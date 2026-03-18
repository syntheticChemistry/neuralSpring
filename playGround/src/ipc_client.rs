// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reusable JSON-RPC 2.0 client over Unix domain sockets.
//!
//! Implements the biomeOS 5-tier socket resolution and newline-delimited
//! JSON-RPC protocol used by all ecoPrimals primals.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

/// biomeOS socket subdirectory name within the XDG runtime directory.
const BIOMEOS_SOCKET_SUBDIR: &str = neural_spring::config::BIOMEOS_SOCKET_SUBDIR;

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Serialize)]
struct JsonRpcRequest<'a> {
    jsonrpc: &'static str,
    method: &'a str,
    params: &'a serde_json::Value,
    id: u64,
}

#[derive(Debug, Deserialize)]
pub struct JsonRpcResponse {
    #[serde(default)]
    pub result: Option<serde_json::Value>,
    #[serde(default)]
    pub error: Option<JsonRpcError>,
    #[serde(default)]
    pub id: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

impl std::fmt::Display for JsonRpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON-RPC error {}: {}", self.code, self.message)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Structured IPC errors (healthSpring V31 / rhizoCrypt V13 pattern)
// ═══════════════════════════════════════════════════════════════════

/// Typed IPC error phases for observability and targeted retry logic.
///
/// Each variant captures the phase where the failure occurred, enabling
/// callers to decide whether a retry is appropriate (e.g. `Connect` and
/// `Timeout` are often transient, while `RpcError` is application-level).
#[derive(Debug)]
pub enum IpcError {
    /// Socket connection failed (primal may not be running).
    Connect(std::io::Error),
    /// Failed to write the request payload.
    Write(std::io::Error),
    /// Failed to read the response.
    Read(std::io::Error),
    /// Response was not valid JSON.
    InvalidJson(serde_json::Error),
    /// Response contained neither `result` nor `error`.
    NoResult,
    /// Remote JSON-RPC error with code and message.
    RpcError { code: i64, message: String },
    /// IPC call exceeded the configured timeout.
    Timeout,
}

impl IpcError {
    /// Whether this error is likely transient and worth retrying.
    #[must_use]
    pub const fn is_recoverable(&self) -> bool {
        matches!(self, Self::Connect(_) | Self::Timeout)
    }
}

impl std::fmt::Display for IpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connect(e) => write!(f, "connect: {e}"),
            Self::Write(e) => write!(f, "write: {e}"),
            Self::Read(e) => write!(f, "read: {e}"),
            Self::InvalidJson(e) => write!(f, "parse: {e}"),
            Self::NoResult => write!(f, "response missing 'result' field"),
            Self::RpcError { code, message } => write!(f, "rpc error {code}: {message}"),
            Self::Timeout => write!(f, "ipc timeout"),
        }
    }
}

impl std::error::Error for IpcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Connect(e) | Self::Write(e) | Self::Read(e) => Some(e),
            Self::InvalidJson(e) => Some(e),
            _ => None,
        }
    }
}

/// Extract a JSON-RPC error code and message from a raw response value.
///
/// Centralizes the `response.get("error")` pattern used across IPC call
/// sites (airSpring V0.8.6 pattern).
#[must_use]
pub fn extract_rpc_error(response: &serde_json::Value) -> Option<(i64, String)> {
    let err = response.get("error")?;
    let code = err.get("code")?.as_i64()?;
    let message = err.get("message")?.as_str()?.to_owned();
    Some((code, message))
}

// ═══════════════════════════════════════════════════════════════════
// Core IPC call
// ═══════════════════════════════════════════════════════════════════

/// Send a single JSON-RPC 2.0 request to a Unix socket and return the result.
///
/// Returns `Result<Value>` for backward compatibility. Use [`call_typed`]
/// for structured `IpcError` reporting.
pub async fn call(
    socket_path: &Path,
    method: &str,
    params: &serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value> {
    call_typed(socket_path, method, params, timeout)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Send a JSON-RPC 2.0 request with structured [`IpcError`] reporting.
///
/// Prefer this over [`call`] in new code — the typed error lets callers
/// distinguish transient failures (connect, timeout) from application
/// errors (RPC error, invalid JSON).
pub async fn call_typed(
    socket_path: &Path,
    method: &str,
    params: &serde_json::Value,
    timeout: Duration,
) -> std::result::Result<serde_json::Value, IpcError> {
    let stream = UnixStream::connect(socket_path)
        .await
        .map_err(IpcError::Connect)?;

    let (reader, mut writer) = stream.into_split();

    let request = JsonRpcRequest {
        jsonrpc: "2.0",
        method,
        params,
        id: next_id(),
    };

    let mut payload = serde_json::to_vec(&request).map_err(IpcError::InvalidJson)?;
    payload.push(b'\n');
    writer.write_all(&payload).await.map_err(IpcError::Write)?;
    writer.flush().await.map_err(IpcError::Write)?;

    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();
    let read_result = tokio::time::timeout(timeout, buf_reader.read_line(&mut line))
        .await
        .map_err(|_| IpcError::Timeout)?
        .map_err(IpcError::Read)?;

    if read_result == 0 {
        return Err(IpcError::Read(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "empty response",
        )));
    }

    let resp: JsonRpcResponse = serde_json::from_str(line.trim()).map_err(IpcError::InvalidJson)?;

    if let Some(err) = resp.error {
        return Err(IpcError::RpcError {
            code: err.code,
            message: err.message,
        });
    }

    resp.result.ok_or(IpcError::NoResult)
}

// ═══════════════════════════════════════════════════════════════════
// biomeOS 5-tier socket resolution
// ═══════════════════════════════════════════════════════════════════

/// Resolve the biomeOS socket directory using the standard 5-tier fallback.
pub fn resolve_socket_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("BIOMEOS_SOCKET_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(xdg).join(BIOMEOS_SOCKET_SUBDIR);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let Ok(meta) = std::fs::metadata("/proc/self") {
            let uid = meta.uid();
            let dir = PathBuf::from(format!("/run/user/{uid}")).join(BIOMEOS_SOCKET_SUBDIR);
            if dir.parent().is_some_and(Path::exists) {
                return dir;
            }
        }
    }
    std::env::temp_dir().join(BIOMEOS_SOCKET_SUBDIR)
}

fn get_family_id() -> String {
    std::env::var("FAMILY_ID")
        .or_else(|_| std::env::var("BIOMEOS_FAMILY_ID"))
        .unwrap_or_else(|_| "default".to_string())
}

/// Discover a primal socket by name using the standard resolution order:
/// 1. `{name}-{family_id}.sock`
/// 2. `{name}.sock`
/// 3. Any `{name}*.sock` in the socket directory
pub fn discover_socket(primal_name: &str) -> Result<PathBuf> {
    let socket_dir = resolve_socket_dir();
    let family_id = get_family_id();

    let with_family = socket_dir.join(format!("{primal_name}-{family_id}.sock"));
    if with_family.exists() {
        return Ok(with_family);
    }

    let without_family = socket_dir.join(format!("{primal_name}.sock"));
    if without_family.exists() {
        return Ok(without_family);
    }

    if let Ok(entries) = std::fs::read_dir(&socket_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(primal_name) && name_str.ends_with(".sock") {
                return Ok(entry.path());
            }
        }
    }

    anyhow::bail!(
        "no socket found for primal '{primal_name}' in {}",
        socket_dir.display()
    )
}

/// Discover a primal socket by required capability.
///
/// Scans the biomeOS socket directory for any primal that advertises the
/// given capability via `capability.list`.  Falls back to `discover_socket`
/// with the `hint_name` if no capability probe succeeds (e.g. primal is
/// not yet running).
///
/// This follows the ecoPrimals self-knowledge principle: a client only
/// knows *what* it needs (a capability), not *who* provides it.
pub fn discover_by_capability(required_capability: &str, hint_name: &str) -> Result<PathBuf> {
    let socket_dir = resolve_socket_dir();

    if let Ok(entries) = std::fs::read_dir(&socket_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.ends_with(".sock") {
                continue;
            }
            if let Ok(caps) = probe_capabilities(&path)
                && caps.iter().any(|c| c == required_capability)
            {
                return Ok(path);
            }
        }
    }

    discover_socket(hint_name).with_context(|| {
        format!(
            "no primal advertising '{required_capability}' found, \
             fallback name '{hint_name}' also failed"
        )
    })
}

/// Probe a primal's capabilities by sending `capability.list` over JSON-RPC.
///
/// Handles all ecosystem response formats (flat, object array, nested,
/// double-nested, result wrapper) via [`parse_capability_list`].
///
/// Returns a list of capability strings, or an error if the primal does
/// not respond within 2 seconds.
fn probe_capabilities(socket_path: &std::path::Path) -> Result<Vec<String>> {
    let params = serde_json::json!({});
    let timeout = Duration::from_secs(2);

    let result = if let Ok(h) = tokio::runtime::Handle::try_current() {
        h.block_on(call(socket_path, "capability.list", &params, timeout))
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        rt.block_on(call(socket_path, "capability.list", &params, timeout))
    }?;

    Ok(parse_capability_list(&result))
}

/// Extract capability strings from any response format used across the ecosystem.
///
/// Handles all 4 formats (airSpring V0.8.7 pattern):
///   - **Flat**: `["cap.a", "cap.b"]`
///   - **Object array**: `[{"name": "cap.a"}, {"capability": "cap.b"}]`
///   - **Nested wrapper**: `{"capabilities": ["cap.a"]}`
///   - **Double-nested**: `{"capabilities": {"capabilities": ["cap.a"]}}`
///   - **Result wrapper**: `{"result": ["cap.a"]}`
///
/// Returns an empty vec (never errors) for unrecognized formats — defensive
/// parsing is safer than panicking during discovery probes.
#[must_use]
pub fn parse_capability_list(value: &serde_json::Value) -> Vec<String> {
    if let serde_json::Value::Object(obj) = value {
        if let Some(inner) = obj.get("capabilities") {
            return parse_capability_list(inner);
        }
        if let Some(inner) = obj.get("result") {
            return parse_capability_list(inner);
        }
    }

    match value {
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| match v {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Object(obj) => obj
                    .get("name")
                    .or_else(|| obj.get("capability"))
                    .and_then(|n| n.as_str())
                    .map(str::to_owned),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Read the IPC timeout from environment, defaulting to 5 seconds.
#[must_use]
pub fn ipc_timeout() -> Duration {
    let secs: u64 = std::env::var("PRIMAL_IPC_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    Duration::from_secs(secs)
}

// ═══════════════════════════════════════════════════════════════════
// Generic primal discovery (sweetGrass / groundSpring V112 pattern)
// ═══════════════════════════════════════════════════════════════════

/// Generate the environment variable name for a primal's socket override.
///
/// Follows the ecosystem convention: `{UPPER_NAME}_SOCKET`.
/// Example: `socket_env_var("toadstool")` → `"TOADSTOOL_SOCKET"`.
#[must_use]
pub fn socket_env_var(primal_name: &str) -> String {
    format!("{}_SOCKET", primal_name.to_uppercase())
}

/// Generate the environment variable name for a primal's address (host:port).
///
/// Follows the ecosystem convention: `{UPPER_NAME}_ADDRESS`.
#[must_use]
pub fn address_env_var(primal_name: &str) -> String {
    format!("{}_ADDRESS", primal_name.to_uppercase())
}

/// Discover a primal socket by name, checking the `{UPPER}_SOCKET` env var
/// first, then falling back to biomeOS socket directory resolution.
///
/// This is the generic discovery helper — primals should not hardcode
/// socket paths or peer names beyond their own `niche`.
pub fn discover_primal(primal_name: &str) -> Result<PathBuf> {
    let env_key = socket_env_var(primal_name);
    if let Ok(path) = std::env::var(&env_key) {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Ok(p);
        }
    }
    discover_socket(primal_name)
}

// ═══════════════════════════════════════════════════════════════════
// RPC response classification (groundSpring V112 pattern)
// ═══════════════════════════════════════════════════════════════════

/// Classifies a JSON-RPC response for graceful degradation.
///
/// Callers can match on `DispatchOutcome` to decide whether a failure is
/// retryable (protocol error) vs application-level (wrong capability).
#[derive(Debug)]
pub enum DispatchOutcome {
    /// The RPC succeeded and returned a result value.
    Ok(serde_json::Value),
    /// Protocol-level error (JSON-RPC spec codes -32700 to -32600).
    ProtocolError { code: i64, message: String },
    /// Application-level error (code >= -32000 or non-standard).
    ApplicationError { code: i64, message: String },
}

impl DispatchOutcome {
    /// Classify a typed [`IpcError`] into a `DispatchOutcome`.
    #[must_use]
    pub fn from_ipc_error(err: &IpcError) -> Self {
        match err {
            IpcError::RpcError { code, message } => {
                if *code <= -32600 && *code >= -32700 {
                    Self::ProtocolError {
                        code: *code,
                        message: message.clone(),
                    }
                } else {
                    Self::ApplicationError {
                        code: *code,
                        message: message.clone(),
                    }
                }
            }
            other => Self::ProtocolError {
                code: -1,
                message: other.to_string(),
            },
        }
    }

    /// Classify a parsed JSON-RPC response value (healthSpring V28 pattern).
    ///
    /// Inspects `"result"` and `"error"` fields to determine outcome type.
    #[must_use]
    pub fn classify_response(response: &serde_json::Value) -> Self {
        if let Some(result) = response.get("result") {
            return Self::Ok(result.clone());
        }
        if let Some((code, message)) = extract_rpc_error(response) {
            return if (-32700..=-32600).contains(&code) {
                Self::ProtocolError { code, message }
            } else {
                Self::ApplicationError { code, message }
            };
        }
        Self::ProtocolError {
            code: -1,
            message: "response missing both 'result' and 'error'".to_owned(),
        }
    }

    /// Whether the outcome represents success.
    #[must_use]
    pub const fn is_ok(&self) -> bool {
        matches!(self, Self::Ok(_))
    }

    /// Whether the outcome is a protocol-level error (JSON-RPC spec codes).
    #[must_use]
    pub const fn is_protocol_error(&self) -> bool {
        matches!(self, Self::ProtocolError { .. })
    }

    /// Whether the error indicates the called method does not exist
    /// (JSON-RPC code -32601).
    #[must_use]
    pub const fn is_method_not_found(&self) -> bool {
        matches!(
            self,
            Self::ProtocolError { code: -32601, .. } | Self::ApplicationError { code: -32601, .. }
        )
    }
}

// ═══════════════════════════════════════════════════════════════════
// Resilient IPC call (healthSpring V32 pattern)
// ═══════════════════════════════════════════════════════════════════

const RESILIENT_MAX_RETRIES: u32 = 2;
const RESILIENT_RETRY_BASE_MS: u64 = 50;
const RESILIENT_CIRCUIT_OPEN_MS: u64 = 5000;

static LAST_FAILURE_EPOCH_MS: AtomicU64 = AtomicU64::new(0);

fn epoch_ms_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "epoch millis fits u64 for ~584 million years"
            )]
            let ms = d.as_millis() as u64;
            ms
        })
}

fn circuit_is_open() -> bool {
    let last = LAST_FAILURE_EPOCH_MS.load(Ordering::Relaxed);
    last > 0 && epoch_ms_now().saturating_sub(last) < RESILIENT_CIRCUIT_OPEN_MS
}

fn record_failure() {
    LAST_FAILURE_EPOCH_MS.store(epoch_ms_now(), Ordering::Relaxed);
}

fn record_success() {
    LAST_FAILURE_EPOCH_MS.store(0, Ordering::Relaxed);
}

/// IPC call with circuit breaker + exponential backoff retry.
///
/// Retries recoverable failures (connect, timeout) up to 2 times with
/// 50ms/100ms backoff. If the circuit breaker is open (failure within
/// last 5 seconds), short-circuits immediately.
///
/// Use this for calls to external primals (provenance trio, biomeOS)
/// where transient unavailability is expected.
pub async fn resilient_call(
    socket_path: &Path,
    method: &str,
    params: &serde_json::Value,
    timeout: Duration,
) -> std::result::Result<serde_json::Value, IpcError> {
    if circuit_is_open() {
        return Err(IpcError::Connect(std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            "circuit open — primal recently unavailable",
        )));
    }

    let mut last_err = IpcError::Timeout;
    for attempt in 0..=RESILIENT_MAX_RETRIES {
        match call_typed(socket_path, method, params, timeout).await {
            Ok(v) => {
                record_success();
                return Ok(v);
            }
            Err(e) => {
                if !e.is_recoverable() {
                    record_failure();
                    return Err(e);
                }
                last_err = e;
                if attempt < RESILIENT_MAX_RETRIES {
                    let delay = RESILIENT_RETRY_BASE_MS * 2_u64.saturating_pow(attempt);
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
            }
        }
    }
    record_failure();
    Err(last_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipc_timeout_default() {
        temp_env::with_var_unset("PRIMAL_IPC_TIMEOUT_SECS", || {
            assert_eq!(ipc_timeout(), Duration::from_secs(5));
        });
    }

    #[test]
    fn resolve_socket_dir_respects_env() {
        let test_dir = std::env::temp_dir().join("ns_test_biomeos");
        let test_str = test_dir.to_str().expect("temp_dir is valid UTF-8");
        temp_env::with_var("BIOMEOS_SOCKET_DIR", Some(test_str), || {
            assert_eq!(resolve_socket_dir(), test_dir);
        });
    }

    #[test]
    fn resolve_socket_dir_falls_through_tiers() {
        let xdg_dir = std::env::temp_dir().join("ns_xdg_test");
        let xdg_str = xdg_dir.to_str().expect("temp_dir is valid UTF-8");
        temp_env::with_vars(
            [
                ("BIOMEOS_SOCKET_DIR", None::<&str>),
                ("XDG_RUNTIME_DIR", Some(xdg_str)),
            ],
            || {
                assert_eq!(resolve_socket_dir(), xdg_dir.join("biomeos"));
            },
        );
    }

    #[test]
    fn get_family_id_default() {
        temp_env::with_vars(
            [("FAMILY_ID", None::<&str>), ("BIOMEOS_FAMILY_ID", None)],
            || {
                assert_eq!(get_family_id(), "default");
            },
        );
    }

    #[test]
    fn get_family_id_from_env() {
        temp_env::with_var("FAMILY_ID", Some("test_family"), || {
            assert_eq!(get_family_id(), "test_family");
        });
    }

    #[test]
    fn discover_socket_fails_when_dir_missing() {
        let missing = std::env::temp_dir().join("ns_nonexistent_biomeos_test_dir");
        let missing_str = missing.to_str().expect("temp_dir is valid UTF-8");
        temp_env::with_var("BIOMEOS_SOCKET_DIR", Some(missing_str), || {
            assert!(discover_socket("some_primal").is_err());
        });
    }

    #[test]
    fn parse_capability_list_flat_array() {
        let val = serde_json::json!(["compute.submit", "compute.probe"]);
        assert_eq!(
            parse_capability_list(&val),
            vec!["compute.submit", "compute.probe"]
        );
    }

    #[test]
    fn parse_capability_list_object_format() {
        let obj = serde_json::json!({
            "primal": "neuralspring",
            "capabilities": ["science.ipr", "science.spectral_analysis"]
        });
        assert_eq!(
            parse_capability_list(&obj),
            vec!["science.ipr", "science.spectral_analysis"]
        );
    }

    #[test]
    fn parse_capability_list_object_array_format() {
        let val = serde_json::json!([
            {"name": "health", "version": "1.0"},
            {"capability": "compute.dispatch"}
        ]);
        assert_eq!(
            parse_capability_list(&val),
            vec!["health", "compute.dispatch"]
        );
    }

    #[test]
    fn parse_capability_list_double_nested() {
        let val = serde_json::json!({
            "capabilities": {"capabilities": ["health", "compute.dispatch"]}
        });
        assert_eq!(
            parse_capability_list(&val),
            vec!["health", "compute.dispatch"]
        );
    }

    #[test]
    fn parse_capability_list_result_wrapper() {
        let val = serde_json::json!({"result": ["health", "data.weather"]});
        assert_eq!(parse_capability_list(&val), vec!["health", "data.weather"]);
    }

    #[test]
    fn parse_capability_list_empty_and_junk() {
        assert!(parse_capability_list(&serde_json::json!(null)).is_empty());
        assert!(parse_capability_list(&serde_json::json!(42)).is_empty());
        assert!(parse_capability_list(&serde_json::json!([])).is_empty());
    }

    #[test]
    fn socket_env_var_uppercases() {
        assert_eq!(
            socket_env_var(neural_spring::primal_names::TOADSTOOL),
            "TOADSTOOL_SOCKET"
        );
        assert_eq!(
            socket_env_var(neural_spring::primal_names::BIOMEOS),
            "BIOMEOS_SOCKET"
        );
    }

    #[test]
    fn address_env_var_uppercases() {
        assert_eq!(
            address_env_var(neural_spring::primal_names::NESTGATE),
            "NESTGATE_ADDRESS"
        );
    }

    #[test]
    fn discover_primal_falls_back_to_socket_dir() {
        let missing = std::env::temp_dir().join("ns_nonexistent_biomeos_test_dir");
        let missing_str = missing.to_str().expect("temp_dir is valid UTF-8");
        temp_env::with_vars(
            [
                ("TOADSTOOL_SOCKET", None::<&str>),
                ("BIOMEOS_SOCKET_DIR", Some(missing_str)),
            ],
            || {
                assert!(discover_primal(neural_spring::primal_names::TOADSTOOL).is_err());
            },
        );
    }

    #[test]
    fn dispatch_outcome_protocol_vs_application() {
        let proto = IpcError::RpcError {
            code: -32601,
            message: "method not found".into(),
        };
        let app = IpcError::RpcError {
            code: -1,
            message: "custom error".into(),
        };
        assert!(matches!(
            DispatchOutcome::from_ipc_error(&proto),
            DispatchOutcome::ProtocolError { .. }
        ));
        assert!(matches!(
            DispatchOutcome::from_ipc_error(&app),
            DispatchOutcome::ApplicationError { .. }
        ));
    }

    #[test]
    fn circuit_breaker_opens_and_closes() {
        record_success();
        assert!(!circuit_is_open());
        record_failure();
        assert!(circuit_is_open());
        record_success();
        assert!(!circuit_is_open());
    }

    #[test]
    fn discover_socket_finds_exact_match() {
        let dir = std::env::temp_dir().join("biomeos_discover_test");
        let _ = std::fs::create_dir_all(&dir);
        let sock = dir.join("testprimal.sock");
        std::fs::write(&sock, b"").unwrap();

        temp_env::with_vars(
            [
                ("BIOMEOS_SOCKET_DIR", Some(dir.to_str().unwrap())),
                ("FAMILY_ID", None),
                ("BIOMEOS_FAMILY_ID", None),
            ],
            || {
                let found = discover_socket("testprimal").unwrap();
                assert_eq!(found, sock);
            },
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
