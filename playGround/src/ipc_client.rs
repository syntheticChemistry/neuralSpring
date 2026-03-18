// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reusable JSON-RPC 2.0 client over Unix domain sockets.
//!
//! Protocol, error types, and call mechanics. Discovery lives in
//! [`crate::discovery`] — re-exported here for backward compatibility.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

pub use crate::discovery::{
    address_env_var, discover_by_capability, discover_primal, discover_socket, ipc_timeout,
    parse_capability_list, resolve_socket_dir, socket_env_var,
};

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
#[must_use]
pub fn extract_rpc_error(response: &serde_json::Value) -> Option<(i64, String)> {
    let err = response.get("error")?;
    let code = err.get("code")?.as_i64()?;
    let message = err.get("message")?.as_str()?.to_owned();
    Some((code, message))
}

/// Extract the `"result"` field from a JSON-RPC response, returning `None`
/// if an `"error"` field is present (healthSpring V37 / wetSpring V127 pattern).
#[must_use]
pub fn extract_rpc_result(response: &serde_json::Value) -> Option<&serde_json::Value> {
    if response.get("error").is_some() {
        return None;
    }
    response.get("result")
}

/// Consuming variant that clones the result value.
#[must_use]
pub fn extract_rpc_result_owned(response: &serde_json::Value) -> Option<serde_json::Value> {
    if response.get("error").is_some() {
        return None;
    }
    response.get("result").cloned()
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
// RPC response classification (groundSpring V112 pattern)
// ═══════════════════════════════════════════════════════════════════

/// Classifies a JSON-RPC response for graceful degradation.
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

    /// Classify a parsed JSON-RPC response value (healthSpring V37 pattern).
    #[must_use]
    pub fn classify_response(response: &serde_json::Value) -> Self {
        if let Some(result) = extract_rpc_result(response) {
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

    /// Whether the error indicates the called method does not exist.
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
#[expect(
    clippy::unwrap_used,
    reason = "test assertions — unwrap documents expected success"
)]
mod tests {
    use super::*;

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
    fn dispatch_outcome_classify_never_panics() {
        let fuzz_values: &[serde_json::Value] = &[
            serde_json::json!(null),
            serde_json::json!(42),
            serde_json::json!({}),
            serde_json::json!({"result": "ok"}),
            serde_json::json!({"error": null}),
            serde_json::json!({"error": {"code": -32600, "message": "invalid"}}),
            serde_json::json!({"error": {"code": -1, "message": "app error"}}),
            serde_json::json!({"result": null, "error": null}),
            serde_json::json!({"error": {"code": "not_int"}}),
        ];
        for val in fuzz_values {
            let _ = DispatchOutcome::classify_response(val);
        }
    }

    #[test]
    fn extract_rpc_error_never_panics() {
        let fuzz_values: &[serde_json::Value] = &[
            serde_json::json!(null),
            serde_json::json!(42),
            serde_json::json!({}),
            serde_json::json!({"error": null}),
            serde_json::json!({"error": {}}),
            serde_json::json!({"error": {"code": "not_int"}}),
            serde_json::json!({"error": {"code": 42}}),
            serde_json::json!({"error": {"code": 42, "message": 123}}),
            serde_json::json!({"error": {"code": -32601, "message": "method not found"}}),
        ];
        for val in fuzz_values {
            let _ = extract_rpc_error(val);
        }
    }

    #[test]
    fn extract_rpc_result_returns_result_when_no_error() {
        let resp = serde_json::json!({"jsonrpc": "2.0", "result": 42, "id": 1});
        assert_eq!(extract_rpc_result(&resp), Some(&serde_json::json!(42)));
    }

    #[test]
    fn extract_rpc_result_returns_none_when_error_present() {
        let resp = serde_json::json!({"jsonrpc": "2.0", "error": {"code": -32601, "message": "nf"}, "id": 1});
        assert!(extract_rpc_result(&resp).is_none());
    }

    #[test]
    fn extract_rpc_result_returns_none_when_neither() {
        let resp = serde_json::json!({"jsonrpc": "2.0", "id": 1});
        assert!(extract_rpc_result(&resp).is_none());
    }

    #[test]
    fn extract_rpc_result_owned_clones_value() {
        let resp = serde_json::json!({"result": [1, 2, 3]});
        assert_eq!(
            extract_rpc_result_owned(&resp),
            Some(serde_json::json!([1, 2, 3]))
        );
    }

    #[test]
    fn extract_rpc_result_never_panics() {
        let fuzz_values: &[serde_json::Value] = &[
            serde_json::json!(null),
            serde_json::json!(42),
            serde_json::json!({}),
            serde_json::json!({"result": "ok"}),
            serde_json::json!({"error": null}),
            serde_json::json!({"error": null, "result": "ok"}),
            serde_json::json!({"error": {"code": -32600, "message": "invalid"}}),
            serde_json::json!({"result": null}),
        ];
        for val in fuzz_values {
            let _ = extract_rpc_result(val);
            let _ = extract_rpc_result_owned(val);
        }
    }

    #[test]
    fn ipc_error_is_recoverable_contract() {
        assert!(
            IpcError::Connect(std::io::Error::new(
                std::io::ErrorKind::ConnectionRefused,
                "test"
            ))
            .is_recoverable()
        );
        assert!(IpcError::Timeout.is_recoverable());
        assert!(!IpcError::NoResult.is_recoverable());
        assert!(
            !IpcError::RpcError {
                code: -1,
                message: "err".into()
            }
            .is_recoverable()
        );
        assert!(
            !IpcError::InvalidJson(serde_json::from_str::<serde_json::Value>("{{").unwrap_err())
                .is_recoverable()
        );
    }
}
