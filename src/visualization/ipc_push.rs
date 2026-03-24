// SPDX-License-Identifier: AGPL-3.0-or-later

//! Push visualization data to petalTongue via JSON-RPC IPC.
//!
//! Follows healthSpring's `PetalTonguePushClient` pattern: runtime
//! socket discovery with no compile-time petalTongue dependency.
//! Uses `visualization.render` and `visualization.render.stream`
//! JSON-RPC methods over Unix sockets.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;

use super::types::{DataChannel, NeuralScenario, ThresholdRange};
use crate::ipc_resilience::{CircuitBreaker, RetryPolicy};

/// Client for pushing visualization data to petalTongue.
///
/// Wraps every IPC call with a [`RetryPolicy`] (exponential backoff on
/// transient failures) and a [`CircuitBreaker`] (prevents hammering an
/// unresponsive petalTongue instance).
pub struct PetalTonguePushClient {
    socket_path: PathBuf,
    retry_policy: RetryPolicy,
    circuit_breaker: CircuitBreaker,
}

/// Result of a petalTongue push or RPC call (`Ok` value or [`PushError`]).
pub type PushResult<T> = Result<T, PushError>;

/// Error type for push operations.
#[derive(Debug)]
pub enum PushError {
    /// petalTongue socket not found.
    NotFound(String),
    /// Connection failed.
    ConnectionFailed(std::io::Error),
    /// JSON serialization error.
    SerializationError(String),
    /// RPC error response.
    RpcError {
        /// JSON-RPC error code from the peer.
        code: i64,
        /// Human-readable error message from the peer.
        message: String,
    },
}

impl std::fmt::Display for PushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(msg) => write!(f, "petalTongue not found: {msg}"),
            Self::ConnectionFailed(e) => write!(f, "connection failed: {e}"),
            Self::SerializationError(e) => write!(f, "serialization error: {e}"),
            Self::RpcError { code, message } => write!(f, "RPC error {code}: {message}"),
        }
    }
}

impl std::error::Error for PushError {}

/// Build JSON-RPC params for `visualization.render` (testable without socket).
fn build_render_params(
    session_id: &str,
    title: &str,
    scenario: &NeuralScenario,
) -> serde_json::Value {
    let bindings: Vec<&DataChannel> = scenario
        .ecosystem
        .primals
        .iter()
        .flat_map(|p| p.data_channels.iter())
        .collect();
    let thresholds: Vec<&ThresholdRange> = scenario
        .ecosystem
        .primals
        .iter()
        .flat_map(|p| p.thresholds.iter())
        .collect();

    serde_json::json!({
        "session_id": session_id,
        "title": title,
        "bindings": bindings,
        "thresholds": thresholds,
        "domain": crate::config::PETALTONGUE_DOMAIN,
    })
}

/// Build JSON-RPC params for `visualization.render.stream` append.
fn build_append_params(
    session_id: &str,
    binding_id: &str,
    x_values: &[f64],
    y_values: &[f64],
) -> serde_json::Value {
    serde_json::json!({
        "session_id": session_id,
        "binding_id": binding_id,
        "operation": {
            "type": "append",
            "x_values": x_values,
            "y_values": y_values,
        },
    })
}

/// Build JSON-RPC params for `visualization.render.stream` gauge update.
fn build_gauge_params(session_id: &str, binding_id: &str, value: f64) -> serde_json::Value {
    serde_json::json!({
        "session_id": session_id,
        "binding_id": binding_id,
        "operation": {
            "type": "set_value",
            "value": value,
        },
    })
}

/// Build JSON-RPC params for `visualization.render.stream` replace.
fn build_replace_params(
    session_id: &str,
    binding_id: &str,
    data: &serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "session_id": session_id,
        "binding_id": binding_id,
        "operation": {
            "type": "replace",
            "data": data,
        },
    })
}

/// Default circuit breaker threshold for petalTongue IPC: trip after 3
/// consecutive connection failures, cooldown for 10 seconds.
const BREAKER_THRESHOLD: u32 = 3;
const BREAKER_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(10);

impl PetalTonguePushClient {
    /// Discover petalTongue socket at runtime.
    ///
    /// Resolution order:
    /// 1. `PETALTONGUE_SOCKET` env var
    /// 2. `$XDG_RUNTIME_DIR/petaltongue/*.sock`
    /// 3. `std::env::temp_dir()/petaltongue-*.sock`
    ///
    /// # Errors
    ///
    /// Returns [`PushError::NotFound`] if no petalTongue socket exists at
    /// any candidate path.
    pub fn discover() -> PushResult<Self> {
        if let Ok(path) = std::env::var(crate::config::ENV_PETALTONGUE_SOCKET) {
            let path = PathBuf::from(path);
            if path.exists() {
                return Ok(Self::with_socket(path));
            }
        }
        if let Ok(runtime) = std::env::var(crate::config::ENV_XDG_RUNTIME_DIR) {
            let dir = PathBuf::from(runtime).join(crate::config::PETALTONGUE_SOCKET_DIR);
            if dir.is_dir()
                && let Ok(entries) = std::fs::read_dir(&dir)
            {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().is_some_and(|e| e == "sock") {
                        return Ok(Self::with_socket(p));
                    }
                }
            }
        }
        if let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with(crate::config::PETALTONGUE_SOCKET_PREFIX)
                    && name.ends_with(".sock")
                {
                    return Ok(Self::with_socket(entry.path()));
                }
            }
        }
        Err(PushError::NotFound("no petalTongue socket found".into()))
    }

    fn with_socket(socket_path: PathBuf) -> Self {
        Self {
            socket_path,
            retry_policy: RetryPolicy::default(),
            circuit_breaker: CircuitBreaker::new(BREAKER_THRESHOLD, BREAKER_COOLDOWN),
        }
    }

    /// Create client with an explicit socket path.
    #[must_use]
    pub fn new(socket_path: PathBuf) -> Self {
        Self::with_socket(socket_path)
    }

    /// Create a headless client that silently drops all push operations.
    ///
    /// Uses a non-existent temp-dir socket so every `send_rpc` call
    /// returns `ConnectionFailed` — callers that ignore errors get a
    /// zero-overhead no-op sink without hardcoded socket names.
    #[must_use]
    pub fn headless() -> Self {
        Self::with_socket(
            std::env::temp_dir().join(format!("neuralspring-headless-{}.sock", std::process::id())),
        )
    }

    /// Socket path accessor (for tests).
    #[cfg(test)]
    #[must_use]
    pub const fn socket_path(&self) -> &PathBuf {
        &self.socket_path
    }

    /// Push a full visualization render request.
    ///
    /// # Errors
    ///
    /// Returns [`PushError::ConnectionFailed`] if the socket is unreachable,
    /// [`PushError::SerializationError`] if the payload cannot be encoded, or
    /// [`PushError::RpcError`] if petalTongue rejects the request.
    pub fn push_render(
        &self,
        session_id: &str,
        title: &str,
        scenario: &NeuralScenario,
    ) -> PushResult<()> {
        let params = build_render_params(session_id, title, scenario);
        self.send_rpc("visualization.render", &params)?;
        Ok(())
    }

    /// Push a stream update (append data points to a `TimeSeries`).
    ///
    /// # Errors
    ///
    /// Returns [`PushError::ConnectionFailed`], [`PushError::SerializationError`],
    /// or [`PushError::RpcError`] on failure.
    pub fn push_append(
        &self,
        session_id: &str,
        binding_id: &str,
        x_values: &[f64],
        y_values: &[f64],
    ) -> PushResult<()> {
        let params = build_append_params(session_id, binding_id, x_values, y_values);
        self.send_rpc("visualization.render.stream", &params)?;
        Ok(())
    }

    /// Push a gauge value update.
    ///
    /// # Errors
    ///
    /// Returns [`PushError::ConnectionFailed`], [`PushError::SerializationError`],
    /// or [`PushError::RpcError`] on failure.
    pub fn push_gauge_update(
        &self,
        session_id: &str,
        binding_id: &str,
        value: f64,
    ) -> PushResult<()> {
        let params = build_gauge_params(session_id, binding_id, value);
        self.send_rpc("visualization.render.stream", &params)?;
        Ok(())
    }

    /// Replace the entire data payload for a binding (Heatmap, Bar, etc.).
    ///
    /// # Errors
    ///
    /// Returns [`PushError::ConnectionFailed`], [`PushError::SerializationError`],
    /// or [`PushError::RpcError`] on failure.
    pub fn push_replace(
        &self,
        session_id: &str,
        binding_id: &str,
        data: &serde_json::Value,
    ) -> PushResult<()> {
        let params = build_replace_params(session_id, binding_id, data);
        self.send_rpc("visualization.render.stream", &params)?;
        Ok(())
    }

    /// Query petalTongue for available renderer capabilities.
    ///
    /// # Errors
    ///
    /// Returns [`PushError::ConnectionFailed`], [`PushError::SerializationError`],
    /// or [`PushError::RpcError`] on failure.
    pub fn query_capabilities(&self) -> PushResult<Vec<String>> {
        let params = serde_json::json!({});
        let response = self.send_rpc("visualization.capabilities", &params)?;
        let caps = response
            .get("result")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        Ok(caps)
    }

    fn send_rpc(&self, method: &str, params: &serde_json::Value) -> PushResult<serde_json::Value> {
        if !self.circuit_breaker.is_allowed() {
            return Err(PushError::ConnectionFailed(std::io::Error::other(
                "circuit breaker open — petalTongue unreachable",
            )));
        }

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1,
        });

        let payload = serde_json::to_vec(&request)
            .map_err(|e| PushError::SerializationError(e.to_string()))?;

        let mut last_err = None;
        for attempt in 0..=self.retry_policy.max_retries {
            match self.try_send(&payload) {
                Ok(response) => {
                    self.circuit_breaker.record_success();
                    return Self::check_rpc_error(response);
                }
                Err(e) => {
                    last_err = Some(e);
                    if attempt < self.retry_policy.max_retries {
                        std::thread::sleep(self.retry_policy.delay_for_attempt(attempt));
                    }
                }
            }
        }

        self.circuit_breaker.record_failure();
        Err(last_err.unwrap_or_else(|| {
            PushError::ConnectionFailed(std::io::Error::other("all retries exhausted"))
        }))
    }

    fn try_send(&self, payload: &[u8]) -> PushResult<serde_json::Value> {
        let mut stream =
            UnixStream::connect(&self.socket_path).map_err(PushError::ConnectionFailed)?;
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .map_err(PushError::ConnectionFailed)?;
        stream
            .write_all(payload)
            .map_err(PushError::ConnectionFailed)?;
        stream
            .write_all(b"\n")
            .map_err(PushError::ConnectionFailed)?;
        stream.flush().map_err(PushError::ConnectionFailed)?;

        let mut buf = vec![0u8; 65536];
        let n = stream.read(&mut buf).map_err(PushError::ConnectionFailed)?;

        serde_json::from_slice(&buf[..n]).map_err(|e| PushError::SerializationError(e.to_string()))
    }

    fn check_rpc_error(response: serde_json::Value) -> PushResult<serde_json::Value> {
        if let Some(error) = response.get("error") {
            return Err(PushError::RpcError {
                code: error
                    .get("code")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(-1),
                message: error
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
            });
        }
        Ok(response)
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;
    use crate::visualization::scenarios;

    #[test]
    fn push_error_display_not_found() {
        let e = PushError::NotFound("no socket".into());
        let s = format!("{e}");
        assert!(s.contains("petalTongue not found"));
        assert!(s.contains("no socket"));
    }

    #[test]
    fn push_error_display_connection_failed() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "connection refused");
        let e = PushError::ConnectionFailed(io_err);
        assert!(format!("{e}").contains("connection failed"));
    }

    #[test]
    fn push_error_display_serialization_error() {
        let e = PushError::SerializationError("invalid json".into());
        let s = format!("{e}");
        assert!(s.contains("serialization error"));
        assert!(s.contains("invalid json"));
    }

    #[test]
    fn push_error_display_rpc_error() {
        let e = PushError::RpcError {
            code: -32600,
            message: "invalid request".into(),
        };
        let s = format!("{e}");
        assert!(s.contains("RPC error"));
        assert!(s.contains("-32600"));
    }

    #[test]
    fn push_error_impl_error() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<PushError>();
    }

    #[test]
    fn client_new_stores_path() {
        let path = std::env::temp_dir().join("test-socket.sock");
        let client = PetalTonguePushClient::new(path.clone());
        assert_eq!(client.socket_path(), &path);
    }

    #[test]
    fn headless_client_uses_pid_unique_path() {
        let client = PetalTonguePushClient::headless();
        let path_str = client.socket_path().to_string_lossy().to_string();
        assert!(
            path_str.contains("neuralspring-headless-"),
            "headless path should contain sentinel name"
        );
        assert!(
            path_str.contains(&std::process::id().to_string()),
            "headless path should include PID for uniqueness"
        );
    }

    #[test]
    fn headless_client_push_returns_connection_error() {
        let client = PetalTonguePushClient::headless();
        let result = client.push_gauge_update("s", "b", 1.0);
        assert!(matches!(result, Err(PushError::ConnectionFailed(_))));
    }

    #[test]
    fn build_render_params_structure() {
        let (scenario, _) = scenarios::spectral_study();
        let params = build_render_params("sess-123", "Spectral Test", &scenario);

        assert_eq!(
            params.get("session_id").and_then(|v| v.as_str()),
            Some("sess-123")
        );
        assert_eq!(
            params.get("title").and_then(|v| v.as_str()),
            Some("Spectral Test")
        );
        assert_eq!(
            params.get("domain").and_then(|v| v.as_str()),
            Some(crate::config::PETALTONGUE_DOMAIN)
        );
        assert!(params.get("bindings").is_some());
        assert!(params.get("thresholds").is_some());

        let bindings = params["bindings"].as_array().expect("bindings is array");
        assert!(!bindings.is_empty(), "spectral study should have bindings");
    }

    #[test]
    fn build_append_params_structure() {
        let params = build_append_params("sess-456", "binding-1", &[1.0, 2.0], &[10.0, 20.0]);

        assert_eq!(params["session_id"], "sess-456");
        assert_eq!(params["binding_id"], "binding-1");
        assert_eq!(params["operation"]["type"], "append");
        let xs = params["operation"]["x_values"]
            .as_array()
            .expect("x_values");
        assert_eq!(xs.len(), 2);
    }

    #[test]
    fn build_gauge_params_structure() {
        let params = build_gauge_params("sess-789", "gauge-binding", 73.5);

        assert_eq!(params["session_id"], "sess-789");
        assert_eq!(params["binding_id"], "gauge-binding");
        assert_eq!(params["operation"]["type"], "set_value");
        assert_eq!(params["operation"]["value"], 73.5);
    }

    fn mock_petaltongue_response(listener: &std::os::unix::net::UnixListener) -> serde_json::Value {
        listener
            .set_nonblocking(false)
            .expect("set listener blocking");
        let (stream, _) = listener.accept().expect("accept");
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .expect("set read timeout");
        let mut reader = std::io::BufReader::new(&stream);
        let mut line = String::new();
        std::io::BufRead::read_line(&mut reader, &mut line).expect("read line");
        let request: serde_json::Value = serde_json::from_str(line.trim()).expect("parse request");
        let response = serde_json::json!({"jsonrpc": "2.0", "result": "ok", "id": 1});
        let mut writer = &stream;
        writer
            .write_all(serde_json::to_vec(&response).expect("ser").as_slice())
            .expect("write");
        request
    }

    fn socket_test_setup(name: &str) -> (PathBuf, std::os::unix::net::UnixListener) {
        let dir = std::env::temp_dir().join(format!(
            "ns_ipc_{name}_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).ok();
        let sock_path = dir.join(format!("{name}.sock"));
        let _ = std::fs::remove_file(&sock_path);
        let listener = std::os::unix::net::UnixListener::bind(&sock_path).expect("bind");
        listener
            .set_nonblocking(false)
            .expect("set listener blocking");
        (sock_path, listener)
    }

    fn socket_test_cleanup(sock_path: &std::path::Path) {
        std::fs::remove_file(sock_path).ok();
        if let Some(parent) = sock_path.parent() {
            std::fs::remove_dir(parent).ok();
        }
    }

    #[test]
    fn push_render_sends_valid_jsonrpc() {
        let (sock_path, listener) = socket_test_setup("render");
        let client = PetalTonguePushClient::new(sock_path.clone());
        let (scenario, _) = scenarios::spectral_study();

        let handle = std::thread::spawn(move || mock_petaltongue_response(&listener));
        std::thread::yield_now();
        let result = client.push_render("sess-1", "Test Render", &scenario);
        let request = handle.join().expect("mock thread");

        assert!(result.is_ok(), "push_render failed: {result:?}");
        assert_eq!(request["jsonrpc"], "2.0");
        assert_eq!(request["method"], "visualization.render");
        assert_eq!(request["params"]["session_id"], "sess-1");
        assert_eq!(
            request["params"]["domain"],
            crate::config::PETALTONGUE_DOMAIN
        );
        socket_test_cleanup(&sock_path);
    }

    #[test]
    fn push_append_sends_valid_jsonrpc() {
        let (sock_path, listener) = socket_test_setup("append");
        let client = PetalTonguePushClient::new(sock_path.clone());

        let handle = std::thread::spawn(move || mock_petaltongue_response(&listener));
        let result = client.push_append("sess-2", "bind-1", &[1.0, 2.0], &[10.0, 20.0]);
        let request = handle.join().expect("mock thread");

        assert!(result.is_ok());
        assert_eq!(request["method"], "visualization.render.stream");
        assert_eq!(request["params"]["operation"]["type"], "append");
        socket_test_cleanup(&sock_path);
    }

    #[test]
    fn push_gauge_update_sends_valid_jsonrpc() {
        let (sock_path, listener) = socket_test_setup("gauge");
        let client = PetalTonguePushClient::new(sock_path.clone());

        let handle = std::thread::spawn(move || mock_petaltongue_response(&listener));
        let result = client.push_gauge_update("sess-3", "gauge-1", 42.5);
        let request = handle.join().expect("mock thread");

        assert!(result.is_ok());
        assert_eq!(request["method"], "visualization.render.stream");
        assert_eq!(request["params"]["operation"]["type"], "set_value");
        assert_eq!(request["params"]["operation"]["value"], 42.5);
        socket_test_cleanup(&sock_path);
    }

    #[test]
    fn push_connection_failed_on_missing_socket() {
        let client =
            PetalTonguePushClient::new(std::env::temp_dir().join("nonexistent_ns_test.sock"));
        let result = client.push_gauge_update("s", "b", 1.0);
        assert!(matches!(result, Err(PushError::ConnectionFailed(_))));
    }

    #[test]
    fn discover_returns_not_found_when_no_socket_exists() {
        let result = PetalTonguePushClient::discover();
        if result.is_ok() {
            return;
        }
        assert!(matches!(result, Err(PushError::NotFound(_))));
    }
}
