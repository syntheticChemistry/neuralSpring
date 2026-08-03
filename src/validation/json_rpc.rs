// SPDX-License-Identifier: AGPL-3.0-or-later

//! JSON-RPC 2.0 helpers for primal IPC probes.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

use crate::error::IpcError;

/// Send a JSON-RPC 2.0 request over a Unix socket and return the response.
///
/// Uses newline-delimited framing per `PRIMAL_IPC_PROTOCOL.md`.
///
/// # Errors
///
/// Returns an error if the socket cannot be connected, the request fails
/// to send, or the response cannot be parsed.
pub fn json_rpc_call(
    socket: &Path,
    method: &str,
    params: &serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value, IpcError> {
    let method_owned = method.to_owned();
    let stream = UnixStream::connect(socket).map_err(|e| IpcError::Transport {
        capability: method_owned.clone(),
        reason: format!("connect {}: {e}", socket.display()),
    })?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| IpcError::Transport {
            capability: method_owned.clone(),
            reason: format!("set_read_timeout: {e}"),
        })?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| IpcError::Transport {
            capability: method_owned.clone(),
            reason: format!("set_write_timeout: {e}"),
        })?;

    json_rpc_on_stream(stream, &method_owned, params)
}

fn json_rpc_on_stream(
    mut stream: UnixStream,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value, IpcError> {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    });

    let method_owned = method.to_owned();
    let mut payload = serde_json::to_vec(&request).map_err(|e| IpcError::Protocol {
        capability: method_owned.clone(),
        reason: format!("serialize: {e}"),
    })?;
    payload.push(b'\n');

    stream
        .write_all(&payload)
        .map_err(|e| IpcError::Transport {
            capability: method_owned.clone(),
            reason: format!("write: {e}"),
        })?;
    stream.flush().map_err(|e| IpcError::Transport {
        capability: method_owned.clone(),
        reason: format!("flush: {e}"),
    })?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| IpcError::Transport {
            capability: method_owned.clone(),
            reason: format!("read: {e}"),
        })?;

    let resp: serde_json::Value =
        serde_json::from_str(line.trim()).map_err(|e| IpcError::Protocol {
            capability: method_owned.clone(),
            reason: format!("parse: {e}"),
        })?;

    if let Some(err) = resp.get("error") {
        return Err(IpcError::Protocol {
            capability: method_owned,
            reason: format!("RPC error: {err}"),
        });
    }

    resp.get("result")
        .cloned()
        .ok_or_else(|| IpcError::Protocol {
            capability: method_owned,
            reason: "response missing 'result' field".into(),
        })
}

/// Probe a primal's `health.liveness` endpoint.
///
/// Returns `Ok(())` if the primal responds, `Err` with reason otherwise.
///
/// # Errors
///
/// Returns an error if the primal is unreachable or responds with an error.
pub fn probe_liveness(socket: &Path, timeout: Duration) -> Result<(), IpcError> {
    json_rpc_call(socket, "health.liveness", &serde_json::json!({}), timeout)?;
    Ok(())
}

/// Probe a primal's `capability.list` endpoint.
///
/// Tries canonical `capability.list` first, then falls back to
/// legacy `capabilities.list` (plural) for older primals.
///
/// # Errors
///
/// Returns an error if the primal is unreachable or doesn't advertise.
pub fn probe_capabilities(socket: &Path, timeout: Duration) -> Result<Vec<String>, IpcError> {
    let result = json_rpc_call(
        socket,
        crate::capabilities::CAPABILITY_LIST,
        &serde_json::json!({}),
        timeout,
    )
    .or_else(|_| json_rpc_call(socket, "capabilities.list", &serde_json::json!({}), timeout))?;

    let caps = result
        .get("capabilities")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    Ok(caps)
}

/// Call a primal capability by name and return the raw result.
///
/// # Errors
///
/// Returns an error if the call fails.
pub fn call_capability(
    socket: &Path,
    capability: &str,
    params: &serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value, IpcError> {
    json_rpc_call(socket, capability, params, timeout)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;
    use std::path::PathBuf;

    fn mock_rpc_socket(response_body: &str) -> (PathBuf, std::thread::JoinHandle<()>) {
        let dir = std::env::temp_dir().join(format!(
            "ns_comp_rpc_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let sock_path = dir.join("mock.sock");
        let _ = std::fs::remove_file(&sock_path);
        let listener = UnixListener::bind(&sock_path).expect("bind mock socket");
        listener
            .set_nonblocking(false)
            .expect("listener blocking mode");
        let response = format!("{response_body}\n");
        let handle = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
                let mut reader = BufReader::new(&stream);
                let mut line = String::new();
                reader.read_line(&mut line).ok();
                let request: serde_json::Value =
                    serde_json::from_str(line.trim()).expect("valid request JSON");
                assert_eq!(request["jsonrpc"], "2.0");
                assert_eq!(request["id"], 1);
                stream
                    .write_all(response.as_bytes())
                    .expect("write response");
            }
        });
        (sock_path, handle)
    }

    fn cleanup_mock_socket(sock_path: &std::path::Path) {
        std::fs::remove_file(sock_path).ok();
        if let Some(parent) = sock_path.parent() {
            std::fs::remove_dir(parent).ok();
        }
    }

    #[test]
    fn json_rpc_call_connection_refused() {
        let err = json_rpc_call(
            std::path::Path::new("/nonexistent/neuralspring.sock"),
            "health.liveness",
            &serde_json::json!({}),
            Duration::from_millis(100),
        )
        .expect_err("missing socket");
        assert!(matches!(err, IpcError::Transport { .. }));
        assert!(err.to_string().contains("connect"));
    }

    #[test]
    fn json_rpc_call_success_returns_result() {
        let (sock, handle) = mock_rpc_socket(r#"{"jsonrpc":"2.0","result":{"ok":true},"id":1}"#);
        std::thread::yield_now();
        let result = json_rpc_call(
            &sock,
            "health.liveness",
            &serde_json::json!({}),
            Duration::from_secs(5),
        )
        .expect("rpc success");
        assert_eq!(result["ok"], true);
        handle.join().expect("mock thread");
        cleanup_mock_socket(&sock);
    }

    #[test]
    fn json_rpc_call_parse_error() {
        let (sock, handle) = mock_rpc_socket("not-json");
        std::thread::yield_now();
        let err = json_rpc_call(
            &sock,
            "health.liveness",
            &serde_json::json!({}),
            Duration::from_secs(5),
        )
        .expect_err("invalid response");
        assert!(matches!(err, IpcError::Protocol { .. }));
        assert!(err.to_string().contains("parse"));
        handle.join().expect("mock thread");
        cleanup_mock_socket(&sock);
    }

    #[test]
    fn json_rpc_call_rpc_error_field() {
        let (sock, handle) = mock_rpc_socket(
            r#"{"jsonrpc":"2.0","error":{"code":-32601,"message":"method not found"},"id":1}"#,
        );
        std::thread::yield_now();
        let err = json_rpc_call(
            &sock,
            "missing.method",
            &serde_json::json!({}),
            Duration::from_secs(5),
        )
        .expect_err("rpc error");
        assert!(matches!(err, IpcError::Protocol { .. }));
        assert!(err.to_string().contains("RPC error"));
        handle.join().expect("mock thread");
        cleanup_mock_socket(&sock);
    }

    #[test]
    fn json_rpc_call_missing_result_field() {
        let (sock, handle) = mock_rpc_socket(r#"{"jsonrpc":"2.0","id":1}"#);
        std::thread::yield_now();
        let err = json_rpc_call(
            &sock,
            "health.liveness",
            &serde_json::json!({}),
            Duration::from_secs(5),
        )
        .expect_err("missing result");
        assert!(matches!(err, IpcError::Protocol { .. }));
        assert!(err.to_string().contains("missing 'result'"));
        handle.join().expect("mock thread");
        cleanup_mock_socket(&sock);
    }

    #[test]
    fn probe_liveness_connection_refused() {
        let err = probe_liveness(
            std::path::Path::new("/nonexistent/primal.sock"),
            Duration::from_millis(100),
        )
        .expect_err("unreachable");
        assert!(matches!(err, IpcError::Transport { .. }));
    }

    #[test]
    fn probe_capabilities_parses_capability_list() {
        let (sock, handle) = mock_rpc_socket(
            r#"{"jsonrpc":"2.0","result":{"capabilities":["stats.mean","tensor.matmul"]},"id":1}"#,
        );
        std::thread::yield_now();
        let caps = probe_capabilities(&sock, Duration::from_secs(5)).expect("capabilities");
        assert_eq!(caps, vec!["stats.mean", "tensor.matmul"]);
        handle.join().expect("mock thread");
        cleanup_mock_socket(&sock);
    }

    #[test]
    fn probe_capabilities_empty_when_malformed() {
        let (sock, handle) =
            mock_rpc_socket(r#"{"jsonrpc":"2.0","result":{"capabilities":"not-an-array"},"id":1}"#);
        std::thread::yield_now();
        let caps = probe_capabilities(&sock, Duration::from_secs(5)).expect("empty caps");
        assert!(caps.is_empty());
        handle.join().expect("mock thread");
        cleanup_mock_socket(&sock);
    }

    #[test]
    fn call_capability_forwards_method() {
        let (sock, handle) = mock_rpc_socket(r#"{"jsonrpc":"2.0","result":{"value":42.5},"id":1}"#);
        std::thread::yield_now();
        let result = call_capability(
            &sock,
            "science.ipr",
            &serde_json::json!({"wavefunction": [0.5, 0.5]}),
            Duration::from_secs(5),
        )
        .expect("call");
        let expected = 42.5_f64;
        assert!((result["value"].as_f64().expect("f64") - expected).abs() < 1e-9);
        handle.join().expect("mock thread");
        cleanup_mock_socket(&sock);
    }

    #[test]
    #[serial_test::serial]
    fn probe_capabilities_legacy_fallback() {
        let dir = std::env::temp_dir().join(format!(
            "ns_legacy_caps_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let sock_path = dir.join("mock.sock");
        let _ = std::fs::remove_file(&sock_path);
        let listener = UnixListener::bind(&sock_path).expect("bind");
        let handle = std::thread::spawn(move || {
            for _ in 0..2 {
                if let Ok((mut stream, _)) = listener.accept() {
                    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
                    let mut reader = BufReader::new(&stream);
                    let mut line = String::new();
                    reader.read_line(&mut line).ok();
                    let request: serde_json::Value =
                        serde_json::from_str(line.trim()).expect("request json");
                    let method = request["method"].as_str().unwrap_or("");
                    let response = if method == crate::capabilities::CAPABILITY_LIST {
                        r#"{"jsonrpc":"2.0","error":{"code":-32601,"message":"not found"},"id":1}"#
                    } else {
                        r#"{"jsonrpc":"2.0","result":{"capabilities":["legacy.cap"]},"id":1}"#
                    };
                    stream
                        .write_all(format!("{response}\n").as_bytes())
                        .expect("write");
                }
            }
        });
        std::thread::yield_now();
        let caps = probe_capabilities(&sock_path, Duration::from_secs(5)).expect("legacy caps");
        assert_eq!(caps, vec!["legacy.cap"]);
        handle.join().expect("mock thread");
        cleanup_mock_socket(&sock_path);
    }

    #[test]
    fn json_rpc_call_serializes_method_and_params() {
        let (sock, handle) = mock_rpc_socket(r#"{"jsonrpc":"2.0","result":{},"id":1}"#);
        std::thread::yield_now();
        let params = serde_json::json!({"dim": 16, "seed": 42});
        json_rpc_call(
            &sock,
            "science.spectral_analysis",
            &params,
            Duration::from_secs(5),
        )
        .expect("rpc");
        handle.join().expect("mock thread");
        cleanup_mock_socket(&sock);
    }
}
