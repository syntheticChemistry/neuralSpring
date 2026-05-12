// SPDX-License-Identifier: AGPL-3.0-or-later

//! NestGate IPC surface — content-addressed storage for model weights.
//!
//! Methods: `content.put`, `content.get`, `content.exists`.
//!
//! NestGate provides content-addressed storage with BLAKE3 hash-as-key
//! and automatic deduplication. Model weights are stored as base64-encoded
//! blobs; the returned hash serves as a stable identifier for retrieval
//! across sessions and springs.

use std::path::Path;
use std::time::Duration;

use crate::error::IpcError;
use crate::validation::composition::call_capability;

/// Store content-addressed data via NestGate `content.put`.
///
/// The payload is base64-encoded before sending. NestGate returns a
/// BLAKE3 hash (64-char hex) that serves as the retrieval key.
///
/// # Errors
///
/// Returns an error if the IPC call to NestGate fails or times out.
pub fn content_put(
    socket: &Path,
    data_base64: &str,
    content_type: Option<&str>,
    timeout: Duration,
) -> Result<ContentPutResult, IpcError> {
    let mut params = serde_json::json!({
        "data": data_base64,
    });
    if let Some(ct) = content_type {
        params["content_type"] = serde_json::Value::String(ct.to_owned());
    }

    let resp = call_capability(socket, "content.put", &params, timeout)?;

    let hash = resp
        .get("hash")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let size = resp
        .get("size")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let deduplicated = resp
        .get("deduplicated")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    Ok(ContentPutResult {
        hash,
        size,
        deduplicated,
    })
}

/// Retrieve content-addressed data via NestGate `content.get`.
///
/// Takes a BLAKE3 hash (64-char hex) and returns the base64-encoded payload.
///
/// # Errors
///
/// Returns an error if the hash is not found, or the IPC call fails.
pub fn content_get(
    socket: &Path,
    hash: &str,
    timeout: Duration,
) -> Result<ContentGetResult, IpcError> {
    let params = serde_json::json!({
        "hash": hash,
    });

    let resp = call_capability(socket, "content.get", &params, timeout)?;

    let data = resp
        .get("data")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let size = resp
        .get("size")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let content_type = resp
        .get("content_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);

    Ok(ContentGetResult {
        data,
        hash: hash.to_owned(),
        size,
        content_type,
    })
}

/// Check whether content exists via NestGate `content.exists`.
///
/// # Errors
///
/// Returns an error if the IPC call fails.
pub fn content_exists(
    socket: &Path,
    hash: &str,
    timeout: Duration,
) -> Result<bool, IpcError> {
    let params = serde_json::json!({
        "hash": hash,
    });

    let resp = call_capability(socket, "content.exists", &params, timeout)?;

    Ok(resp
        .get("exists")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false))
}

/// Result from a `content.put` call.
#[derive(Debug, Clone)]
pub struct ContentPutResult {
    /// BLAKE3 hash (64-char hex) — the retrieval key.
    pub hash: String,
    /// Size of the stored content in bytes.
    pub size: u64,
    /// Whether the content was already present (deduplicated).
    pub deduplicated: bool,
}

/// Result from a `content.get` call.
#[derive(Debug, Clone)]
pub struct ContentGetResult {
    /// Base64-encoded content payload.
    pub data: String,
    /// BLAKE3 hash that was requested.
    pub hash: String,
    /// Size of the content in bytes.
    pub size: u64,
    /// MIME content type if set during `content.put`.
    pub content_type: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::Duration;

    const TIMEOUT: Duration = Duration::from_millis(100);
    const FAKE_SOCKET: &str = "/nonexistent/nestgate.sock";

    #[test]
    fn content_put_returns_err_for_nonexistent_socket() {
        let result = content_put(Path::new(FAKE_SOCKET), "dGVzdA==", None, TIMEOUT);
        assert!(result.is_err());
    }

    #[test]
    fn content_put_with_content_type_returns_err_for_nonexistent_socket() {
        let result = content_put(
            Path::new(FAKE_SOCKET),
            "dGVzdA==",
            Some("application/octet-stream"),
            TIMEOUT,
        );
        assert!(result.is_err());
    }

    #[test]
    fn content_get_returns_err_for_nonexistent_socket() {
        let result = content_get(
            Path::new(FAKE_SOCKET),
            "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            TIMEOUT,
        );
        assert!(result.is_err());
    }

    #[test]
    fn content_exists_returns_err_for_nonexistent_socket() {
        let result = content_exists(
            Path::new(FAKE_SOCKET),
            "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            TIMEOUT,
        );
        assert!(result.is_err());
    }

    #[test]
    fn content_put_result_debug_and_clone() {
        let r = ContentPutResult {
            hash: "abc123".into(),
            size: 42,
            deduplicated: true,
        };
        let cloned = r.clone();
        assert_eq!(format!("{r:?}"), format!("{cloned:?}"));
        assert!(cloned.deduplicated);
    }

    #[test]
    fn content_get_result_debug_and_clone() {
        let r = ContentGetResult {
            data: "dGVzdA==".into(),
            hash: "abc123".into(),
            size: 4,
            content_type: Some("text/plain".into()),
        };
        let cloned = r.clone();
        assert_eq!(cloned.content_type.as_deref(), Some("text/plain"));
        assert_eq!(format!("{r:?}"), format!("{cloned:?}"));
    }
}
