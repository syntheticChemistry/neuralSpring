// SPDX-License-Identifier: AGPL-3.0-or-later

//! Squirrel IPC surface — inference routing and provider lifecycle.
//!
//! Methods: `inference.complete`, `inference.embed`, `inference.models`,
//! `inference.register_provider`, `inference.unregister_provider`.

use std::path::Path;
use std::time::Duration;

use crate::capabilities;
use crate::error::IpcError;
use crate::validation::composition::call_capability;

/// `inference.complete` via Squirrel IPC.
///
/// # Errors
///
/// Returns an error if Squirrel is not reachable or the IPC call fails.
pub fn inference_complete(
    socket: &Path,
    params: &serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value, IpcError> {
    call_capability(socket, capabilities::INFERENCE_COMPLETE, params, timeout)
}

/// `inference.embed` via Squirrel IPC.
///
/// # Errors
///
/// Returns an error if Squirrel is not reachable or the IPC call fails.
pub fn inference_embed(
    socket: &Path,
    params: &serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value, IpcError> {
    call_capability(socket, capabilities::INFERENCE_EMBED, params, timeout)
}

/// `inference.models` via Squirrel IPC.
///
/// # Errors
///
/// Returns an error if Squirrel is not reachable or the IPC call fails.
pub fn inference_models(socket: &Path, timeout: Duration) -> Result<serde_json::Value, IpcError> {
    call_capability(socket, capabilities::INFERENCE_MODELS, &serde_json::json!({}), timeout)
}

/// Register as an inference provider with Squirrel.
///
/// `provider_id` identifies this provider (e.g. `"neuralspring"`).
/// `socket_path` is the UDS path where Squirrel can reach us.
/// `supported_capabilities` lists what this provider can do (e.g.
/// `["completion", "embedding"]`).
///
/// # Errors
///
/// Returns an error if Squirrel is not reachable or registration fails.
pub fn register_provider(
    socket: &Path,
    provider_id: &str,
    socket_path: &str,
    supported_capabilities: &[&str],
    timeout: Duration,
) -> Result<serde_json::Value, IpcError> {
    let params = serde_json::json!({
        "provider_id": provider_id,
        "socket": socket_path,
        "capabilities": {
            "supported_tasks": supported_capabilities,
        },
    });
    call_capability(socket, capabilities::INFERENCE_REGISTER_PROVIDER, &params, timeout)
}

/// Unregister an inference provider from Squirrel.
///
/// # Errors
///
/// Returns an error if Squirrel is not reachable or unregistration fails.
pub fn unregister_provider(
    socket: &Path,
    provider_id: &str,
    timeout: Duration,
) -> Result<serde_json::Value, IpcError> {
    let params = serde_json::json!({ "provider_id": provider_id });
    call_capability(socket, capabilities::INFERENCE_UNREGISTER_PROVIDER, &params, timeout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::Duration;

    const TIMEOUT: Duration = Duration::from_millis(100);
    const FAKE_SOCKET: &str = "/nonexistent/squirrel.sock";

    #[test]
    fn inference_complete_returns_err_for_nonexistent_socket() {
        let result = inference_complete(
            Path::new(FAKE_SOCKET),
            &serde_json::json!({"prompt": "test"}),
            TIMEOUT,
        );
        assert!(result.is_err());
    }

    #[test]
    fn inference_embed_returns_err_for_nonexistent_socket() {
        let result = inference_embed(
            Path::new(FAKE_SOCKET),
            &serde_json::json!({"text": "test"}),
            TIMEOUT,
        );
        assert!(result.is_err());
    }

    #[test]
    fn inference_models_returns_err_for_nonexistent_socket() {
        let result = inference_models(Path::new(FAKE_SOCKET), TIMEOUT);
        assert!(result.is_err());
    }

    #[test]
    fn register_provider_returns_err_for_nonexistent_socket() {
        let result = register_provider(
            Path::new(FAKE_SOCKET),
            "neuralspring",
            "/run/user/1000/biomeos/neuralspring.sock",
            &["completion", "embedding"],
            TIMEOUT,
        );
        assert!(result.is_err());
    }

    #[test]
    fn unregister_provider_returns_err_for_nonexistent_socket() {
        let result = unregister_provider(Path::new(FAKE_SOCKET), "neuralspring", TIMEOUT);
        assert!(result.is_err());
    }
}
