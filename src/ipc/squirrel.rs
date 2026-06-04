// SPDX-License-Identifier: AGPL-3.0-or-later

//! Squirrel IPC surface — inference routing.
//!
//! Methods: `inference.complete`, `inference.embed`, `inference.models`.

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
}
