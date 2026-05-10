// SPDX-License-Identifier: AGPL-3.0-or-later

//! Squirrel IPC surface — inference routing.
//!
//! Methods: `inference.complete`, `inference.embed`, `inference.models`.

use std::path::Path;
use std::time::Duration;

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
) -> Result<serde_json::Value, String> {
    call_capability(socket, "inference.complete", params, timeout)
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
) -> Result<serde_json::Value, String> {
    call_capability(socket, "inference.embed", params, timeout)
}

/// `inference.models` via Squirrel IPC.
///
/// # Errors
///
/// Returns an error if Squirrel is not reachable or the IPC call fails.
pub fn inference_models(socket: &Path, timeout: Duration) -> Result<serde_json::Value, String> {
    call_capability(socket, "inference.models", &serde_json::json!({}), timeout)
}
