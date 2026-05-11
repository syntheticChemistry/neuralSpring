// SPDX-License-Identifier: AGPL-3.0-or-later

//! coralReef IPC surface — shader compilation.
//!
//! Methods: `shader.compile.wgsl`, `shader.compile.capabilities`.

use std::path::Path;
use std::time::Duration;

use crate::error::IpcError;
use crate::validation::composition::call_capability;

/// `shader.compile.wgsl` via coralReef IPC.
///
/// # Errors
///
/// Returns an error if coralReef is not reachable or the IPC call fails.
pub fn shader_compile_wgsl(
    socket: &Path,
    source: &str,
    label: &str,
    timeout: Duration,
) -> Result<serde_json::Value, IpcError> {
    Ok(call_capability(
        socket,
        "shader.compile.wgsl",
        &serde_json::json!({ "source": source, "label": label }),
        timeout,
    )?)
}

/// `shader.compile.capabilities` via coralReef IPC.
///
/// # Errors
///
/// Returns an error if coralReef is not reachable or the IPC call fails.
pub fn shader_capabilities(socket: &Path, timeout: Duration) -> Result<serde_json::Value, IpcError> {
    Ok(call_capability(
        socket,
        "shader.compile.capabilities",
        &serde_json::json!({}),
        timeout,
    )?)
}
