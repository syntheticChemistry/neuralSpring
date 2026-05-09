// SPDX-License-Identifier: AGPL-3.0-or-later

//! coralReef IPC surface — shader compilation.
//!
//! Methods: `shader.compile.wgsl`, `shader.compile.capabilities`.

use std::path::PathBuf;
use std::time::Duration;

use crate::validation::composition::call_capability;

/// `shader.compile.wgsl` via coralReef IPC.
pub fn shader_compile_wgsl(
    socket: &PathBuf,
    source: &str,
    label: &str,
    timeout: Duration,
) -> Result<serde_json::Value, String> {
    call_capability(
        socket,
        "shader.compile.wgsl",
        &serde_json::json!({ "source": source, "label": label }),
        timeout,
    )
}

/// `shader.compile.capabilities` via coralReef IPC.
pub fn shader_capabilities(
    socket: &PathBuf,
    timeout: Duration,
) -> Result<serde_json::Value, String> {
    call_capability(
        socket,
        "shader.compile.capabilities",
        &serde_json::json!({}),
        timeout,
    )
}
