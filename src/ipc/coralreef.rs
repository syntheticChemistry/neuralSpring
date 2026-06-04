// SPDX-License-Identifier: AGPL-3.0-or-later

//! coralReef IPC surface — shader compilation.
//!
//! Methods: `shader.compile.wgsl`, `shader.compile.capabilities`.

use std::path::Path;
use std::time::Duration;

use crate::capabilities;
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
    call_capability(
        socket,
        capabilities::SHADER_COMPILE_WGSL,
        &serde_json::json!({ "source": source, "label": label }),
        timeout,
    )
}

/// `shader.compile.capabilities` via coralReef IPC.
///
/// # Errors
///
/// Returns an error if coralReef is not reachable or the IPC call fails.
pub fn shader_capabilities(socket: &Path, timeout: Duration) -> Result<serde_json::Value, IpcError> {
    call_capability(
        socket,
        capabilities::SHADER_COMPILE_CAPABILITIES,
        &serde_json::json!({}),
        timeout,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::Duration;

    const TIMEOUT: Duration = Duration::from_millis(100);
    const FAKE_SOCKET: &str = "/nonexistent/coralreef.sock";

    #[test]
    fn shader_compile_wgsl_returns_err_for_nonexistent_socket() {
        let result = shader_compile_wgsl(
            Path::new(FAKE_SOCKET),
            "@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }",
            "test.wgsl",
            TIMEOUT,
        );
        assert!(result.is_err());
    }

    #[test]
    fn shader_capabilities_returns_err_for_nonexistent_socket() {
        let result = shader_capabilities(Path::new(FAKE_SOCKET), TIMEOUT);
        assert!(result.is_err());
    }
}
