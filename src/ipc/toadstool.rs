// SPDX-License-Identifier: AGPL-3.0-or-later

//! toadStool IPC surface — compute dispatch.
//!
//! Methods: `compute.dispatch`.

use std::path::Path;
use std::time::Duration;

use crate::capabilities;
use crate::error::IpcError;
use crate::validation::composition::call_capability;

/// `compute.dispatch` via toadStool IPC.
///
/// # Errors
///
/// Returns an error if toadStool is not reachable or the IPC call fails.
pub fn compute_dispatch(
    socket: &Path,
    params: &serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value, IpcError> {
    Ok(call_capability(socket, capabilities::COMPUTE_DISPATCH, params, timeout)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::Duration;

    #[test]
    fn compute_dispatch_returns_err_for_nonexistent_socket() {
        let result = compute_dispatch(
            Path::new("/nonexistent/toadstool.sock"),
            &serde_json::json!({"workload": "test"}),
            Duration::from_millis(100),
        );
        assert!(result.is_err());
    }
}
