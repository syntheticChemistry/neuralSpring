// SPDX-License-Identifier: AGPL-3.0-or-later

//! toadStool IPC surface — compute dispatch.
//!
//! Methods: `compute.dispatch`.

use std::path::PathBuf;
use std::time::Duration;

use crate::validation::composition::call_capability;

/// `compute.dispatch` via toadStool IPC.
pub fn compute_dispatch(
    socket: &PathBuf,
    params: &serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value, String> {
    call_capability(socket, "compute.dispatch", params, timeout)
}
