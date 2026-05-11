// SPDX-License-Identifier: AGPL-3.0-or-later

//! skunkBat IPC surface — audit logging and threat detection.
//!
//! Methods: `security.audit_log`.

use std::path::Path;
use std::time::Duration;

use crate::capabilities;
use crate::error::IpcError;
use crate::validation::composition::call_capability;

/// `security.audit_log` via skunkBat IPC.
///
/// Forwards an audit event to skunkBat for cross-primal logging into
/// rhizoCrypt DAG + sweetGrass braid (JH-5 forwarding).
///
/// # Errors
///
/// Returns an error if the IPC call to skunkBat fails or times out.
pub fn audit_log(
    socket: &Path,
    event_type: &str,
    source: &str,
    payload: &serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value, IpcError> {
    Ok(call_capability(
        socket,
        capabilities::SECURITY_AUDIT_LOG,
        &serde_json::json!({
            "event_type": event_type,
            "source": source,
            "payload": payload,
            "timestamp": chrono_timestamp(),
        }),
        timeout,
    )?)
}

fn chrono_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{now}")
}
