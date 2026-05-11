// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BearDog` IPC surface — cryptographic operations.
//!
//! Methods: `crypto.hash`.

use std::path::Path;
use std::time::Duration;

use crate::error::IpcError;
use crate::validation::composition::call_capability;

/// `crypto.hash` via `BearDog` IPC.
///
/// # Errors
///
/// Returns an error if `BearDog` is not reachable or the response is malformed.
pub fn crypto_hash(
    socket: &Path,
    algorithm: &str,
    data: &str,
    timeout: Duration,
) -> Result<String, IpcError> {
    let result = call_capability(
        socket,
        "crypto.hash",
        &serde_json::json!({ "algorithm": algorithm, "data": data }),
        timeout,
    )?;
    result
        .get("hash")
        .or_else(|| result.get("digest"))
        .or_else(|| result.get("result"))
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| IpcError::Protocol {
            capability: "crypto.hash".into(),
            reason: "response missing hash string".into(),
        })
}
