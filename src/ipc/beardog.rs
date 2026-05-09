// SPDX-License-Identifier: AGPL-3.0-or-later

//! BearDog IPC surface — cryptographic operations.
//!
//! Methods: `crypto.hash`.

use std::path::PathBuf;
use std::time::Duration;

use crate::validation::composition::call_capability;

/// `crypto.hash` via BearDog IPC.
pub fn crypto_hash(
    socket: &PathBuf,
    algorithm: &str,
    data: &str,
    timeout: Duration,
) -> Result<String, String> {
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
        .ok_or_else(|| "crypto.hash: response missing hash string".to_string())
}
