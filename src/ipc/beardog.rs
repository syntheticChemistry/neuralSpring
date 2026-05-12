// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BearDog` IPC surface — cryptographic operations.
//!
//! Methods: `crypto.hash`.

use std::path::Path;
use std::time::Duration;

use crate::capabilities;
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
        capabilities::CRYPTO_HASH,
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
            capability: capabilities::CRYPTO_HASH.into(),
            reason: "response missing hash string".into(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::Duration;

    #[test]
    fn crypto_hash_returns_err_for_nonexistent_socket() {
        let result = crypto_hash(
            Path::new("/nonexistent/beardog.sock"),
            "blake3",
            "hello world",
            Duration::from_millis(100),
        );
        assert!(result.is_err());
    }
}
