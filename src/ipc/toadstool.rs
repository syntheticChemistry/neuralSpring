// SPDX-License-Identifier: AGPL-3.0-or-later

//! toadStool IPC surface — compute dispatch + Tier 2 Science API.
//!
//! Methods: `compute.dispatch`, `toadstool.validate`, `toadstool.list_workloads`.

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

/// Pre-flight validation result from `toadstool.validate`.
#[derive(Debug, Clone)]
pub struct ValidateResult {
    /// Whether the workload TOML is structurally valid.
    pub valid: bool,
    /// Whether GPU compute is available for dispatch.
    pub gpu_available: bool,
    /// Recommended precision tier (`"DF64"`, `"FP32"`, or `"none"`).
    pub precision_tier: String,
    /// Estimated dispatch time in milliseconds.
    pub estimated_dispatch_time_ms: u64,
    /// Advisory warnings (non-fatal).
    pub warnings: Vec<String>,
    /// Capabilities required by this workload.
    pub required_capabilities: Vec<String>,
}

impl ValidateResult {
    fn from_json(v: &serde_json::Value) -> Self {
        Self {
            valid: v.get("valid").and_then(serde_json::Value::as_bool).unwrap_or(false),
            gpu_available: v.get("gpu_available").and_then(serde_json::Value::as_bool).unwrap_or(false),
            precision_tier: v.get("precision_tier")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("none")
                .to_owned(),
            estimated_dispatch_time_ms: v.get("estimated_dispatch_time_ms")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            warnings: v.get("warnings")
                .and_then(serde_json::Value::as_array)
                .map(|arr| arr.iter().filter_map(|w| w.as_str().map(String::from)).collect())
                .unwrap_or_default(),
            required_capabilities: v.get("required_capabilities")
                .and_then(serde_json::Value::as_array)
                .map(|arr| arr.iter().filter_map(|c| c.as_str().map(String::from)).collect())
                .unwrap_or_default(),
        }
    }
}

/// `toadstool.validate` — Tier 2 workload pre-flight validation.
///
/// Validates a workload TOML before dispatch: checks file validity,
/// GPU availability, precision tier, estimated dispatch time, and
/// required capabilities.
///
/// # Errors
///
/// Returns an error if toadStool is not reachable or the IPC call fails.
pub fn validate(
    socket: &Path,
    workload_path: &str,
    dry_run: bool,
    timeout: Duration,
) -> Result<ValidateResult, IpcError> {
    let params = serde_json::json!({
        "workload_path": workload_path,
        "dry_run": dry_run,
    });
    let response = call_capability(socket, capabilities::TOADSTOOL_VALIDATE, &params, timeout)?;
    Ok(ValidateResult::from_json(&response))
}

/// `toadstool.list_workloads` — list available workloads.
///
/// # Errors
///
/// Returns an error if toadStool is not reachable or the IPC call fails.
pub fn list_workloads(
    socket: &Path,
    timeout: Duration,
) -> Result<serde_json::Value, IpcError> {
    Ok(call_capability(socket, capabilities::TOADSTOOL_LIST_WORKLOADS, &serde_json::json!({}), timeout)?)
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

    #[test]
    fn validate_returns_err_for_nonexistent_socket() {
        let result = validate(
            Path::new("/nonexistent/toadstool.sock"),
            "/tmp/test.toml",
            true,
            Duration::from_millis(100),
        );
        assert!(result.is_err());
    }

    #[test]
    fn list_workloads_returns_err_for_nonexistent_socket() {
        let result = list_workloads(
            Path::new("/nonexistent/toadstool.sock"),
            Duration::from_millis(100),
        );
        assert!(result.is_err());
    }

    #[test]
    fn validate_result_from_json_full() {
        let v = serde_json::json!({
            "valid": true,
            "gpu_available": true,
            "precision_tier": "DF64",
            "estimated_dispatch_time_ms": 100,
            "warnings": ["experimental workload"],
            "required_capabilities": ["compute.dispatch", "precision.routing"],
            "dry_run": true,
        });
        let result = ValidateResult::from_json(&v);
        assert!(result.valid);
        assert!(result.gpu_available);
        assert_eq!(result.precision_tier, "DF64");
        assert_eq!(result.estimated_dispatch_time_ms, 100);
        assert_eq!(result.warnings, vec!["experimental workload"]);
        assert_eq!(result.required_capabilities, vec!["compute.dispatch", "precision.routing"]);
    }

    #[test]
    fn validate_result_from_json_minimal() {
        let v = serde_json::json!({});
        let result = ValidateResult::from_json(&v);
        assert!(!result.valid);
        assert!(!result.gpu_available);
        assert_eq!(result.precision_tier, "none");
        assert_eq!(result.estimated_dispatch_time_ms, 0);
        assert!(result.warnings.is_empty());
        assert!(result.required_capabilities.is_empty());
    }
}
