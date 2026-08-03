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
    call_capability(socket, capabilities::COMPUTE_DISPATCH, params, timeout)
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
            valid: v
                .get("valid")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            gpu_available: v
                .get("gpu_available")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            precision_tier: v
                .get("precision_tier")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("none")
                .to_owned(),
            estimated_dispatch_time_ms: v
                .get("estimated_dispatch_time_ms")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            warnings: v
                .get("warnings")
                .and_then(serde_json::Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|w| w.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            required_capabilities: v
                .get("required_capabilities")
                .and_then(serde_json::Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|c| c.as_str().map(String::from))
                        .collect()
                })
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
pub fn list_workloads(socket: &Path, timeout: Duration) -> Result<serde_json::Value, IpcError> {
    call_capability(
        socket,
        capabilities::TOADSTOOL_LIST_WORKLOADS,
        &serde_json::json!({}),
        timeout,
    )
}

/// Structured workload for typed compute dispatch.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ComputeWorkload {
    /// Science capability to dispatch (e.g. `"science.eigensolve"`).
    pub capability: String,
    /// Workload data payload (stage-specific parameters).
    pub data: serde_json::Value,
    /// Substrate hint for routing (`"gpu"`, `"cpu"`, or `"auto"`).
    pub substrate_hint: String,
}

/// Result from a typed workload dispatch.
#[derive(Debug, Clone)]
pub struct WorkloadResult {
    /// Whether the dispatch succeeded.
    pub success: bool,
    /// Actual substrate used for execution.
    pub actual_substrate: String,
    /// Result payload from the computation.
    pub output: serde_json::Value,
    /// Elapsed time in microseconds (as reported by the remote).
    pub elapsed_us: f64,
}

impl WorkloadResult {
    fn from_json(v: &serde_json::Value) -> Self {
        Self {
            success: v
                .get("success")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            actual_substrate: v
                .get("actual_substrate")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown")
                .to_owned(),
            output: v.get("output").cloned().unwrap_or(serde_json::Value::Null),
            elapsed_us: v
                .get("elapsed_us")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0),
        }
    }
}

/// `compute.dispatch` with typed workload — submit a structured workload
/// (capability, data, substrate hint) and get back a structured result.
///
/// # Errors
///
/// Returns an error if toadStool is not reachable or the IPC call fails.
pub fn compute_dispatch_workload(
    socket: &Path,
    workload: &ComputeWorkload,
    timeout: Duration,
) -> Result<WorkloadResult, IpcError> {
    let params = serde_json::to_value(workload)
        .map_err(|e| IpcError::Other(format!("serialize workload: {e}")))?;
    let response = call_capability(socket, capabilities::COMPUTE_DISPATCH, &params, timeout)?;
    Ok(WorkloadResult::from_json(&response))
}

/// `compute.dispatch` with full pipeline graph — submit an entire pipeline for
/// remote execution with substrate-aware routing per stage.
///
/// # Errors
///
/// Returns an error if toadStool is not reachable or the IPC call fails.
pub fn compute_dispatch_pipeline(
    socket: &Path,
    pipeline_name: &str,
    stages: &[ComputeWorkload],
    timeout: Duration,
) -> Result<serde_json::Value, IpcError> {
    let params = serde_json::json!({
        "pipeline": pipeline_name,
        "stages": stages.iter()
            .map(|w| serde_json::to_value(w).unwrap_or_default())
            .collect::<Vec<_>>(),
        "mode": "pipeline_batch",
    });
    call_capability(socket, capabilities::COMPUTE_DISPATCH, &params, timeout)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions")]
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
        assert_eq!(
            result.required_capabilities,
            vec!["compute.dispatch", "precision.routing"]
        );
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

    #[test]
    fn compute_workload_serializes() {
        let wl = ComputeWorkload {
            capability: "science.eigensolve".to_string(),
            data: serde_json::json!({"n": 16}),
            substrate_hint: "gpu".to_string(),
        };
        let json = serde_json::to_value(&wl).unwrap();
        assert_eq!(json["capability"], "science.eigensolve");
        assert_eq!(json["substrate_hint"], "gpu");
    }

    #[test]
    fn workload_result_from_json_full() {
        let v = serde_json::json!({
            "success": true,
            "actual_substrate": "gpu",
            "output": {"eigenvalues": [1.0, 2.0]},
            "elapsed_us": 42.5,
        });
        let result = WorkloadResult::from_json(&v);
        assert!(result.success);
        assert_eq!(result.actual_substrate, "gpu");
        assert!((result.elapsed_us - 42.5).abs() < 1e-10);
    }

    #[test]
    fn workload_result_from_json_minimal() {
        let result = WorkloadResult::from_json(&serde_json::json!({}));
        assert!(!result.success);
        assert_eq!(result.actual_substrate, "unknown");
        assert!(result.output.is_null());
    }

    #[test]
    fn compute_dispatch_workload_returns_err_for_nonexistent_socket() {
        let wl = ComputeWorkload {
            capability: "science.eigensolve".to_string(),
            data: serde_json::json!({}),
            substrate_hint: "auto".to_string(),
        };
        let result = compute_dispatch_workload(
            Path::new("/nonexistent/toadstool.sock"),
            &wl,
            Duration::from_millis(100),
        );
        assert!(result.is_err());
    }

    #[test]
    fn compute_dispatch_pipeline_returns_err_for_nonexistent_socket() {
        let stages = vec![ComputeWorkload {
            capability: "science.eigensolve".to_string(),
            data: serde_json::json!({}),
            substrate_hint: "gpu".to_string(),
        }];
        let result = compute_dispatch_pipeline(
            Path::new("/nonexistent/toadstool.sock"),
            "test_pipeline",
            &stages,
            Duration::from_millis(100),
        );
        assert!(result.is_err());
    }
}
