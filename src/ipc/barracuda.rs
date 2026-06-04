// SPDX-License-Identifier: AGPL-3.0-or-later

//! barraCuda IPC surface — tensor lifecycle, core math, ML inference,
//! and precision routing.
//!
//! Methods: `stats.mean`, `stats.std_dev`, `stats.weighted_mean`,
//! `tensor.matmul`, `tensor.create`, `barracuda.precision.route`,
//! `ml.mlp_infer`.

use std::path::Path;
use std::time::Duration;

use crate::capabilities;
use crate::error::IpcError;
use crate::validation::composition::call_capability;

/// `stats.mean` via barraCuda IPC.
///
/// # Errors
///
/// Returns an error if barraCuda is not reachable or the response is malformed.
pub fn stats_mean(socket: &Path, data: &[f64], timeout: Duration) -> Result<f64, IpcError> {
    let result = call_capability(
        socket,
        capabilities::STATS_MEAN,
        &serde_json::json!({ "data": data }),
        timeout,
    )?;
    super::extract_f64(&result, &["mean", "result", "value"]).ok_or_else(|| IpcError::Protocol {
        capability: capabilities::STATS_MEAN.into(),
        reason: "response missing numeric result".into(),
    })
}

/// `stats.std_dev` via barraCuda IPC.
///
/// # Errors
///
/// Returns an error if barraCuda is not reachable or the response is malformed.
pub fn stats_std_dev(socket: &Path, data: &[f64], timeout: Duration) -> Result<f64, IpcError> {
    let result = call_capability(
        socket,
        capabilities::STATS_STD_DEV,
        &serde_json::json!({ "data": data }),
        timeout,
    )?;
    super::extract_f64(&result, &["std_dev", "result", "value"]).ok_or_else(|| {
        IpcError::Protocol {
            capability: capabilities::STATS_STD_DEV.into(),
            reason: "response missing numeric result".into(),
        }
    })
}

/// `stats.weighted_mean` via barraCuda IPC.
///
/// # Errors
///
/// Returns an error if barraCuda is not reachable or the response is malformed.
pub fn stats_weighted_mean(
    socket: &Path,
    data: &[f64],
    weights: &[f64],
    timeout: Duration,
) -> Result<f64, IpcError> {
    let result = call_capability(
        socket,
        capabilities::STATS_WEIGHTED_MEAN,
        &serde_json::json!({ "data": data, "weights": weights }),
        timeout,
    )?;
    super::extract_f64(&result, &["weighted_mean", "result", "value"]).ok_or_else(|| {
        IpcError::Protocol {
            capability: capabilities::STATS_WEIGHTED_MEAN.into(),
            reason: "response missing numeric result".into(),
        }
    })
}

/// `tensor.matmul` via barraCuda IPC.
///
/// # Errors
///
/// Returns an error if barraCuda is not reachable or the response is malformed.
pub fn tensor_matmul(
    socket: &Path,
    a: &[f64],
    b: &[f64],
    rows_a: usize,
    cols_a: usize,
    cols_b: usize,
    timeout: Duration,
) -> Result<Vec<f64>, IpcError> {
    let result = call_capability(
        socket,
        capabilities::TENSOR_MATMUL,
        &serde_json::json!({
            "a": a, "b": b,
            "rows_a": rows_a, "cols_a": cols_a, "cols_b": cols_b,
        }),
        timeout,
    )?;
    super::extract_f64_array(&result, &["data", "result"]).ok_or_else(|| IpcError::Protocol {
        capability: capabilities::TENSOR_MATMUL.into(),
        reason: "response missing data array".into(),
    })
}

/// `tensor.create` via barraCuda IPC.
///
/// # Errors
///
/// Returns an error if barraCuda is not reachable or the response is malformed.
pub fn tensor_create(
    socket: &Path,
    shape: &[usize],
    fill: &str,
    timeout: Duration,
) -> Result<serde_json::Value, IpcError> {
    call_capability(
        socket,
        capabilities::TENSOR_CREATE,
        &serde_json::json!({ "shape": shape, "fill": fill }),
        timeout,
    )
}

/// Parsed result from `barracuda.precision.route`.
#[derive(Debug)]
pub struct PrecisionRouteResult {
    /// Recommended precision tier (e.g. `"f32"`, `"f64"`, `"DF64"`).
    pub recommended_tier: String,
    /// Whether fused multiply-add is safe for this domain.
    pub fma_safe: bool,
    /// Whether the sovereign shader compiler is required.
    pub requires_compiler: bool,
    /// Hardware hint echoed back (advisory).
    pub hardware_hint: String,
    /// Human-readable rationale for the recommendation.
    pub rationale: Option<String>,
}

impl PrecisionRouteResult {
    fn from_value(v: &serde_json::Value) -> Self {
        Self {
            recommended_tier: v
                .get("recommended_tier")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("f64")
                .to_owned(),
            fma_safe: v
                .get("fma_safe")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            requires_compiler: v
                .get("requires_compiler")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            hardware_hint: v
                .get("hardware_hint")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned(),
            rationale: v
                .get("rationale")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
        }
    }
}

/// `barracuda.precision.route` via barraCuda IPC.
///
/// Queries the optimal precision strategy for a given domain operation.
/// Returns the recommended tier, FMA safety, compiler requirement, and
/// optional rationale.
///
/// # Errors
///
/// Returns an error if barraCuda is not reachable or the response is malformed.
pub fn precision_route(
    socket: &Path,
    domain: &str,
    hardware_hint: Option<&str>,
    timeout: Duration,
) -> Result<PrecisionRouteResult, IpcError> {
    let mut params = serde_json::json!({ "domain": domain });
    if let Some(hint) = hardware_hint {
        params["hardware_hint"] = serde_json::Value::String(hint.to_owned());
    }
    let result = call_capability(
        socket,
        capabilities::PRECISION_ROUTE,
        &params,
        timeout,
    )?;
    Ok(PrecisionRouteResult::from_value(&result))
}

/// `ml.mlp_infer` via barraCuda IPC — MLP forward pass.
///
/// Sends input data and layer specifications (weights, biases, dimensions)
/// to barraCuda for a full MLP forward inference. This capability enables
/// cross-gate dispatch: a neuralSpring on one gate can route MLP inference
/// to a barraCuda instance on another gate via Songbird mesh.
///
/// # Errors
///
/// Returns an error if barraCuda is not reachable or the response is malformed.
pub fn ml_mlp_infer(
    socket: &Path,
    input: &[f64],
    input_dim: usize,
    hidden_dims: &[usize],
    output_dim: usize,
    timeout: Duration,
) -> Result<Vec<f64>, IpcError> {
    let result = call_capability(
        socket,
        capabilities::ML_MLP_INFER,
        &serde_json::json!({
            "input": input,
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
        }),
        timeout,
    )?;
    super::extract_f64_array(&result, &["output", "data", "result"]).ok_or_else(|| {
        IpcError::Protocol {
            capability: capabilities::ML_MLP_INFER.into(),
            reason: "response missing output array".into(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::Duration;

    const TIMEOUT: Duration = Duration::from_millis(100);
    const FAKE_SOCKET: &str = "/nonexistent/barracuda.sock";

    #[test]
    fn stats_mean_returns_err_for_nonexistent_socket() {
        let result = stats_mean(Path::new(FAKE_SOCKET), &[1.0, 2.0, 3.0], TIMEOUT);
        assert!(result.is_err());
    }

    #[test]
    fn stats_std_dev_returns_err_for_nonexistent_socket() {
        let result = stats_std_dev(Path::new(FAKE_SOCKET), &[1.0, 2.0, 3.0], TIMEOUT);
        assert!(result.is_err());
    }

    #[test]
    fn stats_weighted_mean_returns_err_for_nonexistent_socket() {
        let result = stats_weighted_mean(
            Path::new(FAKE_SOCKET),
            &[1.0, 2.0],
            &[0.5, 0.5],
            TIMEOUT,
        );
        assert!(result.is_err());
    }

    #[test]
    fn tensor_matmul_returns_err_for_nonexistent_socket() {
        let result = tensor_matmul(
            Path::new(FAKE_SOCKET),
            &[1.0, 0.0, 0.0, 1.0],
            &[1.0, 2.0, 3.0, 4.0],
            2, 2, 2,
            TIMEOUT,
        );
        assert!(result.is_err());
    }

    #[test]
    fn tensor_create_returns_err_for_nonexistent_socket() {
        let result = tensor_create(Path::new(FAKE_SOCKET), &[2, 3], "zeros", TIMEOUT);
        assert!(result.is_err());
    }

    #[test]
    fn precision_route_returns_err_for_nonexistent_socket() {
        let result = precision_route(Path::new(FAKE_SOCKET), "lattice_qcd", None, TIMEOUT);
        assert!(result.is_err());
    }

    #[test]
    fn precision_route_result_from_value_defaults() {
        let v = serde_json::json!({});
        let r = PrecisionRouteResult::from_value(&v);
        assert_eq!(r.recommended_tier, "f64");
        assert!(!r.fma_safe);
        assert!(!r.requires_compiler);
        assert_eq!(r.hardware_hint, "");
        assert!(r.rationale.is_none());
    }

    #[test]
    fn ml_mlp_infer_returns_err_for_nonexistent_socket() {
        let result = ml_mlp_infer(
            Path::new(FAKE_SOCKET),
            &[1.0, 2.0, 3.0],
            3,
            &[4],
            2,
            TIMEOUT,
        );
        assert!(result.is_err());
    }

    #[test]
    fn precision_route_result_from_value_full() {
        let v = serde_json::json!({
            "recommended_tier": "DF64",
            "fma_safe": true,
            "requires_compiler": true,
            "hardware_hint": "compute",
            "rationale": "lattice QCD needs double-float precision"
        });
        let r = PrecisionRouteResult::from_value(&v);
        assert_eq!(r.recommended_tier, "DF64");
        assert!(r.fma_safe);
        assert!(r.requires_compiler);
        assert_eq!(r.hardware_hint, "compute");
        assert_eq!(r.rationale.as_deref(), Some("lattice QCD needs double-float precision"));
    }
}
