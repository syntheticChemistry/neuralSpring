// SPDX-License-Identifier: AGPL-3.0-or-later

//! barraCuda IPC surface — tensor lifecycle, core math, and ML ops.
//!
//! Methods: `stats.mean`, `stats.std_dev`, `stats.weighted_mean`,
//! `tensor.matmul`, `tensor.create`.

use std::path::Path;
use std::time::Duration;

use crate::validation::composition::call_capability;

/// `stats.mean` via barraCuda IPC.
///
/// # Errors
///
/// Returns an error if barraCuda is not reachable or the response is malformed.
pub fn stats_mean(socket: &Path, data: &[f64], timeout: Duration) -> Result<f64, String> {
    let result = call_capability(
        socket,
        "stats.mean",
        &serde_json::json!({ "data": data }),
        timeout,
    )?;
    super::extract_f64(&result, &["mean", "result", "value"])
        .ok_or_else(|| "stats.mean: response missing numeric result".to_string())
}

/// `stats.std_dev` via barraCuda IPC.
///
/// # Errors
///
/// Returns an error if barraCuda is not reachable or the response is malformed.
pub fn stats_std_dev(socket: &Path, data: &[f64], timeout: Duration) -> Result<f64, String> {
    let result = call_capability(
        socket,
        "stats.std_dev",
        &serde_json::json!({ "data": data }),
        timeout,
    )?;
    super::extract_f64(&result, &["std_dev", "result", "value"])
        .ok_or_else(|| "stats.std_dev: response missing numeric result".to_string())
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
) -> Result<f64, String> {
    let result = call_capability(
        socket,
        "stats.weighted_mean",
        &serde_json::json!({ "data": data, "weights": weights }),
        timeout,
    )?;
    super::extract_f64(&result, &["weighted_mean", "result", "value"])
        .ok_or_else(|| "stats.weighted_mean: response missing numeric result".to_string())
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
) -> Result<Vec<f64>, String> {
    let result = call_capability(
        socket,
        "tensor.matmul",
        &serde_json::json!({
            "a": a, "b": b,
            "rows_a": rows_a, "cols_a": cols_a, "cols_b": cols_b,
        }),
        timeout,
    )?;
    super::extract_f64_array(&result, &["data", "result"])
        .ok_or_else(|| "tensor.matmul: response missing data array".to_string())
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
) -> Result<serde_json::Value, String> {
    call_capability(
        socket,
        "tensor.create",
        &serde_json::json!({ "shape": shape, "fill": fill }),
        timeout,
    )
}
