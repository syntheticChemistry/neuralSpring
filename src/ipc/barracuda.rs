// SPDX-License-Identifier: AGPL-3.0-or-later

//! barraCuda IPC surface — tensor lifecycle, core math, and ML ops.
//!
//! Methods: `stats.mean`, `stats.std_dev`, `stats.weighted_mean`,
//! `tensor.matmul`, `tensor.create`.

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
    Ok(call_capability(
        socket,
        capabilities::TENSOR_CREATE,
        &serde_json::json!({ "shape": shape, "fill": fill }),
        timeout,
    )?)
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
}
