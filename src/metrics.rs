// SPDX-License-Identifier: AGPL-3.0-only

//! Statistical metrics for validation (R², RMSE, MAE, NSE).
//!
//! ## `BarraCUDA` Integration
//!
//! `barracuda::stats` provides building blocks (`variance`, `pearson_correlation`,
//! `covariance`) validated in `validate_barracuda_stats`. These metrics compose
//! those primitives into domain-specific measures. When barracuda adds GPU-resident
//! `mse_loss` / `mae_loss` Tensor ops, we can delegate the hot path there.
//!
//! Re-exports from barracuda for convenience:
//!
//! - [`barracuda::stats::correlation::variance`]
//! - [`barracuda::stats::pearson_correlation`]

/// Coefficient of determination.
///
/// ```
/// # use neural_spring::metrics::r_squared;
/// let y = [1.0, 2.0, 3.0, 4.0, 5.0];
/// assert!((r_squared(&y, &y) - 1.0).abs() < 1e-12);
/// ```
///
/// # Panics
///
/// Panics if `y_true` and `y_pred` have different lengths.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn r_squared(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
    let n = y_true.len() as f64;
    let mean = y_true.iter().sum::<f64>() / n;
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();
    if ss_tot == 0.0 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

/// Root mean squared error.
///
/// # Panics
///
/// Panics if `y_true` and `y_pred` have different lengths.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
    let n = y_true.len() as f64;
    let mse: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>()
        / n;
    mse.sqrt()
}

/// Mean absolute error.
///
/// # Panics
///
/// Panics if `y_true` and `y_pred` have different lengths.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
    let n = y_true.len() as f64;
    y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t - p).abs())
        .sum::<f64>()
        / n
}

/// Nash-Sutcliffe Efficiency (standard hydrology metric, equivalent to R²).
#[must_use]
pub fn nse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    r_squared(y_true, y_pred)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn perfect_prediction() {
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(r_squared(&y, &y), 1.0);
        assert_relative_eq!(rmse(&y, &y), 0.0);
        assert_relative_eq!(mae(&y, &y), 0.0);
    }

    #[test]
    fn mean_prediction_gives_zero_r2() {
        let y_true = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let y_pred = [mean; 5];
        assert_relative_eq!(r_squared(&y_true, &y_pred), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn known_rmse() {
        let y_true = [1.0, 2.0, 3.0];
        let y_pred = [1.1, 2.1, 3.1];
        assert_relative_eq!(rmse(&y_true, &y_pred), 0.1, epsilon = 1e-10);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn metrics_deterministic() {
        let y_true = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = [1.1, 2.2, 2.9, 4.1, 4.8];
        let r2_a = r_squared(&y_true, &y_pred);
        let r2_b = r_squared(&y_true, &y_pred);
        assert_eq!(r2_a, r2_b, "r_squared must be bit-identical across runs");

        let rmse_a = rmse(&y_true, &y_pred);
        let rmse_b = rmse(&y_true, &y_pred);
        assert_eq!(rmse_a, rmse_b, "rmse must be bit-identical across runs");

        let mae_a = mae(&y_true, &y_pred);
        let mae_b = mae(&y_true, &y_pred);
        assert_eq!(mae_a, mae_b, "mae must be bit-identical across runs");
    }
}
