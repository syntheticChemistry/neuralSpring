// SPDX-License-Identifier: AGPL-3.0-or-later

//! Statistical metrics for validation (R², RMSE, MAE, NSE).
//!
//! ## `BarraCUDA` Integration
//!
//! All four metrics delegate to `barracuda::stats` (absorbed from
//! airSpring/groundSpring via `ToadStool` S64–S66, now in `BarraCUDA`). `mae` was rewired in
//! S66 when `barracuda::stats::mae` became available.

/// Coefficient of determination (delegates to `barracuda::stats::r_squared`).
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
pub fn r_squared(y_true: &[f64], y_pred: &[f64]) -> f64 {
    #[cfg(feature = "barracuda")]
    { barracuda::stats::r_squared(y_true, y_pred) }
    #[cfg(not(feature = "barracuda"))]
    {
        let mean: f64 = y_true.iter().sum::<f64>() / y_true.len() as f64;
        let ss_res: f64 = y_true.iter().zip(y_pred).map(|(t, p)| (t - p).powi(2)).sum();
        let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();
        1.0 - ss_res / ss_tot
    }
}

/// Root mean squared error.
///
/// # Panics
///
/// Panics if `y_true` and `y_pred` have different lengths.
#[must_use]
pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    #[cfg(feature = "barracuda")]
    { barracuda::stats::rmse(y_true, y_pred) }
    #[cfg(not(feature = "barracuda"))]
    {
        let mse: f64 = y_true.iter().zip(y_pred)
            .map(|(t, p)| (t - p).powi(2)).sum::<f64>() / y_true.len() as f64;
        mse.sqrt()
    }
}

/// Mean absolute error.
///
/// # Panics
///
/// Panics if `y_true` and `y_pred` have different lengths.
#[must_use]
pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    #[cfg(feature = "barracuda")]
    { barracuda::stats::mae(y_true, y_pred) }
    #[cfg(not(feature = "barracuda"))]
    {
        y_true.iter().zip(y_pred)
            .map(|(t, p)| (t - p).abs()).sum::<f64>() / y_true.len() as f64
    }
}

/// Nash-Sutcliffe Efficiency.
#[must_use]
pub fn nse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    #[cfg(feature = "barracuda")]
    { barracuda::stats::nash_sutcliffe(y_true, y_pred) }
    #[cfg(not(feature = "barracuda"))]
    { r_squared(y_true, y_pred) }
}

#[cfg(all(test, feature = "barracuda"))]
mod tests {
    use super::*;
    use crate::tolerances;
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
        assert_relative_eq!(
            r_squared(&y_true, &y_pred),
            0.0,
            epsilon = tolerances::CROSS_LANGUAGE
        );
    }

    #[test]
    fn known_rmse() {
        let y_true = [1.0, 2.0, 3.0];
        let y_pred = [1.1, 2.1, 3.1];
        assert_relative_eq!(
            rmse(&y_true, &y_pred),
            0.1,
            epsilon = tolerances::CROSS_LANGUAGE
        );
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "determinism test requires exact bit equality"
    )]
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

    #[test]
    fn nse_equals_r_squared() {
        let y_true = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = [1.1, 2.2, 2.9, 4.1, 4.8];
        assert_relative_eq!(nse(&y_true, &y_pred), r_squared(&y_true, &y_pred));
    }

    #[test]
    fn r_squared_constant_true() {
        let y_true = [3.0, 3.0, 3.0];
        let y_pred = [3.0, 3.1, 2.9];
        let r2 = r_squared(&y_true, &y_pred);
        assert!(r2.is_finite(), "constant y_true should not produce NaN");
    }
}
