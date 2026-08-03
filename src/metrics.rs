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
    {
        barracuda::stats::r_squared(y_true, y_pred)
    }
    #[cfg(not(feature = "barracuda"))]
    {
        let mean: f64 = y_true.iter().sum::<f64>() / y_true.len() as f64;
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred)
            .map(|(t, p)| (t - p).powi(2))
            .sum();
        let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();
        if ss_tot < f64::EPSILON {
            if ss_res < f64::EPSILON { 1.0 } else { 0.0 }
        } else {
            1.0 - ss_res / ss_tot
        }
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
    {
        barracuda::stats::rmse(y_true, y_pred)
    }
    #[cfg(not(feature = "barracuda"))]
    {
        let mse: f64 = y_true
            .iter()
            .zip(y_pred)
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>()
            / y_true.len() as f64;
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
    {
        barracuda::stats::mae(y_true, y_pred)
    }
    #[cfg(not(feature = "barracuda"))]
    {
        y_true
            .iter()
            .zip(y_pred)
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>()
            / y_true.len() as f64
    }
}

/// Nash-Sutcliffe Efficiency.
#[must_use]
pub fn nse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    #[cfg(feature = "barracuda")]
    {
        barracuda::stats::nash_sutcliffe(y_true, y_pred)
    }
    #[cfg(not(feature = "barracuda"))]
    {
        r_squared(y_true, y_pred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_prediction() {
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((r_squared(&y, &y) - 1.0).abs() < 1e-12);
        assert!(rmse(&y, &y).abs() < 1e-12);
        assert!(mae(&y, &y).abs() < 1e-12);
    }

    #[test]
    fn mean_prediction_gives_zero_r2() {
        let y_true = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = [3.0; 5];
        assert!(r_squared(&y_true, &y_pred).abs() < 1e-10);
    }

    #[test]
    fn known_rmse() {
        let y_true = [1.0, 2.0, 3.0];
        let y_pred = [1.1, 2.1, 3.1];
        assert!((rmse(&y_true, &y_pred) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn known_mae() {
        let y_true = [1.0, 2.0, 3.0];
        let y_pred = [1.5, 2.5, 3.5];
        assert!((mae(&y_true, &y_pred) - 0.5).abs() < 1e-12);
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "determinism test requires exact bit equality"
    )]
    fn metrics_deterministic() {
        let y_true = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = [1.1, 2.2, 2.9, 4.1, 4.8];
        assert_eq!(r_squared(&y_true, &y_pred), r_squared(&y_true, &y_pred));
        assert_eq!(rmse(&y_true, &y_pred), rmse(&y_true, &y_pred));
        assert_eq!(mae(&y_true, &y_pred), mae(&y_true, &y_pred));
    }

    #[test]
    fn nse_equals_r_squared() {
        let y_true = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = [1.1, 2.2, 2.9, 4.1, 4.8];
        assert!((nse(&y_true, &y_pred) - r_squared(&y_true, &y_pred)).abs() < 1e-12);
    }

    #[test]
    fn r_squared_constant_true() {
        let y_true = [3.0, 3.0, 3.0];
        let y_pred = [3.0, 3.1, 2.9];
        let r2 = r_squared(&y_true, &y_pred);
        assert!(r2.is_finite(), "constant y_true should not produce NaN");
    }

    #[test]
    fn r_squared_good_prediction() {
        let y_true = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = [1.1, 2.0, 2.9, 4.1, 5.0];
        let r2 = r_squared(&y_true, &y_pred);
        assert!(r2 > 0.99, "good prediction should have R² > 0.99, got {r2}");
    }

    #[test]
    fn rmse_nonnegative() {
        let y_true = [1.0, 5.0, 3.0];
        let y_pred = [2.0, 4.0, 6.0];
        assert!(rmse(&y_true, &y_pred) >= 0.0);
    }

    #[test]
    fn mae_symmetric() {
        let a = [1.0, 2.0, 3.0];
        let b = [3.0, 2.0, 1.0];
        assert!((mae(&a, &b) - mae(&b, &a)).abs() < 1e-12);
    }
}
