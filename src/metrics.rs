//! Statistical metrics for validation (R², RMSE, MAE, NSE).
//!
//! These should eventually delegate to `barracuda::reduce` primitives
//! rather than reimplementing reduction ops.

/// Coefficient of determination.
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
}
