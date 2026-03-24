// SPDX-License-Identifier: AGPL-3.0-or-later

//! Autocorrelation, decay-time estimation, and thin wrappers around shared regression metrics.

#![expect(
    clippy::cast_precision_loss,
    reason = "autocorrelation normalizes by series length as f64"
)]

use crate::tolerances;

use super::cgm::ACOR_DECAY_STEPS;

/// Compute normalized autocorrelation up to `max_lag` steps.
#[must_use]
pub fn autocorrelation(series: &[f64], max_lag: usize) -> Vec<f64> {
    let n = series.len();
    let mean = series.iter().sum::<f64>() / n as f64;
    let var = series.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / n as f64;
    let mut acor = Vec::with_capacity(max_lag);
    for lag in 0..max_lag {
        let cov = series[..n - lag]
            .iter()
            .zip(series[lag..].iter())
            .map(|(&a, &b)| (a - mean) * (b - mean))
            .sum::<f64>()
            / n as f64;
        acor.push(cov / var.max(tolerances::LOG_ZERO_GUARD));
    }
    acor
}

/// Estimate autocorrelation decay time τ (in steps).
#[must_use]
pub fn estimate_tau(acor: &[f64]) -> usize {
    let threshold = 1.0 / std::f64::consts::E;
    acor.iter()
        .position(|&a| a < threshold)
        .unwrap_or(ACOR_DECAY_STEPS)
}

/// R² score between actual and predicted values.
///
/// Delegates to [`crate::metrics::r_squared`] → `barracuda::stats::r_squared`.
#[must_use]
pub fn r2_score(actual: &[f64], predicted: &[f64]) -> f64 {
    crate::metrics::r_squared(actual, predicted)
}

/// RMSE between actual and predicted values.
///
/// Delegates to [`crate::metrics::rmse`] → `barracuda::stats::rmse`.
#[must_use]
pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    crate::metrics::rmse(actual, predicted)
}
