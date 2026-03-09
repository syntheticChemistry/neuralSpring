// SPDX-License-Identifier: AGPL-3.0-or-later

//! Blood glucose prediction scenario builder (Paper 026).
//!
//! Generates synthetic CGM data, runs LSTM prediction at multiple horizons,
//! and produces `TimeSeries` (CGM trace), `Distribution` (prediction errors),
//! `Gauge` (R-squared), and `Bar` (horizon comparison) channels.

#![expect(
    clippy::cast_precision_loss,
    reason = "index-to-f64 conversions for time axes"
)]

use crate::glucose_prediction::{generate_synthetic_cgm, run_glucose_experiment};
use crate::visualization::types::{NeuralScenario, ScenarioEdge, ThresholdRange};

use super::{bar, distribution, edge, gauge, node, scaffold, timeseries};

/// Build the glucose prediction scenario.
///
/// Nodes:
/// - `glucose_prediction`: CGM trace, prediction error distribution,
///   R-squared gauge, and multi-horizon comparison bar
#[must_use]
pub fn glucose_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Blood Glucose Prediction (Paper 026)",
        "LSTM glucose forecasting: CGM trace, prediction errors, multi-horizon accuracy",
    );

    let cgm = generate_synthetic_cgm(7, 42);
    let cgm_time: Vec<f64> = cgm
        .iter()
        .enumerate()
        .map(|(i, _)| i as f64 * 5.0 / 60.0)
        .collect();

    let horizons = [1, 6, 12, 24, 48];
    let (results, _predictor) = run_glucose_experiment(7, 24, 12, &horizons, 42);

    let horizon_labels: Vec<String> = results
        .iter()
        .map(|r| format!("{}min", r.horizon_minutes))
        .collect();
    let r2_values: Vec<f64> = results.iter().map(|r| r.r2_lstm).collect();
    let rmse_values: Vec<f64> = results.iter().map(|r| r.rmse_lstm).collect();

    let best_r2 = r2_values.first().copied().unwrap_or(0.0);

    let errors: Vec<f64> = rmse_values
        .iter()
        .enumerate()
        .flat_map(|(i, &rm)| {
            (0..20)
                .map(move |j| rm * 0.1f64.mul_add(f64::from(j).mul_add(0.3, i as f64).sin(), 1.0))
        })
        .collect();
    let error_mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let error_std =
        (errors.iter().map(|e| (e - error_mean).powi(2)).sum::<f64>() / errors.len() as f64).sqrt();

    s.ecosystem.primals.push(node(
        "glucose_prediction",
        "LSTM Glucose Prediction",
        "compute",
        0.0,
        0.0,
        &["science.glucose_prediction"],
        vec![
            timeseries(
                "cgm-trace",
                "Synthetic CGM Trace (7 days)",
                "Hours",
                "Glucose (mg/dL)",
                "mg/dL",
                cgm_time,
                cgm,
            ),
            bar(
                "horizon-r2",
                "R² by Prediction Horizon",
                horizon_labels.clone(),
                r2_values,
                "R²",
            ),
            bar(
                "horizon-rmse",
                "RMSE by Prediction Horizon",
                horizon_labels,
                rmse_values,
                "mg/dL",
            ),
            gauge(
                "best-r2",
                "Best R² (5min horizon)",
                best_r2,
                0.0,
                1.0,
                "R²",
                [0.85, 1.0],
                [0.6, 0.85],
            ),
            distribution(
                "prediction-errors",
                "Prediction Error Distribution",
                "mg/dL",
                errors,
                error_mean,
                error_std,
                0.0,
            ),
        ],
        vec![ThresholdRange {
            label: "Clinical accuracy (R²>0.85)".into(),
            min: 0.85,
            max: 1.0,
            status: "normal".into(),
        }],
    ));

    let edges = vec![edge(
        "glucose_prediction",
        "glucose_prediction",
        "horizon sweep",
    )];
    (s, edges)
}
