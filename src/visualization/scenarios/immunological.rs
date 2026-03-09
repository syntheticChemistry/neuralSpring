// SPDX-License-Identifier: AGPL-3.0-or-later

//! Immunological Anderson scenario builder (baseCamp nS-06).
//!
//! Visualizes dose-response curves, pharmacokinetic decay,
//! and cytokine barrier height spectra from the immunological
//! Anderson localization model.

#![expect(
    clippy::cast_precision_loss,
    reason = "index-to-f64 conversions for visualization axes"
)]

use crate::immunological_anderson::{
    cytokine_barrier_heights, hill_dose_response, ic50_sweep, pk_exponential_decay,
};
use crate::visualization::types::{NeuralScenario, ScenarioEdge, ThresholdRange};

use super::{distribution, edge, gauge, node, scaffold, spectrum, timeseries};

/// Build the immunological Anderson scenario.
///
/// Nodes:
/// - `immuno_anderson`: dose-response, barrier spectrum, PK decay, disorder gauge
#[expect(
    clippy::too_many_lines,
    reason = "scenario builder — single cohesive builder"
)]
#[must_use]
pub fn immunological_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Immunological Anderson Localization",
        "Dose-response pharmacodynamics, cytokine barrier heights, and PK decay in AD skin models",
    );

    let ic50 = 10.0;
    let hill_n = 1.5;
    let concentrations: Vec<f64> = (0..50_i32).map(|i| 0.1 * 1.15_f64.powi(i)).collect();
    let responses = ic50_sweep(ic50, hill_n, &concentrations);

    let barriers = cytokine_barrier_heights(1.0);
    let barrier_names: Vec<f64> = (0..barriers.len()).map(|i| i as f64).collect();
    let barrier_values: Vec<f64> = barriers.iter().map(|&(_, h)| h).collect();

    let time_hours: Vec<f64> = (0..240_i32).map(f64::from).collect();
    let c0 = 100.0;
    let half_life = 14.0 * 24.0;
    let pk_curve: Vec<f64> = time_hours
        .iter()
        .map(|&t| pk_exponential_decay(c0, t, half_life))
        .collect();

    let disorder_w = 2.5;
    let response_at_ic50 = hill_dose_response(ic50, ic50, hill_n, 1.0);

    let dose_dist: Vec<f64> = concentrations
        .iter()
        .map(|&c| hill_dose_response(c, ic50, hill_n, 1.0))
        .collect();
    let dist_mean = dose_dist.iter().sum::<f64>() / dose_dist.len() as f64;
    let dist_std = (dose_dist
        .iter()
        .map(|r| (r - dist_mean).powi(2))
        .sum::<f64>()
        / dose_dist.len() as f64)
        .sqrt();

    s.ecosystem.primals.push(node(
        "immuno_anderson",
        "Immunological Anderson (nS-06)",
        "compute",
        0.0,
        0.0,
        &[
            "science.immunological_anderson",
            "science.dose_response",
            "science.pk_decay",
        ],
        vec![
            timeseries(
                "dose-response",
                "Hill Dose-Response Curve",
                "Concentration (nM)",
                "Response (fraction)",
                "fraction",
                concentrations,
                responses,
            ),
            spectrum(
                "barrier-heights",
                "Cytokine Barrier Heights",
                "eV-equivalent",
                barrier_names,
                barrier_values,
            ),
            timeseries(
                "pk-decay",
                "Lokivetmab PK Decay (10 days)",
                "Hours",
                "Concentration (mg/L)",
                "mg/L",
                time_hours,
                pk_curve,
            ),
            gauge(
                "disorder-parameter",
                "Effective Disorder W",
                disorder_w,
                0.0,
                10.0,
                "dimensionless",
                [0.0, 3.0],
                [3.0, 7.0],
            ),
            distribution(
                "response-distribution",
                "Dose-Response Distribution",
                "fraction",
                dose_dist,
                dist_mean,
                dist_std,
                response_at_ic50,
            ),
        ],
        vec![
            ThresholdRange {
                label: "Low disorder (healthy skin)".into(),
                min: 0.0,
                max: 3.0,
                status: "normal".into(),
            },
            ThresholdRange {
                label: "High disorder (AD flare)".into(),
                min: 7.0,
                max: 10.0,
                status: "warning".into(),
            },
        ],
    ));

    let edges = vec![edge(
        "immuno_anderson",
        "immuno_anderson",
        "dose → response → barrier",
    )];
    (s, edges)
}
