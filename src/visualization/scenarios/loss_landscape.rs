// SPDX-License-Identifier: AGPL-3.0-or-later

//! Loss landscape analysis scenario builder (baseCamp nS-03).
//!
//! Visualizes Hessian eigenvalue spectra, 2D loss surface field maps,
//! and spectral diagnostics (condition number, spectral gap) from
//! landscape analysis at converged minima.

#![expect(
    clippy::cast_precision_loss,
    reason = "grid index conversions for field map coordinates"
)]

use crate::loss_landscape::{landscape_analysis, spectral_gap};
use crate::surrogate::rosenbrock_2d;
use crate::tolerances;
use crate::visualization::types::{NeuralScenario, ScenarioEdge, ThresholdRange};

use super::{edge, fieldmap, gauge, node, scaffold, spectrum};

/// Build the loss landscape analysis scenario.
///
/// Nodes:
/// - `hessian_analysis`: Hessian eigenvalue spectrum + condition / gap gauges
/// - `loss_surface`: 2D loss field map around a minimum
#[expect(
    clippy::too_many_lines,
    reason = "scenario builder — single cohesive builder"
)]
#[must_use]
pub fn loss_landscape_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Loss Landscape Analysis (nS-03)",
        "Hessian eigenanalysis and 2D loss surface topology at converged minima",
    );

    let params = vec![1.0, 1.0];
    let loss_fn = |x: &[f64]| rosenbrock_2d(x[0], x[1]);
    let result = landscape_analysis(
        &loss_fn,
        &params,
        tolerances::HESSIAN_FD_STEP,
        tolerances::ODE_RTOL,
    );

    let mut sorted_evals = result.hessian_eigenvalues.clone();
    sorted_evals.sort_by(f64::total_cmp);
    let eval_indices: Vec<f64> = (0..sorted_evals.len()).map(|i| i as f64).collect();
    let gap = spectral_gap(&sorted_evals);
    let cond = if let (Some(&min), Some(&max)) = (sorted_evals.first(), sorted_evals.last()) {
        if min.abs() > crate::tolerances::ZERO_DETECTION {
            max.abs() / min.abs()
        } else {
            f64::INFINITY
        }
    } else {
        1.0
    };

    s.ecosystem.primals.push(node(
        "hessian_analysis",
        "Hessian Eigenanalysis (Rosenbrock at minimum)",
        "compute",
        0.0,
        0.0,
        &["science.hessian_eigen", "science.loss_landscape"],
        vec![
            spectrum(
                "hessian-eigenvalues",
                "Hessian Eigenvalue Spectrum",
                "dimensionless",
                eval_indices,
                sorted_evals,
            ),
            gauge(
                "spectral-gap",
                "Spectral Gap",
                gap,
                0.0,
                500.0,
                "dimensionless",
                [0.0, 100.0],
                [100.0, 400.0],
            ),
            gauge(
                "condition-number",
                "Condition Number",
                cond.min(1e6),
                1.0,
                1e6,
                "ratio",
                [1.0, 100.0],
                [100.0, 1e4],
            ),
            gauge(
                "saddle-index",
                "Saddle Index",
                result.saddle_index as f64,
                0.0,
                2.0,
                "count",
                [0.0, 0.5],
                [0.5, 2.0],
            ),
        ],
        vec![ThresholdRange {
            label: "Well-conditioned (<100)".into(),
            min: 1.0,
            max: 100.0,
            status: "normal".into(),
        }],
    ));

    let n_grid = 25;
    let grid_x: Vec<f64> = (0..n_grid)
        .map(|i| -0.5 + 3.0 * i as f64 / (n_grid - 1) as f64)
        .collect();
    let grid_y: Vec<f64> = grid_x.clone();
    let mut surface = Vec::with_capacity(n_grid * n_grid);
    for &gy in &grid_y {
        for &gx in &grid_x {
            let v = rosenbrock_2d(gx, gy).ln_1p().min(10.0);
            surface.push(v);
        }
    }

    s.ecosystem.primals.push(node(
        "loss_surface",
        "2D Loss Surface (Rosenbrock)",
        "compute",
        400.0,
        0.0,
        &["science.loss_landscape"],
        vec![fieldmap(
            "rosenbrock-surface",
            "log(1 + Rosenbrock(x,y))",
            grid_x,
            grid_y,
            surface,
            "log-loss",
        )],
        vec![],
    ));

    let edges = vec![edge(
        "hessian_analysis",
        "loss_surface",
        "eigenanalysis ↔ surface topology",
    )];
    (s, edges)
}
