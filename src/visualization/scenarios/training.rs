// SPDX-License-Identifier: AGPL-3.0-or-later

//! Training trajectory scenario builder.
//!
//! Simulates a training run by interpolating between two random symmetric
//! weight matrices and tracking spectral diagnostics at each epoch.

#![expect(
    clippy::cast_precision_loss,
    reason = "epoch/alpha conversion is inherently lossy"
)]

use crate::anderson_localization::mean_ipr;
use crate::eigh::eigh_householder_qr;
use crate::primitives::shannon_entropy;
use crate::rng::Rng;
use crate::visualization::types::{NeuralScenario, ScenarioEdge};
use crate::weight_spectral::{level_spacing_ratio, spectral_bandwidth};

use super::{edge, node, scaffold, timeseries};

/// Build the training trajectory study scenario.
///
/// Produces 1 node with 4 `TimeSeries` channels tracking epoch vs:
/// mean IPR, spectral entropy, level spacing ratio, and bandwidth.
#[must_use]
pub fn training_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Training Trajectory Spectral Analysis",
        "Spectral diagnostics evolving across a simulated training run (nS-050)",
    );

    let dim = 16;
    let n_epochs = 20;
    let mut rng = Rng::new(42);

    let mut w_start = vec![0.0f64; dim * dim];
    let mut w_end = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            w_start[i * dim + j] = rng.uniform() - 0.5;
            w_end[i * dim + j] = rng.uniform() - 0.5;
        }
    }
    for i in 0..dim {
        for j in (i + 1)..dim {
            w_start[j * dim + i] = w_start[i * dim + j];
            w_end[j * dim + i] = w_end[i * dim + j];
        }
    }

    let mut epochs = Vec::with_capacity(n_epochs + 1);
    let mut ipr_vals = Vec::with_capacity(n_epochs + 1);
    let mut entropy_vals = Vec::with_capacity(n_epochs + 1);
    let mut lsr_vals = Vec::with_capacity(n_epochs + 1);
    let mut bw_vals = Vec::with_capacity(n_epochs + 1);

    for epoch in 0..=n_epochs {
        let alpha = epoch as f64 / n_epochs as f64;
        let w: Vec<f64> = w_start
            .iter()
            .zip(&w_end)
            .map(|(&s, &e)| alpha.mul_add(e - s, s))
            .collect();

        let decomp = eigh_householder_qr(&w, dim);
        let ipr_val = mean_ipr(&decomp.eigenvectors, dim);
        let mut evals = decomp.eigenvalues;
        evals.sort_by(f64::total_cmp);
        let entropy = shannon_entropy(&evals);
        let lsr = level_spacing_ratio(&evals);
        let bw = spectral_bandwidth(&evals);

        epochs.push(epoch as f64);
        ipr_vals.push(ipr_val);
        entropy_vals.push(entropy);
        lsr_vals.push(lsr);
        bw_vals.push(bw);
    }

    s.ecosystem.primals.push(node(
        "training_trajectory",
        "Training Trajectory (dim=16, 20 epochs)",
        "compute",
        0.0,
        0.0,
        &["science.training_trajectory"],
        vec![
            timeseries(
                "epoch-vs-ipr",
                "Mean IPR over Training",
                "Epoch",
                "Mean IPR",
                "dimensionless",
                epochs.clone(),
                ipr_vals,
            ),
            timeseries(
                "epoch-vs-entropy",
                "Spectral Entropy over Training",
                "Epoch",
                "Shannon Entropy",
                "nats",
                epochs.clone(),
                entropy_vals,
            ),
            timeseries(
                "epoch-vs-lsr",
                "Level Spacing Ratio over Training",
                "Epoch",
                "LSR",
                "dimensionless",
                epochs.clone(),
                lsr_vals,
            ),
            timeseries(
                "epoch-vs-bandwidth",
                "Spectral Bandwidth over Training",
                "Epoch",
                "Bandwidth",
                "dimensionless",
                epochs,
                bw_vals,
            ),
        ],
        vec![],
    ));

    let edges = vec![edge(
        "training_trajectory",
        "training_trajectory",
        "epoch evolution",
    )];

    (s, edges)
}
