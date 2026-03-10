// SPDX-License-Identifier: AGPL-3.0-or-later

//! Attention Anderson spectral scenario builder (Exp 101).
//!
//! Visualizes how self-attention matrix quality correlates with Anderson
//! localization-like spectral properties.

use crate::attention_anderson::attention_spectral;
use crate::rng::Rng;
use crate::visualization::types::{NeuralScenario, ScenarioEdge, ThresholdRange};

use super::{edge, gauge, node, scaffold, spectrum, timeseries};

/// Build the attention Anderson spectral scenario.
///
/// Nodes:
/// - `attention_quality`: quality sweep with entropy + IPR correlations
/// - `spectral_localization`: eigenvalue spectrum + participation gauge
#[must_use]
#[expect(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "rich scenario builder; matrix dimensions ≤ 16 fit in f64"
)]
pub fn attention_anderson_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Attention Anderson Spectral Analysis",
        "Attention quality → Anderson localization: higher quality → delocalized eigenstates",
    );

    let n = 16;
    let n_configs = 8;
    let mut rng = Rng::new(42);

    let mut qualities = Vec::with_capacity(n_configs);
    let mut entropies = Vec::with_capacity(n_configs);
    let mut ipr_vals = Vec::with_capacity(n_configs);
    let mut xi_vals = Vec::with_capacity(n_configs);
    

    for i in 0..n_configs {
        #[expect(clippy::cast_precision_loss, reason = "i, n_configs ≤ 8")]
        let quality = (i as f64 + 1.0) / n_configs as f64;

        let mut matrix = vec![0.0; n * n];
        for row in 0..n {
            let mut row_vals = Vec::with_capacity(n);
            for col in 0..n {
                let base = if row == col { quality } else { 1.0 - quality };
                row_vals.push(rng.uniform().mul_add(0.1, base));
            }
            let sum: f64 = row_vals.iter().sum();
            for (col, val) in row_vals.into_iter().enumerate() {
                matrix[row * n + col] = val / sum;
            }
        }
        let sym: Vec<f64> = (0..n * n)
            .map(|idx| {
                let r = idx / n;
                let c = idx % n;
                (matrix[r * n + c] + matrix[c * n + r]) * 0.5
            })
            .collect();

        let result = attention_spectral(&sym, n);
        qualities.push(quality);
        entropies.push(result.entropy);
        ipr_vals.push(result.mean_ipr);
        xi_vals.push(result.xi);
        
    }

    s.ecosystem.primals.push(node(
        "attention_quality",
        "Attention Quality Sweep",
        "compute",
        0.0,
        0.0,
        &["science.attention_mechanism", "science.information_theory"],
        vec![
            timeseries(
                "quality-vs-entropy",
                "Quality → Attention Entropy",
                "Quality",
                "Entropy (nats)",
                "nats",
                qualities.clone(),
                entropies,
            ),
            timeseries(
                "quality-vs-ipr",
                "Quality → Mean IPR",
                "Quality",
                "Mean IPR",
                "dimensionless",
                qualities.clone(),
                ipr_vals,
            ),
            timeseries(
                "quality-vs-xi",
                "Quality → Localization Length",
                "Quality",
                "ξ",
                "dimensionless",
                qualities,
                xi_vals,
            ),
        ],
        vec![ThresholdRange {
            label: "Focused attention (low IPR)".into(),
            min: 0.0,
            max: 0.3,
            status: "normal".into(),
        }],
    ));

    let ref_quality = 0.8;
    let mut ref_matrix = vec![0.0; n * n];
    for row in 0..n {
        let mut row_vals = Vec::with_capacity(n);
        for col in 0..n {
            let base = if row == col {
                ref_quality
            } else {
                1.0 - ref_quality
            };
            row_vals.push(base + 0.05_f64);
        }
        let sum: f64 = row_vals.iter().sum();
        for (col, val) in row_vals.into_iter().enumerate() {
            ref_matrix[row * n + col] = val / sum;
        }
    }
    let sym_ref: Vec<f64> = (0..n * n)
        .map(|idx| {
            let r = idx / n;
            let c = idx % n;
            (ref_matrix[r * n + c] + ref_matrix[c * n + r]) * 0.5
        })
        .collect();

    let ref_result = attention_spectral(&sym_ref, n);

    let ref_decomp = crate::eigh::eigh_householder_qr(&sym_ref, n);
    let mut ref_evals = ref_decomp.eigenvalues;
    ref_evals.sort_by(f64::total_cmp);
    #[expect(clippy::cast_possible_truncation, reason = "n ≤ 16")]
    let n_u32 = n as u32;
    let indices: Vec<f64> = (0..n_u32).map(f64::from).collect();

    s.ecosystem.primals.push(node(
        "spectral_localization",
        &format!(
            "Reference Spectrum (q={ref_quality}, IPR={:.3})",
            ref_result.mean_ipr
        ),
        "compute",
        400.0,
        0.0,
        &[
            "science.spectral_analysis",
            "science.anderson_localization",
        ],
        vec![
            spectrum(
                "ref-eigenvalues",
                "Reference Eigenvalue Spectrum",
                "dimensionless",
                indices,
                ref_evals,
            ),
            gauge(
                "participation",
                "Participation Number",
                ref_result.participation,
                1.0,
                n as f64,
                "dimensionless",
                [n as f64 * 0.5, n as f64],
                [1.0, n as f64 * 0.5],
            ),
            gauge(
                "spectral-radius",
                "Spectral Radius",
                ref_result.spectral_radius,
                0.0,
                2.0,
                "dimensionless",
                [0.0, 1.0],
                [1.0, 2.0],
            ),
        ],
        vec![],
    ));

    let edges = vec![edge(
        "attention_quality",
        "spectral_localization",
        "quality sweep → spectral analysis",
    )];

    (s, edges)
}
