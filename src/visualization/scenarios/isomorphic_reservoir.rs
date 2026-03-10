// SPDX-License-Identifier: AGPL-3.0-or-later

//! Isomorphic reservoir ensemble scenario builder (Exp 098).
//!
//! Visualizes spectral universality: three recurrent architectures from
//! unrelated domains produce nearly identical eigenvalue distributions.

use crate::isomorphic_reservoir::{cross_domain_metrics, spectral_properties};
use crate::rng::Rng;
use crate::visualization::types::{NeuralScenario, ScenarioEdge, ThresholdRange};

use super::{bar, edge, gauge, node, scaffold, spectrum};

/// Build the isomorphic reservoir ensemble scenario.
///
/// Nodes:
/// - `reservoir_spectral`: overlaid eigenvalue spectra from 3 domains
/// - `universality_metrics`: cross-domain CV gauges
#[must_use]
#[expect(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "rich scenario builder; matrix dimensions ≤ 32 fit in f64"
)]
pub fn isomorphic_reservoir_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Isomorphic Reservoir Ensemble",
        "Spectral universality: ESN (digester) + LSTM (glucose) + LSTM (weather) produce identical eigenvalue distributions",
    );

    let n = 32;
    let mut rng = Rng::new(42);
    let domains = ["Digester ESN", "Glucose LSTM", "Weather LSTM"];
    let gains = [0.9, 0.85, 0.95];

    let mut profiles = Vec::with_capacity(3);
    let mut all_evals = Vec::new();

    for (i, (domain, gain)) in domains.iter().zip(gains.iter()).enumerate() {
        let mut matrix = vec![0.0; n * n];
        let mut r = Rng::new(42 + i as u64);
        for val in &mut matrix {
            *val = r.uniform().mul_add(2.0, -1.0) * gain / (n as f64).sqrt();
        }
        for j in 0..n {
            matrix[j * n + j] += rng.uniform() * 0.01;
        }
        let sym: Vec<f64> = (0..n * n)
            .map(|idx| {
                let row = idx / n;
                let col = idx % n;
                (matrix[row * n + col] + matrix[col * n + row]) * 0.5
            })
            .collect();

        let profile = spectral_properties(&sym, n, domain);

        let decomp = crate::eigh::eigh_householder_qr(&sym, n);
        let mut evals = decomp.eigenvalues;
        evals.sort_by(f64::total_cmp);
        all_evals.push((domain.to_string(), evals));

        profiles.push(profile);
    }

    let mut channels = Vec::new();

    for (domain, evals) in &all_evals {
        let indices: Vec<f64> = (0..evals.len()).map(|i| i as f64).collect();
        channels.push(spectrum(
            &format!("evals-{}", domain.to_lowercase().replace(' ', "-")),
            &format!("{domain} Eigenvalues"),
            "dimensionless",
            indices,
            evals.clone(),
        ));
    }

    let domain_names: Vec<String> = profiles.iter().map(|p| p.name.clone()).collect();
    let spacing_ratios: Vec<f64> = profiles.iter().map(|p| p.mean_spacing_ratio).collect();
    channels.push(bar(
        "spacing-ratios",
        "Level Spacing Ratio by Domain",
        domain_names.clone(),
        spacing_ratios,
        "dimensionless",
    ));

    let eff_dims: Vec<f64> = profiles.iter().map(|p| p.effective_dimension).collect();
    channels.push(bar(
        "effective-dimensions",
        "Effective Dimension by Domain",
        domain_names,
        eff_dims,
        "dimensionless",
    ));

    s.ecosystem.primals.push(node(
        "reservoir_spectral",
        "Cross-Domain Eigenvalue Spectra",
        "compute",
        0.0,
        0.0,
        &[
            "science.spectral_analysis",
            "science.isomorphic_thesis",
            "science.reservoir_computing",
        ],
        channels,
        vec![],
    ));

    let cdm = cross_domain_metrics(&profiles);

    s.ecosystem.primals.push(node(
        "universality_metrics",
        "Spectral Universality Metrics",
        "compute",
        500.0,
        0.0,
        &["science.cross_domain_universality"],
        vec![
            gauge(
                "eff-ratio-cv",
                "Effective Dimension Ratio CV",
                cdm.eff_ratio_cv,
                0.0,
                0.5,
                "dimensionless",
                [0.0, 0.05],
                [0.05, 0.2],
            ),
            gauge(
                "ipr-cv",
                "IPR CV (cross-domain)",
                cdm.ipr_cv,
                0.0,
                0.5,
                "dimensionless",
                [0.0, 0.05],
                [0.05, 0.2],
            ),
            gauge(
                "spacing-ratio-mean",
                "Mean Spacing Ratio (Wigner ≈ 0.53)",
                cdm.spacing_ratio_mean,
                0.0,
                1.0,
                "dimensionless",
                [0.4, 0.6],
                [0.2, 0.4],
            ),
        ],
        vec![ThresholdRange {
            label: "Universal (CV < 0.05)".into(),
            min: 0.0,
            max: 0.05,
            status: "normal".into(),
        }],
    ));

    let edges = vec![edge(
        "reservoir_spectral",
        "universality_metrics",
        "spectra → universality check",
    )];

    (s, edges)
}
