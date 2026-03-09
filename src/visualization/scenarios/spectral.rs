// SPDX-License-Identifier: AGPL-3.0-or-later

//! Anderson localization, spectral analysis, and Hessian eigenanalysis
//! scenario builder.
//!
//! Produces 3 nodes: disorder sweep, single-system spectral analysis,
//! and Hessian eigenanalysis at a trained minimum.

use crate::anderson_localization::{anderson_hamiltonian_random, disorder_sweep, mean_ipr};
use crate::eigh::eigh_householder_qr;
use crate::rng::Rng;
use crate::visualization::types::{NeuralScenario, ScenarioEdge, ThresholdRange};
use crate::weight_spectral::{
    classify_phase, empirical_spectral_density, level_spacing_ratio, spectral_bandwidth,
    spectral_condition_number,
};

use super::{edge, gauge, node, scaffold, spectrum, timeseries};

/// Build the spectral analysis study scenario.
///
/// Nodes:
/// - `anderson_sweep`: disorder vs mean IPR sweep
/// - `spectral_analysis`: single-system eigenvalue spectrum + phase gauge
/// - `hessian_eigen`: Hessian eigenvalue spectrum + condition gauge
#[must_use]
#[expect(clippy::too_many_lines, reason = "3 nodes with rich data channels")]
pub fn spectral_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Spectral Analysis & Anderson Localization",
        "Weight matrix spectral diagnostics: Anderson sweep, eigenvalue spectrum, Hessian analysis",
    );

    let disorder_values = vec![0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
    let mut rng = Rng::new(42);
    let iprs = disorder_sweep(20, 1.0, &disorder_values, &mut rng);

    s.ecosystem.primals.push(node(
        "anderson_sweep",
        "Anderson Localization Sweep",
        "compute",
        0.0,
        0.0,
        &["science.anderson_localization", "science.disorder_sweep"],
        vec![timeseries(
            "disorder-vs-ipr",
            "Disorder vs Mean IPR",
            "Disorder (W)",
            "Mean IPR",
            "dimensionless",
            disorder_values,
            iprs,
        )],
        vec![
            ThresholdRange {
                label: "Extended (delocalized)".into(),
                min: 0.0,
                max: 0.1,
                status: "normal".into(),
            },
            ThresholdRange {
                label: "Localized (memorization risk)".into(),
                min: 0.5,
                max: 1.0,
                status: "warning".into(),
            },
        ],
    ));

    let mut rng2 = Rng::new(42);
    let h = anderson_hamiltonian_random(20, 1.0, 2.0, &mut rng2);
    let decomp = eigh_householder_qr(&h, 20);
    let ipr_val = mean_ipr(&decomp.eigenvectors, 20);
    let mut evals = decomp.eigenvalues;
    evals.sort_by(f64::total_cmp);
    let lsr = level_spacing_ratio(&evals);
    let phase = classify_phase(lsr);
    let (centers, amps) = empirical_spectral_density(&evals, 20);

    s.ecosystem.primals.push(node(
        "spectral_analysis",
        &format!("Spectral Analysis (W=2.0, phase={phase})"),
        "compute",
        400.0,
        0.0,
        &["science.spectral_analysis"],
        vec![
            spectrum(
                "eigenvalue-esd",
                "Eigenvalue Spectral Density",
                "dimensionless",
                centers,
                amps,
            ),
            gauge(
                "mean-ipr",
                "Mean IPR",
                ipr_val,
                0.0,
                1.0,
                "dimensionless",
                [0.0, 0.1],
                [0.1, 0.5],
            ),
        ],
        vec![],
    ));

    let dim = 20;
    let mut hessian = vec![0.0; dim * dim];
    for i in 0..dim {
        hessian[i * dim + i] = 200.0 + 2.0;
    }
    for i in 0..(dim - 1) {
        hessian[i * dim + i + 1] = -200.0;
        hessian[(i + 1) * dim + i] = -200.0;
    }

    let hess_decomp = eigh_householder_qr(&hessian, dim);
    let mut hess_evals = hess_decomp.eigenvalues;
    hess_evals.sort_by(f64::total_cmp);
    let hess_cond = spectral_condition_number(&hess_evals);
    let hess_bw = spectral_bandwidth(&hess_evals);
    let (hess_centers, hess_amps) = empirical_spectral_density(&hess_evals, 20);

    s.ecosystem.primals.push(node(
        "hessian_eigen",
        "Hessian Eigenanalysis (Rosenbrock)",
        "compute",
        200.0,
        300.0,
        &["science.hessian_eigen"],
        vec![
            spectrum(
                "hessian-spectrum",
                "Hessian Eigenvalue Spectrum",
                "dimensionless",
                hess_centers,
                hess_amps,
            ),
            gauge(
                "hessian-condition",
                "Condition Number",
                hess_cond,
                1.0,
                1000.0,
                "ratio",
                [1.0, 100.0],
                [100.0, 500.0],
            ),
            gauge(
                "hessian-bandwidth",
                "Spectral Bandwidth",
                hess_bw,
                0.0,
                500.0,
                "dimensionless",
                [0.0, 200.0],
                [200.0, 400.0],
            ),
        ],
        vec![],
    ));

    let edges = vec![
        edge(
            "anderson_sweep",
            "spectral_analysis",
            "disorder parameterizes spectral",
        ),
        edge(
            "spectral_analysis",
            "hessian_eigen",
            "spectral theory → loss landscape",
        ),
    ];

    (s, edges)
}
