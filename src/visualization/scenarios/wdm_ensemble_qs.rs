// SPDX-License-Identifier: AGPL-3.0-or-later

//! WDM ensemble quorum sensing scenario builder (Exp 099).
//!
//! Visualizes the pipeline: surrogate disagreement → Anderson disorder →
//! localization → QS cooperation dynamics.

use crate::rng::Rng;
use crate::wdm_ensemble_qs::{
    anderson_from_disorder, disagreement_to_disorder, replicator_final_coop, snowdrift_payoff,
};
use crate::visualization::types::{NeuralScenario, ScenarioEdge, ThresholdRange};

use super::{edge, gauge, node, scaffold, timeseries};

/// Build the WDM ensemble QS scenario.
///
/// Nodes:
/// - `ensemble_disagreement`: disagreement → disorder mapping
/// - `anderson_phase`: disorder → IPR (localization)
/// - `qs_dynamics`: disorder → cooperation via replicator dynamics
#[must_use]
#[expect(clippy::too_many_lines, reason = "3 nodes with rich data channels")]
pub fn wdm_ensemble_qs_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "WDM Ensemble Quorum Sensing",
        "Surrogate disagreement maps to Anderson disorder, driving QS cooperation dynamics",
    );

    let n_slices = 10;
    let mut rng = Rng::new(42);

    let mut disagreements = Vec::with_capacity(n_slices);
    let mut disorders = Vec::with_capacity(n_slices);
    let mut iprs = Vec::with_capacity(n_slices);
    let mut xi_vals = Vec::with_capacity(n_slices);
    let mut coop_freqs = Vec::with_capacity(n_slices);

    for i in 0..n_slices {
        #[expect(clippy::cast_precision_loss, reason = "i ≤ 10")]
        let disagree = (i as f64).mul_add(0.1, 0.01);
        let w = disagreement_to_disorder(disagree, 0.01, 1.0, 16.0);

        let disorder_vec: Vec<f64> = (0..20).map(|_| rng.uniform() * w).collect();
        let (ipr, xi) = anderson_from_disorder(&disorder_vec);

        let w_frac = w / 16.0;
        let payoff = snowdrift_payoff(w_frac.clamp(0.0, 1.0));
        let coop = replicator_final_coop(&payoff, 500);

        disagreements.push(disagree);
        disorders.push(w);
        iprs.push(ipr);
        xi_vals.push(xi);
        coop_freqs.push(coop);
    }

    s.ecosystem.primals.push(node(
        "ensemble_disagreement",
        "Surrogate Ensemble Disagreement",
        "compute",
        0.0,
        0.0,
        &["science.wdm_surrogates", "science.ensemble_variance"],
        vec![timeseries(
            "disagree-to-disorder",
            "Disagreement → Anderson Disorder",
            "Disagreement (σ²)",
            "Disorder W",
            "dimensionless",
            disagreements.clone(),
            disorders.clone(),
        )],
        vec![],
    ));

    s.ecosystem.primals.push(node(
        "anderson_phase",
        "Anderson Localization Phase",
        "compute",
        400.0,
        0.0,
        &["science.anderson_localization"],
        vec![
            timeseries(
                "disorder-vs-ipr",
                "Disorder → Mean IPR",
                "Disorder W",
                "Mean IPR",
                "dimensionless",
                disorders.clone(),
                iprs,
            ),
            timeseries(
                "disorder-vs-xi",
                "Disorder → Localization Length",
                "Disorder W",
                "ξ",
                "dimensionless",
                disorders,
                xi_vals,
            ),
        ],
        vec![ThresholdRange {
            label: "Delocalized (cooperative)".into(),
            min: 0.0,
            max: 0.3,
            status: "normal".into(),
        }],
    ));

    s.ecosystem.primals.push(node(
        "qs_dynamics",
        "Quorum Sensing Cooperation",
        "compute",
        200.0,
        400.0,
        &["science.game_theory", "science.replicator_dynamics"],
        vec![
            timeseries(
                "disagree-vs-coop",
                "Disagreement → Cooperation Frequency",
                "Disagreement",
                "Cooperation freq",
                "fraction",
                disagreements,
                coop_freqs.clone(),
            ),
            gauge(
                "low-w-coop",
                "Low-disorder Cooperation",
                coop_freqs[0],
                0.0,
                1.0,
                "fraction",
                [0.5, 1.0],
                [0.2, 0.5],
            ),
            gauge(
                "high-w-coop",
                "High-disorder Cooperation",
                coop_freqs[n_slices - 1],
                0.0,
                1.0,
                "fraction",
                [0.5, 1.0],
                [0.2, 0.5],
            ),
        ],
        vec![],
    ));

    let edges = vec![
        edge(
            "ensemble_disagreement",
            "anderson_phase",
            "disagreement → disorder",
        ),
        edge(
            "anderson_phase",
            "qs_dynamics",
            "localization → cooperation",
        ),
    ];

    (s, edges)
}
