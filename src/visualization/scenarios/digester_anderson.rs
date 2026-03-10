// SPDX-License-Identifier: AGPL-3.0-or-later

//! Digester×Anderson coupling scenario builder (Exp 097).
//!
//! Visualizes how microbial community disorder (Shannon diversity → Anderson
//! disorder) correlates with ESN prediction accuracy.

use crate::digester_anderson::{community_anderson, evenness_to_disorder};
use crate::rng::Rng;
use crate::visualization::types::{NeuralScenario, ScenarioEdge, ThresholdRange};

use super::{edge, gauge, node, scaffold, timeseries};

/// Build the digester×Anderson coupling scenario.
///
/// Nodes:
/// - `digester_community`: Shannon diversity sweep → disorder mapping
/// - `anderson_coupling`: disorder vs localization (IPR sweep)
/// - `esn_accuracy`: ESN accuracy vs disorder correlation
#[must_use]
#[expect(clippy::too_many_lines, reason = "3 nodes with rich data channels")]
pub fn digester_anderson_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Digester × Anderson Coupling",
        "Community disorder drives Anderson localization, which predicts ESN accuracy degradation",
    );

    let mut rng = Rng::new(42);
    let n_species = 10;
    let n_samples = 8;

    let mut diversities = Vec::with_capacity(n_samples);
    let mut disorders = Vec::with_capacity(n_samples);
    let mut iprs = Vec::with_capacity(n_samples);
    let mut xi_vals = Vec::with_capacity(n_samples);

    let alphas = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0];

    for &alpha in &alphas {
        let (h, _evenness, w, ipr, xi) =
            community_anderson(n_species, alpha, 20, &mut rng);
        diversities.push(h);
        disorders.push(w);
        iprs.push(ipr);
        xi_vals.push(xi);
    }

    s.ecosystem.primals.push(node(
        "digester_community",
        "Community Diversity Sweep",
        "compute",
        0.0,
        0.0,
        &["science.community_ecology", "science.shannon_diversity"],
        vec![
            timeseries(
                "diversity-vs-disorder",
                "Shannon Diversity → Anderson Disorder",
                "Shannon H",
                "Disorder W",
                "dimensionless",
                diversities.clone(),
                disorders.clone(),
            ),
            timeseries(
                "disorder-vs-ipr",
                "Disorder → Mean IPR (localization)",
                "Disorder W",
                "Mean IPR",
                "dimensionless",
                disorders.clone(),
                iprs.clone(),
            ),
        ],
        vec![
            ThresholdRange {
                label: "Delocalized (healthy community)".into(),
                min: 0.0,
                max: 0.2,
                status: "normal".into(),
            },
            ThresholdRange {
                label: "Localized (disturbed)".into(),
                min: 0.5,
                max: 1.0,
                status: "warning".into(),
            },
        ],
    ));

    let w_val = disorders[disorders.len() / 2];
    let ipr_sample = iprs[iprs.len() / 2];

    s.ecosystem.primals.push(node(
        "anderson_coupling",
        &format!("Anderson Localization (W={w_val:.1})"),
        "compute",
        400.0,
        0.0,
        &["science.anderson_localization"],
        vec![
            gauge(
                "coupling-ipr",
                "Mean IPR at midpoint",
                ipr_sample,
                0.0,
                1.0,
                "dimensionless",
                [0.0, 0.2],
                [0.2, 0.6],
            ),
            timeseries(
                "xi-vs-disorder",
                "Localization Length vs Disorder",
                "Disorder W",
                "ξ (localization length)",
                "dimensionless",
                disorders,
                xi_vals,
            ),
        ],
        vec![],
    ));

    let evenness_sweep: Vec<f64> = (1..=8_u32).map(|i| f64::from(i) / 8.0).collect();
    let accuracy_proxy: Vec<f64> = evenness_sweep
        .iter()
        .map(|&e| {
            let w = evenness_to_disorder(e);
            1.0 / 0.1_f64.mul_add(w, 1.0)
        })
        .collect();

    s.ecosystem.primals.push(node(
        "esn_accuracy",
        "ESN Accuracy vs Community State",
        "compute",
        200.0,
        400.0,
        &["science.esn_prediction", "science.digester_performance"],
        vec![timeseries(
            "evenness-vs-accuracy",
            "Community Evenness → Predicted R²",
            "Evenness",
            "R² (proxy)",
            "fraction",
            evenness_sweep,
            accuracy_proxy,
        )],
        vec![ThresholdRange {
            label: "Good prediction (R²>0.8)".into(),
            min: 0.8,
            max: 1.0,
            status: "normal".into(),
        }],
    ));

    let edges = vec![
        edge(
            "digester_community",
            "anderson_coupling",
            "diversity → disorder",
        ),
        edge(
            "anderson_coupling",
            "esn_accuracy",
            "localization → accuracy loss",
        ),
    ];

    (s, edges)
}
