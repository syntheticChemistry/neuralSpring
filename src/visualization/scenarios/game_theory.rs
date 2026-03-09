// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evolutionary game theory scenario builder (Papers 019-021).
//!
//! Produces 2 nodes: replicator dynamics with payoff matrix and
//! quorum-sensing spatial cooperation model.

#![expect(
    clippy::cast_precision_loss,
    reason = "index-to-f64 conversions for visualization axes"
)]

use crate::game_theory::{prisoners_dilemma_payoff, replicator_dynamics, QsConfig, QsResult};
use crate::visualization::types::{NeuralScenario, ScenarioEdge, ThresholdRange};

use super::{edge, gauge, heatmap, node, scaffold, timeseries};

/// Build the evolutionary game theory scenario.
///
/// Nodes:
/// - `replicator_dynamics`: payoff heatmap + cooperation frequency trace
/// - `qs_cooperation`: QS cooperation frequency + mean fitness traces
#[expect(
    clippy::too_many_lines,
    reason = "scenario builder — single cohesive builder"
)]
#[must_use]
pub fn game_theory_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Evolutionary Game Theory",
        "Replicator dynamics and quorum-sensing cooperation in structured populations",
    );

    let b = 3.0;
    let c = 1.0;
    let payoff = prisoners_dilemma_payoff(b, c);
    let payoff_flat = vec![payoff[0][0], payoff[0][1], payoff[1][0], payoff[1][1]];
    let strategy_labels = vec!["Cooperate".into(), "Defect".into()];

    let trajectory = replicator_dynamics(&[0.5, 0.5], &payoff, 200, 0.05);
    let rep_time: Vec<f64> = (0..trajectory.len()).map(|t| t as f64 * 0.05).collect();
    let coop_freq: Vec<f64> = trajectory.iter().map(|s| s[0]).collect();

    let final_coop = coop_freq.last().copied().unwrap_or(0.0);
    let nash_distance = final_coop.min(1.0 - final_coop);

    s.ecosystem.primals.push(node(
        "replicator_dynamics",
        "Replicator Dynamics (PD)",
        "compute",
        0.0,
        0.0,
        &["science.replicator_dynamics"],
        vec![
            heatmap(
                "payoff-matrix",
                "Prisoner's Dilemma Payoff",
                strategy_labels.clone(),
                strategy_labels,
                payoff_flat,
                "fitness",
            ),
            timeseries(
                "cooperation-frequency",
                "Cooperation Frequency",
                "Time",
                "P(Cooperate)",
                "probability",
                rep_time,
                coop_freq,
            ),
            gauge(
                "nash-distance",
                "Distance to Nash Equilibrium",
                nash_distance,
                0.0,
                0.5,
                "probability",
                [0.0, 0.05],
                [0.05, 0.2],
            ),
        ],
        vec![ThresholdRange {
            label: "Near Nash equilibrium".into(),
            min: 0.0,
            max: 0.05,
            status: "normal".into(),
        }],
    ));

    let qs_cfg = QsConfig {
        pop_size: 200,
        n_gen: 100,
        qs_threshold: 0.3,
        cooperation_cost: 0.1,
        cooperation_benefit: 0.5,
        dispersal_bonus: 0.05,
        mutation_rate: 0.008,
        seed: 42,
    };
    let QsResult {
        coop_freq: qs_coop,
        mean_fitness: qs_fit,
    } = crate::game_theory::qs_cooperation_model(&qs_cfg);

    let qs_gen: Vec<f64> = (0..qs_coop.len()).map(|g| g as f64).collect();

    s.ecosystem.primals.push(node(
        "qs_cooperation",
        "Quorum-Sensing Cooperation",
        "compute",
        400.0,
        0.0,
        &["science.qs_cooperation"],
        vec![
            timeseries(
                "qs-cooperation-freq",
                "QS Cooperation Frequency",
                "Generation",
                "Cooperation",
                "fraction",
                qs_gen.clone(),
                qs_coop,
            ),
            timeseries(
                "qs-mean-fitness",
                "QS Mean Fitness",
                "Generation",
                "Fitness",
                "dimensionless",
                qs_gen,
                qs_fit,
            ),
        ],
        vec![],
    ));

    let edges = vec![edge(
        "replicator_dynamics",
        "qs_cooperation",
        "classical → spatial dynamics",
    )];
    (s, edges)
}
