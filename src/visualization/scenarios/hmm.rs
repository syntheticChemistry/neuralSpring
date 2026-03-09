// SPDX-License-Identifier: AGPL-3.0-or-later

//! HMM phylogenetics scenario builder (Papers 016-018).
//!
//! Produces 2 nodes: forward log-likelihood trace and Viterbi decoded states,
//! backed by a small 3-state HMM with real forward/Viterbi computation.

#![expect(
    clippy::cast_precision_loss,
    reason = "index-to-f64 conversions for visualization axes"
)]

use crate::hmm::Hmm;
use crate::rng::Rng;
use crate::visualization::types::{NeuralScenario, ScenarioEdge, ThresholdRange};

use super::{bar, edge, gauge, heatmap, node, scaffold, timeseries};

/// Build the HMM phylogenetics scenario.
///
/// Nodes:
/// - `hmm_forward`: transition matrix heatmap + log-likelihood trace
/// - `hmm_viterbi`: decoded state bar chart + accuracy gauge
#[expect(
    clippy::too_many_lines,
    reason = "scenario builder — single cohesive builder"
)]
#[must_use]
pub fn hmm_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "HMM Phylogenetics",
        "Hidden Markov Model: forward algorithm, Viterbi decoding, transition structure",
    );

    let n_states = 3;
    let n_obs = 4;
    let trans = vec![
        vec![0.7, 0.2, 0.1],
        vec![0.1, 0.6, 0.3],
        vec![0.2, 0.2, 0.6],
    ];
    let emit = vec![
        vec![0.4, 0.3, 0.2, 0.1],
        vec![0.1, 0.4, 0.3, 0.2],
        vec![0.2, 0.1, 0.3, 0.4],
    ];
    let init = vec![0.5, 0.3, 0.2];
    let hmm = Hmm::new(trans.clone(), emit, init);

    let mut rng = Rng::new(42);
    let (true_states, observations) = hmm.generate_sequence(50, &mut rng);

    let trans_flat: Vec<f64> = trans.iter().flat_map(|row| row.iter().copied()).collect();
    let state_labels: Vec<String> = (0..n_states).map(|i| format!("S{i}")).collect();

    let mut ll_trace = Vec::with_capacity(observations.len());
    for t in 1..=observations.len() {
        let (_, ll) = hmm.forward(&observations[..t]);
        ll_trace.push(ll);
    }
    let time_steps: Vec<f64> = (1..=observations.len()).map(|t| t as f64).collect();

    s.ecosystem.primals.push(node(
        "hmm_forward",
        "HMM Forward Algorithm",
        "compute",
        0.0,
        0.0,
        &["science.hmm_forward"],
        vec![
            heatmap(
                "transition-matrix",
                "Transition Matrix P(i→j)",
                state_labels.clone(),
                state_labels.clone(),
                trans_flat,
                "probability",
            ),
            timeseries(
                "log-likelihood-trace",
                "Cumulative Log-Likelihood",
                "Time step",
                "Log P(O₁..ₜ)",
                "nats",
                time_steps,
                ll_trace,
            ),
        ],
        vec![],
    ));

    let (viterbi_path, viterbi_ll) = hmm.viterbi(&observations);
    let correct = true_states
        .iter()
        .zip(viterbi_path.iter())
        .filter(|(a, b)| a == b)
        .count();
    let accuracy = correct as f64 / true_states.len() as f64;

    let mut state_counts = vec![0.0; n_states];
    for &st in &viterbi_path {
        if st < n_states {
            state_counts[st] += 1.0;
        }
    }

    let obs_labels: Vec<String> = (0..n_obs).map(|i| format!("O{i}")).collect();
    let _ = obs_labels;

    s.ecosystem.primals.push(node(
        "hmm_viterbi",
        &format!("Viterbi Decoding (LL={viterbi_ll:.2})"),
        "compute",
        400.0,
        0.0,
        &["science.hmm_viterbi"],
        vec![
            bar(
                "decoded-state-counts",
                "Viterbi State Distribution",
                state_labels,
                state_counts,
                "count",
            ),
            gauge(
                "viterbi-accuracy",
                "Decode Accuracy",
                accuracy,
                0.0,
                1.0,
                "fraction",
                [0.7, 1.0],
                [0.4, 0.7],
            ),
        ],
        vec![ThresholdRange {
            label: "Good decoding (>70%)".into(),
            min: 0.7,
            max: 1.0,
            status: "normal".into(),
        }],
    ));

    let edges = vec![edge("hmm_forward", "hmm_viterbi", "forward → decode")];
    (s, edges)
}
