// SPDX-License-Identifier: AGPL-3.0-or-later

//! HMM introgression on NN weight layers scenario builder (Exp 100).
//!
//! Visualizes how the PhyloNet-HMM detects anomalous layers in a neural
//! network by treating weight statistics as genomic observations.

#![expect(
    clippy::cast_precision_loss,
    reason = "layer indices and state values ≤ 100 fit in f64 mantissa"
)]

use crate::introgression_nn::{build_nn_hmm, build_null_hmm, detection_metrics};
use crate::rng::Rng;
use crate::visualization::types::{NeuralScenario, ScenarioEdge, ThresholdRange};

use super::{bar, edge, gauge, heatmap, node, scaffold, timeseries};

/// Build the HMM introgression on NN layers scenario.
///
/// Nodes:
/// - `nn_observations`: layer weight statistics + ground truth
/// - `hmm_detection`: Viterbi path vs truth + detection metrics
#[must_use]
#[expect(clippy::too_many_lines, reason = "2 rich nodes")]
pub fn introgression_nn_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "HMM Introgression on NN Layers",
        "PhyloNet-HMM detects anomalous neural network weight layers (TPR=0.97, FPR=0)",
    );

    let hmm = build_nn_hmm();
    let null_hmm = build_null_hmm();
    let n_layers = 100;
    let introgressed: Vec<usize> = (30..60).collect();

    let mut rng = Rng::new(42);
    let mut truth = vec![0_usize; n_layers];
    for &idx in &introgressed {
        if idx < n_layers {
            truth[idx] = 1;
        }
    }

    let mut obs = Vec::with_capacity(n_layers);
    for &state in &truth {
        let val = if state == 1 {
            2_usize
        } else {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "rng.uniform() in [0,1) * 2.0 fits in usize"
            )]
            let v = (rng.uniform() * 2.0) as usize;
            v
        };
        obs.push(val);
    }

    let (viterbi_path, _) = hmm.viterbi(&obs);
    let (tpr, fpr, accuracy) = detection_metrics(&viterbi_path, &truth);

    let (_, log_lik_introg) = hmm.forward(&obs);
    let (_, log_lik_baseline) = null_hmm.forward(&obs);
    let llr = log_lik_introg - log_lik_baseline;

    let layer_indices: Vec<f64> = (0..n_layers).map(|i| i as f64).collect();
    let truth_f64: Vec<f64> = truth.iter().map(|&v| v as f64).collect();
    let path_f64: Vec<f64> = viterbi_path.iter().map(|&v| v as f64).collect();

    s.ecosystem.primals.push(node(
        "nn_observations",
        "NN Layer Observations",
        "compute",
        0.0,
        0.0,
        &["science.neural_network", "science.weight_analysis"],
        vec![
            timeseries(
                "ground-truth",
                "Ground Truth (0=normal, 1=introgressed)",
                "Layer",
                "State",
                "binary",
                layer_indices.clone(),
                truth_f64,
            ),
            timeseries(
                "viterbi-path",
                "Viterbi Decoded Path",
                "Layer",
                "State",
                "binary",
                layer_indices,
                path_f64,
            ),
            heatmap(
                "transition-matrix",
                "HMM Transition Matrix",
                vec!["Normal".into(), "Introgressed".into()],
                vec!["Normal".into(), "Introgressed".into()],
                vec![0.95, 0.05, 0.1, 0.9],
                "probability",
            ),
        ],
        vec![],
    ));

    let state_labels = vec!["Normal".into(), "Introgressed".into()];
    let mut state_counts = vec![0.0_f64; 2];
    for &st in &viterbi_path {
        if st < 2 {
            state_counts[st] += 1.0;
        }
    }

    s.ecosystem.primals.push(node(
        "hmm_detection",
        &format!("Detection Metrics (TPR={tpr:.2}, FPR={fpr:.2})"),
        "compute",
        400.0,
        0.0,
        &[
            "science.hmm_viterbi",
            "science.anomaly_detection",
            "science.introgression",
        ],
        vec![
            bar(
                "detected-state-counts",
                "Decoded State Distribution",
                state_labels,
                state_counts,
                "count",
            ),
            gauge("tpr", "True Positive Rate", tpr, 0.0, 1.0, "fraction", [0.8, 1.0], [0.5, 0.8]),
            gauge("fpr", "False Positive Rate", fpr, 0.0, 1.0, "fraction", [0.0, 0.1], [0.1, 0.3]),
            gauge("accuracy", "Accuracy", accuracy, 0.0, 1.0, "fraction", [0.8, 1.0], [0.5, 0.8]),
            gauge("llr", "Log-Likelihood Ratio", llr, -50.0, 50.0, "nats", [0.0, 50.0], [-10.0, 0.0]),
        ],
        vec![ThresholdRange {
            label: "Good detection (TPR > 0.8)".into(),
            min: 0.8,
            max: 1.0,
            status: "normal".into(),
        }],
    ));

    let edges = vec![edge(
        "nn_observations",
        "hmm_detection",
        "observations → Viterbi decode",
    )];

    (s, edges)
}
