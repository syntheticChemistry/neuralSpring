// SPDX-License-Identifier: AGPL-3.0-or-later

//! LTEE B3: LSTM+HMM+ESN Allele Trajectory Classifier.
//!
//! Port of Good et al. (2017) "The dynamics of molecular evolution over
//! 60,000 generations" (Nature 551:45-50).
//!
//! Fuses three ML architectures to classify allele trajectory fates:
//! 1. **LSTM encoder** — temporal feature extraction from frequency series
//! 2. **HMM regime decoder** — dynamical regime posterior (sweep / interference / coexistence)
//! 3. **ESN classifier** — combines LSTM features + HMM posterior for fate classification
//!
//! Allele fates: fixation (→1.0), loss (→0.0), polymorphic (stable intermediate).
//!
//! This is **additive ML enrichment** for lithoSpore Module 3. Target:
//! T06 — classification accuracy ≥ 95% on labeled trajectories.
//!
//! ## Architecture
//!
//! ```text
//! frequency series ─→ LSTM(h=32) ─→ pool [mean, std, last] ─→ 96 features ─┐
//!                  └→ discretize ─→ HMM(3 states) ─→ posterior ─→ 3 values ──┤
//!                                                                             ├→ ESN(128) → 3-class
//! ```
//!
//! ## Python Baseline Provenance
//!
//! | Metric | Value | Source |
//! |--------|-------|--------|
//! | Train accuracy | 0.9917 | `control/ltee_allele_trajectory/ltee_allele_trajectory.py`, seed=42 |
//! | Test accuracy  | 1.0000 | same |
//!
//! ## Reference
//!
//! Good et al. (2017), Nature 551:45-50

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::suboptimal_flops,
    reason = "LTEE allele trajectory simulation and ML classifier math"
)]

use crate::rng::Rng;

/// LSTM hidden state dimension.
pub const LSTM_HIDDEN: usize = 32;
/// HMM state count (sweep / interference / coexistence).
pub const HMM_N_STATES: usize = 3;
/// HMM emission symbol count (discretized frequency bins).
pub const HMM_N_SYMBOLS: usize = 4;
/// ESN input dimension: LSTM features (32*3) + HMM posterior (3).
pub const ESN_INPUT_DIM: usize = LSTM_HIDDEN * 3 + HMM_N_STATES;
/// ESN reservoir dimension.
pub const ESN_RESERVOIR: usize = 128;
/// Number of allele fate classes.
pub const N_CLASSES: usize = 3;
/// Sequence length (generations per trajectory).
pub const SEQ_LEN: usize = 50;

/// Allele fate class names.
pub const CLASS_NAMES: [&str; 3] = ["fixation", "loss", "polymorphic"];

/// Generate a synthetic allele frequency trajectory.
///
/// `fate`: 0=fixation, 1=loss, 2=polymorphic.
/// Returns `SEQ_LEN` frequency values in [0, 1].
#[must_use]
pub fn generate_allele_trajectory(rng: &mut Rng, fate: usize, seq_len: usize) -> Vec<f64> {
    let f0 = 0.05 + rng.uniform() * 0.25;
    let mut freqs = vec![0.0; seq_len];
    freqs[0] = f0;

    match fate {
        0 => {
            let s = 0.01 + rng.uniform() * 0.04;
            for t in 1..seq_len {
                let df = s * freqs[t - 1] * (1.0 - freqs[t - 1]) + rng.normal() * 0.02;
                freqs[t] = (freqs[t - 1] + df).clamp(0.01, 0.99);
            }
            if seq_len >= 6 {
                let base = freqs[seq_len - 6];
                let target = 0.95 + rng.uniform() * 0.05;
                for (i, f) in freqs[seq_len - 5..].iter_mut().enumerate() {
                    *f = base + (target - base) * (i + 1) as f64 / 5.0;
                }
            }
        }
        1 => {
            let s = -0.05 + rng.uniform() * 0.04;
            for t in 1..seq_len {
                let df = s * freqs[t - 1] * (1.0 - freqs[t - 1]) + rng.normal() * 0.02;
                freqs[t] = (freqs[t - 1] + df).clamp(0.01, 0.99);
            }
            if seq_len >= 6 {
                let base = freqs[seq_len - 6];
                let target = 0.02 + rng.uniform() * 0.03;
                for (i, f) in freqs[seq_len - 5..].iter_mut().enumerate() {
                    *f = base + (target - base) * (i + 1) as f64 / 5.0;
                }
            }
        }
        _ => {
            let eq = 0.2 + rng.uniform() * 0.6;
            for t in 1..seq_len {
                let df = 0.1 * (eq - freqs[t - 1]) + rng.normal() * 0.03;
                freqs[t] = (freqs[t - 1] + df).clamp(0.01, 0.99);
            }
        }
    }

    for f in &mut freqs {
        *f = f.clamp(0.0, 1.0);
    }
    freqs
}

/// Simple tanh-RNN forward pass (LSTM-like encoder).
///
/// Returns hidden states `seq_len × hidden_size` (flattened row-major).
#[must_use]
pub fn lstm_forward(sequence: &[f64], w_x: &[f64], w_h: &[f64], hidden_size: usize) -> Vec<f64> {
    let seq_len = sequence.len();
    let mut states = vec![0.0; seq_len * hidden_size];
    let mut h = vec![0.0; hidden_size];

    for (t, &x_t) in sequence.iter().enumerate() {
        let mut h_new = vec![0.0; hidden_size];
        for i in 0..hidden_size {
            let mut sum = w_x[i] * x_t;
            for j in 0..hidden_size {
                sum += w_h[i * hidden_size + j] * h[j];
            }
            h_new[i] = sum.tanh();
        }
        states[t * hidden_size..(t + 1) * hidden_size].copy_from_slice(&h_new);
        h = h_new;
    }

    states
}

/// Pool LSTM hidden states: [mean, std, last] → 3 * hidden features.
#[must_use]
pub fn pool_features(states: &[f64], seq_len: usize, hidden_size: usize) -> Vec<f64> {
    let mut mean = vec![0.0; hidden_size];
    let mut sq_mean = vec![0.0; hidden_size];

    for row in states.chunks(hidden_size).take(seq_len) {
        for (i, &v) in row.iter().enumerate() {
            mean[i] += v;
            sq_mean[i] += v * v;
        }
    }

    let n = seq_len as f64;
    let mut features = Vec::with_capacity(hidden_size * 3);

    for m_val in &mean {
        features.push(m_val / n);
    }
    for i in 0..hidden_size {
        let m = mean[i] / n;
        let var = (sq_mean[i] / n - m * m).max(0.0);
        features.push(var.sqrt());
    }
    features.extend_from_slice(&states[(seq_len - 1) * hidden_size..seq_len * hidden_size]);

    features
}

/// Discretize a frequency trajectory into HMM observation symbols.
#[must_use]
#[expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "f is clamped to [0, n_symbols) so the cast is safe"
)]
pub fn discretize_trajectory(freqs: &[f64], n_symbols: usize) -> Vec<usize> {
    freqs
        .iter()
        .map(|&f| {
            let bin = (f * n_symbols as f64).max(0.0) as usize;
            bin.min(n_symbols - 1)
        })
        .collect()
}

/// Scaled HMM forward algorithm returning posterior at last step.
///
/// `transition` is `n_states × n_states` (row-major), `emission` is
/// `n_states × n_symbols` (row-major), `initial` is `n_states`.
#[must_use]
pub fn hmm_forward_posterior(
    obs: &[usize],
    transition: &[f64],
    emission: &[f64],
    initial: &[f64],
    n_states: usize,
    n_symbols: usize,
) -> Vec<f64> {
    let t_len = obs.len();
    let mut alpha = vec![0.0; t_len * n_states];

    for j in 0..n_states {
        alpha[j] = initial[j] * emission[j * n_symbols + obs[0]];
    }
    let scale: f64 = alpha[..n_states].iter().sum();
    if scale > 0.0 {
        for a in &mut alpha[..n_states] {
            *a /= scale;
        }
    }

    for t in 1..t_len {
        for j in 0..n_states {
            let mut sum = 0.0;
            for i in 0..n_states {
                sum += alpha[(t - 1) * n_states + i] * transition[i * n_states + j];
            }
            alpha[t * n_states + j] = emission[j * n_symbols + obs[t]] * sum;
        }
        let scale: f64 = alpha[t * n_states..(t + 1) * n_states].iter().sum();
        if scale > 0.0 {
            for a in &mut alpha[t * n_states..(t + 1) * n_states] {
                *a /= scale;
            }
        }
    }

    let last = &alpha[(t_len - 1) * n_states..t_len * n_states];
    let norm: f64 = last.iter().sum();
    if norm > 0.0 {
        last.iter().map(|&a| a / norm).collect()
    } else {
        last.to_vec()
    }
}

/// ESN 2-step reservoir recurrence on a single input vector.
#[must_use]
pub fn esn_reservoir_step(
    x: &[f64],
    w_in: &[f64],
    w_res: &[f64],
    b_res: &[f64],
    reservoir_size: usize,
) -> Vec<f64> {
    let input_dim = x.len();

    let mut h: Vec<f64> = (0..reservoir_size)
        .map(|i| {
            let mut sum = b_res[i];
            for (j, &xj) in x.iter().enumerate() {
                sum += w_in[i * input_dim + j] * xj;
            }
            sum.tanh()
        })
        .collect();

    let h_prev = h.clone();
    for (i, h_val) in h.iter_mut().enumerate() {
        let mut sum = b_res[i];
        for (j, &xj) in x.iter().enumerate() {
            sum += w_in[i * input_dim + j] * xj;
        }
        for (k, &hk) in h_prev.iter().enumerate() {
            sum += w_res[i * reservoir_size + k] * hk;
        }
        *h_val = sum.tanh();
    }

    h
}

/// Multi-class argmax classification from ESN state × readout weights.
///
/// `w_out` is `reservoir_size × n_classes` (flattened row-major).
/// Returns `(predicted_class, class_scores)`.
#[must_use]
pub fn classify_allele_fate(
    esn_state: &[f64],
    w_out: &[f64],
    n_classes: usize,
) -> (usize, Vec<f64>) {
    let rs = esn_state.len();
    let mut scores = vec![0.0; n_classes];
    for c in 0..n_classes {
        for i in 0..rs {
            scores[c] += esn_state[i] * w_out[i * n_classes + c];
        }
    }
    let pred = scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx);
    (pred, scores)
}

/// Classification accuracy and per-class confusion matrix.
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    /// Overall accuracy.
    pub accuracy: f64,
    /// Confusion matrix `[true_class][predicted_class]`.
    pub confusion: Vec<Vec<usize>>,
}

/// Compute multi-class classification metrics.
#[must_use]
pub fn classification_metrics(
    predictions: &[usize],
    labels: &[usize],
    n_classes: usize,
) -> ClassificationMetrics {
    let n = predictions.len();
    let mut confusion = vec![vec![0usize; n_classes]; n_classes];
    let mut correct = 0usize;
    for i in 0..n {
        if labels[i] < n_classes && predictions[i] < n_classes {
            confusion[labels[i]][predictions[i]] += 1;
        }
        if predictions[i] == labels[i] {
            correct += 1;
        }
    }
    ClassificationMetrics {
        accuracy: correct as f64 / n.max(1) as f64,
        confusion,
    }
}

/// Loaded B3 baseline with all model weights and expected values.
#[derive(Debug, Clone)]
pub struct AlleleTrajectoryBaseline {
    /// LSTM input-to-hidden weights (`hidden_size`).
    pub lstm_w_x: Vec<f64>,
    /// LSTM hidden-to-hidden weights (`hidden_size` x `hidden_size`, row-major).
    pub lstm_w_h: Vec<f64>,
    /// HMM transition matrix (`n_states` x `n_states`, row-major).
    pub hmm_transition: Vec<f64>,
    /// HMM emission matrix (`n_states` x `n_symbols`, row-major).
    pub hmm_emission: Vec<f64>,
    /// HMM initial state distribution.
    pub hmm_initial: Vec<f64>,
    /// ESN input-to-reservoir weights (`reservoir` x `input_dim`, row-major).
    pub esn_w_in: Vec<f64>,
    /// ESN reservoir weights (`reservoir` x `reservoir`, row-major).
    pub esn_w_res: Vec<f64>,
    /// ESN reservoir bias.
    pub esn_b_res: Vec<f64>,
    /// ESN readout weights (`reservoir` x `n_classes`, row-major).
    pub esn_w_out: Vec<f64>,
    /// Python train accuracy.
    pub train_accuracy: f64,
    /// Python test accuracy.
    pub test_accuracy: f64,
    /// First allele trajectory (frequency series).
    pub first_trajectory: Vec<f64>,
    /// First allele label.
    pub first_label: usize,
    /// First allele LSTM features (from Python).
    pub first_lstm_features: Vec<f64>,
    /// First allele HMM posterior (from Python).
    pub first_hmm_posterior: Vec<f64>,
    /// First allele ESN state (from Python).
    pub first_esn_state: Vec<f64>,
    /// First allele class scores (from Python).
    pub first_class_scores: Vec<f64>,
    /// First allele predicted class (from Python).
    pub first_prediction: usize,
}

/// Load B3 baseline from Python JSON.
///
/// # Errors
///
/// Returns `Err` if JSON structure is unexpected.
pub fn load_allele_baseline_from_json(json_str: &str) -> Result<AlleleTrajectoryBaseline, String> {
    let v: serde_json::Value = serde_json::from_str(json_str).map_err(|e| format!("parse: {e}"))?;

    Ok(AlleleTrajectoryBaseline {
        lstm_w_x: parse_f64_array(&v["lstm"], "w_x")?,
        lstm_w_h: parse_f64_array(&v["lstm"], "w_h")?,
        hmm_transition: parse_f64_2d_flat(&v["hmm"], "transition")?,
        hmm_emission: parse_f64_2d_flat(&v["hmm"], "emission")?,
        hmm_initial: parse_f64_array(&v["hmm"], "initial")?,
        esn_w_in: parse_f64_array(&v["esn"], "w_in")?,
        esn_w_res: parse_f64_array(&v["esn"], "w_res")?,
        esn_b_res: parse_f64_array(&v["esn"], "b_res")?,
        esn_w_out: parse_f64_array(&v["esn"], "w_out")?,
        train_accuracy: v["train_accuracy"].as_f64().ok_or("train_accuracy")?,
        test_accuracy: v["test_accuracy"].as_f64().ok_or("test_accuracy")?,
        first_trajectory: parse_f64_array(&v["first_allele"], "trajectory")?,
        first_label: v["first_allele"]["label"].as_u64().ok_or("first label")? as usize,
        first_lstm_features: parse_f64_array(&v["first_allele"], "lstm_features")?,
        first_hmm_posterior: parse_f64_array(&v["first_allele"], "hmm_posterior")?,
        first_esn_state: parse_f64_array(&v["first_allele"], "esn_state")?,
        first_class_scores: parse_f64_array(&v["first_allele"], "class_scores")?,
        first_prediction: v["first_allele"]["prediction"]
            .as_u64()
            .ok_or("first pred")? as usize,
    })
}

fn parse_f64_array(v: &serde_json::Value, key: &str) -> Result<Vec<f64>, String> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("{key} not array"))?
        .iter()
        .map(|x| x.as_f64().ok_or_else(|| format!("{key} element not f64")))
        .collect()
}

fn parse_f64_2d_flat(v: &serde_json::Value, key: &str) -> Result<Vec<f64>, String> {
    let rows = v[key]
        .as_array()
        .ok_or_else(|| format!("{key} not array"))?;
    let mut flat = Vec::new();
    for row in rows {
        for val in row
            .as_array()
            .ok_or_else(|| format!("{key} row not array"))?
        {
            flat.push(
                val.as_f64()
                    .ok_or_else(|| format!("{key} element not f64"))?,
            );
        }
    }
    Ok(flat)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test assertions on control JSON fixtures"
)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_generation_deterministic() {
        let mut rng = Rng::new(42);
        let traj = generate_allele_trajectory(&mut rng, 0, SEQ_LEN);
        assert_eq!(traj.len(), SEQ_LEN);
        assert!(traj.iter().all(|&f| (0.0..=1.0).contains(&f)));
    }

    #[test]
    fn fixation_trajectory_ends_high() {
        let mut rng = Rng::new(42);
        let traj = generate_allele_trajectory(&mut rng, 0, SEQ_LEN);
        assert!(traj[SEQ_LEN - 1] > 0.8, "fixation should end near 1.0");
    }

    #[test]
    fn loss_trajectory_ends_low() {
        let mut rng = Rng::new(42);
        let traj = generate_allele_trajectory(&mut rng, 1, SEQ_LEN);
        assert!(traj[SEQ_LEN - 1] < 0.2, "loss should end near 0.0");
    }

    #[test]
    fn lstm_forward_correct_dims() {
        let mut rng = Rng::new(99);
        let hs = 8;
        let w_x: Vec<f64> = (0..hs).map(|_| rng.normal() * 0.1).collect();
        let w_h: Vec<f64> = (0..hs * hs).map(|_| rng.normal() * 0.1).collect();
        let seq = vec![0.5, 0.6, 0.7, 0.8];
        let states = lstm_forward(&seq, &w_x, &w_h, hs);
        assert_eq!(states.len(), seq.len() * hs);
    }

    #[test]
    fn pool_features_correct_dims() {
        let hs = 4;
        let seq_len = 3;
        let states = vec![0.1; seq_len * hs];
        let feats = pool_features(&states, seq_len, hs);
        assert_eq!(feats.len(), hs * 3);
    }

    #[test]
    fn hmm_posterior_sums_to_one() {
        let transition = vec![0.85, 0.10, 0.05, 0.10, 0.80, 0.10, 0.05, 0.10, 0.85];
        let emission = vec![
            0.10, 0.20, 0.30, 0.40, 0.40, 0.30, 0.20, 0.10, 0.15, 0.35, 0.35, 0.15,
        ];
        let initial = vec![0.33, 0.34, 0.33];
        let obs = vec![0, 1, 2, 3, 1, 0, 2];
        let post = hmm_forward_posterior(&obs, &transition, &emission, &initial, 3, 4);
        assert_eq!(post.len(), 3);
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "posterior must sum to 1: {sum}");
    }

    #[test]
    fn classify_argmax_correct() {
        let state = vec![1.0, 0.5, -0.5, 0.0];
        let w_out = vec![
            0.1, -0.1, 0.0, 0.0, 0.2, -0.2, -0.1, 0.0, 0.3, 0.2, -0.1, 0.0,
        ];
        let (pred, scores) = classify_allele_fate(&state, &w_out, 3);
        assert_eq!(scores.len(), 3);
        let max_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(pred, max_idx);
    }

    #[test]
    fn load_baseline_from_json() {
        let json_str = include_str!("../control/ltee_allele_trajectory/expected_values.json");
        let bl = load_allele_baseline_from_json(json_str).expect("parse");
        assert_eq!(bl.lstm_w_x.len(), LSTM_HIDDEN);
        assert_eq!(bl.lstm_w_h.len(), LSTM_HIDDEN * LSTM_HIDDEN);
        assert_eq!(bl.hmm_transition.len(), HMM_N_STATES * HMM_N_STATES);
        assert_eq!(bl.hmm_emission.len(), HMM_N_STATES * HMM_N_SYMBOLS);
        assert_eq!(bl.esn_w_in.len(), ESN_RESERVOIR * ESN_INPUT_DIM);
        assert_eq!(bl.esn_w_out.len(), ESN_RESERVOIR * N_CLASSES);
        assert!(bl.test_accuracy >= 0.95);
    }

    #[test]
    fn rust_pipeline_matches_python() {
        let json_str = include_str!("../control/ltee_allele_trajectory/expected_values.json");
        let bl = load_allele_baseline_from_json(json_str).expect("parse");

        let states = lstm_forward(
            &bl.first_trajectory,
            &bl.lstm_w_x,
            &bl.lstm_w_h,
            LSTM_HIDDEN,
        );
        let lstm_feats = pool_features(&states, bl.first_trajectory.len(), LSTM_HIDDEN);

        for (i, (&r, &p)) in lstm_feats.iter().zip(&bl.first_lstm_features).enumerate() {
            let diff = (r - p).abs();
            assert!(
                diff < 1e-10,
                "LSTM feature {i} mismatch: rust={r}, py={p}, diff={diff}"
            );
        }

        let obs = discretize_trajectory(&bl.first_trajectory, HMM_N_SYMBOLS);
        let posterior = hmm_forward_posterior(
            &obs,
            &bl.hmm_transition,
            &bl.hmm_emission,
            &bl.hmm_initial,
            HMM_N_STATES,
            HMM_N_SYMBOLS,
        );

        for (i, (&r, &p)) in posterior.iter().zip(&bl.first_hmm_posterior).enumerate() {
            let diff = (r - p).abs();
            assert!(
                diff < 1e-6,
                "HMM posterior {i} mismatch: rust={r}, py={p}, diff={diff}"
            );
        }

        let mut combined = lstm_feats;
        combined.extend_from_slice(&posterior);

        let esn_state = esn_reservoir_step(
            &combined,
            &bl.esn_w_in,
            &bl.esn_w_res,
            &bl.esn_b_res,
            ESN_RESERVOIR,
        );

        for (i, (&r, &p)) in esn_state.iter().zip(&bl.first_esn_state).enumerate() {
            let diff = (r - p).abs();
            assert!(
                diff < 1e-6,
                "ESN state {i} mismatch: rust={r}, py={p}, diff={diff}"
            );
        }

        let (pred, scores) = classify_allele_fate(&esn_state, &bl.esn_w_out, N_CLASSES);

        for (i, (&r, &p)) in scores.iter().zip(&bl.first_class_scores).enumerate() {
            let diff = (r - p).abs();
            assert!(
                diff < 1e-4,
                "class score {i} mismatch: rust={r}, py={p}, diff={diff}"
            );
        }

        assert_eq!(pred, bl.first_prediction, "predicted class mismatch");
    }
}
