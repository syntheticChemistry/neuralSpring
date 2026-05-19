// SPDX-License-Identifier: AGPL-3.0-or-later

//! LTEE B4: ESN Early-Warning Classifier for Citrate Innovation.
//!
//! Port of Blount et al. (2008) "Historical contingency and the evolution
//! of a key innovation in an experimental population of *Escherichia coli*"
//! (PNAS 105:7899-7906).
//!
//! Trains an ESN reservoir on synthetic LTEE population trajectories to
//! detect pre-potentiation regime shifts before the Cit+ innovation event.
//! This is **additive ML enrichment** for lithoSpore Module 4: the module
//! already validates groundSpring statistics (T07), and this classifier
//! adds early-warning detection capability.
//!
//! ## Architecture
//!
//! ESN(input=4, reservoir=256, output=2) — binary early-warning
//! (pre-potentiation vs normal). Ridge regression readout.
//!
//! Input features per generation window:
//! 1. Mean fitness (relative to ancestral)
//! 2. Fitness variance across clones
//! 3. Allele frequency entropy (Shannon H)
//! 4. Frequency change rate (delta-f per generation)
//!
//! ## Python Baseline Provenance
//!
//! | Metric | Value | Source |
//! |--------|-------|--------|
//! | Train accuracy | 0.9311 | `control/ltee_citrate_esn/ltee_citrate_esn.py`, seed=42 |
//! | Test accuracy  | 0.9433 | same |
//! | Train TPR | 0.4152 | same |
//! | Test TPR  | 0.4194 | same |
//!
//! ## Reference
//!
//! Blount et al. (2008), PNAS 105:7899-7906

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::suboptimal_flops,
    reason = "LTEE population trajectory simulation and ESN reservoir math"
)]

use crate::rng::Rng;

/// Number of input features per generation (fitness, variance, entropy, delta-f).
pub const INPUT_DIM: usize = 4;
/// ESN reservoir dimension.
pub const RESERVOIR_SIZE: usize = 256;
/// Target spectral radius for reservoir weight matrix.
pub const SPECTRAL_RADIUS: f64 = 0.9;
/// Input weight scaling factor.
pub const INPUT_SCALE: f64 = 0.5;
/// Ridge regression regularization strength.
pub const RIDGE_ALPHA: f64 = 0.01;
/// Number of generations per trajectory.
pub const N_GENERATIONS: usize = 100;
/// Generation when Cit+ innovation appeared (Ara-3).
pub const CIT_PLUS_GEN: usize = 63;
/// Generation when potentiating mutations began accumulating.
pub const POTENTIATION_GEN: usize = 43;
/// Label window: generations before Cit+ marked as pre-potentiation.
pub const WINDOW_GENS: usize = 20;

fn exponential(rng: &mut Rng, lambda: f64) -> f64 {
    -rng.uniform().max(1e-15).ln() * lambda
}

fn poisson(rng: &mut Rng, lambda: f64) -> usize {
    let l = (-lambda).exp();
    let mut k = 0usize;
    let mut p = 1.0_f64;
    loop {
        k += 1;
        p *= rng.uniform();
        if p <= l {
            break;
        }
    }
    k.saturating_sub(1)
}

fn dirichlet(rng: &mut Rng, n: usize) -> Vec<f64> {
    let mut samples: Vec<f64> = (0..n)
        .map(|_| {
            let u = rng.uniform().max(1e-15);
            -u.ln()
        })
        .collect();
    let sum: f64 = samples.iter().sum();
    if sum > 0.0 {
        for s in &mut samples {
            *s /= sum;
        }
    }
    samples
}

/// Generate one synthetic LTEE population trajectory.
///
/// Returns `(features, labels)` where features is `N_GENERATIONS × INPUT_DIM`
/// (flattened row-major) and labels is `N_GENERATIONS` binary values.
#[must_use]
pub fn generate_trajectory(rng: &mut Rng, has_potentiation: bool) -> (Vec<f64>, Vec<u8>) {
    let n = N_GENERATIONS;
    let mut fitness = vec![1.0; n];
    let mut variance = vec![0.0; n];
    let mut entropy = vec![0.0; n];
    let mut delta_f = vec![0.0; n];

    let base_rate = 0.001;
    for g in 1..n {
        fitness[g] = fitness[g - 1] + base_rate + rng.normal() * 0.0005;
        variance[g] = 0.001 + exponential(rng, 0.001);
        let n_alleles = (3 + poisson(rng, 1.0)).max(2);
        let freqs = dirichlet(rng, n_alleles);
        entropy[g] = -freqs
            .iter()
            .map(|&f| {
                if f > 1e-10 {
                    f * (f + 1e-10).ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        delta_f[g] = rng.normal() * 0.01;
    }

    if has_potentiation {
        let pot_start = POTENTIATION_GEN;
        let cit_gen = CIT_PLUS_GEN;
        for g in pot_start..cit_gen.min(n) {
            let t = (g - pot_start) as f64 / (cit_gen - pot_start).max(1) as f64;
            variance[g] += 0.005 * t;
            entropy[g] += 0.3 * t;
            delta_f[g] += 0.02 * t + rng.normal() * 0.005;
        }
        if cit_gen < n {
            for g in cit_gen..n {
                fitness[g] += 0.05;
                variance[g] *= 0.5;
            }
        }
    }

    let mut features = Vec::with_capacity(n * INPUT_DIM);
    for g in 0..n {
        features.push(fitness[g]);
        features.push(variance[g]);
        features.push(entropy[g]);
        features.push(delta_f[g]);
    }

    let mut labels = vec![0u8; n];
    if has_potentiation {
        let start = CIT_PLUS_GEN.saturating_sub(WINDOW_GENS);
        let end = CIT_PLUS_GEN.min(n);
        for l in labels.iter_mut().take(end).skip(start) {
            *l = 1;
        }
    }

    (features, labels)
}

/// ESN predictor for citrate early-warning detection.
#[derive(Debug, Clone)]
pub struct CitrateEsnPredictor {
    /// Input-to-reservoir weight matrix (reservoir_size × input_dim, flattened).
    pub w_in: Vec<f64>,
    /// Reservoir recurrent weight matrix (reservoir_size × reservoir_size, flattened).
    pub w_res: Vec<f64>,
    /// Reservoir bias vector.
    pub b_res: Vec<f64>,
    /// Readout weights.
    pub w_out: Vec<f64>,
    /// Reservoir dimension.
    pub reservoir_size: usize,
}

impl CitrateEsnPredictor {
    /// Drive the reservoir on a single input vector (2-step recurrence).
    ///
    /// Returns the reservoir state after the second step.
    #[must_use]
    pub fn reservoir_step(&self, x: &[f64]) -> Vec<f64> {
        let rs = self.reservoir_size;

        let mut h: Vec<f64> = (0..rs)
            .map(|i| {
                let mut sum = self.b_res[i];
                for (j, &xj) in x.iter().enumerate() {
                    sum += self.w_in[i * INPUT_DIM + j] * xj;
                }
                sum.tanh()
            })
            .collect();

        let h_prev = h.clone();
        for (i, h_val) in h.iter_mut().enumerate() {
            let mut sum = self.b_res[i];
            for (j, &xj) in x.iter().enumerate() {
                sum += self.w_in[i * INPUT_DIM + j] * xj;
            }
            for (k, &hk) in h_prev.iter().enumerate() {
                sum += self.w_res[i * rs + k] * hk;
            }
            *h_val = sum.tanh();
        }

        h
    }

    /// Drive the reservoir over an entire generation sequence.
    ///
    /// `features` is `n_gens × INPUT_DIM` flattened row-major.
    /// Returns `n_gens × reservoir_size` flattened row-major.
    #[must_use]
    pub fn reservoir_drive(&self, features: &[f64], n_gens: usize) -> Vec<f64> {
        let rs = self.reservoir_size;
        let mut states = vec![0.0; n_gens * rs];
        for g in 0..n_gens {
            let x = &features[g * INPUT_DIM..(g + 1) * INPUT_DIM];
            let h = self.reservoir_step(x);
            states[g * rs..(g + 1) * rs].copy_from_slice(&h);
        }
        states
    }

    /// Classify each generation as normal (0) or pre-potentiation (1).
    ///
    /// Returns `(predictions, scores)` for each generation.
    #[must_use]
    pub fn classify(&self, states: &[f64], n_gens: usize, threshold: f64) -> (Vec<u8>, Vec<f64>) {
        let rs = self.reservoir_size;
        let mut predictions = Vec::with_capacity(n_gens);
        let mut scores = Vec::with_capacity(n_gens);
        for g in 0..n_gens {
            let h = &states[g * rs..(g + 1) * rs];
            let score: f64 = h.iter().zip(&self.w_out).map(|(&hi, &wi)| hi * wi).sum();
            scores.push(score);
            predictions.push(u8::from(score > threshold));
        }
        (predictions, scores)
    }
}

/// Early-warning detection metrics.
#[derive(Debug, Clone)]
pub struct EarlyWarningMetrics {
    /// Overall classification accuracy.
    pub accuracy: f64,
    /// True positive rate (sensitivity).
    pub tpr: f64,
    /// False positive rate.
    pub fpr: f64,
    /// Precision.
    pub precision: f64,
    /// Counts: (TP, FP, TN, FN).
    pub confusion: (usize, usize, usize, usize),
}

/// Compute early-warning detection metrics.
#[must_use]
pub fn early_warning_metrics(predictions: &[u8], labels: &[u8]) -> EarlyWarningMetrics {
    let n = predictions.len();
    let (mut tp, mut fp, mut tn, mut fn_) = (0usize, 0usize, 0usize, 0usize);
    for i in 0..n {
        match (predictions[i], labels[i]) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 0) => tn += 1,
            (0, 1) => fn_ += 1,
            _ => {}
        }
    }
    let n_pos = (tp + fn_).max(1);
    let n_neg = (tn + fp).max(1);
    EarlyWarningMetrics {
        accuracy: (tp + tn) as f64 / n.max(1) as f64,
        tpr: tp as f64 / n_pos as f64,
        fpr: fp as f64 / n_neg as f64,
        precision: tp as f64 / (tp + fp).max(1) as f64,
        confusion: (tp, fp, tn, fn_),
    }
}

/// Load ESN predictor from Python baseline JSON.
///
/// # Errors
///
/// Returns `Err` if JSON structure is unexpected.
pub fn load_citrate_esn_from_json(json_str: &str) -> Result<CitrateEsnBaseline, String> {
    let v: serde_json::Value = serde_json::from_str(json_str).map_err(|e| format!("parse: {e}"))?;

    let w_in = parse_f64_array(&v, "w_in")?;
    let w_res = parse_f64_array(&v, "w_res")?;
    let b_res = parse_f64_array(&v, "b_res")?;
    let w_out = parse_f64_array(&v, "w_out")?;

    let reservoir_size = v["reservoir_size"]
        .as_u64()
        .ok_or("reservoir_size missing")? as usize;

    let predictor = CitrateEsnPredictor {
        w_in,
        w_res,
        b_res,
        w_out,
        reservoir_size,
    };

    let train = &v["train_metrics"];
    let test = &v["test_metrics"];

    Ok(CitrateEsnBaseline {
        predictor,
        seed: v["seed"].as_u64().ok_or("seed")? as u64,
        n_trajectories: v["n_trajectories"].as_u64().ok_or("n_traj")? as usize,
        n_generations: v["n_generations"].as_u64().ok_or("n_gen")? as usize,
        cit_plus_gen: v["cit_plus_gen"].as_u64().ok_or("cit_plus_gen")? as usize,
        potentiation_gen: v["potentiation_gen"].as_u64().ok_or("pot_gen")? as usize,
        window_gens: v["window_gens"].as_u64().ok_or("window_gens")? as usize,
        train_accuracy: train["accuracy"].as_f64().ok_or("train acc")?,
        train_tpr: train["tpr"].as_f64().ok_or("train tpr")?,
        test_accuracy: test["accuracy"].as_f64().ok_or("test acc")?,
        test_tpr: test["tpr"].as_f64().ok_or("test tpr")?,
        first_trajectory_labels: parse_u8_array(&v["first_trajectory"], "labels")?,
        first_trajectory_predictions: parse_u8_array(&v["first_trajectory"], "predictions")?,
        first_trajectory_scores: parse_f64_array(&v["first_trajectory"], "scores")?,
    })
}

/// Loaded baseline with predictor and expected values.
#[derive(Debug, Clone)]
pub struct CitrateEsnBaseline {
    /// Trained ESN predictor.
    pub predictor: CitrateEsnPredictor,
    /// Random seed.
    pub seed: u64,
    /// Number of trajectories in dataset.
    pub n_trajectories: usize,
    /// Generations per trajectory.
    pub n_generations: usize,
    /// Generation when Cit+ appeared.
    pub cit_plus_gen: usize,
    /// Generation when potentiation started.
    pub potentiation_gen: usize,
    /// Window size for early-warning labels.
    pub window_gens: usize,
    /// Python train accuracy.
    pub train_accuracy: f64,
    /// Python train TPR.
    pub train_tpr: f64,
    /// Python test accuracy.
    pub test_accuracy: f64,
    /// Python test TPR.
    pub test_tpr: f64,
    /// First trajectory labels (from Python).
    pub first_trajectory_labels: Vec<u8>,
    /// First trajectory predictions (from Python).
    pub first_trajectory_predictions: Vec<u8>,
    /// First trajectory scores (from Python).
    pub first_trajectory_scores: Vec<f64>,
}

fn parse_f64_array(v: &serde_json::Value, key: &str) -> Result<Vec<f64>, String> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("{key} not array"))?
        .iter()
        .map(|x| x.as_f64().ok_or_else(|| format!("{key} element not f64")))
        .collect()
}

fn parse_u8_array(v: &serde_json::Value, key: &str) -> Result<Vec<u8>, String> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("{key} not array"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .ok_or_else(|| format!("{key} element not u64"))
                .map(|n| n as u8)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_generation_deterministic() {
        let mut rng = Rng::new(42);
        let (features, labels) = generate_trajectory(&mut rng, true);
        assert_eq!(features.len(), N_GENERATIONS * INPUT_DIM);
        assert_eq!(labels.len(), N_GENERATIONS);

        let pos_count: usize = labels.iter().map(|&l| l as usize).sum();
        assert_eq!(pos_count, WINDOW_GENS);
    }

    #[test]
    fn normal_trajectory_has_no_labels() {
        let mut rng = Rng::new(42);
        let (_, labels) = generate_trajectory(&mut rng, false);
        let pos_count: usize = labels.iter().map(|&l| l as usize).sum();
        assert_eq!(pos_count, 0);
    }

    #[test]
    fn reservoir_step_produces_correct_dim() {
        let mut rng = Rng::new(99);
        let rs = 16;
        let predictor = CitrateEsnPredictor {
            w_in: (0..rs * INPUT_DIM).map(|_| rng.normal() * 0.1).collect(),
            w_res: (0..rs * rs).map(|_| rng.normal() * 0.01).collect(),
            b_res: (0..rs).map(|_| rng.normal() * 0.01).collect(),
            w_out: (0..rs).map(|_| rng.normal() * 0.01).collect(),
            reservoir_size: rs,
        };
        let x = [1.0, 0.001, 0.5, 0.0];
        let h = predictor.reservoir_step(&x);
        assert_eq!(h.len(), rs);
        for &val in &h {
            assert!(val.abs() <= 1.0, "tanh output must be in [-1,1]");
        }
    }

    #[test]
    fn metrics_computation() {
        let preds = [1, 1, 0, 0, 1, 0];
        let truth = [1, 0, 0, 1, 1, 0];
        let m = early_warning_metrics(
            &preds.map(|x| x as u8),
            &truth.map(|x| x as u8),
        );
        assert_eq!(m.confusion, (2, 1, 2, 1));
        assert!((m.accuracy - 4.0 / 6.0).abs() < 1e-10);
        assert!((m.tpr - 2.0 / 3.0).abs() < 1e-10);
        assert!((m.fpr - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn load_baseline_from_json() {
        let json_str = include_str!("../control/ltee_citrate_esn/expected_values.json");
        let baseline = load_citrate_esn_from_json(json_str).expect("parse baseline");
        assert_eq!(baseline.predictor.reservoir_size, RESERVOIR_SIZE);
        assert_eq!(baseline.predictor.w_in.len(), RESERVOIR_SIZE * INPUT_DIM);
        assert_eq!(
            baseline.predictor.w_res.len(),
            RESERVOIR_SIZE * RESERVOIR_SIZE
        );
        assert_eq!(baseline.predictor.w_out.len(), RESERVOIR_SIZE);
        assert!(baseline.test_accuracy > 0.80);
    }

    #[test]
    fn rust_predictions_match_python() {
        let json_str = include_str!("../control/ltee_citrate_esn/expected_values.json");
        let baseline = load_citrate_esn_from_json(json_str).expect("parse baseline");

        let v: serde_json::Value =
            serde_json::from_str(json_str).expect("parse raw json");
        let features: Vec<f64> = v["first_trajectory"]["features"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
            })
            .collect();

        let n_gens = baseline.n_generations;
        let states = baseline.predictor.reservoir_drive(&features, n_gens);
        let (preds, scores) = baseline.predictor.classify(&states, n_gens, 0.5);

        for (i, (&rust_score, &py_score)) in
            scores.iter().zip(&baseline.first_trajectory_scores).enumerate()
        {
            let diff = (rust_score - py_score).abs();
            assert!(
                diff < 1e-6,
                "score mismatch at gen {i}: rust={rust_score}, py={py_score}, diff={diff}"
            );
        }

        assert_eq!(
            preds, baseline.first_trajectory_predictions,
            "prediction mismatch vs Python baseline"
        );
    }
}
