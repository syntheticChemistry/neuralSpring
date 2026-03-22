// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paper 026: LSTM blood glucose prediction with horizon limit analysis.
//!
//! Port of Chuna (2020) "Setting Limits on Neural Network's Predictive
//! Capacity in T1D Blood Glucose Concentration" (medRxiv 2020.08.04.20117812).
//!
//! Validates that LSTM prediction accuracy degrades with forecast horizon,
//! with autocorrelation decay τ ≈ 1.5–3 hrs setting the fundamental limit.
//! Same LSTM primitives as Exp 003 (weather), Exp 009 (ERA5), nW-03 (S(q,ω)).
//!
//! ## Python Baseline Provenance
//!
//! | Metric | Value | Source |
//! |--------|-------|--------|
//! | R²(5min) | 0.9629 | `control/glucose_prediction/glucose_prediction.py`, seed=42 |
//! | R²(30min) | 0.7790 | same |
//! | R²(60min) | 0.4698 | same |
//! | R²(120min) | 0.2159 | same |
//! | R²(240min) | 0.1641 | same |
//! | τ (autocorrelation) | 1.5 hrs | same |
//!
//! ## Architecture
//!
//! LSTM(input_size=1, hidden=24) reservoir → pooled features
//! [mean, std, last] → per-horizon linear readout → glucose (mg/dL).
//!
//! ## Reference
//!
//! Chuna (2020), medRxiv 2020.08.04.20117812

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::too_many_lines,
    reason = "domain-specific numeric patterns and CGM simulation require index↔float casts and long experiment orchestration"
)]

use std::f64::consts::PI;

use crate::rng::Rng;
use crate::sequence::{LstmWeights, lstm_cell};
use crate::tolerances;

const WASHOUT: usize = 4;
const BASAL_GLUCOSE: f64 = 120.0;
const NOISE_STD: f64 = 8.0;
const ACOR_DECAY_STEPS: usize = 36;
const INSULIN_DECAY_RATE: f64 = 0.02;
const DT_MINUTES: f64 = 5.0;
const SAMPLES_PER_DAY: usize = 288;

/// Ridge regression regularization strength for LSTM readout.
///
/// Small enough to not bias predictions, large enough to stabilize
/// the Cholesky solve for the ill-conditioned feature matrices
/// produced by LSTM hidden-state pooling.
const RIDGE_ALPHA: f64 = 1e-3;

/// Per-horizon prediction results.
#[derive(Debug, Clone)]
pub struct HorizonResult {
    /// Forecast horizon in CGM samples (steps ahead).
    pub horizon_steps: usize,
    /// Forecast horizon in minutes (steps × sample interval).
    pub horizon_minutes: usize,
    /// Coefficient of determination for the LSTM predictor.
    pub r2_lstm: f64,
    /// Root mean squared error for the LSTM predictor (mg/dL).
    pub rmse_lstm: f64,
    /// R² for the naive persistence baseline (last value).
    pub r2_persistence: f64,
    /// RMSE for the persistence baseline (mg/dL).
    pub rmse_persistence: f64,
    /// Percent improvement of LSTM RMSE over persistence.
    pub lstm_improvement_pct: f64,
}

/// Trained glucose predictor for a single horizon.
#[derive(Debug, Clone)]
pub struct GlucoseReadout {
    /// Linear readout weight vector on pooled LSTM features.
    pub w_out: Vec<f64>,
    /// Linear readout bias term.
    pub b_out: f64,
}

/// Complete glucose prediction model (multi-horizon).
#[derive(Debug, Clone)]
pub struct GlucosePredictor {
    /// LSTM input weight matrix (flattened).
    pub w_i: Vec<f64>,
    /// LSTM hidden recurrent weight matrix (flattened).
    pub w_h: Vec<f64>,
    /// LSTM input bias vector.
    pub b_i: Vec<f64>,
    /// LSTM hidden bias vector.
    pub b_h: Vec<f64>,
    /// LSTM hidden state width.
    pub hidden_size: usize,
    /// Input window length (past CGM samples).
    pub seq_len: usize,
    /// CGM training mean used for normalization (mg/dL).
    pub cgm_mean: f64,
    /// CGM training standard deviation used for normalization.
    pub cgm_std: f64,
    /// Per-horizon linear readouts `(horizon_steps, readout)`.
    pub readouts: Vec<(usize, GlucoseReadout)>,
}

/// Generate synthetic CGM trace capturing T1D statistical structure.
///
/// Models basal glucose, circadian variation (dawn phenomenon), three
/// daily meals with postprandial spikes and insulin decay, plus
/// Ornstein-Uhlenbeck autocorrelated noise with τ ≈ 3 hrs.
#[must_use]
pub fn generate_synthetic_cgm(n_days: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let n = n_days * SAMPLES_PER_DAY;
    let mut glucose = vec![0.0_f64; n];

    for (t, g) in glucose.iter_mut().enumerate() {
        let hours = (t % SAMPLES_PER_DAY) as f64 * DT_MINUTES / 60.0;
        let dawn = 8.0 * (-0.5 * ((hours - 5.0) / 1.5).powi(2)).exp();
        let circadian = 3.0f64.mul_add((2.0 * PI * hours / 24.0).sin(), dawn);
        *g = BASAL_GLUCOSE + circadian;
    }

    let meal_times_hr = [7.0_f64, 12.0, 18.0];
    let meal_sizes = [50.0_f64, 65.0, 55.0];

    for day in 0..n_days {
        for (&mt, &ms) in meal_times_hr.iter().zip(meal_sizes.iter()) {
            let jitter_hr = rng.normal_params(0.0, 0.3);
            let jitter_size = rng.normal_params(0.0, 8.0);
            let meal_step = day * SAMPLES_PER_DAY + ((mt + jitter_hr) * 60.0 / DT_MINUTES) as usize;
            if meal_step < n {
                let amp = ms + jitter_size;
                for k in 0..48.min(n - meal_step) {
                    let decay = (-INSULIN_DECAY_RATE * k as f64).exp();
                    let rise = 1.0 - (-0.15 * k as f64).exp();
                    glucose[meal_step + k] += amp * rise * decay;
                }
            }
        }
    }

    let alpha = (-1.0 / ACOR_DECAY_STEPS as f64).exp();
    let sigma_scale = (1.0 - alpha * alpha).sqrt();
    let mut noise_prev = rng.normal_params(0.0, NOISE_STD);
    glucose[0] += noise_prev;

    for g in glucose.iter_mut().skip(1) {
        let noise = alpha.mul_add(noise_prev, sigma_scale * rng.normal_params(0.0, NOISE_STD));
        noise_prev = noise;
        *g += noise;
    }

    for g in &mut glucose {
        *g = g.clamp(40.0, 400.0);
    }

    glucose
}

/// Create (input_window, target) pairs for forecasting.
#[must_use]
pub fn create_sequences(data: &[f64], seq_len: usize, horizon: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = data.len();
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for i in seq_len..=(n.saturating_sub(horizon)) {
        if i + horizon - 1 < n {
            inputs.push(data[i - seq_len..i].to_vec());
            targets.push(data[i + horizon - 1]);
        }
    }
    (inputs, targets)
}

/// Compute normalized autocorrelation up to `max_lag` steps.
#[must_use]
pub fn autocorrelation(series: &[f64], max_lag: usize) -> Vec<f64> {
    let n = series.len();
    let mean = series.iter().sum::<f64>() / n as f64;
    let var = series.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / n as f64;
    let mut acor = Vec::with_capacity(max_lag);
    for lag in 0..max_lag {
        let cov = series[..n - lag]
            .iter()
            .zip(series[lag..].iter())
            .map(|(&a, &b)| (a - mean) * (b - mean))
            .sum::<f64>()
            / n as f64;
        acor.push(cov / var.max(tolerances::LOG_ZERO_GUARD));
    }
    acor
}

/// Estimate autocorrelation decay time τ (in steps).
#[must_use]
pub fn estimate_tau(acor: &[f64]) -> usize {
    let threshold = 1.0 / std::f64::consts::E;
    acor.iter()
        .position(|&a| a < threshold)
        .unwrap_or(ACOR_DECAY_STEPS)
}

/// Extract LSTM hidden state features [mean, std, last] from a window.
fn extract_features(window: &[f64], lstm_w: &LstmWeights<'_>) -> Vec<f64> {
    let hs = lstm_w.hidden_size;
    let mut h = vec![0.0; hs];
    let mut c = vec![0.0; hs];
    let mut all_h = Vec::with_capacity(window.len());

    for val in window {
        let (h_new, c_new) = lstm_cell(&[*val], &h, &c, lstm_w);
        h = h_new;
        c = c_new;
        all_h.push(h.clone());
    }

    let valid_h = &all_h[WASHOUT.min(all_h.len())..];
    let n_valid = valid_h.len() as f64;

    let mut h_mean = vec![0.0; hs];
    for state in valid_h {
        for (m, s) in h_mean.iter_mut().zip(state.iter()) {
            *m += s;
        }
    }
    if n_valid > 0.0 {
        for m in &mut h_mean {
            *m /= n_valid;
        }
    }

    let mut h_std = vec![0.0; hs];
    for state in valid_h {
        for (j, s) in state.iter().enumerate() {
            h_std[j] += (s - h_mean[j]).powi(2);
        }
    }
    if n_valid > 0.0 {
        for s in &mut h_std {
            *s = (*s / n_valid).sqrt();
        }
    }

    let h_last = all_h.last().cloned().unwrap_or_else(|| vec![0.0; hs]);

    let mut features = Vec::with_capacity(3 * hs);
    features.extend_from_slice(&h_mean);
    features.extend_from_slice(&h_std);
    features.extend_from_slice(&h_last);
    features
}

/// R² score between actual and predicted values.
///
/// Delegates to [`crate::metrics::r_squared`] → `barracuda::stats::r_squared`.
#[must_use]
pub fn r2_score(actual: &[f64], predicted: &[f64]) -> f64 {
    crate::metrics::r_squared(actual, predicted)
}

/// RMSE between actual and predicted values.
///
/// Delegates to [`crate::metrics::rmse`] → `barracuda::stats::rmse`.
#[must_use]
pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    crate::metrics::rmse(actual, predicted)
}

/// Run the full glucose prediction experiment at multiple horizons.
///
/// Returns per-horizon results and the trained predictor model.
#[must_use]
pub fn run_glucose_experiment(
    n_days: usize,
    hidden_size: usize,
    seq_len: usize,
    horizons: &[usize],
    seed: u64,
) -> (Vec<HorizonResult>, GlucosePredictor) {
    let glucose = generate_synthetic_cgm(n_days, seed);

    let g_mean = glucose.iter().sum::<f64>() / glucose.len() as f64;
    let g_var = glucose.iter().map(|&g| (g - g_mean).powi(2)).sum::<f64>() / glucose.len() as f64;
    let g_std = g_var.sqrt().max(tolerances::VARIANCE_DIVISION_GUARD);

    let glucose_norm: Vec<f64> = glucose.iter().map(|&g| (g - g_mean) / g_std).collect();

    let mut rng_w = Rng::new(seed);
    let hs = hidden_size;

    let input_scale = 0.5;
    let spectral_radius = 0.9;
    let forget_bias = 1.0;
    let ridge_alpha = RIDGE_ALPHA;
    let test_fraction = 0.2;

    let w_i: Vec<f64> = (0..4 * hs).map(|_| rng_w.normal() * input_scale).collect();

    let mut w_h: Vec<f64> = (0..4 * hs * hs).map(|_| rng_w.normal() * 0.1).collect();

    let rho_max = spectral_radius_estimate(&w_h[..hs * hs], hs);
    if rho_max > tolerances::RELATIVE_ERROR_FLOOR {
        let scale = spectral_radius / rho_max;
        for w in &mut w_h {
            *w *= scale;
        }
    }

    let mut b_i = vec![0.0; 4 * hs];
    for b in b_i.iter_mut().take(hs) {
        *b = forget_bias;
    }
    let b_h = vec![0.0; 4 * hs];

    let lstm_w = LstmWeights {
        w_input: &w_i,
        w_hidden: &w_h,
        b_input: &b_i,
        b_hidden: &b_h,
        hidden_size: hs,
    };

    let mut results = Vec::with_capacity(horizons.len());
    let mut readouts = Vec::with_capacity(horizons.len());

    for &horizon in horizons {
        let (inputs, targets) = create_sequences(&glucose_norm, seq_len, horizon);

        let n = inputs.len();
        let n_test = (n as f64 * test_fraction).max(1.0) as usize;

        let mut rng_split = Rng::new(seed + horizon as u64);
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng_split.next_u64() as usize % (i + 1);
            perm.swap(i, j);
        }

        let test_idx = &perm[..n_test];
        let train_idx = &perm[n_test..];

        let feat_dim = 3 * hs;
        let n_train = train_idx.len();
        let mut h_train = vec![0.0; n_train * feat_dim];
        let mut y_train = Vec::with_capacity(n_train);

        for (row, &idx) in train_idx.iter().enumerate() {
            let features = extract_features(&inputs[idx], &lstm_w);
            h_train[row * feat_dim..(row + 1) * feat_dim].copy_from_slice(&features);
            y_train.push(targets[idx]);
        }

        let augmented_dim = feat_dim + 1;
        let mut hth = vec![0.0; augmented_dim * augmented_dim];
        let mut hty = vec![0.0; augmented_dim];

        for (row, &y_val) in y_train.iter().enumerate() {
            let feat_start = row * feat_dim;
            for i in 0..feat_dim {
                let fi = h_train[feat_start + i];
                for j in i..feat_dim {
                    let fj = h_train[feat_start + j];
                    hth[i * augmented_dim + j] = fi.mul_add(fj, hth[i * augmented_dim + j]);
                    if i != j {
                        hth[j * augmented_dim + i] = fi.mul_add(fj, hth[j * augmented_dim + i]);
                    }
                }
                hth[i * augmented_dim + feat_dim] += fi;
                hth[feat_dim * augmented_dim + i] += fi;
                hty[i] = fi.mul_add(y_val, hty[i]);
            }
            hth[feat_dim * augmented_dim + feat_dim] += 1.0;
            hty[feat_dim] += y_val;
        }

        for i in 0..feat_dim {
            hth[i * augmented_dim + i] += ridge_alpha;
        }

        let w_out_aug = solve_symmetric(&hth, &hty, augmented_dim);

        let w_out = w_out_aug[..feat_dim].to_vec();
        let b_out = w_out_aug[feat_dim];

        let mut pred_test = Vec::with_capacity(n_test);
        let mut actual_test = Vec::with_capacity(n_test);
        let mut persist_pred = Vec::with_capacity(n_test);

        for &idx in test_idx {
            let features = extract_features(&inputs[idx], &lstm_w);
            let pred_norm: f64 = features
                .iter()
                .zip(w_out.iter())
                .fold(b_out, |acc, (&f, &w)| f.mul_add(w, acc));
            pred_test.push(pred_norm.mul_add(g_std, g_mean));
            actual_test.push(targets[idx].mul_add(g_std, g_mean));
            persist_pred.push(
                inputs[idx]
                    .last()
                    .copied()
                    .unwrap_or(0.0)
                    .mul_add(g_std, g_mean),
            );
        }

        let r2_lstm = r2_score(&actual_test, &pred_test);
        let rmse_lstm = rmse(&actual_test, &pred_test);
        let r2_persist = r2_score(&actual_test, &persist_pred);
        let rmse_persist = rmse(&actual_test, &persist_pred);

        let improvement =
            (rmse_persist - rmse_lstm) / rmse_persist.max(tolerances::RELATIVE_ERROR_FLOOR) * 100.0;

        results.push(HorizonResult {
            horizon_steps: horizon,
            horizon_minutes: horizon * DT_MINUTES as usize,
            r2_lstm,
            rmse_lstm,
            r2_persistence: r2_persist,
            rmse_persistence: rmse_persist,
            lstm_improvement_pct: improvement,
        });

        readouts.push((horizon, GlucoseReadout { w_out, b_out }));
    }

    let predictor = GlucosePredictor {
        w_i,
        w_h,
        b_i,
        b_h,
        hidden_size: hs,
        seq_len,
        cgm_mean: g_mean,
        cgm_std: g_std,
        readouts,
    };

    (results, predictor)
}

/// Estimate the spectral radius of a square matrix via power iteration.
fn spectral_radius_estimate(matrix: &[f64], n: usize) -> f64 {
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    for _ in 0..50 {
        let mut w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] = matrix[i * n + j].mul_add(v[j], w[i]);
            }
        }
        let norm: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < tolerances::LOG_ZERO_GUARD {
            return 0.0;
        }
        for x in &mut w {
            *x /= norm;
        }
        v = w;
    }

    let mut mv = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            mv[i] = matrix[i * n + j].mul_add(v[j], mv[i]);
        }
    }
    mv.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Solve Ax = b for symmetric positive-definite A.
///
/// Delegates to `barracuda::linalg::solve::solve_f64_cpu` (Gaussian
/// elimination with partial pivoting).  Falls back to a
/// ridge-regularized system if the matrix is near-singular (degenerate
/// reservoir states produce rank-deficient gram matrices).
fn solve_symmetric(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    barracuda::linalg::solve::solve_f64_cpu(a, b, n).unwrap_or_else(|_| {
        let mut regularized = a.to_vec();
        for i in 0..n {
            regularized[i * n + i] += tolerances::RELATIVE_ERROR_FLOOR;
        }
        barracuda::linalg::solve::solve_f64_cpu(&regularized, b, n).unwrap_or_else(|_| vec![0.0; n])
    })
}

/// Load a [`GlucosePredictor`] from the Python baseline JSON.
///
/// # Errors
///
/// Returns `Err` if the JSON structure is unexpected.
pub fn load_glucose_from_json(json_str: &str) -> Result<GlucosePredictor, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

    let cgm = parsed.get("cgm_stats").ok_or("Missing 'cgm_stats'")?;
    let cgm_mean = cgm
        .get("mean")
        .and_then(serde_json::Value::as_f64)
        .ok_or("Missing cgm mean")?;
    let cgm_std = cgm
        .get("std")
        .and_then(serde_json::Value::as_f64)
        .ok_or("Missing cgm std")?;

    let w = parsed.get("weights").ok_or("Missing 'weights'")?;
    let hs = usize::try_from(
        w.get("hidden_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or("Missing hidden_size")?,
    )
    .map_err(|e| format!("hidden_size: {e}"))?;

    let cfg = parsed.get("lstm_config").ok_or("Missing 'lstm_config'")?;
    let seq_len = usize::try_from(
        cfg.get("seq_len")
            .and_then(serde_json::Value::as_u64)
            .ok_or("Missing seq_len")?,
    )
    .map_err(|e| format!("seq_len: {e}"))?;

    let w_i = parse_f64_vec(w, "W_i")?;
    let w_h = parse_f64_vec(w, "W_h")?;
    let b_i = parse_f64_vec(w, "b_i")?;
    let b_h = parse_f64_vec(w, "b_h")?;

    let horizons = parsed
        .get("horizons")
        .and_then(serde_json::Value::as_array)
        .ok_or("Missing 'horizons'")?;

    let mut readouts = Vec::with_capacity(horizons.len());
    for h in horizons {
        let steps = usize::try_from(
            h.get("horizon_steps")
                .and_then(serde_json::Value::as_u64)
                .ok_or("Missing horizon_steps")?,
        )
        .map_err(|e| format!("horizon_steps: {e}"))?;
        let w_out = parse_f64_vec(h, "W_out")?;
        let b_out = h
            .get("b_out")
            .and_then(serde_json::Value::as_f64)
            .ok_or("Missing b_out")?;
        readouts.push((steps, GlucoseReadout { w_out, b_out }));
    }

    Ok(GlucosePredictor {
        w_i,
        w_h,
        b_i,
        b_h,
        hidden_size: hs,
        seq_len,
        cgm_mean,
        cgm_std,
        readouts,
    })
}

fn parse_f64_vec(parent: &serde_json::Value, key: &str) -> Result<Vec<f64>, String> {
    parent
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("Missing '{key}'"))?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| format!("Non-numeric in '{key}'")))
        .collect()
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn synthetic_cgm_range() {
        let cgm = generate_synthetic_cgm(14, 42);
        assert_eq!(cgm.len(), 14 * 288);
        assert!(cgm.iter().all(|&g| (40.0..=400.0).contains(&g)));
    }

    #[test]
    fn synthetic_cgm_statistics() {
        let cgm = generate_synthetic_cgm(14, 42);
        let mean = cgm.iter().sum::<f64>() / cgm.len() as f64;
        assert!(
            mean > 100.0 && mean < 200.0,
            "mean glucose should be physiological"
        );
        let std = (cgm.iter().map(|&g| (g - mean).powi(2)).sum::<f64>() / cgm.len() as f64).sqrt();
        assert!(std > 5.0 && std < 50.0, "std should be reasonable");
    }

    #[test]
    fn synthetic_cgm_deterministic() {
        let cgm1 = generate_synthetic_cgm(7, 42);
        let cgm2 = generate_synthetic_cgm(7, 42);
        assert_eq!(cgm1, cgm2, "CGM generation must be deterministic");
    }

    #[test]
    fn autocorrelation_unit_lag() {
        let cgm = generate_synthetic_cgm(14, 42);
        let acor = autocorrelation(&cgm, 144);
        assert!((acor[0] - 1.0).abs() < tolerances::ZERO_DETECTION);
        assert!(
            acor[1] > 0.9,
            "adjacent CGM samples should be highly correlated"
        );
        assert!(acor[72] < acor[0], "correlation should decay over 6 hours");
    }

    #[test]
    fn tau_estimate_reasonable() {
        let cgm = generate_synthetic_cgm(14, 42);
        let acor = autocorrelation(&cgm, 144);
        let tau = estimate_tau(&acor);
        let tau_hours = tau as f64 * DT_MINUTES / 60.0;
        assert!(
            (1.0..=6.0).contains(&tau_hours),
            "τ should be 1-6 hours, got {tau_hours}"
        );
    }

    #[test]
    fn create_sequences_lengths() {
        let data: Vec<f64> = (0..100).map(f64::from).collect();
        let (inputs, targets) = create_sequences(&data, 12, 6);
        assert_eq!(inputs.len(), targets.len());
        assert!(!inputs.is_empty());
        assert_eq!(inputs[0].len(), 12);
    }

    #[test]
    fn r2_score_perfect() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r2 = r2_score(&actual, &actual);
        assert!((r2 - 1.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn rmse_zero_for_identical() {
        let actual = vec![1.0, 2.0, 3.0];
        let r = rmse(&actual, &actual);
        assert!(r.abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn run_experiment_produces_results() {
        let horizons = [1, 6];
        let (results, predictor) = run_glucose_experiment(7, 8, 6, &horizons, 42);
        assert_eq!(results.len(), 2);
        assert!(results[0].r2_lstm.is_finite());
        assert!(results[1].r2_lstm.is_finite());
        assert_eq!(predictor.hidden_size, 8);
        assert_eq!(predictor.readouts.len(), 2);
    }

    #[test]
    fn short_horizon_better_than_long() {
        let horizons = [1, 12];
        let (results, _) = run_glucose_experiment(14, 16, 12, &horizons, 42);
        assert!(
            results[0].r2_lstm > results[1].r2_lstm,
            "R²(1-step) should exceed R²(12-step)"
        );
    }

    #[test]
    fn experiment_deterministic() {
        let horizons = [1, 6];
        let (r1, _) = run_glucose_experiment(7, 8, 6, &horizons, 42);
        let (r2, _) = run_glucose_experiment(7, 8, 6, &horizons, 42);
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert!(
                (a.r2_lstm - b.r2_lstm).abs() < tolerances::ZERO_DETECTION,
                "experiment must be deterministic"
            );
        }
    }

    #[test]
    fn extract_features_dimension() {
        let cgm = generate_synthetic_cgm(7, 42);
        let g_mean = cgm.iter().sum::<f64>() / cgm.len() as f64;
        let g_var = cgm.iter().map(|&g| (g - g_mean).powi(2)).sum::<f64>() / cgm.len() as f64;
        let g_std = g_var.sqrt().max(tolerances::VARIANCE_DIVISION_GUARD);
        let norm: Vec<f64> = cgm.iter().map(|&g| (g - g_mean) / g_std).collect();
        let window = &norm[..24];

        let hs = 8;
        let mut rng = crate::rng::Rng::new(42);
        let w_input: Vec<f64> = (0..4 * hs).map(|_| rng.normal() * 0.5).collect();
        let w_hidden: Vec<f64> = (0..4 * hs * hs).map(|_| rng.normal() * 0.1).collect();
        let mut b_input = vec![0.0; 4 * hs];
        let b_hidden = vec![0.0; 4 * hs];
        for b in &mut b_input[hs..2 * hs] {
            *b = 1.0;
        }
        let lstm_w = crate::sequence::LstmWeights {
            w_input: &w_input,
            w_hidden: &w_hidden,
            b_input: &b_input,
            b_hidden: &b_hidden,
            hidden_size: hs,
        };

        let features = extract_features(window, &lstm_w);
        assert_eq!(
            features.len(),
            3 * hs,
            "features should be [mean, std, last] × hidden_size"
        );
        assert!(
            features.iter().all(|v| v.is_finite()),
            "all features must be finite"
        );
    }

    #[test]
    fn solve_symmetric_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 7.0];
        let x = solve_symmetric(&a, &b, 2);
        assert!(
            (x[0] - 3.0).abs() < tolerances::CROSS_LANGUAGE,
            "I·x = b → x = b"
        );
        assert!((x[1] - 7.0).abs() < tolerances::CROSS_LANGUAGE);
    }

    #[test]
    fn solve_symmetric_known_system() {
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let b = vec![8.0, 7.0];
        let x = solve_symmetric(&a, &b, 2);
        let residual_0 = (4.0f64.mul_add(x[0], 2.0 * x[1]) - 8.0).abs();
        let residual_1 = (2.0f64.mul_add(x[0], 3.0 * x[1]) - 7.0).abs();
        assert!(
            residual_0 < tolerances::ODE_ATOL,
            "residual[0] = {residual_0}"
        );
        assert!(
            residual_1 < tolerances::ODE_ATOL,
            "residual[1] = {residual_1}"
        );
    }

    #[test]
    fn load_glucose_from_json_valid() {
        let json = r#"{
            "cgm_stats": {"mean": 120.0, "std": 15.0},
            "weights": {
                "hidden_size": 2,
                "W_i": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "W_h": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                         0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
                "b_i": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "b_h": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "lstm_config": {"seq_len": 12},
            "horizons": [
                {"horizon_steps": 1, "W_out": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "b_out": 0.0}
            ]
        }"#;
        let predictor = load_glucose_from_json(json).expect("valid JSON should parse");
        assert_eq!(predictor.hidden_size, 2);
        assert_eq!(predictor.seq_len, 12);
        assert!((predictor.cgm_mean - 120.0).abs() < tolerances::CROSS_LANGUAGE);
        assert_eq!(predictor.readouts.len(), 1);
        assert_eq!(predictor.readouts[0].0, 1);
    }

    #[test]
    fn load_glucose_from_json_missing_field() {
        let json = r#"{"cgm_stats": {"mean": 120.0}}"#;
        assert!(load_glucose_from_json(json).is_err());
    }

    #[test]
    fn load_glucose_from_json_invalid_json() {
        assert!(load_glucose_from_json("not json").is_err());
    }
}
