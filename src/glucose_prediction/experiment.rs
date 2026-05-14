// SPDX-License-Identifier: AGPL-3.0-or-later

//! LSTM reservoir forward pass, ridge readouts, spectral scaling, and the full multi-horizon experiment.
//!
//! Also loads [`super::GlucosePredictor`] weights from the Python baseline JSON for parity checks.

#![expect(
    clippy::cast_precision_loss,
    reason = "domain-specific numeric patterns for Paper 026"
)]

#[cfg(feature = "barracuda")]
use crate::sequence::{LstmWeights, lstm_cell};
use crate::tolerances;

#[cfg(feature = "barracuda")]
use super::cgm::{DT_MINUTES, WASHOUT, create_sequences, generate_synthetic_cgm};
#[cfg(feature = "barracuda")]
use super::analysis::{r2_score, rmse};
#[cfg(feature = "barracuda")]
use super::{HorizonResult, RIDGE_ALPHA};
use super::{GlucosePredictor, GlucoseReadout};

/// Run the full glucose prediction experiment at multiple horizons.
///
/// Returns per-horizon results and the trained predictor model.
#[cfg(feature = "barracuda")]
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

    let mut rng_w = crate::rng::Rng::new(seed);
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
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "test_fraction ∈ [0,1] and n is small → product is non-negative and fits usize"
        )]
        let n_test = (n as f64 * test_fraction).max(1.0) as usize;

        let mut rng_split = crate::rng::Rng::new(seed + horizon as u64);
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "modulus i+1 fits usize since i comes from a Vec index"
            )]
            let j = (rng_split.next_u64() % (i as u64 + 1)) as usize;
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
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "DT_MINUTES is 5.0 — fits usize"
            )]
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

/// Extract LSTM hidden state features [mean, std, last] from a window.
#[cfg(feature = "barracuda")]
#[must_use]
pub fn extract_features(window: &[f64], lstm_w: &LstmWeights<'_>) -> Vec<f64> {
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

/// Estimate the spectral radius of a square matrix via power iteration.
#[must_use]
pub fn spectral_radius_estimate(matrix: &[f64], n: usize) -> f64 {
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
#[cfg(feature = "barracuda")]
#[must_use]
pub fn solve_symmetric(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
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
