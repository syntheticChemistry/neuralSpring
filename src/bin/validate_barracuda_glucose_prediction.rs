// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paper 026: `BarraCUDA` CPU + GPU validator for LSTM blood glucose prediction.
//!
//! Validates the glucose prediction LSTM reservoir through two tiers:
//!
//! **Tier 1 — `BarraCUDA` CPU**: Verifies that `barracuda::stats` primitives
//! (variance, Pearson correlation) produce identical results to the pure
//! Rust implementation. Also validates autocorrelation decay and CGM
//! statistics using barracuda math.
//!
//! **Tier 2 — `BarraCUDA` GPU**: Runs the LSTM gate projections through
//! `Tensor::matmul` + CPU-side sigmoid/tanh, and the readout through
//! `Tensor::matmul` + `Tensor::add`. Compares GPU predictions against
//! CPU reference at each horizon.
//!
//! Evolution chain:
//! ```text
//! Chuna CGM LSTM → Python reservoir → Rust CPU → BarraCUDA (CPU stats) → BarraCUDA (GPU Tensor)
//! ```
//!
//! ## Provenance
//!
//! | Baseline | Source |
//! |----------|--------|
//! | Python baseline | `control/glucose_prediction/glucose_prediction.py` |
//! | Reference | Chuna (2020), medRxiv 2020.08.04.20117812 |
//! | Baseline commit | in-tree (`control/glucose_prediction/`) |
//! | Baseline date | 2026-03-05 |
//! | Command | `python glucose_prediction.py --seed 42` |

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "validation binary with multi-tier LSTM GPU promotion"
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::glucose_prediction::{
    autocorrelation, create_sequences, estimate_tau, generate_synthetic_cgm,
    load_glucose_from_json, r2_score, rmse, run_glucose_experiment,
};
use neural_spring::gpu::Gpu;
use neural_spring::sequence::{lstm_cell, LstmWeights};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

const BASELINE_JSON: &str =
    include_str!("../../control/glucose_prediction/glucose_prediction_baseline.json");

const WASHOUT: usize = 4;

type Dev = Arc<WgpuDevice>;

// ═══════════════════════════════════════════════════════════════════
// GPU LSTM helpers (same pattern as validate_barracuda_wdm_sqw.rs)
// ═══════════════════════════════════════════════════════════════════

struct LstmGpuWeights<'a> {
    w_i: &'a [f32],
    w_h: &'a [f32],
    b_i: &'a [f32],
    b_h: &'a [f32],
    hs: usize,
}

fn gpu_lstm_step(
    x_val: f32,
    h_prev: &[f32],
    c_prev: &[f32],
    wt: &LstmGpuWeights<'_>,
    device: &Dev,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let hs = wt.hs;

    let x_tensor =
        Tensor::from_data(&[x_val], vec![1, 1], device.clone()).map_err(|e| format!("x: {e}"))?;
    let h_tensor =
        Tensor::from_data(h_prev, vec![1, hs], device.clone()).map_err(|e| format!("h: {e}"))?;

    let wi_tensor = Tensor::from_data(wt.w_i, vec![4 * hs, 1], device.clone())
        .map_err(|e| format!("W_i: {e}"))?;
    let wi_t = wi_tensor.transpose().map_err(|e| format!("W_i^T: {e}"))?;

    let wh_tensor = Tensor::from_data(wt.w_h, vec![4 * hs, hs], device.clone())
        .map_err(|e| format!("W_h: {e}"))?;
    let wh_t = wh_tensor.transpose().map_err(|e| format!("W_h^T: {e}"))?;

    let bi_tensor = Tensor::from_data(wt.b_i, vec![1, 4 * hs], device.clone())
        .map_err(|e| format!("b_i: {e}"))?;
    let bh_tensor = Tensor::from_data(wt.b_h, vec![1, 4 * hs], device.clone())
        .map_err(|e| format!("b_h: {e}"))?;

    let input_proj = x_tensor.matmul(&wi_t).map_err(|e| format!("x@Wi: {e}"))?;
    let hidden_proj = h_tensor.matmul(&wh_t).map_err(|e| format!("h@Wh: {e}"))?;
    let sum1 = input_proj
        .add(&hidden_proj)
        .map_err(|e| format!("add1: {e}"))?;
    let sum2 = sum1.add(&bi_tensor).map_err(|e| format!("add_bi: {e}"))?;
    let gates = sum2.add(&bh_tensor).map_err(|e| format!("add_bh: {e}"))?;

    let gates_vec = gates.to_vec().map_err(|e| format!("readback: {e}"))?;

    let f_gate: Vec<f32> = gates_vec[..hs].iter().map(|v| sigmoid_f32(*v)).collect();
    let i_gate: Vec<f32> = gates_vec[hs..2 * hs]
        .iter()
        .map(|v| sigmoid_f32(*v))
        .collect();
    let g_gate: Vec<f32> = gates_vec[2 * hs..3 * hs].iter().map(|v| v.tanh()).collect();
    let o_gate: Vec<f32> = gates_vec[3 * hs..]
        .iter()
        .map(|v| sigmoid_f32(*v))
        .collect();

    let c_new: Vec<f32> = (0..hs)
        .map(|j| f_gate[j].mul_add(c_prev[j], i_gate[j] * g_gate[j]))
        .collect();
    let h_new: Vec<f32> = (0..hs).map(|j| o_gate[j] * c_new[j].tanh()).collect();

    Ok((h_new, c_new))
}

use neural_spring::primitives::sigmoid_f32;

/// Run the full LSTM reservoir on GPU for a single input window.
///
/// Returns the pooled feature vector [mean, std, last] from GPU LSTM unroll.
fn gpu_lstm_features(
    window: &[f64],
    wt: &LstmGpuWeights<'_>,
    device: &Dev,
) -> Result<Vec<f32>, String> {
    let hs = wt.hs;
    let mut h = vec![0.0_f32; hs];
    let mut c = vec![0.0_f32; hs];
    let mut all_h: Vec<Vec<f32>> = Vec::with_capacity(window.len());

    for val in window {
        let (h_new, c_new) = gpu_lstm_step(*val as f32, &h, &c, wt, device)?;
        h = h_new;
        c = c_new;
        all_h.push(h.clone());
    }

    let valid_h = &all_h[WASHOUT.min(all_h.len())..];
    let n_valid = valid_h.len() as f32;

    let mut h_mean = vec![0.0_f32; hs];
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

    let mut h_std = vec![0.0_f32; hs];
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
    Ok(features)
}

/// CPU LSTM feature extraction using f64 weights (reference for GPU parity).
fn extract_cpu_features(window: &[f64], lstm_w: &LstmWeights<'_>) -> Vec<f64> {
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

/// GPU readout: features @ `W_out` + `b_out` via Tensor matmul.
fn gpu_readout(
    features: &[f32],
    w_out: &[f32],
    b_out: f32,
    feat_dim: usize,
    device: &Dev,
) -> Result<f32, String> {
    let feat_tensor = Tensor::from_data(features, vec![1, feat_dim], device.clone())
        .map_err(|e| format!("feat: {e}"))?;
    let w_tensor = Tensor::from_data(w_out, vec![feat_dim, 1], device.clone())
        .map_err(|e| format!("W_out: {e}"))?;
    let b_tensor = Tensor::from_data(&[b_out], vec![1, 1], device.clone())
        .map_err(|e| format!("b_out: {e}"))?;

    let readout = feat_tensor
        .matmul(&w_tensor)
        .map_err(|e| format!("readout matmul: {e}"))?;
    let biased = readout
        .add(&b_tensor)
        .map_err(|e| format!("readout add: {e}"))?;
    let out = biased.to_vec().map_err(|e| format!("readback: {e}"))?;
    Ok(out[0])
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("barracuda_glucose_prediction");

    // ── Tier 1: BarraCUDA CPU primitives ──
    eprintln!("\n═══ Tier 1: BarraCUDA CPU primitives ═══");
    validate_barracuda_cpu(&mut h);

    // ── Tier 2: BarraCUDA GPU Tensor ──
    eprintln!("\n═══ Tier 2: BarraCUDA GPU Tensor ═══");
    match Gpu::new().await {
        Ok(gpu) => {
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                gpu.adapter_name, gpu.device_type, gpu.backend,
            );
            let device: Dev = gpu.wgpu_device().clone();
            validate_barracuda_gpu(&mut h, &device);
        }
        Err(_) => {
            eprintln!("  [skip] No GPU available — Tier 2 skipped");
        }
    }

    h.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Tier 1: BarraCUDA CPU
// ═══════════════════════════════════════════════════════════════════

fn validate_barracuda_cpu(h: &mut ValidationHarness) {
    let cgm = generate_synthetic_cgm(14, 42);
    let n = cgm.len() as f64;

    let mean = cgm.iter().sum::<f64>() / n;
    let cpu_var = cgm.iter().map(|&g| (g - mean).powi(2)).sum::<f64>() / n;

    let bc_var = barracuda::stats::correlation::variance(&cgm).unwrap_or(0.0);
    let ddof_correction = bc_var * (n - 1.0) / n;
    h.check_abs(
        "bC CPU: variance matches (corrected ddof)",
        ddof_correction,
        cpu_var,
        tolerances::CROSS_LANGUAGE,
    );

    let acor = autocorrelation(&cgm, 144);
    let tau = estimate_tau(&acor);
    let tau_hrs = tau as f64 * 5.0 / 60.0;
    h.check_bool(
        "bC CPU: τ in [1.0, 5.0] hrs",
        (1.0..=5.0).contains(&tau_hrs),
    );

    let cgm_norm: Vec<f64> = cgm
        .iter()
        .map(|&g| (g - mean) / cpu_var.sqrt().max(tolerances::VARIANCE_DIVISION_GUARD))
        .collect();
    let (inputs, targets) = create_sequences(&cgm_norm, 12, 6);

    let Ok(baseline) = load_glucose_from_json(BASELINE_JSON) else {
        h.check_bool("bC CPU: baseline JSON load", false);
        return;
    };
    let lstm_w = LstmWeights {
        w_input: &baseline.w_i,
        w_hidden: &baseline.w_h,
        b_input: &baseline.b_i,
        b_hidden: &baseline.b_h,
        hidden_size: baseline.hidden_size,
    };

    let window = &inputs[0];
    let hs = baseline.hidden_size;
    let mut hh = vec![0.0; hs];
    let mut cc = vec![0.0; hs];
    for val in window {
        let (h_new, c_new) = lstm_cell(&[*val], &hh, &cc, &lstm_w);
        hh = h_new;
        cc = c_new;
    }

    h.check_bool(
        "bC CPU: LSTM hidden state finite",
        hh.iter().all(|v| v.is_finite()),
    );
    h.check_bool(
        "bC CPU: LSTM cell state finite",
        cc.iter().all(|v| v.is_finite()),
    );
    h.check_bool("bC CPU: LSTM hidden size correct", hh.len() == hs);

    let (results, _predictor) = run_glucose_experiment(14, 24, 12, &[1, 6, 12, 24, 48], 42);
    h.check_bool("bC CPU: 5 horizon results", results.len() == 5);
    h.check_bool("bC CPU: R²(5min) > 0.85", results[0].r2_lstm > 0.85);
    h.check_bool(
        "bC CPU: R² degrades with horizon",
        results[0].r2_lstm > results[4].r2_lstm,
    );

    let actual = vec![100.0, 120.0, 140.0, 160.0, 180.0];
    let predicted = vec![102.0, 118.0, 142.0, 157.0, 183.0];
    let bc_r2 = barracuda::stats::r_squared(&actual, &predicted);
    let cpu_r2 = r2_score(&actual, &predicted);
    h.check_abs(
        "bC CPU: R² parity barracuda vs local",
        bc_r2,
        cpu_r2,
        tolerances::CROSS_LANGUAGE,
    );

    let bc_rmse_val = barracuda::stats::rmse(&actual, &predicted);
    let cpu_rmse_val = rmse(&actual, &predicted);
    h.check_abs(
        "bC CPU: RMSE parity barracuda vs local",
        bc_rmse_val,
        cpu_rmse_val,
        tolerances::CROSS_LANGUAGE,
    );

    let x: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.1).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&v| 0.01f64.mul_add(v.sin(), 2.0f64.mul_add(v, 1.0)))
        .collect();
    let bc_corr = barracuda::stats::correlation::pearson_correlation(&x, &y).unwrap_or(0.0);
    h.check_bool(
        &format!("bC CPU: Pearson correlation near 1.0 (got {bc_corr:.6})"),
        bc_corr > 0.999,
    );

    eprintln!("  bC CPU: LSTM inference, stats primitives, and experiment orchestration verified");
    let _ = (inputs, targets);
}

// ═══════════════════════════════════════════════════════════════════
// Tier 2: BarraCUDA GPU
// ═══════════════════════════════════════════════════════════════════

fn validate_barracuda_gpu(h: &mut ValidationHarness, device: &Dev) {
    let baseline = match load_glucose_from_json(BASELINE_JSON) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool("GPU: JSON load", false);
            eprintln!("FATAL: {e}");
            return;
        }
    };

    let hs = baseline.hidden_size;
    let cgm = generate_synthetic_cgm(14, 42);
    let mean = cgm.iter().sum::<f64>() / cgm.len() as f64;
    let var = cgm.iter().map(|&g| (g - mean).powi(2)).sum::<f64>() / cgm.len() as f64;
    let std = var.sqrt().max(tolerances::VARIANCE_DIVISION_GUARD);

    let cgm_norm: Vec<f64> = cgm.iter().map(|&g| (g - mean) / std).collect();

    let w_i_f32: Vec<f32> = baseline.w_i.iter().map(|&v| v as f32).collect();
    let w_h_f32: Vec<f32> = baseline.w_h.iter().map(|&v| v as f32).collect();
    let b_i_f32: Vec<f32> = baseline.b_i.iter().map(|&v| v as f32).collect();
    let b_h_f32: Vec<f32> = baseline.b_h.iter().map(|&v| v as f32).collect();
    let wt = LstmGpuWeights {
        w_i: &w_i_f32,
        w_h: &w_h_f32,
        b_i: &b_i_f32,
        b_h: &b_h_f32,
        hs,
    };

    // ── Single-window GPU LSTM vs CPU LSTM ──
    eprintln!("\n  GPU LSTM gate-level parity check...");

    let test_window = &cgm_norm[0..12];

    let cpu_lstm_w = LstmWeights {
        w_input: &baseline.w_i,
        w_hidden: &baseline.w_h,
        b_input: &baseline.b_i,
        b_hidden: &baseline.b_h,
        hidden_size: hs,
    };

    let mut cpu_h = vec![0.0; hs];
    let mut cpu_c = vec![0.0; hs];
    let mut cpu_all_h = Vec::with_capacity(test_window.len());
    for val in test_window {
        let (h_new, c_new) = lstm_cell(&[*val], &cpu_h, &cpu_c, &cpu_lstm_w);
        cpu_h = h_new;
        cpu_c = c_new;
        cpu_all_h.push(cpu_h.clone());
    }

    let gpu_features = match gpu_lstm_features(test_window, &wt, device) {
        Ok(f) => f,
        Err(e) => {
            h.check_bool("GPU: LSTM forward pass", false);
            eprintln!("  GPU LSTM failed: {e}");
            return;
        }
    };

    h.check_bool("GPU: features length = 3*hs", gpu_features.len() == 3 * hs);
    h.check_bool(
        "GPU: features finite",
        gpu_features.iter().all(|v| v.is_finite()),
    );

    let valid_cpu_h = &cpu_all_h[WASHOUT.min(cpu_all_h.len())..];
    let n_valid = valid_cpu_h.len() as f64;
    let mut cpu_mean = vec![0.0; hs];
    for state in valid_cpu_h {
        for (m, s) in cpu_mean.iter_mut().zip(state.iter()) {
            *m += s;
        }
    }
    for m in &mut cpu_mean {
        *m /= n_valid;
    }

    let mean_diff: f64 = cpu_mean
        .iter()
        .zip(gpu_features[..hs].iter())
        .map(|(&c, &g)| (c - f64::from(g)).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        &format!("GPU: hidden mean parity (diff={mean_diff:.2e})"),
        mean_diff < tolerances::ML_MLP_F32,
    );

    // ── Multi-horizon GPU vs CPU parity ──
    eprintln!("\n  Multi-horizon GPU vs CPU parity...");

    for &(horizon, ref readout) in &baseline.readouts {
        let (inputs, _targets) = create_sequences(&cgm_norm, baseline.seq_len, horizon);
        let horizon_min = horizon * 5;

        let n_test = 20.min(inputs.len());
        let mut gpu_preds = Vec::with_capacity(n_test);
        let mut cpu_ref_preds = Vec::with_capacity(n_test);

        let w_out_f32: Vec<f32> = readout.w_out.iter().map(|&v| v as f32).collect();
        let b_out_f32 = readout.b_out as f32;
        let feat_dim = 3 * hs;

        for (i, window) in inputs.iter().take(n_test).enumerate() {
            let cpu_feat = extract_cpu_features(window, &cpu_lstm_w);
            let cpu_pred_norm: f64 = cpu_feat
                .iter()
                .zip(readout.w_out.iter())
                .map(|(&f, &w)| f * w)
                .sum::<f64>()
                + readout.b_out;
            cpu_ref_preds.push(cpu_pred_norm.mul_add(std, mean));

            let gpu_feat = match gpu_lstm_features(window, &wt, device) {
                Ok(f) => f,
                Err(e) => {
                    h.check_bool(&format!("GPU: horizon {horizon_min}min sample {i}"), false);
                    eprintln!("  GPU failed: {e}");
                    return;
                }
            };

            let gpu_pred_norm =
                match gpu_readout(&gpu_feat, &w_out_f32, b_out_f32, feat_dim, device) {
                    Ok(p) => p,
                    Err(e) => {
                        h.check_bool(&format!("GPU: readout {horizon_min}min sample {i}"), false);
                        eprintln!("  GPU readout failed: {e}");
                        return;
                    }
                };
            gpu_preds.push(f64::from(gpu_pred_norm).mul_add(std, mean));
        }

        let gpu_finite = gpu_preds.iter().all(|v| v.is_finite());
        h.check_bool(
            &format!("GPU: {horizon_min}min predictions finite"),
            gpu_finite,
        );

        let max_rel_diff: f64 = gpu_preds
            .iter()
            .zip(cpu_ref_preds.iter())
            .map(|(&g, &c)| {
                let floor = c.abs().max(tolerances::RELATIVE_ERROR_FLOOR);
                (g - c).abs() / floor
            })
            .fold(0.0_f64, f64::max);
        h.check_bool(
            &format!("GPU: {horizon_min}min CPU parity (rel={max_rel_diff:.2e})"),
            max_rel_diff < tolerances::ML_MLP_F32,
        );
        eprintln!("    {horizon_min:>3}min: GPU↔CPU max_rel={max_rel_diff:.2e}, n={n_test}");
    }

    // ── GPU determinism ──
    eprintln!("\n  GPU determinism check...");
    let test_win = &cgm_norm[100..112];

    let f1 = gpu_lstm_features(test_win, &wt, device);
    let f2 = gpu_lstm_features(test_win, &wt, device);

    match (f1, f2) {
        (Ok(a), Ok(b)) => {
            let max_diff: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f32, f32::max);
            h.check_bool(
                &format!("GPU: deterministic (diff={max_diff:.2e})"),
                max_diff < tolerances::CROSS_LANGUAGE as f32,
            );
        }
        _ => {
            h.check_bool("GPU: determinism (runs failed)", false);
        }
    }
}
