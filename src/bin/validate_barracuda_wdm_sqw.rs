// SPDX-License-Identifier: AGPL-3.0-or-later

//! nW-03: `BarraCUDA` GPU validator for WDM S(q,ω) peak predictor.
//!
//! Runs LSTM reservoir + pooled readout through `BarraCUDA` Tensor ops
//! on GPU, comparing with Rust CPU reference path.
//!
//! The LSTM unroll uses per-step GPU matmuls for gate projections,
//! then CPU-side sigmoid/tanh for gate activations (validated separately
//! in `validate_barracuda_lstm`). The final readout is a GPU matmul.
//!
//! Evolution chain:
//! ```text
//! Hansen MD S(q,ω) → Python LSTM → Rust LSTM (CPU) → BarraCUDA (GPU)
//! ```

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    reason = "validation binary"
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::wdm_sqw::{self, SqwPredictor};
use std::sync::Arc;

const BASELINE_JSON: &str = include_str!("../../control/wdm/sqw_peak_baseline.json");
const WASHOUT: usize = 4;

type Dev = Arc<WgpuDevice>;

#[tokio::main]
async fn main() {
    let Ok(gpu) = Gpu::new().await else {
        neural_spring::validation::exit_no_gpu();
    };
    eprintln!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );
    let device: Dev = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_wdm_sqw");

    let pred = match wdm_sqw::load_sqw_from_json(BASELINE_JSON) {
        Ok(p) => p,
        Err(e) => {
            h.check_bool("JSON load", false);
            eprintln!("FATAL: failed to load SQW predictor: {e}");
            h.finish();
        }
    };

    validate_gpu_sqw(&mut h, &pred, &device);

    h.finish();
}

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

    // gates = x @ W_i^T + h @ W_h^T + b_i + b_h
    let input_proj = x_tensor.matmul(&wi_t).map_err(|e| format!("x@Wi: {e}"))?;
    let hidden_proj = h_tensor.matmul(&wh_t).map_err(|e| format!("h@Wh: {e}"))?;
    let sum1 = input_proj
        .add(&hidden_proj)
        .map_err(|e| format!("add1: {e}"))?;
    let sum2 = sum1.add(&bi_tensor).map_err(|e| format!("add_bi: {e}"))?;
    let gates = sum2.add(&bh_tensor).map_err(|e| format!("add_bh: {e}"))?;

    let gates_vec = gates.to_vec().map_err(|e| format!("readback: {e}"))?;

    // Split gates and apply activations (CPU-side for precision).
    // Gate order matches sequence::lstm_cell: forget, input, candidate, output.
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

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn gpu_sqw_predict(
    pred: &SqwPredictor,
    time_series: &[f64],
    device: &Dev,
) -> Result<(f64, f64), String> {
    let hs = pred.hidden_size;

    let normalized: Vec<f64> = time_series
        .iter()
        .map(|&v| (v - pred.norm.series_mean) / pred.norm.series_std)
        .collect();

    let w_i_f32: Vec<f32> = pred.w_i.iter().map(|&v| v as f32).collect();
    let w_h_f32: Vec<f32> = pred.w_h.iter().map(|&v| v as f32).collect();
    let b_i_f32: Vec<f32> = pred.b_i.iter().map(|&v| v as f32).collect();
    let b_h_f32: Vec<f32> = pred.b_h.iter().map(|&v| v as f32).collect();
    let wt = LstmGpuWeights {
        w_i: &w_i_f32,
        w_h: &w_h_f32,
        b_i: &b_i_f32,
        b_h: &b_h_f32,
        hs,
    };

    let mut h = vec![0.0_f32; hs];
    let mut c = vec![0.0_f32; hs];
    let mut all_h: Vec<Vec<f32>> = Vec::with_capacity(normalized.len());

    for val in &normalized {
        let (h_new, c_new) = gpu_lstm_step(*val as f32, &h, &c, &wt, device)?;
        h = h_new;
        c = c_new;
        all_h.push(h.clone());
    }

    let valid_h = &all_h[WASHOUT..];
    let n_valid = valid_h.len() as f32;

    let mut h_mean = vec![0.0_f32; hs];
    for state in valid_h {
        for (m, s) in h_mean.iter_mut().zip(state.iter()) {
            *m += s;
        }
    }
    for m in &mut h_mean {
        *m /= n_valid;
    }

    let mut h_std = vec![0.0_f32; hs];
    for state in valid_h {
        for (j, s) in state.iter().enumerate() {
            h_std[j] += (s - h_mean[j]).powi(2);
        }
    }
    for s in &mut h_std {
        *s = (*s / n_valid).sqrt();
    }

    let h_last = &all_h[all_h.len() - 1];

    let out_dim = 2;
    let weight_feat_dim = pred.w_out.len() / out_dim;

    let features: Vec<f32> = if weight_feat_dim >= 3 * hs {
        let mut f = Vec::with_capacity(3 * hs);
        f.extend_from_slice(&h_mean);
        f.extend_from_slice(&h_std);
        f.extend_from_slice(h_last);
        f
    } else {
        h_last.clone()
    };

    let feat_dim = features.len();
    let w_out_f32: Vec<f32> = pred.w_out.iter().map(|&v| v as f32).collect();
    let b_out_f32: Vec<f32> = pred.b_out.iter().map(|&v| v as f32).collect();

    let feat_tensor = Tensor::from_data(&features, vec![1, feat_dim], device.clone())
        .map_err(|e| format!("feat: {e}"))?;
    let w_out_tensor = Tensor::from_data(&w_out_f32, vec![feat_dim, out_dim], device.clone())
        .map_err(|e| format!("W_out: {e}"))?;
    let b_out_tensor = Tensor::from_data(&b_out_f32, vec![1, out_dim], device.clone())
        .map_err(|e| format!("b_out: {e}"))?;

    let readout = feat_tensor
        .matmul(&w_out_tensor)
        .map_err(|e| format!("readout matmul: {e}"))?;
    let readout_biased = readout
        .add(&b_out_tensor)
        .map_err(|e| format!("readout add: {e}"))?;

    let out_vec = readout_biased
        .to_vec()
        .map_err(|e| format!("readback: {e}"))?;

    let omega = f64::from(out_vec[0]).mul_add(pred.norm.y_std[0], pred.norm.y_mean[0]);
    let gamma = f64::from(out_vec[1]).mul_add(pred.norm.y_std[1], pred.norm.y_mean[1]);

    Ok((omega, gamma))
}

fn validate_gpu_sqw(h: &mut ValidationHarness, pred: &SqwPredictor, device: &Dev) {
    let ts: Vec<f64> = (0..20).map(|i| (f64::from(i) * 0.3).cos()).collect();

    let (cpu_omega, cpu_gamma) = pred.predict(&ts);

    h.check_bool("CPU omega finite", cpu_omega.is_finite());
    h.check_bool("CPU gamma finite", cpu_gamma.is_finite());

    let (gpu_omega, gpu_gamma) = match gpu_sqw_predict(pred, &ts, device) {
        Ok(r) => r,
        Err(e) => {
            h.check_bool("GPU forward pass", false);
            eprintln!("  GPU forward failed: {e}");
            return;
        }
    };

    h.check_bool("GPU omega finite", gpu_omega.is_finite());
    h.check_bool("GPU gamma finite", gpu_gamma.is_finite());

    let omega_diff = (gpu_omega - cpu_omega).abs();
    let gamma_diff = (gpu_gamma - cpu_gamma).abs();

    let omega_rel = if cpu_omega.abs() > tolerances::RELATIVE_ERROR_FLOOR {
        omega_diff / cpu_omega.abs()
    } else {
        omega_diff
    };
    let gamma_rel = if cpu_gamma.abs() > tolerances::RELATIVE_ERROR_FLOOR {
        gamma_diff / cpu_gamma.abs()
    } else {
        gamma_diff
    };

    h.check_bool(
        &format!("GPU omega within tol, rel={omega_rel:.2e}"),
        omega_rel < tolerances::ML_MLP_F32,
    );
    h.check_bool(
        &format!("GPU gamma within tol, rel={gamma_rel:.2e}"),
        gamma_rel < tolerances::ML_MLP_F32,
    );

    // Determinism
    let Ok((o1, g1)) = gpu_sqw_predict(pred, &ts, device) else {
        h.check_bool("GPU determinism (re-run failed)", false);
        return;
    };
    h.check_bool(
        "GPU determinism",
        (o1 - gpu_omega).abs() < tolerances::EXACT_F64
            && (g1 - gpu_gamma).abs() < tolerances::EXACT_F64,
    );
}
