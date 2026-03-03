// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: Exp 002 transformer inference (gT).
//!
//! Validates GPU Tensor ops for transformer primitive operations:
//! matmul for Q/K/V projections, scaled dot-product attention (Q·K^T),
//! and FFN block (matmul → tanh → matmul).
//!
//! ## S-14 workaround
//!
//! All matmul operations use A × B^T (transpose second operand).
//!
//! ## S-15 workaround
//!
//! All data uses `rng.uniform() * 0.5 + 0.5` ([0.5, 1.0)) to avoid
//! matmul hang on RTX 4070 Vulkan with values near zero.
//!
//! ## Provenance
//!
//! CPU baseline: `control/transformer/transformer_inference.py`

#![expect(
    clippy::cast_possible_truncation,
    clippy::similar_names,
    reason = "validation binary"
)]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_tensor;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{gpu_readback, max_abs_diff_gpu_vs_cpu, ValidationHarness};
use std::sync::Arc;

/// CPU A × B^T: A is M×K (Vec of rows), B is N×K (Vec of rows).
fn cpu_a_bt(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b.len();
    let depth = a[0].len();
    let mut out = vec![vec![0.0_f64; cols]; rows];
    for row_idx in 0..rows {
        for col_idx in 0..cols {
            for inner_idx in 0..depth {
                out[row_idx][col_idx] += a[row_idx][inner_idx] * b[col_idx][inner_idx];
            }
        }
    }
    out
}

fn flatten_f32(data: &[Vec<f64>]) -> Vec<f32> {
    data.iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect()
}

fn flatten_f64(data: &[Vec<f64>]) -> Vec<f64> {
    data.iter().flat_map(|r| r.iter().copied()).collect()
}

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };
    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_gpu_transformer");

    validate_qk_projection(&mut h, &device);
    validate_attention_scores(&mut h, &device);
    validate_ffn_block(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

/// Q = X·Wq^T and K = X·Wk^T for single-head attention.
/// `seq_len=8`, `d_model=16`, `d_k=16`.
fn validate_qk_projection(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(42);
    let seq_len = 8_usize;
    let d_model = 16_usize;
    let d_k = 16_usize;

    let mut safe_val = || rng.uniform().mul_add(0.5, 0.5);

    let x: Vec<Vec<f64>> = (0..seq_len)
        .map(|_| (0..d_model).map(|_| safe_val()).collect())
        .collect();
    let wq: Vec<Vec<f64>> = (0..d_k)
        .map(|_| (0..d_model).map(|_| safe_val()).collect())
        .collect();
    let wk: Vec<Vec<f64>> = (0..d_k)
        .map(|_| (0..d_model).map(|_| safe_val()).collect())
        .collect();

    let cpu_q = cpu_a_bt(&x, &wq);
    let cpu_k = cpu_a_bt(&x, &wk);

    let x_flat = flatten_f32(&x);
    let x_t_q = gpu_tensor!(h, &x_flat, &[seq_len, d_model], device);
    let x_t_k = gpu_tensor!(h, &x_flat, &[seq_len, d_model], device);
    let wq_t = gpu_tensor!(h, &flatten_f32(&wq), &[d_k, d_model], device);
    let wk_t = gpu_tensor!(h, &flatten_f32(&wk), &[d_k, d_model], device);

    let wq_t_t = match wq_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("Wq transpose: {e}"), false);
            return;
        }
    };
    let wk_t_t = match wk_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("Wk transpose: {e}"), false);
            return;
        }
    };

    let gpu_q_t = match x_t_q.matmul(&wq_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("Q matmul: {e}"), false);
            return;
        }
    };
    let gpu_k_t = match x_t_k.matmul(&wk_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("K matmul: {e}"), false);
            return;
        }
    };

    let Some(q_gpu) = gpu_readback(h, &gpu_q_t) else {
        return;
    };
    let Some(k_gpu) = gpu_readback(h, &gpu_k_t) else {
        return;
    };

    let cpu_q_flat = flatten_f64(&cpu_q);
    let cpu_k_flat = flatten_f64(&cpu_k);

    let diff_q = max_abs_diff_gpu_vs_cpu(&q_gpu, &cpu_q_flat);
    let diff_k = max_abs_diff_gpu_vs_cpu(&k_gpu, &cpu_k_flat);

    h.check_upper("Q projection", diff_q, tolerances::BARRACUDA_GPU_ECO_F32);
    h.check_upper("K projection", diff_k, tolerances::BARRACUDA_GPU_ECO_F32);
}

/// Attention scores = Q·K^T (scaled dot-product, no softmax on GPU).
fn validate_attention_scores(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(43);
    let seq_len = 8_usize;
    let d_k = 16_usize;

    let mut safe_val = || rng.uniform().mul_add(0.5, 0.5);

    let q: Vec<Vec<f64>> = (0..seq_len)
        .map(|_| (0..d_k).map(|_| safe_val()).collect())
        .collect();
    let k: Vec<Vec<f64>> = (0..seq_len)
        .map(|_| (0..d_k).map(|_| safe_val()).collect())
        .collect();

    let cpu_scores = cpu_a_bt(&q, &k);

    let q_t = gpu_tensor!(h, &flatten_f32(&q), &[seq_len, d_k], device);
    let k_t = gpu_tensor!(h, &flatten_f32(&k), &[seq_len, d_k], device);

    let k_t_t = match k_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("K transpose: {e}"), false);
            return;
        }
    };

    let scores_t = match q_t.matmul(&k_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("attention scores matmul: {e}"), false);
            return;
        }
    };

    let Some(scores) = gpu_readback(h, &scores_t) else {
        return;
    };

    h.check_bool("scores shape (8×8)", scores.len() == 64);

    let cpu_scores_flat = flatten_f64(&cpu_scores);
    let diff = max_abs_diff_gpu_vs_cpu(&scores, &cpu_scores_flat);
    h.check_upper("attention scores", diff, tolerances::BARRACUDA_GPU_ECO_F32);
}

/// FFN: hidden = X·W1^T → tanh → output = hidden·W2^T.
/// `d_model=16`, `d_ff=32`.
fn validate_ffn_block(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(44);
    let seq_len = 8_usize;
    let d_model = 16_usize;
    let d_ff = 32_usize;

    let mut safe_val = || rng.uniform().mul_add(0.5, 0.5);

    let x: Vec<Vec<f64>> = (0..seq_len)
        .map(|_| (0..d_model).map(|_| safe_val()).collect())
        .collect();
    let w1: Vec<Vec<f64>> = (0..d_ff)
        .map(|_| (0..d_model).map(|_| safe_val()).collect())
        .collect();
    let w2: Vec<Vec<f64>> = (0..d_model)
        .map(|_| (0..d_ff).map(|_| safe_val()).collect())
        .collect();

    let cpu_hidden = cpu_a_bt(&x, &w1);
    let cpu_hidden_tanh: Vec<Vec<f64>> = cpu_hidden
        .iter()
        .map(|row| row.iter().map(|&v| v.tanh()).collect())
        .collect();
    let cpu_output = cpu_a_bt(&cpu_hidden_tanh, &w2);

    let x_t = gpu_tensor!(h, &flatten_f32(&x), &[seq_len, d_model], device);
    let w1_t = gpu_tensor!(h, &flatten_f32(&w1), &[d_ff, d_model], device);
    let w2_t = gpu_tensor!(h, &flatten_f32(&w2), &[d_model, d_ff], device);

    let w1_t_t = match w1_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("W1 transpose: {e}"), false);
            return;
        }
    };
    let w2_t_t = match w2_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("W2 transpose: {e}"), false);
            return;
        }
    };

    let hidden_t = match x_t.matmul(&w1_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("FFN hidden matmul: {e}"), false);
            return;
        }
    };
    let hidden_tanh_t = match hidden_t.tanh() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("FFN tanh: {e}"), false);
            return;
        }
    };
    let Some(hidden_gpu) = gpu_readback(h, &hidden_tanh_t) else {
        return;
    };
    let out_t = match hidden_tanh_t.matmul(&w2_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("FFN output matmul: {e}"), false);
            return;
        }
    };
    let Some(out_gpu) = gpu_readback(h, &out_t) else {
        return;
    };

    let cpu_hidden_tanh_flat = flatten_f64(&cpu_hidden_tanh);
    let cpu_output_flat = flatten_f64(&cpu_output);

    let diff_hidden = max_abs_diff_gpu_vs_cpu(&hidden_gpu, &cpu_hidden_tanh_flat);
    let diff_output = max_abs_diff_gpu_vs_cpu(&out_gpu, &cpu_output_flat);

    h.check_upper(
        "FFN hidden",
        diff_hidden,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
    h.check_upper("FFN output", diff_output, tolerances::BARRACUDA_GPU_ECO_F32);
}

/// Run Q projection twice, check bit-identical readback.
fn validate_determinism(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(45);
    let seq_len = 8_usize;
    let d_model = 16_usize;
    let d_k = 16_usize;

    let mut safe_val = || rng.uniform().mul_add(0.5, 0.5);

    let x: Vec<f32> = (0..seq_len * d_model).map(|_| safe_val() as f32).collect();
    let wq: Vec<f32> = (0..d_k * d_model).map(|_| safe_val() as f32).collect();

    let run = || -> Option<Vec<f32>> {
        let x_t = Tensor::from_data(&x, vec![seq_len, d_model], device.clone()).ok()?;
        let wq_t = Tensor::from_data(&wq, vec![d_k, d_model], device.clone()).ok()?;
        let wq_t_t = wq_t.transpose().ok()?;
        let q_t = x_t.matmul(&wq_t_t).ok()?;
        q_t.to_vec().ok()
    };

    let Some(run1) = run() else {
        h.check_bool("deterministic Q projection run1", false);
        return;
    };
    let Some(run2) = run() else {
        h.check_bool("deterministic Q projection run2", false);
        return;
    };

    h.check_bool("deterministic Q projection", run1 == run2);
}
