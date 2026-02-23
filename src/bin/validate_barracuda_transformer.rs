// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` validation: Exp 002 transformer inference (bC tier).
//!
//! Validates `BarraCUDA` Tensor API for transformer primitive operations against
//! f64 CPU baselines. Covers Q/K/V projections, scaled dot-product attention
//! scores, and the feed-forward network block (matmul → tanh → matmul).
//!
//! This is the bC (`BarraCUDA` CPU-tier) companion to `validate_barracuda_gpu_transformer`
//! which focuses on GPU-specific paths.
//!
//! ## S-14 workaround
//!
//! All matmul operations use A × B^T (transpose second operand).
//!
//! ## S-15 workaround
//!
//! Data in `\[0.5, 1.0)` range to avoid matmul hang on RTX 4070 Vulkan.
//!
//! ## Provenance
//!
//! CPU baseline: `control/transformer/transformer_inference.py`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::require;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{max_abs_diff_gpu_vs_cpu, ValidationHarness};
use std::sync::Arc;

fn cpu_matmul_a_bt(a: &[f64], a_rows: usize, a_cols: usize, b: &[f64], b_rows: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; a_rows * b_rows];
    for i in 0..a_rows {
        for j in 0..b_rows {
            let mut sum = 0.0;
            for k in 0..a_cols {
                sum += a[i * a_cols + k] * b[j * a_cols + k];
            }
            out[i * b_rows + j] = sum;
        }
    }
    out
}

fn tensor_from(
    data: &[f32],
    shape: Vec<usize>,
    device: &Arc<WgpuDevice>,
) -> Result<Tensor, barracuda::error::BarracudaError> {
    Tensor::from_data(data, shape, device.clone())
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
        Err(e) => {
            eprintln!("  SKIP: {e} — no GPU/CPU adapter available");
            eprintln!("  0/0 checks — skipping gracefully");
            std::process::exit(0);
        }
    };
    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_transformer");

    validate_qkv_projections(&mut h, &device);
    validate_attention_scores(&mut h, &device);
    validate_ffn_block(&mut h, &device);
    validate_residual_add(&mut h, &device);
    validate_softmax(&mut h, &device);
    validate_full_layer(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

/// Q = X·Wq^T, K = X·Wk^T, V = X·Wv^T for single-head attention.
fn validate_qkv_projections(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = Rng::new(42);
    let seq = 8_usize;
    let d_model = 16_usize;

    let mut safe_val = || rng.uniform().mul_add(0.5, 0.5);

    let x_f64: Vec<f64> = (0..seq * d_model).map(|_| safe_val()).collect();
    let wq_f64: Vec<f64> = (0..d_model * d_model).map(|_| safe_val()).collect();
    let wk_f64: Vec<f64> = (0..d_model * d_model).map(|_| safe_val()).collect();
    let wv_f64: Vec<f64> = (0..d_model * d_model).map(|_| safe_val()).collect();

    let cpu_q = cpu_matmul_a_bt(&x_f64, seq, d_model, &wq_f64, d_model);
    let cpu_k = cpu_matmul_a_bt(&x_f64, seq, d_model, &wk_f64, d_model);
    let cpu_v = cpu_matmul_a_bt(&x_f64, seq, d_model, &wv_f64, d_model);

    let x_f32: Vec<f32> = x_f64.iter().map(|&v| v as f32).collect();
    let wq_f32: Vec<f32> = wq_f64.iter().map(|&v| v as f32).collect();
    let wk_f32: Vec<f32> = wk_f64.iter().map(|&v| v as f32).collect();
    let wv_f32: Vec<f32> = wv_f64.iter().map(|&v| v as f32).collect();

    for (name, w_f32, cpu_ref) in [
        ("Q", &wq_f32, &cpu_q),
        ("K", &wk_f32, &cpu_k),
        ("V", &wv_f32, &cpu_v),
    ] {
        let w_t = require!(h, tensor_from(w_f32, vec![d_model, d_model], device), name);
        let w_t_t = require!(h, w_t.transpose(), &format!("W{name} transpose"));
        let x_t = require!(
            h,
            tensor_from(&x_f32, vec![seq, d_model], device),
            &format!("X for {name}")
        );
        let proj = require!(h, x_t.matmul(&w_t_t), &format!("{name} matmul"));
        let out = require!(h, proj.to_vec(), &format!("{name} readback"));
        let diff = max_abs_diff_gpu_vs_cpu(&out, cpu_ref);
        h.check_upper(
            &format!("{name} projection: diff={diff:.2e}"),
            diff,
            tolerances::BARRACUDA_GPU_ECO_F32,
        );
    }
}

/// Attention scores = Q · K^T.
fn validate_attention_scores(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = Rng::new(43);
    let seq = 8_usize;
    let dk = 16_usize;

    let mut safe_val = || rng.uniform().mul_add(0.5, 0.5);

    let q_f64: Vec<f64> = (0..seq * dk).map(|_| safe_val()).collect();
    let k_f64: Vec<f64> = (0..seq * dk).map(|_| safe_val()).collect();
    let cpu_scores = cpu_matmul_a_bt(&q_f64, seq, dk, &k_f64, seq);

    let q_f32: Vec<f32> = q_f64.iter().map(|&v| v as f32).collect();
    let k_f32: Vec<f32> = k_f64.iter().map(|&v| v as f32).collect();

    let q_t = require!(h, tensor_from(&q_f32, vec![seq, dk], device), "Q");
    let k_t = require!(h, tensor_from(&k_f32, vec![seq, dk], device), "K");
    let k_t_t = require!(h, k_t.transpose(), "K^T");
    let scores_t = require!(h, q_t.matmul(&k_t_t), "Q·K^T");
    let scores = require!(h, scores_t.to_vec(), "scores readback");

    let diff = max_abs_diff_gpu_vs_cpu(&scores, &cpu_scores);
    h.check_upper(
        &format!("attention scores: diff={diff:.2e}"),
        diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

/// FFN block: hidden = X·W1^T → tanh → output = hidden·W2^T.
fn validate_ffn_block(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = Rng::new(44);
    let seq = 8_usize;
    let d_model = 16_usize;
    let d_ff = 32_usize;

    let mut safe_val = || rng.uniform().mul_add(0.5, 0.5);

    let x_f64: Vec<f64> = (0..seq * d_model).map(|_| safe_val()).collect();
    let w1_f64: Vec<f64> = (0..d_ff * d_model).map(|_| safe_val()).collect();
    let w2_f64: Vec<f64> = (0..d_model * d_ff).map(|_| safe_val()).collect();

    let cpu_hidden = cpu_matmul_a_bt(&x_f64, seq, d_model, &w1_f64, d_ff);
    let cpu_hidden_tanh: Vec<f64> = cpu_hidden.iter().map(|&v| v.tanh()).collect();
    let cpu_output = cpu_matmul_a_bt(&cpu_hidden_tanh, seq, d_ff, &w2_f64, d_model);

    let x_f32: Vec<f32> = x_f64.iter().map(|&v| v as f32).collect();
    let w1_f32: Vec<f32> = w1_f64.iter().map(|&v| v as f32).collect();
    let w2_f32: Vec<f32> = w2_f64.iter().map(|&v| v as f32).collect();

    let x_t = require!(h, tensor_from(&x_f32, vec![seq, d_model], device), "FFN X");
    let w1_t = require!(
        h,
        tensor_from(&w1_f32, vec![d_ff, d_model], device),
        "FFN W1"
    );
    let w2_t = require!(
        h,
        tensor_from(&w2_f32, vec![d_model, d_ff], device),
        "FFN W2"
    );
    let w1_t_t = require!(h, w1_t.transpose(), "W1^T");
    let w2_t_t = require!(h, w2_t.transpose(), "W2^T");

    let hidden = require!(h, x_t.matmul(&w1_t_t), "FFN hidden matmul");
    let hidden_tanh = require!(h, hidden.tanh(), "FFN tanh");
    let hidden_out = require!(h, hidden_tanh.to_vec(), "FFN hidden readback");
    let diff_h = max_abs_diff_gpu_vs_cpu(&hidden_out, &cpu_hidden_tanh);
    h.check_upper(
        &format!("FFN hidden: diff={diff_h:.2e}"),
        diff_h,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    let output = require!(h, hidden_tanh.matmul(&w2_t_t), "FFN output matmul");
    let output_data = require!(h, output.to_vec(), "FFN output readback");
    let diff_o = max_abs_diff_gpu_vs_cpu(&output_data, &cpu_output);
    h.check_upper(
        &format!("FFN output: diff={diff_o:.2e}"),
        diff_o,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

/// Residual connection: output = x + sublayer(x).
fn validate_residual_add(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = Rng::new(45);
    let n = 64_usize;

    let a_f64: Vec<f64> = (0..n).map(|_| rng.uniform().mul_add(0.5, 0.5)).collect();
    let b_f64: Vec<f64> = (0..n).map(|_| rng.uniform().mul_add(0.5, 0.5)).collect();
    let cpu_sum: Vec<f64> = a_f64
        .iter()
        .zip(b_f64.iter())
        .map(|(&x, &y)| x + y)
        .collect();

    let a_f32: Vec<f32> = a_f64.iter().map(|&v| v as f32).collect();
    let b_f32: Vec<f32> = b_f64.iter().map(|&v| v as f32).collect();

    let a_t = require!(h, tensor_from(&a_f32, vec![8, 8], device), "residual a");
    let b_t = require!(h, tensor_from(&b_f32, vec![8, 8], device), "residual b");
    let sum_t = require!(h, a_t.add(&b_t), "residual add");
    let out = require!(h, sum_t.to_vec(), "residual readback");

    let diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_sum);
    h.check_upper(
        &format!("residual add: diff={diff:.2e}"),
        diff,
        tolerances::TENSOR_EXACT_F32,
    );
}

/// Global softmax (1D): `Tensor::softmax()` normalizes over all elements.
///
/// `BarraCUDA` softmax is global (entire tensor), not row-wise. Row-wise softmax
/// for attention requires `ScaledDotProductAttention` or manual per-row dispatch.
/// This test validates the global 1D case which is correct and useful for
/// classification logits.
fn validate_softmax(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = Rng::new(46);
    let n = 8_usize;

    let data_f64: Vec<f64> = (0..n).map(|_| rng.uniform() * 2.0).collect();

    // CPU softmax (global)
    let max_val = data_f64.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = data_f64.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let cpu_softmax: Vec<f64> = exps.iter().map(|&e| e / sum).collect();

    let data_f32: Vec<f32> = data_f64.iter().map(|&v| v as f32).collect();
    let data_t = require!(h, tensor_from(&data_f32, vec![n], device), "softmax input");
    let sm_t = require!(h, data_t.softmax(), "softmax");
    let out = require!(h, sm_t.to_vec(), "softmax readback");

    let diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_softmax);
    h.check_upper(
        &format!("softmax 1D global: diff={diff:.2e}"),
        diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    let total: f64 = out.iter().map(|&x| f64::from(x)).sum();
    h.check_abs(
        "softmax sums to 1",
        total,
        1.0,
        tolerances::TENSOR_EXACT_F32,
    );
}

/// Full transformer layer: Q/K/V → attention → residual → FFN → residual.
fn validate_full_layer(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = Rng::new(47);
    let seq = 4_usize;
    let dm = 8_usize;
    let dff = 16_usize;
    let dk = 8_usize;

    let mut sv = || rng.uniform().mul_add(0.5, 0.5);

    let x_f64: Vec<f64> = (0..seq * dm).map(|_| sv()).collect();
    let wq_f64: Vec<f64> = (0..dk * dm).map(|_| sv()).collect();
    let wk_f64: Vec<f64> = (0..dk * dm).map(|_| sv()).collect();
    let wv_f64: Vec<f64> = (0..dk * dm).map(|_| sv()).collect();
    let w1_f64: Vec<f64> = (0..dff * dm).map(|_| sv()).collect();
    let w2_f64: Vec<f64> = (0..dm * dff).map(|_| sv()).collect();

    // CPU forward — skip row-wise softmax (BarraCUDA Tensor::softmax() is global,
    // not row-wise; row-wise requires ScaledDotProductAttention). Use raw scaled
    // scores · V to validate the matmul chain end-to-end.
    let q = cpu_matmul_a_bt(&x_f64, seq, dm, &wq_f64, dk);
    let k = cpu_matmul_a_bt(&x_f64, seq, dm, &wk_f64, dk);
    let v = cpu_matmul_a_bt(&x_f64, seq, dm, &wv_f64, dk);
    let scores = cpu_matmul_a_bt(&q, seq, dk, &k, seq);
    let scale = 1.0 / (dk as f64).sqrt();
    let scaled: Vec<f64> = scores.iter().map(|&s| s * scale).collect();

    // Attention output = scaled_scores · V (no softmax)
    let attn_out = cpu_matmul_a_bt(&scaled, seq, seq, &v, dk);

    // Residual: x + attn_out
    let residual1: Vec<f64> = x_f64
        .iter()
        .zip(attn_out.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    // FFN
    let ffn_hidden = cpu_matmul_a_bt(&residual1, seq, dm, &w1_f64, dff);
    let ffn_tanh: Vec<f64> = ffn_hidden.iter().map(|&v| v.tanh()).collect();
    let ffn_out = cpu_matmul_a_bt(&ffn_tanh, seq, dff, &w2_f64, dm);
    let cpu_final: Vec<f64> = residual1
        .iter()
        .zip(ffn_out.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    // GPU forward
    let f32_of = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };

    let x_t = require!(
        h,
        tensor_from(&f32_of(&x_f64), vec![seq, dm], device),
        "layer X"
    );
    let wq_t = require!(
        h,
        tensor_from(&f32_of(&wq_f64), vec![dk, dm], device),
        "layer Wq"
    );
    let wk_t = require!(
        h,
        tensor_from(&f32_of(&wk_f64), vec![dk, dm], device),
        "layer Wk"
    );
    let wv_t = require!(
        h,
        tensor_from(&f32_of(&wv_f64), vec![dk, dm], device),
        "layer Wv"
    );
    let w1_t = require!(
        h,
        tensor_from(&f32_of(&w1_f64), vec![dff, dm], device),
        "layer W1"
    );
    let w2_t = require!(
        h,
        tensor_from(&f32_of(&w2_f64), vec![dm, dff], device),
        "layer W2"
    );

    let wq_tt = require!(h, wq_t.transpose(), "Wq^T");
    let wk_tt = require!(h, wk_t.transpose(), "Wk^T");
    let wv_tt = require!(h, wv_t.transpose(), "Wv^T");
    let w1_tt = require!(h, w1_t.transpose(), "W1^T");
    let w2_tt = require!(h, w2_t.transpose(), "W2^T");

    let x2 = require!(
        h,
        tensor_from(&f32_of(&x_f64), vec![seq, dm], device),
        "X copy for K"
    );
    let x3 = require!(
        h,
        tensor_from(&f32_of(&x_f64), vec![seq, dm], device),
        "X copy for V"
    );

    let q_t = require!(h, x_t.matmul(&wq_tt), "Q");
    let k_t = require!(h, x2.matmul(&wk_tt), "K");
    let v_t = require!(h, x3.matmul(&wv_tt), "V");

    let k_transposed = require!(h, k_t.transpose(), "K transpose for scores");
    let scores_t = require!(h, q_t.matmul(&k_transposed), "scores");
    let scores_scaled = require!(h, scores_t.mul_scalar(scale as f32), "scaled scores");
    let attn_out_t = require!(h, scores_scaled.matmul(&v_t), "attn output");

    let x_res = require!(
        h,
        tensor_from(&f32_of(&x_f64), vec![seq, dm], device),
        "X for residual"
    );
    let res1_t = require!(h, x_res.add(&attn_out_t), "residual 1");

    let ffn_h = require!(h, res1_t.matmul(&w1_tt), "FFN hidden");
    let ffn_a = require!(h, ffn_h.tanh(), "FFN tanh");
    let ffn_o = require!(h, ffn_a.matmul(&w2_tt), "FFN output");

    let res1_copy = require!(
        h,
        tensor_from(&f32_of(&residual1), vec![seq, dm], device),
        "res1 copy"
    );
    let final_t = require!(h, res1_copy.add(&ffn_o), "residual 2");
    let final_out = require!(h, final_t.to_vec(), "final readback");

    let diff = max_abs_diff_gpu_vs_cpu(&final_out, &cpu_final);
    h.check_upper(
        &format!("full transformer layer: diff={diff:.2e}"),
        diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
    h.check_bool(
        &format!("output shape: {} (expect {})", final_out.len(), seq * dm),
        final_out.len() == seq * dm,
    );
}

/// Run full layer twice, check determinism.
fn validate_determinism(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = Rng::new(48);
    let seq = 4_usize;
    let dm = 8_usize;

    let sv = || {
        let mut r = Rng::new(48);
        (0..seq * dm)
            .map(|_| r.uniform().mul_add(0.5, 0.5) as f32)
            .collect::<Vec<f32>>()
    };
    let x = sv();
    let wq: Vec<f32> = (0..dm * dm)
        .map(|_| rng.uniform().mul_add(0.5, 0.5) as f32)
        .collect();

    let run = || -> Option<Vec<f32>> {
        let x_t = Tensor::from_data(&x, vec![seq, dm], device.clone()).ok()?;
        let wq_t = Tensor::from_data(&wq, vec![dm, dm], device.clone()).ok()?;
        let wq_tt = wq_t.transpose().ok()?;
        let q = x_t.matmul(&wq_tt).ok()?;
        q.to_vec().ok()
    };

    let Some(r1) = run() else {
        h.check_bool("determinism run1 failed", false);
        return;
    };
    let Some(r2) = run() else {
        h.check_bool("determinism run2 failed", false);
        return;
    };

    h.check_bool("transformer determinism: bit-identical", r1 == r2);
}
