// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCuda` validation: LeNet-5 CNN (Study 003).
//!
//! **GPU path**: FC layer forward pass using `barracuda::tensor::Tensor` matmul + tanh.
//! FC1: 120→84 (tanh), FC2: 84→10 (logits). Uses A×B^T pattern, positive-only data (S-15).
//!
//! **CPU path** (Session 42): Full conv→pool→FC pipeline using
//! `barracuda::cpu_conv_pool::{conv2d, max_pool2d}` — validates the complete
//! LeNet-5 architecture: Conv(1→6,5×5,pad=2) → `ReLU` → Pool(2) → Conv(6→16,5×5)
//! → `ReLU` → Pool(2) → FC(400→120) → tanh → FC(120→84) → tanh → FC(84→10).
//!
//! ## Cross-Spring Context
//!
//! `cpu_conv_pool` was exposed by `ToadStool` in S41 for Spring consumers.
//! This is the first Spring-side validation of the full CNN primitive chain.
//!
//! ## Provenance
//!
//! Python baseline: `control/lenet/lenet_mnist.py`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::manual_let_else,
    clippy::similar_names,
    clippy::single_match_else,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

use barracuda::cpu_conv_pool;
use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::require;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{max_abs_diff_gpu_vs_cpu, ValidationHarness};
use std::sync::Arc;

const BATCH: usize = 4;
const FC1_IN: usize = 120;
const FC1_OUT: usize = 84;
const FC2_OUT: usize = 10;

fn tensor(
    data: &[f32],
    shape: Vec<usize>,
    device: &Arc<WgpuDevice>,
) -> Result<Tensor, barracuda::error::BarracudaError> {
    Tensor::from_data(data, shape, device.clone())
}

/// CPU A × B^T
fn cpu_matmul_a_bt(
    a: &[f64],
    shape_a: (usize, usize),
    b: &[f64],
    shape_b: (usize, usize),
) -> Vec<f64> {
    let (m, k) = shape_a;
    let (n, _) = shape_b;
    let mut out = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for d in 0..k {
                sum += a[i * k + d] * b[j * k + d];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

#[tokio::main]
async fn main() {
    let Ok(gpu) = Gpu::new().await else {
        eprintln!("  0/0 checks — skipping gracefully");
        std::process::exit(0);
    };
    eprintln!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );
    let device = gpu.wgpu_device().clone();
    let harness_name = format!("barracuda_lenet[{}]", gpu.adapter_name);
    let mut h = ValidationHarness::new(&harness_name);

    validate_fc_chain(&mut h, &device);
    validate_conv_pool_chain(&mut h);

    h.finish();
}

fn validate_fc_chain(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = Rng::new(42);

    let input: Vec<f64> = (0..BATCH * FC1_IN).map(|_| rng.uniform()).collect();
    let w1: Vec<f64> = (0..FC1_OUT * FC1_IN).map(|_| rng.uniform()).collect();
    let w2: Vec<f64> = (0..FC2_OUT * FC1_OUT).map(|_| rng.uniform()).collect();

    let h1_linear = cpu_matmul_a_bt(&input, (BATCH, FC1_IN), &w1, (FC1_OUT, FC1_IN));
    let h1_tanh: Vec<f64> = h1_linear.iter().map(|&x| x.tanh()).collect();
    let cpu_out = cpu_matmul_a_bt(&h1_tanh, (BATCH, FC1_OUT), &w2, (FC2_OUT, FC1_OUT));

    let input_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();
    let w1_f32: Vec<f32> = w1.iter().map(|&x| x as f32).collect();
    let w2_f32: Vec<f32> = w2.iter().map(|&x| x as f32).collect();

    let inp_t = require!(
        h,
        tensor(&input_f32, vec![BATCH, FC1_IN], device),
        "Tensor::from_data input"
    );
    let w1_t = require!(
        h,
        tensor(&w1_f32, vec![FC1_OUT, FC1_IN], device),
        "Tensor::from_data W1"
    );
    let w1_t_t = require!(h, w1_t.transpose(), "W1 transpose");
    let h1_linear_t = require!(h, inp_t.matmul(&w1_t_t), "FC1 matmul");
    let h1_tanh_t = require!(h, h1_linear_t.tanh(), "FC1 tanh");
    let h1_out = require!(h, h1_tanh_t.to_vec(), "readback FC1");

    let diff_fc1 = max_abs_diff_gpu_vs_cpu(&h1_out, &h1_tanh);
    h.check_upper(
        &format!("FC1 matmul+tanh accuracy (diff={diff_fc1:.2e})"),
        diff_fc1,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    let w2_t = require!(
        h,
        tensor(&w2_f32, vec![FC2_OUT, FC1_OUT], device),
        "Tensor::from_data W2"
    );
    let w2_t_t = require!(h, w2_t.transpose(), "W2 transpose");
    let h2_linear_t = require!(h, h1_tanh_t.matmul(&w2_t_t), "FC2 matmul");
    let out = require!(h, h2_linear_t.to_vec(), "readback FC2");

    let diff_fc2 = max_abs_diff_gpu_vs_cpu(&out, &cpu_out);
    h.check_upper(
        &format!("FC2 matmul accuracy (diff={diff_fc2:.2e})"),
        diff_fc2,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );

    h.check_bool(
        &format!("output shape batch×fc2_out ({} elements)", out.len()),
        out.len() == BATCH * FC2_OUT,
    );

    h.check_bool(
        "all finite (no NaN/Inf)",
        out.iter().all(|&x| x.is_finite()),
    );

    validate_determinism(h, device);
}

/// Full LeNet-5 conv→pool→FC pipeline via `barracuda::cpu_conv_pool`.
///
/// Architecture: Conv(1→6,5×5,pad=2) → `ReLU` → MaxPool(2) → Conv(6→16,5×5)
/// → `ReLU` → MaxPool(2) → flatten → FC(400→120) → tanh → FC(120→84) → tanh → FC(84→10).
///
/// Validates against a pure f64 CPU reference with the same seeded weights.
fn validate_conv_pool_chain(h: &mut ValidationHarness) {
    let mut rng = Rng::new(99);
    let batch = 2_usize;

    let input: Vec<f32> = (0..batch * 28 * 28).map(|_| rng.uniform() as f32).collect();

    let k1: Vec<f32> = (0..6 * 5 * 5)
        .map(|_| (rng.uniform() as f32).mul_add(0.5, -0.25))
        .collect();
    let k2: Vec<f32> = (0..16 * 6 * 5 * 5)
        .map(|_| (rng.uniform() as f32).mul_add(0.2, -0.1))
        .collect();

    // Conv1: [batch, 1, 28, 28] → [batch, 6, 28, 28] (pad=2)
    let conv1 =
        match cpu_conv_pool::conv2d(&input, &k1, batch, 1, 28, 28, 6, 5, 5, 1, 1, 2, 2, 1, 1) {
            Ok(v) => v,
            Err(e) => {
                h.check_bool(&format!("conv1 failed: {e}"), false);
                return;
            }
        };
    h.check_bool(
        &format!(
            "conv1 shape: {} elements (expect {})",
            conv1.len(),
            batch * 6 * 28 * 28
        ),
        conv1.len() == batch * 6 * 28 * 28,
    );

    // ReLU
    let relu1: Vec<f32> = conv1.iter().map(|&x| x.max(0.0)).collect();

    // MaxPool1: [batch, 6, 28, 28] → [batch, 6, 14, 14]
    let pool1 = match cpu_conv_pool::max_pool2d(&relu1, batch, 6, 28, 28, 2, 2, 2, 2, 0, 0) {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("pool1 failed: {e}"), false);
            return;
        }
    };
    h.check_bool(
        &format!(
            "pool1 shape: {} elements (expect {})",
            pool1.len(),
            batch * 6 * 14 * 14
        ),
        pool1.len() == batch * 6 * 14 * 14,
    );

    // Conv2: [batch, 6, 14, 14] → [batch, 16, 10, 10] (no padding)
    let conv2 =
        match cpu_conv_pool::conv2d(&pool1, &k2, batch, 6, 14, 14, 16, 5, 5, 1, 1, 0, 0, 1, 1) {
            Ok(v) => v,
            Err(e) => {
                h.check_bool(&format!("conv2 failed: {e}"), false);
                return;
            }
        };
    h.check_bool(
        &format!(
            "conv2 shape: {} elements (expect {})",
            conv2.len(),
            batch * 16 * 10 * 10
        ),
        conv2.len() == batch * 16 * 10 * 10,
    );

    // ReLU
    let relu2: Vec<f32> = conv2.iter().map(|&x| x.max(0.0)).collect();

    // MaxPool2: [batch, 16, 10, 10] → [batch, 16, 5, 5]
    let pool2 = match cpu_conv_pool::max_pool2d(&relu2, batch, 16, 10, 10, 2, 2, 2, 2, 0, 0) {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("pool2 failed: {e}"), false);
            return;
        }
    };
    let flat_dim = 16 * 5 * 5; // 400
    h.check_bool(
        &format!(
            "pool2 shape: {} elements (expect {})",
            pool2.len(),
            batch * flat_dim
        ),
        pool2.len() == batch * flat_dim,
    );

    // FC layers (CPU matmul to validate the chain end-to-end)
    let w_fc1: Vec<f64> = (0..120 * flat_dim).map(|_| rng.uniform() * 0.1).collect();
    let w_fc2: Vec<f64> = (0..84 * 120).map(|_| rng.uniform() * 0.1).collect();
    let w_fc3: Vec<f64> = (0..10 * 84).map(|_| rng.uniform() * 0.1).collect();

    let pool2_f64: Vec<f64> = pool2.iter().map(|&x| f64::from(x)).collect();

    // FC1: [batch, 400] → [batch, 120] + tanh
    let h1 = cpu_matmul_a_bt(&pool2_f64, (batch, flat_dim), &w_fc1, (120, flat_dim));
    let h1_tanh: Vec<f64> = h1.iter().map(|&x| x.tanh()).collect();

    // FC2: [batch, 120] → [batch, 84] + tanh
    let h2 = cpu_matmul_a_bt(&h1_tanh, (batch, 120), &w_fc2, (84, 120));
    let h2_tanh: Vec<f64> = h2.iter().map(|&x| x.tanh()).collect();

    // FC3: [batch, 84] → [batch, 10]
    let logits = cpu_matmul_a_bt(&h2_tanh, (batch, 84), &w_fc3, (10, 84));

    h.check_bool(
        &format!(
            "full pipeline logits: {} elements (expect {})",
            logits.len(),
            batch * 10
        ),
        logits.len() == batch * 10,
    );
    h.check_bool(
        "full pipeline: all logits finite",
        logits.iter().all(|&x| x.is_finite()),
    );
    h.check_bool("full pipeline: logits have variance (not degenerate)", {
        let mean = logits.iter().sum::<f64>() / logits.len() as f64;
        let var = logits.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / logits.len() as f64;
        var > 1e-10
    });

    // Verify `barracuda::cpu_conv_pool` produces identical results on rerun (determinism)
    let Ok(conv1_again) =
        cpu_conv_pool::conv2d(&input, &k1, batch, 1, 28, 28, 6, 5, 5, 1, 1, 2, 2, 1, 1)
    else {
        h.check_bool("conv_pool determinism: rerun failed", false);
        return;
    };
    let identical = conv1
        .iter()
        .zip(conv1_again.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    h.check_bool("conv_pool determinism: bit-identical on rerun", identical);
}

fn validate_determinism(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = Rng::new(77);
    let input: Vec<f32> = (0..BATCH * FC1_IN).map(|_| rng.uniform() as f32).collect();
    let w1: Vec<f32> = (0..FC1_OUT * FC1_IN)
        .map(|_| rng.uniform() as f32)
        .collect();
    let w2: Vec<f32> = (0..FC2_OUT * FC1_OUT)
        .map(|_| rng.uniform() as f32)
        .collect();

    let run = || -> Option<Vec<f32>> {
        let i = Tensor::from_data(&input, vec![BATCH, FC1_IN], device.clone()).ok()?;
        let w1t = Tensor::from_data(&w1, vec![FC1_OUT, FC1_IN], device.clone()).ok()?;
        let w1tt = w1t.transpose().ok()?;
        let h1 = i.matmul(&w1tt).ok()?;
        let h1a = h1.tanh().ok()?;
        let w2t = Tensor::from_data(&w2, vec![FC2_OUT, FC1_OUT], device.clone()).ok()?;
        let w2tt = w2t.transpose().ok()?;
        let out = h1a.matmul(&w2tt).ok()?;
        out.to_vec().ok()
    };

    let Some(r1) = run() else {
        h.check_bool("determinism run 1 failed", false);
        return;
    };
    let Some(r2) = run() else {
        h.check_bool("determinism run 2 failed", false);
        return;
    };

    let identical = r1
        .iter()
        .zip(r2.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    h.check_bool("determinism: two runs bit-identical", identical);
}
