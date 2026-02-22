// SPDX-License-Identifier: AGPL-3.0-or-later

//! BarraCUDA Tensor validation: LeNet-5 CNN (Study 003).
//!
//! Tests LeNet-5 FC layer forward pass using barracuda Tensor matmul + tanh.
//! FC1: 120→84 (tanh), FC2: 84→10 (logits). Uses A×B^T pattern and positive-only data.
//!
//! ## S-14 workaround
//!
//! Uses A × B^T pattern: transpose weights before matmul.
//!
//! ## S-15 workaround
//!
//! All data uses `rng.uniform()` (positive-only).
//!
//! ## Provenance
//!
//! Python baseline: `control/lenet/lenet_mnist.py`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::manual_let_else,
    clippy::similar_names,
    clippy::single_match_else,
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
