// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` bC/gT validator for Exp 004 (Transfer Learning).
//!
//! Exercises `barracuda::tensor` for a transfer learning MLP forward pass
//! and domain adaptation (source vs target distributions).
//!
//! ## Provenance
//!
//! Python baseline: `control/transfer/transfer_learning.py`
//! Rust baseline: `validate_transfer`

#![expect(
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    reason = "validation binary"
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::metrics::{r_squared, rmse};
use neural_spring::require;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{max_abs_diff_gpu_vs_cpu, ValidationHarness};
use std::sync::Arc;

type Dev = Arc<WgpuDevice>;

fn t(data: &[f32], shape: Vec<usize>, device: &Dev) -> Result<Tensor, String> {
    Tensor::from_data(data, shape, device.clone()).map_err(|e| e.to_string())
}

/// CPU A × B^T reference for MLP forward.
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
        neural_spring::validation::exit_no_gpu();
    };
    println!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );
    let device: Dev = gpu.wgpu_device().clone();
    let harness_name = format!("barracuda_transfer[{}]", gpu.adapter_name);
    let mut h = ValidationHarness::new(&harness_name);

    validate_mlp_forward(&mut h, &device);
    validate_domain_adaptation(&mut h, &device);
    validate_metrics_parity(&mut h);
    validate_determinism(&mut h, &device);

    h.finish();
}

/// Build transfer MLP: input(6) → FC1(6→32) → tanh → FC2(32→1).
fn validate_mlp_forward(h: &mut ValidationHarness, device: &Dev) {
    const IN: usize = 6;
    const H1: usize = 32;
    const OUT: usize = 1;

    let mut rng = Rng::new(42);
    let input: Vec<f64> = (0..IN).map(|_| rng.uniform() * 0.5 + 0.5).collect();
    let w1: Vec<f64> = (0..(H1 * IN)).map(|_| rng.uniform() * 0.5 + 0.5).collect();
    let w2: Vec<f64> = (0..(OUT * H1)).map(|_| rng.uniform() * 0.5 + 0.5).collect();

    let h1_linear = cpu_matmul_a_bt(&input, (1, IN), &w1, (H1, IN));
    let h1_tanh: Vec<f64> = h1_linear.iter().map(|&x| x.tanh()).collect();
    let cpu_out = cpu_matmul_a_bt(&h1_tanh, (1, H1), &w2, (OUT, H1));

    let input_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();
    let w1_f32: Vec<f32> = w1.iter().map(|&x| x as f32).collect();
    let w2_f32: Vec<f32> = w2.iter().map(|&x| x as f32).collect();

    let inp_t = require!(h, t(&input_f32, vec![1, IN], device), "create input tensor");
    let w1_t = require!(h, t(&w1_f32, vec![H1, IN], device), "create W1 tensor");
    let w1_t_t = require!(h, w1_t.transpose(), "W1 transpose");
    let h1_linear_t = require!(
        h,
        inp_t.matmul(&w1_t_t).map_err(|e| e.to_string()),
        "FC1 matmul"
    );
    let h1_tanh_t = require!(h, h1_linear_t.tanh().map_err(|e| e.to_string()), "FC1 tanh");

    let h1_out = require!(h, h1_tanh_t.to_vec(), "readback FC1");
    let diff_fc1 = max_abs_diff_gpu_vs_cpu(&h1_out, &h1_tanh);
    h.check_upper(
        "FC1 accuracy",
        diff_fc1,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    let w2_t = require!(h, t(&w2_f32, vec![OUT, H1], device), "create W2 tensor");
    let w2_t_t = require!(h, w2_t.transpose(), "W2 transpose");
    let h2_linear_t = require!(
        h,
        h1_tanh_t.matmul(&w2_t_t).map_err(|e| e.to_string()),
        "FC2 matmul"
    );
    let gpu_out = require!(h, h2_linear_t.to_vec(), "readback FC2");

    let diff_fc2 = max_abs_diff_gpu_vs_cpu(&gpu_out, &cpu_out);
    h.check_upper("FC2 accuracy", diff_fc2, tolerances::BARRACUDA_GPU_ECO_F32);
}

/// Run same MLP with source and target domain inputs; verify GPU matches CPU for both.
fn validate_domain_adaptation(h: &mut ValidationHarness, device: &Dev) {
    const IN: usize = 6;
    const H1: usize = 32;
    const OUT: usize = 1;

    let mut rng = Rng::new(123);
    let w1: Vec<f64> = (0..(H1 * IN)).map(|_| rng.uniform() * 0.5 + 0.5).collect();
    let w2: Vec<f64> = (0..(OUT * H1)).map(|_| rng.uniform() * 0.5 + 0.5).collect();
    let w1_f32: Vec<f32> = w1.iter().map(|&x| x as f32).collect();
    let w2_f32: Vec<f32> = w2.iter().map(|&x| x as f32).collect();

    let w1_t = require!(h, t(&w1_f32, vec![H1, IN], device), "W1 for domain");
    let w2_t = require!(h, t(&w2_f32, vec![OUT, H1], device), "W2 for domain");
    let w1_t_t = require!(h, w1_t.transpose(), "W1 transpose");
    let w2_t_t = require!(h, w2_t.transpose(), "W2 transpose");

    let forward = |input: &[f64]| -> Vec<f64> {
        let h1 = cpu_matmul_a_bt(input, (1, IN), &w1, (H1, IN));
        let h1_tanh: Vec<f64> = h1.iter().map(|&x| x.tanh()).collect();
        cpu_matmul_a_bt(&h1_tanh, (1, H1), &w2, (OUT, H1))
    };

    let run_gpu = |input: &[f32]| -> Option<Vec<f32>> {
        let inp_t = Tensor::from_data(input, vec![1, IN], device.clone()).ok()?;
        let h1 = inp_t.matmul(&w1_t_t).ok()?;
        let h1a = h1.tanh().ok()?;
        let out = h1a.matmul(&w2_t_t).ok()?;
        out.to_vec().ok()
    };

    let source: Vec<f64> = (0..IN).map(|_| rng.uniform() * 0.5 + 0.5).collect();
    let target: Vec<f64> = (0..IN).map(|_| rng.uniform() * 0.3 + 0.6).collect();

    let cpu_source = forward(&source);
    let cpu_target = forward(&target);

    let source_f32: Vec<f32> = source.iter().map(|&x| x as f32).collect();
    let target_f32: Vec<f32> = target.iter().map(|&x| x as f32).collect();

    let Some(gpu_source) = run_gpu(&source_f32) else {
        h.check_bool("domain adaptation: source GPU run", false);
        return;
    };
    let Some(gpu_target) = run_gpu(&target_f32) else {
        h.check_bool("domain adaptation: target GPU run", false);
        return;
    };

    let diff_source = max_abs_diff_gpu_vs_cpu(&gpu_source, &cpu_source);
    let diff_target = max_abs_diff_gpu_vs_cpu(&gpu_target, &cpu_target);

    h.check_upper(
        "domain adaptation: source GPU matches CPU",
        diff_source,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
    h.check_upper(
        "domain adaptation: target GPU matches CPU",
        diff_target,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

/// Pure math: R² and RMSE on output vs reference.
fn validate_metrics_parity(h: &mut ValidationHarness) {
    let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    h.check_abs(
        "R² perfect = 1.0",
        r_squared(&y_true, &y_pred),
        1.0,
        tolerances::METRIC_EXACT,
    );
    h.check_abs(
        "RMSE perfect = 0",
        rmse(&y_true, &y_pred),
        0.0,
        tolerances::METRIC_EXACT,
    );
}

/// Run forward twice, check bit-identical.
fn validate_determinism(h: &mut ValidationHarness, device: &Dev) {
    const IN: usize = 6;
    const H1: usize = 32;
    const OUT: usize = 1;

    let mut rng = Rng::new(77);
    let input: Vec<f32> = (0..IN)
        .map(|_| (rng.uniform() * 0.5 + 0.5) as f32)
        .collect();
    let w1: Vec<f32> = (0..(H1 * IN))
        .map(|_| (rng.uniform() * 0.5 + 0.5) as f32)
        .collect();
    let w2: Vec<f32> = (0..(OUT * H1))
        .map(|_| (rng.uniform() * 0.5 + 0.5) as f32)
        .collect();

    let run = || -> Option<Vec<f32>> {
        let i = Tensor::from_data(&input, vec![1, IN], device.clone()).ok()?;
        let w1t = Tensor::from_data(&w1, vec![H1, IN], device.clone()).ok()?;
        let w1tt = w1t.transpose().ok()?;
        let h1 = i.matmul(&w1tt).ok()?;
        let h1a = h1.tanh().ok()?;
        let w2t = Tensor::from_data(&w2, vec![OUT, H1], device.clone()).ok()?;
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
