// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU validator for Exp 001 (Neural Surrogate).
//!
//! Exercises `barracuda::tensor` Tensor ops for a simple MLP forward pass
//! that approximates benchmark functions (Rastrigin, Rosenbrock, Ackley).
//!
//! ## Provenance
//!
//! Python baseline: `control/surrogate/surrogate_validation.py`
//! Rust baseline: `validate_surrogate`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::require;
use neural_spring::rng::Rng;
use neural_spring::surrogate::{ackley_2d, rastrigin_2d, rosenbrock_2d};
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
        eprintln!("  0/0 checks — skipping gracefully");
        std::process::exit(0);
    };
    eprintln!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );
    let device: Dev = gpu.wgpu_device().clone();
    let harness_name = format!("barracuda_surrogate[{}]", gpu.adapter_name);
    let mut h = ValidationHarness::new(&harness_name);

    validate_benchmark_functions(&mut h);
    validate_mlp_forward(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

/// Pure math check (no GPU): benchmark function global minima.
fn validate_benchmark_functions(h: &mut ValidationHarness) {
    h.check_abs(
        "Rastrigin(0,0) == 0",
        rastrigin_2d(0.0, 0.0),
        0.0,
        tolerances::BENCHMARK_GLOBAL_MIN,
    );
    h.check_abs(
        "Rosenbrock(1,1) == 0",
        rosenbrock_2d(1.0, 1.0),
        0.0,
        tolerances::BENCHMARK_GLOBAL_MIN,
    );
    h.check_abs(
        "Ackley(0,0) == 0",
        ackley_2d(0.0, 0.0),
        0.0,
        tolerances::BENCHMARK_GLOBAL_MIN,
    );
}

/// Build 2-layer MLP: input(4) → FC1(4→16) → tanh → FC2(16→4).
fn validate_mlp_forward(h: &mut ValidationHarness, device: &Dev) {
    let mut rng = Rng::new(42);
    const IN: usize = 4;
    const H1: usize = 16;
    const OUT: usize = 4;

    // S-15 safe: values in [0.5, 1.0)
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

    h.check_bool("all outputs finite", gpu_out.iter().all(|&x| x.is_finite()));
}

/// Run MLP forward twice with same data, check bit-identical.
fn validate_determinism(h: &mut ValidationHarness, device: &Dev) {
    let mut rng = Rng::new(99);
    const IN: usize = 4;
    const H1: usize = 16;
    const OUT: usize = 4;

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
