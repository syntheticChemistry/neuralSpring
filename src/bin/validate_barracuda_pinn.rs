// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU port: PINN Burgers' equation (Study 001).
//!
//! Validates that `barracuda::tensor` ops reproduce the PINN MLP forward
//! pass and Cole-Hopf exact solution from `pinn.rs`.
//!
//! Evolution path:
//! ```text
//! Python (PyTorch autograd) → Rust (hand-rolled MLP + Cole-Hopf)
//!   → BarraCUDA CPU (barracuda::tensor matmul + tanh)
//!   → BarraCUDA GPU (gemm_f64.wgsl + elementwise tanh)
//! ```
//!
//! ## Provenance
//!
//! Python baseline: `control/pinn/pinn_burgers.py`
//! Rust baseline: `validate_pinn` (16/16 PASS)

#![allow(clippy::cast_precision_loss)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::pinn::{burgers_exact_point, max_gradient, BURGERS_NU};
use neural_spring::require;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

type Dev = Arc<WgpuDevice>;

fn t(data: &[f32], shape: Vec<usize>, device: &Dev) -> Result<Tensor, String> {
    Tensor::from_data(data, shape, device.clone()).map_err(|e| e.to_string())
}

fn readback(tensor: &Tensor) -> Result<Vec<f32>, String> {
    tensor.to_vec().map_err(|e| e.to_string())
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

    let harness_name = format!("barracuda_pinn[{}]", gpu.adapter_name);
    let mut h = ValidationHarness::new(&harness_name);

    validate_cole_hopf_cross(&mut h);
    validate_mlp_forward_barracuda(&mut h, &device);
    validate_shock_steepening(&mut h);

    h.finish();
}

/// Cross-validate Cole-Hopf exact solution (pure math, no GPU needed).
fn validate_cole_hopf_cross(h: &mut ValidationHarness) {
    let ic_points = [0.0, 0.5, -0.5, 1.0, -1.0];
    for &x in &ic_points {
        let u = burgers_exact_point(0.0, x, BURGERS_NU);
        let expected = -(std::f64::consts::PI * x).sin();
        h.check_abs(
            &format!("Cole-Hopf IC: u(0,{x})"),
            u,
            expected,
            tolerances::PINN_IC_EXACT,
        );
    }

    for &t in &[0.25, 0.5, 0.75] {
        let u_left = burgers_exact_point(t, -1.0, BURGERS_NU);
        let u_right = burgers_exact_point(t, 1.0, BURGERS_NU);
        h.check_upper(
            &format!("Cole-Hopf BC: |u({t},-1)|"),
            u_left.abs(),
            tolerances::PINN_BC_TOLERANCE,
        );
        h.check_upper(
            &format!("Cole-Hopf BC: |u({t},+1)|"),
            u_right.abs(),
            tolerances::PINN_BC_TOLERANCE,
        );
    }
}

/// Validate MLP forward pass (matmul + tanh) through `BarraCUDA` tensors.
fn validate_mlp_forward_barracuda(h: &mut ValidationHarness, device: &Dev) {
    #[allow(clippy::cast_possible_truncation)]
    let w1: Vec<f32> = [1.0_f64, 0.0, 0.0, 1.0].iter().map(|&v| v as f32).collect();
    let b1: Vec<f32> = vec![0.0, 0.0];
    let input: Vec<f32> = vec![0.5, -0.3];

    let inp = require!(h, t(&input, vec![1, 2], device), "create input tensor");
    let w = require!(h, t(&w1, vec![2, 2], device), "create weight tensor");
    let b = require!(h, t(&b1, vec![1, 2], device), "create bias tensor");

    let mm = require!(h, inp.matmul(&w).map_err(|e| e.to_string()), "matmul");
    let biased = require!(h, mm.add(&b).map_err(|e| e.to_string()), "add bias");
    let activated = require!(h, biased.tanh().map_err(|e| e.to_string()), "tanh");

    let result = require!(h, readback(&activated), "readback");

    let expected_0 = (0.5_f64).tanh();
    let expected_1 = (-0.3_f64).tanh();

    h.check_abs(
        "barracuda MLP [0]",
        f64::from(result[0]),
        expected_0,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
    h.check_abs(
        "barracuda MLP [1]",
        f64::from(result[1]),
        expected_1,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

/// Validate shock steepening (pure math cross-validation).
fn validate_shock_steepening(h: &mut ValidationHarness) {
    let nx = 128;
    let x_grid: Vec<f64> = (0..nx)
        .map(|i| 2.0f64.mul_add(f64::from(i) / f64::from(nx - 1), -1.0))
        .collect();

    let u_t0: Vec<f64> = x_grid
        .iter()
        .map(|&x| burgers_exact_point(0.0, x, BURGERS_NU))
        .collect();
    let u_t1: Vec<f64> = x_grid
        .iter()
        .map(|&x| burgers_exact_point(1.0, x, BURGERS_NU))
        .collect();

    let grad_t0 = max_gradient(&u_t0);
    let grad_t1 = max_gradient(&u_t1);

    h.check_lower(
        "shock steepening ratio",
        if grad_t0 > 0.0 {
            grad_t1 / grad_t0
        } else {
            0.0
        },
        tolerances::PINN_SHOCK_RATIO_MIN,
    );
}
