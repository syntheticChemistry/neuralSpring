// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU port: `DeepONet` operator learning (Study 002).
//!
//! Validates that `barracuda::tensor` ops reproduce the `DeepONet`
//! branch-trunk inference from `deeponet.rs`.
//!
//! Evolution path:
//! ```text
//! Python (PyTorch) → Rust (hand-rolled MLP + dot product)
//!   → BarraCUDA CPU (barracuda::tensor matmul + dot)
//!   → BarraCUDA GPU (gemm_f64.wgsl + sum_reduce)
//! ```
//!
//! ## Provenance
//!
//! Python baseline: `control/deeponet/deeponet_antideriv.py`
//! Rust baseline: `validate_deeponet` (17/17 PASS)

#![allow(clippy::cast_precision_loss)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::deeponet::{
    eval_polynomial, exact_antiderivative, l2_relative_error, linspace, rmse,
};
use neural_spring::gpu::Gpu;
use neural_spring::require;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

type Dev = Arc<WgpuDevice>;

fn t(data: &[f32], shape: Vec<usize>, device: &Dev) -> Result<Tensor, String> {
    Tensor::from_data(data, shape, device.clone()).map_err(|e| e.to_string())
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

    let harness_name = format!("barracuda_deeponet[{}]", gpu.adapter_name);
    let mut h = ValidationHarness::new(&harness_name);

    validate_antiderivative_cross(&mut h);
    validate_dot_product_barracuda(&mut h, &device);
    validate_polynomial_cross(&mut h);
    validate_error_metrics(&mut h);

    h.finish();
}

/// Cross-validate exact antiderivative (pure math).
fn validate_antiderivative_cross(h: &mut ValidationHarness) {
    let y = linspace(0.0, 1.0, 50);

    let g_const = exact_antiderivative(&[1.0], &y);
    let max_err: f64 = g_const
        .iter()
        .zip(y.iter())
        .map(|(&g, &yi)| (g - yi).abs())
        .fold(0.0_f64, f64::max);
    h.check_abs(
        "∫1 dy = y",
        max_err,
        0.0,
        tolerances::DEEPONET_EXACT_ANTIDERIV,
    );

    let g_linear = exact_antiderivative(&[0.0, 1.0], &y);
    let max_err2: f64 = g_linear
        .iter()
        .zip(y.iter())
        .map(|(&g, &yi)| (g - yi.powi(2) / 2.0).abs())
        .fold(0.0_f64, f64::max);
    h.check_abs(
        "∫x dy = y²/2",
        max_err2,
        0.0,
        tolerances::DEEPONET_EXACT_ANTIDERIV,
    );

    let g_quad = exact_antiderivative(&[0.0, 0.0, 1.0], &y);
    let max_err3: f64 = g_quad
        .iter()
        .zip(y.iter())
        .map(|(&g, &yi)| (g - yi.powi(3) / 3.0).abs())
        .fold(0.0_f64, f64::max);
    h.check_abs(
        "∫x² dy = y³/3",
        max_err3,
        0.0,
        tolerances::DEEPONET_EXACT_ANTIDERIV,
    );
}

/// Validate branch-trunk dot product through `BarraCUDA` tensors.
fn validate_dot_product_barracuda(h: &mut ValidationHarness, device: &Dev) {
    #[allow(clippy::cast_possible_truncation)]
    let branch: Vec<f32> = [1.0_f64, 2.0, 3.0, 4.0].iter().map(|&v| v as f32).collect();
    #[allow(clippy::cast_possible_truncation)]
    let trunk: Vec<f32> = [4.0_f64, 3.0, 2.0, 1.0].iter().map(|&v| v as f32).collect();

    let b = require!(h, t(&branch, vec![1, 4], device), "create branch tensor");
    let tr = require!(h, t(&trunk, vec![4, 1], device), "create trunk tensor");

    let dot = require!(h, b.matmul(&tr).map_err(|e| e.to_string()), "dot product");
    let result = require!(h, dot.to_vec(), "readback");

    h.check_abs(
        "barracuda branch·trunk",
        f64::from(result[0]),
        20.0,
        tolerances::TENSOR_EXACT_F32,
    );
}

/// Cross-validate polynomial evaluation (pure math).
fn validate_polynomial_cross(h: &mut ValidationHarness) {
    let x_pts = linspace(0.0, 1.0, 5);
    let poly = eval_polynomial(&[1.0, 2.0, 3.0], &x_pts);

    h.check_abs(
        "p(0) = 1",
        poly[0],
        1.0,
        tolerances::DEEPONET_POLYNOMIAL_EXACT,
    );
    h.check_abs(
        "p(1) = 6",
        poly[x_pts.len() - 1],
        6.0,
        tolerances::DEEPONET_POLYNOMIAL_EXACT,
    );
}

/// Validate error metrics (L2 relative, RMSE).
fn validate_error_metrics(h: &mut ValidationHarness) {
    let a = [1.0, 2.0, 3.0];
    h.check_abs(
        "L2 error (perfect)",
        l2_relative_error(&a, &a),
        0.0,
        tolerances::EXACT_F64,
    );
    h.check_abs("RMSE (perfect)", rmse(&a, &a), 0.0, tolerances::EXACT_F64);

    let b = [1.1, 2.1, 3.1];
    h.check_bool("L2 > 0 imperfect", l2_relative_error(&b, &a) > 0.0);
}
