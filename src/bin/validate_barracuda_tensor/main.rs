// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: `BarraCUDA` unified Tensor API (WGSL shaders via `wgpu`).
//!
//! Proves that the same WGSL shaders produce identical results on **any** hardware.
//! `BarraCUDA` is the unified math; `ToadStool` (`wgpu`) runs it on GPU, CPU, or NPU.
//!
//! ## Backend selection
//!
//! Set `GPU_BACKEND` to control the `wgpu` adapter:
//!
//! | Value | Behavior |
//! |-------|----------|
//! | `auto` (default) | Best available (GPU → CPU software fallback) |
//! | `cpu` | Force CPU software rasterizer (lavapipe / llvmpipe) |
//! | `gpu` | Force discrete / integrated GPU |
//!
//! Run all three to prove the math is universal:
//! ```text
//! GPU_BACKEND=cpu  cargo run --bin validate_barracuda_tensor
//! GPU_BACKEND=gpu  cargo run --bin validate_barracuda_tensor
//! GPU_BACKEND=auto cargo run --bin validate_barracuda_tensor
//! ```
//!
//! ## Provenance
//!
//! Expected values: analytical formulas cross-validated with `PyTorch` 2.2.
//!
//! ## Structure
//!
//! Core validations (activations, arithmetic, matmul, losses, normalization)
//! live here. Extended validations (transcendental math, reductions, upstream-
//! fixed activations) live in `extended.rs`.

mod extended;

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::require;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, check_gpu_points};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let gpu = neural_spring::validation::gpu_or_exit().await;
    println!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );
    let device = gpu.wgpu_device().clone();

    let harness_name = format!("barracuda_tensor[{}]", gpu.adapter_name);
    let mut h = ValidationHarness::new(&harness_name);

    // Core tensor ops
    validate_relu(&mut h, &device);
    validate_gelu(&mut h, &device);
    validate_sigmoid(&mut h, &device);
    validate_softmax(&mut h, &device);
    validate_layer_norm(&mut h, &device);
    validate_matmul(&mut h, &device);
    validate_arithmetic(&mut h, &device);
    validate_mse_loss(&mut h, &device);

    // Extended: transcendental math, reductions, upstream-fixed activations
    extended::validate_tanh(&mut h, &device);
    extended::validate_exp_log_sqrt(&mut h, &device);
    extended::validate_scalar_ops(&mut h, &device);
    extended::validate_div(&mut h, &device);
    extended::validate_reductions(&mut h, &device);
    extended::validate_activations_extended(&mut h, &device);
    extended::validate_losses_extended(&mut h, &device);
    extended::validate_transpose(&mut h, &device);
    extended::validate_log_softmax(&mut h, &device);
    extended::validate_leaky_relu(&mut h, &device);
    extended::validate_elu(&mut h, &device);

    h.finish();
}

// ── Helpers ─────────────────────────────────────────────────────────────

pub(crate) fn tensor(
    data: &[f32],
    shape: Vec<usize>,
    device: &Arc<WgpuDevice>,
) -> Result<Tensor, barracuda::error::BarracudaError> {
    Tensor::from_data(data, shape, device.clone())
}

pub(crate) fn readback(t: &Tensor) -> Result<Vec<f32>, barracuda::error::BarracudaError> {
    t.to_vec()
}

pub(crate) fn check_binary_op(
    h: &mut ValidationHarness,
    device: &Arc<WgpuDevice>,
    lhs_data: &[f32],
    rhs_data: &[f32],
    op: fn(&Tensor, &Tensor) -> Result<Tensor, barracuda::error::BarracudaError>,
    name: &str,
    checks: &[(usize, f32)],
) {
    let lhs = require!(h, tensor(lhs_data, vec![lhs_data.len()], device), "alloc");
    let rhs = require!(h, tensor(rhs_data, vec![rhs_data.len()], device), "alloc");
    match op(&lhs, &rhs) {
        Ok(out) => {
            let v = require!(h, readback(&out), "readback");
            let tol = tolerances::TENSOR_EXACT_F32;
            for &(idx, expected) in checks {
                h.check_abs(
                    &format!("{name} [{idx}] = {expected}"),
                    f64::from(v[idx]),
                    f64::from(expected),
                    tol,
                );
            }
        }
        Err(e) => h.check_bool(&format!("{name} [ERROR: {e}]"), false),
    }
}

// ── Activations ─────────────────────────────────────────────────────────

fn validate_relu(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    use neural_spring::validation::validate_tensor_unary;
    let tol = tolerances::TENSOR_EXACT_F32;
    validate_tensor_unary(
        h,
        device,
        &[-2.0, -1.0, 0.0, 0.5, 1.0, 3.0],
        &[6],
        |t| t.clone().relu(),
        "relu",
        &[
            ("relu(-2) == 0", 0, 0.0, tol),
            ("relu(-1) == 0", 1, 0.0, tol),
            ("relu(0) == 0", 2, 0.0, tol),
            ("relu(0.5) == 0.5", 3, 0.5, tol),
            ("relu(1) == 1", 4, 1.0, tol),
            ("relu(3) == 3", 5, 3.0, tol),
        ],
    );
}

fn validate_gelu(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let input = require!(
        h,
        tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], vec![6], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match input.gelu_wgsl() {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            h.check_abs(
                "gelu(0) == 0",
                f64::from(v[2]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
            let tol = tolerances::TENSOR_TRANSCENDENTAL_F32;
            // True GELU(3) = 0.5*3*(1+erf(3/√2)) ≈ 2.9964 (not 3.0).
            // Provenance: scipy.special.erf → 2.996_362_607_918_227.
            check_gpu_points(
                h,
                &v,
                &[
                    ("gelu(1) ≈ 0.8412", 3, 0.8412, tol),
                    ("gelu(-2) ≈ -0.0454", 0, -0.0454, tol),
                    ("gelu(3) ≈ 2.9964", 5, 2.996_362_607_918_227, tol),
                ],
            );
            h.check_bool("gelu monotonic: g(1) < g(2)", v[3] < v[4]);
        }
        Err(e) => h.check_bool(&format!("gelu_wgsl [ERROR: {e}]"), false),
    }
}

fn validate_sigmoid(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let input = require!(
        h,
        tensor(&[-10.0, -1.0, 0.0, 1.0, 10.0], vec![5], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match input.sigmoid() {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let tex = tolerances::TENSOR_EXACT_F32;
            let ttf = tolerances::TENSOR_TRANSCENDENTAL_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("sigmoid(0) == 0.5", 2, 0.5, tex),
                    ("sigmoid(-10) ≈ 0", 0, 0.0, ttf),
                    ("sigmoid(10) ≈ 1", 4, 1.0, ttf),
                ],
            );
            h.check_abs(
                "sigmoid symmetry",
                f64::from(v[1]) + f64::from(v[3]),
                1.0,
                tex,
            );
        }
        Err(e) => h.check_bool(&format!("sigmoid [ERROR: {e}]"), false),
    }
}

fn validate_softmax(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let input = require!(
        h,
        tensor(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match input.softmax() {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let sum: f64 = v.iter().map(|&x| f64::from(x)).sum();
            h.check_abs("softmax sums to 1", sum, 1.0, tolerances::TENSOR_EXACT_F32);
            h.check_bool("softmax ordering: s[0] < s[4]", v[0] < v[4]);
            h.check_bool("softmax all positive", v.iter().all(|&x| x > 0.0));
            let lse: f64 =
                (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp() + 4.0_f64.exp() + 5.0_f64.exp())
                    .ln();
            let expected_last = (5.0_f64 - lse).exp();
            h.check_abs(
                "softmax[4] analytical",
                f64::from(v[4]),
                expected_last,
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
        }
        Err(e) => h.check_bool(&format!("softmax [ERROR: {e}]"), false),
    }
}

// ── Normalization ───────────────────────────────────────────────────────

fn validate_layer_norm(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let data = [1.0_f32, 2.0, 3.0, 4.0];
    let input = require!(
        h,
        tensor(&data, vec![1, 4], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    #[expect(clippy::cast_possible_truncation, reason = "validation binary")]
    let eps = tolerances::LAYER_NORM_EPS as f32;

    match input.layer_norm_wgsl(eps) {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let mean = 2.5_f64;
            let var = [1.0, 2.0, 3.0, 4.0]
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / 4.0;
            let std = (var + f64::from(eps)).sqrt();

            let tol = tolerances::TENSOR_NORM_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("layer_norm[0]", 0, (1.0 - mean) / std, tol),
                    ("layer_norm[1]", 1, (2.0 - mean) / std, tol),
                    ("layer_norm[3]", 3, (4.0 - mean) / std, tol),
                ],
            );

            let out_mean: f64 = v.iter().map(|&x| f64::from(x)).sum::<f64>() / 4.0;
            h.check_abs(
                "layer_norm zero-mean",
                out_mean,
                0.0,
                tolerances::TENSOR_NORM_F32,
            );
        }
        Err(e) => h.check_bool(&format!("layer_norm_wgsl [ERROR: {e}]"), false),
    }
}

// ── Arithmetic ──────────────────────────────────────────────────────────

fn validate_arithmetic(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    check_binary_op(
        h,
        device,
        &[1.0, 2.0, 3.0, 4.0],
        &[5.0, 6.0, 7.0, 8.0],
        Tensor::add,
        "add",
        &[(0, 6.0), (3, 12.0)],
    );
    check_binary_op(
        h,
        device,
        &[10.0, 20.0, 30.0, 40.0],
        &[1.0, 2.0, 3.0, 4.0],
        Tensor::sub,
        "sub",
        &[(0, 9.0), (3, 36.0)],
    );
    check_binary_op(
        h,
        device,
        &[2.0, 3.0, 4.0, 5.0],
        &[10.0, 20.0, 30.0, 40.0],
        Tensor::mul,
        "mul",
        &[(0, 20.0), (3, 200.0)],
    );
}

// ── MatMul ──────────────────────────────────────────────────────────────

fn validate_matmul(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mat_a = require!(
        h,
        tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    let mat_b = require!(
        h,
        tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], device),
        "Tensor::from_data: GPU buffer alloc"
    );

    match mat_a.matmul(&mat_b) {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let tol = tolerances::TENSOR_MATMUL_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("matmul [0,0] = 58", 0, 58.0, tol),
                    ("matmul [0,1] = 64", 1, 64.0, tol),
                    ("matmul [1,0] = 139", 2, 139.0, tol),
                    ("matmul [1,1] = 154", 3, 154.0, tol),
                ],
            );
        }
        Err(e) => h.check_bool(&format!("matmul [ERROR: {e}]"), false),
    }

    let identity = require!(
        h,
        tensor(&[1.0, 0.0, 0.0, 1.0], vec![2, 2], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    let vec_x = require!(
        h,
        tensor(&[3.0, 7.0], vec![2, 1], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match identity.matmul(&vec_x) {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let tol = tolerances::TENSOR_MATMUL_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("I @ x [0] = 3", 0, 3.0, tol),
                    ("I @ x [1] = 7", 1, 7.0, tol),
                ],
            );
        }
        Err(e) => h.check_bool(&format!("matmul identity [ERROR: {e}]"), false),
    }
}

// ── Losses ──────────────────────────────────────────────────────────────

fn validate_mse_loss(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let pred = require!(
        h,
        tensor(&[1.0, 2.0, 3.0], vec![3], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    let target = require!(
        h,
        tensor(&[1.0, 2.0, 3.0], vec![3], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match pred.mse_loss(target) {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            h.check_abs(
                "mse(same) == 0",
                f64::from(v[0]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("mse_loss [ERROR: {e}]"), false),
    }

    let pred2 = require!(
        h,
        tensor(&[1.0, 2.0, 3.0], vec![3], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    let target2 = require!(
        h,
        tensor(&[4.0, 5.0, 6.0], vec![3], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match pred2.mse_loss(target2) {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            h.check_abs(
                "mse([1,2,3],[4,5,6]) == 9",
                f64::from(v[0]),
                9.0,
                tolerances::TENSOR_MATMUL_F32,
            );
        }
        Err(e) => h.check_bool(&format!("mse_loss known [ERROR: {e}]"), false),
    }
}
