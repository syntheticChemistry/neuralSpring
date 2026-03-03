// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: `BarraCUDA` unified Tensor API (WGSL shaders via `wgpu`).
//!
//! Proves that the same WGSL shaders produce identical results on **any** hardware.
//! `BarraCUDA` is the unified math; `ToadStool` (`wgpu`) runs it on GPU, CPU, or NPU.
//!
//! ## Backend selection
//!
//! Set `NEURALSPRING_BACKEND` to control the `wgpu` adapter:
//!
//! | Value | Behavior |
//! |-------|----------|
//! | `auto` (default) | Best available (GPU → CPU software fallback) |
//! | `cpu` | Force CPU software rasterizer (lavapipe / llvmpipe) |
//! | `gpu` | Force discrete / integrated GPU |
//!
//! Run all three to prove the math is universal:
//! ```text
//! NEURALSPRING_BACKEND=cpu  cargo run --bin validate_barracuda_tensor
//! NEURALSPRING_BACKEND=gpu  cargo run --bin validate_barracuda_tensor
//! NEURALSPRING_BACKEND=auto cargo run --bin validate_barracuda_tensor
//! ```
//!
//! ## Provenance
//!
//! Expected values: analytical formulas cross-validated with `PyTorch` 2.2.

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::require;
use neural_spring::tolerances;
use neural_spring::validation::{check_gpu_points, ValidationHarness};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let Ok(gpu) = Gpu::new().await else {
        neural_spring::validation::exit_no_gpu();
    };
    eprintln!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );
    let device = gpu.wgpu_device().clone();

    let harness_name = format!("barracuda_tensor[{}]", gpu.adapter_name);
    let mut h = ValidationHarness::new(&harness_name);

    validate_relu(&mut h, &device);
    validate_gelu(&mut h, &device);
    validate_sigmoid(&mut h, &device);
    validate_softmax(&mut h, &device);
    validate_layer_norm(&mut h, &device);
    validate_matmul(&mut h, &device);
    validate_arithmetic(&mut h, &device);
    validate_mse_loss(&mut h, &device);

    validate_tanh(&mut h, &device);
    validate_exp_log_sqrt(&mut h, &device);
    validate_scalar_ops(&mut h, &device);
    validate_div(&mut h, &device);
    validate_reductions(&mut h, &device);
    validate_activations_extended(&mut h, &device);
    validate_losses_extended(&mut h, &device);
    validate_transpose(&mut h, &device);

    validate_log_softmax(&mut h, &device);
    validate_leaky_relu(&mut h, &device);
    validate_elu(&mut h, &device);

    h.finish();
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn tensor(
    data: &[f32],
    shape: Vec<usize>,
    device: &Arc<WgpuDevice>,
) -> Result<Tensor, barracuda::error::BarracudaError> {
    Tensor::from_data(data, shape, device.clone())
}

fn readback(t: &Tensor) -> Result<Vec<f32>, barracuda::error::BarracudaError> {
    t.to_vec()
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
            // Previous test used 3.0 which is ~0.004 away — outside 1e-3 tol.
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
    let lhs = require!(
        h,
        tensor(&[1.0, 2.0, 3.0, 4.0], vec![4], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    let rhs = require!(
        h,
        tensor(&[5.0, 6.0, 7.0, 8.0], vec![4], device),
        "Tensor::from_data: GPU buffer alloc"
    );

    match lhs.add(&rhs) {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let tol = tolerances::TENSOR_EXACT_F32;
            check_gpu_points(
                h,
                &v,
                &[("add [0] = 6", 0, 6.0, tol), ("add [3] = 12", 3, 12.0, tol)],
            );
        }
        Err(e) => h.check_bool(&format!("add [ERROR: {e}]"), false),
    }

    let lhs2 = require!(
        h,
        tensor(&[10.0, 20.0, 30.0, 40.0], vec![4], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    let rhs2 = require!(
        h,
        tensor(&[1.0, 2.0, 3.0, 4.0], vec![4], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match lhs2.sub(&rhs2) {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let tol = tolerances::TENSOR_EXACT_F32;
            check_gpu_points(
                h,
                &v,
                &[("sub [0] = 9", 0, 9.0, tol), ("sub [3] = 36", 3, 36.0, tol)],
            );
        }
        Err(e) => h.check_bool(&format!("sub [ERROR: {e}]"), false),
    }

    let lhs3 = require!(
        h,
        tensor(&[2.0, 3.0, 4.0, 5.0], vec![4], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    let rhs3 = require!(
        h,
        tensor(&[10.0, 20.0, 30.0, 40.0], vec![4], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match lhs3.mul(&rhs3) {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let tol = tolerances::TENSOR_EXACT_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("mul [0] = 20", 0, 20.0, tol),
                    ("mul [3] = 200", 3, 200.0, tol),
                ],
            );
        }
        Err(e) => h.check_bool(&format!("mul [ERROR: {e}]"), false),
    }
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

// ── Tanh ────────────────────────────────────────────────────────────────

fn validate_tanh(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let input = require!(
        h,
        tensor(&[-10.0, -1.0, 0.0, 1.0, 10.0], vec![5], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match input.tanh() {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            h.check_abs(
                "tanh(0) == 0",
                f64::from(v[2]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
            let tol = tolerances::TENSOR_TRANSCENDENTAL_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("tanh(1) ≈ 0.7616", 3, 0.7616, tol),
                    ("tanh(-10) ≈ -1", 0, -1.0, tol),
                    ("tanh(10) ≈ 1", 4, 1.0, tol),
                ],
            );
            h.check_abs(
                "tanh antisymmetry",
                f64::from(v[1]) + f64::from(v[3]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("tanh [ERROR: {e}]"), false),
    }
}

// ── Exp / Log / Sqrt ────────────────────────────────────────────────────

fn validate_exp_log_sqrt(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let exp_input = require!(
        h,
        tensor(&[0.0, 1.0, 2.0, -1.0], vec![4], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match exp_input.exp_wgsl() {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let ttf = tolerances::TENSOR_TRANSCENDENTAL_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("exp(0) == 1", 0, 1.0, tolerances::TENSOR_EXACT_F32),
                    ("exp(1) ≈ e", 1, std::f64::consts::E, ttf),
                    ("exp(-1) ≈ 1/e", 3, 1.0 / std::f64::consts::E, ttf),
                ],
            );
        }
        Err(e) => h.check_bool(&format!("exp_wgsl [ERROR: {e}]"), false),
    }

    let log_input = require!(
        h,
        tensor(&[1.0, std::f32::consts::E, 10.0], vec![3], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match log_input.log_wgsl() {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let ttf = tolerances::TENSOR_TRANSCENDENTAL_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("log(1) == 0", 0, 0.0, tolerances::TENSOR_EXACT_F32),
                    ("log(e) ≈ 1", 1, 1.0, ttf),
                    ("log(10) ≈ 2.3026", 2, 10.0_f64.ln(), ttf),
                ],
            );
        }
        Err(e) => h.check_bool(&format!("log_wgsl [ERROR: {e}]"), false),
    }

    let sqrt_input = require!(
        h,
        tensor(&[0.0, 1.0, 4.0, 9.0, 16.0], vec![5], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match sqrt_input.sqrt_wgsl() {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let tol = tolerances::TENSOR_EXACT_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("sqrt(0) == 0", 0, 0.0, tol),
                    ("sqrt(4) == 2", 2, 2.0, tol),
                    ("sqrt(9) == 3", 3, 3.0, tol),
                ],
            );
        }
        Err(e) => h.check_bool(&format!("sqrt_wgsl [ERROR: {e}]"), false),
    }
}

// ── Scalar ops ──────────────────────────────────────────────────────────

fn validate_scalar_ops(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let input = require!(
        h,
        tensor(&[2.0, 4.0, 6.0, 8.0], vec![4], device),
        "Tensor::from_data: GPU buffer alloc"
    );

    match input.mul_scalar(3.0) {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let tol = tolerances::TENSOR_EXACT_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("mul_scalar [0] = 6", 0, 6.0, tol),
                    ("mul_scalar [3] = 24", 3, 24.0, tol),
                ],
            );
        }
        Err(e) => h.check_bool(&format!("mul_scalar [ERROR: {e}]"), false),
    }

    let input2 = require!(
        h,
        tensor(&[1.0, 2.0, 3.0, 4.0], vec![4], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match input2.add_scalar(10.0) {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let tol = tolerances::TENSOR_EXACT_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("add_scalar [0] = 11", 0, 11.0, tol),
                    ("add_scalar [3] = 14", 3, 14.0, tol),
                ],
            );
        }
        Err(e) => h.check_bool(&format!("add_scalar [ERROR: {e}]"), false),
    }

    let input3 = require!(
        h,
        tensor(&[10.0, 20.0, 30.0, 40.0], vec![4], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match input3.div_scalar(5.0) {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let tol = tolerances::TENSOR_EXACT_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("div_scalar [0] = 2", 0, 2.0, tol),
                    ("div_scalar [3] = 8", 3, 8.0, tol),
                ],
            );
        }
        Err(e) => h.check_bool(&format!("div_scalar [ERROR: {e}]"), false),
    }
}

// ── Element-wise div ────────────────────────────────────────────────────

fn validate_div(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let lhs = require!(
        h,
        tensor(&[10.0, 20.0, 30.0, 40.0], vec![4], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    let rhs = require!(
        h,
        tensor(&[2.0, 4.0, 5.0, 8.0], vec![4], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match lhs.div(&rhs) {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let tol = tolerances::TENSOR_EXACT_F32;
            check_gpu_points(
                h,
                &v,
                &[("div [0] = 5", 0, 5.0, tol), ("div [3] = 5", 3, 5.0, tol)],
            );
        }
        Err(e) => h.check_bool(&format!("div [ERROR: {e}]"), false),
    }
}

// ── Reductions ──────────────────────────────────────────────────────────

fn validate_reductions(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    use neural_spring::validation::{validate_tensor_reduction, ReductionExpected};
    let tex = tolerances::TENSOR_EXACT_F32;

    validate_tensor_reduction(
        h,
        device,
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[5],
        Tensor::sum,
        &ReductionExpected {
            label: "sum([1..5]) == 15",
            value: 15.0,
            tolerance: tex,
        },
    );
    validate_tensor_reduction(
        h,
        device,
        &[2.0, 4.0, 6.0, 8.0, 10.0],
        &[5],
        Tensor::mean,
        &ReductionExpected {
            label: "mean([2,4,6,8,10]) == 6",
            value: 6.0,
            tolerance: tex,
        },
    );
    validate_tensor_reduction(
        h,
        device,
        &[3.0, 1.0, 7.0, 2.0, 5.0],
        &[5],
        Tensor::max,
        &ReductionExpected {
            label: "max([3,1,7,2,5]) == 7",
            value: 7.0,
            tolerance: tex,
        },
    );
    validate_tensor_reduction(
        h,
        device,
        &[3.0, 1.0, 7.0, 2.0, 5.0],
        &[5],
        Tensor::min,
        &ReductionExpected {
            label: "min([3,1,7,2,5]) == 1",
            value: 1.0,
            tolerance: tex,
        },
    );
    validate_tensor_reduction(
        h,
        device,
        &[3.0, 4.0],
        &[2],
        Tensor::norm,
        &ReductionExpected {
            label: "norm([3,4]) == 5",
            value: 5.0,
            tolerance: tolerances::TENSOR_NORM_F32,
        },
    );
}

// ── Extended activations ────────────────────────────────────────────────

fn validate_activations_extended(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    // leaky_relu and elu bugs (S-05, S-06) now fixed upstream — tested in
    // validate_leaky_relu() and validate_elu() below.

    let input2 = require!(
        h,
        tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match input2.swish_wgsl() {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            h.check_abs(
                "swish(0) == 0",
                f64::from(v[2]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_bool("swish(2) > 0", v[4] > 0.0);
            h.check_bool(
                "swish monotonic: s(-1) < s(0) < s(1)",
                v[1] < v[2] && v[2] < v[3],
            );
        }
        Err(e) => h.check_bool(&format!("swish [ERROR: {e}]"), false),
    }

    let input4 = require!(
        h,
        tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match input4.mish_wgsl() {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            h.check_abs(
                "mish(0) == 0",
                f64::from(v[2]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_bool(
                "mish monotonic: m(-1) < m(0) < m(1)",
                v[1] < v[2] && v[2] < v[3],
            );
        }
        Err(e) => h.check_bool(&format!("mish [ERROR: {e}]"), false),
    }
}

// ── Extended losses ─────────────────────────────────────────────────────

fn validate_losses_extended(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let pred = require!(
        h,
        tensor(&[1.0, 2.0, 3.0], vec![3], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    let target = require!(
        h,
        tensor(&[4.0, 5.0, 6.0], vec![3], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match pred.mae_loss(&target) {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            h.check_abs(
                "mae([1,2,3],[4,5,6]) == 3",
                f64::from(v[0]),
                3.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("mae_loss [ERROR: {e}]"), false),
    }

    let pred2 = require!(
        h,
        tensor(&[1.0, 2.0, 3.0], vec![3], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    let target2 = require!(
        h,
        tensor(&[1.0, 2.0, 3.0], vec![3], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match pred2.huber_loss(&target2, 1.0) {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            h.check_abs(
                "huber(same, delta=1) == 0",
                f64::from(v[0]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("huber_loss [ERROR: {e}]"), false),
    }
}

// ── Transpose ───────────────────────────────────────────────────────────

fn validate_transpose(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mat = require!(
        h,
        tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match mat.transpose() {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            h.check_bool("transpose shape [3,2]", *out.shape() == [3, 2]);
            let tol = tolerances::TENSOR_EXACT_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("transpose [0,0] = 1", 0, 1.0, tol),
                    ("transpose [0,1] = 4", 1, 4.0, tol),
                    ("transpose [1,0] = 2", 2, 2.0, tol),
                ],
            );
        }
        Err(e) => h.check_bool(&format!("transpose [ERROR: {e}]"), false),
    }
}

// ── Log-softmax (fixed upstream in BarraCUDA — formerly S-04 workaround) ─

fn validate_log_softmax(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let data = [1.0_f32, 2.0, 3.0];

    let input = require!(
        h,
        tensor(&data, vec![1, 3], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match input.log_softmax_wgsl() {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            h.check_bool("log_softmax_wgsl all negative", v.iter().all(|&x| x < 0.0));

            let max_val = 3.0_f64;
            let lse = ((-2.0_f64).exp() + (-1.0_f64).exp() + 0.0_f64.exp()).ln();
            let expected: Vec<f64> = data.iter().map(|&x| f64::from(x) - max_val - lse).collect();

            let tol = tolerances::TENSOR_TRANSCENDENTAL_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("log_softmax_wgsl[0]", 0, expected[0], tol),
                    ("log_softmax_wgsl[1]", 1, expected[1], tol),
                    ("log_softmax_wgsl[2]", 2, expected[2], tol),
                ],
            );

            let log_sum: f64 = v.iter().map(|&x| f64::from(x).exp()).sum::<f64>().ln();
            h.check_abs(
                "log_softmax_wgsl logsumexp ≈ 0",
                log_sum,
                0.0,
                tolerances::TENSOR_NORM_F32,
            );
        }
        Err(e) => h.check_bool(&format!("log_softmax_wgsl [ERROR: {e}]"), false),
    }
}

// ── Activations now fixed upstream (S-05, S-06) ────────────────────────

fn validate_leaky_relu(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    use neural_spring::validation::validate_tensor_unary;
    let tol = tolerances::TENSOR_EXACT_F32;
    validate_tensor_unary(
        h,
        device,
        &[-2.0, -1.0, 0.0, 1.0, 2.0],
        &[5],
        |t| t.clone().leaky_relu_wgsl_with_slope(0.01),
        "leaky_relu",
        &[
            ("leaky_relu(-2, 0.01) ≈ -0.02", 0, -0.02, tol),
            ("leaky_relu(0) == 0", 2, 0.0, tol),
            ("leaky_relu(2) == 2", 4, 2.0, tol),
        ],
    );
}

fn validate_elu(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    use neural_spring::validation::validate_tensor_unary;
    let tex = tolerances::TENSOR_EXACT_F32;
    validate_tensor_unary(
        h,
        device,
        &[-2.0, -1.0, 0.0, 1.0, 2.0],
        &[5],
        |t| t.clone().elu_wgsl(),
        "elu",
        &[
            (
                "elu(-2, 1.0)",
                0,
                (-2.0_f64).exp_m1(),
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            ),
            ("elu(0) == 0", 2, 0.0, tex),
            ("elu(2) == 2", 4, 2.0, tex),
        ],
    );
}
