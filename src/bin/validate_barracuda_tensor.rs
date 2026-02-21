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
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

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

#[allow(clippy::expect_used)]
fn tensor(data: &[f32], shape: Vec<usize>, device: &Arc<WgpuDevice>) -> Tensor {
    Tensor::from_data(data, shape, device.clone()).expect("Tensor::from_data: GPU buffer alloc")
}

#[allow(clippy::expect_used)]
fn readback(t: &Tensor) -> Vec<f32> {
    t.to_vec().expect("tensor readback from GPU")
}

// ── Activations ─────────────────────────────────────────────────────────

fn validate_relu(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let input = tensor(&[-2.0, -1.0, 0.0, 0.5, 1.0, 3.0], vec![6], device);
    match input.relu() {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "relu(-2) == 0",
                f64::from(v[0]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "relu(-1) == 0",
                f64::from(v[1]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "relu(0) == 0",
                f64::from(v[2]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "relu(0.5) == 0.5",
                f64::from(v[3]),
                0.5,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "relu(1) == 1",
                f64::from(v[4]),
                1.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "relu(3) == 3",
                f64::from(v[5]),
                3.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("relu [ERROR: {e}]"), false),
    }
}

fn validate_gelu(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let input = tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], vec![6], device);
    match input.gelu_wgsl() {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "gelu(0) == 0",
                f64::from(v[2]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "gelu(1) ≈ 0.8412",
                f64::from(v[3]),
                0.8412,
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
            h.check_abs(
                "gelu(-2) ≈ -0.0454",
                f64::from(v[0]),
                -0.0454,
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
            h.check_abs(
                "gelu(3) ≈ 3.0",
                f64::from(v[5]),
                3.0,
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
            h.check_bool("gelu monotonic: g(1) < g(2)", v[3] < v[4]);
        }
        Err(e) => h.check_bool(&format!("gelu_wgsl [ERROR: {e}]"), false),
    }
}

fn validate_sigmoid(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let input = tensor(&[-10.0, -1.0, 0.0, 1.0, 10.0], vec![5], device);
    match input.sigmoid() {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "sigmoid(0) == 0.5",
                f64::from(v[2]),
                0.5,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "sigmoid(-10) ≈ 0",
                f64::from(v[0]),
                0.0,
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
            h.check_abs(
                "sigmoid(10) ≈ 1",
                f64::from(v[4]),
                1.0,
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
            h.check_abs(
                "sigmoid symmetry",
                f64::from(v[1]) + f64::from(v[3]),
                1.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("sigmoid [ERROR: {e}]"), false),
    }
}

fn validate_softmax(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let input = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5], device);
    match input.softmax() {
        Ok(out) => {
            let v = readback(&out);
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
    let input = tensor(&data, vec![1, 4], device);
    let eps = 1e-5_f32;

    match input.layer_norm_wgsl(eps) {
        Ok(out) => {
            let v = readback(&out);
            let mean = 2.5_f64;
            let var = [1.0, 2.0, 3.0, 4.0]
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / 4.0;
            let std = (var + f64::from(eps)).sqrt();

            h.check_abs(
                "layer_norm[0]",
                f64::from(v[0]),
                (1.0 - mean) / std,
                tolerances::TENSOR_NORM_F32,
            );
            h.check_abs(
                "layer_norm[1]",
                f64::from(v[1]),
                (2.0 - mean) / std,
                tolerances::TENSOR_NORM_F32,
            );
            h.check_abs(
                "layer_norm[3]",
                f64::from(v[3]),
                (4.0 - mean) / std,
                tolerances::TENSOR_NORM_F32,
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
    let lhs = tensor(&[1.0, 2.0, 3.0, 4.0], vec![4], device);
    let rhs = tensor(&[5.0, 6.0, 7.0, 8.0], vec![4], device);

    match lhs.add(&rhs) {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "add [0] = 6",
                f64::from(v[0]),
                6.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "add [3] = 12",
                f64::from(v[3]),
                12.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("add [ERROR: {e}]"), false),
    }

    let lhs2 = tensor(&[10.0, 20.0, 30.0, 40.0], vec![4], device);
    let rhs2 = tensor(&[1.0, 2.0, 3.0, 4.0], vec![4], device);
    match lhs2.sub(&rhs2) {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "sub [0] = 9",
                f64::from(v[0]),
                9.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "sub [3] = 36",
                f64::from(v[3]),
                36.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("sub [ERROR: {e}]"), false),
    }

    let lhs3 = tensor(&[2.0, 3.0, 4.0, 5.0], vec![4], device);
    let rhs3 = tensor(&[10.0, 20.0, 30.0, 40.0], vec![4], device);
    match lhs3.mul(&rhs3) {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "mul [0] = 20",
                f64::from(v[0]),
                20.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "mul [3] = 200",
                f64::from(v[3]),
                200.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("mul [ERROR: {e}]"), false),
    }
}

// ── MatMul ──────────────────────────────────────────────────────────────

fn validate_matmul(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mat_a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], device);
    let mat_b = tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], device);

    match mat_a.matmul(&mat_b) {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "matmul [0,0] = 58",
                f64::from(v[0]),
                58.0,
                tolerances::TENSOR_MATMUL_F32,
            );
            h.check_abs(
                "matmul [0,1] = 64",
                f64::from(v[1]),
                64.0,
                tolerances::TENSOR_MATMUL_F32,
            );
            h.check_abs(
                "matmul [1,0] = 139",
                f64::from(v[2]),
                139.0,
                tolerances::TENSOR_MATMUL_F32,
            );
            h.check_abs(
                "matmul [1,1] = 154",
                f64::from(v[3]),
                154.0,
                tolerances::TENSOR_MATMUL_F32,
            );
        }
        Err(e) => h.check_bool(&format!("matmul [ERROR: {e}]"), false),
    }

    let identity = tensor(&[1.0, 0.0, 0.0, 1.0], vec![2, 2], device);
    let vec_x = tensor(&[3.0, 7.0], vec![2, 1], device);
    match identity.matmul(&vec_x) {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "I @ x [0] = 3",
                f64::from(v[0]),
                3.0,
                tolerances::TENSOR_MATMUL_F32,
            );
            h.check_abs(
                "I @ x [1] = 7",
                f64::from(v[1]),
                7.0,
                tolerances::TENSOR_MATMUL_F32,
            );
        }
        Err(e) => h.check_bool(&format!("matmul identity [ERROR: {e}]"), false),
    }
}

// ── Losses ──────────────────────────────────────────────────────────────

fn validate_mse_loss(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let pred = tensor(&[1.0, 2.0, 3.0], vec![3], device);
    let target = tensor(&[1.0, 2.0, 3.0], vec![3], device);
    match pred.mse_loss(target) {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "mse(same) == 0",
                f64::from(v[0]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("mse_loss [ERROR: {e}]"), false),
    }

    let pred2 = tensor(&[1.0, 2.0, 3.0], vec![3], device);
    let target2 = tensor(&[4.0, 5.0, 6.0], vec![3], device);
    match pred2.mse_loss(target2) {
        Ok(out) => {
            let v = readback(&out);
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
    let input = tensor(&[-10.0, -1.0, 0.0, 1.0, 10.0], vec![5], device);
    match input.tanh() {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "tanh(0) == 0",
                f64::from(v[2]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "tanh(1) ≈ 0.7616",
                f64::from(v[3]),
                0.7616,
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
            h.check_abs(
                "tanh(-10) ≈ -1",
                f64::from(v[0]),
                -1.0,
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
            h.check_abs(
                "tanh(10) ≈ 1",
                f64::from(v[4]),
                1.0,
                tolerances::TENSOR_TRANSCENDENTAL_F32,
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
    let exp_input = tensor(&[0.0, 1.0, 2.0, -1.0], vec![4], device);
    match exp_input.exp_wgsl() {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "exp(0) == 1",
                f64::from(v[0]),
                1.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "exp(1) ≈ e",
                f64::from(v[1]),
                std::f64::consts::E,
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
            h.check_abs(
                "exp(-1) ≈ 1/e",
                f64::from(v[3]),
                1.0 / std::f64::consts::E,
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
        }
        Err(e) => h.check_bool(&format!("exp_wgsl [ERROR: {e}]"), false),
    }

    let log_input = tensor(&[1.0, std::f32::consts::E, 10.0], vec![3], device);
    match log_input.log_wgsl() {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "log(1) == 0",
                f64::from(v[0]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "log(e) ≈ 1",
                f64::from(v[1]),
                1.0,
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
            h.check_abs(
                "log(10) ≈ 2.3026",
                f64::from(v[2]),
                10.0_f64.ln(),
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
        }
        Err(e) => h.check_bool(&format!("log_wgsl [ERROR: {e}]"), false),
    }

    let sqrt_input = tensor(&[0.0, 1.0, 4.0, 9.0, 16.0], vec![5], device);
    match sqrt_input.sqrt_wgsl() {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "sqrt(0) == 0",
                f64::from(v[0]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "sqrt(4) == 2",
                f64::from(v[2]),
                2.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "sqrt(9) == 3",
                f64::from(v[3]),
                3.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("sqrt_wgsl [ERROR: {e}]"), false),
    }
}

// ── Scalar ops ──────────────────────────────────────────────────────────

fn validate_scalar_ops(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let input = tensor(&[2.0, 4.0, 6.0, 8.0], vec![4], device);

    match input.mul_scalar(3.0) {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "mul_scalar [0] = 6",
                f64::from(v[0]),
                6.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "mul_scalar [3] = 24",
                f64::from(v[3]),
                24.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("mul_scalar [ERROR: {e}]"), false),
    }

    let input2 = tensor(&[1.0, 2.0, 3.0, 4.0], vec![4], device);
    match input2.add_scalar(10.0) {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "add_scalar [0] = 11",
                f64::from(v[0]),
                11.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "add_scalar [3] = 14",
                f64::from(v[3]),
                14.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("add_scalar [ERROR: {e}]"), false),
    }

    let input3 = tensor(&[10.0, 20.0, 30.0, 40.0], vec![4], device);
    match input3.div_scalar(5.0) {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "div_scalar [0] = 2",
                f64::from(v[0]),
                2.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "div_scalar [3] = 8",
                f64::from(v[3]),
                8.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("div_scalar [ERROR: {e}]"), false),
    }
}

// ── Element-wise div ────────────────────────────────────────────────────

fn validate_div(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let lhs = tensor(&[10.0, 20.0, 30.0, 40.0], vec![4], device);
    let rhs = tensor(&[2.0, 4.0, 5.0, 8.0], vec![4], device);
    match lhs.div(&rhs) {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "div [0] = 5",
                f64::from(v[0]),
                5.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "div [3] = 5",
                f64::from(v[3]),
                5.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("div [ERROR: {e}]"), false),
    }
}

// ── Reductions ──────────────────────────────────────────────────────────

fn validate_reductions(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let input = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5], device);

    match input.sum() {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "sum([1..5]) == 15",
                f64::from(v[0]),
                15.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("sum [ERROR: {e}]"), false),
    }

    let input2 = tensor(&[2.0, 4.0, 6.0, 8.0, 10.0], vec![5], device);
    match input2.mean() {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "mean([2,4,6,8,10]) == 6",
                f64::from(v[0]),
                6.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("mean [ERROR: {e}]"), false),
    }

    let input3 = tensor(&[3.0, 1.0, 7.0, 2.0, 5.0], vec![5], device);
    match input3.max() {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "max([3,1,7,2,5]) == 7",
                f64::from(v[0]),
                7.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("max [ERROR: {e}]"), false),
    }

    let input4 = tensor(&[3.0, 1.0, 7.0, 2.0, 5.0], vec![5], device);
    match input4.min() {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "min([3,1,7,2,5]) == 1",
                f64::from(v[0]),
                1.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("min [ERROR: {e}]"), false),
    }

    let input5 = tensor(&[3.0, 4.0], vec![2], device);
    match input5.norm() {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "norm([3,4]) == 5",
                f64::from(v[0]),
                5.0,
                tolerances::TENSOR_NORM_F32,
            );
        }
        Err(e) => h.check_bool(&format!("norm [ERROR: {e}]"), false),
    }
}

// ── Extended activations ────────────────────────────────────────────────

fn validate_activations_extended(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    // leaky_relu and elu bugs (S-05, S-06) now fixed upstream — tested in
    // validate_leaky_relu() and validate_elu() below.

    let input2 = tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], device);
    match input2.swish_wgsl() {
        Ok(out) => {
            let v = readback(&out);
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

    let input4 = tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], device);
    match input4.mish_wgsl() {
        Ok(out) => {
            let v = readback(&out);
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
    let pred = tensor(&[1.0, 2.0, 3.0], vec![3], device);
    let target = tensor(&[4.0, 5.0, 6.0], vec![3], device);
    match pred.mae_loss(&target) {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "mae([1,2,3],[4,5,6]) == 3",
                f64::from(v[0]),
                3.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("mae_loss [ERROR: {e}]"), false),
    }

    let pred2 = tensor(&[1.0, 2.0, 3.0], vec![3], device);
    let target2 = tensor(&[1.0, 2.0, 3.0], vec![3], device);
    match pred2.huber_loss(&target2, 1.0) {
        Ok(out) => {
            let v = readback(&out);
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
    let mat = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], device);
    match mat.transpose() {
        Ok(out) => {
            let v = readback(&out);
            h.check_bool("transpose shape [3,2]", *out.shape() == [3, 2]);
            h.check_abs(
                "transpose [0,0] = 1",
                f64::from(v[0]),
                1.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "transpose [0,1] = 4",
                f64::from(v[1]),
                4.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "transpose [1,0] = 2",
                f64::from(v[2]),
                2.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("transpose [ERROR: {e}]"), false),
    }
}

// ── Log-softmax (fixed upstream in BarraCUDA — formerly S-04 workaround) ─

fn validate_log_softmax(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let data = [1.0_f32, 2.0, 3.0];

    let input = tensor(&data, vec![1, 3], device);
    match input.log_softmax_wgsl() {
        Ok(out) => {
            let v = readback(&out);
            h.check_bool("log_softmax_wgsl all negative", v.iter().all(|&x| x < 0.0));

            let max_val = 3.0_f64;
            let lse = ((-2.0_f64).exp() + (-1.0_f64).exp() + 0.0_f64.exp()).ln();
            let expected: Vec<f64> = data.iter().map(|&x| f64::from(x) - max_val - lse).collect();

            h.check_abs(
                "log_softmax_wgsl[0]",
                f64::from(v[0]),
                expected[0],
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
            h.check_abs(
                "log_softmax_wgsl[1]",
                f64::from(v[1]),
                expected[1],
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
            h.check_abs(
                "log_softmax_wgsl[2]",
                f64::from(v[2]),
                expected[2],
                tolerances::TENSOR_TRANSCENDENTAL_F32,
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
    let input = tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], device);
    match input.leaky_relu_wgsl_with_slope(0.01) {
        Ok(out) => {
            let v = readback(&out);
            h.check_abs(
                "leaky_relu(-2, 0.01) ≈ -0.02",
                f64::from(v[0]),
                -0.02,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "leaky_relu(0) == 0",
                f64::from(v[2]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "leaky_relu(2) == 2",
                f64::from(v[4]),
                2.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("leaky_relu [ERROR: {e}]"), false),
    }
}

fn validate_elu(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let input = tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], device);
    match input.elu_wgsl() {
        Ok(out) => {
            let v = readback(&out);
            let expect_neg2 = (-2.0_f64).exp_m1();
            h.check_abs(
                "elu(-2, 1.0)",
                f64::from(v[0]),
                expect_neg2,
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
            h.check_abs(
                "elu(0) == 0",
                f64::from(v[2]),
                0.0,
                tolerances::TENSOR_EXACT_F32,
            );
            h.check_abs(
                "elu(2) == 2",
                f64::from(v[4]),
                2.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => h.check_bool(&format!("elu [ERROR: {e}]"), false),
    }
}
