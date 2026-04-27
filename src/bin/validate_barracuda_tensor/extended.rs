// SPDX-License-Identifier: AGPL-3.0-or-later

//! Extended tensor validations: transcendental math, element-wise ops,
//! reductions, extended activations/losses, and upstream-fixed activations.

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::require;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, check_gpu_points};
use std::sync::Arc;

use super::{readback, tensor};

// ── Tanh ────────────────────────────────────────────────────────────────

pub fn validate_tanh(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let input = require!(
        h,
        tensor(&[-10.0, -1.0, 0.0, 1.0, 10.0], vec![5], device),
        "Tensor::from_data: GPU buffer alloc"
    );
    match input.tanh() {
        Ok(out) => {
            let v = require!(h, readback(&out), "tensor readback from GPU");
            let tex = tolerances::TENSOR_EXACT_F32;
            let ttf = tolerances::TENSOR_TRANSCENDENTAL_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("tanh(0) == 0", 2, 0.0, tex),
                    ("tanh(-10) ≈ -1", 0, -1.0, ttf),
                    ("tanh(10) ≈ 1", 4, 1.0, ttf),
                    ("tanh(1) ≈ 0.7616", 3, 0.761_594_155_955_764, ttf),
                ],
            );
            h.check_bool("tanh symmetric: |tanh(-1) + tanh(1)| ≈ 0", {
                (f64::from(v[1]) + f64::from(v[3])).abs() < ttf
            });
        }
        Err(e) => h.check_bool(&format!("tanh [ERROR: {e}]"), false),
    }
}

// ── Exp / Log / Sqrt ────────────────────────────────────────────────────

pub fn validate_exp_log_sqrt(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let exp_in = require!(h, tensor(&[0.0, 1.0, 2.0, -1.0], vec![4], device), "alloc");
    match exp_in.exp_wgsl() {
        Ok(out) => {
            let v = require!(h, readback(&out), "readback");
            let tol = tolerances::TENSOR_TRANSCENDENTAL_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("exp(0) == 1", 0, 1.0, tol),
                    ("exp(1) ≈ 2.7183", 1, std::f64::consts::E, tol),
                    ("exp(2) ≈ 7.389", 2, (2.0_f64).exp(), tol),
                    ("exp(-1) ≈ 0.3679", 3, (-1.0_f64).exp(), tol),
                ],
            );
        }
        Err(e) => h.check_bool(&format!("exp [ERROR: {e}]"), false),
    }

    let log_in = require!(
        h,
        tensor(&[1.0, std::f32::consts::E, 10.0, 100.0], vec![4], device),
        "alloc"
    );
    match log_in.log_wgsl() {
        Ok(out) => {
            let v = require!(h, readback(&out), "readback");
            let tol = tolerances::TENSOR_TRANSCENDENTAL_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("log(1) == 0", 0, 0.0, tol),
                    ("log(e) ≈ 1", 1, 1.0, tol),
                    ("log(10) ≈ 2.302", 2, 10.0_f64.ln(), tol),
                    ("log(100) ≈ 4.605", 3, 100.0_f64.ln(), tol),
                ],
            );
        }
        Err(e) => h.check_bool(&format!("log [ERROR: {e}]"), false),
    }

    let sqrt_in = require!(
        h,
        tensor(&[0.0, 1.0, 4.0, 9.0, 16.0, 25.0], vec![6], device),
        "alloc"
    );
    match sqrt_in.sqrt_wgsl() {
        Ok(out) => {
            let v = require!(h, readback(&out), "readback");
            let tex = tolerances::TENSOR_EXACT_F32;
            check_gpu_points(
                h,
                &v,
                &[
                    ("sqrt(0) == 0", 0, 0.0, tex),
                    ("sqrt(1) == 1", 1, 1.0, tex),
                    ("sqrt(4) == 2", 2, 2.0, tex),
                    ("sqrt(9) == 3", 3, 3.0, tex),
                    ("sqrt(16) == 4", 4, 4.0, tex),
                    ("sqrt(25) == 5", 5, 5.0, tex),
                ],
            );
        }
        Err(e) => h.check_bool(&format!("sqrt [ERROR: {e}]"), false),
    }
}

// ── Scalar ops ──────────────────────────────────────────────────────────

fn check_scalar_op(
    h: &mut ValidationHarness,
    device: &Arc<WgpuDevice>,
    data: &[f32],
    op: impl FnOnce(Tensor) -> Result<Tensor, barracuda::error::BarracudaError>,
    name: &str,
    checks: &[(usize, f32)],
) {
    let t = require!(h, tensor(data, vec![data.len()], device), "alloc");
    match op(t) {
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

pub fn validate_scalar_ops(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    check_scalar_op(
        h,
        device,
        &[2.0, 4.0, 6.0, 8.0],
        |t| t.mul_scalar(3.0),
        "mul_scalar",
        &[(0, 6.0), (3, 24.0)],
    );
    check_scalar_op(
        h,
        device,
        &[1.0, 2.0, 3.0, 4.0],
        |t| t.add_scalar(10.0),
        "add_scalar",
        &[(0, 11.0), (3, 14.0)],
    );
    check_scalar_op(
        h,
        device,
        &[10.0, 20.0, 30.0, 40.0],
        |t| t.div_scalar(5.0),
        "div_scalar",
        &[(0, 2.0), (3, 8.0)],
    );
}

// ── Element-wise div ────────────────────────────────────────────────────

pub fn validate_div(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    super::check_binary_op(
        h,
        device,
        &[10.0, 20.0, 30.0, 40.0],
        &[2.0, 4.0, 5.0, 8.0],
        Tensor::div,
        "div",
        &[(0, 5.0), (3, 5.0)],
    );
}

// ── Reductions ──────────────────────────────────────────────────────────

pub fn validate_reductions(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    use neural_spring::validation::{ReductionExpected, validate_tensor_reduction};
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

pub fn validate_activations_extended(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
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

pub fn validate_losses_extended(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
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

pub fn validate_transpose(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
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

pub fn validate_log_softmax(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
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

pub fn validate_leaky_relu(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
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

pub fn validate_elu(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
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
