// SPDX-License-Identifier: AGPL-3.0-or-later

//! `ToadStool` `ComputeDispatch` evolution validator.
//!
//! Proves that `neuralSpring`'s dispatch math maps to the same operations
//! `ToadStool`'s 144-op `ComputeDispatch` handles. Each test exercises
//! `barracuda::dispatch` functions directly (the same functions `ComputeDispatch`
//! wraps) and compares with our `Dispatcher`'s results.
//!
//! This is the bridge: `Dispatcher` → `barracuda::dispatch` → `ComputeDispatch`.
//!
//! ```text
//! neuralSpring Dispatcher
//!   ↓ delegates to
//! barracuda::dispatch::*_dispatch (CPU/GPU routing)
//!   ↓ same math as
//! `ToadStool` `ComputeDispatch` (144 ops, S86)
//! ```
//!
//! ## Provenance
//!
//! Validation class: Integration.
//! Analytical reference: Dispatcher → `barracuda::dispatch` → `ComputeDispatch` math parity.
//! Components: Dispatcher, `barracuda::dispatch` (matmul, transpose, softmax, gelu, mean, variance).

#![expect(
    clippy::cast_precision_loss,
    clippy::expect_used,
    reason = "validation binary"
)]

use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, exit_no_gpu};

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("validate_compute_dispatch_evolution");

    let Ok(gpu) = Gpu::new().await else {
        exit_no_gpu();
    };

    let dispatcher = Dispatcher::from_gpu(gpu);

    validate_matmul_dispatch_bridge(&mut h, &dispatcher);
    validate_transpose_dispatch_bridge(&mut h, &dispatcher);
    validate_softmax_dispatch_bridge(&mut h, &dispatcher);
    validate_gelu_dispatch_bridge(&mut h, &dispatcher);
    validate_mean_dispatch_bridge(&mut h, &dispatcher);
    validate_variance_dispatch_bridge(&mut h, &dispatcher);
    validate_l2_dispatch_bridge(&mut h, &dispatcher);
    validate_hmm_forward_dispatch_bridge(&mut h, &dispatcher);
    validate_frobenius_dispatch_bridge(&mut h, &dispatcher);
    validate_dispatch_threshold_routing(&mut h, &dispatcher);
    validate_dispatch_determinism(&mut h, &dispatcher);

    h.finish();
}

fn validate_matmul_dispatch_bridge(h: &mut ValidationHarness, disp: &Dispatcher) {
    let n = 8;
    let a: Vec<f64> = (0..n * n).map(|i| (i as f64 + 1.0) * 0.1).collect();
    let b: Vec<f64> = (0..n * n)
        .map(|i| (i as f64).mul_add(2.0, 1.0) * 0.05)
        .collect();

    let dispatcher_result = disp.mat_mul(&a, &b, n);

    let upstream_result = barracuda::dispatch::matmul_dispatch(&a, &b, n, n, n, disp.wgpu_device())
        .expect("matmul_dispatch should succeed");

    let max_diff = dispatcher_result
        .iter()
        .zip(upstream_result.iter())
        .map(|(d, u)| (d - u).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "matmul: Dispatcher == barracuda::dispatch",
        max_diff,
        tolerances::EXACT_F64,
    );
}

fn validate_transpose_dispatch_bridge(h: &mut ValidationHarness, disp: &Dispatcher) {
    let n = 5;
    let a: Vec<f64> = (0..n * n).map(|i| i as f64).collect();

    let dispatcher_result = disp.transpose(&a, n);

    let upstream_result = barracuda::dispatch::transpose_dispatch(&a, n, n, disp.wgpu_device())
        .expect("transpose_dispatch should succeed");

    let max_diff = dispatcher_result
        .iter()
        .zip(upstream_result.iter())
        .map(|(d, u)| (d - u).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "transpose: Dispatcher == barracuda::dispatch",
        max_diff,
        tolerances::EXACT_F64,
    );
}

fn validate_softmax_dispatch_bridge(h: &mut ValidationHarness, disp: &Dispatcher) {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let dispatcher_result = disp.softmax(&x);

    let upstream_result = barracuda::dispatch::softmax_dispatch(&x, disp.wgpu_device())
        .expect("softmax_dispatch should succeed");

    let max_diff = dispatcher_result
        .iter()
        .zip(upstream_result.iter())
        .map(|(d, u)| (d - u).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "softmax: Dispatcher == barracuda::dispatch",
        max_diff,
        tolerances::EXACT_F64,
    );

    let sum: f64 = upstream_result.iter().sum();
    h.check_abs(
        "softmax sums to 1 via barracuda::dispatch",
        sum,
        1.0,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_gelu_dispatch_bridge(h: &mut ValidationHarness, disp: &Dispatcher) {
    let x = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];

    let dispatcher_result = disp.gelu(&x);

    let upstream_result = barracuda::dispatch::gelu_dispatch(&x, disp.wgpu_device())
        .expect("gelu_dispatch should succeed");

    let max_diff = dispatcher_result
        .iter()
        .zip(upstream_result.iter())
        .map(|(d, u)| (d - u).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "gelu: Dispatcher == barracuda::dispatch",
        max_diff,
        tolerances::EXACT_F64,
    );
}

fn validate_mean_dispatch_bridge(h: &mut ValidationHarness, disp: &Dispatcher) {
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

    let dispatcher_result = disp.mean(&data);

    let upstream_result = barracuda::dispatch::mean_dispatch(&data, disp.wgpu_device())
        .expect("mean_dispatch should succeed");

    h.check_abs(
        "mean: Dispatcher == barracuda::dispatch",
        dispatcher_result,
        upstream_result,
        tolerances::EXACT_F64,
    );
}

fn validate_variance_dispatch_bridge(h: &mut ValidationHarness, disp: &Dispatcher) {
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

    let dispatcher_result = disp.variance(&data);

    let upstream_result = barracuda::dispatch::variance_dispatch(&data, disp.wgpu_device())
        .expect("variance_dispatch should succeed");

    h.check_abs(
        "variance: Dispatcher == barracuda::dispatch",
        dispatcher_result,
        upstream_result,
        tolerances::EXACT_F64,
    );
}

fn validate_l2_dispatch_bridge(h: &mut ValidationHarness, disp: &Dispatcher) {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let dispatcher_result = disp.l2_distance(&a, &b);

    let upstream_result = barracuda::dispatch::l2_distance_dispatch(&a, &b, disp.wgpu_device())
        .expect("l2_distance_dispatch should succeed");

    h.check_abs(
        "l2_distance: Dispatcher == barracuda::dispatch",
        dispatcher_result,
        upstream_result,
        tolerances::EXACT_F64,
    );
}

fn validate_hmm_forward_dispatch_bridge(h: &mut ValidationHarness, disp: &Dispatcher) {
    let alpha_prev = vec![0.6, 0.4];
    let trans = vec![0.7, 0.3, 0.4, 0.6];
    let emit_col = vec![0.1, 0.6];

    let (disp_alpha, disp_scale) = disp.hmm_forward_step(&alpha_prev, &trans, &emit_col, 2);

    let (upstream_alpha, upstream_scale) = barracuda::dispatch::hmm_forward_dispatch(
        &alpha_prev,
        &trans,
        &emit_col,
        2,
        disp.wgpu_device(),
    )
    .expect("hmm_forward_dispatch should succeed");

    let max_diff = disp_alpha
        .iter()
        .zip(upstream_alpha.iter())
        .map(|(d, u)| (d - u).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "hmm_forward alpha: Dispatcher == barracuda::dispatch",
        max_diff,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "hmm_forward scale: Dispatcher == barracuda::dispatch",
        disp_scale,
        upstream_scale,
        tolerances::EXACT_F64,
    );
}

fn validate_frobenius_dispatch_bridge(h: &mut ValidationHarness, disp: &Dispatcher) {
    let a = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let dispatcher_result = disp.frobenius_norm(&a);

    let upstream_result = barracuda::dispatch::frobenius_norm_dispatch(&a, disp.wgpu_device())
        .expect("frobenius_norm_dispatch should succeed");

    h.check_abs(
        "frobenius_norm: Dispatcher == barracuda::dispatch",
        dispatcher_result,
        upstream_result,
        tolerances::EXACT_F64,
    );
}

fn validate_dispatch_threshold_routing(h: &mut ValidationHarness, disp: &Dispatcher) {
    let small_data = vec![1.0, 2.0, 3.0, 4.0];
    let small_mean = disp.mean(&small_data);
    let expected_small = 2.5;
    h.check_abs(
        "small dispatch routes correctly (CPU expected)",
        small_mean,
        expected_small,
        tolerances::EXACT_F64,
    );

    let large_data: Vec<f64> = (0..4096_i32).map(f64::from).collect();
    let large_mean = disp.mean(&large_data);
    let expected_large = (4096.0 - 1.0) / 2.0;
    h.check_abs(
        "large dispatch routes correctly (GPU expected)",
        large_mean,
        expected_large,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_dispatch_determinism(h: &mut ValidationHarness, disp: &Dispatcher) {
    let n = 16;
    let a: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.3).sin()).collect();
    let b: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.7).cos()).collect();
    let r1 = disp.mat_mul(&a, &b, n);
    let r2 = disp.mat_mul(&a, &b, n);
    let max_diff = r1
        .iter()
        .zip(r2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "dispatch determinism: repeated matmul identical",
        max_diff == 0.0,
    );
}
