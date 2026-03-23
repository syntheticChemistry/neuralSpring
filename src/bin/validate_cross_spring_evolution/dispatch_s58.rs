// SPDX-License-Identifier: AGPL-3.0-or-later

//! S58 rewired dispatcher methods: matmul, Frobenius, transpose, softmax, L2, mean, variance.

use crate::helpers::gen_f64_vec;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, bench_once, max_abs_diff_f64};

pub fn validate_rewired_matmul(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    cpu: &Dispatcher,
) {
    let n = 64;
    let a = gen_f64_vec(n * n, 0.001);
    let b: Vec<f64> = (0..n * n).map(|i| (n * n - i) as f64 * 0.001).collect();

    let (result, _) = bench_once("matmul upstream", || dispatcher.mat_mul(&a, &b, n));
    let (reference, _) = bench_once("matmul CPU ref", || cpu.mat_mul(&a, &b, n));

    h.check_abs(
        "rewired matmul parity (64x64)",
        max_abs_diff_f64(&result, &reference),
        0.0,
        tolerances::DISPATCH_MATMUL_F64,
    );
}

pub fn validate_rewired_frobenius(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    cpu: &Dispatcher,
) {
    let data = gen_f64_vec(1024, 0.01);

    let (result, _) = bench_once("frobenius upstream", || dispatcher.frobenius_norm(&data));
    let (reference, _) = bench_once("frobenius CPU ref", || cpu.frobenius_norm(&data));

    h.check_abs(
        "rewired frobenius parity",
        result,
        reference,
        tolerances::DISPATCH_FROBENIUS_F64,
    );
}

pub fn validate_rewired_transpose(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    cpu: &Dispatcher,
) {
    let n = 32;
    let a = gen_f64_vec(n * n, 0.1);

    let (result, _) = bench_once("transpose upstream", || dispatcher.transpose(&a, n));
    let (reference, _) = bench_once("transpose CPU ref", || cpu.transpose(&a, n));

    h.check_abs(
        "rewired transpose parity (32x32)",
        max_abs_diff_f64(&result, &reference),
        0.0,
        tolerances::DISPATCH_TRANSPOSE_F64,
    );
}

pub fn validate_rewired_softmax(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    cpu: &Dispatcher,
) {
    let x: Vec<f64> = (0..256_i32)
        .map(|i| f64::from(i).mul_add(0.02, -2.56))
        .collect();

    let (result, _) = bench_once("softmax upstream", || dispatcher.softmax(&x));
    let (reference, _) = bench_once("softmax CPU ref", || cpu.softmax(&x));

    let sum: f64 = result.iter().sum();
    h.check_abs(
        "rewired softmax sums to 1",
        sum,
        1.0,
        tolerances::DISPATCH_ELEMENTWISE_F64,
    );
    h.check_abs(
        "rewired softmax parity",
        max_abs_diff_f64(&result, &reference),
        0.0,
        tolerances::DISPATCH_ELEMENTWISE_F64,
    );
}

pub fn validate_rewired_l2(h: &mut ValidationHarness, dispatcher: &Dispatcher, cpu: &Dispatcher) {
    let a = gen_f64_vec(512, 0.01);
    let b: Vec<f64> = (0..512_i32)
        .map(|i| f64::from(i).mul_add(0.01, 1.0))
        .collect();

    let (result, _) = bench_once("l2_distance upstream", || dispatcher.l2_distance(&a, &b));
    let (reference, _) = bench_once("l2_distance CPU ref", || cpu.l2_distance(&a, &b));

    h.check_abs(
        "rewired l2_distance parity",
        result,
        reference,
        tolerances::DISPATCH_TWOPASS_F64,
    );
}

pub fn validate_rewired_mean(h: &mut ValidationHarness, dispatcher: &Dispatcher, cpu: &Dispatcher) {
    let data = gen_f64_vec(2048, 0.001);

    let (result, _) = bench_once("mean upstream", || dispatcher.mean(&data));
    let (reference, _) = bench_once("mean CPU ref", || cpu.mean(&data));

    h.check_abs(
        "rewired mean parity",
        result,
        reference,
        tolerances::DISPATCH_ELEMENTWISE_F64,
    );
}

pub fn validate_rewired_variance(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    cpu: &Dispatcher,
) {
    let data = gen_f64_vec(2048, 0.001);

    let (result, _) = bench_once("variance upstream", || dispatcher.variance(&data));
    let (reference, _) = bench_once("variance CPU ref", || cpu.variance(&data));

    h.check_abs(
        "rewired variance parity",
        result,
        reference,
        tolerances::DISPATCH_TWOPASS_F64,
    );
}
