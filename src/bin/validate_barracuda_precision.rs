// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: `barracuda::shaders::precision::cpu` primitives.
//!
//! Validates `elementwise_add`, `elementwise_mul`, `elementwise_fma`,
//! `dot_product`, `kahan_sum`, and `reduce_sum` against analytically
//! known values. These are the CPU implementations that match GPU
//! algorithms exactly — the foundation of cross-device agreement.
//!
//! ## Provenance
//!
//! Expected values: pure arithmetic (exact in f64).

use barracuda::shaders::precision::cpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_precision");

    validate_elementwise(&mut h);
    validate_dot(&mut h);
    validate_reduction(&mut h);

    h.finish();
}

fn validate_elementwise(h: &mut ValidationHarness) {
    let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0];
    let c: Vec<f64> = vec![0.5, 0.5, 0.5, 0.5];

    // --- add ---
    let mut out = vec![0.0_f64; 4];
    cpu::elementwise_add(&a, &b, &mut out);
    h.check_abs("add [0]: 1+10=11", out[0], 11.0, tolerances::EXACT_F64);
    h.check_abs("add [3]: 4+40=44", out[3], 44.0, tolerances::EXACT_F64);

    // --- mul ---
    cpu::elementwise_mul(&a, &b, &mut out);
    h.check_abs("mul [0]: 1*10=10", out[0], 10.0, tolerances::EXACT_F64);
    h.check_abs("mul [2]: 3*30=90", out[2], 90.0, tolerances::EXACT_F64);

    // --- fma: a*b+c ---
    cpu::elementwise_fma(&a, &b, &c, &mut out);
    h.check_abs(
        "fma [0]: 1*10+0.5=10.5",
        out[0],
        10.5,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "fma [3]: 4*40+0.5=160.5",
        out[3],
        160.5,
        tolerances::EXACT_F64,
    );
}

fn validate_dot(h: &mut ValidationHarness) {
    let lhs: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let rhs: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0];

    // dot(lhs,rhs) = 1*10 + 2*20 + 3*30 + 4*40 = 10+40+90+160 = 300
    let dot = cpu::dot_product(&lhs, &rhs);
    h.check_abs(
        "dot([1..4],[10..40]) == 300",
        dot,
        300.0,
        tolerances::EXACT_F64,
    );

    // dot with self: ||lhs||² = 1+4+9+16 = 30
    let norm_sq = cpu::dot_product(&lhs, &lhs);
    h.check_abs("dot(a,a) == 30", norm_sq, 30.0, tolerances::EXACT_F64);

    // orthogonal vectors
    let unit_x: Vec<f64> = vec![1.0, 0.0];
    let unit_y: Vec<f64> = vec![0.0, 1.0];
    let ortho = cpu::dot_product(&unit_x, &unit_y);
    h.check_abs("dot(e1, e2) == 0", ortho, 0.0, tolerances::EXACT_F64);
}

fn validate_reduction(h: &mut ValidationHarness) {
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // reduce_sum
    let sum = cpu::reduce_sum(&data);
    h.check_abs("reduce_sum([1..5]) == 15", sum, 15.0, tolerances::EXACT_F64);

    // kahan_sum — same result for well-conditioned data
    let kahan = cpu::kahan_sum(&data);
    h.check_abs(
        "kahan_sum([1..5]) == 15",
        kahan,
        15.0,
        tolerances::EXACT_F64,
    );

    // Kahan vs naive on pathological data: large + tiny values
    let pathological: Vec<f64> = {
        let mut v = vec![1e16_f64, -1e16];
        v.extend(std::iter::repeat_n(1.0, 1000));
        v
    };
    let kahan_path = cpu::kahan_sum(&pathological);
    h.check_abs(
        "kahan_sum(pathological) == 1000",
        kahan_path,
        1000.0,
        tolerances::EXACT_F64,
    );
}
