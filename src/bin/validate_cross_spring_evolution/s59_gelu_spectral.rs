// SPDX-License-Identifier: AGPL-3.0-or-later

//! S59: GELU + HMM forward on dispatcher; spectral library (ESD, MP bounds, effective rank).

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::neural_pgm;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, bench_once, max_abs_diff_f64};
use neural_spring::weight_spectral;

pub fn validate_rewired_esd(h: &mut ValidationHarness) {
    let eigenvalues: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
    let (centers, counts) = weight_spectral::empirical_spectral_density(&eigenvalues, 20);

    h.check_bool("rewired ESD returns 20 bins", centers.len() == 20);
    let sum: f64 = counts.iter().sum();
    h.check_abs("rewired ESD sums to 1", sum, 1.0, tolerances::EXACT_F64);
}

pub fn validate_rewired_mp_bounds(h: &mut ValidationHarness) {
    let (lo, hi) = weight_spectral::marchenko_pastur_bounds(1.0);
    h.check_abs("rewired MP lower bound γ=1", lo, 0.0, tolerances::EXACT_F64);
    h.check_abs("rewired MP upper bound γ=1", hi, 4.0, tolerances::EXACT_F64);

    let (lo2, hi2) = weight_spectral::marchenko_pastur_bounds(0.25);
    h.check_bool("rewired MP bounds ordered", lo2 < hi2);
}

pub fn validate_rewired_effective_rank(h: &mut ValidationHarness) {
    let full_rank = vec![1.0; 8];
    let rank = neural_pgm::effective_rank(&full_rank);
    h.check_abs(
        "rewired effective_rank full",
        rank,
        8.0,
        tolerances::DISPATCH_ELEMENTWISE_F64,
    );

    let mut low_rank = vec![0.0; 8];
    low_rank[0] = 1.0;
    let rank_low = neural_pgm::effective_rank(&low_rank);
    h.check_abs(
        "rewired effective_rank single",
        rank_low,
        1.0,
        tolerances::DISPATCH_ELEMENTWISE_F64,
    );
}

pub fn validate_rewired_gelu(h: &mut ValidationHarness, dispatcher: &Dispatcher, cpu: &Dispatcher) {
    let x: Vec<f64> = (-50..50).map(|i| f64::from(i) * 0.1).collect();

    let (result, _) = bench_once("gelu upstream", || dispatcher.gelu(&x));
    let (reference, _) = bench_once("gelu CPU ref", || cpu.gelu(&x));

    h.check_abs(
        "rewired gelu parity",
        max_abs_diff_f64(&result, &reference),
        0.0,
        tolerances::DISPATCH_ELEMENTWISE_F64,
    );

    h.check_abs(
        "gelu(0) ≈ 0",
        result[50],
        0.0,
        tolerances::DISPATCH_NEAR_ZERO_F64,
    );
}

pub fn validate_rewired_hmm_forward(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    cpu: &Dispatcher,
) {
    let n = 3;
    let alpha = vec![0.5, 0.3, 0.2];
    let transition = vec![0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.2, 0.2, 0.6];
    let emission = vec![0.4, 0.3, 0.3];

    let (result, _) = bench_once("hmm_forward upstream", || {
        dispatcher.hmm_forward_step(&alpha, &transition, &emission, n)
    });
    let (reference, _) = bench_once("hmm_forward CPU ref", || {
        cpu.hmm_forward_step(&alpha, &transition, &emission, n)
    });

    h.check_abs(
        "rewired hmm_forward alpha parity",
        max_abs_diff_f64(&result.0, &reference.0),
        0.0,
        tolerances::DISPATCH_ELEMENTWISE_F64,
    );
    h.check_abs(
        "rewired hmm_forward scale parity",
        result.1,
        reference.1,
        tolerances::DISPATCH_ELEMENTWISE_F64,
    );

    let alpha_sum: f64 = result.0.iter().sum();
    h.check_abs(
        "hmm_forward alpha normalized",
        alpha_sum,
        1.0,
        tolerances::DISPATCH_ELEMENTWISE_F64,
    );
}
