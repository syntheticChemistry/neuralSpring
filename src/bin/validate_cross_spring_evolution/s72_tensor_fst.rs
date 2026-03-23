// SPDX-License-Identifier: AGPL-3.0-or-later

//! S72 rewires: Tensor row-wise softmax, FST population-genetics, Viterbi argmax path.

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, bench_once, max_abs_diff_f64};

pub fn validate_rewired_softmax_row_wise(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    cpu: &Dispatcher,
) {
    let n_rows = 4;
    let n_cols = 8;
    let matrix: Vec<f64> = (0..n_rows * n_cols)
        .map(|i| (i as f64 - 16.0) * 0.1)
        .collect();

    let (result, _) = bench_once("softmax_row_wise upstream", || {
        dispatcher.softmax_row_wise(&matrix, n_rows, n_cols)
    });
    let (reference, _) = bench_once("softmax_row_wise CPU ref", || {
        cpu.softmax_row_wise(&matrix, n_rows, n_cols)
    });

    for row in 0..n_rows {
        let row_sum: f64 = result[row * n_cols..(row + 1) * n_cols].iter().sum();
        h.check_abs(
            &format!("softmax_row_wise row {row} sums to 1"),
            row_sum,
            1.0,
            tolerances::DISPATCH_F32_ROUNDTRIP,
        );
    }

    h.check_abs(
        "softmax_row_wise parity (f32 path)",
        max_abs_diff_f64(&result, &reference),
        0.0,
        tolerances::DISPATCH_F32_ROUNDTRIP,
    );
}

pub fn validate_rewired_fst_single_locus(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    let freqs_diverged = [0.1, 0.9];
    let sizes = [50, 50];

    let result = dispatcher.fst_single_locus(&freqs_diverged, &sizes);
    h.check_bool("fst_single_locus returns Ok", result.is_ok());

    if let Ok((fst, f_is, f_it)) = result {
        h.check_bool("fst_single_locus θ > 0.5 (diverged pops)", fst > 0.5);
        h.check_bool("fst_single_locus f_it defined", f_it.is_finite());
        h.check_bool("fst_single_locus f_is defined", f_is.is_finite());

        let identity_check = (1.0 - f_is).mul_add(-(1.0 - fst), 1.0 - f_it);
        h.check_abs(
            "fst_single_locus Wright identity (1-F_IT)=(1-F_IS)(1-F_ST)",
            identity_check,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );
    }

    let freqs_identical = [0.5, 0.5];
    if let Ok((fst, _, _)) = dispatcher.fst_single_locus(&freqs_identical, &sizes) {
        h.check_abs(
            "fst_single_locus θ near 0 (identical pops, W-C sample correction)",
            fst,
            0.0,
            tolerances::FST_IDENTICAL_POP_TOL,
        );
    }
}

pub fn validate_rewired_pairwise_fst_full(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    cpu: &Dispatcher,
) {
    let n_a = 20;
    let n_b = 20;
    let n_loci = 10;
    let pop_a: Vec<f64> = (0..n_a * n_loci).map(|i| (i % 2) as f64).collect();
    let pop_b: Vec<f64> = (0..n_b * n_loci).map(|i| ((i + 1) % 2) as f64).collect();

    let (fst, f_is, f_it) = dispatcher.pairwise_fst_full(&pop_a, n_a, &pop_b, n_b, n_loci);
    let fst_theta_only = cpu.pairwise_fst(&pop_a, n_a, &pop_b, n_b, n_loci);

    h.check_abs(
        "pairwise_fst_full θ ≈ pairwise_fst (mean-of-ratios vs ratio-of-sums)",
        fst,
        fst_theta_only,
        tolerances::FST_ESTIMATOR_AGREEMENT,
    );
    h.check_bool("pairwise_fst_full f_is defined", f_is.is_finite());
    h.check_bool("pairwise_fst_full f_it defined", f_it.is_finite());
    h.check_bool("pairwise_fst_full θ > 0 (diverged pops)", fst > 0.0);
}

pub fn validate_rewired_viterbi_argmax(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    cpu: &Dispatcher,
) {
    let n_states = 3;
    let n_obs = 2;
    let initial = vec![0.6, 0.3, 0.1];
    let transition = vec![0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.2, 0.2, 0.6];
    let emission = vec![0.5, 0.5, 0.4, 0.6, 0.7, 0.3];
    let observations = vec![0, 1, 0, 1, 0];

    let (path_gpu, logp_gpu) = dispatcher.hmm_viterbi_chain(
        &initial,
        &transition,
        &emission,
        &observations,
        n_states,
        n_obs,
    );
    let (path_cpu, logp_cpu) = cpu.hmm_viterbi_chain(
        &initial,
        &transition,
        &emission,
        &observations,
        n_states,
        n_obs,
    );

    h.check_bool("viterbi argmax_dim path matches CPU", path_gpu == path_cpu);
    h.check_abs(
        "viterbi argmax_dim log-prob (f32 GPU path)",
        logp_gpu,
        logp_cpu,
        tolerances::DISPATCH_VITERBI_F32,
    );
}
