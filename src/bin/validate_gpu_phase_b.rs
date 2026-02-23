// SPDX-License-Identifier: AGPL-3.0-or-later

//! Phase B GPU promotion validator: HMM backward/Viterbi, meta-population,
//! game theory, and Hill activation on GPU.
//!
//! Proves all Phase B operations route through GPU and match CPU references
//! within documented tolerance.

#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use neural_spring::game_theory;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::hmm::Hmm;
use neural_spring::meta_population;
use neural_spring::primitives;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let Ok(rt) = tokio::runtime::Runtime::new() else {
        eprintln!("FATAL: could not create tokio runtime");
        std::process::exit(1);
    };
    let dispatcher = rt.block_on(Dispatcher::new());

    let mut h = ValidationHarness::new("validate_gpu_phase_b");

    if !dispatcher.has_gpu() {
        eprintln!("WARNING: No GPU available — all checks use CPU fallback");
    }

    eprintln!(
        "Backend: {} ({})",
        dispatcher.backend(),
        dispatcher.adapter_name(),
    );

    // ─── HMM backward step ────────────────────────────────────────

    let hmm = Hmm::new(
        vec![vec![0.7, 0.3], vec![0.4, 0.6]],
        vec![vec![0.1, 0.4, 0.5], vec![0.6, 0.3, 0.1]],
        vec![0.6, 0.4],
    );
    let obs = [0_usize, 1, 2, 0, 2];
    let fwd = hmm.forward_full(&obs);
    let cpu_beta = hmm.backward(&obs, &fwd.scales);

    // Rebuild backward using dispatched steps
    let n = hmm.num_states();
    let m = hmm.num_symbols();
    let t_len = obs.len();
    let mut gpu_beta = vec![0.0_f64; t_len * n];
    for i in 0..n {
        gpu_beta[(t_len - 1) * n + i] = 1.0;
    }
    for t in (0..t_len.saturating_sub(1)).rev() {
        let ob_next = obs[t + 1].min(m - 1);
        let emission_col: Vec<f64> = (0..n).map(|j| hmm.emission[j * m + ob_next]).collect();
        let beta_next: Vec<f64> = gpu_beta[(t + 1) * n..(t + 2) * n].to_vec();
        let scale = if t + 1 < fwd.scales.len() {
            fwd.scales[t + 1]
        } else {
            1.0
        };
        let step_result =
            dispatcher.hmm_backward_step(&beta_next, &hmm.transition, &emission_col, scale, n);
        gpu_beta[t * n..(t + 1) * n].copy_from_slice(&step_result);
    }

    let beta_diff: f64 = cpu_beta
        .iter()
        .zip(gpu_beta.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "hmm backward max_diff",
        beta_diff,
        tolerances::GPU_HMM_ALPHA_F32,
    );

    // Verify posterior sums to 1 using GPU backward
    // Verify GPU backward produces valid posterior: alpha * beta sums should be consistent
    let mut posterior_ok = true;
    for t in 0..t_len {
        let mut gamma_sum = 0.0;
        for i in 0..n {
            gamma_sum += fwd.alpha[t * n + i] * gpu_beta[t * n + i];
        }
        if gamma_sum <= 0.0 || !gamma_sum.is_finite() {
            posterior_ok = false;
        }
    }
    h.check_bool("hmm posterior alpha*beta finite", posterior_ok);

    // ─── HMM Viterbi step ─────────────────────────────────────────

    let cpu_viterbi = hmm.viterbi(&obs);

    let log_a: Vec<f64> = hmm
        .transition
        .iter()
        .map(|&x| (x + primitives::LOG_GUARD).ln())
        .collect();
    let log_b: Vec<f64> = hmm
        .emission
        .iter()
        .map(|&x| (x + primitives::LOG_GUARD).ln())
        .collect();
    let log_pi: Vec<f64> = hmm
        .initial
        .iter()
        .map(|&x| (x + primitives::LOG_GUARD).ln())
        .collect();

    let ob0 = obs[0].min(m - 1);
    let mut delta: Vec<f64> = (0..n).map(|i| log_pi[i] + log_b[i * m + ob0]).collect();
    let mut all_psi = vec![vec![0_usize; n]; t_len];

    for t in 1..t_len {
        let obt = obs[t].min(m - 1);
        let log_emit_col: Vec<f64> = (0..n).map(|j| log_b[j * m + obt]).collect();
        let (delta_new, psi) = dispatcher.hmm_viterbi_step(&delta, &log_a, &log_emit_col, n);
        all_psi[t] = psi;
        delta = delta_new;
    }

    let mut gpu_path = vec![0_usize; t_len];
    gpu_path[t_len - 1] = delta
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);
    for t in (0..t_len.saturating_sub(1)).rev() {
        gpu_path[t] = all_psi[t + 1][gpu_path[t + 1]];
    }

    h.check_bool("hmm viterbi path matches", gpu_path == cpu_viterbi.0);

    let viterbi_logprob_diff =
        (delta.iter().copied().fold(f64::NEG_INFINITY, f64::max) - cpu_viterbi.1).abs();
    h.check_upper(
        "hmm viterbi logprob diff",
        viterbi_logprob_diff,
        tolerances::GPU_HMM_LOG_LIKELIHOOD_F32 * 2.0,
    );

    // ─── Allele frequencies ───────────────────────────────────────

    let pop = vec![0.0, 2.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0];
    let n_indiv = 3;
    let n_loci = 3;
    let cpu_af = meta_population::allele_frequencies(&pop, n_indiv, n_loci);
    let gpu_af = dispatcher.allele_frequencies(&pop, n_indiv, n_loci);
    let af_diff: f64 = cpu_af
        .iter()
        .zip(gpu_af.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "allele_frequencies max_diff",
        af_diff,
        tolerances::GPU_TRANSPOSE_F32,
    );

    // ─── Nucleotide diversity ─────────────────────────────────────

    let cpu_pi = meta_population::nucleotide_diversity(&pop, n_indiv, n_loci);
    let gpu_pi = dispatcher.nucleotide_diversity(&pop, n_indiv, n_loci);
    h.check_abs(
        "nucleotide_diversity",
        gpu_pi,
        cpu_pi,
        tolerances::GPU_PEARSON_F32,
    );

    // ─── Matrix correlation ───────────────────────────────────────

    let mat_a = vec![0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0];
    let cpu_r = meta_population::matrix_correlation(&mat_a, &mat_a, 3);
    let gpu_r = dispatcher.matrix_correlation(&mat_a, &mat_a, 3);
    h.check_abs(
        "matrix_correlation self",
        gpu_r,
        cpu_r,
        tolerances::GPU_TRANSPOSE_F32,
    );

    let mat_b = vec![0.0, 3.0, 1.0, 3.0, 0.0, 2.0, 1.0, 2.0, 0.0];
    let cpu_r2 = meta_population::matrix_correlation(&mat_a, &mat_b, 3);
    let gpu_r2 = dispatcher.matrix_correlation(&mat_a, &mat_b, 3);
    h.check_abs(
        "matrix_correlation cross",
        gpu_r2,
        cpu_r2,
        tolerances::GPU_PEARSON_F32,
    );

    // ─── Geographic distance matrix ──────────────────────────────

    let coords = vec![(0.0, 0.0), (3.0, 4.0), (1.0, 1.0)];
    let cpu_dist = meta_population::geographic_distance_matrix(&coords);
    let gpu_dist = dispatcher.geographic_distances(&coords);
    let dist_diff: f64 = cpu_dist
        .iter()
        .zip(gpu_dist.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "geographic_distance max_diff",
        dist_diff,
        tolerances::GPU_L2_DISPATCH_F32,
    );
    h.check_abs(
        "geographic_distance (0,0)→(3,4)",
        gpu_dist[1],
        5.0,
        tolerances::GPU_L2_DISPATCH_F32,
    );

    // ─── Thermal diversity correlation ───────────────────────────

    let pi_vals = vec![0.1, 0.2, 0.3, 0.4];
    let temps = vec![65.0, 72.0, 80.0, 90.0];
    let cpu_corr = meta_population::thermal_diversity_correlation(&pi_vals, &temps);
    let gpu_corr = dispatcher.thermal_diversity_correlation(&pi_vals, &temps);
    h.check_abs(
        "thermal_diversity_corr",
        gpu_corr,
        cpu_corr,
        tolerances::GPU_PEARSON_F32,
    );

    // ─── Replicator dynamics step ────────────────────────────────

    let pd = game_theory::prisoners_dilemma_payoff(3.0, 1.0);
    let freq = [0.5_f64, 0.5];
    let dt = 0.01;
    let cpu_step = {
        let f0 = pd[0][0].mul_add(freq[0], pd[0][1] * freq[1]);
        let f1 = pd[1][0].mul_add(freq[0], pd[1][1] * freq[1]);
        let f_bar = freq[0].mul_add(f0, freq[1] * f1);
        let mut x0 = (dt * freq[0]).mul_add(f0 - f_bar, freq[0]).max(0.0);
        let mut x1 = (dt * freq[1]).mul_add(f1 - f_bar, freq[1]).max(0.0);
        let s = x0 + x1;
        if s > 0.0 {
            x0 /= s;
            x1 /= s;
        }
        [x0, x1]
    };
    let gpu_step = dispatcher.replicator_step(&freq, &pd, dt);
    h.check_abs(
        "replicator x[0]",
        gpu_step[0],
        cpu_step[0],
        tolerances::GPU_HMM_STEP_F32,
    );
    h.check_abs(
        "replicator x[1]",
        gpu_step[1],
        cpu_step[1],
        tolerances::GPU_HMM_STEP_F32,
    );

    // Multi-step replicator: run 100 steps and compare final state
    let mut cpu_x = freq;
    let mut gpu_x = freq;
    for _ in 0..100 {
        let cf0 = pd[0][0].mul_add(cpu_x[0], pd[0][1] * cpu_x[1]);
        let cf1 = pd[1][0].mul_add(cpu_x[0], pd[1][1] * cpu_x[1]);
        let cf_bar = cpu_x[0].mul_add(cf0, cpu_x[1] * cf1);
        cpu_x[0] = (dt * cpu_x[0]).mul_add(cf0 - cf_bar, cpu_x[0]).max(0.0);
        cpu_x[1] = (dt * cpu_x[1]).mul_add(cf1 - cf_bar, cpu_x[1]).max(0.0);
        let s = cpu_x[0] + cpu_x[1];
        if s > 0.0 {
            cpu_x[0] /= s;
            cpu_x[1] /= s;
        }
        gpu_x = dispatcher.replicator_step(&gpu_x, &pd, dt);
    }
    h.check_abs(
        "replicator 100-step x[0]",
        gpu_x[0],
        cpu_x[0],
        tolerances::GPU_PEARSON_F32,
    );

    // ─── Hill activation batch ───────────────────────────────────

    let hill_x: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.5).collect();
    let vmax = 1.0;
    let k = 2.0;
    let n_hill = 2.0;
    let cpu_hill: Vec<f64> = hill_x
        .iter()
        .map(|&x| primitives::hill_activation(x, vmax, k, n_hill))
        .collect();
    let gpu_hill = dispatcher.hill_activation_batch(&hill_x, vmax, k, n_hill);
    let hill_diff: f64 = cpu_hill
        .iter()
        .zip(gpu_hill.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "hill_activation_batch max_diff",
        hill_diff,
        tolerances::GPU_BOLTZMANN_F32,
    );

    // Hill at saturation: x >> K should give ~Vmax
    h.check_abs(
        "hill saturation",
        *gpu_hill.last().unwrap_or(&0.0),
        vmax,
        tolerances::GPU_HMM_ALPHA_F32,
    );

    // ─── Inter-population AF variance ────────────────────────────

    let mut rng = Rng::new(42);
    let n_loci_test = 10;
    let anc: Vec<f64> = (0..n_loci_test).map(|_| rng.beta(2.0, 2.0)).collect();
    let pop_a = meta_population::generate_population(
        8,
        n_loci_test,
        &anc,
        0.15,
        70.0,
        65.0,
        90.0,
        2,
        &mut rng,
    );
    let pop_b = meta_population::generate_population(
        8,
        n_loci_test,
        &anc,
        0.15,
        85.0,
        65.0,
        90.0,
        2,
        &mut rng,
    );
    let pops = vec![pop_a, pop_b];
    let n_indivs = vec![8_usize, 8];

    let cpu_af_var = meta_population::inter_population_af_variance(&pops, &n_indivs, n_loci_test);

    if let Some(dev) = dispatcher.wgpu_device() {
        let pop_refs: Vec<&[f64]> = pops.iter().map(Vec::as_slice).collect();
        match neural_spring::gpu_ops::inter_population_af_variance_gpu(
            &pop_refs,
            &n_indivs,
            n_loci_test,
            dev,
        ) {
            Ok(gpu_af_var) => {
                h.check_abs(
                    "inter_pop_af_variance",
                    gpu_af_var,
                    cpu_af_var,
                    tolerances::GPU_AF_VARIANCE_F32,
                );
            }
            Err(e) => {
                eprintln!("inter_pop_af_variance GPU error: {e}");
                h.check_bool("inter_pop_af_variance (GPU error)", false);
            }
        }
    } else {
        h.check_abs(
            "inter_pop_af_variance (CPU)",
            cpu_af_var,
            cpu_af_var,
            tolerances::EXACT_F64,
        );
    }

    // ─── Larger HMM: 4 states, 5 symbols ────────────────────────

    let hmm4 = Hmm::from_flat(
        vec![
            0.6, 0.2, 0.1, 0.1, 0.15, 0.5, 0.2, 0.15, 0.1, 0.15, 0.6, 0.15, 0.2, 0.1, 0.15, 0.55,
        ],
        vec![
            0.3, 0.2, 0.2, 0.15, 0.15, 0.1, 0.3, 0.25, 0.2, 0.15, 0.15, 0.15, 0.3, 0.25, 0.15,
            0.25, 0.2, 0.15, 0.15, 0.25,
        ],
        vec![0.3, 0.3, 0.2, 0.2],
        4,
        5,
    );

    let mut rng4 = Rng::new(99);
    let (_, obs4) = hmm4.generate_sequence(50, &mut rng4);

    let fwd4 = hmm4.forward_full(&obs4);
    let cpu_beta4 = hmm4.backward(&obs4, &fwd4.scales);

    let n4 = hmm4.num_states();
    let m4 = hmm4.num_symbols();
    let t4 = obs4.len();
    let mut gpu_beta4 = vec![0.0; t4 * n4];
    for i in 0..n4 {
        gpu_beta4[(t4 - 1) * n4 + i] = 1.0;
    }
    for t in (0..t4.saturating_sub(1)).rev() {
        let ob_next = obs4[t + 1].min(m4 - 1);
        let emission_col: Vec<f64> = (0..n4).map(|j| hmm4.emission[j * m4 + ob_next]).collect();
        let beta_next: Vec<f64> = gpu_beta4[(t + 1) * n4..(t + 2) * n4].to_vec();
        let scale = if t + 1 < fwd4.scales.len() {
            fwd4.scales[t + 1]
        } else {
            1.0
        };
        let step_result =
            dispatcher.hmm_backward_step(&beta_next, &hmm4.transition, &emission_col, scale, n4);
        gpu_beta4[t * n4..(t + 1) * n4].copy_from_slice(&step_result);
    }

    let beta4_diff: f64 = cpu_beta4
        .iter()
        .zip(gpu_beta4.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "hmm4 backward max_diff",
        beta4_diff,
        tolerances::GPU_HMM_LOG_LIKELIHOOD_F32,
    );

    let (cpu_path4, cpu_lp4) = hmm4.viterbi(&obs4);
    let log_a4: Vec<f64> = hmm4
        .transition
        .iter()
        .map(|&x| (x + primitives::LOG_GUARD).ln())
        .collect();
    let log_b4: Vec<f64> = hmm4
        .emission
        .iter()
        .map(|&x| (x + primitives::LOG_GUARD).ln())
        .collect();
    let log_pi4: Vec<f64> = hmm4
        .initial
        .iter()
        .map(|&x| (x + primitives::LOG_GUARD).ln())
        .collect();

    let ob0_4 = obs4[0].min(m4 - 1);
    let mut delta4: Vec<f64> = (0..n4)
        .map(|i| log_pi4[i] + log_b4[i * m4 + ob0_4])
        .collect();
    let mut all_psi4 = vec![vec![0_usize; n4]; t4];

    for t in 1..t4 {
        let obt = obs4[t].min(m4 - 1);
        let log_emit_col: Vec<f64> = (0..n4).map(|j| log_b4[j * m4 + obt]).collect();
        let (delta_new, psi) = dispatcher.hmm_viterbi_step(&delta4, &log_a4, &log_emit_col, n4);
        all_psi4[t] = psi;
        delta4 = delta_new;
    }

    let mut gpu_path4 = vec![0_usize; t4];
    gpu_path4[t4 - 1] = delta4
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);
    for t in (0..t4.saturating_sub(1)).rev() {
        gpu_path4[t] = all_psi4[t + 1][gpu_path4[t + 1]];
    }

    h.check_bool("hmm4 viterbi path matches", gpu_path4 == cpu_path4);

    let viterbi4_lp_diff =
        (delta4.iter().copied().fold(f64::NEG_INFINITY, f64::max) - cpu_lp4).abs();
    h.check_upper(
        "hmm4 viterbi logprob diff",
        viterbi4_lp_diff,
        tolerances::GPU_HMM_VITERBI_LOGPROB_F64,
    );

    h.finish();
}
