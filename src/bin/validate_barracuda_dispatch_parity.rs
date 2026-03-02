// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` dispatch parity — proves CPU and GPU paths produce identical math.
//!
//! For each `Dispatcher` operation that has both a GPU and CPU path, this binary:
//! 1. Runs the operation via the CPU-only `Dispatcher` (no GPU)
//! 2. Runs the same operation via the GPU `Dispatcher` (if GPU available)
//! 3. Verifies numeric parity within documented tolerances
//!
//! This is the authoritative proof that `BarraCUDA` math is portable:
//! the same Rust code dispatches to CPU or GPU and produces identical results.
//!
//! ```text
//! CPU (pure Rust) ←→ GPU (`WGSL` via wgpu) ←→ future NPU / `ToadStool` dispatch
//!                   ↑ parity proven here ↑
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::many_single_char_names
)]

use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::gpu_ops::HillGateConfig;
use neural_spring::tolerances;
use neural_spring::validation::{exit_no_gpu, ValidationHarness};

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("validate_barracuda_dispatch_parity");

    let Ok(gpu) = Gpu::new().await else {
        exit_no_gpu();
    };

    let gpu_disp = Dispatcher::from_gpu(gpu);
    let cpu_disp = Dispatcher::cpu_only();

    validate_mat_mul(&mut h, &gpu_disp, &cpu_disp);
    validate_frobenius_norm(&mut h, &gpu_disp, &cpu_disp);
    validate_transpose(&mut h, &gpu_disp, &cpu_disp);
    validate_commutator(&mut h, &gpu_disp, &cpu_disp);
    validate_distance_to_normal(&mut h, &gpu_disp, &cpu_disp);
    validate_softmax(&mut h, &gpu_disp, &cpu_disp);
    validate_boltzmann(&mut h, &gpu_disp, &cpu_disp);
    validate_gelu(&mut h, &gpu_disp, &cpu_disp);
    validate_hill_activation(&mut h, &gpu_disp, &cpu_disp);
    validate_l2_distance(&mut h, &gpu_disp, &cpu_disp);
    validate_mean_variance(&mut h, &gpu_disp, &cpu_disp);
    validate_pearson(&mut h, &gpu_disp, &cpu_disp);
    validate_chi_squared(&mut h, &gpu_disp, &cpu_disp);
    validate_shannon_entropy(&mut h, &gpu_disp, &cpu_disp);
    validate_hmm_forward(&mut h, &gpu_disp, &cpu_disp);
    validate_hmm_viterbi(&mut h, &gpu_disp, &cpu_disp);
    validate_replicator(&mut h, &gpu_disp, &cpu_disp);
    validate_eigh(&mut h, &gpu_disp, &cpu_disp);
    validate_allele_frequencies(&mut h, &gpu_disp, &cpu_disp);
    validate_nucleotide_diversity(&mut h, &gpu_disp, &cpu_disp);
    validate_matrix_correlation(&mut h, &gpu_disp, &cpu_disp);
    validate_geographic_distances(&mut h, &gpu_disp, &cpu_disp);
    validate_pairwise_fst(&mut h, &gpu_disp, &cpu_disp);
    validate_global_fst(&mut h, &gpu_disp, &cpu_disp);
    validate_spectrum_chi_squared(&mut h, &gpu_disp, &cpu_disp);
    validate_selection_coefficient(&mut h, &gpu_disp, &cpu_disp);
    validate_kl_divergence(&mut h, &gpu_disp, &cpu_disp);
    validate_softmax_row_wise(&mut h, &gpu_disp, &cpu_disp);
    validate_hmm_forward_step(&mut h, &gpu_disp, &cpu_disp);
    validate_hill_gate(&mut h, &gpu_disp, &cpu_disp);
    validate_thermal_diversity(&mut h, &gpu_disp, &cpu_disp);
    validate_global_fst_variance_decomposition(&mut h, &gpu_disp, &cpu_disp);
    validate_pairwise_fst_full(&mut h, &gpu_disp, &cpu_disp);

    // S115: expanded dispatch parity — bio, ODE, HMM, popgen ops
    validate_multi_obj_fitness(&mut h, &gpu_disp, &cpu_disp);
    validate_swarm_nn_forward(&mut h, &gpu_disp, &cpu_disp);
    validate_integrate_ode_batch(&mut h, &gpu_disp, &cpu_disp);
    validate_inter_pop_af_variance(&mut h, &gpu_disp, &cpu_disp);
    validate_hmm_backward_step(&mut h, &gpu_disp, &cpu_disp);
    validate_hmm_viterbi_step(&mut h, &gpu_disp, &cpu_disp);
    validate_hmm_chain(&mut h, &gpu_disp, &cpu_disp);
    validate_detect_introgression(&mut h, &gpu_disp, &cpu_disp);

    h.finish();
}

fn validate_mat_mul(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let n = 4;
    let a: Vec<f64> = (0..n * n).map(|i| (i + 1) as f64).collect();
    let b: Vec<f64> = (0..n * n).map(|i| (i * 2 + 1) as f64).collect();
    let g = gpu.mat_mul(&a, &b, n);
    let c = cpu.mat_mul(&a, &b, n);
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "matmul CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_MATMUL_F32,
    );
}

fn validate_frobenius_norm(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let a = vec![3.0, 4.0, 5.0, 6.0];
    let g = gpu.frobenius_norm(&a);
    let c = cpu.frobenius_norm(&a);
    h.check_abs("frobenius CPU↔GPU", g, c, tolerances::TENSOR_EXACT_F32);
}

fn validate_transpose(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let n = 3;
    let a: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
    let g = gpu.transpose(&a, n);
    let c = cpu.transpose(&a, n);
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "transpose CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_commutator(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let g = gpu.commutator(&a, &b, 2);
    let c = cpu.commutator(&a, &b, 2);
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "commutator CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_MATMUL_F32,
    );
}

fn validate_distance_to_normal(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let a = vec![2.0, 1.0, 1.0, 2.0];
    let g = gpu.distance_to_normal(&a, 2);
    let c = cpu.distance_to_normal(&a, 2);
    h.check_abs(
        "dist_to_normal CPU↔GPU",
        g,
        c,
        tolerances::TENSOR_MATMUL_F32,
    );
}

fn validate_softmax(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let g = gpu.softmax(&x);
    let c = cpu.softmax(&x);
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "softmax CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_boltzmann(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let f = vec![0.1, 0.5, 0.9, 1.3];
    let g = gpu.boltzmann(&f, 2.0);
    let c = cpu.boltzmann(&f, 2.0);
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "boltzmann CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_gelu(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let g = gpu.gelu(&x);
    let c = cpu.gelu(&x);
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "gelu CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_hill_activation(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let x = vec![0.1, 0.5, 1.0, 2.0, 5.0];
    let g = gpu.hill_activation_batch(&x, 1.0, 0.5, 2.0);
    let c = cpu.hill_activation_batch(&x, 1.0, 0.5, 2.0);
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "hill_activation CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_l2_distance(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let g = gpu.l2_distance(&a, &b);
    let c = cpu.l2_distance(&a, &b);
    h.check_abs("l2_distance CPU↔GPU", g, c, tolerances::TENSOR_EXACT_F32);
}

fn validate_mean_variance(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let g_mean = gpu.mean(&data);
    let c_mean = cpu.mean(&data);
    h.check_abs("mean CPU↔GPU", g_mean, c_mean, tolerances::TENSOR_EXACT_F32);

    let g_var = gpu.variance(&data);
    let c_var = cpu.variance(&data);
    h.check_abs(
        "variance CPU↔GPU",
        g_var,
        c_var,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_pearson(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];
    let g = gpu.pearson_correlation(&x, &y);
    let c = cpu.pearson_correlation(&x, &y);
    h.check_abs("pearson CPU↔GPU", g, c, tolerances::TENSOR_EXACT_F32);
}

fn validate_chi_squared(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let obs = vec![10.0, 20.0, 30.0, 40.0];
    let exp = vec![25.0, 25.0, 25.0, 25.0];
    let g = gpu.chi_squared(&obs, &exp);
    let c = cpu.chi_squared(&obs, &exp);
    h.check_abs("chi_squared CPU↔GPU", g, c, tolerances::TENSOR_EXACT_F32);
}

fn validate_shannon_entropy(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let g = gpu.shannon_entropy(&p);
    let c = cpu.shannon_entropy(&p);
    h.check_abs(
        "entropy CPU↔GPU",
        g,
        c,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_hmm_forward(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let trans = vec![0.7, 0.3, 0.4, 0.6];
    let emission = vec![0.1, 0.4, 0.5, 0.6, 0.3, 0.1];
    let initial = vec![0.6, 0.4];
    let obs = vec![0, 1, 2, 0, 1];
    let g = gpu.hmm_forward_chain(&initial, &trans, &emission, &obs, 2, 3);
    let c = cpu.hmm_forward_chain(&initial, &trans, &emission, &obs, 2, 3);
    h.check_abs(
        "hmm_forward_chain CPU↔GPU",
        g,
        c,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_hmm_viterbi(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let trans = vec![0.7, 0.3, 0.4, 0.6];
    let emission = vec![0.1, 0.4, 0.5, 0.6, 0.3, 0.1];
    let initial = vec![0.6, 0.4];
    let obs = vec![0, 1, 2, 0, 1];
    let (g_path, g_prob) = gpu.hmm_viterbi_chain(&initial, &trans, &emission, &obs, 2, 3);
    let (c_path, c_prob) = cpu.hmm_viterbi_chain(&initial, &trans, &emission, &obs, 2, 3);
    h.check_bool("hmm_viterbi path CPU↔GPU", g_path == c_path);
    h.check_abs(
        "hmm_viterbi prob CPU↔GPU",
        g_prob,
        c_prob,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_replicator(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let freq = [0.6, 0.4];
    let payoff = [[3.0, 0.0], [5.0, 1.0]];
    let g = gpu.replicator_step(&freq, &payoff, 0.01);
    let c = cpu.replicator_step(&freq, &payoff, 0.01);
    h.check_abs(
        "replicator[0] CPU↔GPU",
        g[0],
        c[0],
        tolerances::TENSOR_EXACT_F32,
    );
    h.check_abs(
        "replicator[1] CPU↔GPU",
        g[1],
        c[1],
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_eigh(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let a = vec![4.0, 1.0, 1.0, 3.0];
    let (g_vals, _) = gpu.eigh(&a, 2);
    let (c_vals, _) = cpu.eigh(&a, 2);
    let mut g_sorted = g_vals;
    let mut c_sorted = c_vals;
    g_sorted.sort_by(f64::total_cmp);
    c_sorted.sort_by(f64::total_cmp);
    for (i, (&g, &c)) in g_sorted.iter().zip(c_sorted.iter()).enumerate() {
        h.check_abs(
            &format!("eigh λ[{i}] CPU↔GPU"),
            g,
            c,
            tolerances::TENSOR_MATMUL_F32,
        );
    }
}

fn validate_allele_frequencies(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let pop = vec![2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.5, 0.5];
    let g = gpu.allele_frequencies(&pop, 4, 2);
    let c = cpu.allele_frequencies(&pop, 4, 2);
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "allele_freq CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_nucleotide_diversity(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let pop = vec![0.0, 1.0, 1.0, 0.0, 0.5, 0.5];
    let g = gpu.nucleotide_diversity(&pop, 3, 2);
    let c = cpu.nucleotide_diversity(&pop, 3, 2);
    h.check_abs(
        "nucleotide_diversity CPU↔GPU",
        g,
        c,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_matrix_correlation(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let a = vec![0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0];
    let b = vec![0.0, 2.0, 4.0, 2.0, 0.0, 6.0, 4.0, 6.0, 0.0];
    let g = gpu.matrix_correlation(&a, &b, 3);
    let c = cpu.matrix_correlation(&a, &b, 3);
    h.check_abs(
        "matrix_correlation CPU↔GPU",
        g,
        c,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_geographic_distances(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let coords = vec![(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)];
    let g = gpu.geographic_distances(&coords);
    let c = cpu.geographic_distances(&coords);
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "geographic_dist CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_pairwise_fst(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    // Large populations with dense allele data minimize WC estimator
    // sensitivity to f32 intermediate precision in the GPU path.
    let n = 20;
    let n_loci = 4;
    let pop_a: Vec<f64> = (0..n * n_loci)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let pop_b: Vec<f64> = (0..n * n_loci)
        .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
        .collect();
    let g = gpu.pairwise_fst(&pop_a, n, &pop_b, n, n_loci);
    let c = cpu.pairwise_fst(&pop_a, n, &pop_b, n, n_loci);
    h.check_abs(
        "pairwise_fst CPU↔GPU",
        g,
        c,
        tolerances::GPU_FST_PAIRWISE_F32,
    );
}

fn validate_global_fst(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let pop1 = vec![2.0, 0.0, 2.0, 0.0, 2.0, 0.0];
    let pop2 = vec![0.0, 2.0, 0.0, 2.0, 0.0, 2.0];
    let pops = vec![pop1, pop2];
    let g = gpu.global_fst(&pops, &[3, 3], 2);
    let c = cpu.global_fst(&pops, &[3, 3], 2);
    h.check_abs("global_fst CPU↔GPU", g, c, tolerances::TENSOR_EXACT_F32);
}

fn validate_spectrum_chi_squared(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let obs = vec![30.0, 25.0, 25.0, 20.0];
    let fracs = vec![0.25, 0.25, 0.25, 0.25];
    let g = gpu.spectrum_chi_squared(&obs, &fracs);
    let c = cpu.spectrum_chi_squared(&obs, &fracs);
    h.check_abs("spectrum_chi2 CPU↔GPU", g, c, tolerances::TENSOR_EXACT_F32);
}

fn validate_selection_coefficient(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let obs = vec![30.0, 25.0, 25.0, 20.0];
    let neutral = vec![0.25, 0.25, 0.25, 0.25];
    let g = gpu.selection_coefficient(&obs, &neutral);
    let c = cpu.selection_coefficient(&obs, &neutral);
    h.check_abs(
        "selection_coeff CPU↔GPU",
        g,
        c,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_kl_divergence(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let p = vec![0.4, 0.3, 0.2, 0.1];
    let q = vec![0.25, 0.25, 0.25, 0.25];
    let g = gpu.kl_divergence(&p, &q);
    let c = cpu.kl_divergence(&p, &q);
    h.check_abs(
        "kl_divergence CPU↔GPU",
        g,
        c,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_softmax_row_wise(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let g = gpu.softmax_row_wise(&matrix, 3, 3);
    let c = cpu.softmax_row_wise(&matrix, 3, 3);
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "softmax_row_wise CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
    let row_sum: f64 = g[0..3].iter().sum();
    h.check_abs(
        "softmax_row_wise row[0] sums to 1",
        row_sum,
        1.0,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_hmm_forward_step(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let alpha_prev = vec![0.6, 0.4];
    let trans = vec![0.7, 0.3, 0.4, 0.6];
    let emit_col = vec![0.1, 0.6];
    let (g_alpha, g_scale) = gpu.hmm_forward_step(&alpha_prev, &trans, &emit_col, 2);
    let (c_alpha, c_scale) = cpu.hmm_forward_step(&alpha_prev, &trans, &emit_col, 2);
    let max_diff = g_alpha
        .iter()
        .zip(c_alpha.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "hmm_forward_step alpha CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
    h.check_abs(
        "hmm_forward_step scale CPU↔GPU",
        g_scale,
        c_scale,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_hill_gate(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let a = vec![0.1, 0.5, 1.0, 2.0];
    let b = vec![0.2, 0.4, 0.8, 1.6];
    let cfg = HillGateConfig {
        vmax: 1.0,
        k_a: 0.5,
        k_b: 0.5,
        n_a: 2.0,
        n_b: 2.0,
    };
    let g = gpu.hill_gate(&a, &b, &cfg);
    let c = cpu.hill_gate(&a, &b, &cfg);
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "hill_gate CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_thermal_diversity(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let pi = vec![0.3, 0.5, 0.7, 0.9];
    let temps = vec![1000.0, 5000.0, 10000.0, 50000.0];
    let g = gpu.thermal_diversity_correlation(&pi, &temps);
    let c = cpu.thermal_diversity_correlation(&pi, &temps);
    h.check_abs(
        "thermal_diversity_corr CPU↔GPU",
        g,
        c,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_global_fst_variance_decomposition(
    h: &mut ValidationHarness,
    gpu: &Dispatcher,
    cpu: &Dispatcher,
) {
    let pop1 = vec![2.0, 0.0, 2.0, 0.0, 2.0, 0.0];
    let pop2 = vec![0.0, 2.0, 0.0, 2.0, 0.0, 2.0];
    let pops = vec![pop1, pop2];
    let g = gpu.global_fst_variance_decomposition(&pops, &[3, 3], 2);
    let c = cpu.global_fst_variance_decomposition(&pops, &[3, 3], 2);
    h.check_abs(
        "global_fst_var_decomp CPU↔GPU",
        g,
        c,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_pairwise_fst_full(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let n = 20;
    let n_loci = 4;
    let pop_a: Vec<f64> = (0..n * n_loci)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let pop_b: Vec<f64> = (0..n * n_loci)
        .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
        .collect();
    let (g_fst, g_fis, g_fit) = gpu.pairwise_fst_full(&pop_a, n, &pop_b, n, n_loci);
    let (c_fst, c_fis, c_fit) = cpu.pairwise_fst_full(&pop_a, n, &pop_b, n, n_loci);
    h.check_abs(
        "pairwise_fst_full FST CPU↔GPU",
        g_fst,
        c_fst,
        tolerances::GPU_FST_PAIRWISE_F32,
    );
    h.check_abs(
        "pairwise_fst_full FIS CPU↔GPU",
        g_fis,
        c_fis,
        tolerances::GPU_FST_PAIRWISE_F32,
    );
    h.check_abs(
        "pairwise_fst_full FIT CPU↔GPU",
        g_fit,
        c_fit,
        tolerances::GPU_FST_PAIRWISE_F32,
    );
}

// ─── S115: expanded dispatch parity — bio, ODE, HMM, popgen ─────

fn validate_multi_obj_fitness(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let pop_size = 8;
    let genome_len = 6;
    let n_objectives = 3;
    let genotypes: Vec<f64> = (0..pop_size * genome_len)
        .map(|i| (i as f64).mul_add(0.1, 0.05).sin().abs())
        .collect();
    let g = gpu.multi_obj_fitness(&genotypes, pop_size, genome_len, n_objectives);
    let c = cpu.multi_obj_fitness(&genotypes, pop_size, genome_len, n_objectives);
    assert_eq!(g.len(), c.len(), "multi_obj_fitness length mismatch");
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "multi_obj_fitness CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_MATMUL_F32,
    );
}

fn validate_swarm_nn_forward(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    use neural_spring::gpu_ops::SwarmNnDims;

    let dims = SwarmNnDims {
        n_controllers: 4,
        n_evals: 2,
        input_dim: 1,
        hidden_dim: 4,
        output_dim: 5,
    };
    let weights_per = dims.input_dim * dims.hidden_dim
        + dims.hidden_dim
        + dims.hidden_dim * dims.output_dim
        + dims.output_dim;
    let weights: Vec<f64> = (0..dims.n_controllers * weights_per)
        .map(|i| (i as f64 * 0.3).sin())
        .collect();
    let inputs: Vec<f64> = (0..dims.n_controllers * dims.n_evals * dims.input_dim)
        .map(|i| (i as f64 * 0.7).cos())
        .collect();
    let g = gpu.swarm_nn_forward(&weights, &inputs, &dims);
    let c = cpu.swarm_nn_forward(&weights, &inputs, &dims);
    h.check_bool("swarm_nn_forward CPU↔GPU action vectors match", g == c);
}

fn validate_integrate_ode_batch(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let dim = 2;
    let n_systems = 4;
    let n_steps = 100;
    let dt = 0.01;
    let states: Vec<f64> = (0..n_systems * dim)
        .map(|i| (i as f64).mul_add(0.05, 0.1))
        .collect();
    let n_coeffs = dim * 3;
    let coeffs: Vec<f64> = (0..n_systems * n_coeffs)
        .map(|i| ((i % n_coeffs) as f64).mul_add(0.1, 0.5))
        .collect();
    let g = gpu.integrate_ode_batch(&states, &coeffs, n_systems, dim, n_steps, dt);
    let c = cpu.integrate_ode_batch(&states, &coeffs, n_systems, dim, n_steps, dt);
    assert_eq!(g.len(), c.len(), "ODE batch output length mismatch");
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "integrate_ode_batch CPU↔GPU max diff",
        max_diff,
        tolerances::GPU_RK4_F32,
    );
}

fn validate_inter_pop_af_variance(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let n_loci = 2;
    let pop_a: Vec<f64> = vec![2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.5, 0.5];
    let pop_b: Vec<f64> = vec![0.0, 2.0, 0.5, 1.5, 2.0, 0.0, 0.5, 1.5];
    let populations: Vec<&[f64]> = vec![&pop_a, &pop_b];
    let n_individuals = vec![4, 4];
    let g = gpu.inter_population_af_variance(&populations, &n_individuals, n_loci);
    let c = cpu.inter_population_af_variance(&populations, &n_individuals, n_loci);
    h.check_abs(
        "inter_pop_af_variance CPU↔GPU",
        g,
        c,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_hmm_backward_step(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let beta_next = vec![0.5, 0.5];
    let trans = vec![0.7, 0.3, 0.4, 0.6];
    let emit_col = vec![0.1, 0.6];
    let scale = 0.5;
    let g = gpu.hmm_backward_step(&beta_next, &trans, &emit_col, scale, 2);
    let c = cpu.hmm_backward_step(&beta_next, &trans, &emit_col, scale, 2);
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "hmm_backward_step CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_hmm_viterbi_step(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let delta_prev = vec![-0.5, -1.0];
    let log_trans: Vec<f64> = vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()];
    let log_emit = vec![0.1_f64.ln(), 0.6_f64.ln()];
    let (g_delta, g_psi) = gpu.hmm_viterbi_step(&delta_prev, &log_trans, &log_emit, 2);
    let (c_delta, c_psi) = cpu.hmm_viterbi_step(&delta_prev, &log_trans, &log_emit, 2);
    let max_diff = g_delta
        .iter()
        .zip(c_delta.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "hmm_viterbi_step delta CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
    h.check_bool("hmm_viterbi_step psi CPU↔GPU", g_psi == c_psi);
}

fn validate_hmm_chain(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let trans = vec![0.7, 0.3, 0.4, 0.6];
    let emission = vec![0.1, 0.4, 0.5, 0.6, 0.3, 0.1];
    let initial = vec![0.6, 0.4];
    let obs = vec![0, 1, 2, 0, 1];
    let (g_path, g_prob, g_lik) = gpu.hmm_chain(&initial, &trans, &emission, &obs, 2, 3);
    let (c_path, c_prob, c_lik) = cpu.hmm_chain(&initial, &trans, &emission, &obs, 2, 3);
    h.check_bool("hmm_chain path CPU↔GPU", g_path == c_path);
    h.check_abs(
        "hmm_chain log_prob CPU↔GPU",
        g_prob,
        c_prob,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
    h.check_abs(
        "hmm_chain log_lik CPU↔GPU",
        g_lik,
        c_lik,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_detect_introgression(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let trans = vec![0.95, 0.05, 0.10, 0.90];
    let emission = vec![0.9, 0.1, 0.1, 0.9];
    let initial = vec![0.7, 0.3];
    let hmm = neural_spring::hmm::Hmm::from_flat(trans, emission, initial, 2, 2);
    let obs = vec![0, 0, 0, 1, 1, 1, 0, 0];
    let (g_path, g_prob) = gpu.detect_introgression(&hmm, &obs);
    let (c_path, c_prob) = cpu.detect_introgression(&hmm, &obs);
    h.check_bool("detect_introgression path CPU↔GPU", g_path == c_path);
    h.check_abs(
        "detect_introgression log_prob CPU↔GPU",
        g_prob,
        c_prob,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}
