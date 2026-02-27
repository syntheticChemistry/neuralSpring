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
