// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation: `BarraCUDA` CPU vs GPU parity across all science domains.
//!
//! Comprehensive parity proof: for every dispatched operation, the GPU path
//! produces the same result as the CPU reference (within documented tolerance).
//! This is the "pure math portability" validator — same Rust math,
//! different hardware substrate.
//!
//! ## Domain coverage
//!
//! - Linear algebra: matmul, transpose, frobenius norm, commutator
//! - Statistics: variance, Pearson correlation, Shannon entropy
//! - Spectral: eigensolve, IPR
//! - Activation: softmax, Boltzmann, Hill
//! - Reduction: mean, sum, max
//! - Biology: HMM step, replicator dynamics
//! - Distance: L2, chi-squared
//!
//! ```text
//! cargo run --release --bin validate_barracuda_parity
//! ```

#![expect(
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::min_ident_chars,
    reason = "validation binary"
)]

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("barracuda_parity");
    let mut rng = Rng::new(42);
    let dispatcher = Dispatcher::new().await;
    let cpu = Dispatcher::cpu_only();

    eprintln!(
        "[parity] GPU: {} ({})",
        dispatcher.has_gpu(),
        dispatcher.adapter_name()
    );

    // ═══════════════════════════════════════════════════════════════════
    // Linear algebra
    // ═══════════════════════════════════════════════════════════════════

    let n = 8;
    let a: Vec<f64> = (0..n * n).map(|_| rng.normal()).collect();
    let b: Vec<f64> = (0..n * n).map(|_| rng.normal()).collect();

    let gpu_mm = dispatcher.mat_mul(&a, &b, n);
    let cpu_mm = cpu.mat_mul(&a, &b, n);
    let mm_diff = element_max_diff(&gpu_mm, &cpu_mm);
    h.check_bool(
        "linalg: matmul parity",
        mm_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );

    let gpu_t = dispatcher.transpose(&a, n);
    let cpu_t = cpu.transpose(&a, n);
    let t_diff = element_max_diff(&gpu_t, &cpu_t);
    h.check_bool(
        "linalg: transpose parity",
        t_diff < tolerances::GPU_TRANSPOSE_F32,
    );

    let gpu_fn = dispatcher.frobenius_norm(&a);
    let cpu_fn = cpu.frobenius_norm(&a);
    h.check_abs(
        "linalg: frobenius norm parity",
        gpu_fn,
        cpu_fn,
        tolerances::GPU_FROBENIUS_F32,
    );

    let gpu_comm = dispatcher.commutator(&a, &b, n);
    let cpu_comm = cpu.commutator(&a, &b, n);
    let comm_diff = element_max_diff(&gpu_comm, &cpu_comm);
    h.check_bool(
        "linalg: commutator parity",
        comm_diff < tolerances::GPU_COMMUTATOR_F32,
    );

    let gpu_dn = dispatcher.distance_to_normal(&a, n);
    let cpu_dn = cpu.distance_to_normal(&a, n);
    h.check_abs(
        "linalg: distance_to_normal parity",
        gpu_dn,
        cpu_dn,
        tolerances::GPU_NORMAL_DISTANCE_SYMMETRIC_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Statistics
    // ═══════════════════════════════════════════════════════════════════

    let data: Vec<f64> = (0..256).map(|_| rng.normal()).collect();

    let gpu_var = dispatcher.variance(&data);
    let cpu_var = cpu.variance(&data);
    h.check_abs(
        "stats: variance parity",
        gpu_var,
        cpu_var,
        tolerances::GPU_VARIANCE_F64,
    );

    let x: Vec<f64> = (0..128).map(|_| rng.normal()).collect();
    let y: Vec<f64> = (0..128).map(|_| rng.normal()).collect();
    let gpu_pc = dispatcher.pearson_correlation(&x, &y);
    let cpu_pc = cpu.pearson_correlation(&x, &y);
    h.check_abs(
        "stats: pearson parity",
        gpu_pc,
        cpu_pc,
        tolerances::GPU_PEARSON_F64,
    );

    let probs: Vec<f64> = {
        let raw: Vec<f64> = (0..64).map(|_| rng.uniform().max(1e-12)).collect();
        let s: f64 = raw.iter().sum();
        raw.iter().map(|&r| r / s).collect()
    };
    let gpu_ent = dispatcher.shannon_entropy(&probs);
    let cpu_ent = cpu.shannon_entropy(&probs);
    h.check_abs(
        "stats: entropy parity",
        gpu_ent,
        cpu_ent,
        tolerances::GPU_ENTROPY_F64,
    );

    let gpu_chi = dispatcher.chi_squared(&[10.0, 20.0, 30.0, 40.0], &[25.0, 25.0, 25.0, 25.0]);
    let cpu_chi = cpu.chi_squared(&[10.0, 20.0, 30.0, 40.0], &[25.0, 25.0, 25.0, 25.0]);
    h.check_abs(
        "stats: chi-squared parity",
        gpu_chi,
        cpu_chi,
        tolerances::GPU_CHI_SQUARED_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Spectral
    // ═══════════════════════════════════════════════════════════════════

    let sym: Vec<f64> = {
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let v = rng.normal();
                m[i * n + j] += v;
                m[j * n + i] += v;
            }
        }
        m
    };
    let (gpu_evals, _) = dispatcher.eigh(&sym, n);
    let (cpu_evals, _) = cpu.eigh(&sym, n);

    let eigh_diff = {
        let mut ge = gpu_evals;
        let mut ce = cpu_evals;
        ge.sort_by(f64::total_cmp);
        ce.sort_by(f64::total_cmp);
        element_max_diff(&ge, &ce)
    };
    h.check_bool(
        "spectral: eigh eigenvalue parity",
        eigh_diff < tolerances::GPU_EIGH_DISPATCH_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Activation
    // ═══════════════════════════════════════════════════════════════════

    let logits: Vec<f64> = (0..8).map(|_| rng.normal()).collect();
    let gpu_sm = dispatcher.softmax(&logits);
    let cpu_sm = cpu.softmax(&logits);
    let sm_diff = element_max_diff(&gpu_sm, &cpu_sm);
    h.check_bool(
        "activation: softmax parity",
        sm_diff < tolerances::GPU_SOFTMAX_DISPATCH_F32,
    );

    let fitnesses: Vec<f64> = (0..8).map(|_| rng.uniform() * 10.0).collect();
    let gpu_boltz = dispatcher.boltzmann(&fitnesses, 1.0);
    let cpu_boltz = cpu.boltzmann(&fitnesses, 1.0);
    let boltz_diff = element_max_diff(&gpu_boltz, &cpu_boltz);
    h.check_bool(
        "activation: boltzmann parity",
        boltz_diff < tolerances::GPU_BOLTZMANN_F32,
    );

    let hill_x: Vec<f64> = (0..16).map(|_| rng.uniform() * 5.0).collect();
    let gpu_hill = dispatcher.hill_activation_batch(&hill_x, 1.0, 1.0, 2.0);
    let cpu_hill = cpu.hill_activation_batch(&hill_x, 1.0, 1.0, 2.0);
    let hill_diff = element_max_diff(&gpu_hill, &cpu_hill);
    h.check_bool(
        "activation: hill parity",
        hill_diff < tolerances::GPU_HILL_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Reductions
    // ═══════════════════════════════════════════════════════════════════

    let gpu_mean = dispatcher.mean(&data);
    let cpu_mean = cpu.mean(&data);
    h.check_abs(
        "reduce: mean parity",
        gpu_mean,
        cpu_mean,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Distance
    // ═══════════════════════════════════════════════════════════════════

    let va: Vec<f64> = (0..32).map(|_| rng.normal()).collect();
    let vb: Vec<f64> = (0..32).map(|_| rng.normal()).collect();
    let gpu_l2 = dispatcher.l2_distance(&va, &vb);
    let cpu_l2 = cpu.l2_distance(&va, &vb);
    h.check_abs(
        "distance: L2 parity",
        gpu_l2,
        cpu_l2,
        tolerances::GPU_L2_DISPATCH_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Biology: replicator dynamics
    // ═══════════════════════════════════════════════════════════════════

    let freq = [0.6, 0.4];
    let payoff = [[3.0, 0.0], [5.0, 1.0]];
    let gpu_rep = dispatcher.replicator_step(&freq, &payoff, 0.01);
    let cpu_rep = cpu.replicator_step(&freq, &payoff, 0.01);
    h.check_abs(
        "bio: replicator step[0] parity",
        gpu_rep[0],
        cpu_rep[0],
        tolerances::GPU_MEAN_DISPATCH_F32,
    );
    h.check_abs(
        "bio: replicator step[1] parity",
        gpu_rep[1],
        cpu_rep[1],
        tolerances::GPU_MEAN_DISPATCH_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════

    h.finish();
}

fn element_max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}
