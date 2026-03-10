// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU workload validation for all 5 baseCamp sub-theses.
//!
//! Each sub-thesis dispatches its core computation through the GPU-aware
//! `Dispatcher`, then compares scalar summaries against CPU references.
//!
//! ## Sub-theses validated
//!
//! | Sub | Module | GPU path |
//! |-----|--------|----------|
//! | 01 | `weight_spectral` | `eigh_gpu` (Hamiltonian eigensolve) |
//! | 02 | `information_flow` | `eigh_gpu` + `matmul_dispatch` (attention + signal) |
//! | 03 | `loss_landscape` | `eigh_gpu` (Hessian eigensolve) |
//! | 04 | `neural_pgm` | `matmul_dispatch` f64 (belief propagation GEMV) |
//! | 05 | `agent_coordination` | `pairwise_l2_matrix_gpu` (interaction graph) |
//!
//! ## Provenance
//!
//! CPU reference: neuralSpring lib (baseCamp modules Rust CPU math).
//! GPU path: `BarraCUDA` Dispatcher GPU path (`eigh_gpu`, `matmul_dispatch`, `pairwise_l2`) via wgpu.
//! Evolution: Python baseline → Rust CPU → `BarraCUDA` GPU → Pure GPU (scalar readback).

use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::primitives::PROBABILITY_FLOOR;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::time::Instant;

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let dispatcher = Dispatcher::from_gpu(gpu);
    let cpu = Dispatcher::cpu_only();
    let mut h = ValidationHarness::new("basecamp_gpu_pure");
    let t0 = Instant::now();

    validate_sub01_weight_spectral(&mut h, &dispatcher, &cpu);
    validate_sub02_information_flow(&mut h, &dispatcher, &cpu);
    validate_sub03_loss_landscape(&mut h, &dispatcher, &cpu);
    validate_sub04_neural_pgm(&mut h, &dispatcher, &cpu);
    validate_sub05_agent_coordination(&mut h, &dispatcher, &cpu);
    validate_cross_dispatch(&mut h, &dispatcher, &cpu);

    let elapsed = t0.elapsed();
    eprintln!(
        "\n  total baseCamp pure-GPU time: {:.1}ms (5 sub-theses + cross)",
        elapsed.as_secs_f64() * 1000.0,
    );

    h.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Sub-01: Weight Matrices as Disordered Hamiltonians
// ═══════════════════════════════════════════════════════════════════

fn validate_sub01_weight_spectral(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(101);
    let m = 8;
    let n = 6;
    let weights: Vec<f64> = (0..m * n).map(|_| rng.normal()).collect();

    let cpu_result = cpu.weight_spectral_analysis(&weights, m, n);
    let gpu_result = gpu.weight_spectral_analysis(&weights, m, n);

    h.check_bool(
        "Sub-01: eigenvalue count parity",
        cpu_result.eigenvalues.len() == gpu_result.eigenvalues.len(),
    );

    let max_eval_diff = cpu_result
        .eigenvalues
        .iter()
        .zip(gpu_result.eigenvalues.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "Sub-01: eigenvalue parity (GPU eigh vs CPU Householder-QR)",
        max_eval_diff < tolerances::GPU_EIGH_DISPATCH_F64,
    );

    h.check_abs(
        "Sub-01: mean IPR parity",
        gpu_result.mean_ipr,
        cpu_result.mean_ipr,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    h.check_abs(
        "Sub-01: spectral entropy parity",
        gpu_result.spectral_entropy,
        cpu_result.spectral_entropy,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    h.check_abs(
        "Sub-01: MP departure parity",
        gpu_result.mp_departure,
        cpu_result.mp_departure,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    // Non-square matrix: verify gamma is correctly propagated
    let rect_weights: Vec<f64> = (0..16 * 4).map(|_| rng.normal()).collect();
    let gpu_rect = gpu.weight_spectral_analysis(&rect_weights, 16, 4);
    h.check_bool(
        "Sub-01: non-square Hamiltonian produces correct dim",
        gpu_rect.eigenvalues.len() == 20,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Sub-02: Information Flow as Wave Propagation
// ═══════════════════════════════════════════════════════════════════

fn validate_sub02_information_flow(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(202);
    let n = 8;
    let attention: Vec<f64> = (0..n * n).map(|_| rng.uniform()).collect();

    let cpu_result = neural_spring::information_flow::attention_spectral_analysis(&attention, n);
    let gpu_result = gpu.attention_spectral_analysis(&attention, n);

    let max_eval_diff = cpu_result
        .eigenvalues
        .iter()
        .zip(gpu_result.eigenvalues.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "Sub-02: attention eigenvalue parity",
        max_eval_diff < tolerances::GPU_EIGH_DISPATCH_F64,
    );

    h.check_abs(
        "Sub-02: attention mean IPR parity",
        gpu_result.mean_ipr,
        cpu_result.mean_ipr,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    // Signal propagation via GPU matmul
    let input: Vec<f64> = (0..4).map(|_| rng.normal()).collect();
    let w1: Vec<f64> = (0..4 * 3).map(|_| rng.normal() * 0.5).collect();
    let w2: Vec<f64> = (0..3 * 2).map(|_| rng.normal() * 0.5).collect();

    let cpu_vars = cpu.mlp_signal_propagation(&input, &[&w1, &w2], &[3, 2]);
    let gpu_vars = gpu.mlp_signal_propagation(&input, &[&w1, &w2], &[3, 2]);

    h.check_bool(
        "Sub-02: signal propagation layer count parity",
        cpu_vars.len() == gpu_vars.len(),
    );

    let max_var_diff = cpu_vars
        .iter()
        .zip(gpu_vars.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "Sub-02: signal propagation variance parity",
        max_var_diff < tolerances::DISPATCH_MATMUL_F64,
    );

    // Depth scale from GPU-computed variances
    let gpu_xi = neural_spring::information_flow::depth_scale(&gpu_vars);
    let cpu_xi = neural_spring::information_flow::depth_scale(&cpu_vars);
    h.check_bool(
        "Sub-02: depth scale parity",
        (gpu_xi - cpu_xi).abs() < tolerances::DISPATCH_MATMUL_F64
            || (gpu_xi.is_infinite() && cpu_xi.is_infinite()),
    );
}

// ═══════════════════════════════════════════════════════════════════
// Sub-03: Loss Landscapes as Energy Landscapes
// ═══════════════════════════════════════════════════════════════════

fn validate_sub03_loss_landscape(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    fn quadratic(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    let params = vec![0.5, -0.3, 0.8, 0.1];
    let eps = tolerances::HESSIAN_FD_STEP;

    let cpu_result = cpu.landscape_analysis(&quadratic, &params, eps, 0.1);
    let gpu_result = gpu.landscape_analysis(&quadratic, &params, eps, 0.1);

    h.check_abs(
        "Sub-03: loss value parity",
        gpu_result.loss,
        cpu_result.loss,
        tolerances::GPU_F64_EXACT,
    );

    h.check_abs(
        "Sub-03: sharpness parity (max |eigenvalue|)",
        gpu_result.sharpness,
        cpu_result.sharpness,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    h.check_bool(
        "Sub-03: saddle index parity",
        gpu_result.saddle_index == cpu_result.saddle_index,
    );

    h.check_abs(
        "Sub-03: flatness parity",
        gpu_result.flatness,
        cpu_result.flatness,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    // Analytical: f(x) = x₁² + x₂² → H = diag(2,2), eigenvalues = 2.0 exactly.
    let expected_sharpness = 2.0;
    h.check_abs(
        "Sub-03: quadratic Hessian eigenvalue ≈ 2",
        gpu_result.sharpness,
        expected_sharpness,
        tolerances::OPTIMIZER_VALUE_AT_MIN,
    );

    h.check_bool(
        "Sub-03: origin of quadratic is a minimum (saddle_index=0)",
        gpu_result.saddle_index == 0,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Sub-04: Neural Networks as PGMs
// ═══════════════════════════════════════════════════════════════════

fn validate_sub04_neural_pgm(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(404);

    // Build transition matrices (row-stochastic)
    let n1 = 4;
    let n2 = 3;
    let n3 = 2;

    let t1 = make_stochastic(n1, n2, &mut rng);
    let t2 = make_stochastic(n2, n3, &mut rng);

    let input: Vec<f64> = {
        let raw: Vec<f64> = (0..n1)
            .map(|_| rng.uniform().max(PROBABILITY_FLOOR))
            .collect();
        let s: f64 = raw.iter().sum();
        raw.iter().map(|&v| v / s).collect()
    };

    let cpu_dists = cpu.belief_propagation(&input, &[&t1, &t2], &[n2, n3]);
    let gpu_dists = gpu.belief_propagation(&input, &[&t1, &t2], &[n2, n3]);

    h.check_bool(
        "Sub-04: belief propagation layer count parity",
        cpu_dists.len() == gpu_dists.len(),
    );

    // Check each layer's output distribution
    for (layer, (cpu_d, gpu_d)) in cpu_dists.iter().zip(gpu_dists.iter()).enumerate() {
        let max_diff = cpu_d
            .iter()
            .zip(gpu_d.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0_f64, f64::max);
        h.check_bool(
            &format!("Sub-04: BP layer {layer} parity (f64 GEMV)"),
            max_diff < tolerances::DISPATCH_MATMUL_F64,
        );
    }

    let Some(gpu_final) = gpu_dists.last() else {
        h.check_bool("Sub-04: GPU BP produced at least one layer", false);
        return;
    };
    let sum: f64 = gpu_final.iter().sum();
    h.check_abs(
        "Sub-04: GPU BP final distribution sums to 1",
        sum,
        1.0,
        tolerances::PGM_NORMALIZATION_SUM,
    );
}

fn make_stochastic(rows: usize, cols: usize, rng: &mut Rng) -> Vec<f64> {
    let mut mat = vec![0.0; rows * cols];
    for i in 0..rows {
        let mut row_sum = 0.0;
        for j in 0..cols {
            let v = rng.uniform().max(PROBABILITY_FLOOR);
            mat[i * cols + j] = v;
            row_sum += v;
        }
        for j in 0..cols {
            mat[i * cols + j] /= row_sum;
        }
    }
    mat
}

// ═══════════════════════════════════════════════════════════════════
// Sub-05: Multi-Agent AI as Quorum Sensing
// ═══════════════════════════════════════════════════════════════════

fn validate_sub05_agent_coordination(
    h: &mut ValidationHarness,
    gpu: &Dispatcher,
    cpu: &Dispatcher,
) {
    let mut rng = Rng::new(505);
    let n_agents = 8;
    let dim = 3;
    let comm_range = 5.0;

    let positions: Vec<f64> = (0..n_agents * dim).map(|_| rng.uniform() * 10.0).collect();

    let cpu_adj = cpu.agent_interaction_graph(&positions, n_agents, dim, comm_range);
    let gpu_adj = gpu.agent_interaction_graph(&positions, n_agents, dim, comm_range);

    h.check_bool(
        "Sub-05: adjacency matrix size parity",
        cpu_adj.len() == gpu_adj.len(),
    );

    let max_adj_diff = cpu_adj
        .iter()
        .zip(gpu_adj.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "Sub-05: interaction graph parity (GPU pairwise L2)",
        max_adj_diff < tolerances::GPU_MODES_L2_F32,
    );

    // Symmetry check
    for i in 0..n_agents {
        for j in 0..n_agents {
            h.check_bool(
                &format!("Sub-05: adjacency symmetric ({i},{j})"),
                (gpu_adj[i * n_agents + j] - gpu_adj[j * n_agents + i]).abs() < f64::EPSILON,
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Cross-module: Dispatcher GPU routing
// ═══════════════════════════════════════════════════════════════════

fn validate_cross_dispatch(h: &mut ValidationHarness, gpu: &Dispatcher, _cpu: &Dispatcher) {
    h.check_bool("Cross: GPU dispatcher has_gpu", gpu.has_gpu());
    h.check_bool(
        "Cross: GPU dispatcher backend is GPU",
        gpu.backend() == neural_spring::gpu_dispatch::Backend::Gpu,
    );
}
