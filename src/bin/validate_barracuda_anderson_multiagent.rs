// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: Exp-053 — Anderson Multi-Agent Coordination.
//!
//! Proves GPU parity for the multi-agent spectral pipeline:
//! interaction graph → Laplacian → disordered Laplacian → GPU eigensolve → IPR.
//!
//! Progression: Python (open data) → Rust CPU → **`BarraCUDA` GPU** (this) → pure GPU
//!
//! Papers: Sub-thesis 05 (Multi-Agent QS), Paper C (AAMAS 2027).
//!
//! ## Provenance
//!
//! Validation class: Integration.
//! Python baseline: multi-agent spectral pipeline (interaction graph → Laplacian → eigensolve → IPR).
//! Components: `agent_coordination`, `anderson_localization`, eigh, `gpu_ops`, `BarraCUDA` GPU.

#![expect(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::too_many_lines,
    reason = "validation binary"
)]

use neural_spring::agent_coordination::{
    coordination_spectral_analysis, generate_lattice_agents, graph_laplacian, interaction_graph,
};
use neural_spring::anderson_localization::mean_ipr;
use neural_spring::eigh::eigh_householder_qr;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_ops;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, exit_no_gpu};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("barracuda_anderson_multiagent_gpu");

    let gpu = match Gpu::new().await {
        Ok(g) => {
            println!(
                "GPU: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => exit_no_gpu(),
    };
    let dev = Arc::clone(gpu.wgpu_device());

    // ═══════════════════════════════════════════════════════════════════
    // 1. GPU eigensolve on agent Laplacian: clean graph (W=0)
    // ═══════════════════════════════════════════════════════════════════

    let n_side = 4;
    let dim = 3;
    let comm_range = 2.5;
    let mut rng = Rng::new(42);
    let agents = generate_lattice_agents(n_side, dim, 0.3, &mut rng);
    let n = agents.len();

    let adj = interaction_graph(&agents, comm_range);
    let lap = graph_laplacian(&adj, n);

    let cpu_decomp = eigh_householder_qr(&lap, n);
    let (gpu_evals, gpu_evecs) = gpu_ops::eigh_gpu(&lap, n, &dev).expect("eigh_gpu Laplacian");

    let mut cpu_sorted = cpu_decomp.eigenvalues.clone();
    cpu_sorted.sort_by(f64::total_cmp);
    let mut gpu_sorted = gpu_evals;
    gpu_sorted.sort_by(f64::total_cmp);

    let max_eval_diff = cpu_sorted
        .iter()
        .zip(gpu_sorted.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        "GPU Laplacian eigenvalues match CPU",
        max_eval_diff < tolerances::GPU_EIGH_DISPATCH_F64,
    );

    h.check_bool(
        "GPU: smallest Laplacian eigenvalue near 0 (connected graph)",
        gpu_sorted[0].abs() < 0.01,
    );

    let gpu_alg_conn = gpu_sorted.get(1).copied().unwrap_or(0.0);
    let cpu_alg_conn = cpu_sorted.get(1).copied().unwrap_or(0.0);

    h.check_abs(
        "GPU algebraic connectivity matches CPU",
        gpu_alg_conn,
        cpu_alg_conn,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 2. GPU IPR from Laplacian eigenvectors
    // ═══════════════════════════════════════════════════════════════════

    let cpu_ipr = mean_ipr(&cpu_decomp.eigenvectors, n);
    let gpu_ipr = mean_ipr(&gpu_evecs, n);

    h.check_abs(
        "GPU IPR matches CPU (clean Laplacian)",
        gpu_ipr,
        cpu_ipr,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 3. Disordered Laplacian: GPU vs CPU eigensolve
    // ═══════════════════════════════════════════════════════════════════

    let disorder = 4.0;
    let positions: Vec<f64> = agents
        .iter()
        .flat_map(|a| a.position.iter().copied())
        .collect();
    let pos_dim = agents.first().map_or(3, |a| a.position.len());

    let disorder_vec: Vec<f64> = {
        let mut rng_d = Rng::new(42);
        (0..n).map(|_| rng_d.uniform() * disorder).collect()
    };

    let mut disordered_lap = lap;
    for i in 0..n {
        disordered_lap[i * n + i] += disorder_vec[i];
    }

    let cpu_dis_decomp = eigh_householder_qr(&disordered_lap, n);
    let (gpu_dis_evals, gpu_dis_evecs) =
        gpu_ops::eigh_gpu(&disordered_lap, n, &dev).expect("eigh_gpu disordered");

    let mut cpu_dis_sorted = cpu_dis_decomp.eigenvalues.clone();
    cpu_dis_sorted.sort_by(f64::total_cmp);
    let mut gpu_dis_sorted = gpu_dis_evals;
    gpu_dis_sorted.sort_by(f64::total_cmp);

    let dis_max_diff = cpu_dis_sorted
        .iter()
        .zip(gpu_dis_sorted.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        "GPU disordered Laplacian eigenvalues match CPU",
        dis_max_diff < tolerances::GPU_EIGH_DISPATCH_F64,
    );

    let cpu_dis_ipr = mean_ipr(&cpu_dis_decomp.eigenvectors, n);
    let gpu_dis_ipr = mean_ipr(&gpu_dis_evecs, n);

    h.check_abs(
        "GPU disordered IPR matches CPU",
        gpu_dis_ipr,
        cpu_dis_ipr,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 4. Disorder localizes: GPU IPR(W=4) > GPU IPR(W=0)
    // ═══════════════════════════════════════════════════════════════════

    h.check_bool(
        "GPU: disorder increases IPR (W=4 > W=0)",
        gpu_dis_ipr > gpu_ipr,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 5. GPU pairwise L2 for interaction graph construction
    // ═══════════════════════════════════════════════════════════════════

    let gpu_dists =
        gpu_ops::pairwise_l2_matrix_gpu(&positions, n, pos_dim, &dev).expect("pairwise_l2");

    let mut cpu_upper_tri = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = (0..pos_dim)
                .map(|k| {
                    let diff = positions[i * pos_dim + k] - positions[j * pos_dim + k];
                    diff * diff
                })
                .sum::<f64>()
                .sqrt();
            cpu_upper_tri.push(d);
        }
    }

    let l2_max_diff = cpu_upper_tri
        .iter()
        .zip(gpu_dists.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        "GPU pairwise L2 matches CPU (interaction graph)",
        l2_max_diff < tolerances::GPU_L2_DISPATCH_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 6. Full CPU pipeline vs GPU eigensolve parity
    // ═══════════════════════════════════════════════════════════════════

    let cpu_result = coordination_spectral_analysis(&agents, comm_range, disorder);

    h.check_abs(
        "Full pipeline: GPU IPR matches CPU coordination_spectral_analysis",
        gpu_dis_ipr,
        cpu_result.mean_ipr,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 7. GPU variance on eigenvalue spectrum
    // ═══════════════════════════════════════════════════════════════════

    let cpu_var = {
        let mean = cpu_dis_sorted.iter().sum::<f64>() / n as f64;
        cpu_dis_sorted
            .iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>()
            / n as f64
    };
    let gpu_var = gpu_ops::variance_gpu(&gpu_dis_sorted, &dev).expect("variance_gpu");

    h.check_abs(
        "GPU disordered eigenvalue variance matches CPU",
        gpu_var,
        cpu_var,
        tolerances::GPU_VARIANCE_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 8. Determinism
    // ═══════════════════════════════════════════════════════════════════

    let (evals_a, _) = gpu_ops::eigh_gpu(&disordered_lap, n, &dev).expect("determinism a");
    let (evals_b, _) = gpu_ops::eigh_gpu(&disordered_lap, n, &dev).expect("determinism b");

    let det_diff = evals_a
        .iter()
        .zip(evals_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        "GPU disordered eigensolve is deterministic",
        det_diff < tolerances::NUMERICAL_DISTINCTNESS,
    );

    h.finish();
}
