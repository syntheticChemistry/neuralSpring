// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU promotion validator: baseCamp domain-specific GPU paths.
//!
//! Validates the planned GPU promotions from the `PAPER_REVIEW_QUEUE`:
//!
//! 1. `weight_spectral` → `matmul`: GPU matmul for Hamiltonian construction
//! 2. `loss_landscape` → batch eigensolve: GPU Hessian spectral analysis
//! 3. `neural_pgm` → HMM forward chain: GPU belief propagation
//! 4. `agent_coordination` → `pairwise_l2`: GPU interaction graph distances
//!
//! Progression: Python (open data) → Rust CPU → `BarraCUDA` CPU → `BarraCUDA` GPU (this)
//!
//! Each check runs CPU (reference) → GPU (test) → compare (parity).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]

use neural_spring::anderson_localization::mean_ipr;
use neural_spring::eigh::eigh_householder_qr;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_ops;
use neural_spring::primitives::PROBABILITY_FLOOR;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{exit_no_gpu, ValidationHarness};
use neural_spring::weight_spectral;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("barracuda_basecamp_gpu_promotions");

    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "GPU: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => exit_no_gpu(),
    };
    let dev = Arc::clone(gpu.wgpu_device());
    let mut rng = Rng::new(42);

    // ═══════════════════════════════════════════════════════════════════
    // PROMOTION 1: weight_spectral → GPU matmul for Hamiltonian H·H
    // ═══════════════════════════════════════════════════════════════════
    // The Hamiltonian H has eigenvalues ±σ_i. H² = H·H has eigenvalues σ_i².
    // We validate that GPU matmul(H, H) matches CPU matmul(H, H).

    let ws_m = 16;
    let ws_n = 16;
    let ws_weights: Vec<f64> = (0..ws_m * ws_n).map(|_| rng.normal()).collect();
    let ham = weight_spectral::weight_to_hamiltonian(&ws_weights, ws_m, ws_n);
    let dim = ws_m + ws_n;

    let cpu_h_sq = {
        let mut c = vec![0.0; dim * dim];
        for i in 0..dim {
            for k in 0..dim {
                let a_ik = ham[i * dim + k];
                for j in 0..dim {
                    c[i * dim + j] += a_ik * ham[k * dim + j];
                }
            }
        }
        c
    };
    let gpu_h_sq = gpu_ops::mat_mul_gpu(&ham, &ham, dim, &dev).expect("matmul H² GPU");

    let h_sq_diff = cpu_h_sq
        .iter()
        .zip(gpu_h_sq.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "WS-GPU: H² matmul parity (Hamiltonian squared)",
        h_sq_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );

    // GPU eigensolve of H² to get σ_i² directly
    let (cpu_evals, _) = {
        let d = eigh_householder_qr(&cpu_h_sq, dim);
        let mut ev = d.eigenvalues;
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        (ev, d.eigenvectors)
    };
    let (gpu_evals_raw, _) = gpu_ops::eigh_gpu(&gpu_h_sq, dim, &dev).expect("eigh H² GPU");
    let mut gpu_evals = gpu_evals_raw;
    gpu_evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let h_sq_eval_diff = cpu_evals
        .iter()
        .zip(gpu_evals.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "WS-GPU: H² eigenvalues CPU vs GPU",
        h_sq_eval_diff < tolerances::GPU_EIGH_DISPATCH_F64,
    );

    // All eigenvalues of H² must be non-negative (σ_i² ≥ 0)
    h.check_bool(
        "WS-GPU: H² eigenvalues are non-negative (σ² ≥ 0)",
        gpu_evals.iter().all(|&e| e > -0.01),
    );

    // GPU IPR on H² eigenvectors matches CPU
    let cpu_ipr_h = {
        let d = eigh_householder_qr(&ham, dim);
        mean_ipr(&d.eigenvectors, dim)
    };
    let (_, gpu_evecs_h) = gpu_ops::eigh_gpu(&ham, dim, &dev).expect("eigh H GPU");
    let gpu_ipr_h = mean_ipr(&gpu_evecs_h, dim);
    h.check_abs(
        "WS-GPU: IPR parity (H eigenvectors)",
        gpu_ipr_h,
        cpu_ipr_h,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    // Spectral variance via GPU
    let cpu_var_evals = {
        let mean = cpu_evals.iter().sum::<f64>() / cpu_evals.len() as f64;
        cpu_evals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / cpu_evals.len() as f64
    };
    let gpu_var_evals = gpu_ops::variance_gpu(&gpu_evals, &dev).expect("variance H² evals GPU");
    h.check_abs(
        "WS-GPU: H² eigenvalue variance CPU vs GPU",
        gpu_var_evals,
        cpu_var_evals,
        tolerances::GPU_VARIANCE_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // PROMOTION 2: loss_landscape → GPU Hessian spectral analysis
    // ═══════════════════════════════════════════════════════════════════
    // For GPU promotion we: compute Hessian on CPU (function evals are inherently
    // serial), then eigensolve the Hessian on GPU and compare with CPU eigensolve.

    let loss_dim = 8;
    let loss_params: Vec<f64> = (0..loss_dim).map(|_| rng.normal()).collect();
    let loss_fn = |x: &[f64]| -> f64 {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| f64::from(i as u16 + 1) * xi * xi)
            .sum()
    };

    let hessian = neural_spring::loss_landscape::numerical_hessian(&loss_fn, &loss_params, 1e-5);
    let cpu_hess_evals = neural_spring::loss_landscape::hessian_spectrum(&hessian, loss_dim);

    let (gpu_hess_evals_raw, _) =
        gpu_ops::eigh_gpu(&hessian, loss_dim, &dev).expect("eigh Hessian GPU");
    let mut gpu_hess_evals = gpu_hess_evals_raw;
    gpu_hess_evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let hess_eval_diff = cpu_hess_evals
        .iter()
        .zip(gpu_hess_evals.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "LL-GPU: Hessian eigenvalues CPU vs GPU",
        hess_eval_diff < tolerances::GPU_EIGH_DISPATCH_F64,
    );

    // For diagonal quadratic f(x) = Σ_i a_i x_i², H = diag(2a_i)
    // Eigenvalues should be 2, 4, 6, 8, 10, 12, 14, 16
    h.check_bool(
        "LL-GPU: smallest eigenvalue ≈ 2.0 (analytical)",
        (gpu_hess_evals[0] - 2.0).abs() < 0.1,
    );
    h.check_bool(
        "LL-GPU: largest eigenvalue ≈ 16.0 (analytical)",
        (gpu_hess_evals[loss_dim - 1] - 16.0).abs() < 0.1,
    );

    // Spectral entropy of Hessian eigenvalues via GPU
    let hess_probs: Vec<f64> = {
        let abs_evals: Vec<f64> = gpu_hess_evals
            .iter()
            .map(|&v| v.abs().max(PROBABILITY_FLOOR))
            .collect();
        let sum: f64 = abs_evals.iter().sum();
        abs_evals.iter().map(|&v| v / sum).collect()
    };
    let cpu_hess_entropy = neural_spring::primitives::shannon_entropy(&hess_probs);
    let gpu_hess_entropy =
        gpu_ops::shannon_entropy_gpu(&hess_probs, &dev).expect("entropy Hessian GPU");
    h.check_abs(
        "LL-GPU: Hessian spectral entropy CPU vs GPU",
        gpu_hess_entropy,
        cpu_hess_entropy,
        tolerances::GPU_ENTROPY_F64,
    );

    // GPU matmul: H×x for gradient-like computation
    let hess_col: Vec<f64> = (0..loss_dim).map(|_| rng.normal()).collect();
    let cpu_hx = {
        let mut r = vec![0.0; loss_dim];
        for i in 0..loss_dim {
            for j in 0..loss_dim {
                r[i] += hessian[i * loss_dim + j] * hess_col[j];
            }
        }
        r
    };
    let mut padded_b = vec![0.0; loss_dim * loss_dim];
    for i in 0..loss_dim {
        padded_b[i * loss_dim] = hess_col[i];
    }
    let gpu_hx_full =
        gpu_ops::mat_mul_gpu(&hessian, &padded_b, loss_dim, &dev).expect("matmul H×x GPU");
    let gpu_hx: Vec<f64> = (0..loss_dim).map(|i| gpu_hx_full[i * loss_dim]).collect();
    let hx_diff = cpu_hx
        .iter()
        .zip(gpu_hx.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "LL-GPU: Hessian-vector product CPU vs GPU",
        hx_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // PROMOTION 3: neural_pgm → GPU HMM forward for belief propagation
    // ═══════════════════════════════════════════════════════════════════
    // Belief propagation chain is mat-vec multiplication over layers.
    // GPU promotion: HMM forward chain (same math: α_t = T × α_{t-1}).

    let pgm_states = 4;
    let pgm_obs_sym = 4;
    let pgm_initial: Vec<f64> = {
        let raw: Vec<f64> = (0..pgm_states)
            .map(|_| rng.uniform().max(PROBABILITY_FLOOR))
            .collect();
        let s: f64 = raw.iter().sum();
        raw.iter().map(|&v| v / s).collect()
    };

    // Stochastic transition matrix (row-stochastic)
    let pgm_transition: Vec<f64> = {
        let mut t = Vec::with_capacity(pgm_states * pgm_states);
        for _ in 0..pgm_states {
            let raw: Vec<f64> = (0..pgm_states)
                .map(|_| rng.uniform().max(PROBABILITY_FLOOR))
                .collect();
            let s: f64 = raw.iter().sum();
            t.extend(raw.iter().map(|&v| v / s));
        }
        t
    };

    // Emission matrix
    let pgm_emission: Vec<f64> = {
        let mut e = Vec::with_capacity(pgm_states * pgm_obs_sym);
        for _ in 0..pgm_states {
            let raw: Vec<f64> = (0..pgm_obs_sym)
                .map(|_| rng.uniform().max(PROBABILITY_FLOOR))
                .collect();
            let s: f64 = raw.iter().sum();
            e.extend(raw.iter().map(|&v| v / s));
        }
        e
    };

    let observations = [0_usize, 1, 2, 3, 0, 1, 2, 3, 0, 1];

    // CPU HMM forward chain via neuralSpring Hmm struct
    let cpu_hmm = neural_spring::hmm::Hmm::from_flat(
        pgm_transition.clone(),
        pgm_emission.clone(),
        pgm_initial.clone(),
        pgm_states,
        pgm_obs_sym,
    );
    let (_, cpu_ll) = cpu_hmm.forward(&observations);

    // GPU HMM forward chain
    let gpu_ll = gpu_ops::hmm_forward_chain_gpu(
        &pgm_initial,
        &pgm_transition,
        &pgm_emission,
        &observations,
        pgm_states,
        pgm_obs_sym,
        &dev,
    )
    .expect("hmm_forward_chain_gpu");

    h.check_abs(
        "PGM-GPU: HMM log-likelihood CPU vs GPU",
        gpu_ll,
        cpu_ll,
        tolerances::GPU_HMM_LOG_LIKELIHOOD_F32,
    );
    h.check_bool(
        "PGM-GPU: log-likelihood is finite and negative",
        gpu_ll.is_finite() && gpu_ll < 0.0,
    );

    // CPU belief propagation chain for comparison
    let transition_ref: &[f64] = &pgm_transition;
    let cpu_bp = neural_spring::neural_pgm::belief_propagation_chain(
        &pgm_initial,
        &[transition_ref, transition_ref, transition_ref],
        &[pgm_states, pgm_states, pgm_states, pgm_states],
    );
    h.check_bool(
        "PGM-GPU: BP chain produces 3 output layers",
        cpu_bp.len() == 3,
    );

    // GPU mat-vec for a single BP step: out = T × input
    let bp_out_cpu = &cpu_bp[0];
    let mut padded_input = vec![0.0; pgm_states * pgm_states];
    for i in 0..pgm_states {
        padded_input[i * pgm_states] = pgm_initial[i];
    }
    let gpu_bp_full = gpu_ops::mat_mul_gpu(&pgm_transition, &padded_input, pgm_states, &dev)
        .expect("matmul BP step GPU");
    let gpu_bp_out: Vec<f64> = (0..pgm_states)
        .map(|i| gpu_bp_full[i * pgm_states])
        .collect();
    let bp_out_sum: f64 = gpu_bp_out.iter().sum();
    let gpu_bp_normed: Vec<f64> = gpu_bp_out.iter().map(|&v| v / bp_out_sum).collect();

    let bp_step_diff = bp_out_cpu
        .iter()
        .zip(gpu_bp_normed.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "PGM-GPU: single BP step matmul parity",
        bp_step_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );

    // KL divergence between GPU and CPU BP outputs
    let cpu_kl_bp = neural_spring::neural_pgm::pgm_nn_divergence(bp_out_cpu, &gpu_bp_normed);
    h.check_bool("PGM-GPU: KL(CPU_BP || GPU_BP) near zero", cpu_kl_bp < 0.01);

    // ═══════════════════════════════════════════════════════════════════
    // PROMOTION 4: agent_coordination → GPU pairwise L2
    // ═══════════════════════════════════════════════════════════════════
    // interaction_graph computes pairwise Euclidean distances between agents.
    // GPU promotion: PairwiseL2Gpu for the O(n²) distance computation.

    let n_agents = 16;
    let agent_dim = 3;
    let agent_positions: Vec<f64> = (0..n_agents * agent_dim).map(|_| rng.normal()).collect();

    // CPU pairwise L2 distances (upper triangle)
    let n_pairs = n_agents * (n_agents - 1) / 2;
    let cpu_l2_pairs: Vec<f64> = {
        let mut pairs = Vec::with_capacity(n_pairs);
        for i in 0..n_agents {
            for j in (i + 1)..n_agents {
                let d: f64 = (0..agent_dim)
                    .map(|k| {
                        let diff =
                            agent_positions[i * agent_dim + k] - agent_positions[j * agent_dim + k];
                        diff * diff
                    })
                    .sum::<f64>()
                    .sqrt();
                pairs.push(d);
            }
        }
        pairs
    };

    let gpu_l2_pairs = gpu_ops::pairwise_l2_matrix_gpu(&agent_positions, n_agents, agent_dim, &dev)
        .expect("pairwise_l2_gpu");

    h.check_bool(
        "AC-GPU: pairwise L2 count matches",
        gpu_l2_pairs.len() == cpu_l2_pairs.len(),
    );

    let l2_max_diff = cpu_l2_pairs
        .iter()
        .zip(gpu_l2_pairs.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "AC-GPU: pairwise L2 distances CPU vs GPU",
        l2_max_diff < tolerances::GPU_L2_DISPATCH_F32,
    );

    // Verify distance properties
    h.check_bool(
        "AC-GPU: all GPU distances non-negative",
        gpu_l2_pairs.iter().all(|&d| d >= 0.0),
    );

    // Build Laplacian from GPU distances (via agent_coordination module)
    let agents: Vec<neural_spring::agent_coordination::Agent> = (0..n_agents)
        .map(|i| neural_spring::agent_coordination::Agent {
            position: agent_positions[i * agent_dim..(i + 1) * agent_dim].to_vec(),
            capability: 1.0,
            signal_level: 1.0,
            cooperating: true,
        })
        .collect();
    let cpu_adj = neural_spring::agent_coordination::interaction_graph(&agents, 5.0);
    let cpu_laplacian = neural_spring::agent_coordination::graph_laplacian(&cpu_adj, n_agents);
    let cpu_lap_decomp = eigh_householder_qr(&cpu_laplacian, n_agents);
    let mut cpu_lap_evals = cpu_lap_decomp.eigenvalues;
    cpu_lap_evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let (gpu_lap_evals_raw, _) =
        gpu_ops::eigh_gpu(&cpu_laplacian, n_agents, &dev).expect("eigh Laplacian GPU");
    let mut gpu_lap_evals = gpu_lap_evals_raw;
    gpu_lap_evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lap_eval_diff = cpu_lap_evals
        .iter()
        .zip(gpu_lap_evals.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "AC-GPU: Laplacian eigenvalues CPU vs GPU",
        lap_eval_diff < tolerances::GPU_EIGH_DISPATCH_F64,
    );

    h.check_bool(
        "AC-GPU: smallest Laplacian eigenvalue ≈ 0 (connected graph)",
        gpu_lap_evals[0].abs() < 0.5,
    );

    // GPU chi² for agent distribution comparison
    let agent_obs: Vec<f64> = (0..8).map(|_| rng.uniform() * 100.0).collect();
    let agent_mean = agent_obs.iter().sum::<f64>() / agent_obs.len() as f64;
    let agent_exp = vec![agent_mean; 8];
    let cpu_chi2 = barracuda::special::chi_squared_statistic(&agent_obs, &agent_exp).unwrap_or(0.0);
    let gpu_chi2 = gpu_ops::chi_squared_gpu(&agent_obs, &agent_exp, &dev).expect("chi² agent GPU");
    h.check_abs(
        "AC-GPU: chi² distribution test CPU vs GPU",
        gpu_chi2,
        cpu_chi2,
        tolerances::GPU_CHI_SQUARED_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // CROSS-MODULE: GPU determinism across baseCamp operations
    // ═══════════════════════════════════════════════════════════════════

    let (evals_a, _) = gpu_ops::eigh_gpu(&ham, dim, &dev).expect("determinism check a");
    let (evals_b, _) = gpu_ops::eigh_gpu(&ham, dim, &dev).expect("determinism check b");
    let det_diff = evals_a
        .iter()
        .zip(evals_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool("CROSS: GPU eigensolve determinism", det_diff < 1e-15);

    let gpu_c1 = gpu_ops::mat_mul_gpu(&ham, &ham, dim, &dev).expect("determinism matmul a");
    let gpu_c2 = gpu_ops::mat_mul_gpu(&ham, &ham, dim, &dev).expect("determinism matmul b");
    let matmul_det_diff = gpu_c1
        .iter()
        .zip(gpu_c2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool("CROSS: GPU matmul determinism", matmul_det_diff < 1e-15);

    h.finish();
}
