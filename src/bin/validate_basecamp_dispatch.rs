// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation: baseCamp Dispatcher GPU promotion methods.
//!
//! Exercises all 5 `Dispatcher` baseCamp methods through the dispatch layer
//! (GPU when available, CPU fallback otherwise). Validates:
//!
//! 1. `weight_spectral_analysis` — Sub-01 eigensolve via dispatch
//! 2. `numerical_hessian` — Sub-03 Hessian computation
//! 3. `belief_propagation` — Sub-04 GEMV chain through dispatch
//! 4. `agent_interaction_graph` — Sub-05 pairwise L2 via dispatch
//! 5. Cross-path parity: dispatch result ≈ direct library call
//!
//! This is the "dispatch portability proof" for baseCamp — same science,
//! routed through the `Dispatcher` abstraction.

#![expect(clippy::too_many_lines, reason = "validation binary")]

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::neural_pgm::weight_to_transition;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("basecamp_dispatch");
    let mut rng = Rng::new(42);
    let dispatcher = Dispatcher::new().await;

    eprintln!(
        "[basecamp_dispatch] backend: {}, GPU: {}",
        dispatcher.backend(),
        dispatcher.has_gpu()
    );

    // ═══════════════════════════════════════════════════════════════════
    // Sub-01: Weight spectral analysis via Dispatcher
    // ═══════════════════════════════════════════════════════════════════

    let ws_rows = 8;
    let ws_cols = 8;
    let ws_w: Vec<f64> = (0..ws_rows * ws_cols).map(|_| rng.normal()).collect();

    let dispatch_result = dispatcher.weight_spectral_analysis(&ws_w, ws_rows, ws_cols);
    let direct_result =
        neural_spring::weight_spectral::weight_spectral_analysis(&ws_w, ws_rows, ws_cols);

    h.check_bool(
        "Sub-01: dispatch eigenvalue count matches direct",
        dispatch_result.eigenvalues.len() == direct_result.eigenvalues.len(),
    );

    let eval_max_diff = dispatch_result
        .eigenvalues
        .iter()
        .zip(direct_result.eigenvalues.iter())
        .map(|(d, r)| (d - r).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "Sub-01: dispatch eigenvalues match direct",
        eval_max_diff < tolerances::GPU_EIGH_DISPATCH_F64,
    );

    h.check_bool(
        "Sub-01: dispatch IPR finite and positive",
        dispatch_result.mean_ipr.is_finite() && dispatch_result.mean_ipr > 0.0,
    );
    h.check_bool(
        "Sub-01: dispatch LSR finite",
        dispatch_result.level_spacing_ratio.is_finite(),
    );
    h.check_bool(
        "Sub-01: dispatch spectral entropy finite",
        dispatch_result.spectral_entropy.is_finite(),
    );

    // ═══════════════════════════════════════════════════════════════════
    // Sub-03: Numerical Hessian via Dispatcher
    // ═══════════════════════════════════════════════════════════════════

    let rosenbrock = |x: &[f64]| {
        let a = 1.0 - x[0];
        let b = (-x[0]).mul_add(x[0], x[1]);
        a.mul_add(a, 100.0 * b * b)
    };
    let point = vec![1.0, 1.0];

    let dispatch_hessian =
        dispatcher.numerical_hessian(rosenbrock, &point, tolerances::HESSIAN_FD_STEP);
    let direct_hessian = neural_spring::loss_landscape::numerical_hessian(
        &rosenbrock,
        &point,
        tolerances::HESSIAN_FD_STEP,
    );

    h.check_bool(
        "Sub-03: dispatch Hessian size correct (2x2=4)",
        dispatch_hessian.len() == 4,
    );

    let hess_max_diff = dispatch_hessian
        .iter()
        .zip(direct_hessian.iter())
        .map(|(d, r)| (d - r).abs())
        .fold(0.0_f64, f64::max);
    h.check_abs(
        "Sub-03: dispatch Hessian matches direct",
        hess_max_diff,
        0.0,
        tolerances::EXACT_F64,
    );

    // Analytical: at (1,1) the Rosenbrock Hessian is [[802, -400], [-400, 200]]
    h.check_abs(
        "Sub-03: H[0,0] ≈ 802 (Rosenbrock at minimum)",
        dispatch_hessian[0],
        802.0,
        tolerances::HESSIAN_FD_ABS,
    );
    h.check_abs(
        "Sub-03: H[1,1] ≈ 200 (Rosenbrock at minimum)",
        dispatch_hessian[3],
        200.0,
        tolerances::HESSIAN_FD_ABS,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Sub-04: Belief propagation via Dispatcher
    // ═══════════════════════════════════════════════════════════════════

    let input_dist = vec![0.25, 0.25, 0.25, 0.25];
    let w1: Vec<f64> = (0..16).map(|_| rng.normal()).collect();
    let t1 = weight_to_transition(&w1, 4, 4);
    let w2: Vec<f64> = (0..16).map(|_| rng.normal()).collect();
    let t2 = weight_to_transition(&w2, 4, 4);

    let dispatch_dists =
        dispatcher.belief_propagation(&input_dist, &[t1.as_slice(), t2.as_slice()], &[4, 4]);

    h.check_bool(
        "Sub-04: dispatch BP produces 3 distributions (input + 2 layers)",
        dispatch_dists.len() == 3,
    );

    for (i, dist) in dispatch_dists.iter().enumerate() {
        let sum: f64 = dist.iter().sum();
        h.check_abs(
            &format!("Sub-04: dispatch BP layer {i} sums to 1"),
            sum,
            1.0,
            tolerances::PGM_NORMALIZATION_SUM,
        );
    }

    let direct_dists = neural_spring::neural_pgm::belief_propagation_chain(
        &input_dist,
        &[t1.as_slice(), t2.as_slice()],
        &[4, 4],
    );
    let final_max_diff = dispatch_dists
        .last()
        .unwrap_or(&vec![])
        .iter()
        .zip(direct_dists.last().unwrap_or(&vec![]).iter())
        .map(|(d, r)| (d - r).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "Sub-04: dispatch BP final layer matches direct (tol 0.05)",
        final_max_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Sub-05: Agent interaction graph via Dispatcher
    // ═══════════════════════════════════════════════════════════════════

    let n_agents = 8;
    let dim = 2;
    let positions: Vec<f64> = (0..n_agents * dim).map(|_| rng.uniform() * 5.0).collect();
    let comm_range = 3.0;

    let dispatch_adj = dispatcher.agent_interaction_graph(&positions, n_agents, dim, comm_range);

    h.check_bool(
        "Sub-05: dispatch adjacency matrix correct size",
        dispatch_adj.len() == n_agents * n_agents,
    );

    let mut symmetric = true;
    for i in 0..n_agents {
        for j in 0..n_agents {
            if (dispatch_adj[i * n_agents + j] - dispatch_adj[j * n_agents + i]).abs()
                > tolerances::ZERO_DETECTION
            {
                symmetric = false;
            }
        }
    }
    h.check_bool("Sub-05: dispatch adjacency symmetric", symmetric);

    let mut diagonal_zero = true;
    for i in 0..n_agents {
        if dispatch_adj[i * n_agents + i].abs() > tolerances::ZERO_DETECTION {
            diagonal_zero = false;
        }
    }
    h.check_bool("Sub-05: dispatch adjacency diagonal zero", diagonal_zero);

    let direct_agents: Vec<neural_spring::agent_coordination::Agent> = (0..n_agents)
        .map(|i| neural_spring::agent_coordination::Agent {
            position: positions[i * dim..(i + 1) * dim].to_vec(),
            capability: 1.0,
            signal_level: 0.0,
            cooperating: false,
        })
        .collect();
    let direct_adj =
        neural_spring::agent_coordination::interaction_graph(&direct_agents, comm_range);

    let adj_max_diff = dispatch_adj
        .iter()
        .zip(direct_adj.iter())
        .map(|(d, r)| (d - r).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "Sub-05: dispatch adjacency matches direct library (tol 0.01)",
        adj_max_diff < tolerances::GPU_L2_DISPATCH_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Determinism: dispatch is reproducible
    // ═══════════════════════════════════════════════════════════════════

    let dispatch_result2 = dispatcher.weight_spectral_analysis(&ws_w, ws_rows, ws_cols);
    h.check_bool(
        "Determinism: repeated dispatch eigenvalue count",
        dispatch_result.eigenvalues.len() == dispatch_result2.eigenvalues.len(),
    );

    h.finish();
}
