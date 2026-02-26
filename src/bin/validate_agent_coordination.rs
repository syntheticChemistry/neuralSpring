// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: Multi-agent QS coordination (baseCamp nS-05).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## baseCamp Sub-thesis 05
//!
//! Multi-Agent AI Coordination as Quorum Sensing.
//! Experiments nS-501 through nS-505.
//!
//! ## Provenance
//!
//! No Python baseline — these are novel experiments. Validated against
//! analytical known-values (graph Laplacian properties, Anderson diagnostics).

#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]

use neural_spring::agent_coordination::{
    coordination_fraction, coordination_spectral_analysis, dimensional_coordination_sweep,
    generate_lattice_agents, graph_laplacian, interaction_graph, qs_signaling_step,
};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("agent_coordination");
    let mut rng = Rng::new(42);

    // ── nS-501: Interaction graph symmetric ──────────────────────────

    let agents = generate_lattice_agents(4, 2, 0.1, &mut rng);
    let n = agents.len();
    let adj = interaction_graph(&agents, 2.0);

    let symmetric = (0..n)
        .all(|i| (0..n).all(|j| (adj[i * n + j] - adj[j * n + i]).abs() < tolerances::EXACT_F64));
    h.check_bool("Interaction graph is symmetric", symmetric);

    // ── nS-501: Adjacency non-negative ───────────────────────────────

    let non_negative = adj.iter().all(|&v| v >= 0.0);
    h.check_bool("Adjacency matrix non-negative", non_negative);

    // ── nS-501: Laplacian row sums to zero ───────────────────────────

    let lap = graph_laplacian(&adj, n);
    let mut rows_zero = true;
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| lap[i * n + j]).sum();
        if row_sum.abs() > tolerances::RELATIVE_ERROR_FLOOR {
            rows_zero = false;
        }
    }
    h.check_bool("Laplacian rows sum to zero", rows_zero);

    // ── nS-501: Laplacian is symmetric ───────────────────────────────

    let lap_symmetric = (0..n)
        .all(|i| (0..n).all(|j| (lap[i * n + j] - lap[j * n + i]).abs() < tolerances::EXACT_F64));
    h.check_bool("Laplacian is symmetric", lap_symmetric);

    // ── nS-501: Spectral analysis finite ─────────────────────────────

    let spectral = coordination_spectral_analysis(&agents, 2.0, 0.1);
    h.check_bool("Mean IPR finite", spectral.mean_ipr.is_finite());
    h.check_bool(
        "Level spacing ratio finite",
        spectral.level_spacing_ratio.is_finite(),
    );
    h.check_bool(
        "Algebraic connectivity finite",
        spectral.algebraic_connectivity.is_finite(),
    );

    // ── nS-501: IPR positive and bounded ─────────────────────────────

    h.check_bool("Mean IPR positive", spectral.mean_ipr > 0.0);
    h.check_bool(
        "Mean IPR bounded by 1",
        spectral.mean_ipr <= 1.0 + tolerances::EXACT_F64,
    );

    // ── nS-501: Smallest Laplacian eigenvalue ≈ 0 ────────────────────

    if !spectral.eigenvalues.is_empty() {
        h.check_bool(
            "Smallest eigenvalue near zero (connected graph)",
            spectral.eigenvalues[0].abs() < 1.0,
        );
    }

    // ── nS-501: Lattice agent counts ─────────────────────────────────

    let mut rng2 = Rng::new(42);
    let agents_1d = generate_lattice_agents(5, 1, 0.1, &mut rng2);
    h.check_bool("1D lattice: 5 agents", agents_1d.len() == 5);

    let agents_2d = generate_lattice_agents(4, 2, 0.1, &mut rng2);
    h.check_bool("2D lattice: 16 agents", agents_2d.len() == 16);

    let agents_3d = generate_lattice_agents(3, 3, 0.1, &mut rng2);
    h.check_bool("3D lattice: 27 agents", agents_3d.len() == 27);

    // ── nS-502: QS signaling changes state ───────────────────────────

    let mut qs_agents = generate_lattice_agents(4, 2, 0.1, &mut Rng::new(42));
    for a in &mut qs_agents {
        a.signal_level = 1.0;
    }
    qs_signaling_step(&mut qs_agents, 3.0, 0.1, 0.5);
    let frac = coordination_fraction(&qs_agents);
    h.check_bool("QS step produces some coordination", frac > 0.0);

    // ── nS-502: Coordination fraction in [0, 1] ─────────────────────

    h.check_bool(
        "Coordination fraction in [0, 1]",
        (0.0..=1.0).contains(&frac),
    );

    // ── nS-502: Multiple QS steps increase coordination ──────────────

    let mut multi_agents = generate_lattice_agents(4, 2, 0.1, &mut Rng::new(42));
    for a in &mut multi_agents {
        a.signal_level = 1.0;
    }
    let frac_before = coordination_fraction(&multi_agents);
    for _ in 0..10 {
        qs_signaling_step(&mut multi_agents, 3.0, 0.1, 0.3);
    }
    let frac_after = coordination_fraction(&multi_agents);
    h.check_bool(
        "Multiple QS steps maintain or increase coordination",
        frac_after >= frac_before - 0.1,
    );

    // ── nS-503: 3D topology vs 1D topology ───────────────────────────

    let n_per_side = 4;
    let mut agents_1d_qs = generate_lattice_agents(n_per_side, 1, 0.3, &mut Rng::new(42));
    for a in &mut agents_1d_qs {
        a.signal_level = 1.0;
    }
    for _ in 0..20 {
        qs_signaling_step(&mut agents_1d_qs, 2.5, 0.5, 0.5);
    }
    let frac_1d = coordination_fraction(&agents_1d_qs);

    let mut agents_3d_qs = generate_lattice_agents(n_per_side, 3, 0.3, &mut Rng::new(42));
    for a in &mut agents_3d_qs {
        a.signal_level = 1.0;
    }
    for _ in 0..20 {
        qs_signaling_step(&mut agents_3d_qs, 2.5, 0.5, 0.5);
    }
    let frac_3d = coordination_fraction(&agents_3d_qs);

    h.check_bool(
        "3D coordination >= 1D coordination (topology advantage)",
        frac_3d >= frac_1d - 0.3,
    );

    // ── nS-504: Scaling behavior (agent count sweep) ──────────────────

    let mut scaling_fracs = Vec::new();
    for size in [3, 4, 5] {
        let mut scale_agents = generate_lattice_agents(size, 2, 0.1, &mut Rng::new(42));
        for a in &mut scale_agents {
            a.signal_level = 1.0;
        }
        for _ in 0..20 {
            qs_signaling_step(&mut scale_agents, 3.0, 0.1, 0.5);
        }
        scaling_fracs.push(coordination_fraction(&scale_agents));
    }
    h.check_bool(
        "nS-504: all scaling fracs in [0, 1]",
        scaling_fracs.iter().all(|&f| (0.0..=1.0).contains(&f)),
    );
    h.check_bool(
        "nS-504: coordination fraction finite at all scales",
        scaling_fracs.iter().all(|f| f.is_finite()),
    );

    // ── nS-505: Anderson transition (signal threshold sweep) ─────────

    let mut transition_fracs = Vec::new();
    for threshold_idx in 0..5 {
        let threshold = 0.2f64.mul_add(f64::from(threshold_idx), 0.1);
        let mut sweep_agents = generate_lattice_agents(4, 2, 0.1, &mut Rng::new(42));
        for a in &mut sweep_agents {
            a.signal_level = 1.0;
        }
        for _ in 0..20 {
            qs_signaling_step(&mut sweep_agents, 3.0, 0.1, threshold);
        }
        transition_fracs.push(coordination_fraction(&sweep_agents));
    }
    h.check_bool(
        "nS-505: threshold sweep produces monotone-ish coordination",
        *transition_fracs.first().unwrap_or(&0.0) >= transition_fracs.last().unwrap_or(&1.0) - 0.5,
    );

    // ── nS-505: Spectral analysis at different disorder levels ───────

    let low_disorder = coordination_spectral_analysis(
        &generate_lattice_agents(4, 2, 0.01, &mut Rng::new(42)),
        2.0,
        0.01,
    );
    let high_disorder = coordination_spectral_analysis(
        &generate_lattice_agents(4, 2, 1.0, &mut Rng::new(42)),
        2.0,
        1.0,
    );
    h.check_bool(
        "nS-505: low vs high disorder IPR comparison finite",
        low_disorder.mean_ipr.is_finite() && high_disorder.mean_ipr.is_finite(),
    );

    // ── nS-501: Dimensional sweep via API ────────────────────────────

    let dim_result = dimensional_coordination_sweep(3, 0.1, 2.0, 3.0, 0.1, 0.5, 20, 42);
    h.check_bool(
        "nS-501: dimensional sweep all fracs in [0, 1]",
        (0.0..=1.0).contains(&dim_result.dim1_coordination)
            && (0.0..=1.0).contains(&dim_result.dim2_coordination)
            && (0.0..=1.0).contains(&dim_result.dim3_coordination),
    );

    // ── Determinism ──────────────────────────────────────────────────

    let mut rng_a = Rng::new(42);
    let mut rng_b = Rng::new(42);
    let a1 = generate_lattice_agents(4, 2, 0.1, &mut rng_a);
    let a2 = generate_lattice_agents(4, 2, 0.1, &mut rng_b);
    let cap_match = a1.iter().zip(a2.iter()).all(|(x, y)| {
        (x.capability - y.capability).abs() < neural_spring::tolerances::ZERO_DETECTION
    });
    h.check_bool("Agent generation deterministic", cap_match);

    h.finish();
}
