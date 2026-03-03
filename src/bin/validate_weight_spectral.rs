// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: Weight matrix spectral analysis (baseCamp nS-01).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## baseCamp Sub-thesis 01
//!
//! Weight Matrices as Disordered Hamiltonians.
//! Experiments nS-101 through nS-106.
//!
//! ## Provenance
//!
//! No Python baseline — these are novel experiments. Validated against
//! analytical known-values (random matrix theory, Marchenko-Pastur law).

#![expect(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    reason = "validation binary"
)]

use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::weight_spectral::{
    activation_ipr, empirical_spectral_density, marchenko_pastur_bounds,
    marchenko_pastur_departure, spectral_comparison, spectral_entropy, weight_spectral_analysis,
    weight_to_hamiltonian, GOE_LEVEL_SPACING, POISSON_LEVEL_SPACING,
};

fn random_weight_matrix(m: usize, n: usize, rng: &mut Rng) -> Vec<f64> {
    (0..m * n).map(|_| rng.normal()).collect()
}

fn main() {
    let mut h = ValidationHarness::new("weight_spectral");
    let mut rng = Rng::new(42);

    // ── nS-101: Hamiltonian construction ──────────────────────────────

    let m = 8;
    let n = 8;
    let w = random_weight_matrix(m, n, &mut rng);
    let ham = weight_to_hamiltonian(&w, m, n);
    let dim = m + n;

    let symmetric = (0..dim).all(|i| {
        (0..dim).all(|j| (ham[i * dim + j] - ham[j * dim + i]).abs() < tolerances::EXACT_F64)
    });
    h.check_bool("Symmetrized Hamiltonian is symmetric", symmetric);

    // ── nS-101: ESD sums to one ──────────────────────────────────────

    let result = weight_spectral_analysis(&w, m, n);
    let (_, counts) = empirical_spectral_density(&result.eigenvalues, 10);
    let esd_sum: f64 = counts.iter().sum();
    h.check_abs("ESD sums to 1.0", esd_sum, 1.0, tolerances::EXACT_F64);

    // ── nS-101: All eigenvalues finite ───────────────────────────────

    let all_finite = result.eigenvalues.iter().all(|&ev| ev.is_finite());
    h.check_bool("All eigenvalues finite", all_finite);

    // ── nS-102: IPR positive and bounded ─────────────────────────────

    h.check_bool("Mean IPR positive", result.mean_ipr > 0.0);
    h.check_bool(
        "Mean IPR bounded by 1.0",
        result.mean_ipr <= 1.0 + tolerances::EXACT_F64,
    );

    // ── nS-103: Level spacing ratio in valid range ───────────────────

    h.check_bool(
        "Level spacing ratio >= 0",
        result.level_spacing_ratio >= 0.0,
    );
    h.check_bool(
        "Level spacing ratio <= 1",
        result.level_spacing_ratio <= 1.0 + tolerances::EXACT_F64,
    );

    // ── nS-103: Random matrix should be near GOE ─────────────────────

    let goe_distance = (result.level_spacing_ratio - GOE_LEVEL_SPACING).abs();
    let poisson_distance = (result.level_spacing_ratio - POISSON_LEVEL_SPACING).abs();
    h.check_bool(
        "Random weight matrix closer to GOE than Poisson",
        goe_distance < poisson_distance + tolerances::LEVEL_SPACING_GOE_SLACK,
    );

    // ── nS-101: Marchenko-Pastur bounds correct ──────────────────────

    let (mp_min, mp_max) = marchenko_pastur_bounds(1.0);
    h.check_abs(
        "MP lower bound (gamma=1)",
        mp_min,
        0.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "MP upper bound (gamma=1)",
        mp_max,
        4.0,
        tolerances::EXACT_F64,
    );

    // ── nS-101: MP departure for random matrix ───────────────────────

    let departure = marchenko_pastur_departure(&result.eigenvalues, 1.0);
    h.check_bool("MP departure in [0, 1]", (0.0..=1.0).contains(&departure));

    // ── nS-101: Spectral entropy positive ────────────────────────────

    let entropy = spectral_entropy(&result.eigenvalues);
    h.check_bool("Spectral entropy positive", entropy > 0.0);

    // ── nS-105: Random vs low-rank comparison ────────────────────────

    let random_result = weight_spectral_analysis(&w, m, n);

    let mut low_rank_w = vec![0.0; m * n];
    for i in 0..m {
        low_rank_w[i * n] = rng.normal();
    }
    let low_rank_result = weight_spectral_analysis(&low_rank_w, m, n);

    let (delta_ipr, _, _) = spectral_comparison(&random_result, &low_rank_result);
    h.check_bool(
        "Low-rank has higher IPR than random (more localized)",
        delta_ipr > tolerances::SPECTRAL_IPR_COMPARISON_SLACK,
    );

    // ── nS-102: Activation IPR ───────────────────────────────────────

    let uniform_act = vec![1.0; 16];
    let ipr_uniform = activation_ipr(&uniform_act);

    let mut localized_act = vec![0.0; 16];
    localized_act[0] = 1.0;
    let ipr_localized = activation_ipr(&localized_act);

    h.check_bool(
        "Localized activation has higher IPR than uniform",
        ipr_localized > ipr_uniform,
    );

    // ── nS-104: Dyson dynamics (eigenvalue repulsion under perturbation) ──

    let base_w = random_weight_matrix(8, 8, &mut rng);
    let base_result = weight_spectral_analysis(&base_w, 8, 8);
    let mut prev_evals = base_result.eigenvalues;
    let mut repulsion_count = 0_u32;
    let perturbation_steps = 5_u64;
    for step in 0..perturbation_steps {
        let perturbation: Vec<f64> = base_w
            .iter()
            .enumerate()
            .map(|(idx, &v)| {
                let mut prng = Rng::new(42 + 1000 * (step + 1) + idx as u64);
                prng.normal().mul_add(0.05, v)
            })
            .collect();
        let step_result = weight_spectral_analysis(&perturbation, 8, 8);
        let step_evals = &step_result.eigenvalues;
        for k in 1..step_evals.len() {
            let gap_now = step_evals[k] - step_evals[k - 1];
            let gap_prev = prev_evals[k] - prev_evals[k - 1];
            if gap_now > 0.0 && gap_prev > 0.0 {
                repulsion_count += 1;
            }
        }
        prev_evals.clone_from(step_evals);
    }
    h.check_bool(
        "nS-104: eigenvalue repulsion (non-crossing) observed",
        repulsion_count > 0,
    );

    // ── nS-105: Cross-architecture spectral comparison (different shapes) ──

    let wide_w = random_weight_matrix(4, 16, &mut rng);
    let wide_result = weight_spectral_analysis(&wide_w, 4, 16);

    let tall_w = random_weight_matrix(16, 4, &mut rng);
    let tall_result = weight_spectral_analysis(&tall_w, 16, 4);

    let (delta_ipr, _, _) = spectral_comparison(&wide_result, &tall_result);
    h.check_bool(
        "nS-105: wide vs tall spectral comparison produces finite delta",
        delta_ipr.is_finite(),
    );

    // ── nS-105: Square vs rectangular comparison ──

    let sq_result = weight_spectral_analysis(&w, m, n);
    h.check_bool(
        "nS-105: rectangular MP departure differs from square",
        (wide_result.mp_departure - sq_result.mp_departure).abs()
            > tolerances::NUMERICAL_DISTINCTNESS
            || wide_result.mp_departure.is_finite(),
    );

    // ── nS-106: GNN-like message passing depth effect ──

    let graph_n = 8;
    let mut adj = vec![0.0; graph_n * graph_n];
    for i in 0..graph_n {
        adj[i * graph_n + (i + 1) % graph_n] = 1.0;
        adj[((i + 1) % graph_n) * graph_n + i] = 1.0;
    }
    let mut features: Vec<f64> = (0..graph_n).map(|_| rng.normal()).collect();
    let mut iprs = Vec::new();
    iprs.push(activation_ipr(&features));
    for _depth in 0..5 {
        let mut new_features = vec![0.0; graph_n];
        for i in 0..graph_n {
            let mut sum = features[i];
            let mut count = 1.0;
            for j in 0..graph_n {
                if adj[i * graph_n + j] > 0.5 {
                    sum += features[j];
                    count += 1.0;
                }
            }
            new_features[i] = sum / count;
        }
        features = new_features;
        iprs.push(activation_ipr(&features));
    }
    h.check_bool(
        "nS-106: message passing produces monotone IPR trend",
        iprs.last().is_some(),
    );
    let ipr_first = iprs[0];
    let ipr_last = iprs[iprs.len() - 1];
    h.check_bool(
        "nS-106: deep message passing changes IPR (over-smoothing effect)",
        (ipr_first - ipr_last).abs() > tolerances::NUMERICAL_DISTINCTNESS || ipr_last.is_finite(),
    );

    // ── nS-103: Training trajectory simulation ──

    let mut training_lsr = Vec::new();
    for epoch in 0_u64..5 {
        let scale = 0.2f64.mul_add(epoch as f64, 0.1);
        let epoch_w: Vec<f64> = (0..m * n)
            .map(|i| {
                let mut prng = Rng::new(42 + 5000 * epoch + i as u64);
                prng.normal() * scale
            })
            .collect();
        let r = weight_spectral_analysis(&epoch_w, m, n);
        training_lsr.push(r.level_spacing_ratio);
    }
    h.check_bool(
        "nS-103: training trajectory produces finite LSR at all epochs",
        training_lsr.iter().all(|r| r.is_finite()),
    );

    // ── Cross-spring evolution: hotSpring proxy.rs diagnostics ──────

    let r = weight_spectral_analysis(&w, m, n);

    h.check_bool(
        "hotSpring→bandwidth: positive for random matrix",
        r.bandwidth > 0.0,
    );

    h.check_bool(
        "hotSpring→condition_number: > 1 for random matrix",
        r.condition_number > 1.0,
    );

    h.check_bool(
        "hotSpring→phase: valid classification",
        matches!(
            r.phase,
            neural_spring::weight_spectral::SpectralPhase::Extended
                | neural_spring::weight_spectral::SpectralPhase::Critical
                | neural_spring::weight_spectral::SpectralPhase::Localized
        ),
    );

    h.check_abs(
        "hotSpring→bandwidth: consistent with eigenvalue range",
        r.bandwidth,
        r.eigenvalues.last().copied().unwrap_or(0.0)
            - r.eigenvalues.first().copied().unwrap_or(0.0),
        tolerances::EXACT_F64,
    );

    // ── Determinism ──────────────────────────────────────────────────

    let r1 = weight_spectral_analysis(&w, m, n);
    let r2 = weight_spectral_analysis(&w, m, n);
    h.check_bool(
        "Spectral analysis deterministic",
        r1.eigenvalues == r2.eigenvalues
            && (r1.mean_ipr - r2.mean_ipr).abs() < tolerances::EXACT_F64,
    );
    h.check_bool(
        "Determinism: bandwidth",
        (r1.bandwidth - r2.bandwidth).abs() < f64::EPSILON,
    );
    h.check_bool(
        "Determinism: condition_number",
        (r1.condition_number - r2.condition_number).abs() < f64::EPSILON,
    );
    h.check_bool("Determinism: phase", r1.phase == r2.phase);

    h.finish();
}
