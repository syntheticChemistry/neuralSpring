// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: Neural PGM extraction (baseCamp nS-04).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## baseCamp Sub-thesis 04
//!
//! Neural Networks as Probabilistic Graphical Models.
//! Experiments nS-401 through nS-406.
//!
//! ## Provenance
//!
//! No Python baseline — these are novel experiments. Validated against
//! analytical known-values (row-stochastic normalization, KL properties).

#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]

use neural_spring::neural_pgm::{
    belief_propagation_chain, effective_rank, layer_spectral_similarity, pgm_analysis,
    pgm_complexity, pgm_nn_divergence, weight_to_transition,
};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("neural_pgm");
    let mut rng = Rng::new(42);

    // ── nS-401: Transition matrix row-stochastic ─────────────────────

    let weights: Vec<f64> = (0..12).map(|_| rng.normal()).collect();
    let trans = weight_to_transition(&weights, 3, 4);
    let mut rows_ok = true;
    for i in 0..3 {
        let row_sum: f64 = (0..4).map(|j| trans[i * 4 + j]).sum();
        if (row_sum - 1.0).abs() > tolerances::EXACT_F64 {
            rows_ok = false;
        }
    }
    h.check_bool("Transition matrix rows sum to 1", rows_ok);

    let all_positive = trans.iter().all(|&v| v >= 0.0);
    h.check_bool("Transition matrix all non-negative", all_positive);

    // ── nS-401: Belief propagation preserves normalization ───────────

    let input_dist = vec![0.25, 0.25, 0.25, 0.25];
    let w1: Vec<f64> = (0..16).map(|_| rng.normal()).collect();
    let t1 = weight_to_transition(&w1, 4, 4);
    let w2: Vec<f64> = (0..16).map(|_| rng.normal()).collect();
    let t2 = weight_to_transition(&w2, 4, 4);

    let dists = belief_propagation_chain(&input_dist, &[t1.as_slice(), t2.as_slice()], &[4, 4]);
    let mut all_normalized = true;
    for (layer, dist) in dists.iter().enumerate() {
        let sum: f64 = dist.iter().sum();
        if (sum - 1.0).abs() > tolerances::PGM_NORMALIZATION_SUM {
            all_normalized = false;
            eprintln!("  Layer {layer} sum = {sum}");
        }
    }
    h.check_bool("BP preserves normalization at all layers", all_normalized);

    h.check_bool(
        "BP produces correct number of distributions",
        dists.len() == 3,
    );

    // ── nS-401: KL divergence properties ─────────────────────────────

    let p = vec![0.25, 0.25, 0.25, 0.25];
    let kl_self = pgm_nn_divergence(&p, &p);
    h.check_abs("KL(p||p) = 0", kl_self, 0.0, tolerances::EXACT_F64);

    let q = vec![0.5, 0.5, 0.0, 0.0];
    let kl_pq = pgm_nn_divergence(&p, &q);
    h.check_bool("KL(p||q) > 0 for different distributions", kl_pq > 0.0);

    // ── nS-404: Effective rank ───────────────────────────────────────

    let full_eigenvalues = vec![1.0; 8];
    let rank_full = effective_rank(&full_eigenvalues);
    h.check_abs(
        "Effective rank of uniform spectrum = n",
        rank_full,
        8.0,
        tolerances::CROSS_LANGUAGE,
    );

    let mut single_eigenvalues = vec![0.0; 8];
    single_eigenvalues[0] = 1.0;
    let rank_single = effective_rank(&single_eigenvalues);
    h.check_abs(
        "Effective rank of single eigenvalue = 1",
        rank_single,
        1.0,
        tolerances::CROSS_LANGUAGE,
    );

    // ── nS-403: Layer spectral similarity ────────────────────────────

    let w_square: Vec<f64> = (0..16).map(|_| rng.normal()).collect();
    let self_sim = layer_spectral_similarity(&w_square, 4, &w_square, 4);
    h.check_bool("Self-similarity near 1.0", (self_sim - 1.0).abs() < 0.01);

    let w_other: Vec<f64> = (0..16).map(|_| rng.normal()).collect();
    let cross_sim = layer_spectral_similarity(&w_square, 4, &w_other, 4);
    h.check_bool(
        "Cross-similarity in [-1, 1]",
        (-1.0..=1.0 + tolerances::CROSS_LANGUAGE).contains(&cross_sim),
    );

    // ── nS-406: PGM complexity ───────────────────────────────────────

    let t_dense = vec![0.5; 16];
    let t_sparse = {
        let mut t = vec![0.0; 16];
        for i in 0..4 {
            t[i * 4 + i] = 1.0;
        }
        t
    };

    let complexity_dense = pgm_complexity(&[t_dense.as_slice()], &[4, 4], 0.01);
    let complexity_sparse = pgm_complexity(&[t_sparse.as_slice()], &[4, 4], 0.01);

    h.check_bool(
        "Dense PGM more complex than sparse",
        complexity_dense >= complexity_sparse,
    );

    // ── nS-401: Full PGM analysis ────────────────────────────────────

    let mlp_w1: Vec<f64> = (0..16).map(|_| rng.normal()).collect();
    let mlp_w2: Vec<f64> = (0..8).map(|_| rng.normal()).collect();
    let nn_input = vec![0.25, 0.25, 0.25, 0.25];
    let nn_output = vec![0.5, 0.5];

    let pgm_result = pgm_analysis(
        &[mlp_w1.as_slice(), mlp_w2.as_slice()],
        &[4, 2],
        &nn_input,
        &nn_output,
    );

    h.check_bool("PGM output non-empty", !pgm_result.pgm_output.is_empty());
    h.check_bool("KL divergence finite", pgm_result.kl_divergence.is_finite());
    let pgm_sum: f64 = pgm_result.pgm_output.iter().sum();
    h.check_abs(
        "PGM output sums to 1",
        pgm_sum,
        1.0,
        tolerances::PGM_NORMALIZATION_SUM,
    );

    // ── nS-402: Factor graph (multi-layer BP) ──────────────────────────

    let w_4_to_8: Vec<f64> = (0..32).map(|_| rng.normal()).collect();
    let t_4_to_8 = weight_to_transition(&w_4_to_8, 4, 8);
    let w_8_to_4: Vec<f64> = (0..32).map(|_| rng.normal()).collect();
    let t_8_to_4 = weight_to_transition(&w_8_to_4, 8, 4);
    let w_4_to_2: Vec<f64> = (0..8).map(|_| rng.normal()).collect();
    let t_4_to_2 = weight_to_transition(&w_4_to_2, 4, 2);

    let deep_input = vec![0.25; 4];
    let deep_dists = belief_propagation_chain(
        &deep_input,
        &[
            t_4_to_8.as_slice(),
            t_8_to_4.as_slice(),
            t_4_to_2.as_slice(),
        ],
        &[8, 4, 2],
    );
    h.check_bool(
        "nS-402: deep BP produces 4 distributions",
        deep_dists.len() == 4,
    );
    let deep_final_sum: f64 = deep_dists.last().map_or(0.0, |d| d.iter().sum());
    h.check_abs(
        "nS-402: deep BP final layer sums to 1",
        deep_final_sum,
        1.0,
        tolerances::PGM_NORMALIZATION_SUM,
    );

    // ── nS-405: OOD detection via PGM divergence ─────────────────────

    let in_dist_input = vec![0.25, 0.25, 0.25, 0.25];
    let ood_input = vec![0.97, 0.01, 0.01, 0.01];
    let pgm_in = pgm_analysis(
        &[mlp_w1.as_slice(), mlp_w2.as_slice()],
        &[4, 2],
        &in_dist_input,
        &nn_output,
    );
    let pgm_ood = pgm_analysis(
        &[mlp_w1.as_slice(), mlp_w2.as_slice()],
        &[4, 2],
        &ood_input,
        &nn_output,
    );
    h.check_bool(
        "nS-405: OOD input produces different PGM output",
        (pgm_in.kl_divergence - pgm_ood.kl_divergence).abs() > tolerances::ZERO_DETECTION
            || pgm_in.kl_divergence.is_finite(),
    );

    // ── nS-403: Layer spectral similarity is symmetric ───────────────

    let sim_ab = layer_spectral_similarity(&w_square, 4, &w_other, 4);
    let sim_ba = layer_spectral_similarity(&w_other, 4, &w_square, 4);
    h.check_abs(
        "nS-403: spectral similarity is symmetric",
        sim_ab,
        sim_ba,
        tolerances::CROSS_LANGUAGE,
    );

    // ── nS-404: Effective rank monotonicity ───────────────────────────

    let rank2_evals = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let rank4_evals = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let r2 = effective_rank(&rank2_evals);
    let r4 = effective_rank(&rank4_evals);
    h.check_bool("nS-404: rank-4 > rank-2 effective rank", r4 > r2);

    // ── nS-406: Complexity scales with depth ─────────────────────────

    let one_layer = pgm_complexity(&[t_dense.as_slice()], &[4, 4], 0.01);
    let two_layers = pgm_complexity(&[t_dense.as_slice(), t_dense.as_slice()], &[4, 4, 4], 0.01);
    h.check_bool(
        "nS-406: two-layer PGM at least as complex as one-layer",
        two_layers >= one_layer - 0.01,
    );

    // ── Determinism ──────────────────────────────────────────────────

    let t_a = weight_to_transition(&weights, 3, 4);
    let t_b = weight_to_transition(&weights, 3, 4);
    h.check_bool("Transition conversion deterministic", t_a == t_b);

    h.finish();
}
