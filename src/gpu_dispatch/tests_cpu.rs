// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU-path tests for [`Dispatcher`] operations.
//!
//! These tests exercise CPU fallback paths — no GPU adapter required.
//! They validate correctness of the local CPU reference implementations
//! and metadata queries when running in CPU-only mode.

#![allow(clippy::expect_used)]

use super::*;
use crate::tolerances;

fn cpu() -> Dispatcher {
    Dispatcher::cpu_only()
}

// ── Metadata ────────────────────────────────────────────────

#[test]
fn cpu_only_no_gpu() {
    let d = cpu();
    assert!(!d.has_gpu());
    assert_eq!(d.backend(), Backend::Cpu);
    assert!(d.capabilities().is_none());
    assert_eq!(d.adapter_name(), "(none)");
    assert!(d.wgpu_device().is_none());
    assert!(d.gpu().is_none());
}

#[test]
fn backend_display() {
    assert_eq!(format!("{}", Backend::Gpu), "GPU");
    assert_eq!(format!("{}", Backend::Cpu), "CPU");
}

// ── Linear algebra ──────────────────────────────────────────

#[test]
fn cpu_mat_mul_identity() {
    let d = cpu();
    #[rustfmt::skip]
    let eye = vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    let result = d.mat_mul(&eye, &eye, 3);
    for (i, &v) in result.iter().enumerate() {
        let expected = if i / 3 == i % 3 { 1.0 } else { 0.0 };
        assert!(
            (v - expected).abs() < tolerances::ZERO_DETECTION,
            "mat_mul identity [{i}]"
        );
    }
}

#[test]
fn cpu_frobenius_norm() {
    let d = cpu();
    let a = vec![3.0, 4.0];
    assert!((d.frobenius_norm(&a) - 5.0).abs() < tolerances::EXACT_F64);
}

#[test]
fn cpu_transpose() {
    let d = cpu();
    #[rustfmt::skip]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let t = d.transpose(&a, 2);
    assert!((t[0] - 1.0).abs() < tolerances::ZERO_DETECTION);
    assert!((t[1] - 3.0).abs() < tolerances::ZERO_DETECTION);
    assert!((t[2] - 2.0).abs() < tolerances::ZERO_DETECTION);
    assert!((t[3] - 4.0).abs() < tolerances::ZERO_DETECTION);
}

#[test]
fn cpu_distance_to_normal() {
    let d = cpu();
    #[rustfmt::skip]
    let sym = vec![
        2.0, 1.0,
        1.0, 2.0,
    ];
    let dist = d.distance_to_normal(&sym, 2);
    assert!(
        dist < tolerances::EXACT_F64,
        "symmetric matrix should commute with transpose"
    );
}

#[test]
fn cpu_commutator_symmetric_zero() {
    let d = cpu();
    let a = vec![1.0, 0.0, 0.0, 1.0];
    let comm = d.commutator(&a, &a, 2);
    for &v in &comm {
        assert!(
            v.abs() < tolerances::ZERO_DETECTION,
            "A commutes with itself"
        );
    }
}

// ── Activations / distributions ─────────────────────────────

#[test]
fn cpu_softmax_sums_to_one() {
    let d = cpu();
    let result = d.softmax(&[1.0, 2.0, 3.0]);
    let total: f64 = result.iter().sum();
    assert!((total - 1.0).abs() < tolerances::EXACT_F64);
    assert!(result[2] > result[1] && result[1] > result[0]);
}

#[test]
fn cpu_boltzmann_sums_to_one() {
    let d = cpu();
    let result = d.boltzmann(&[1.0, 2.0, 3.0], 1.0);
    let total: f64 = result.iter().sum();
    assert!((total - 1.0).abs() < tolerances::EXACT_F64);
}

// ── Reductions / statistics ─────────────────────────────────

#[test]
fn cpu_l2_distance() {
    let d = cpu();
    let dist = d.l2_distance(&[0.0, 0.0], &[3.0, 4.0]);
    assert!((dist - 5.0).abs() < tolerances::EXACT_F64);
}

#[test]
fn cpu_shannon_entropy() {
    let d = cpu();
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let h = d.shannon_entropy(&p);
    let expected = 4.0_f64.ln();
    assert!((h - expected).abs() < tolerances::CROSS_LANGUAGE);
}

#[test]
fn cpu_mean() {
    let d = cpu();
    assert!((d.mean(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < tolerances::ZERO_DETECTION);
    assert!((d.mean(&[]) - 0.0).abs() < tolerances::ZERO_DETECTION);
}

#[test]
fn cpu_variance() {
    let d = cpu();
    let v = d.variance(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    assert!((v - 4.0).abs() < tolerances::EXACT_F64);
    assert!((d.variance(&[]) - 0.0).abs() < tolerances::ZERO_DETECTION);
}

#[test]
fn cpu_pearson_correlation() {
    let d = cpu();
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let r = d.pearson_correlation(&x, &y);
    assert!(
        (r - 1.0).abs() < tolerances::EXACT_F64,
        "perfect positive correlation"
    );
}

#[test]
fn cpu_pearson_short() {
    let d = cpu();
    assert!((d.pearson_correlation(&[1.0], &[2.0]) - 0.0).abs() < tolerances::ZERO_DETECTION);
}

#[test]
fn cpu_pearson_zero_variance() {
    let d = cpu();
    let r = d.pearson_correlation(&[3.0, 3.0, 3.0], &[1.0, 2.0, 3.0]);
    assert!((r - 0.0).abs() < tolerances::ZERO_DETECTION);
}

#[test]
fn cpu_chi_squared() {
    let d = cpu();
    let obs = vec![10.0, 20.0, 30.0];
    let exp = vec![20.0, 20.0, 20.0];
    let chi2 = d.chi_squared(&obs, &exp);
    assert!((chi2 - 10.0).abs() < tolerances::CROSS_LANGUAGE);
}

#[test]
fn cpu_chi_squared_zero_expected() {
    let d = cpu();
    let chi2 = d.chi_squared(&[5.0], &[0.0]);
    assert!(
        (chi2 - 0.0).abs() < tolerances::ZERO_DETECTION,
        "zero expected → 0 contribution"
    );
}

// ── HMM ─────────────────────────────────────────────────────

#[test]
fn cpu_hmm_backward_step_basic() {
    let d = cpu();
    let beta_next = vec![1.0, 1.0];
    #[rustfmt::skip]
    let trans = vec![0.7, 0.3, 0.4, 0.6];
    let emit = vec![0.5, 0.5];
    let result = d.hmm_backward_step(&beta_next, &trans, &emit, 1.0, 2);
    assert_eq!(result.len(), 2);
    assert!((result[0] - 0.5).abs() < tolerances::EXACT_F64);
    assert!((result[1] - 0.5).abs() < tolerances::EXACT_F64);
}

#[test]
fn cpu_hmm_backward_step_zero_scale() {
    let d = cpu();
    let result = d.hmm_backward_step(&[1.0], &[1.0], &[1.0], 0.0, 1);
    assert!(result[0].is_finite(), "zero scale should use guard");
}

#[test]
fn cpu_hmm_viterbi_step() {
    let d = cpu();
    let delta_prev = vec![0.0_f64.ln(), (-1.0_f64).exp().ln()];
    #[rustfmt::skip]
    let log_trans = vec![
        0.7_f64.ln(), 0.3_f64.ln(),
        0.4_f64.ln(), 0.6_f64.ln(),
    ];
    let log_emit = vec![0.6_f64.ln(), 0.4_f64.ln()];
    let (delta, psi) = d.hmm_viterbi_step(&delta_prev, &log_trans, &log_emit, 2);
    assert_eq!(delta.len(), 2);
    assert_eq!(psi.len(), 2);
}

// ── Population genetics ─────────────────────────────────────

#[test]
fn cpu_allele_frequencies() {
    let d = cpu();
    let pop = vec![2.0, 0.0, 0.0, 2.0];
    let freq = d.allele_frequencies(&pop, 2, 2);
    assert_eq!(freq.len(), 2);
    assert!((freq[0] - 0.5).abs() < tolerances::EXACT_F64);
    assert!((freq[1] - 0.5).abs() < tolerances::EXACT_F64);
}

#[test]
fn cpu_nucleotide_diversity() {
    let d = cpu();
    let pop = vec![0.0, 1.0, 1.0, 0.0];
    let pi = d.nucleotide_diversity(&pop, 2, 2);
    assert!(pi >= 0.0);
}

#[test]
fn cpu_matrix_correlation() {
    let d = cpu();
    #[rustfmt::skip]
    let a = vec![
        0.0, 1.0, 2.0,
        1.0, 0.0, 3.0,
        2.0, 3.0, 0.0,
    ];
    let r = d.matrix_correlation(&a, &a, 3);
    assert!(
        (r - 1.0).abs() < tolerances::CROSS_LANGUAGE,
        "self-correlation = 1.0"
    );
}

#[test]
fn cpu_geographic_distances() {
    let d = cpu();
    let coords = vec![(0.0, 0.0), (3.0, 4.0)];
    let dist = d.geographic_distances(&coords);
    assert_eq!(dist.len(), 4);
    assert!((dist[0] - 0.0).abs() < tolerances::EXACT_F64);
    assert!((dist[1] - 5.0).abs() < tolerances::EXACT_F64);
    assert!((dist[3] - 0.0).abs() < tolerances::EXACT_F64);
}

#[test]
fn cpu_thermal_diversity_correlation() {
    let d = cpu();
    let r = d.thermal_diversity_correlation(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]);
    assert!(
        (r - 1.0).abs() < tolerances::CROSS_LANGUAGE,
        "perfect linear → r≈1"
    );
}

#[test]
fn cpu_thermal_diversity_short() {
    let d = cpu();
    let r = d.thermal_diversity_correlation(&[1.0], &[10.0]);
    assert!((r - 0.0).abs() < tolerances::ZERO_DETECTION, "n<2 → 0");
}

// ── Game theory ─────────────────────────────────────────────

#[test]
fn cpu_replicator_step_preserves_simplex() {
    let d = cpu();
    let freq = [0.6, 0.4];
    let payoff = [[3.0, 0.0], [5.0, 1.0]];
    let next = d.replicator_step(&freq, &payoff, 0.01);
    let sum: f64 = next.iter().sum();
    assert!(
        (sum - 1.0).abs() < tolerances::EXACT_F64,
        "frequencies sum to 1"
    );
    assert!(next[0] >= 0.0 && next[1] >= 0.0, "non-negative");
}

// ── Regulatory ──────────────────────────────────────────────

#[test]
fn cpu_hill_activation_batch() {
    let d = cpu();
    let result = d.hill_activation_batch(&[0.0, 1.0, 10.0], 1.0, 1.0, 2.0);
    assert_eq!(result.len(), 3);
    assert!(
        (result[0] - 0.0).abs() < tolerances::CROSS_LANGUAGE,
        "hill(0)≈0"
    );
    assert!(
        (result[1] - 0.5).abs() < tolerances::NORM_PPF_TAIL,
        "hill(k)≈Vmax/2"
    );
    assert!(result[2] > 0.9, "hill(10k)≈Vmax");
}

// ── Eigensolvers ────────────────────────────────────────────

#[test]
fn cpu_eigh_diagonal() {
    let d = cpu();
    let a = vec![2.0, 0.0, 0.0, 3.0];
    let (vals, _vecs) = d.eigh(&a, 2);
    let mut sorted = vals;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    assert!((sorted[0] - 2.0).abs() < tolerances::CROSS_LANGUAGE);
    assert!((sorted[1] - 3.0).abs() < tolerances::CROSS_LANGUAGE);
}

#[test]
fn cpu_disorder_sweep_no_gpu() {
    let d = cpu();
    assert!(d.disorder_sweep(&[1.0, 0.0, 0.0, 1.0], 2, 1).is_none());
}

// ── Inter-population AF variance ──────────────────────────

#[test]
fn cpu_inter_population_af_variance_basic() {
    let d = cpu();
    let population_a = vec![2.0, 0.0, 0.0, 2.0];
    let population_b = vec![0.0, 2.0, 2.0, 0.0];
    let populations: Vec<&[f64]> = vec![&population_a, &population_b];
    let var = d.inter_population_af_variance(&populations, &[2, 2], 2);
    assert!(var >= 0.0, "AF variance must be non-negative");
}

// ── FST ──────────────────────────────────────────────────────

#[test]
fn cpu_pairwise_fst_divergent() {
    let d = cpu();
    let pop_a = vec![2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0];
    let pop_b = vec![0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0];
    let fst = d.pairwise_fst(&pop_a, 5, &pop_b, 5, 2);
    assert!(fst.is_finite(), "FST must be finite");
}

#[test]
fn cpu_global_fst_two_pops() {
    let d = cpu();
    let pop1 = vec![2.0, 0.0, 2.0, 0.0];
    let pop2 = vec![0.0, 2.0, 0.0, 2.0];
    let fst = d.global_fst(&[pop1, pop2], &[2, 2], 2);
    assert!(fst.is_finite(), "FST must be finite");
}

// ── HMM chains ──────────────────────────────────────────────

#[test]
fn cpu_hmm_forward_chain_basic() {
    let d = cpu();
    let initial = vec![0.6, 0.4];
    #[rustfmt::skip]
    let transition = vec![0.7, 0.3, 0.4, 0.6];
    #[rustfmt::skip]
    let emission = vec![0.5, 0.4, 0.1, 0.1, 0.3, 0.6];
    let obs = vec![0, 1, 2, 0];
    let ll = d.hmm_forward_chain(&initial, &transition, &emission, &obs, 2, 3);
    assert!(ll.is_finite(), "log-likelihood must be finite");
    assert!(ll < 0.0, "log-likelihood should be negative");
}

#[test]
fn cpu_hmm_viterbi_chain_basic() {
    let d = cpu();
    let initial = vec![0.6, 0.4];
    #[rustfmt::skip]
    let transition = vec![0.7, 0.3, 0.4, 0.6];
    #[rustfmt::skip]
    let emission = vec![0.5, 0.4, 0.1, 0.1, 0.3, 0.6];
    let obs = vec![0, 1, 2, 0];
    let (path, log_prob) = d.hmm_viterbi_chain(&initial, &transition, &emission, &obs, 2, 3);
    assert_eq!(path.len(), 4);
    assert!(log_prob.is_finite());
    for &s in &path {
        assert!(s < 2, "state must be < n_states");
    }
}

// ── Pangenome selection ─────────────────────────────────────

#[test]
fn cpu_spectrum_chi_squared() {
    let d = cpu();
    let obs = vec![10.0, 20.0, 30.0];
    let frac = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let chi2 = d.spectrum_chi_squared(&obs, &frac);
    assert!(chi2 >= 0.0);
}

#[test]
fn cpu_selection_coefficient() {
    let d = cpu();
    let obs = vec![10.0, 20.0, 30.0];
    let neutral = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let s = d.selection_coefficient(&obs, &neutral);
    assert!(s.is_finite());
}

// ── baseCamp (gpu_dispatch/basecamp.rs coverage) ─────────────

#[test]
fn basecamp_weight_spectral_analysis() {
    let d = cpu();
    let weights = vec![1.0, 0.0, 0.0, 1.0];
    let result = d.weight_spectral_analysis(&weights, 2, 2);
    assert_eq!(result.eigenvalues.len(), 4);
    assert!(result.mean_ipr.is_finite());
    assert!(result.level_spacing_ratio.is_finite());
    assert!(result.spectral_entropy.is_finite());
    assert!(result.mp_departure.is_finite());
}

#[test]
fn basecamp_numerical_hessian_quadratic() {
    let d = cpu();
    let quadratic = |x: &[f64]| -> f64 { x.iter().map(|&v| v * v).sum() };
    let point = vec![1.0, 2.0];
    let hess = d.numerical_hessian(quadratic, &point, tolerances::HESSIAN_FD_STEP);
    assert_eq!(hess.len(), 4);
    assert!(
        (hess[0] - 2.0).abs() < tolerances::OPTIMIZER_VALUE_AT_MIN,
        "d²/dx² of x² = 2"
    );
    assert!(
        (hess[3] - 2.0).abs() < tolerances::OPTIMIZER_VALUE_AT_MIN,
        "d²/dy² of y² = 2"
    );
    assert!(
        hess[1].abs() < tolerances::OPTIMIZER_VALUE_AT_MIN,
        "cross-term ≈ 0"
    );
}

#[test]
fn basecamp_belief_propagation_preserves_probability() {
    let d = cpu();
    let input = vec![0.25, 0.25, 0.25, 0.25];
    #[rustfmt::skip]
    let transition = vec![
        0.7, 0.3,
        0.6, 0.4,
        0.5, 0.5,
        0.4, 0.6,
    ];
    let dists = d.belief_propagation(&input, &[transition.as_slice()], &[2]);
    assert_eq!(dists.len(), 2);
    let final_sum: f64 = dists.last().expect("non-empty").iter().sum();
    assert!(
        (final_sum - 1.0).abs() < tolerances::CROSS_LANGUAGE,
        "output should be normalized, got sum={final_sum}"
    );
}

#[test]
fn basecamp_agent_interaction_graph() {
    let d = cpu();
    let positions = vec![0.0, 0.0, 1.0, 0.0, 5.0, 5.0];
    let adj = d.agent_interaction_graph(&positions, 3, 2, 2.0);
    assert_eq!(adj.len(), 9);
    assert!(adj[1] > 0.0, "agents 0-1 within range");
    assert!(adj[3] > 0.0, "symmetric: adj[1][0]");
    assert!(
        adj[2].abs() < tolerances::ZERO_DETECTION,
        "agents 0-2 outside range"
    );
}

#[test]
fn basecamp_landscape_analysis_quadratic() {
    let d = cpu();
    let quadratic = |x: &[f64]| -> f64 { x[0].mul_add(x[0], x[1] * x[1]) };
    let result = d.landscape_analysis(&quadratic, &[1.0, 1.0], tolerances::HESSIAN_FD_STEP, 0.1);
    assert!(result.loss.is_finite(), "loss must be finite");
    assert!(
        (result.loss - 2.0).abs() < tolerances::EXACT_F64,
        "f(1,1)=2, got {}",
        result.loss
    );
    assert!(result.flatness.is_finite(), "flatness must be finite");
    assert!(result.sharpness.is_finite(), "sharpness must be finite");
    assert_eq!(
        result.saddle_index, 0,
        "quadratic has no negative curvature"
    );
    assert!(result.spectral_gap.is_finite());
    assert_eq!(result.hessian_eigenvalues.len(), 2);
    for ev in &result.hessian_eigenvalues {
        assert!(
            (*ev - 2.0).abs() < tolerances::HESSIAN_FD_ABS,
            "eigenvalue should be ~2, got {ev}"
        );
    }
}

#[test]
fn basecamp_landscape_analysis_saddle() {
    let d = cpu();
    let saddle = |x: &[f64]| -> f64 { x[0].mul_add(x[0], -(x[1] * x[1])) };
    let result = d.landscape_analysis(&saddle, &[0.0, 0.0], tolerances::HESSIAN_FD_STEP, 0.1);
    assert_eq!(
        result.saddle_index, 1,
        "monkey saddle has 1 negative eigenvalue"
    );
}

#[test]
fn basecamp_attention_spectral_analysis() {
    let d = cpu();
    #[rustfmt::skip]
    let attention = vec![
        0.5, 0.5,
        0.5, 0.5,
    ];
    let result = d.attention_spectral_analysis(&attention, 2);
    assert_eq!(result.eigenvalues.len(), 2);
    assert!(result.mean_ipr.is_finite());
    assert!(result.level_spacing_ratio.is_finite());
    for ev in &result.eigenvalues {
        assert!(ev.is_finite(), "eigenvalue must be finite");
    }
}

#[test]
fn basecamp_attention_spectral_identity() {
    let d = cpu();
    #[rustfmt::skip]
    let identity = vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    let result = d.attention_spectral_analysis(&identity, 3);
    assert_eq!(result.eigenvalues.len(), 3);
    for ev in &result.eigenvalues {
        assert!(
            (*ev - 1.0).abs() < tolerances::EIGH_JACOBI_EIGENVALUE,
            "identity eigenvalue should be ~1, got {ev}"
        );
    }
}

#[test]
fn basecamp_mlp_signal_propagation() {
    let d = cpu();
    let input = vec![1.0, 0.5, -0.3];
    #[rustfmt::skip]
    let w0 = vec![
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
    ];
    let variances = d.mlp_signal_propagation(&input, &[w0.as_slice()], &[2]);
    assert_eq!(variances.len(), 2, "input variance + 1 layer variance");
    for v in &variances {
        assert!(v.is_finite(), "variance must be finite");
        assert!(*v >= 0.0, "variance must be non-negative");
    }
}

#[test]
fn basecamp_mlp_signal_propagation_deep() {
    let d = cpu();
    let input = vec![1.0, 0.5];
    #[rustfmt::skip]
    let w0 = vec![0.5, 0.5, 0.3, 0.7, 0.1, 0.9];
    #[rustfmt::skip]
    let w1 = vec![0.4, 0.4, 0.4, 0.6, 0.6, 0.6];
    let variances = d.mlp_signal_propagation(&input, &[w0.as_slice(), w1.as_slice()], &[3, 2]);
    assert_eq!(variances.len(), 3, "input + 2 layers");
    assert!(variances[0] > 0.0, "non-zero input has positive variance");
}

#[test]
fn basecamp_belief_propagation_chain() {
    let d = cpu();
    let input = vec![0.5, 0.5];
    #[rustfmt::skip]
    let t1 = vec![0.9, 0.1, 0.2, 0.8];
    #[rustfmt::skip]
    let t2 = vec![0.7, 0.3, 0.4, 0.6];
    let dists = d.belief_propagation(&input, &[t1.as_slice(), t2.as_slice()], &[2, 2]);
    assert_eq!(dists.len(), 3, "input + 2 transitions");
    for (i, dist) in dists.iter().enumerate() {
        let sum: f64 = dist.iter().sum();
        assert!(
            (sum - 1.0).abs() < tolerances::CROSS_LANGUAGE,
            "distribution {i} not normalized: sum={sum}"
        );
    }
}

#[test]
fn basecamp_belief_propagation_identity_transition() {
    let d = cpu();
    let input = vec![0.3, 0.7];
    #[rustfmt::skip]
    let identity = vec![1.0, 0.0, 0.0, 1.0];
    let dists = d.belief_propagation(&input, &[identity.as_slice()], &[2]);
    let output = &dists[1];
    assert!(
        (output[0] - 0.3).abs() < tolerances::CROSS_LANGUAGE,
        "identity transition preserves distribution"
    );
    assert!(
        (output[1] - 0.7).abs() < tolerances::CROSS_LANGUAGE,
        "identity transition preserves distribution"
    );
}

#[test]
fn basecamp_agent_interaction_graph_no_connections() {
    let d = cpu();
    let positions = vec![0.0, 0.0, 100.0, 100.0];
    let adj = d.agent_interaction_graph(&positions, 2, 2, 1.0);
    assert_eq!(adj.len(), 4);
    assert!(
        adj.iter().all(|&v| v.abs() < tolerances::ZERO_DETECTION),
        "agents far apart should have no connections"
    );
}

#[test]
fn basecamp_agent_interaction_graph_symmetric() {
    let d = cpu();
    let positions = vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5];
    let adj = d.agent_interaction_graph(&positions, 3, 2, 2.0);
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (adj[i * 3 + j] - adj[j * 3 + i]).abs() < tolerances::ZERO_DETECTION,
                "adjacency matrix must be symmetric"
            );
        }
    }
}

// ── Dispatcher metadata (CPU-only) ────────────────────────

#[test]
fn cpu_fp64_strategy_native() {
    let d = cpu();
    assert_eq!(
        d.fp64_strategy(),
        barracuda::device::driver_profile::Fp64Strategy::Native
    );
}

#[test]
fn cpu_needs_pow_workaround_false() {
    let d = cpu();
    assert!(!d.needs_pow_workaround());
}

#[test]
fn cpu_bandwidth_tier_unknown() {
    let d = cpu();
    assert_eq!(
        d.bandwidth_tier(),
        barracuda::unified_hardware::BandwidthTier::Unknown
    );
}

#[test]
fn cpu_check_allocation_safe_ok() {
    let d = cpu();
    assert!(d.check_allocation_safe(1_000_000).is_ok());
}

#[test]
fn cpu_driver_profile_none() {
    let d = cpu();
    assert!(d.driver_profile().is_none());
}

// ── mixed_dispatch (CPU-only path) ─────────────────────────

#[test]
fn mixed_dispatch_cpu_only_small() {
    let d = cpu();
    let (result, substrate) = d.mixed_dispatch(
        "test_add",
        1.0,
        32,
        false,
        false,
        |_dev| Ok(42.0_f64),
        || 42.0_f64,
    );
    assert!((result - 42.0).abs() < tolerances::ZERO_DETECTION);
    assert_eq!(
        substrate,
        neural_spring_forge::mixed::MixedSubstrate::CpuOnly
    );
}

#[test]
fn mixed_dispatch_cpu_only_large() {
    let d = cpu();
    let (result, _substrate) = d.mixed_dispatch(
        "test_matmul",
        1000.0,
        8_000_000,
        false,
        false,
        |_dev| Ok(99.0_f64),
        || 99.0_f64,
    );
    assert!((result - 99.0).abs() < tolerances::ZERO_DETECTION);
}

// ── Dispatch ops via CPU fallback ──────────────────────────

#[test]
fn cpu_gelu_basic() {
    let d = cpu();
    let result = d.gelu(&[0.0, 1.0, -1.0]);
    assert_eq!(result.len(), 3);
    assert!(
        (result[0] - 0.0).abs() < tolerances::GELU_LARGE_INPUT,
        "gelu(0)≈0"
    );
    assert!(result[1] > 0.5, "gelu(1)>0.5");
    assert!(result[2] < 0.0, "gelu(-1)<0");
}

#[test]
fn cpu_hmm_forward_step_basic() {
    let d = cpu();
    let alpha = vec![0.6, 0.4];
    #[rustfmt::skip]
    let trans = vec![0.7, 0.3, 0.4, 0.6];
    let emit = vec![0.5, 0.5];
    let (new_alpha, scale) = d.hmm_forward_step(&alpha, &trans, &emit, 2);
    assert_eq!(new_alpha.len(), 2);
    assert!(scale > 0.0, "scale must be positive");
    let sum: f64 = new_alpha.iter().sum();
    assert!(
        (sum - 1.0).abs() < tolerances::EXACT_F64,
        "forward step normalizes"
    );
}
