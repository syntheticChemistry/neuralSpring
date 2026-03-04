// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU-path tests for [`Dispatcher`] operations.
//!
//! These tests exercise CPU fallback paths — no GPU adapter required.
//! They validate correctness of the local CPU reference implementations
//! and metadata queries when running in CPU-only mode.

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
        &MixedWorkload {
            op: "test_add",
            compute_us: 1.0,
            data_bytes: 32,
            npu_available: false,
            needs_realtime: false,
        },
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
        &MixedWorkload {
            op: "test_matmul",
            compute_us: 1000.0,
            data_bytes: 8_000_000,
            npu_available: false,
            needs_realtime: false,
        },
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

// ── dispatch_bio CPU fallback ─────────────────────────────

#[test]
fn cpu_hill_gate_basic() {
    let d = cpu();
    let cfg = crate::gpu_ops::HillGateConfig {
        vmax: 1.0,
        k_a: 0.5,
        k_b: 0.5,
        n_a: 2.0,
        n_b: 2.0,
    };
    let result = d.hill_gate(&[0.0, 0.5, 1.0], &[1.0], &cfg);
    assert_eq!(result.len(), 3, "3 inputs × 1 input = 3 outputs");
    assert!(
        result[0].abs() < tolerances::ZERO_DETECTION,
        "hill(0, _) ≈ 0"
    );
    assert!(result[1] > 0.0, "hill(0.5, 1.0) > 0");
    assert!(result[2] > result[1], "hill(1.0, _) > hill(0.5, _)");
}

#[test]
fn cpu_hill_gate_symmetric_inputs() {
    let d = cpu();
    let cfg = crate::gpu_ops::HillGateConfig {
        vmax: 2.0,
        k_a: 1.0,
        k_b: 1.0,
        n_a: 2.0,
        n_b: 2.0,
    };
    let r1 = d.hill_gate(&[1.0], &[1.0], &cfg);
    assert_eq!(r1.len(), 1);
    assert!(
        (r1[0] - 0.5).abs() < 0.1,
        "at K with n=2, Hill ≈ 0.5, got {}",
        r1[0]
    );
}

#[test]
fn cpu_hill_gate_empty_inputs() {
    let d = cpu();
    let cfg = crate::gpu_ops::HillGateConfig {
        vmax: 1.0,
        k_a: 0.5,
        k_b: 0.5,
        n_a: 2.0,
        n_b: 2.0,
    };
    let result = d.hill_gate(&[], &[1.0], &cfg);
    assert!(result.is_empty(), "empty input_a → empty output");
}

#[test]
fn cpu_multi_obj_fitness_basic() {
    let d = cpu();
    let genome_len = 4;
    let n_objectives = 2;
    let genotypes = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let result = d.multi_obj_fitness(&genotypes, 2, genome_len, n_objectives);
    assert_eq!(result.len(), 2 * n_objectives, "2 genotypes × 2 objectives");
    assert!(
        result.iter().all(|v| v.is_finite()),
        "all fitness values finite"
    );
}

#[test]
fn cpu_multi_obj_fitness_single() {
    let d = cpu();
    let genotypes = vec![0.5; 8];
    let result = d.multi_obj_fitness(&genotypes, 1, 8, 3);
    assert_eq!(result.len(), 3, "1 genotype × 3 objectives");
}

#[test]
fn cpu_swarm_nn_forward_basic() {
    let d = cpu();
    // neural_forward expects fixed layout: 4 input, 4 bias, 4×5 h→o, 5 o_bias = 33
    let dims = crate::gpu_ops::SwarmNnDims {
        n_controllers: 2,
        n_evals: 1,
        input_dim: 1,
        hidden_dim: 4,
        output_dim: 5,
    };
    let weights_per = dims.input_dim * dims.hidden_dim
        + dims.hidden_dim
        + dims.hidden_dim * dims.output_dim
        + dims.output_dim;
    assert_eq!(weights_per, 33, "swarm NN expects 33 params per controller");
    let weights = vec![0.5_f64; dims.n_controllers * weights_per];
    let inputs = vec![0.5_f64; dims.n_controllers * dims.n_evals * dims.input_dim];
    let result = d.swarm_nn_forward(&weights, &inputs, &dims);
    assert_eq!(
        result.len(),
        dims.n_controllers * dims.n_evals,
        "one action per controller per eval"
    );
}

// ── Activation dispatch (cpu fallback paths) ────────────────

#[test]
fn cpu_softmax_row_wise_basic() {
    let d = cpu();
    #[rustfmt::skip]
    let matrix = vec![
        1.0, 2.0, 3.0,
        3.0, 2.0, 1.0,
    ];
    let result = d.softmax_row_wise(&matrix, 2, 3);
    assert_eq!(result.len(), 6);
    let row0_sum: f64 = result[0..3].iter().sum();
    let row1_sum: f64 = result[3..6].iter().sum();
    assert!(
        (row0_sum - 1.0).abs() < tolerances::CROSS_LANGUAGE,
        "row 0 sum = {row0_sum}"
    );
    assert!(
        (row1_sum - 1.0).abs() < tolerances::CROSS_LANGUAGE,
        "row 1 sum = {row1_sum}"
    );
}

#[test]
fn cpu_softmax_row_wise_single_row() {
    let d = cpu();
    let row = vec![0.0, 0.0, 0.0];
    let result = d.softmax_row_wise(&row, 1, 3);
    assert_eq!(result.len(), 3);
    for &v in &result {
        assert!(
            (v - 1.0 / 3.0).abs() < tolerances::CROSS_LANGUAGE,
            "uniform softmax expected 1/3, got {v}"
        );
    }
}

#[test]
fn cpu_hill_activation_batch_single() {
    let d = cpu();
    let x = vec![1.0];
    let result = d.hill_activation_batch(&x, 1.0, 0.5, 2.0);
    assert_eq!(result.len(), 1);
    assert!(result[0] > 0.0 && result[0] <= 1.0);
}

#[test]
fn cpu_gelu_matches_transformer_gelu() {
    let d = cpu();
    let xs = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let dispatched = d.gelu(&xs);
    for (i, &x) in xs.iter().enumerate() {
        let expected = crate::transformer::gelu(x);
        assert!(
            (dispatched[i] - expected).abs() < tolerances::EXACT_F64,
            "gelu({x}): dispatch={}, direct={expected}",
            dispatched[i]
        );
    }
}

// ── Dispatcher metadata and accessors ──────────────────────

#[test]
fn cpu_only_driver_profile_none() {
    let d = cpu();
    assert!(d.driver_profile().is_none());
    assert!(!d.needs_pow_workaround());
    assert!(d.check_allocation_safe(1_000_000).is_ok());
}

#[test]
fn cpu_only_bandwidth_tier_unknown() {
    let d = cpu();
    assert_eq!(
        format!("{:?}", d.bandwidth_tier()),
        format!("{:?}", barracuda::unified_hardware::BandwidthTier::Unknown)
    );
}

#[test]
fn cpu_fp64_strategy_defaults_native() {
    let d = cpu();
    assert_eq!(
        format!("{:?}", d.fp64_strategy()),
        format!(
            "{:?}",
            barracuda::device::driver_profile::Fp64Strategy::Native
        )
    );
}

#[test]
fn cpu_mixed_dispatch_routes_cpu() {
    let d = cpu();
    let workload = MixedWorkload {
        op: "test_op",
        compute_us: 100.0,
        data_bytes: 1024,
        npu_available: false,
        needs_realtime: false,
    };
    let (result, _substrate) = d.mixed_dispatch(&workload, |_dev| Ok(42.0_f64), || 99.0);
    assert!((result - 99.0).abs() < tolerances::ZERO_DETECTION);
}

// ── Additional dispatch_stats coverage ─────────────────────

#[test]
fn cpu_l2_distance_known() {
    let d = cpu();
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    // Analytical: sqrt((1-0)^2 + (0-1)^2) = sqrt(2) ≈ 1.4142
    let dist = d.l2_distance(&a, &b);
    assert!(
        (dist - std::f64::consts::SQRT_2).abs() < tolerances::CROSS_LANGUAGE,
        "L2 distance mismatch: {dist}"
    );
}

#[test]
fn cpu_kl_divergence_identical() {
    let d = cpu();
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let kl = d.kl_divergence(&p, &p);
    assert!(kl.abs() < tolerances::EXACT_F64);
}

#[test]
fn cpu_softmax_row_wise_sums_to_one() {
    let d = cpu();
    let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = d.softmax_row_wise(&matrix, 2, 3);
    let sum_r0: f64 = result[..3].iter().sum();
    let sum_r1: f64 = result[3..].iter().sum();
    assert!((sum_r0 - 1.0).abs() < tolerances::CROSS_LANGUAGE);
    assert!((sum_r1 - 1.0).abs() < tolerances::CROSS_LANGUAGE);
}

#[test]
fn cpu_boltzmann_normalizes() {
    let d = cpu();
    let fitnesses = vec![1.0, 2.0, 3.0, 4.0];
    let probs = d.boltzmann(&fitnesses, 1.0);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < tolerances::CROSS_LANGUAGE);
    assert!(probs[3] > probs[0], "higher fitness → higher probability");
}

#[test]
fn cpu_shannon_entropy_uniform() {
    let d = cpu();
    // Analytical: H(uniform(4)) = ln(4) ≈ 1.386
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let h = d.shannon_entropy(&p);
    assert!(
        (h - 4.0_f64.ln()).abs() < tolerances::CROSS_LANGUAGE,
        "Shannon entropy mismatch: {h}"
    );
}
