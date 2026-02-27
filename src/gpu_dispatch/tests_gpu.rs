// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-path tests for [`Dispatcher`] operations.
//!
//! These tests exercise the GPU code paths through `barracuda::dispatch`
//! and `crate::gpu_ops`. They are skipped when no GPU adapter is available.

#![allow(clippy::expect_used)]

use super::*;
use crate::tolerances;

fn try_gpu_dispatcher() -> Option<Dispatcher> {
    let gpu_arc = crate::gpu::tests::shared_gpu()?;
    let dev = gpu_arc.wgpu_device().clone();
    let gpu = crate::gpu::Gpu::from_device(dev);
    Some(Dispatcher::from_gpu(gpu))
}

#[test]
fn gpu_dispatcher_metadata() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    assert!(d.has_gpu());
    assert_eq!(d.backend(), Backend::Gpu);
    assert!(d.capabilities().is_some());
    assert_ne!(d.adapter_name(), "(none)");
    assert!(d.wgpu_device().is_some());
    assert!(d.gpu().is_some());
    assert!(d.driver_profile().is_some());
}

#[test]
fn gpu_mat_mul_identity() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
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
            (v - expected).abs() < tolerances::CROSS_LANGUAGE,
            "gpu mat_mul identity [{i}]: {v} vs {expected}"
        );
    }
}

#[test]
fn gpu_frobenius_norm() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let a = vec![3.0, 4.0];
    assert!((d.frobenius_norm(&a) - 5.0).abs() < tolerances::DISPATCH_FROBENIUS_F64);
}

#[test]
fn gpu_transpose() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let t = d.transpose(&a, 2);
    assert!((t[0] - 1.0).abs() < tolerances::DISPATCH_TRANSPOSE_F64);
    assert!((t[1] - 3.0).abs() < tolerances::DISPATCH_TRANSPOSE_F64);
    assert!((t[2] - 2.0).abs() < tolerances::DISPATCH_TRANSPOSE_F64);
    assert!((t[3] - 4.0).abs() < tolerances::DISPATCH_TRANSPOSE_F64);
}

#[test]
fn gpu_softmax() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let result = d.softmax(&[1.0, 2.0, 3.0]);
    let total: f64 = result.iter().sum();
    assert!((total - 1.0).abs() < tolerances::GPU_SOFTMAX_SUM_F32);
}

#[test]
fn gpu_gelu() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let result = d.gelu(&[0.0, 1.0, -1.0]);
    assert_eq!(result.len(), 3);
    assert!((result[0]).abs() < tolerances::GPU_GELU_F32);
}

#[test]
fn gpu_l2_distance() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let dist = d.l2_distance(&[0.0, 0.0], &[3.0, 4.0]);
    assert!((dist - 5.0).abs() < tolerances::GPU_L2_DISPATCH_F32);
}

#[test]
fn gpu_mean() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    assert!((d.mean(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < tolerances::GPU_MEAN_DISPATCH_F32);
}

#[test]
fn gpu_variance() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let v = d.variance(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    assert!((v - 4.0).abs() < tolerances::GPU_VARIANCE_DISPATCH_F32);
}

#[test]
fn gpu_hmm_forward_step() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let alpha = vec![0.6, 0.4];
    #[rustfmt::skip]
    let trans = vec![0.7, 0.3, 0.4, 0.6];
    let emit = vec![0.5, 0.5];
    let (new_alpha, scale) = d.hmm_forward_step(&alpha, &trans, &emit, 2);
    assert_eq!(new_alpha.len(), 2);
    assert!(scale > 0.0);
}

#[test]
fn gpu_commutator() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let a = vec![1.0, 0.0, 0.0, 1.0];
    let comm = d.commutator(&a, &a, 2);
    for &v in &comm {
        assert!(
            v.abs() < tolerances::CPU_NORMAL_DISTANCE_SYMMETRIC_F64,
            "identity commutes with itself"
        );
    }
}

#[test]
fn gpu_distance_to_normal() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    #[rustfmt::skip]
    let sym = vec![2.0, 1.0, 1.0, 2.0];
    let dist = d.distance_to_normal(&sym, 2);
    assert!(
        dist < tolerances::CPU_NORMAL_DISTANCE_SYMMETRIC_F64,
        "symmetric matrix is normal"
    );
}

#[test]
fn gpu_boltzmann() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let result = d.boltzmann(&[1.0, 2.0, 3.0], 1.0);
    let total: f64 = result.iter().sum();
    assert!((total - 1.0).abs() < tolerances::GPU_BOLTZMANN_F32);
}

#[test]
fn gpu_shannon_entropy() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let h = d.shannon_entropy(&p);
    let expected = 4.0_f64.ln();
    assert!((h - expected).abs() < tolerances::GPU_ENTROPY_F32);
}

#[test]
fn gpu_pearson_correlation() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let r = d.pearson_correlation(&x, &y);
    assert!((r - 1.0).abs() < tolerances::GPU_PEARSON_F32);
}

#[test]
fn gpu_chi_squared() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let chi2 = d.chi_squared(&[10.0, 20.0, 30.0], &[20.0, 20.0, 20.0]);
    assert!((chi2 - 10.0).abs() < tolerances::GPU_CHI_SQUARED_F32);
}

#[test]
fn gpu_hill_activation_batch() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let result = d.hill_activation_batch(&[0.0, 1.0, 10.0], 1.0, 1.0, 2.0);
    assert_eq!(result.len(), 3);
    assert!((result[0]).abs() < tolerances::GPU_GELU_F32);
}

#[test]
fn gpu_eigh_diagonal() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let (vals, _) = d.eigh(&[2.0, 0.0, 0.0, 3.0], 2);
    let mut sorted = vals;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    assert!((sorted[0] - 2.0).abs() < tolerances::GPU_EIGH_DISPATCH_F64);
    assert!((sorted[1] - 3.0).abs() < tolerances::GPU_EIGH_DISPATCH_F64);
}

#[test]
fn gpu_mixed_dispatch_routes_gpu_path() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let (result, substrate) = d.mixed_dispatch(
        &MixedWorkload {
            op: "test_gpu_heavy",
            compute_us: 100_000.0,
            data_bytes: 256_000_000,
            npu_available: false,
            needs_realtime: false,
        },
        |_dev| Ok(42.0_f64),
        || 99.0_f64,
    );
    let _ = substrate;
    assert!(result.is_finite());
}

#[test]
fn gpu_mixed_dispatch_cpu_path() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let (result, _substrate) = d.mixed_dispatch(
        &MixedWorkload {
            op: "test_gpu_tiny",
            compute_us: 0.1,
            data_bytes: 16,
            npu_available: false,
            needs_realtime: false,
        },
        |_dev| Ok(42.0_f64),
        || 99.0_f64,
    );
    assert!(result.is_finite());
}

#[test]
fn gpu_mixed_dispatch_npu_fallback() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let (result, _substrate) = d.mixed_dispatch(
        &MixedWorkload {
            op: "test_npu",
            compute_us: 1000.0,
            data_bytes: 8_000_000,
            npu_available: true,
            needs_realtime: true,
        },
        |_dev| Ok(42.0_f64),
        || 99.0_f64,
    );
    assert!(result.is_finite());
}

#[test]
fn gpu_bandwidth_tier_detected() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let _tier = d.bandwidth_tier();
}

#[test]
fn gpu_fp64_strategy_detected() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let _strategy = d.fp64_strategy();
}

#[test]
fn gpu_check_allocation_safe() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    assert!(d.check_allocation_safe(1024).is_ok());
}

#[test]
fn gpu_allele_frequencies() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let pop = vec![2.0, 0.0, 0.0, 2.0];
    let freq = d.allele_frequencies(&pop, 2, 2);
    assert_eq!(freq.len(), 2);
}

#[test]
fn gpu_replicator_step() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let next = d.replicator_step(&[0.6, 0.4], &[[3.0, 0.0], [5.0, 1.0]], 0.01);
    let sum: f64 = next.iter().sum();
    assert!((sum - 1.0).abs() < tolerances::REPLICATOR_DYNAMICS);
}

#[test]
fn gpu_spectrum_chi_squared() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let obs = vec![10.0, 20.0, 30.0];
    let frac = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let chi2 = d.spectrum_chi_squared(&obs, &frac);
    assert!(chi2 >= 0.0);
}

#[test]
fn gpu_selection_coefficient() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let s = d.selection_coefficient(&[10.0, 20.0, 30.0], &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    assert!(s.is_finite());
}

#[test]
fn gpu_hmm_backward_step() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let beta_next = vec![1.0, 1.0];
    #[rustfmt::skip]
    let trans = vec![0.7, 0.3, 0.4, 0.6];
    let emit = vec![0.5, 0.5];
    let result = d.hmm_backward_step(&beta_next, &trans, &emit, 1.0, 2);
    assert_eq!(result.len(), 2);
}

#[test]
fn gpu_hmm_viterbi_step() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let delta_prev = vec![0.0, -1.0];
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

#[test]
fn gpu_hmm_forward_chain() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let initial = vec![0.6, 0.4];
    #[rustfmt::skip]
    let transition = vec![0.7, 0.3, 0.4, 0.6];
    #[rustfmt::skip]
    let emission = vec![0.5, 0.4, 0.1, 0.1, 0.3, 0.6];
    let obs = vec![0, 1, 2, 0];
    let ll = d.hmm_forward_chain(&initial, &transition, &emission, &obs, 2, 3);
    assert!(ll.is_finite());
}

#[test]
fn gpu_hmm_viterbi_chain() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let initial = vec![0.6, 0.4];
    #[rustfmt::skip]
    let transition = vec![0.7, 0.3, 0.4, 0.6];
    #[rustfmt::skip]
    let emission = vec![0.5, 0.4, 0.1, 0.1, 0.3, 0.6];
    let obs = vec![0, 1, 2, 0];
    let (path, log_prob) = d.hmm_viterbi_chain(&initial, &transition, &emission, &obs, 2, 3);
    assert_eq!(path.len(), 4);
    assert!(log_prob.is_finite());
}

#[test]
fn gpu_nucleotide_diversity() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let pop = vec![0.0, 1.0, 1.0, 0.0];
    let pi = d.nucleotide_diversity(&pop, 2, 2);
    assert!(pi >= 0.0);
}

#[test]
fn gpu_matrix_correlation() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    #[rustfmt::skip]
    let a = vec![
        0.0, 1.0, 2.0,
        1.0, 0.0, 3.0,
        2.0, 3.0, 0.0,
    ];
    let r = d.matrix_correlation(&a, &a, 3);
    assert!((r - 1.0).abs() < tolerances::GPU_PEARSON_F32);
}

#[test]
fn gpu_geographic_distances() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let coords = vec![(0.0, 0.0), (3.0, 4.0)];
    let dist = d.geographic_distances(&coords);
    assert_eq!(dist.len(), 4);
}

#[test]
fn gpu_thermal_diversity_correlation() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let r = d.thermal_diversity_correlation(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]);
    assert!((r - 1.0).abs() < tolerances::GPU_PEARSON_F32);
}

#[test]
fn gpu_inter_population_af_variance() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let pop_a = vec![2.0, 0.0, 0.0, 2.0];
    let pop_b = vec![0.0, 2.0, 2.0, 0.0];
    let populations: Vec<&[f64]> = vec![&pop_a, &pop_b];
    let var = d.inter_population_af_variance(&populations, &[2, 2], 2);
    assert!(var >= 0.0);
}

#[test]
fn gpu_pairwise_fst() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let pop_a = vec![2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0];
    let pop_b = vec![0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0];
    let fst = d.pairwise_fst(&pop_a, 5, &pop_b, 5, 2);
    assert!(fst.is_finite());
}

#[test]
fn gpu_global_fst() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let pop1 = vec![2.0, 0.0, 2.0, 0.0];
    let pop2 = vec![0.0, 2.0, 0.0, 2.0];
    let fst = d.global_fst(&[pop1, pop2], &[2, 2], 2);
    assert!(fst.is_finite());
}

#[test]
fn gpu_disorder_sweep() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let ham = vec![1.0, 0.0, 0.0, 1.0];
    let result = d.disorder_sweep(&ham, 2, 1);
    if let Some(iprs) = result {
        assert!(!iprs.is_empty());
    }
}

// ── baseCamp GPU paths ──────────────────────────────────────

#[test]
fn gpu_weight_spectral_analysis() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let weights = vec![1.0, 0.0, 0.0, 1.0];
    let result = d.weight_spectral_analysis(&weights, 2, 2);
    assert!(result.mean_ipr.is_finite());
}

#[test]
fn gpu_belief_propagation() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
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
}

#[test]
fn gpu_agent_interaction_graph() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(d) = try_gpu_dispatcher() else {
        return;
    };
    let positions = vec![0.0, 0.0, 1.0, 0.0, 5.0, 5.0];
    let adj = d.agent_interaction_graph(&positions, 3, 2, 2.0);
    assert_eq!(adj.len(), 9);
}
