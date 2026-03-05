// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    reason = "test infrastructure — GPU op validation"
)]

use super::*;
use crate::tolerances;
use std::sync::Arc;

/// Returns a shared GPU device plus a mutex guard that serializes access.
/// Reuses the crate-level shared Gpu instance so all GPU tests share
/// one Vulkan device (preventing driver-level resource races).
pub fn test_device() -> Option<(
    std::sync::MutexGuard<'static, ()>,
    Arc<barracuda::device::WgpuDevice>,
)> {
    let guard = crate::test_gpu_lock::acquire();
    let gpu = crate::gpu::tests::shared_gpu()?;
    Some((guard, gpu.wgpu_device().clone()))
}

#[test]
#[expect(clippy::cast_possible_truncation, reason = "intentional suppression")]
fn f32_f64_roundtrip_precision() {
    let x = [1.0_f64, 2.0, 3.0, 0.5, -1.0];
    for &orig in &x {
        let rt = f64::from(orig as f32);
        assert!(
            (orig - rt).abs() < tolerances::TENSOR_EXACT_F32,
            "roundtrip: {orig} -> {rt}"
        );
    }
}

#[test]
fn gpu_matmul_identity() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let n = 4;
    let a: Vec<f64> = (0..n * n).map(|i| (i + 1) as f64).collect();
    let mut ident = vec![0.0; n * n];
    for i in 0..n {
        ident[i * n + i] = 1.0;
    }
    let result = mat_mul_gpu(&a, &ident, n, &dev).unwrap();
    for (i, (&got, &want)) in result.iter().zip(a.iter()).enumerate() {
        assert!(
            (got - want).abs() < tolerances::GPU_MATMUL_IDENTITY_F32,
            "matmul identity mismatch at {i}"
        );
    }
}

#[test]
fn gpu_transpose_roundtrip() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let n = 3;
    let a: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
    let t = transpose_gpu(&a, n, &dev).unwrap();
    let tt = transpose_gpu(&t, n, &dev).unwrap();
    for (i, (&got, &want)) in tt.iter().zip(a.iter()).enumerate() {
        assert!(
            (got - want).abs() < tolerances::GPU_TRANSPOSE_F32,
            "transpose roundtrip at {i}"
        );
    }
}

#[test]
fn gpu_frobenius_3_4() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let norm = frobenius_norm_gpu(&[3.0, 4.0], &dev).unwrap();
    assert!(
        (norm - 5.0).abs() < tolerances::GPU_FROBENIUS_F32,
        "||[3,4]|| should be 5, got {norm}"
    );
}

#[test]
fn gpu_softmax_sums_to_one() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let sm = softmax_gpu(&[1.0, 2.0, 3.0], &dev).unwrap();
    let sum: f64 = sm.iter().sum();
    assert!(
        (sum - 1.0).abs() < tolerances::GPU_SOFTMAX_SUM_F32,
        "softmax sum = {sum}"
    );
    assert!(
        sm.iter().all(|&v| v > 0.0),
        "softmax values must be positive"
    );
}

#[test]
fn gpu_boltzmann_sums_to_one() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let b = boltzmann_gpu(&[0.1, 0.5, 0.9], 2.0, &dev).unwrap();
    let sum: f64 = b.iter().sum();
    assert!(
        (sum - 1.0).abs() < tolerances::GPU_SOFTMAX_SUM_F32,
        "boltzmann sum = {sum}"
    );
}

#[test]
fn gpu_gelu_zero() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let g = gelu_gpu(&[0.0], &dev).unwrap();
    assert!(
        g[0].abs() < tolerances::GPU_GELU_F32,
        "GELU(0) should be ~0, got {}",
        g[0]
    );
}

#[test]
fn gpu_l2_distance_known() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let d = l2_distance_gpu(&[0.0, 0.0], &[3.0, 4.0], &dev).unwrap();
    assert!(
        (d - 5.0).abs() < tolerances::GPU_L2_DISPATCH_F32,
        "L2([0,0],[3,4]) should be 5, got {d}"
    );
}

#[test]
fn gpu_mean_known() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let m = mean_gpu(&[2.0, 4.0, 6.0], &dev).unwrap();
    assert!(
        (m - 4.0).abs() < tolerances::GPU_MEAN_DISPATCH_F32,
        "mean should be 4, got {m}"
    );
}

#[test]
fn gpu_sum_known() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let s = sum_gpu(&[1.0, 2.0, 3.0], &dev).unwrap();
    assert!(
        (s - 6.0).abs() < tolerances::GPU_SUM_DISPATCH_F32,
        "sum should be 6, got {s}"
    );
}

#[test]
fn gpu_max_known() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let m = max_gpu(&[1.0, 5.0, 3.0], &dev).unwrap();
    assert!(
        (m - 5.0).abs() < tolerances::GPU_MAX_DISPATCH_F32,
        "max should be 5, got {m}"
    );
}

#[test]
fn gpu_variance_known() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let v = variance_gpu(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], &dev).unwrap();
    assert!(v > 0.0, "variance must be positive");
    assert!(
        (v - 4.0).abs() < tolerances::GPU_CHI_SQUARED_F32,
        "variance ≈ 4, got {v}"
    );
}

#[test]
fn gpu_entropy_uniform() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let h = shannon_entropy_gpu(&[0.25, 0.25, 0.25, 0.25], &dev).unwrap();
    let expected = (4.0_f64).ln();
    assert!(
        (h - expected).abs() < tolerances::GPU_ENTROPY_F32,
        "entropy(uniform4) ≈ {expected}, got {h}"
    );
}

#[test]
fn gpu_chi_squared_known() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let chi2 = chi_squared_gpu(&[10.0, 20.0, 30.0, 40.0], &[25.0, 25.0, 25.0, 25.0], &dev).unwrap();
    let expected = 20.0; // (15² + 5² + 5² + 15²) / 25
    assert!(
        (chi2 - expected).abs() < tolerances::GPU_CHI_SQUARED_F32,
        "chi2 ≈ {expected}, got {chi2}"
    );
}

#[test]
fn gpu_kl_divergence_identical() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let kl = kl_divergence_gpu(&[0.25, 0.25, 0.25, 0.25], &[0.25, 0.25, 0.25, 0.25], &dev).unwrap();
    assert!(
        kl.abs() < tolerances::GPU_KL_DISPATCH_F32,
        "KL(p,p) should be 0, got {kl}"
    );
}

#[test]
fn gpu_replicator_step_preserves_sum() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let freq = [0.6, 0.4];
    let payoff = [[3.0, 0.0], [5.0, 1.0]];
    let new_freq = replicator_step_gpu(&freq, &payoff, 0.01, &dev).unwrap();
    let sum = new_freq[0] + new_freq[1];
    assert!(
        (sum - 1.0).abs() < tolerances::GPU_HMM_STEP_F32,
        "replicator sum = {sum}"
    );
}

#[test]
fn gpu_allele_frequencies_known() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let pop = vec![2.0, 0.0, 1.0, 1.0, 0.0, 2.0];
    let freqs = allele_frequencies_gpu(&pop, 3, 2, &dev).unwrap();
    assert_eq!(freqs.len(), 2);
    assert!((freqs[0] - 0.5).abs() < tolerances::GPU_VARIANCE_DISPATCH_F32);
    assert!((freqs[1] - 0.5).abs() < tolerances::GPU_VARIANCE_DISPATCH_F32);
}

#[test]
fn gpu_pearson_perfect_correlation() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let r = pearson_correlation_gpu(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0], &dev).unwrap();
    assert!(
        (r - 1.0).abs() < tolerances::GPU_PEARSON_F32,
        "Pearson r(x, 2x) ≈ 1, got {r}"
    );
}

#[test]
fn gpu_commutator_identity_is_zero() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let n = 3;
    let mut ident = vec![0.0; n * n];
    for i in 0..n {
        ident[i * n + i] = 1.0;
    }
    let comm = commutator_gpu(&ident, &ident, n, &dev).unwrap();
    for &v in &comm {
        assert!(
            v.abs() < tolerances::GPU_GELU_F32,
            "[I,I] should be zero, got {v}"
        );
    }
}

#[test]
fn gpu_spectrum_chi_squared_uniform() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let observed = vec![25.0, 25.0, 25.0, 25.0];
    let fracs = vec![0.25, 0.25, 0.25, 0.25];
    let chi2 = spectrum_chi_squared_gpu(&observed, &fracs, &dev).unwrap();
    assert!(
        chi2.abs() < tolerances::GPU_CHI_SQUARED_F32,
        "uniform should give chi2 ≈ 0, got {chi2}"
    );
}

#[test]
fn gpu_selection_coefficient_neutral() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let obs = vec![25.0, 25.0, 25.0, 25.0];
    let neutral = vec![0.25, 0.25, 0.25, 0.25];
    let s = selection_coefficient_gpu(&obs, &neutral, &dev).unwrap();
    assert!(
        s.abs() < tolerances::GPU_SOFTMAX_DISPATCH_F32,
        "neutral should give s ≈ 0, got {s}"
    );
}

// ── Eigensolver ─────────────────────────────────────────────

#[test]
fn gpu_eigh_diagonal() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let a = vec![2.0, 0.0, 0.0, 3.0];
    let (vals, _vecs) = eigh_gpu(&a, 2, &dev).unwrap();
    let mut sorted = vals;
    sorted.sort_by(f64::total_cmp);
    assert!((sorted[0] - 2.0).abs() < tolerances::GPU_CHI_SQUARED_F32);
    assert!((sorted[1] - 3.0).abs() < tolerances::GPU_CHI_SQUARED_F32);
}

#[test]
fn gpu_disorder_sweep_basic() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    #[rustfmt::skip]
        let hamiltonians = vec![
            1.0, 0.1, 0.1, 2.0,
            3.0, 0.2, 0.2, 4.0,
        ];
    let iprs = disorder_sweep_gpu(&hamiltonians, 2, 2, &dev).unwrap();
    assert_eq!(iprs.len(), 2);
    for &v in &iprs {
        assert!(v > 0.0 && v.is_finite(), "IPR should be positive");
    }
}

// ── Population genetics ─────────────────────────────────────

#[test]
fn gpu_nucleotide_diversity_basic() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let pop = vec![0.0, 1.0, 1.0, 0.0];
    let pi = nucleotide_diversity_gpu(&pop, 2, 2, &dev).unwrap();
    assert!(pi >= 0.0 && pi.is_finite());
}

#[test]
fn gpu_matrix_correlation_self() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let a = vec![0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0];
    let r = matrix_correlation_gpu(&a, &a, 3, &dev).unwrap();
    assert!(
        (r - 1.0).abs() < tolerances::GPU_PEARSON_F32,
        "self-correlation ≈ 1, got {r}"
    );
}

#[test]
fn gpu_geographic_distances_basic() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let coords = vec![(0.0, 0.0), (3.0, 4.0)];
    let dist = geographic_distance_matrix_gpu(&coords, &dev).unwrap();
    assert_eq!(dist.len(), 4);
    assert!(
        dist[0].abs() < tolerances::GPU_VARIANCE_DISPATCH_F32,
        "self-distance ≈ 0"
    );
    assert!(
        (dist[1] - 5.0).abs() < tolerances::GPU_CHI_SQUARED_F32,
        "dist ≈ 5"
    );
}

#[test]
fn gpu_thermal_diversity_basic() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let pi = vec![1.0, 2.0, 3.0];
    let temp = vec![10.0, 20.0, 30.0];
    let r = thermal_diversity_correlation_gpu(&pi, &temp, &dev).unwrap();
    assert!(
        (r - 1.0).abs() < tolerances::GPU_PEARSON_F32,
        "perfect linear → r ≈ 1, got {r}"
    );
}

#[test]
fn gpu_inter_population_af_variance_basic() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let pop1 = vec![2.0, 0.0, 0.0, 2.0];
    let pop2 = vec![0.0, 2.0, 2.0, 0.0];
    let populations: Vec<&[f64]> = vec![&pop1, &pop2];
    let n_individuals = vec![2, 2];
    let var = inter_population_af_variance_gpu(&populations, &n_individuals, 2, &dev).unwrap();
    assert!(var >= 0.0 && var.is_finite());
}

// ── Linalg (distance to normal) ─────────────────────────────

#[test]
fn gpu_distance_to_normal_symmetric() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let sym = vec![2.0, 1.0, 1.0, 2.0];
    let d = distance_to_normal_gpu(&sym, 2, &dev).unwrap();
    assert!(
        d < tolerances::GPU_COMMUTATOR_F32,
        "symmetric → normal → d ≈ 0, got {d}"
    );
}

// ── Reduction (neural forward) ──────────────────────────────

#[test]
fn gpu_neural_forward_basic() {
    use super::NeuralForwardParams;

    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let params = NeuralForwardParams {
        input: &[1.0, 0.5],
        weights_hidden: &[1.0, 0.0, 0.0, 1.0],
        bias_hidden: &[0.0, 0.0],
        weights_output: &[1.0, 1.0],
        bias_output: &[0.0],
        hidden_size: 2,
        output_size: 1,
    };
    let result = neural_forward_gpu(&params, &dev).unwrap();
    assert_eq!(result.len(), 1);
    assert!(result[0].is_finite());
}

// ── Population: FST ─────────────────────────────────────────

#[test]
fn gpu_pairwise_fst_identical_populations() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    // Large uniform populations: WC estimator converges to 0 for identical pops.
    // Small N can produce negative FST (documented WC estimator property).
    let n = 20;
    let pop: Vec<f64> = (0..n * 4)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let fst = pairwise_fst_gpu(&pop, n, &pop, n, 4, &dev).unwrap();
    assert!(
        fst.abs() < tolerances::GPU_CHI_SQUARED_F32,
        "FST of identical populations should be near 0, got {fst}"
    );
    assert!(fst.is_finite());
}

#[test]
fn gpu_pairwise_fst_divergent_populations() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let pop_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    let pop_b = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let fst = pairwise_fst_gpu(&pop_a, 3, &pop_b, 3, 2, &dev).unwrap();
    assert!(fst.is_finite(), "FST should be finite");
    assert!(fst > 0.0, "divergent populations → FST > 0, got {fst}");
}

#[test]
fn gpu_global_fst_single_population() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let pops = vec![vec![1.0, 0.0, 0.0, 1.0]];
    let fst = global_fst_gpu(&pops, &[2], 2, &dev).unwrap();
    assert!(
        fst.abs() < tolerances::CROSS_LANGUAGE,
        "single population → FST = 0, got {fst}"
    );
}

#[test]
fn gpu_global_fst_divergent() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let fixed_a = vec![2.0, 2.0, 2.0, 2.0];
    let fixed_b = vec![0.0, 0.0, 0.0, 0.0];
    let pops = vec![fixed_a, fixed_b];
    let fst = global_fst_gpu(&pops, &[2, 2], 2, &dev).unwrap();
    assert!(fst.is_finite());
    assert!(fst > 0.0, "divergent → FST > 0, got {fst}");
}

#[test]
fn gpu_nucleotide_diversity_single_individual() {
    let Some((_guard, dev)) = test_device() else {
        return;
    };
    let pop = vec![1.0, 0.0];
    let pi = nucleotide_diversity_gpu(&pop, 1, 2, &dev).unwrap();
    assert!(
        pi.abs() < tolerances::CROSS_LANGUAGE,
        "single individual → zero diversity, got {pi}"
    );
}
