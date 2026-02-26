// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spectral analysis of neural network weight matrices.
//!
//! baseCamp Sub-thesis 01: Weight Matrices as Disordered Hamiltonians.
//!
//! Treats trained weight matrices as Anderson Hamiltonians and applies
//! condensed-matter diagnostics (IPR, level spacing ratio, ESD) to
//! predict generalization vs memorization.
//!
//! ## Grounding papers
//!
//! - Martin & Mahoney (2021) "Implicit Self-Regularization in DNNs" (JMLR)
//! - Gurbuzbalaban et al. (2025) "From SGD to Spectra"
//! - Ouyang (2025) "Rethinking Over-Smoothing via Anderson Localization"
//!
//! ## Validated primitives
//!
//! - [`crate::eigh::eigh_householder_qr`] — eigendecomposition
//! - [`crate::anderson_localization::ipr`] — inverse participation ratio
//! - [`crate::anderson_localization::mean_ipr`] — mean IPR over eigenvectors

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::needless_range_loop
)]

use crate::anderson_localization::{ipr, mean_ipr};
use crate::eigh::eigh_householder_qr;
use crate::primitives::LOG_GUARD;

/// Symmetrize a rectangular weight matrix into a square Hamiltonian.
///
/// For an m×n weight matrix W, constructs the (m+n)×(m+n) block matrix:
/// ```text
///     H = [  0    W  ]
///         [ W^T   0  ]
/// ```
/// This preserves the singular value spectrum: eigenvalues of H are
/// ±σ_i where σ_i are singular values of W.
#[must_use]
pub fn weight_to_hamiltonian(weights: &[f64], m: usize, n: usize) -> Vec<f64> {
    let dim = m + n;
    let mut h = vec![0.0; dim * dim];
    for i in 0..m {
        for j in 0..n {
            let w = weights[i * n + j];
            h[i * dim + (m + j)] = w;
            h[(m + j) * dim + i] = w;
        }
    }
    h
}

/// Empirical spectral density: histogram of eigenvalues into `n_bins` bins.
///
/// Returns `(bin_centers, bin_counts)` normalized so that counts sum to 1.
///
/// Delegates to `barracuda::stats::empirical_spectral_density` (absorbed S54, M-011).
#[must_use]
pub fn empirical_spectral_density(eigenvalues: &[f64], n_bins: usize) -> (Vec<f64>, Vec<f64>) {
    barracuda::stats::empirical_spectral_density(eigenvalues, n_bins)
}

/// Level spacing ratio: r = min(s_i, s_{i+1}) / max(s_i, s_{i+1}).
///
/// For GOE (extended states): mean r ≈ 0.531.
/// For Poisson (localized states): mean r ≈ 0.386.
///
/// Input eigenvalues must be sorted in ascending order.
///
/// Delegates to `barracuda::spectral::level_spacing_ratio` (upstream).
#[must_use]
pub fn level_spacing_ratio(eigenvalues: &[f64]) -> f64 {
    barracuda::spectral::level_spacing_ratio(eigenvalues)
}

/// GOE (Gaussian Orthogonal Ensemble) expected level spacing ratio.
pub const GOE_LEVEL_SPACING: f64 = 0.530_95;

/// Poisson expected level spacing ratio (localized regime).
pub const POISSON_LEVEL_SPACING: f64 = 0.386_29;

/// Marchenko-Pastur quarter-circle law: expected spectral density for
/// random matrices with aspect ratio γ = m/n.
///
/// Returns the MP bounds (λ_min, λ_max) for singular values squared.
///
/// Delegates to `barracuda::stats::marchenko_pastur_bounds` (absorbed S54, M-012).
#[must_use]
pub fn marchenko_pastur_bounds(gamma: f64) -> (f64, f64) {
    barracuda::stats::marchenko_pastur_bounds(gamma)
}

/// Fraction of eigenvalues outside the Marchenko-Pastur bulk.
///
/// Eigenvalues beyond MP bounds indicate learned structure (not random).
/// Higher fraction = more departure from random = more learned features.
#[must_use]
pub fn marchenko_pastur_departure(eigenvalues: &[f64], gamma: f64) -> f64 {
    if eigenvalues.is_empty() {
        return 0.0;
    }
    let (mp_min, mp_max) = marchenko_pastur_bounds(gamma);
    let outside = eigenvalues
        .iter()
        .filter(|&&ev| ev < mp_min || ev > mp_max)
        .count();
    outside as f64 / eigenvalues.len() as f64
}

/// Spectral entropy: Shannon entropy of the normalized eigenvalue distribution.
///
/// High entropy = uniform spectrum (random matrix).
/// Low entropy = concentrated spectrum (structured/learned matrix).
///
/// Delegates to `barracuda::stats::shannon_from_frequencies` after normalizing
/// absolute eigenvalues to a probability distribution (sum = 1).
#[must_use]
pub fn spectral_entropy(eigenvalues: &[f64]) -> f64 {
    if eigenvalues.is_empty() {
        return 0.0;
    }
    let abs_vals: Vec<f64> = eigenvalues.iter().map(|&ev| ev.abs()).collect();
    let total: f64 = abs_vals.iter().sum();
    if total < LOG_GUARD {
        return 0.0;
    }
    let frequencies: Vec<f64> = abs_vals.iter().map(|&v| v / total).collect();
    barracuda::stats::shannon_from_frequencies(&frequencies)
}

/// Compute full weight matrix spectral analysis.
///
/// Given a weight matrix (m×n, row-major), returns:
/// - eigenvalues of the symmetrized Hamiltonian
/// - mean IPR of the eigenstates
/// - level spacing ratio
/// - spectral entropy
#[must_use]
pub fn weight_spectral_analysis(weights: &[f64], m: usize, n: usize) -> WeightSpectralResult {
    let h = weight_to_hamiltonian(weights, m, n);
    let dim = m + n;
    let decomp = eigh_householder_qr(&h, dim);

    let mut eigenvalues = decomp.eigenvalues.clone();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let ipr_val = mean_ipr(&decomp.eigenvectors, dim);
    let lsr = level_spacing_ratio(&eigenvalues);
    let entropy = spectral_entropy(&eigenvalues);
    let gamma = m as f64 / n.max(1) as f64;
    let mp_departure = marchenko_pastur_departure(&eigenvalues, gamma);

    WeightSpectralResult {
        eigenvalues,
        mean_ipr: ipr_val,
        level_spacing_ratio: lsr,
        spectral_entropy: entropy,
        mp_departure,
    }
}

/// Result of weight matrix spectral analysis.
#[derive(Debug, Clone)]
pub struct WeightSpectralResult {
    /// Sorted eigenvalues of the symmetrized Hamiltonian.
    pub eigenvalues: Vec<f64>,
    /// Mean inverse participation ratio of eigenstates.
    pub mean_ipr: f64,
    /// Mean level spacing ratio (GOE ≈ 0.531, Poisson ≈ 0.386).
    pub level_spacing_ratio: f64,
    /// Shannon entropy of eigenvalue distribution.
    pub spectral_entropy: f64,
    /// Fraction of eigenvalues outside Marchenko-Pastur bulk.
    pub mp_departure: f64,
}

/// Construct a `WeightSpectralResult` from pre-computed eigen-decomposition.
///
/// Used by [`crate::gpu_dispatch::Dispatcher::weight_spectral_analysis`]
/// where the eigensolve is performed on GPU. `gamma` is the aspect ratio
/// m/n of the original weight matrix (needed for Marchenko-Pastur bounds).
#[must_use]
pub fn spectral_result_from_decomposition(
    mut eigenvalues: Vec<f64>,
    eigenvectors: &[f64],
    dim: usize,
    gamma: f64,
) -> WeightSpectralResult {
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ipr_val = mean_ipr(eigenvectors, dim);
    let lsr = level_spacing_ratio(&eigenvalues);
    let entropy = spectral_entropy(&eigenvalues);
    let mp_departure = marchenko_pastur_departure(&eigenvalues, gamma);
    WeightSpectralResult {
        eigenvalues,
        mean_ipr: ipr_val,
        level_spacing_ratio: lsr,
        spectral_entropy: entropy,
        mp_departure,
    }
}

/// Compare spectral properties of two weight matrices.
///
/// Returns (delta_ipr, delta_lsr, delta_entropy) as signed differences
/// (result2 - result1). Positive delta_ipr means matrix 2 is more
/// delocalized (better generalization prediction).
#[must_use]
pub fn spectral_comparison(
    r1: &WeightSpectralResult,
    r2: &WeightSpectralResult,
) -> (f64, f64, f64) {
    (
        r2.mean_ipr - r1.mean_ipr,
        r2.level_spacing_ratio - r1.level_spacing_ratio,
        r2.spectral_entropy - r1.spectral_entropy,
    )
}

/// Compute IPR of a single activation vector.
///
/// High IPR = information concentrated in few neurons (localized).
/// Low IPR = information spread across neurons (delocalized).
#[must_use]
pub fn activation_ipr(activations: &[f64]) -> f64 {
    let norm_sq: f64 = activations.iter().map(|&x| x * x).sum();
    if norm_sq < LOG_GUARD {
        return 0.0;
    }
    let normalized: Vec<f64> = activations.iter().map(|&x| x / norm_sq.sqrt()).collect();
    ipr(&normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::tolerances;

    fn random_weight_matrix(m: usize, n: usize, rng: &mut Rng) -> Vec<f64> {
        (0..m * n).map(|_| rng.normal()).collect()
    }

    #[test]
    fn hamiltonian_is_symmetric() {
        let mut rng = Rng::new(42);
        let w = random_weight_matrix(4, 6, &mut rng);
        let h = weight_to_hamiltonian(&w, 4, 6);
        let dim = 10;
        for i in 0..dim {
            for j in 0..dim {
                assert!(
                    (h[i * dim + j] - h[j * dim + i]).abs() < tolerances::ZERO_DETECTION,
                    "H not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn esd_sums_to_one() {
        let eigenvalues: Vec<f64> = (0..20).map(|i| f64::from(i) * 0.5).collect();
        let (_, counts) = empirical_spectral_density(&eigenvalues, 10);
        let sum: f64 = counts.iter().sum();
        assert!(
            (sum - 1.0).abs() < tolerances::EXACT_F64,
            "ESD should sum to 1, got {sum}"
        );
    }

    #[test]
    fn level_spacing_bounds() {
        let sorted: Vec<f64> = (0..50).map(f64::from).collect();
        let r = level_spacing_ratio(&sorted);
        assert!(r > 0.9, "uniform spacing should have r near 1.0, got {r}");
    }

    #[test]
    fn mp_bounds_unit_aspect() {
        let (lo, hi) = marchenko_pastur_bounds(1.0);
        assert!((lo - 0.0).abs() < tolerances::EXACT_F64);
        assert!((hi - 4.0).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn spectral_entropy_positive() {
        let eigenvalues = vec![1.0, 2.0, 3.0, 4.0];
        let s = spectral_entropy(&eigenvalues);
        assert!(s > 0.0, "entropy should be positive, got {s}");
    }

    #[test]
    fn random_vs_structured_ipr() {
        let mut rng = Rng::new(42);
        let random_w = random_weight_matrix(8, 8, &mut rng);
        let r_random = weight_spectral_analysis(&random_w, 8, 8);

        let mut low_rank = vec![0.0; 64];
        for i in 0..8 {
            low_rank[i * 8] = 1.0;
        }
        let r_lowrank = weight_spectral_analysis(&low_rank, 8, 8);

        assert!(
            r_lowrank.mean_ipr > r_random.mean_ipr * 0.5,
            "low-rank should have comparable or higher IPR"
        );
    }

    #[test]
    fn determinism() {
        let mut rng = Rng::new(42);
        let w = random_weight_matrix(8, 8, &mut rng);
        let r1 = weight_spectral_analysis(&w, 8, 8);
        let r2 = weight_spectral_analysis(&w, 8, 8);
        assert!(
            r1.eigenvalues
                .iter()
                .zip(r2.eigenvalues.iter())
                .all(|(a, b)| (a - b).abs() < f64::EPSILON),
            "eigenvalue determinism failure"
        );
        assert!(
            (r1.mean_ipr - r2.mean_ipr).abs() < f64::EPSILON,
            "mean_ipr determinism: {} vs {}",
            r1.mean_ipr,
            r2.mean_ipr
        );
    }

    // ── Edge cases for coverage ──────────────────────────────────

    #[test]
    fn esd_empty_eigenvalues() {
        let (centers, counts) = empirical_spectral_density(&[], 10);
        assert!(centers.is_empty());
        assert!(counts.is_empty());
    }

    #[test]
    fn esd_zero_bins() {
        let (centers, counts) = empirical_spectral_density(&[1.0, 2.0], 0);
        assert!(centers.is_empty());
        assert!(counts.is_empty());
    }

    #[test]
    fn esd_identical_eigenvalues() {
        let eigenvalues = vec![5.0; 20];
        let (_, counts) = empirical_spectral_density(&eigenvalues, 4);
        let sum: f64 = counts.iter().sum();
        assert!(
            (sum - 1.0).abs() < tolerances::EXACT_F64,
            "ESD should sum to 1 even for identical values"
        );
    }

    #[test]
    fn level_spacing_too_few() {
        assert!((level_spacing_ratio(&[]) - 0.0).abs() < tolerances::ZERO_DETECTION);
        assert!((level_spacing_ratio(&[1.0]) - 0.0).abs() < tolerances::ZERO_DETECTION);
        assert!((level_spacing_ratio(&[1.0, 2.0]) - 0.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn level_spacing_degenerate() {
        let degen = vec![1.0; 10];
        let r = level_spacing_ratio(&degen);
        assert!(r.is_finite(), "degenerate eigenvalues should not NaN");
    }

    #[test]
    fn mp_departure_empty() {
        assert!((marchenko_pastur_departure(&[], 1.0) - 0.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn mp_departure_all_inside() {
        let (lo, hi) = marchenko_pastur_bounds(1.0);
        let eigenvalues = vec![f64::midpoint(lo, hi); 10];
        let dep = marchenko_pastur_departure(&eigenvalues, 1.0);
        assert!(
            dep.abs() < tolerances::ZERO_DETECTION,
            "all inside MP bounds → 0 departure, got {dep}"
        );
    }

    #[test]
    fn mp_departure_all_outside() {
        let eigenvalues = vec![100.0; 10];
        let dep = marchenko_pastur_departure(&eigenvalues, 1.0);
        assert!(
            (dep - 1.0).abs() < tolerances::ZERO_DETECTION,
            "all outside MP bounds → 1.0 departure, got {dep}"
        );
    }

    #[test]
    fn spectral_entropy_empty() {
        assert!((spectral_entropy(&[]) - 0.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn spectral_entropy_all_zero() {
        assert!((spectral_entropy(&[0.0; 5]) - 0.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn spectral_entropy_single_nonzero() {
        let s = spectral_entropy(&[0.0, 0.0, 5.0, 0.0]);
        assert!(
            s.abs() < tolerances::ZERO_DETECTION,
            "single nonzero → 0 entropy, got {s}"
        );
    }

    #[test]
    fn spectral_comparison_signed() {
        let mut rng = Rng::new(42);
        let w1 = random_weight_matrix(4, 4, &mut rng);
        let w2 = random_weight_matrix(4, 4, &mut rng);
        let r1 = weight_spectral_analysis(&w1, 4, 4);
        let r2 = weight_spectral_analysis(&w2, 4, 4);
        let (d_ipr, d_lsr, d_ent) = spectral_comparison(&r1, &r2);
        assert!(
            (d_ipr - (r2.mean_ipr - r1.mean_ipr)).abs() < tolerances::ZERO_DETECTION,
            "delta_ipr should be r2 - r1"
        );
        assert!(
            (d_lsr - (r2.level_spacing_ratio - r1.level_spacing_ratio)).abs()
                < tolerances::ZERO_DETECTION,
            "delta_lsr should be r2 - r1"
        );
        assert!(
            (d_ent - (r2.spectral_entropy - r1.spectral_entropy)).abs()
                < tolerances::ZERO_DETECTION,
            "delta_entropy should be r2 - r1"
        );
    }

    #[test]
    fn activation_ipr_zero_vector() {
        let zeros = vec![0.0; 8];
        let ipr_val = activation_ipr(&zeros);
        assert!(
            ipr_val.abs() < tolerances::ZERO_DETECTION,
            "zero vector → 0 IPR, got {ipr_val}"
        );
    }

    #[test]
    fn activation_ipr_localized() {
        let mut v = vec![0.0; 8];
        v[3] = 5.0;
        let ipr_val = activation_ipr(&v);
        assert!(
            (ipr_val - 1.0).abs() < tolerances::EXACT_F64,
            "single-neuron activation → IPR=1, got {ipr_val}"
        );
    }

    #[test]
    fn activation_ipr_uniform() {
        let v = vec![1.0; 8];
        let ipr_val = activation_ipr(&v);
        assert!(
            (ipr_val - 1.0 / 8.0).abs() < tolerances::EXACT_F64,
            "uniform activation → IPR=1/n, got {ipr_val}"
        );
    }

    #[test]
    fn full_analysis_identity_hamiltonian() {
        let w: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0];
        let r = weight_spectral_analysis(&w, 2, 2);
        assert_eq!(r.eigenvalues.len(), 4);
        assert!(r.mean_ipr.is_finite());
        assert!(r.level_spacing_ratio.is_finite());
        assert!(r.spectral_entropy.is_finite());
        assert!(r.mp_departure.is_finite());
    }

    #[test]
    fn mp_bounds_non_unit_aspect() {
        let (lo, hi) = marchenko_pastur_bounds(0.25);
        assert!(lo >= 0.0, "lower bound must be non-negative");
        assert!(hi > lo, "upper bound must exceed lower bound");
    }
}
