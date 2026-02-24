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
#[must_use]
pub fn empirical_spectral_density(eigenvalues: &[f64], n_bins: usize) -> (Vec<f64>, Vec<f64>) {
    if eigenvalues.is_empty() || n_bins == 0 {
        return (vec![], vec![]);
    }
    let min = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let max = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    let bin_width = if range < 1e-300 {
        1.0
    } else {
        range / n_bins as f64
    };

    let mut counts = vec![0.0_f64; n_bins];
    let n_total = eigenvalues.len() as f64;
    for &ev in eigenvalues {
        let idx = ((ev - min) / bin_width) as usize;
        let idx = idx.min(n_bins - 1);
        counts[idx] += 1.0 / n_total;
    }

    let centers: Vec<f64> = (0..n_bins)
        .map(|i| (i as f64 + 0.5).mul_add(bin_width, min))
        .collect();

    (centers, counts)
}

/// Level spacing ratio: r = min(s_i, s_{i+1}) / max(s_i, s_{i+1}).
///
/// For GOE (extended states): mean r ≈ 0.531.
/// For Poisson (localized states): mean r ≈ 0.386.
///
/// Input eigenvalues must be sorted in ascending order.
#[must_use]
pub fn level_spacing_ratio(eigenvalues: &[f64]) -> f64 {
    if eigenvalues.len() < 3 {
        return 0.0;
    }

    let spacings: Vec<f64> = eigenvalues.windows(2).map(|w| w[1] - w[0]).collect();

    let mut sum_r = 0.0;
    let mut count = 0usize;
    for pair in spacings.windows(2) {
        let (s1, s2) = (pair[0], pair[1]);
        if s1.abs() < 1e-300 && s2.abs() < 1e-300 {
            continue;
        }
        let r = s1.min(s2) / s1.max(s2).max(1e-300);
        sum_r += r;
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        sum_r / count as f64
    }
}

/// GOE (Gaussian Orthogonal Ensemble) expected level spacing ratio.
pub const GOE_LEVEL_SPACING: f64 = 0.530_95;

/// Poisson expected level spacing ratio (localized regime).
pub const POISSON_LEVEL_SPACING: f64 = 0.386_29;

/// Marchenko-Pastur quarter-circle law: expected spectral density for
/// random matrices with aspect ratio γ = m/n.
///
/// Returns the MP bounds (λ_min, λ_max) for singular values squared.
#[must_use]
pub fn marchenko_pastur_bounds(gamma: f64) -> (f64, f64) {
    let sq = gamma.sqrt();
    let lambda_min = (1.0 - sq).powi(2);
    let lambda_max = (1.0 + sq).powi(2);
    (lambda_min, lambda_max)
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
#[must_use]
pub fn spectral_entropy(eigenvalues: &[f64]) -> f64 {
    if eigenvalues.is_empty() {
        return 0.0;
    }
    let abs_vals: Vec<f64> = eigenvalues.iter().map(|&ev| ev.abs()).collect();
    let total: f64 = abs_vals.iter().sum();
    if total < 1e-300 {
        return 0.0;
    }
    let mut entropy = 0.0;
    for &v in &abs_vals {
        let p = v / total;
        if p > 1e-300 {
            entropy -= p * p.ln();
        }
    }
    entropy
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
    if norm_sq < 1e-300 {
        return 0.0;
    }
    let normalized: Vec<f64> = activations.iter().map(|&x| x / norm_sq.sqrt()).collect();
    ipr(&normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

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
                    (h[i * dim + j] - h[j * dim + i]).abs() < 1e-14,
                    "H not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn esd_sums_to_one() {
        let eigenvalues: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let (_, counts) = empirical_spectral_density(&eigenvalues, 10);
        let sum: f64 = counts.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "ESD should sum to 1, got {sum}");
    }

    #[test]
    fn level_spacing_bounds() {
        let sorted: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let r = level_spacing_ratio(&sorted);
        assert!(r > 0.9, "uniform spacing should have r near 1.0, got {r}");
    }

    #[test]
    fn mp_bounds_unit_aspect() {
        let (lo, hi) = marchenko_pastur_bounds(1.0);
        assert!((lo - 0.0).abs() < 1e-12);
        assert!((hi - 4.0).abs() < 1e-12);
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
            low_rank[i * 8 + 0] = 1.0;
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
        assert_eq!(r1.eigenvalues, r2.eigenvalues);
        assert_eq!(r1.mean_ipr, r2.mean_ipr);
    }
}
