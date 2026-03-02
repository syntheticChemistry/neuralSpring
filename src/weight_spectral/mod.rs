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

pub mod metrics;
pub mod phase;

pub use metrics::*;
pub use phase::*;

use crate::anderson_localization::{ipr, mean_ipr};
use crate::eigh::eigh_householder_qr;
use crate::primitives::LOG_GUARD;

/// Result of weight matrix spectral analysis.
///
/// Core fields (IPR, level spacing, entropy, MP departure) are original to
/// neuralSpring baseCamp nS-01. Extended fields (bandwidth, condition number,
/// phase label) evolved from hotSpring's `proxy.rs` Anderson 3D diagnostics
/// via cross-spring evolution through ToadStool.
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

    // ── Cross-spring evolution: hotSpring proxy.rs diagnostics ───────
    /// Spectral bandwidth: λ_max − λ_min. Larger bandwidth indicates
    /// wider energy spread (more complex learned representation).
    ///
    /// Evolved from hotSpring `ProxyFeatures::bandwidth` (Anderson 3D).
    pub bandwidth: f64,
    /// Spectral condition number: |λ_max| / |λ_min| (non-zero eigenvalues).
    /// High condition number signals ill-conditioning; predicts training
    /// difficulty and CG solver convergence rate.
    ///
    /// Evolved from hotSpring `ProxyFeatures::lambda_min` and
    /// barracuda `SvdDecomposition::condition_number()`.
    pub condition_number: f64,
    /// Discrete spectral phase label derived from level spacing ratio.
    /// - `Extended`: ⟨r⟩ ≥ 0.48 (GOE-like, delocalized, good generalization)
    /// - `Critical`: 0.42 ≤ ⟨r⟩ < 0.48 (Anderson transition)
    /// - `Localized`: ⟨r⟩ < 0.42 (Poisson-like, memorization risk)
    ///
    /// Evolved from hotSpring `ProxyFeatures::phase` (Anderson 3D).
    pub phase: SpectralPhase,
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
    let mut decomp = eigh_householder_qr(&h, dim);

    decomp
        .eigenvalues
        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let ipr_val = mean_ipr(&decomp.eigenvectors, dim);
    let lsr = level_spacing_ratio(&decomp.eigenvalues);
    let entropy = spectral_entropy(&decomp.eigenvalues);
    let gamma = m as f64 / n.max(1) as f64;
    let mp_departure = marchenko_pastur_departure(&decomp.eigenvalues, gamma);
    let bw = spectral_bandwidth(&decomp.eigenvalues);
    let cond = spectral_condition_number(&decomp.eigenvalues);
    let phase = classify_phase(lsr);

    WeightSpectralResult {
        eigenvalues: decomp.eigenvalues,
        mean_ipr: ipr_val,
        level_spacing_ratio: lsr,
        spectral_entropy: entropy,
        mp_departure,
        bandwidth: bw,
        condition_number: cond,
        phase,
    }
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
    let bw = spectral_bandwidth(&eigenvalues);
    let cond = spectral_condition_number(&eigenvalues);
    let phase = classify_phase(lsr);
    WeightSpectralResult {
        eigenvalues,
        mean_ipr: ipr_val,
        level_spacing_ratio: lsr,
        spectral_entropy: entropy,
        mp_departure,
        bandwidth: bw,
        condition_number: cond,
        phase,
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
    fn full_analysis_populates_cross_spring_fields() {
        let mut rng = Rng::new(42);
        let w = random_weight_matrix(8, 8, &mut rng);
        let r = weight_spectral_analysis(&w, 8, 8);

        assert!(
            r.bandwidth > 0.0,
            "bandwidth should be positive for random matrix"
        );
        assert!(
            r.condition_number > 1.0,
            "condition number should exceed 1 for random matrix"
        );
        assert!(
            r.phase == SpectralPhase::Extended || r.phase == SpectralPhase::Critical,
            "random matrix should be extended or critical, got {}",
            r.phase
        );
    }

    #[test]
    fn decomposition_path_matches_direct() {
        let mut rng = Rng::new(42);
        let w = random_weight_matrix(6, 6, &mut rng);
        let direct = weight_spectral_analysis(&w, 6, 6);

        let h = weight_to_hamiltonian(&w, 6, 6);
        let decomp = crate::eigh::eigh_householder_qr(&h, 12);
        let via_decomp =
            spectral_result_from_decomposition(decomp.eigenvalues, &decomp.eigenvectors, 12, 1.0);

        assert!(
            (direct.bandwidth - via_decomp.bandwidth).abs() < tolerances::EXACT_F64,
            "bandwidth mismatch: direct={} decomp={}",
            direct.bandwidth,
            via_decomp.bandwidth
        );
        assert_eq!(
            direct.phase, via_decomp.phase,
            "phase mismatch: direct={} decomp={}",
            direct.phase, via_decomp.phase
        );
    }
}
