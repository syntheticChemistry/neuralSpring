// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spectral metrics: Hamiltonian, ESD, level spacing, Marchenko-Pastur, entropy.

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

/// Compute spectral bandwidth (λ_max − λ_min) from eigenvalues.
///
/// Delegates to `barracuda::spectral::spectral_bandwidth` (absorbed S79,
/// neuralSpring V69 handoff). Upstream handles unsorted input.
#[must_use]
pub fn spectral_bandwidth(eigenvalues: &[f64]) -> f64 {
    barracuda::spectral::spectral_bandwidth(eigenvalues)
}

/// Compute spectral condition number from eigenvalues.
///
/// Uses the ratio |λ_max| / |λ_min| over ALL eigenvalues.
/// Returns `f64::INFINITY` if any eigenvalue is effectively zero.
///
/// Delegates to `barracuda::spectral::spectral_condition_number` (absorbed S79,
/// neuralSpring V69 handoff).
#[must_use]
pub fn spectral_condition_number(eigenvalues: &[f64]) -> f64 {
    barracuda::spectral::spectral_condition_number(eigenvalues)
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
    fn mp_bounds_non_unit_aspect() {
        let (lo, hi) = marchenko_pastur_bounds(0.25);
        assert!(lo >= 0.0, "lower bound must be non-negative");
        assert!(hi > lo, "upper bound must exceed lower bound");
    }

    #[test]
    fn bandwidth_positive_for_distinct_eigenvalues() {
        let evals = vec![-2.0, -1.0, 0.0, 1.0, 3.0];
        let bw = spectral_bandwidth(&evals);
        assert!(
            (bw - 5.0).abs() < tolerances::EXACT_F64,
            "bandwidth should be 3-(-2)=5, got {bw}"
        );
    }

    #[test]
    fn bandwidth_zero_for_single_eigenvalue() {
        assert!(spectral_bandwidth(&[]).abs() < tolerances::ZERO_DETECTION);
        assert!(spectral_bandwidth(&[7.0]).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn condition_number_identity_spectrum() {
        let evals = vec![-1.0, 1.0];
        let cond = spectral_condition_number(&evals);
        assert!(
            (cond - 1.0).abs() < tolerances::EXACT_F64,
            "condition number of ±1 spectrum should be 1, got {cond}"
        );
    }

    #[test]
    fn condition_number_singular_spectrum() {
        let evals = vec![0.0, 0.0, 1.0];
        let cond = spectral_condition_number(&evals);
        assert!(
            cond.is_infinite(),
            "condition number with zero eigenvalues should be inf"
        );
    }

    #[test]
    fn condition_number_well_conditioned() {
        let evals = vec![1.0, 2.0, 3.0, 4.0];
        let cond = spectral_condition_number(&evals);
        assert!(
            (cond - 4.0).abs() < tolerances::EXACT_F64,
            "cond = max/min = 4/1 = 4, got {cond}"
        );
    }
}
