// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::suboptimal_flops
)]

//! Pangenome selection dynamics (Paper 024).
//!
//! Port of `control/pangenome_selection/pangenome_selection.py`.
//!
//! Moulana, Anderson et al. (2020)
//! "Selection is a significant driver of gene gain and loss in the
//!  pangenome of the deep-sea pathogen Vibrio parahaemolyticus"
//! mSystems 5:e00673-19.
//!
//! Core thesis: Gene gain/loss in bacteria is driven by environmental
//! selection, not neutral drift.  The gene frequency spectrum deviates
//! from the U-shaped neutral expectation.
//!
//! ## `BarraCUDA` connection
//!
//! - Binary PA matrix: sparse GEMM / bitwise ops
//! - Chi-squared: map-reduce statistical test
//! - Pairwise Jaccard: `barracuda::ops::pairwise_distance` (GPU, f32)
//!
//! ## WGSL shader (absorption-ready)
//!
//! [`WGSL_PAIRWISE_JACCARD`] — pairwise Jaccard distance from PA matrix.
//! One thread per genome pair, O(G) per pair. Validated in
//! `validate_gpu_pangenome` (6/6 PASS, max error ~1e-8).
//! - Jaccard distance: pairwise GEMV

use crate::primitives;
use crate::rng::Rng;

/// WGSL shader: pairwise Jaccard distance from a pangenome PA matrix.
///
/// Absorption target: `barracuda::ops::pairwise_distance`.
/// Validated: `validate_gpu_pangenome` (6/6 PASS).
pub use neural_spring_forge::shaders::PAIRWISE_JACCARD as WGSL_PAIRWISE_JACCARD;

/// Generate a synthetic gene presence/absence matrix.
///
/// Returns a flat `Vec<f64>` of shape `(n_genes, n_genomes)` in row-major order.
/// - Core genes: present in all genomes.
/// - Singleton genes: present in exactly one genome.
/// - Accessory genes: frequency drawn from Beta, with some environment-associated.
pub fn generate_pa_matrix(
    n_genomes: usize,
    n_genes: usize,
    core_frac: f64,
    singleton_frac: f64,
    rng: &mut Rng,
    env_labels: &[usize],
) -> Vec<f64> {
    let mut pa = vec![0.0_f64; n_genes * n_genomes];

    let n_core = (n_genes as f64 * core_frac) as usize;
    let n_singleton = (n_genes as f64 * singleton_frac) as usize;
    let n_accessory = n_genes - n_core - n_singleton;

    for i in 0..n_core {
        for j in 0..n_genomes {
            pa[i * n_genomes + j] = 1.0;
        }
    }

    for i in n_core..(n_core + n_singleton) {
        let col = (rng.next_u64() as usize) % n_genomes;
        pa[i * n_genomes + col] = 1.0;
    }

    for i in (n_core + n_singleton)..n_genes {
        let gene_idx = i - n_core - n_singleton;
        if gene_idx < n_accessory / 3 {
            let freq = rng.beta(0.3, 0.3);
            for j in 0..n_genomes {
                pa[i * n_genomes + j] = if rng.next_f64() < freq { 1.0 } else { 0.0 };
            }
        } else if gene_idx < 2 * n_accessory / 3 {
            let env_type = gene_idx % 2;
            for j in 0..n_genomes {
                let prob = if env_labels[j] == env_type { 0.8 } else { 0.15 };
                pa[i * n_genomes + j] = if rng.next_f64() < prob { 1.0 } else { 0.0 };
            }
        } else {
            let freq = rng.beta(2.0, 5.0);
            for j in 0..n_genomes {
                pa[i * n_genomes + j] = if rng.next_f64() < freq { 1.0 } else { 0.0 };
            }
        }
    }

    pa
}

/// Compute per-gene frequency (fraction of genomes containing each gene).
#[must_use]
pub fn gene_frequencies(pa: &[f64], n_genes: usize, n_genomes: usize) -> Vec<f64> {
    let mut freqs = vec![0.0; n_genes];
    for i in 0..n_genes {
        let sum: f64 = (0..n_genomes).map(|j| pa[i * n_genomes + j]).sum();
        freqs[i] = sum / n_genomes as f64;
    }
    freqs
}

/// Partition pangenome into (core, accessory, singleton) counts.
#[must_use]
pub fn partition_pangenome(
    freqs: &[f64],
    n_genomes: usize,
    core_threshold: f64,
) -> (usize, usize, usize) {
    let singleton_freq = 1.0 / n_genomes as f64;
    let n_core = freqs.iter().filter(|&&f| f >= core_threshold).count();
    let n_singleton = freqs
        .iter()
        .filter(|&&f| (f - singleton_freq).abs() < 1e-10)
        .count();
    let n_accessory = freqs.len() - n_core - n_singleton;
    (n_core, n_accessory, n_singleton)
}

/// Histogram of gene frequencies (excluding core and absent).
#[must_use]
pub fn frequency_spectrum(freqs: &[f64], n_bins: usize) -> Vec<f64> {
    let mut counts = vec![0.0_f64; n_bins];
    for &f in freqs {
        if f > 0.0 && f < 1.0 {
            let bin = ((f * n_bins as f64) as usize).min(n_bins - 1);
            counts[bin] += 1.0;
        }
    }
    counts
}

/// Neutral U-shaped frequency spectrum (Wright-Fisher 1/f(1-f)).
#[must_use]
pub fn neutral_spectrum(n_bins: usize) -> Vec<f64> {
    let mut spec = vec![0.0_f64; n_bins];
    for i in 0..n_bins {
        let c = (i as f64 + 0.5) / n_bins as f64;
        if c > 0.01 && c < 0.99 {
            spec[i] = 1.0 / (c * (1.0 - c));
        }
    }
    let total: f64 = spec.iter().sum();
    if total > 0.0 {
        for v in &mut spec {
            *v /= total;
        }
    }
    spec
}

/// Chi-squared statistic comparing observed spectrum to expected fractions.
#[must_use]
pub fn spectrum_chi_squared(observed: &[f64], expected_frac: &[f64]) -> f64 {
    let total: f64 = observed.iter().sum();
    if total == 0.0 {
        return 0.0;
    }
    let mut chi2 = 0.0;
    for (&o, &e_frac) in observed.iter().zip(expected_frac.iter()) {
        let e = e_frac * total;
        if e > 0.5 {
            chi2 += (o - e).powi(2) / e;
        }
    }
    chi2
}

/// Per-gene chi-squared test for environmental association (2x2 contingency).
#[must_use]
pub fn env_association_chi2(
    pa: &[f64],
    n_genes: usize,
    n_genomes: usize,
    env_labels: &[usize],
) -> Vec<f64> {
    let mut chi2_vals = vec![0.0_f64; n_genes];
    let n0: f64 = env_labels.iter().filter(|&&e| e == 0).count() as f64;
    let n1: f64 = n_genomes as f64 - n0;

    if n0 == 0.0 || n1 == 0.0 {
        return chi2_vals;
    }

    let n = n_genomes as f64;
    for i in 0..n_genes {
        let mut a = 0.0_f64;
        let mut b = 0.0_f64;
        for j in 0..n_genomes {
            if env_labels[j] == 0 {
                a += pa[i * n_genomes + j];
            } else {
                b += pa[i * n_genomes + j];
            }
        }
        let c = n0 - a;
        let d = n1 - b;

        let expected = [
            (a + b) * (a + c) / n,
            (a + b) * (b + d) / n,
            (c + d) * (a + c) / n,
            (c + d) * (b + d) / n,
        ];
        let obs = [a, b, c, d];
        for (&o, &e) in obs.iter().zip(expected.iter()) {
            if e > 0.5 {
                chi2_vals[i] += (o - e).powi(2) / e;
            }
        }
    }
    chi2_vals
}

/// Selection coefficient: L2 deviation of normalized spectrum from neutral.
#[must_use]
pub fn selection_coefficient(observed: &[f64], neutral: &[f64]) -> f64 {
    let total: f64 = observed.iter().sum();
    if total == 0.0 {
        return 0.0;
    }
    let sum_sq: f64 = observed
        .iter()
        .zip(neutral.iter())
        .map(|(&o, &n)| (o / total - n).powi(2))
        .sum();
    sum_sq.sqrt()
}

/// Shannon diversity of gene repertoire sizes across genomes.
#[must_use]
pub fn gene_repertoire_diversity(pa: &[f64], n_genes: usize, n_genomes: usize) -> f64 {
    let mut sizes = vec![0_usize; n_genomes];
    for j in 0..n_genomes {
        let mut sum = 0;
        for i in 0..n_genes {
            if pa[i * n_genomes + j] > 0.5 {
                sum += 1;
            }
        }
        sizes[j] = sum;
    }

    let mut counts = std::collections::HashMap::new();
    for &s in &sizes {
        *counts.entry(s).or_insert(0_usize) += 1;
    }

    let total = n_genomes as f64;
    let freqs: Vec<f64> = counts.values().map(|&c| c as f64 / total).collect();
    primitives::shannon_entropy(&freqs)
}

/// Pairwise Jaccard distance between genomes (columns of PA matrix).
#[must_use]
pub fn jaccard_distance_matrix(pa: &[f64], n_genes: usize, n_genomes: usize) -> Vec<f64> {
    let mut dist = vec![0.0_f64; n_genomes * n_genomes];
    for i in 0..n_genomes {
        for j in (i + 1)..n_genomes {
            let mut intersection = 0.0_f64;
            let mut union = 0.0_f64;
            for g in 0..n_genes {
                let a = pa[g * n_genomes + i];
                let b = pa[g * n_genomes + j];
                intersection += a * b;
                union += a.max(b);
            }
            let d = if union > 0.0 {
                1.0 - intersection / union
            } else {
                0.0
            };
            dist[i * n_genomes + j] = d;
            dist[j * n_genomes + i] = d;
        }
    }
    dist
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::tolerances;

    fn test_rng() -> Rng {
        Rng::new(42)
    }

    #[test]
    fn pa_matrix_is_binary() {
        let mut rng = test_rng();
        let env = vec![0, 0, 0, 1, 1, 1];
        let pa = generate_pa_matrix(6, 20, 0.25, 0.1, &mut rng, &env);
        assert!(pa.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn partition_sums_to_total() {
        let freqs = vec![1.0, 1.0, 0.5, 0.5, 0.1];
        let (c, a, s) = partition_pangenome(&freqs, 10, 0.95);
        assert_eq!(c + a + s, 5);
    }

    #[test]
    fn neutral_spectrum_normalized() {
        let spec = neutral_spectrum(10);
        let sum: f64 = spec.iter().sum();
        assert!(
            (sum - 1.0).abs() < tolerances::CROSS_LANGUAGE,
            "neutral spectrum should sum to 1"
        );
    }

    #[test]
    fn chi_squared_zero_for_matching() {
        let obs = vec![10.0, 10.0, 10.0, 10.0];
        let exp = vec![0.25, 0.25, 0.25, 0.25];
        let chi2 = spectrum_chi_squared(&obs, &exp);
        assert!(chi2 < tolerances::CROSS_LANGUAGE);
    }

    #[test]
    fn jaccard_symmetric_and_bounded() {
        let pa = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let dist = jaccard_distance_matrix(&pa, 2, 3);
        for i in 0..3 {
            assert!((dist[i * 3 + i]).abs() < tolerances::CROSS_LANGUAGE);
            for j in 0..3 {
                assert!((dist[i * 3 + j] - dist[j * 3 + i]).abs() < tolerances::CROSS_LANGUAGE);
                assert!(dist[i * 3 + j] >= 0.0 && dist[i * 3 + j] <= 1.0);
            }
        }
    }

    #[test]
    fn determinism() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        let env = vec![0, 0, 1, 1];
        let pa1 = generate_pa_matrix(4, 10, 0.3, 0.1, &mut r1, &env);
        let pa2 = generate_pa_matrix(4, 10, 0.3, 0.1, &mut r2, &env);
        assert_eq!(pa1, pa2);
    }

    #[test]
    fn gene_frequencies_bounded() {
        let mut rng = Rng::new(42);
        let env = vec![0, 0, 1, 1];
        let pa = generate_pa_matrix(4, 20, 0.3, 0.1, &mut rng, &env);
        let freqs = gene_frequencies(&pa, 20, 4);
        assert_eq!(freqs.len(), 20);
        assert!(freqs.iter().all(|&f| (0.0..=1.0).contains(&f)));
    }

    #[test]
    fn frequency_spectrum_sums_correctly() {
        let freqs = vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.95];
        let spec = frequency_spectrum(&freqs, 5);
        assert_eq!(spec.len(), 5);
        let total: f64 = spec.iter().sum();
        let expected = freqs.len() as f64;
        assert!(
            (total - expected).abs() < tolerances::ZERO_DETECTION,
            "frequency spectrum total {total} != expected {expected}"
        );
    }

    #[test]
    fn env_association_chi2_nonneg() {
        let mut rng = Rng::new(42);
        let env = vec![0, 0, 1, 1];
        let pa = generate_pa_matrix(4, 10, 0.3, 0.1, &mut rng, &env);
        let chi2 = env_association_chi2(&pa, 10, 4, &env);
        assert_eq!(chi2.len(), 10);
        assert!(chi2.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn selection_coefficient_nonneg() {
        let obs = vec![5.0, 10.0, 15.0, 12.0, 8.0];
        let neu = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        let s = selection_coefficient(&obs, &neu);
        assert!(s >= 0.0 && s.is_finite());
    }

    #[test]
    fn gene_repertoire_diversity_positive() {
        let mut rng = Rng::new(42);
        let env = vec![0, 0, 1, 1];
        let pa = generate_pa_matrix(4, 20, 0.3, 0.1, &mut rng, &env);
        let d = gene_repertoire_diversity(&pa, 20, 4);
        assert!(d >= 0.0 && d.is_finite());
    }
}
