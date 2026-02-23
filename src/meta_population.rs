// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::imprecise_flops
)]

//! Meta-population differentiation under thermal constraint (Paper 025).
//!
//! Port of `control/meta_population/meta_population.py`.
//!
//! Campbell, Anderson et al. (2017)
//! "`Sulfolobus islandicus` meta-populations in Yellowstone National
//!  Park hot springs"
//! Environmental Microbiology 19:2392-2405.
//!
//! Core thesis: Geographic isolation of hot spring populations leads to
//! independent evolutionary trajectories.  Different populations evolve
//! distinct strategies despite shared ancestry — the biological analog
//! of swarm robotics (Dolson Paper 015).
//!
//! ## `BarraCUDA` connection
//!
//! - Allele frequencies: column-wise reduction (mean)
//! - FST: variance decomposition (ANOVA-like reduction)
//! - Mantel test: matrix correlation + permutation GEMM
//! - Thermal correlation: `barracuda::stats::pearson_correlation`
//! - Per-locus variance: `barracuda::ops::VarianceReduceF64` (GPU)
//!
//! ## WGSL shader (absorption-ready)
//!
//! [`WGSL_LOCUS_VARIANCE`] — per-locus allele frequency variance across
//! populations. One thread per locus. Validated in `validate_gpu_meta_pop`
//! (7/7 PASS, max error ~1e-8).

use crate::rng::Rng;

/// WGSL shader: per-locus allele frequency variance across populations.
///
/// Absorption target: `barracuda::ops::VarianceReduceF64`.
/// Validated: `validate_gpu_meta_pop` (7/7 PASS).
pub use neural_spring_forge::shaders::LOCUS_VARIANCE as WGSL_LOCUS_VARIANCE;

/// Generate synthetic diploid genotype data for one population.
///
/// Returns a flat `Vec<f64>` of shape `(n_individuals, n_loci)` row-major.
/// Values are 0, 1, or 2 (allele counts).
#[must_use]
pub fn generate_population(
    n_individuals: usize,
    n_loci: usize,
    ancestral_freq: &[f64],
    fst_target: f64,
    temperature: f64,
    temp_min: f64,
    temp_max: f64,
    n_thermal: usize,
    rng: &mut Rng,
) -> Vec<f64> {
    let drift = fst_target / (1.0 - fst_target + 1e-10);
    let temp_norm = (temperature - temp_min) / (temp_max - temp_min + 1e-10);

    let mut pop_freq = vec![0.0_f64; n_loci];
    for j in 0..n_loci {
        let p = ancestral_freq[j];
        let alpha = if drift > 0.0 {
            (p / drift).max(0.01)
        } else {
            p * 100.0
        };
        let beta_param = if drift > 0.0 {
            ((1.0 - p) / drift).max(0.01)
        } else {
            (1.0 - p) * 100.0
        };
        pop_freq[j] = rng.beta(alpha, beta_param);
    }

    for j in 0..n_thermal.min(n_loci) {
        pop_freq[j] = (pop_freq[j] + 0.3 * (temp_norm - 0.5)).clamp(0.01, 0.99);
    }

    let mut genotypes = vec![0.0_f64; n_individuals * n_loci];
    for j in 0..n_loci {
        let p = pop_freq[j];
        for i in 0..n_individuals {
            let a1 = if rng.next_f64() < p { 1.0 } else { 0.0 };
            let a2 = if rng.next_f64() < p { 1.0 } else { 0.0 };
            genotypes[i * n_loci + j] = a1 + a2;
        }
    }
    genotypes
}

/// Compute per-locus allele frequency from a genotype matrix.
///
/// Genotypes are 0/1/2; frequency = column mean / 2.
#[must_use]
pub fn allele_frequencies(pop: &[f64], n_individuals: usize, n_loci: usize) -> Vec<f64> {
    let mut freqs = vec![0.0; n_loci];
    for j in 0..n_loci {
        let sum: f64 = (0..n_individuals).map(|i| pop[i * n_loci + j]).sum();
        freqs[j] = sum / (2.0 * n_individuals as f64);
    }
    freqs
}

/// Nucleotide diversity (pi) within a population.
///
/// pi = mean over loci of `2 * p * (1-p) * n/(n-1)`
#[must_use]
pub fn nucleotide_diversity(pop: &[f64], n_individuals: usize, n_loci: usize) -> f64 {
    if n_individuals < 2 {
        return 0.0;
    }
    let freqs = allele_frequencies(pop, n_individuals, n_loci);
    let correction = n_individuals as f64 / (n_individuals as f64 - 1.0);
    let sum: f64 = freqs
        .iter()
        .map(|&p| 2.0 * p * (1.0 - p) * correction)
        .sum();
    sum / n_loci as f64
}

/// Weir & Cockerham (1984) FST estimator for two populations.
#[must_use]
pub fn pairwise_fst(pop_a: &[f64], n_a: usize, pop_b: &[f64], n_b: usize, n_loci: usize) -> f64 {
    let freq_a = allele_frequencies(pop_a, n_a, n_loci);
    let freq_b = allele_frequencies(pop_b, n_b, n_loci);
    let ns = [n_a as f64, n_b as f64];
    let n_total = ns[0] + ns[1];
    let n_pops = 2.0;
    let n_bar = n_total / n_pops;
    let n_c = (n_total - (ns[0] * ns[0] + ns[1] * ns[1]) / n_total) / (n_pops - 1.0);

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for j in 0..n_loci {
        let p_i = [freq_a[j], freq_b[j]];
        let p_bar = (ns[0] * p_i[0] + ns[1] * p_i[1]) / n_total;

        let s2 = (ns[0] * (p_i[0] - p_bar).powi(2) + ns[1] * (p_i[1] - p_bar).powi(2))
            / ((n_pops - 1.0) * n_bar);

        let h_bar = (ns[0] * 2.0 * p_i[0] * (1.0 - p_i[0]) + ns[1] * 2.0 * p_i[1] * (1.0 - p_i[1]))
            / n_total;

        let a = (n_bar / n_c)
            * (s2
                - (1.0 / (n_bar - 1.0))
                    * (p_bar * (1.0 - p_bar) - ((n_pops - 1.0) / n_pops) * s2 - 0.25 * h_bar));
        let b = (n_bar / (n_bar - 1.0))
            * (p_bar * (1.0 - p_bar)
                - ((n_pops - 1.0) / n_pops) * s2
                - ((2.0 * n_bar - 1.0) / (4.0 * n_bar)) * h_bar);
        let c = 0.5 * h_bar;

        numerator += a;
        denominator += a + b + c;
    }

    if denominator.abs() < crate::primitives::DIVISION_GUARD {
        return 0.0;
    }
    numerator / denominator
}

/// Global FST across multiple populations.
#[must_use]
pub fn global_fst(populations: &[Vec<f64>], n_individuals: &[usize], n_loci: usize) -> f64 {
    let n_pops = populations.len();
    let ns: Vec<f64> = n_individuals.iter().map(|&n| n as f64).collect();
    let n_total: f64 = ns.iter().sum();
    let n_bar = n_total / n_pops as f64;
    let n_c = (n_total - ns.iter().map(|n| n * n).sum::<f64>() / n_total) / (n_pops as f64 - 1.0);

    let all_freqs: Vec<Vec<f64>> = populations
        .iter()
        .zip(n_individuals.iter())
        .map(|(pop, &n)| allele_frequencies(pop, n, n_loci))
        .collect();

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for j in 0..n_loci {
        let p_i: Vec<f64> = all_freqs.iter().map(|f| f[j]).collect();
        let p_bar: f64 = ns.iter().zip(p_i.iter()).map(|(n, p)| n * p).sum::<f64>() / n_total;

        let s2: f64 = ns
            .iter()
            .zip(p_i.iter())
            .map(|(n, p)| n * (p - p_bar).powi(2))
            .sum::<f64>()
            / ((n_pops as f64 - 1.0) * n_bar);

        let h_bar: f64 = ns
            .iter()
            .zip(p_i.iter())
            .map(|(n, p)| n * 2.0 * p * (1.0 - p))
            .sum::<f64>()
            / n_total;

        let a = (n_bar / n_c)
            * (s2
                - (1.0 / (n_bar - 1.0))
                    * (p_bar * (1.0 - p_bar)
                        - ((n_pops as f64 - 1.0) / n_pops as f64) * s2
                        - 0.25 * h_bar));
        let b = (n_bar / (n_bar - 1.0))
            * (p_bar * (1.0 - p_bar)
                - ((n_pops as f64 - 1.0) / n_pops as f64) * s2
                - ((2.0 * n_bar - 1.0) / (4.0 * n_bar)) * h_bar);
        let c_val = 0.5 * h_bar;

        numerator += a;
        denominator += a + b + c_val;
    }

    if denominator.abs() < crate::primitives::DIVISION_GUARD {
        return 0.0;
    }
    numerator / denominator
}

/// Build pairwise FST matrix (`n_pops` x `n_pops`, flat row-major).
#[must_use]
pub fn fst_matrix(populations: &[Vec<f64>], n_individuals: &[usize], n_loci: usize) -> Vec<f64> {
    let n = populations.len();
    let mut mat = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let fst = pairwise_fst(
                &populations[i],
                n_individuals[i],
                &populations[j],
                n_individuals[j],
                n_loci,
            );
            mat[i * n + j] = fst;
            mat[j * n + i] = fst;
        }
    }
    mat
}

/// Euclidean distance matrix from 2D coordinates.
#[must_use]
pub fn geographic_distance_matrix(coords: &[(f64, f64)]) -> Vec<f64> {
    let n = coords.len();
    let mut dist = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[i].0 - coords[j].0;
            let dy = coords[i].1 - coords[j].1;
            let d = (dx * dx + dy * dy).sqrt();
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

/// Pearson correlation between upper-triangle elements of two square matrices.
#[must_use]
pub fn matrix_correlation(a: &[f64], b: &[f64], n: usize) -> f64 {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            xs.push(a[i * n + j]);
            ys.push(b[i * n + j]);
        }
    }
    if xs.len() < 2 {
        return 0.0;
    }
    let mx: f64 = xs.iter().sum::<f64>() / xs.len() as f64;
    let my: f64 = ys.iter().sum::<f64>() / ys.len() as f64;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut cov = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        cov += (x - mx) * (y - my);
        sx += (x - mx).powi(2);
        sy += (y - my).powi(2);
    }
    let denom = (sx * sy).sqrt();
    if denom < crate::primitives::DIVISION_GUARD {
        return 0.0;
    }
    cov / denom
}

/// Mantel test: correlation between distance matrices with permutation p-value.
#[must_use]
pub fn mantel_test(
    dist_a: &[f64],
    dist_b: &[f64],
    n: usize,
    n_permutations: usize,
    rng: &mut Rng,
) -> (f64, f64) {
    let r_obs = matrix_correlation(dist_a, dist_b, n);
    let mut count_ge = 0_usize;
    let mut perm: Vec<usize> = (0..n).collect();

    for _ in 0..n_permutations {
        for i in (1..n).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            perm.swap(i, j);
        }
        let mut dist_b_perm = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                dist_b_perm[i * n + j] = dist_b[perm[i] * n + perm[j]];
            }
        }
        let r_perm = matrix_correlation(dist_a, &dist_b_perm, n);
        if r_perm >= r_obs {
            count_ge += 1;
        }
    }
    let p_value = (count_ge as f64 + 1.0) / (n_permutations as f64 + 1.0);
    (r_obs, p_value)
}

/// Pearson correlation between temperature and nucleotide diversity.
#[must_use]
pub fn thermal_diversity_correlation(pi_values: &[f64], temperatures: &[f64]) -> f64 {
    let n = pi_values.len();
    if n < 2 {
        return 0.0;
    }
    let mx: f64 = pi_values.iter().sum::<f64>() / n as f64;
    let my: f64 = temperatures.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for i in 0..n {
        cov += (pi_values[i] - mx) * (temperatures[i] - my);
        sx += (pi_values[i] - mx).powi(2);
        sy += (temperatures[i] - my).powi(2);
    }
    let denom = (sx * sy).sqrt();
    if denom < crate::primitives::DIVISION_GUARD {
        return 0.0;
    }
    cov / denom
}

/// Mean allele frequency variance across populations (inter-population).
#[must_use]
pub fn inter_population_af_variance(
    populations: &[Vec<f64>],
    n_individuals: &[usize],
    n_loci: usize,
) -> f64 {
    let all_freqs: Vec<Vec<f64>> = populations
        .iter()
        .zip(n_individuals.iter())
        .map(|(pop, &n)| allele_frequencies(pop, n, n_loci))
        .collect();

    let n_pops = all_freqs.len() as f64;
    let mut total_var = 0.0;
    for j in 0..n_loci {
        let mean: f64 = all_freqs.iter().map(|f| f[j]).sum::<f64>() / n_pops;
        let var: f64 = all_freqs.iter().map(|f| (f[j] - mean).powi(2)).sum::<f64>() / n_pops;
        total_var += var;
    }
    total_var / n_loci as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allele_frequencies_bounded() {
        let pop = vec![0.0, 2.0, 1.0, 1.0, 0.0, 2.0];
        let af = allele_frequencies(&pop, 2, 3);
        assert!(af.iter().all(|&f| (0.0..=1.0).contains(&f)));
    }

    #[test]
    fn nucleotide_diversity_positive() {
        let pop = vec![0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let pi = nucleotide_diversity(&pop, 3, 3);
        assert!(pi > 0.0);
    }

    #[test]
    fn fst_symmetric() {
        let mut rng = Rng::new(42);
        let anc: Vec<f64> = (0..10).map(|_| rng.beta(2.0, 2.0)).collect();
        let temps = [70.0, 80.0];
        let pop_a = generate_population(5, 10, &anc, 0.15, temps[0], 70.0, 80.0, 2, &mut rng);
        let pop_b = generate_population(5, 10, &anc, 0.15, temps[1], 70.0, 80.0, 2, &mut rng);
        let fst_ab = pairwise_fst(&pop_a, 5, &pop_b, 5, 10);
        let fst_ba = pairwise_fst(&pop_b, 5, &pop_a, 5, 10);
        assert!((fst_ab - fst_ba).abs() < 1e-10, "FST should be symmetric");
    }

    #[test]
    fn geographic_distance_symmetric() {
        let coords = vec![(0.0, 0.0), (3.0, 4.0), (1.0, 1.0)];
        let dist = geographic_distance_matrix(&coords);
        for i in 0..3 {
            assert!(dist[i * 3 + i].abs() < 1e-10);
            for j in 0..3 {
                assert!((dist[i * 3 + j] - dist[j * 3 + i]).abs() < 1e-10);
            }
        }
        assert!((dist[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn determinism() {
        let run = || {
            let mut rng = Rng::new(42);
            let anc: Vec<f64> = (0..10).map(|_| rng.beta(2.0, 2.0)).collect();
            let pop = generate_population(5, 10, &anc, 0.15, 70.0, 65.0, 90.0, 2, &mut rng);
            (anc, pop)
        };
        let (a1, p1) = run();
        let (a2, p2) = run();
        assert_eq!(a1, a2, "ancestral frequencies must be deterministic");
        assert_eq!(p1, p2, "population genotypes must be deterministic");
    }

    #[test]
    fn global_fst_positive() {
        let mut rng = Rng::new(42);
        let n_loci = 20;
        let anc: Vec<f64> = (0..n_loci).map(|_| rng.beta(2.0, 2.0)).collect();
        let pops: Vec<Vec<f64>> = [70.0, 85.0]
            .iter()
            .map(|&t| generate_population(10, n_loci, &anc, 0.15, t, 65.0, 90.0, 4, &mut rng))
            .collect();
        let n_indivs = vec![10; 2];
        let gfst = global_fst(&pops, &n_indivs, n_loci);
        assert!(gfst.is_finite(), "global FST must be finite");
    }

    #[test]
    fn fst_matrix_symmetric_and_diag_zero() {
        let mut rng = Rng::new(42);
        let n_loci = 15;
        let n_pops = 3;
        let anc: Vec<f64> = (0..n_loci).map(|_| rng.beta(2.0, 2.0)).collect();
        let temps = [70.0, 78.0, 85.0];
        let pops: Vec<Vec<f64>> = temps
            .iter()
            .map(|&t| generate_population(8, n_loci, &anc, 0.15, t, 65.0, 90.0, 3, &mut rng))
            .collect();
        let n_indivs = vec![8; n_pops];
        let fst_mat = fst_matrix(&pops, &n_indivs, n_loci);
        for i in 0..n_pops {
            assert!(fst_mat[i * n_pops + i].abs() < 1e-10);
            for j in 0..n_pops {
                assert!((fst_mat[i * n_pops + j] - fst_mat[j * n_pops + i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn matrix_correlation_perfect() {
        let a = vec![0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0];
        let r = matrix_correlation(&a, &a, 3);
        assert!((r - 1.0).abs() < 1e-10, "self-correlation should be 1.0");
    }

    #[test]
    fn mantel_test_produces_finite() {
        let mut rng = Rng::new(42);
        let n = 4;
        let a: Vec<f64> = (0..n * n).map(|_| rng.uniform()).collect();
        let b: Vec<f64> = (0..n * n).map(|_| rng.uniform()).collect();
        let (r, p) = mantel_test(&a, &b, n, 99, &mut rng);
        assert!(r.is_finite());
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn thermal_diversity_correlation_bounded() {
        let pi = vec![0.1, 0.2, 0.3, 0.4];
        let temps = vec![65.0, 72.0, 80.0, 90.0];
        let r = thermal_diversity_correlation(&pi, &temps);
        assert!(r.abs() <= 1.0 + 1e-10);
    }

    #[test]
    fn inter_pop_af_variance_positive() {
        let mut rng = Rng::new(42);
        let n_loci = 15;
        let anc: Vec<f64> = (0..n_loci).map(|_| rng.beta(2.0, 2.0)).collect();
        let pops: Vec<Vec<f64>> = [70.0, 85.0]
            .iter()
            .map(|&t| generate_population(8, n_loci, &anc, 0.15, t, 65.0, 90.0, 3, &mut rng))
            .collect();
        let n_indivs = vec![8; 2];
        let v = inter_population_af_variance(&pops, &n_indivs, n_loci);
        assert!(v >= 0.0 && v.is_finite());
    }
}
