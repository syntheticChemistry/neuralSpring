// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::imprecise_flops,
    reason = "population genetics uses multi-parameter models with inherent numeric casts"
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

use crate::primitives::DIVISION_GUARD;
use crate::rng::Rng;

pub mod fst;
pub mod geography;

pub use fst::*;
pub use geography::*;

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
    let drift = fst_target / (1.0 - fst_target + DIVISION_GUARD);
    let temp_norm = (temperature - temp_min) / (temp_max - temp_min + DIVISION_GUARD);

    let mut pop_freq: Vec<f64> = ancestral_freq
        .iter()
        .map(|&p| {
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
            rng.beta(alpha, beta_param)
        })
        .collect();

    pop_freq
        .iter_mut()
        .take(n_thermal.min(n_loci))
        .for_each(|p| *p = (*p + 0.3 * (temp_norm - 0.5)).clamp(0.01, 0.99));

    let mut genotypes = vec![0.0_f64; n_individuals * n_loci];
    for (j, &p) in pop_freq.iter().enumerate() {
        for i in 0..n_individuals {
            let a1 = f64::from(u8::from(rng.next_f64() < p));
            let a2 = f64::from(u8::from(rng.next_f64() < p));
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
    let denom = 2.0 * n_individuals as f64;
    (0..n_loci)
        .map(|j| {
            let sum: f64 = (0..n_individuals).map(|i| pop[i * n_loci + j]).sum();
            sum / denom
        })
        .collect()
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

/// Pearson correlation between temperature and nucleotide diversity.
///
/// Delegates to `barracuda::stats::pearson_correlation` (absorbed from
/// airSpring/groundSpring hydrology metrics in `ToadStool` S64).
#[must_use]
pub fn thermal_diversity_correlation(pi_values: &[f64], temperatures: &[f64]) -> f64 {
    if pi_values.len() < 2 {
        return 0.0;
    }
    barracuda::stats::pearson_correlation(pi_values, temperatures).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;

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
    fn thermal_diversity_correlation_bounded() {
        let pi = vec![0.1, 0.2, 0.3, 0.4];
        let temps = vec![65.0, 72.0, 80.0, 90.0];
        let r = thermal_diversity_correlation(&pi, &temps);
        assert!(r.abs() <= 1.0 + tolerances::CROSS_LANGUAGE);
    }
}
