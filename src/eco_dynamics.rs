// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop
)]

//! Ecological dynamics in evolutionary computation.
//!
//! Port of `control/eco_dynamics/eco_dynamics.py`.
//!
//! Reproduces key insights from:
//! Dolson & Ofria (2018) "Ecological Theory Provides Insights about
//! Evolutionary Computation" GECCO '18 Companion, pp 105-106.
//!
//! Core thesis: EA populations behave like ecological communities —
//! competitive exclusion, niche partitioning, frequency-dependent selection.
//!
//! ## `BarraCUDA` connection
//!
//! - Gaussian kernel niche fitness: `barracuda::ops::pairwise_distance` + elementwise exp
//! - Batch fitness evaluation: `barracuda::ops::batch_gemm` (population × niche matrix)
//! - Species diversity metrics: `barracuda::stats::variance` + entropy reduction
//! - Hamming distance for genotype comparison: `barracuda::ops::pairwise_distance`

use crate::primitives;
use crate::rng::Rng;
use std::collections::{HashMap, HashSet};

/// Multi-niche fitness landscape with Gaussian kernel niches.
///
/// Each niche rewards a different genotype pattern. Fitness = max over
/// niches, optionally penalized by crowding (frequency-dependent).
#[derive(Debug, Clone)]
pub struct MultiNicheLandscape {
    n_loci: usize,
    niche_optima: Vec<Vec<u8>>,
    niche_capacity: Vec<f64>,
    niche_width: Vec<f64>,
}

impl MultiNicheLandscape {
    /// Create landscape with random binary niche optima.
    #[must_use]
    pub fn new(n_loci: usize, n_niches: usize, niche_width: f64, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let niche_optima = (0..n_niches)
            .map(|_| (0..n_loci).map(|_| rng.usize(2) as u8).collect())
            .collect();
        let niche_capacity = vec![1.0; n_niches];
        let niche_width_vec = vec![niche_width; n_niches];
        Self {
            n_loci,
            niche_optima,
            niche_capacity,
            niche_width: niche_width_vec,
        }
    }

    /// Vectorized fitness for the entire population.
    #[must_use]
    pub fn batch_fitness(&self, population: &[Vec<u8>], frequency_dependent: bool) -> Vec<f64> {
        let n_niches = self.niche_optima.len();
        let n_pop = population.len();

        let mut dists: Vec<Vec<f64>> = vec![vec![0.0; n_niches]; n_pop];
        for (i, ind) in population.iter().enumerate() {
            for (j, optimum) in self.niche_optima.iter().enumerate() {
                let hamming = ind
                    .iter()
                    .zip(optimum.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                dists[i][j] = hamming as f64 / self.n_loci as f64;
            }
        }

        let mut niche_fits: Vec<Vec<f64>> = vec![vec![0.0; n_niches]; n_pop];
        for i in 0..n_pop {
            for j in 0..n_niches {
                let d = dists[i][j];
                let w = self.niche_width[j];
                let fit = self.niche_capacity[j] * (-(d * d) / (2.0 * w * w)).exp();
                niche_fits[i][j] = fit;
            }
        }

        if frequency_dependent {
            let mut occupancy = vec![0.0; n_niches];
            for i in 0..n_pop {
                for j in 0..n_niches {
                    if dists[i][j] < 0.25 {
                        occupancy[j] += 1.0;
                    }
                }
            }
            let crowding: Vec<f64> = occupancy
                .iter()
                .map(|&o| 1.0 / 0.05f64.mul_add(o, 1.0))
                .collect();
            for i in 0..n_pop {
                for j in 0..n_niches {
                    niche_fits[i][j] *= crowding[j];
                }
            }
        }

        niche_fits
            .into_iter()
            .map(|row| row.into_iter().fold(0.0f64, f64::max))
            .collect()
    }
}

/// Result of running the evolutionary algorithm.
#[derive(Debug, Clone)]
pub struct EaResult {
    pub diversity: Vec<f64>,
    pub richness: Vec<usize>,
    pub dominance: Vec<f64>,
    pub mean_fitness: Vec<f64>,
}

/// Run tournament-selection EA with ecological metric tracking.
#[must_use]
pub fn run_ea(
    landscape: &MultiNicheLandscape,
    pop_size: usize,
    n_generations: usize,
    mutation_rate: f64,
    frequency_dependent: bool,
    tournament_size: usize,
    seed: u64,
) -> EaResult {
    let mut rng = Rng::new(seed);
    let n_loci = landscape.n_loci;

    let mut population: Vec<Vec<u8>> = (0..pop_size)
        .map(|_| (0..n_loci).map(|_| rng.usize(2) as u8).collect())
        .collect();

    let mut diversity_trace = Vec::with_capacity(n_generations);
    let mut richness_trace = Vec::with_capacity(n_generations);
    let mut dominance_trace = Vec::with_capacity(n_generations);
    let mut mean_fitness_trace = Vec::with_capacity(n_generations);

    for _ in 0..n_generations {
        let mut fitnesses = landscape.batch_fitness(&population, frequency_dependent);
        for f in &mut fitnesses {
            *f = (*f).max(1e-10);
        }

        diversity_trace.push(shannon_diversity(&population));
        richness_trace.push(genotype_richness(&population));
        dominance_trace.push(dominance_index(&population));
        mean_fitness_trace.push(fitnesses.iter().sum::<f64>() / fitnesses.len() as f64);

        let mut children = vec![vec![0u8; n_loci]; pop_size];
        for i in 0..pop_size {
            let chosen = rng.choose_distinct(pop_size, tournament_size);
            let mut best_idx = chosen[0];
            for &idx in &chosen[1..] {
                if fitnesses[idx] > fitnesses[best_idx] {
                    best_idx = idx;
                }
            }
            children[i].clone_from(&population[best_idx]);
        }

        for i in 0..pop_size {
            let mask = rng.bernoulli_mask(n_loci, mutation_rate);
            for (j, &flip) in mask.iter().enumerate() {
                if flip {
                    children[i][j] = 1 - children[i][j];
                }
            }
        }

        population = children;
    }

    EaResult {
        diversity: diversity_trace,
        richness: richness_trace,
        dominance: dominance_trace,
        mean_fitness: mean_fitness_trace,
    }
}

/// Shannon diversity (equitability) of genotype frequency distribution.
///
/// `H/H_max` where H = -sum(p*ln(p)), `H_max` = ln(S).
/// Uses `&[u8]` references as `HashMap` keys to avoid cloning genotypes.
#[must_use]
pub fn shannon_diversity(population: &[Vec<u8>]) -> f64 {
    let mut counts: HashMap<&[u8], usize> = HashMap::new();
    for g in population {
        *counts.entry(g.as_slice()).or_insert(0) += 1;
    }
    let total = population.len() as f64;
    let freqs: Vec<f64> = counts.values().map(|&c| c as f64 / total).collect();
    primitives::shannon_equitability(&freqs)
}

/// Number of unique genotypes in the population.
#[must_use]
pub fn genotype_richness(population: &[Vec<u8>]) -> usize {
    let mut seen: HashSet<&[u8]> = HashSet::new();
    for g in population {
        seen.insert(g.as_slice());
    }
    seen.len()
}

/// Berger-Parker dominance: frequency of the most common genotype.
#[must_use]
pub fn dominance_index(population: &[Vec<u8>]) -> f64 {
    let mut counts: HashMap<&[u8], usize> = HashMap::new();
    for g in population {
        *counts.entry(g.as_slice()).or_insert(0) += 1;
    }
    let total = population.len();
    if total == 0 {
        return 0.0;
    }
    counts.values().copied().max().unwrap_or(0) as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn single_niche_fitness_nonnegative() {
        let landscape = MultiNicheLandscape::new(20, 1, 0.12, 42);
        let pop: Vec<Vec<u8>> = (0..50).map(|_| vec![0; 20]).collect();
        let fits = landscape.batch_fitness(&pop, false);
        assert!(fits.iter().all(|&f| f >= 0.0));
    }

    #[test]
    fn ea_mean_fitness_increases() {
        let landscape = MultiNicheLandscape::new(20, 1, 0.12, 42);
        let result = run_ea(&landscape, 200, 300, 0.008, false, 5, 42);
        let early: f64 = result.mean_fitness[..20].iter().sum::<f64>() / 20.0;
        let late: f64 = result.mean_fitness[280..].iter().sum::<f64>() / 20.0;
        assert!(
            late >= early,
            "fitness should increase: early={early}, late={late}"
        );
    }

    #[test]
    fn shannon_diversity_in_range() {
        let pop = vec![vec![0, 0, 1], vec![0, 1, 0], vec![1, 0, 0]];
        let h = shannon_diversity(&pop);
        assert!((0.0..=1.0).contains(&h));
    }

    #[test]
    fn dominance_in_range() {
        let pop = vec![vec![0, 0], vec![0, 0], vec![1, 1]];
        let d = dominance_index(&pop);
        assert!(d > 0.0 && d <= 1.0);
    }

    #[test]
    fn richness_bounded_by_pop_size() {
        let landscape = MultiNicheLandscape::new(10, 2, 0.15, 42);
        let result = run_ea(&landscape, 100, 50, 0.01, false, 5, 42);
        assert!(result.richness.iter().all(|&r| r <= 100));
    }

    #[test]
    fn determinism() {
        let landscape = MultiNicheLandscape::new(20, 4, 0.12, 42);
        let r1 = run_ea(&landscape, 100, 50, 0.008, false, 5, 42);
        let r2 = run_ea(&landscape, 100, 50, 0.008, false, 5, 42);
        assert_relative_eq!(
            r1.mean_fitness[r1.mean_fitness.len() - 1],
            r2.mean_fitness[r2.mean_fitness.len() - 1],
            epsilon = 1e-10
        );
    }
}
