// SPDX-License-Identifier: AGPL-3.0-or-later

//! Directed evolution via selection algorithms.
//!
//! Port of `control/directed_evolution/directed_evolution.py`.
//!
//! Reproduces key results from:
//! Dolson, Banzhaf, Ofria (2022)
//! "Artificial selection methods from evolutionary computing show promise
//!  for directed evolution of microbes"
//! eLife 11:e79665. doi:10.7554/eLife.79665
//!
//! Core thesis: computational selection algorithms (tournament, lexicase,
//! down-sampled lexicase) outperform random and truncation selection for
//! multi-objective optimization in directed evolution.
//!
//! ## GPU-ready layout
//!
//! Population and fitnesses use **flat row-major `Vec<f64>`**:
//! - Population: `pop[individual_i]` at `i * genome_len .. (i+1) * genome_len`
//! - Fitnesses: `fitnesses[individual_i]` at `i * n_objectives .. (i+1) * n_objectives`
//!
//! Maps to GPU buffers for `barracuda::ops::batch_gemm`.
//!
//! ## `BarraCUDA` connection
//!
//! - Multi-objective fitness: `barracuda::stats::variance` (per-chunk statistics)
//! - Tournament selection: `barracuda::ops::batch_gemm` (fitness comparison)
//! - Lexicase selection: sequential per-case filtering (not GPU-friendly)
//! - Population evolution: `barracuda::ops::batch_gemm` (genotype × weight)

/// WGSL shader: multi-objective fitness evaluation (mean + 0.1×std per chunk).
///
/// One thread per (individual, objective) pair. Flat row-major genotype buffer.
/// Paper 014 (Directed Evolution multi-objective).
///
/// Absorption target: `barracuda::ops::batch_gemm`.
/// Validated: `validate_gpu_directed` (6/6 PASS).
#[cfg(feature = "barracuda")]
pub use neural_spring_forge::shaders::MULTI_OBJ_FITNESS as WGSL_MULTI_OBJ_FITNESS;

use crate::rng::Rng;
use crate::tolerances;

const EPSILON: f64 = tolerances::LEXICASE_EPSILON;

/// Compute fitness on multiple objectives.
///
/// Each objective rewards a different portion of the genome.
/// `fitness[i] = mean(chunk_i) + 0.1 * std(chunk_i)`.
/// Last chunk gets remainder of loci.
#[expect(
    clippy::cast_precision_loss,
    reason = "genome chunk sizes → f64 for mean/std computation"
)]
#[must_use]
pub fn multi_objective_fitness(genotype: &[f64], n_objectives: usize) -> Vec<f64> {
    let n = genotype.len();
    let chunk = n / n_objectives;
    let mut fitnesses = Vec::with_capacity(n_objectives);
    for i in 0..n_objectives {
        let start = i * chunk;
        let end = if i < n_objectives - 1 {
            start + chunk
        } else {
            n
        };
        let segment = &genotype[start..end];
        let mean: f64 = segment.iter().sum::<f64>() / segment.len() as f64;
        let variance: f64 =
            segment.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / segment.len() as f64;
        let std = variance.sqrt();
        fitnesses.push(0.1f64.mul_add(std, mean));
    }
    fitnesses
}

/// Random selection: no fitness pressure.
///
/// Population and fitnesses are flat row-major: `pop_size × genome_len`, `pop_size × n_objectives`.
#[must_use]
pub fn random_selection(
    population: &[f64],
    _fitnesses: &[f64],
    pop_size: usize,
    genome_len: usize,
    _n_objectives: usize,
    n_select: usize,
    rng: &mut Rng,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_select * genome_len);
    for _ in 0..n_select {
        let idx = rng.usize(pop_size);
        out.extend_from_slice(&population[idx * genome_len..(idx + 1) * genome_len]);
    }
    out
}

/// Truncation: select top fraction by aggregate fitness.
#[must_use]
pub fn truncation_selection(
    population: &[f64],
    fitnesses: &[f64],
    pop_size: usize,
    genome_len: usize,
    n_objectives: usize,
    n_select: usize,
    rng: &mut Rng,
) -> Vec<f64> {
    let mut agg: Vec<(usize, f64)> = (0..pop_size)
        .map(|i| {
            (
                i,
                fitnesses[i * n_objectives..(i + 1) * n_objectives]
                    .iter()
                    .sum(),
            )
        })
        .collect();
    agg.sort_by(|a, b| a.1.total_cmp(&b.1));
    let top_k = (n_select / 4).max(2).min(agg.len());
    let best_idx: Vec<usize> = agg.iter().rev().take(top_k).map(|x| x.0).collect();
    let mut out = Vec::with_capacity(n_select * genome_len);
    for _ in 0..n_select {
        let idx = best_idx[rng.usize(best_idx.len())];
        out.extend_from_slice(&population[idx * genome_len..(idx + 1) * genome_len]);
    }
    out
}

/// Tournament selection: aggregate fitness comparison.
#[must_use]
pub fn tournament_selection(
    population: &[f64],
    fitnesses: &[f64],
    pop_size: usize,
    genome_len: usize,
    n_objectives: usize,
    n_select: usize,
    rng: &mut Rng,
) -> Vec<f64> {
    let agg: Vec<f64> = (0..pop_size)
        .map(|i| {
            fitnesses[i * n_objectives..(i + 1) * n_objectives]
                .iter()
                .sum()
        })
        .collect();
    let tournament_size = 5usize.min(pop_size);
    let mut out = Vec::with_capacity(n_select * genome_len);
    for _ in 0..n_select {
        let contestants = rng.choose_distinct(pop_size, tournament_size);
        let winner = contestants
            .iter()
            .max_by(|a, b| f64::total_cmp(&agg[**a], &agg[**b]))
            .copied()
            .unwrap_or(0);
        out.extend_from_slice(&population[winner * genome_len..(winner + 1) * genome_len]);
    }
    out
}

/// Lexicase selection: filter by shuffled per-case fitness.
#[must_use]
pub fn lexicase_selection(
    population: &[f64],
    fitnesses: &[f64],
    pop_size: usize,
    genome_len: usize,
    n_objectives: usize,
    n_select: usize,
    rng: &mut Rng,
) -> Vec<f64> {
    let mut selected = Vec::with_capacity(n_select * genome_len);
    for _ in 0..n_select {
        let mut candidates: Vec<usize> = (0..pop_size).collect();
        let obj_order = rng.permutation(n_objectives);
        for obj in obj_order {
            if candidates.len() <= 1 {
                break;
            }
            let best = candidates
                .iter()
                .map(|&i| fitnesses[i * n_objectives + obj])
                .fold(f64::NEG_INFINITY, f64::max);
            candidates.retain(|&i| fitnesses[i * n_objectives + obj] >= best - EPSILON);
        }
        let winner = candidates[rng.usize(candidates.len())];
        selected.extend_from_slice(&population[winner * genome_len..(winner + 1) * genome_len]);
    }
    selected
}

/// Down-sampled lexicase: use random subset (50%) of objectives.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "population indices and objective subset sizing via f64 → usize"
)]
pub fn downsampled_lexicase_selection(
    population: &[f64],
    fitnesses: &[f64],
    pop_size: usize,
    genome_len: usize,
    n_objectives: usize,
    n_select: usize,
    rng: &mut Rng,
) -> Vec<f64> {
    let n_sub = (n_objectives as f64 * 0.5).ceil() as usize;
    let n_sub = n_sub.max(2).min(n_objectives);
    let mut selected = Vec::with_capacity(n_select * genome_len);
    for _ in 0..n_select {
        let mut candidates: Vec<usize> = (0..pop_size).collect();
        let obj_order = rng.choose_distinct(n_objectives, n_sub);
        for obj in obj_order {
            if candidates.len() <= 1 {
                break;
            }
            let best = candidates
                .iter()
                .map(|&i| fitnesses[i * n_objectives + obj])
                .fold(f64::NEG_INFINITY, f64::max);
            candidates.retain(|&i| fitnesses[i * n_objectives + obj] >= best - EPSILON);
        }
        let winner = candidates[rng.usize(candidates.len())];
        selected.extend_from_slice(&population[winner * genome_len..(winner + 1) * genome_len]);
    }
    selected
}

/// Count Pareto-optimal individuals.
///
/// Individual i is dominated if exists j where all `fitnesses[j] >= fitnesses[i]`
/// and at least one strict. Fitnesses flat row-major: `n × n_objectives`.
#[must_use]
pub fn pareto_front_count(fitnesses: &[f64], n: usize, n_objectives: usize) -> usize {
    let mut is_pareto = vec![true; n];
    for i in 0..n {
        if !is_pareto[i] {
            continue;
        }
        for j in 0..n {
            if i == j || !is_pareto[j] {
                continue;
            }
            let base_i = i * n_objectives;
            let base_j = j * n_objectives;
            let all_ge = (0..n_objectives).all(|k| fitnesses[base_j + k] >= fitnesses[base_i + k]);
            let any_strict =
                (0..n_objectives).any(|k| fitnesses[base_j + k] > fitnesses[base_i + k]);
            if all_ge && any_strict {
                is_pareto[i] = false;
                break;
            }
        }
    }
    is_pareto.iter().filter(|&&x| x).count()
}

/// Result of a selection experiment run.
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    /// Mean pairwise phenotype distance across sampled individuals per generation.
    pub diversity: Vec<f64>,
    /// Pareto-optimal individual count per generation.
    pub pareto_front: Vec<usize>,
    /// Mean per-individual aggregate fitness per generation.
    pub mean_fitness: Vec<f64>,
}

#[expect(
    clippy::cast_precision_loss,
    reason = "pairwise distance count → f64 for mean"
)]
fn phenotype_diversity(fitnesses: &[f64], n: usize, n_objectives: usize, rng: &mut Rng) -> f64 {
    let sample_size = 50.min(n);
    if sample_size < 2 {
        return 0.0;
    }
    let idx = rng.choose_distinct(n, sample_size);
    let mut dists = Vec::with_capacity(sample_size * (sample_size - 1) / 2);
    for (ii, &i) in idx.iter().enumerate() {
        for &j in idx.iter().skip(ii + 1) {
            let base_i = i * n_objectives;
            let base_j = j * n_objectives;
            let d: f64 = (0..n_objectives)
                .map(|k| (fitnesses[base_i + k] - fitnesses[base_j + k]).powi(2))
                .sum::<f64>()
                .sqrt();
            dists.push(d);
        }
    }
    dists.iter().sum::<f64>() / dists.len() as f64
}

/// Run EA with a given selection algorithm, track multi-objective metrics.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "population metrics use usize→f64 for averaging"
)]
pub fn run_selection_experiment<F>(
    selection_fn: F,
    n_loci: usize,
    n_objectives: usize,
    pop_size: usize,
    n_gen: usize,
    mutation_rate: f64,
    seed: u64,
) -> ExperimentResult
where
    F: Fn(&[f64], &[f64], usize, usize, usize, usize, &mut Rng) -> Vec<f64>,
{
    let mut rng = Rng::new(seed);
    let mut population: Vec<f64> = (0..pop_size * n_loci).map(|_| rng.uniform()).collect();

    let mut diversity = Vec::with_capacity(n_gen);
    let mut pareto_front = Vec::with_capacity(n_gen);
    let mut mean_fitness = Vec::with_capacity(n_gen);

    for _ in 0..n_gen {
        let mut fitnesses: Vec<f64> = Vec::with_capacity(pop_size * n_objectives);
        for i in 0..pop_size {
            let genotype = &population[i * n_loci..(i + 1) * n_loci];
            fitnesses.extend(multi_objective_fitness(genotype, n_objectives));
        }

        diversity.push(phenotype_diversity(
            &fitnesses,
            pop_size,
            n_objectives,
            &mut rng,
        ));
        pareto_front.push(pareto_front_count(&fitnesses, pop_size, n_objectives));
        let sum_all: f64 = (0..pop_size)
            .map(|i| {
                fitnesses[i * n_objectives..(i + 1) * n_objectives]
                    .iter()
                    .sum::<f64>()
            })
            .sum();
        mean_fitness.push(sum_all / pop_size as f64);

        let selected = selection_fn(
            &population,
            &fitnesses,
            pop_size,
            n_loci,
            n_objectives,
            pop_size,
            &mut rng,
        );
        let mut next_pop = Vec::with_capacity(pop_size * n_loci);
        for ind in selected.chunks_exact(n_loci) {
            for &x in ind {
                next_pop.push((x + rng.normal_params(0.0, mutation_rate)).clamp(0.0, 1.0));
            }
        }
        population = next_pop;
    }

    ExperimentResult {
        diversity,
        pareto_front,
        mean_fitness,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_objective_fitness_produces_n_objectives() {
        let g: Vec<f64> = (0..40).map(|i| f64::from(i) / 40.0).collect();
        let f = multi_objective_fitness(&g, 4);
        assert_eq!(f.len(), 4);
    }

    #[test]
    fn pareto_front_count_bounded() {
        let fits = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]; // 3×2 flat
        let c = pareto_front_count(&fits, 3, 2);
        assert!(c <= 3);
    }

    #[test]
    fn lexicase_preserves_more_diversity_than_truncation() {
        let r1 = run_selection_experiment(lexicase_selection, 40, 4, 200, 50, 0.03, 42);
        let r2 = run_selection_experiment(truncation_selection, 40, 4, 200, 50, 0.03, 42);
        let lex_div: f64 = r1.diversity[r1.diversity.len() - 10..].iter().sum::<f64>() / 10.0;
        let trunc_div: f64 = r2.diversity[r2.diversity.len() - 10..].iter().sum::<f64>() / 10.0;
        assert!(
            lex_div > trunc_div,
            "lexicase {lex_div} > truncation {trunc_div}"
        );
    }

    #[test]
    fn tournament_beats_random_on_fitness() {
        let r1 = run_selection_experiment(tournament_selection, 40, 4, 200, 50, 0.03, 42);
        let r2 = run_selection_experiment(random_selection, 40, 4, 200, 50, 0.03, 42);
        let tourn: f64 = r1.mean_fitness[r1.mean_fitness.len() - 10..]
            .iter()
            .sum::<f64>()
            / 10.0;
        let rand: f64 = r2.mean_fitness[r2.mean_fitness.len() - 10..]
            .iter()
            .sum::<f64>()
            / 10.0;
        assert!(tourn > rand, "tournament {tourn} > random {rand}");
    }

    #[test]
    fn downsampled_lexicase_runs_and_selects() {
        let r = run_selection_experiment(downsampled_lexicase_selection, 40, 4, 100, 30, 0.03, 42);
        assert_eq!(r.mean_fitness.len(), 30);
        assert!(r.pareto_front.iter().all(|&p| p > 0));
    }

    #[test]
    fn determinism() {
        let r1 = run_selection_experiment(lexicase_selection, 40, 4, 100, 20, 0.03, 123);
        let r2 = run_selection_experiment(lexicase_selection, 40, 4, 100, 20, 0.03, 123);
        assert_eq!(r1.mean_fitness, r2.mean_fitness);
        assert_eq!(r1.pareto_front, r2.pareto_front);
        assert_eq!(r1.diversity, r2.diversity);
    }
}
