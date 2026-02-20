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

use crate::rng::Rng;

const EPSILON: f64 = 1e-8;

/// Compute fitness on multiple objectives.
///
/// Each objective rewards a different portion of the genome.
/// `fitness[i] = mean(chunk_i) + 0.1 * std(chunk_i)`.
/// Last chunk gets remainder of loci.
#[allow(clippy::cast_precision_loss)]
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
pub fn random_selection(
    population: &[Vec<f64>],
    _fitnesses: &[Vec<f64>],
    n_select: usize,
    rng: &mut Rng,
) -> Vec<Vec<f64>> {
    (0..n_select)
        .map(|_| population[rng.usize(population.len())].clone())
        .collect()
}

/// Truncation: select top fraction by aggregate fitness.
pub fn truncation_selection(
    population: &[Vec<f64>],
    fitnesses: &[Vec<f64>],
    n_select: usize,
    rng: &mut Rng,
) -> Vec<Vec<f64>> {
    let mut agg: Vec<(usize, f64)> = fitnesses
        .iter()
        .enumerate()
        .map(|(i, f)| (i, f.iter().sum()))
        .collect();
    agg.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k = (n_select / 4).max(2).min(agg.len());
    let best_idx: Vec<usize> = agg.iter().rev().take(top_k).map(|x| x.0).collect();
    (0..n_select)
        .map(|_| population[best_idx[rng.usize(best_idx.len())]].clone())
        .collect()
}

/// Tournament selection: aggregate fitness comparison.
pub fn tournament_selection(
    population: &[Vec<f64>],
    fitnesses: &[Vec<f64>],
    n_select: usize,
    rng: &mut Rng,
) -> Vec<Vec<f64>> {
    let agg: Vec<f64> = fitnesses.iter().map(|f| f.iter().sum()).collect();
    let tournament_size = 5usize.min(population.len());
    (0..n_select)
        .map(|_| {
            let contestants = rng.choose_distinct(population.len(), tournament_size);
            let winner = contestants
                .iter()
                .max_by(|a, b| {
                    agg[**a]
                        .partial_cmp(&agg[**b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .unwrap_or(0);
            population[winner].clone()
        })
        .collect()
}

/// Lexicase selection: filter by shuffled per-case fitness.
pub fn lexicase_selection(
    population: &[Vec<f64>],
    fitnesses: &[Vec<f64>],
    n_select: usize,
    rng: &mut Rng,
) -> Vec<Vec<f64>> {
    let n_obj = fitnesses[0].len();
    let mut selected = Vec::with_capacity(n_select);
    for _ in 0..n_select {
        let mut candidates: Vec<usize> = (0..population.len()).collect();
        let obj_order = rng.permutation(n_obj);
        for obj in obj_order {
            if candidates.len() <= 1 {
                break;
            }
            let best = candidates
                .iter()
                .map(|&i| fitnesses[i][obj])
                .fold(f64::NEG_INFINITY, f64::max);
            candidates.retain(|&i| fitnesses[i][obj] >= best - EPSILON);
        }
        let winner = candidates[rng.usize(candidates.len())];
        selected.push(population[winner].clone());
    }
    selected
}

/// Down-sampled lexicase: use random subset (50%) of objectives.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn downsampled_lexicase_selection(
    population: &[Vec<f64>],
    fitnesses: &[Vec<f64>],
    n_select: usize,
    rng: &mut Rng,
) -> Vec<Vec<f64>> {
    let n_obj = fitnesses[0].len();
    let n_sub = (n_obj as f64 * 0.5).ceil() as usize;
    let n_sub = n_sub.max(2).min(n_obj);
    let mut selected = Vec::with_capacity(n_select);
    for _ in 0..n_select {
        let mut candidates: Vec<usize> = (0..population.len()).collect();
        let obj_order = rng.choose_distinct(n_obj, n_sub);
        for obj in obj_order {
            if candidates.len() <= 1 {
                break;
            }
            let best = candidates
                .iter()
                .map(|&i| fitnesses[i][obj])
                .fold(f64::NEG_INFINITY, f64::max);
            candidates.retain(|&i| fitnesses[i][obj] >= best - EPSILON);
        }
        let winner = candidates[rng.usize(candidates.len())];
        selected.push(population[winner].clone());
    }
    selected
}

/// Count Pareto-optimal individuals.
///
/// Individual i is dominated if exists j where all `fitnesses[j] >= fitnesses[i]`
/// and at least one strict.
#[must_use]
pub fn pareto_front_count(fitnesses: &[Vec<f64>]) -> usize {
    let n = fitnesses.len();
    let mut is_pareto = vec![true; n];
    for i in 0..n {
        if !is_pareto[i] {
            continue;
        }
        for j in 0..n {
            if i == j || !is_pareto[j] {
                continue;
            }
            let all_ge = fitnesses[j]
                .iter()
                .zip(fitnesses[i].iter())
                .all(|(a, b)| a >= b);
            let any_strict = fitnesses[j]
                .iter()
                .zip(fitnesses[i].iter())
                .any(|(a, b)| a > b);
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
    pub diversity: Vec<f64>,
    pub pareto_front: Vec<usize>,
    pub mean_fitness: Vec<f64>,
}

#[allow(clippy::cast_precision_loss)]
fn phenotype_diversity(fitnesses: &[Vec<f64>], rng: &mut Rng) -> f64 {
    let n = 50.min(fitnesses.len());
    if n < 2 {
        return 0.0;
    }
    let idx = rng.choose_distinct(fitnesses.len(), n);
    let subset: Vec<&Vec<f64>> = idx.iter().map(|&i| &fitnesses[i]).collect();
    let mut dists = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = subset[i]
                .iter()
                .zip(subset[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            dists.push(d);
        }
    }
    dists.iter().sum::<f64>() / dists.len() as f64
}

/// Run EA with a given selection algorithm, track multi-objective metrics.
#[allow(clippy::cast_precision_loss)]
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
    F: Fn(&[Vec<f64>], &[Vec<f64>], usize, &mut Rng) -> Vec<Vec<f64>>,
{
    let mut rng = Rng::new(seed);
    let mut population: Vec<Vec<f64>> = (0..pop_size)
        .map(|_| (0..n_loci).map(|_| rng.uniform()).collect())
        .collect();

    let mut diversity = Vec::with_capacity(n_gen);
    let mut pareto_front = Vec::with_capacity(n_gen);
    let mut mean_fitness = Vec::with_capacity(n_gen);

    for _ in 0..n_gen {
        let fitnesses: Vec<Vec<f64>> = population
            .iter()
            .map(|g| multi_objective_fitness(g, n_objectives))
            .collect();

        diversity.push(phenotype_diversity(&fitnesses, &mut rng));
        pareto_front.push(pareto_front_count(&fitnesses));
        mean_fitness
            .push(fitnesses.iter().map(|f| f.iter().sum::<f64>()).sum::<f64>() / pop_size as f64);

        let selected = selection_fn(&population, &fitnesses, pop_size, &mut rng);
        population = selected
            .into_iter()
            .map(|ind| {
                ind.into_iter()
                    .map(|x| (x + rng.normal_params(0.0, mutation_rate)).clamp(0.0, 1.0))
                    .collect()
            })
            .collect();
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
        let fits = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        let c = pareto_front_count(&fits);
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
    fn determinism() {
        let r1 = run_selection_experiment(lexicase_selection, 40, 4, 100, 20, 0.03, 123);
        let r2 = run_selection_experiment(lexicase_selection, 40, 4, 100, 20, 0.03, 123);
        assert_eq!(r1.mean_fitness, r2.mean_fitness);
        assert_eq!(r1.pareto_front, r2.pareto_front);
        assert_eq!(r1.diversity, r2.diversity);
    }
}
