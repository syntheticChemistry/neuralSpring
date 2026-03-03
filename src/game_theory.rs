// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cooperative game theory and quorum sensing.
//!
//! Port of `control/game_theory/game_theory.py`.
//!
//! Reproduces key dynamics from:
//! Bruger & Waters (2018)
//! "Maximizing Growth Yield and Dispersal via Quorum Sensing Promotes
//!  Cooperation in Vibrio bacteria"
//! Applied and Environmental Microbiology 84(6):e00402-18.
//!
//! Core thesis: quorum sensing (QS) promotes cooperation by linking
//! individual growth yield to collective dispersal.
//!
//! ## `BarraCUDA` connection
//!
//! - Replicator dynamics: 2×2 GEMV + normalization (small, CPU-only)
//! - QS cooperation model: `barracuda::ops::batch_gemm` (fitness-weighted selection)
//! - Spatial PD fitness stencil: `barracuda::ops::stencil` (GPU `spatial_payoff.wgsl`)
//! - Population evolution: fitness proportional selection via `barracuda::stats`
//!
//! ## WGSL shader (absorption-ready)
//!
//! [`WGSL_SPATIAL_PAYOFF`] — spatial PD payoff stencil. One thread per
//! grid cell, Moore neighborhood with periodic boundary. Validated in
//! `validate_gpu_game_theory`.

use crate::rng::Rng;

/// WGSL shader: spatial PD payoff stencil on a 2D grid.
///
/// Absorption target: `barracuda::ops::stencil`.
/// Validated: `validate_gpu_game_theory`.
pub use neural_spring_forge::shaders::SPATIAL_PAYOFF as WGSL_SPATIAL_PAYOFF;

/// Standard prisoner's dilemma payoff matrix.
///
/// `Payoff[i][j]` = payoff to row player when opponent plays column.
/// Row 0 = cooperator, row 1 = defector.
#[must_use]
pub fn prisoners_dilemma_payoff(b: f64, c: f64) -> [[f64; 2]; 2] {
    [[b - c, -c], [b, 0.0]]
}

/// Snowdrift (hawk-dove) game: cooperation coexists with defection.
#[must_use]
pub fn snowdrift_payoff(b: f64, c: f64) -> [[f64; 2]; 2] {
    [[b - c / 2.0, b - c], [b, 0.0]]
}

/// Continuous replicator dynamics.
///
/// `dx_i/dt` = `x_i` * (`f_i` - `f_bar`) where f = payoff @ x, `f_bar` = x^T f.
/// Clamp to max(0), normalize.
#[must_use]
pub fn replicator_dynamics(
    freq: &[f64],
    payoff: &[[f64; 2]; 2],
    n_steps: usize,
    dt: f64,
) -> Vec<[f64; 2]> {
    let mut trace = Vec::with_capacity(n_steps + 1);
    let mut x = [freq[0], freq[1]];
    trace.push(x);

    for _ in 0..n_steps {
        let f0 = payoff[0][0].mul_add(x[0], payoff[0][1] * x[1]);
        let f1 = payoff[1][0].mul_add(x[0], payoff[1][1] * x[1]);
        let f_bar = x[0].mul_add(f0, x[1] * f1);

        let dx0 = x[0] * (f0 - f_bar);
        let dx1 = x[1] * (f1 - f_bar);

        x[0] = dt.mul_add(dx0, x[0]).max(0.0);
        x[1] = dt.mul_add(dx1, x[1]).max(0.0);

        let sum = x[0] + x[1];
        if sum > 0.0 {
            x[0] /= sum;
            x[1] /= sum;
        }
        trace.push(x);
    }
    trace
}

/// Configuration for QS cooperation model.
#[derive(Debug, Clone)]
pub struct QsConfig {
    pub pop_size: usize,
    pub n_gen: usize,
    pub qs_threshold: f64,
    pub cooperation_cost: f64,
    pub cooperation_benefit: f64,
    pub dispersal_bonus: f64,
    pub mutation_rate: f64,
    pub seed: u64,
}

/// Result of QS cooperation simulation.
#[derive(Debug, Clone)]
pub struct QsResult {
    pub coop_freq: Vec<f64>,
    pub mean_fitness: Vec<f64>,
}

/// Simulate QS-mediated cooperation dynamics.
///
/// When signal density > threshold, cooperators get dispersal bonus.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    clippy::naive_bytecount,
    reason = "small population counts → f64 for frequencies; bytecount crate overkill for pop_size"
)]
pub fn qs_cooperation_model(config: &QsConfig) -> QsResult {
    let mut rng = Rng::new(config.seed);
    let mut strategies: Vec<u8> = (0..config.pop_size)
        .map(|_| u8::from(rng.uniform() >= 0.5))
        .collect();

    let mut coop_freq = Vec::with_capacity(config.n_gen);
    let mut mean_fitness = Vec::with_capacity(config.n_gen);

    for _ in 0..config.n_gen {
        let freq = strategies.iter().filter(|&&s| s == 1).count() as f64 / config.pop_size as f64;
        coop_freq.push(freq);

        let qs_active = freq > config.qs_threshold;

        let mut fitness = vec![1.0; config.pop_size];
        for (i, &s) in strategies.iter().enumerate() {
            if s == 1 {
                fitness[i] -= config.cooperation_cost;
                fitness[i] += config.cooperation_benefit * freq;
                if qs_active {
                    fitness[i] += config.dispersal_bonus;
                }
            } else {
                fitness[i] += config.cooperation_benefit * freq * 0.5;
            }
            fitness[i] = fitness[i].max(0.01);
        }

        let sum_f: f64 = fitness.iter().sum();
        let probs: Vec<f64> = fitness.iter().map(|f| f / sum_f).collect();

        let parents: Vec<usize> = (0..config.pop_size)
            .map(|_| rng.categorical(&probs))
            .collect();
        strategies = parents.into_iter().map(|i| strategies[i]).collect();

        let mutants = rng.bernoulli_mask(config.pop_size, config.mutation_rate);
        for (i, &m) in mutants.iter().enumerate() {
            if m {
                strategies[i] = 1 - strategies[i];
            }
        }

        mean_fitness.push(fitness.iter().sum::<f64>() / config.pop_size as f64);
    }

    QsResult {
        coop_freq,
        mean_fitness,
    }
}

/// Spatial prisoner's dilemma on a grid.
///
/// Moore neighborhood (8 neighbors, periodic). Each cell copies strategy
/// of fittest neighbor. Rare mutation (2% chance of one random flip).
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "grid indices use isize for periodic boundary wrap; grid_size² → f64 for cooperation fraction"
)]
#[must_use]
pub fn spatial_cooperation(grid_size: usize, n_gen: usize, b: f64, c: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut grid: Vec<Vec<u8>> = (0..grid_size)
        .map(|_| {
            (0..grid_size)
                .map(|_| u8::from(rng.uniform() >= 0.5))
                .collect()
        })
        .collect();

    let mut coop_trace = Vec::with_capacity(n_gen);

    let neighbors: [(i32, i32); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];

    for _ in 0..n_gen {
        let total: usize = grid
            .iter()
            .flat_map(|r| r.iter())
            .filter(|&&c| c == 1)
            .count();
        coop_trace.push(total as f64 / (grid_size * grid_size) as f64);

        let mut fitness_grid = vec![vec![0.0; grid_size]; grid_size];
        for i in 0..grid_size {
            for j in 0..grid_size {
                let mut total = 0.0;
                for (di, dj) in &neighbors {
                    let ni = ((i as i32 + di).rem_euclid(grid_size as i32)) as usize;
                    let nj = ((j as i32 + dj).rem_euclid(grid_size as i32)) as usize;
                    let me = grid[i][j];
                    let other = grid[ni][nj];
                    total += match (me, other) {
                        (1, 1) => b - c,
                        (1, 0) => -c,
                        (0, 1) => b,
                        _ => 0.0,
                    };
                }
                fitness_grid[i][j] = total;
            }
        }

        let mut new_grid = vec![vec![0u8; grid_size]; grid_size];
        for i in 0..grid_size {
            for j in 0..grid_size {
                let mut best_fit = fitness_grid[i][j];
                let mut best_strategy = grid[i][j];
                for (di, dj) in &neighbors {
                    let ni = ((i as i32 + di).rem_euclid(grid_size as i32)) as usize;
                    let nj = ((j as i32 + dj).rem_euclid(grid_size as i32)) as usize;
                    if fitness_grid[ni][nj] > best_fit {
                        best_fit = fitness_grid[ni][nj];
                        best_strategy = grid[ni][nj];
                    }
                }
                new_grid[i][j] = best_strategy;
            }
        }

        if rng.uniform() < 0.02 {
            let mi = rng.usize(grid_size);
            let mj = rng.usize(grid_size);
            new_grid[mi][mj] = 1 - new_grid[mi][mj];
        }

        grid = new_grid;
    }

    coop_trace
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pd_replicator_defection_dominates() {
        let pd = prisoners_dilemma_payoff(3.0, 1.0);
        let freq = [0.5, 0.5];
        let trace = replicator_dynamics(&freq, &pd, 2000, 0.01);
        let final_coop = trace[trace.len() - 1][0];
        assert!(
            final_coop < 0.1,
            "PD: defection should dominate, got {final_coop}"
        );
    }

    #[test]
    fn snowdrift_coexistence() {
        let sd = snowdrift_payoff(3.0, 1.0);
        let freq = [0.5, 0.5];
        let trace = replicator_dynamics(&freq, &sd, 2000, 0.01);
        let final_coop = trace[trace.len() - 1][0];
        assert!(
            (0.1..0.9).contains(&final_coop),
            "Snowdrift: coexistence expected, got {final_coop}"
        );
    }

    #[test]
    fn qs_model_cooperation_above_03() {
        let config = QsConfig {
            pop_size: 300,
            n_gen: 500,
            qs_threshold: 0.3,
            cooperation_cost: 0.1,
            cooperation_benefit: 0.3,
            dispersal_bonus: 0.5,
            mutation_rate: 0.02,
            seed: 42,
        };
        let result = qs_cooperation_model(&config);
        let late_coop: f64 = result.coop_freq[result.coop_freq.len() - 50..]
            .iter()
            .sum::<f64>()
            / 50.0;
        assert!(
            late_coop > 0.3,
            "QS should maintain cooperation > 0.3, got {late_coop}"
        );
    }

    #[test]
    fn replicator_trace_length() {
        let pd = prisoners_dilemma_payoff(3.0, 1.0);
        let trace = replicator_dynamics(&[0.5, 0.5], &pd, 100, 0.01);
        assert_eq!(trace.len(), 101);
    }

    #[test]
    fn determinism() {
        let pd = prisoners_dilemma_payoff(3.0, 1.0);
        let t1 = replicator_dynamics(&[0.5, 0.5], &pd, 500, 0.01);
        let t2 = replicator_dynamics(&[0.5, 0.5], &pd, 500, 0.01);
        for (step, (a, b)) in t1.iter().zip(t2.iter()).enumerate() {
            assert!(
                (a[0] - b[0]).abs() < f64::EPSILON && (a[1] - b[1]).abs() < f64::EPSILON,
                "replicator dynamics not deterministic at step {step}"
            );
        }

        let config = QsConfig {
            pop_size: 100,
            n_gen: 50,
            qs_threshold: 0.3,
            cooperation_cost: 0.1,
            cooperation_benefit: 0.3,
            dispersal_bonus: 0.5,
            mutation_rate: 0.02,
            seed: 42,
        };
        let r1 = qs_cooperation_model(&config);
        let r2 = qs_cooperation_model(&config);
        assert_eq!(r1.coop_freq, r2.coop_freq);

        let s1 = spatial_cooperation(10, 20, 3.0, 1.0, 42);
        let s2 = spatial_cooperation(10, 20, 3.0, 1.0, 42);
        assert_eq!(s1, s2);
    }
}
