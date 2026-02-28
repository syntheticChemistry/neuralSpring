// SPDX-License-Identifier: AGPL-3.0-or-later

//! Counterdiabatic driving of evolution on NK fitness landscapes.
//!
//! Port of `control/counterdiabatic/counterdiabatic_evolution.py`.
//!
//! Reproduces key results from:
//! Iram, Dolson, Chiel, Hu, Nicholson, Ponce, Butts, Raman, Ohno (2020)
//! "Controlling the speed and trajectory of evolution with counterdiabatic
//!  driving" Nature Physics 17, 135–142. doi:10.1038/s41567-020-0989-3
//!
//! Model: Wright-Fisher population dynamics on NK fitness landscapes.
//! CD protocol minimizes geodesic length via Fisher information metric.
//!
//! ## `BarraCUDA` connection
//!
//! - NK fitness evaluation: `barracuda::ops::batch_gemm` (population × landscape)
//! - Boltzmann distribution: `barracuda::ops::softmax` (free energy → probabilities)
//! - Fisher information: `barracuda::ops::FusedMapReduceF64` (gradient inner product)
//! - Geodesic distance: scalar reduction (L2 norm of Fisher transport)

use crate::primitives::LOG_GUARD;
use crate::rng::Rng;

/// Probability floor for Boltzmann weights to prevent `log(0)` in KL divergence.
///
/// Domain-specific: 1e-30 is larger than [`crate::primitives::LOG_GUARD`] (1e-300)
/// because NK model Boltzmann weights at β=1 can legitimately be O(1e-20)
/// and we need this floor to be well below the smallest real probability
/// while staying above the threshold where f64 arithmetic degrades.
const SAFETY_EPS: f64 = 1e-30;

/// Floor for Fisher information metric to prevent `ds/dt → ∞`.
///
/// Domain-specific: the Fisher metric `g(s) = β²·Var_s[F]` vanishes at
/// landscape saddle points. This floor caps the geodesic speed while
/// being negligible compared to typical `g(s) ∈ [1e-4, 1]` for β=1.
/// See also: [`crate::primitives::DIVISION_GUARD`] for the generic guard.
const FISHER_EPS: f64 = 1e-10;

/// NK fitness landscape: N binary loci, K epistatic interactions.
///
/// Each locus has its fitness contribution depend on K other loci.
/// Total fitness = mean of per-locus contributions.
#[derive(Debug, Clone)]
pub struct NkLandscape {
    n: usize,
    k: usize,
    /// neighbors[i] = K indices (other loci) that affect locus i
    neighbors: Vec<Vec<usize>>,
    /// tables[i][idx] = fitness contribution for locus i, idx = bit pattern
    tables: Vec<Vec<f64>>,
}

impl NkLandscape {
    /// Create NK landscape with given N, K and seed for reproducibility.
    #[must_use]
    pub fn new(n: usize, k: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let neighbors: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                let candidates: Vec<usize> = (0..n).filter(|&j| j != i).collect();
                let chosen = rng.choose_distinct(candidates.len(), k.min(candidates.len()));
                chosen.iter().map(|&idx| candidates[idx]).collect()
            })
            .collect();
        let n_entries = 1 << (k + 1);
        let tables: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..n_entries).map(|_| rng.uniform()).collect())
            .collect();
        Self {
            n,
            k,
            neighbors,
            tables,
        }
    }

    /// Fitness of a binary genotype (slice of 0/1).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn fitness(&self, genotype: &[u8]) -> f64 {
        let mut total = 0.0;
        for i in 0..self.n {
            let mut idx = genotype[i] as usize;
            for (p, &j) in self.neighbors[i].iter().enumerate() {
                idx += (genotype[j] as usize) << (p + 1);
            }
            total += self.tables[i][idx.min(self.tables[i].len() - 1)];
        }
        total / self.n as f64
    }

    /// Fitness for all 2^N genotypes (genotype g = integer 0..2^N).
    /// Number of epistatic interactions per locus.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn all_fitnesses(&self) -> Vec<f64> {
        let n_geno = 1 << self.n;
        let mut fitnesses = Vec::with_capacity(n_geno);
        for g in 0..n_geno {
            let geno: Vec<u8> = (0..self.n).map(|i| ((g >> i) & 1) as u8).collect();
            fitnesses.push(self.fitness(&geno));
        }
        fitnesses
    }
}

/// Equilibrium distribution at inverse temperature β.
/// `p_i` ∝ exp(β * `f_i`)
#[must_use]
pub fn boltzmann_distribution(fitnesses: &[f64], beta: f64) -> Vec<f64> {
    let max_f: f64 = fitnesses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_w: Vec<f64> = fitnesses
        .iter()
        .map(|&f| (beta * (f - max_f)).exp())
        .collect();
    let sum: f64 = log_w.iter().sum();
    log_w.iter().map(|w| w / sum).collect()
}

/// Fitness landscape at drug concentration s ∈ [0, 1].
/// F(s) = (1-s)*f0 + s*f1
#[must_use]
pub fn interpolated_fitness(f0: &[f64], f1: &[f64], s: f64) -> Vec<f64> {
    f0.iter()
        .zip(f1.iter())
        .map(|(&a, &b)| (1.0 - s).mul_add(a, s * b))
        .collect()
}

/// KL(p ‖ q) with numerical safeguards.
#[must_use]
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    let p_norm: f64 = p.iter().sum();
    let q_norm: f64 = q.iter().sum();
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            let pi = (pi / p_norm).max(SAFETY_EPS);
            let qi = (qi / q_norm).max(SAFETY_EPS);
            pi * (pi / qi).ln()
        })
        .sum()
}

/// Counterdiabatic schedule: ds/dt ∝ 1/√g(s), g(s) = β² `Var_s[F]`.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn compute_cd_schedule(f0: &[f64], f1: &[f64], t: usize, beta: f64) -> Vec<f64> {
    // Grid resolution for Fisher information integration.
    // At N_STEPS=1000 the trapezoidal rule achieves O(1e-6) relative
    // error for smooth β²Var[F] profiles. Matches Python baseline.
    const N_STEPS: usize = 1000;
    let n_steps_f = N_STEPS as f64;
    let s_grid: Vec<f64> = (0..=N_STEPS).map(|i| i as f64 / n_steps_f).collect();

    let mut fisher_info = Vec::with_capacity(N_STEPS + 1);
    for &s in &s_grid {
        let f_s = interpolated_fitness(f0, f1, s);
        let p_s = boltzmann_distribution(&f_s, beta);
        let mean_f: f64 = barracuda::stats::dot(&p_s, &f_s);
        let var_f: f64 = p_s
            .iter()
            .zip(f_s.iter())
            .map(|(p, f)| p * (f - mean_f).powi(2))
            .sum();
        fisher_info.push((beta * beta).mul_add(var_f, FISHER_EPS));
    }

    let integrand: Vec<f64> = fisher_info.iter().map(|g| g.sqrt()).collect();
    let mut cumulative: Vec<f64> = integrand
        .iter()
        .scan(0.0, |acc, &v| {
            *acc += v / n_steps_f;
            Some(*acc)
        })
        .collect();
    let total = cumulative.last().copied().unwrap_or(1.0).max(LOG_GUARD);
    for c in &mut cumulative {
        *c /= total;
    }

    let t_uniform: Vec<f64> = (0..t).map(|i| i as f64 / (t - 1).max(1) as f64).collect();
    let schedule: Vec<f64> = t_uniform
        .iter()
        .map(|&tv| {
            if tv <= cumulative[0] {
                return s_grid[0];
            }
            if tv >= *cumulative.last().unwrap_or(&1.0) {
                return *s_grid.last().unwrap_or(&1.0);
            }
            let idx = cumulative.iter().position(|&c| c >= tv).unwrap_or(N_STEPS);
            let idx = idx.min(N_STEPS);
            let i0 = idx.saturating_sub(1);
            let i1 = (idx + 1).min(N_STEPS);
            let c0 = cumulative[i0];
            let c1 = cumulative[i1];
            let frac = if (c1 - c0).abs() < 1e-15 {
                0.0
            } else {
                ((tv - c0) / (c1 - c0)).clamp(0.0, 1.0)
            };
            let s0 = s_grid[i0];
            let s1 = s_grid[i1];
            (s0 + frac * (s1 - s0)).clamp(0.0, 1.0)
        })
        .collect();
    schedule
}

/// Result of running deterministic protocol.
#[derive(Debug, Clone)]
pub struct ProtocolResult {
    pub mean_kl: Vec<f64>,
    pub final_dist: f64,
}

/// Deterministic (mean-field) Wright-Fisher under drug schedule.
/// p'_i = `p_i` * `f_i` / \<f\>, then clamp and normalize.
#[must_use]
pub fn run_protocol_deterministic(f0: &[f64], f1: &[f64], schedule: &[f64]) -> ProtocolResult {
    let mut freq = boltzmann_distribution(f0, 1.0);
    let target = boltzmann_distribution(f1, 1.0);
    let mut mean_kl = Vec::with_capacity(schedule.len());

    for &s in schedule {
        let f_t = interpolated_fitness(f0, f1, s);
        let w: Vec<f64> = freq.iter().zip(f_t.iter()).map(|(a, b)| a * b).collect();
        let w_sum: f64 = w.iter().sum();
        if w_sum > 0.0 {
            freq = w.iter().map(|x| (x / w_sum).max(SAFETY_EPS)).collect();
        }
        let sum: f64 = freq.iter().sum();
        for x in &mut freq {
            *x /= sum;
        }
        let eq_t = boltzmann_distribution(&f_t, 1.0);
        mean_kl.push(kl_divergence(&freq, &eq_t));
    }

    let final_dist: f64 = freq
        .iter()
        .zip(target.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    ProtocolResult {
        mean_kl,
        final_dist,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;
    use approx::assert_relative_eq;

    #[test]
    fn nk_landscape_valid_fitnesses() {
        let l = NkLandscape::new(4, 2, 42);
        let f = l.all_fitnesses();
        assert_eq!(f.len(), 16);
        for &x in &f {
            assert!((0.0..=1.0).contains(&x), "fitness {x} out of [0,1]");
        }
    }

    #[test]
    fn boltzmann_sums_to_one() {
        let f = [1.0, 2.0, 0.5, 1.5];
        let p = boltzmann_distribution(&f, 1.0);
        let sum: f64 = p.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = tolerances::EXACT_F64);
    }

    #[test]
    fn interpolation_at_s0_returns_f0() {
        let f0 = [0.1, 0.2, 0.3];
        let f1 = [0.5, 0.6, 0.7];
        let interp = interpolated_fitness(&f0, &f1, 0.0);
        for (a, b) in interp.iter().zip(f0.iter()) {
            assert_relative_eq!(a, b, epsilon = tolerances::EXACT_F64);
        }
    }

    #[test]
    fn kl_self_zero() {
        let p = [0.25, 0.25, 0.5];
        let kl = kl_divergence(&p, &p);
        assert_relative_eq!(kl, 0.0, epsilon = tolerances::CROSS_LANGUAGE);
    }

    #[test]
    fn cd_schedule_endpoints() {
        let f0 = [0.1, 0.2, 0.3, 0.4];
        let f1 = [0.5, 0.6, 0.7, 0.8];
        let sched = compute_cd_schedule(&f0, &f1, 20, 1.0);
        assert!(sched[0] < 0.1, "start near 0");
        assert!(sched[sched.len() - 1] > 0.9, "end near 1");
    }

    #[test]
    fn deterministic_protocol_deterministic() {
        let f0 = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let f1 = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2];
        let sched = compute_cd_schedule(&f0, &f1, 50, 1.0);
        let r1 = run_protocol_deterministic(&f0, &f1, &sched);
        let r2 = run_protocol_deterministic(&f0, &f1, &sched);
        assert_relative_eq!(
            r1.final_dist,
            r2.final_dist,
            epsilon = tolerances::EXACT_F64
        );
    }
}
