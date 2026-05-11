// SPDX-License-Identifier: AGPL-3.0-or-later

// PRNG internals: u64→f64 casts lose precision by design; single-char state
// variables (`s`, `x`, `u`, `v`) match Blackman & Vigna's reference; and
// mul_add / complex float expressions are intentional for the algorithm.
#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::suboptimal_flops,
    reason = "domain-specific numeric patterns"
)]

//! Deterministic pseudo-random number generator for reproducible experiments.
//!
//! Uses Xoshiro256** (Blackman & Vigna, 2018) seeded via `SplitMix64`.
//! Provides the sampling primitives needed by evolutionary and stochastic
//! algorithms without external dependencies.

use crate::primitives::LOG_GUARD;
use std::f64::consts::PI;

/// WGSL shader: GPU-parallel PRNG (Xoshiro128**).
///
/// Absorption target: `barracuda::ops::prng`.
/// Validated: `validate_gpu_prng` (5/5 PASS).
#[cfg(feature = "barracuda")]
pub use neural_spring_forge::shaders::XOSHIRO128SS as WGSL_XOSHIRO128SS;

/// Deterministic PRNG based on Xoshiro256**.
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Create from an integer seed (seeded via `SplitMix64`).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            *slot = splitmix64(&mut sm);
        }
        Self { s }
    }

    /// Raw u64 output (xoshiro256**).
    #[must_use]
    pub const fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform `f64` in `[0, 1)`.
    #[must_use]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Standard normal via Box-Muller transform.
    #[must_use]
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(LOG_GUARD);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Normal with given mean and standard deviation.
    #[must_use]
    pub fn normal_params(&mut self, mean: f64, std: f64) -> f64 {
        std.mul_add(self.normal(), mean)
    }

    /// Uniform integer in `[0, n)`.
    #[must_use]
    pub const fn usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Choose index from a probability distribution (categorical sample).
    #[must_use]
    pub fn categorical(&mut self, probs: &[f64]) -> usize {
        let u = self.uniform();
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if u < cum {
                return i;
            }
        }
        probs.len() - 1
    }

    /// Multinomial sample: draw `n` items from a probability distribution.
    #[must_use]
    pub fn multinomial(&mut self, n: usize, probs: &[f64]) -> Vec<f64> {
        let mut counts = vec![0.0; probs.len()];
        for _ in 0..n {
            counts[self.categorical(probs)] += 1.0;
        }
        counts
    }

    /// Choose `k` distinct indices from `[0, n)` (Fisher-Yates partial).
    #[must_use]
    pub fn choose_distinct(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n).collect();
        let k = k.min(n);
        for i in 0..k {
            let j = i + self.usize(n - i);
            indices.swap(i, j);
        }
        indices[..k].to_vec()
    }

    /// Random permutation of `[0, n)`.
    #[must_use]
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        self.choose_distinct(n, n)
    }

    /// Fill a slice with `uniform() < threshold` coin flips.
    #[must_use]
    pub fn bernoulli_mask(&mut self, n: usize, p: f64) -> Vec<bool> {
        (0..n).map(|_| self.uniform() < p).collect()
    }

    /// Alias for `uniform()` — `f64` in `[0, 1)`.
    #[must_use]
    pub fn next_f64(&mut self) -> f64 {
        self.uniform()
    }

    /// Gamma variate via Marsaglia & Tsang (2000) for shape >= 1,
    /// with Ahrens-Dieter shift for shape < 1.
    #[must_use]
    pub fn gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            return self.gamma(shape + 1.0) * self.uniform().max(LOG_GUARD).powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v_base = 1.0 + c * x;
            if v_base <= 0.0 {
                continue;
            }
            let v = v_base * v_base * v_base;
            let u = self.uniform().max(LOG_GUARD);
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// Beta(α, β) variate via the gamma ratio method.
    #[must_use]
    pub fn beta(&mut self, alpha: f64, beta_param: f64) -> f64 {
        let x = self.gamma(alpha);
        let y = self.gamma(beta_param);
        let sum = x + y;
        if sum < LOG_GUARD {
            return 0.5;
        }
        x / sum
    }
}

const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn deterministic_across_runs() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn uniform_in_range() {
        let mut rng = Rng::new(42);
        for _ in 0..1000 {
            let v = rng.uniform();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn categorical_sums_correctly() {
        let mut rng = Rng::new(42);
        let probs = [0.2, 0.3, 0.5];
        let mut counts = [0u32; 3];
        let n: i32 = 10_000;
        for _ in 0..n {
            counts[rng.categorical(&probs)] += 1;
        }
        for (i, &p) in probs.iter().enumerate() {
            let frac = f64::from(counts[i]) / f64::from(n);
            assert!((frac - p).abs() < 0.05, "bin {i}: {frac} vs {p}");
        }
    }

    #[test]
    fn choose_distinct_no_repeats() {
        let mut rng = Rng::new(42);
        let chosen = rng.choose_distinct(20, 10);
        assert_eq!(chosen.len(), 10);
        let mut sorted = chosen;
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 10);
    }

    #[test]
    fn multinomial_sums_to_n() {
        let mut rng = Rng::new(42);
        let counts = rng.multinomial(1000, &[0.3, 0.3, 0.4]);
        let total: f64 = counts.iter().sum();
        assert!((total - 1000.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn normal_reasonable_range() {
        let mut rng = Rng::new(42);
        let mut sum = 0.0;
        let n: i32 = 10_000;
        for _ in 0..n {
            sum += rng.normal();
        }
        let mean = sum / f64::from(n);
        assert!(mean.abs() < 0.1, "normal mean={mean}");
    }

    #[test]
    fn determinism_rerun_identical() {
        let run = || {
            let mut rng = Rng::new(42);
            (0..100).map(|_| rng.uniform()).collect::<Vec<_>>()
        };
        let a = run();
        let b = run();
        assert_eq!(
            a, b,
            "two runs with same seed must produce identical sequences"
        );
    }

    #[test]
    fn determinism_normal_rerun_identical() {
        let run = || {
            let mut rng = Rng::new(42);
            (0..100).map(|_| rng.normal()).collect::<Vec<_>>()
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "normal() must be deterministic");
    }

    #[test]
    fn determinism_categorical_rerun_identical() {
        let run = || {
            let mut rng = Rng::new(42);
            let probs = [0.2, 0.3, 0.5];
            (0..100)
                .map(|_| rng.categorical(&probs))
                .collect::<Vec<_>>()
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "categorical() must be deterministic");
    }
}
