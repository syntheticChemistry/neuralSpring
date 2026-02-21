// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hidden Markov Model: forward, backward, Viterbi, and posterior.
//!
//! Port of `control/hmm_phylo/hmm_phylo.py`.
//!
//! Liu et al. (2014) "An HMM-based Comparative Genomic Framework for
//! Detecting Introgression in the Presence of Incomplete Lineage Sorting"
//! `PLoS` Computational Biology 10(4):e1003649.
//!
//! ## Layout
//!
//! All matrices use **flat row-major** `Vec<f64>` storage — cache-friendly
//! on CPU and directly uploadable to GPU buffers.
//!
//! - `transition[i*n + j]` = P(s\_{t+1}=j | `s_t`=i), shape N×N
//! - `emission[i*m + k]`   = P(`o_t`=k | `s_t`=i),    shape N×M
//! - `alpha[t*n + i]`      = P(`s_t`=i | o\_{1..t}),   shape T×N
//!
//! ## `BarraCUDA` connection
//!
//! - Forward/backward: sequential GEMV chain → `barracuda::staging::StatefulPipeline`
//! - Log-domain numerics: `barracuda::ops::logsumexp` (5/5 PASS)
//! - Transition GEMM: `barracuda::ops::matmul`
//!
//! ## WGSL shader (absorption-ready)
//!
//! [`WGSL_HMM_FORWARD_LOG`] — log-domain HMM forward pass. One thread
//! per hidden state, logsumexp reduction per step. Validated in
//! `validate_gpu_hmm_forward` (13/13 PASS).

#![allow(clippy::needless_range_loop)]

use crate::primitives;
use crate::rng::Rng;

/// WGSL shader: log-domain HMM forward pass.
///
/// Absorption target: `barracuda::ops::hmm` or `StatefulPipeline`.
/// Validated: `validate_gpu_hmm_forward` (13/13 PASS).
pub const WGSL_HMM_FORWARD_LOG: &str = include_str!("../metalForge/shaders/hmm_forward_log.wgsl");

/// Discrete HMM with N hidden states and M observation symbols.
///
/// All matrices are flat row-major `Vec<f64>` for GPU compatibility.
#[derive(Debug, Clone)]
pub struct Hmm {
    /// N×N transition matrix: `A[i*n + j]` = P(s\_{t+1}=j | `s_t`=i)
    pub transition: Vec<f64>,
    /// N×M emission matrix: `B[i*m + k]` = P(`o_t`=k | `s_t`=i)
    pub emission: Vec<f64>,
    /// Initial distribution π\[i\] = P(`s_1`=i)
    pub initial: Vec<f64>,
    /// Number of hidden states.
    n: usize,
    /// Number of observation symbols.
    m: usize,
}

/// Result of forward algorithm.
#[derive(Debug, Clone)]
pub struct ForwardResult {
    /// T×N alpha matrix (flat row-major): `alpha[t*n + i]`.
    pub alpha: Vec<f64>,
    pub scales: Vec<f64>,
    pub log_likelihood: f64,
    /// Number of hidden states (stride for alpha).
    pub n: usize,
}

impl ForwardResult {
    /// Access alpha values at timestep `t`.
    #[must_use]
    pub fn alpha_at(&self, t: usize) -> &[f64] {
        &self.alpha[t * self.n..(t + 1) * self.n]
    }
}

impl Hmm {
    /// Number of hidden states.
    #[must_use]
    pub const fn num_states(&self) -> usize {
        self.n
    }

    /// Number of observation symbols.
    #[must_use]
    pub const fn num_symbols(&self) -> usize {
        self.m
    }

    /// Create HMM from flat row-major matrices.
    ///
    /// - `transition`: N×N flat row-major
    /// - `emission`: N×M flat row-major
    /// - `initial`: length N
    #[must_use]
    pub fn from_flat(
        transition: Vec<f64>,
        emission: Vec<f64>,
        initial: Vec<f64>,
        n: usize,
        m: usize,
    ) -> Self {
        debug_assert_eq!(transition.len(), n * n);
        debug_assert_eq!(emission.len(), n * m);
        debug_assert_eq!(initial.len(), n);
        Self {
            transition,
            emission,
            initial,
            n,
            m,
        }
    }

    /// Create HMM from nested Vecs (convenience for test/validation code).
    ///
    /// Converts `Vec<Vec<f64>>` → flat row-major layout internally.
    #[must_use]
    pub fn new(transition: Vec<Vec<f64>>, emission: Vec<Vec<f64>>, initial: Vec<f64>) -> Self {
        let n = transition.len();
        let m = emission.first().map_or(0, Vec::len);
        let flat_trans: Vec<f64> = transition.into_iter().flatten().collect();
        let flat_emit: Vec<f64> = emission.into_iter().flatten().collect();
        Self::from_flat(flat_trans, flat_emit, initial, n, m)
    }

    /// Forward algorithm with scaling. Returns (alpha as T×N flat, `log_likelihood`).
    #[must_use]
    pub fn forward(&self, obs: &[usize]) -> (Vec<f64>, f64) {
        let r = self.forward_full(obs);
        (r.alpha, r.log_likelihood)
    }

    /// Forward algorithm returning full result including scales for backward.
    #[must_use]
    pub fn forward_full(&self, obs: &[usize]) -> ForwardResult {
        let t_len = obs.len();
        let mut alpha = vec![0.0; t_len * self.n];
        let mut scales = vec![0.0; t_len];

        let ob0 = obs[0].min(self.m - 1);
        for i in 0..self.n {
            alpha[i] = self.initial[i] * self.emission[i * self.m + ob0];
        }
        scales[0] = alpha[..self.n].iter().sum();
        if scales[0] > 0.0 {
            for x in &mut alpha[..self.n] {
                *x /= scales[0];
            }
        }

        for t in 1..t_len {
            let obt = obs[t].min(self.m - 1);
            for j in 0..self.n {
                let mut sum = 0.0;
                for i in 0..self.n {
                    sum += alpha[(t - 1) * self.n + i] * self.transition[i * self.n + j];
                }
                alpha[t * self.n + j] = sum * self.emission[j * self.m + obt];
            }
            scales[t] = alpha[t * self.n..(t + 1) * self.n].iter().sum();
            if scales[t] > 0.0 {
                for x in &mut alpha[t * self.n..(t + 1) * self.n] {
                    *x /= scales[t];
                }
            }
        }

        let log_lik: f64 = scales
            .iter()
            .map(|s| (s + primitives::LOG_GUARD).ln())
            .sum();
        ForwardResult {
            alpha,
            scales,
            log_likelihood: log_lik,
            n: self.n,
        }
    }

    /// Backward algorithm. Requires scales from forward pass.
    /// Returns T×N flat row-major beta matrix.
    #[must_use]
    pub fn backward(&self, obs: &[usize], scales: &[f64]) -> Vec<f64> {
        let t_len = obs.len();
        let mut beta = vec![0.0; t_len * self.n];

        for i in 0..self.n {
            beta[(t_len - 1) * self.n + i] = 1.0;
        }

        for t in (0..t_len.saturating_sub(1)).rev() {
            let ob_next = obs[t + 1].min(self.m - 1);
            for i in 0..self.n {
                let mut sum = 0.0;
                for j in 0..self.n {
                    sum += self.transition[i * self.n + j]
                        * self.emission[j * self.m + ob_next]
                        * beta[(t + 1) * self.n + j];
                }
                beta[t * self.n + i] = if t + 1 < scales.len() && scales[t + 1] > 0.0 {
                    sum / scales[t + 1]
                } else {
                    sum
                };
            }
        }
        beta
    }

    /// Viterbi algorithm: most likely state sequence. Returns (path, `log_prob`).
    #[must_use]
    pub fn viterbi(&self, obs: &[usize]) -> (Vec<usize>, f64) {
        let t_len = obs.len();

        let log_a: Vec<f64> = self
            .transition
            .iter()
            .map(|&x| (x + primitives::LOG_GUARD).ln())
            .collect();
        let log_b: Vec<f64> = self
            .emission
            .iter()
            .map(|&x| (x + primitives::LOG_GUARD).ln())
            .collect();
        let log_pi: Vec<f64> = self
            .initial
            .iter()
            .map(|&x| (x + primitives::LOG_GUARD).ln())
            .collect();

        let mut delta = vec![0.0; t_len * self.n];
        let mut psi = vec![0_usize; t_len * self.n];

        let ob0 = obs[0].min(self.m - 1);
        for i in 0..self.n {
            delta[i] = log_pi[i] + log_b[i * self.m + ob0];
        }

        for t in 1..t_len {
            let obt = obs[t].min(self.m - 1);
            for j in 0..self.n {
                let (best_i, best) = (0..self.n)
                    .map(|i| (i, delta[(t - 1) * self.n + i] + log_a[i * self.n + j]))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, f64::NEG_INFINITY));
                psi[t * self.n + j] = best_i;
                delta[t * self.n + j] = best + log_b[j * self.m + obt];
            }
        }

        let mut path = vec![0; t_len];
        let last_row = &delta[(t_len - 1) * self.n..t_len * self.n];
        let (best_j, log_prob) = last_row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or((0, f64::NEG_INFINITY), |(i, &v)| (i, v));
        path[t_len - 1] = best_j;

        for t in (0..t_len.saturating_sub(1)).rev() {
            path[t] = psi[(t + 1) * self.n + path[t + 1]];
        }
        (path, log_prob)
    }

    /// Posterior `P(s_t=i | O)` via forward-backward.
    /// Returns T×N flat row-major gamma matrix.
    #[must_use]
    pub fn posterior(&self, obs: &[usize]) -> Vec<f64> {
        let t_len = obs.len();
        let fwd = self.forward_full(obs);
        let beta = self.backward(obs, &fwd.scales);

        let mut gamma = vec![0.0; t_len * self.n];
        for t in 0..t_len {
            let mut sum = 0.0;
            for i in 0..self.n {
                gamma[t * self.n + i] = fwd.alpha[t * self.n + i] * beta[t * self.n + i];
                sum += gamma[t * self.n + i];
            }
            if sum > 0.0 {
                for i in 0..self.n {
                    gamma[t * self.n + i] /= sum;
                }
            }
        }
        gamma
    }

    /// Generate (states, observations) from the model.
    pub fn generate_sequence(&self, length: usize, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
        let mut states = vec![0; length];
        let mut observations = vec![0; length];

        states[0] = rng.categorical(&self.initial);
        observations[0] =
            rng.categorical(&self.emission[states[0] * self.m..(states[0] + 1) * self.m]);

        for t in 1..length {
            let prev = states[t - 1];
            states[t] = rng.categorical(&self.transition[prev * self.n..(prev + 1) * self.n]);
            observations[t] =
                rng.categorical(&self.emission[states[t] * self.m..(states[t] + 1) * self.m]);
        }
        (states, observations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn weather_hmm() -> Hmm {
        let transition = vec![vec![0.7, 0.3], vec![0.4, 0.6]];
        let emission = vec![vec![0.1, 0.4, 0.5], vec![0.6, 0.3, 0.1]];
        let initial = vec![0.6, 0.4];
        Hmm::new(transition, emission, initial)
    }

    #[test]
    fn forward_finite_neg_loglik() {
        let hmm = weather_hmm();
        let obs = [0, 1, 2, 0, 2];
        let (_, log_lik) = hmm.forward(&obs);
        assert!(log_lik.is_finite() && log_lik < 0.0);
    }

    #[test]
    fn forward_alpha_sums_to_one() {
        let hmm = weather_hmm();
        let obs = [0, 1, 2, 0, 2];
        let fwd = hmm.forward_full(&obs);
        for t in 0..obs.len() {
            let sum: f64 = fwd.alpha_at(t).iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn viterbi_path_valid() {
        let hmm = weather_hmm();
        let mut rng = Rng::new(42);
        let (_, obs) = hmm.generate_sequence(100, &mut rng);
        let (path, _) = hmm.viterbi(&obs);
        for &s in &path {
            assert!(s < hmm.n, "state {s} out of range");
        }
    }

    #[test]
    fn posterior_sums_to_one() {
        let hmm = weather_hmm();
        let mut rng = Rng::new(42);
        let (_, obs) = hmm.generate_sequence(50, &mut rng);
        let gamma = hmm.posterior(&obs);
        for t in 0..obs.len() {
            let sum: f64 = gamma[t * hmm.n..(t + 1) * hmm.n].iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn forward_loglik_finite_length100() {
        let hmm = weather_hmm();
        let mut rng = Rng::new(42);
        let (_, obs) = hmm.generate_sequence(100, &mut rng);
        let (_, log_lik) = hmm.forward(&obs);
        assert!(log_lik.is_finite());
    }

    #[test]
    fn generate_deterministic_with_seed() {
        let hmm = weather_hmm();
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        let (s1, o1) = hmm.generate_sequence(20, &mut rng1);
        let (s2, o2) = hmm.generate_sequence(20, &mut rng2);
        assert_eq!(s1, s2);
        assert_eq!(o1, o2);
    }

    #[test]
    fn from_flat_matches_new() {
        let trans = vec![vec![0.7, 0.3], vec![0.4, 0.6]];
        let emit = vec![vec![0.1, 0.4, 0.5], vec![0.6, 0.3, 0.1]];
        let init = vec![0.6, 0.4];
        let nested = Hmm::new(trans, emit, init.clone());

        let flat = Hmm::from_flat(
            vec![0.7, 0.3, 0.4, 0.6],
            vec![0.1, 0.4, 0.5, 0.6, 0.3, 0.1],
            init,
            2,
            3,
        );

        assert_eq!(nested.transition, flat.transition);
        assert_eq!(nested.emission, flat.emission);
    }
}
