// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hidden Markov Model: forward, backward, Viterbi, and posterior.
//!
//! Port of `control/hmm_phylo/hmm_phylo.py`.
//!
//! Liu et al. (2014) "An HMM-based Comparative Genomic Framework for
//! Detecting Introgression in the Presence of Incomplete Lineage Sorting"
//! `PLoS` Computational Biology 10(4):e1003649.

use crate::rng::Rng;

const LOG_EPS: f64 = 1e-300;

/// Discrete HMM with N hidden states and M observation symbols.
///
/// A: N×N transition matrix, B: N×M emission matrix, π: initial distribution.
#[derive(Debug, Clone)]
pub struct Hmm {
    /// `A[i][j]` = P(s\_{t+1}=j | `s_t`=i)
    pub transition: Vec<Vec<f64>>,
    /// `B[i][k]` = P(`o_t`=k | `s_t`=i)
    pub emission: Vec<Vec<f64>>,
    /// `π[i]` = P(`s_1`=i)
    pub initial: Vec<f64>,
    n: usize,
    m: usize,
}

/// Result of forward algorithm.
#[derive(Debug, Clone)]
pub struct ForwardResult {
    pub alpha: Vec<Vec<f64>>,
    pub scales: Vec<f64>,
    pub log_likelihood: f64,
}

impl Hmm {
    /// Number of hidden states.
    #[must_use]
    pub const fn num_states(&self) -> usize {
        self.n
    }

    /// Create HMM from transition, emission, and initial distributions.
    #[must_use]
    pub fn new(transition: Vec<Vec<f64>>, emission: Vec<Vec<f64>>, initial: Vec<f64>) -> Self {
        let n = transition.len();
        let m = emission.first().map_or(0, Vec::len);
        Self {
            transition,
            emission,
            initial,
            n,
            m,
        }
    }

    /// Forward algorithm with scaling. Returns (alpha, `log_likelihood`).
    #[must_use]
    pub fn forward(&self, obs: &[usize]) -> (Vec<Vec<f64>>, f64) {
        let r = self.forward_full(obs);
        (r.alpha, r.log_likelihood)
    }

    /// Forward algorithm returning full result including scales for backward.
    #[allow(clippy::needless_range_loop)]
    #[must_use]
    pub fn forward_full(&self, obs: &[usize]) -> ForwardResult {
        let t_len = obs.len();
        let mut alpha = vec![vec![0.0; self.n]; t_len];
        let mut scales = vec![0.0; t_len];

        for j in 0..self.n {
            let ob0 = obs[0].min(self.m - 1);
            alpha[0][j] = self.initial[j] * self.emission[j][ob0];
        }
        scales[0] = alpha[0].iter().sum();
        if scales[0] > 0.0 {
            for x in &mut alpha[0] {
                *x /= scales[0];
            }
        }

        for t in 1..t_len {
            for j in 0..self.n {
                let mut sum = 0.0;
                for i in 0..self.n {
                    sum += alpha[t - 1][i] * self.transition[i][j];
                }
                let obt = obs[t].min(self.m - 1);
                alpha[t][j] = sum * self.emission[j][obt];
            }
            scales[t] = alpha[t].iter().sum();
            if scales[t] > 0.0 {
                for x in &mut alpha[t] {
                    *x /= scales[t];
                }
            }
        }

        let log_lik: f64 = scales.iter().map(|s| (s + LOG_EPS).ln()).sum();
        ForwardResult {
            alpha,
            scales,
            log_likelihood: log_lik,
        }
    }

    /// Backward algorithm. Requires scales from forward pass.
    #[allow(clippy::needless_range_loop)]
    #[must_use]
    pub fn backward(&self, obs: &[usize], scales: &[f64]) -> Vec<Vec<f64>> {
        let t_len = obs.len();
        let mut beta = vec![vec![0.0; self.n]; t_len];
        for j in 0..self.n {
            beta[t_len - 1][j] = 1.0;
        }

        for t in (0..t_len.saturating_sub(1)).rev() {
            let ob_next = obs[t + 1].min(self.m - 1);
            for i in 0..self.n {
                let mut sum = 0.0;
                for j in 0..self.n {
                    sum += self.transition[i][j] * self.emission[j][ob_next] * beta[t + 1][j];
                }
                if t + 1 < scales.len() && scales[t + 1] > 0.0 {
                    sum /= scales[t + 1];
                }
                beta[t][i] = sum;
            }
        }
        beta
    }

    /// Viterbi algorithm: most likely state sequence. Returns (path, `log_prob`).
    #[allow(clippy::needless_range_loop)]
    #[must_use]
    pub fn viterbi(&self, obs: &[usize]) -> (Vec<usize>, f64) {
        let t_len = obs.len();
        let log_a: Vec<Vec<f64>> = self
            .transition
            .iter()
            .map(|row| row.iter().map(|&x| (x + LOG_EPS).ln()).collect())
            .collect();
        let log_b: Vec<Vec<f64>> = self
            .emission
            .iter()
            .map(|row| row.iter().map(|&x| (x + LOG_EPS).ln()).collect())
            .collect();
        let log_pi: Vec<f64> = self.initial.iter().map(|&x| (x + LOG_EPS).ln()).collect();

        let mut delta = vec![vec![0.0; self.n]; t_len];
        let mut psi = vec![vec![0usize; self.n]; t_len];

        for j in 0..self.n {
            let ob0 = obs[0].min(self.m - 1);
            delta[0][j] = log_pi[j] + log_b[j][ob0];
        }

        for t in 1..t_len {
            let obt = obs[t].min(self.m - 1);
            for j in 0..self.n {
                let mut best = f64::NEG_INFINITY;
                let mut best_i = 0;
                for i in 0..self.n {
                    let v = delta[t - 1][i] + log_a[i][j];
                    if v > best {
                        best = v;
                        best_i = i;
                    }
                }
                psi[t][j] = best_i;
                delta[t][j] = best + log_b[j][obt];
            }
        }

        let mut path = vec![0; t_len];
        let mut best = f64::NEG_INFINITY;
        for j in 0..self.n {
            if delta[t_len - 1][j] > best {
                best = delta[t_len - 1][j];
                path[t_len - 1] = j;
            }
        }
        let log_prob = best;

        for t in (0..t_len.saturating_sub(1)).rev() {
            path[t] = psi[t + 1][path[t + 1]];
        }
        (path, log_prob)
    }

    /// Posterior `P(s_t=i | O)` via forward-backward.
    #[must_use]
    pub fn posterior(&self, obs: &[usize]) -> Vec<Vec<f64>> {
        let fwd = self.forward_full(obs);
        let beta = self.backward(obs, &fwd.scales);
        let mut gamma: Vec<Vec<f64>> = fwd
            .alpha
            .iter()
            .zip(beta.iter())
            .map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
            .collect();
        for row in &mut gamma {
            let sum: f64 = row.iter().sum();
            if sum > 0.0 {
                for x in row.iter_mut() {
                    *x /= sum;
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
        observations[0] = rng.categorical(&self.emission[states[0]]);

        for t in 1..length {
            states[t] = rng.categorical(&self.transition[states[t - 1]]);
            observations[t] = rng.categorical(&self.emission[states[t]]);
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
        let (alpha, _) = hmm.forward(&obs);
        for row in &alpha {
            let sum: f64 = row.iter().sum();
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
        for row in &gamma {
            let sum: f64 = row.iter().sum();
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
}
