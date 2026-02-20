// SPDX-License-Identifier: AGPL-3.0-only

//! PhyloNet-HMM for introgression detection (Paper 018).
//!
//! Liu et al. (2015) "Interspecific Introgressive Origin of Genomic
//! Diversity in the House Mouse" PNAS 112:196-201.
//!
//! Port of `control/introgression/introgression.py`.
//! Uses [`crate::hmm::Hmm`] for forward, backward, and Viterbi.

#![allow(clippy::cast_precision_loss)]

use crate::hmm::Hmm;
use crate::rng::Rng;

/// Gene tree topology: concordant, introgression-like, or other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum GeneTreeTopology {
    /// ((B,C),A) — concordant with species tree (ILS).
    Concordant = 0,
    /// ((A,B),C) — introgression-like (A→B gene flow).
    IntrogressionLike = 1,
    /// ((A,C),B) — other.
    Other = 2,
}

impl GeneTreeTopology {
    /// Observation index for HMM emission matrix.
    #[must_use]
    pub const fn to_obs(&self) -> usize {
        *self as usize
    }
}

impl From<usize> for GeneTreeTopology {
    fn from(i: usize) -> Self {
        match i {
            0 => Self::Concordant,
            1 => Self::IntrogressionLike,
            _ => Self::Other,
        }
    }
}

/// PhyloNet-HMM parameters matching Python baseline.
/// 2 states: `ILS_only` (0), Introgression (1). 3 observations.
#[must_use]
pub fn phylonet_hmm() -> Hmm {
    let transition = vec![vec![0.98, 0.02], vec![0.05, 0.95]];
    let emission = vec![vec![0.70, 0.06, 0.24], vec![0.15, 0.75, 0.10]];
    let initial = vec![0.70, 0.30];
    Hmm::new(transition, emission, initial)
}

/// ILS-only model: single state, emission from ILS row only.
#[must_use]
pub fn ils_only_hmm() -> Hmm {
    let emission = vec![vec![0.70, 0.06, 0.24]];
    let transition = vec![vec![1.0]];
    let initial = vec![1.0];
    Hmm::new(transition, emission, initial)
}

/// HMM that stays in ILS forever (for generating no-introgression data).
#[must_use]
fn ils_only_generating_hmm() -> Hmm {
    let emission = vec![vec![0.70, 0.06, 0.24], vec![0.70, 0.06, 0.24]];
    let transition = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let initial = vec![1.0, 0.0];
    Hmm::new(transition, emission, initial)
}

/// Generate synthetic loci with no introgression (for FPR test).
pub fn generate_ils_only_loci(n_loci: usize, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
    ils_only_generating_hmm().generate_sequence(n_loci, rng)
}

/// Generate synthetic gene tree observations from PhyloNet-HMM.
pub fn generate_synthetic_loci(
    n_loci: usize,
    hmm: &Hmm,
    rng: &mut Rng,
) -> (Vec<usize>, Vec<usize>) {
    hmm.generate_sequence(n_loci, rng)
}

/// Log-likelihood ratio: 2 * (log `L_introg` - log `L_ils_only`).
/// Positive = introgression model preferred.
#[must_use]
pub fn log_likelihood_ratio(log_lik_introg: f64, log_lik_ils_only: f64) -> f64 {
    2.0 * (log_lik_introg - log_lik_ils_only)
}

/// Detect introgression regions via Viterbi decoding.
/// Returns (path, `log_prob`). `path[i]` = 0 for ILS, 1 for Introgression.
#[must_use]
pub fn detect_introgression(hmm: &Hmm, observations: &[usize]) -> (Vec<usize>, f64) {
    hmm.viterbi(observations)
}

/// Fraction of loci in introgression state.
#[must_use]
pub fn introgression_fraction(path: &[usize]) -> f64 {
    if path.is_empty() {
        return 0.0;
    }
    path.iter().filter(|&&s| s == 1).count() as f64 / path.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_to_obs() {
        assert_eq!(GeneTreeTopology::Concordant.to_obs(), 0);
        assert_eq!(GeneTreeTopology::IntrogressionLike.to_obs(), 1);
        assert_eq!(GeneTreeTopology::Other.to_obs(), 2);
    }

    #[test]
    fn phylonet_hmm_forward_finite() {
        let hmm = phylonet_hmm();
        let mut rng = Rng::new(42);
        let (_, obs) = generate_synthetic_loci(500, &hmm, &mut rng);
        let (_, log_lik) = hmm.forward(&obs);
        assert!(log_lik.is_finite());
    }

    #[test]
    fn viterbi_accuracy_above_chance() {
        let hmm = phylonet_hmm();
        let mut rng = Rng::new(42);
        let (true_states, obs) = generate_synthetic_loci(500, &hmm, &mut rng);
        let (path, _) = hmm.viterbi(&obs);
        let acc = path
            .iter()
            .zip(true_states.iter())
            .filter(|(a, b)| a == b)
            .count() as f64
            / path.len() as f64;
        assert!(acc > 0.55);
    }
}
