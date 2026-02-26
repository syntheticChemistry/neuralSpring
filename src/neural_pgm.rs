// SPDX-License-Identifier: AGPL-3.0-or-later

//! Neural networks as probabilistic graphical models.
//!
//! baseCamp Sub-thesis 04: Neural Networks as PGMs.
//!
//! Decomposes trained neural networks into equivalent probabilistic
//! graphical models (tree-structured PGMs) whose structure reveals
//! the network's reasoning as conditional probability chains.
//!
//! ## Grounding papers
//!
//! - Li et al. (2023) "DNNs as Infinite Tree-Structured PGMs"
//! - Nabarro et al. (2024) "Learning in Deep Factor Graphs" (ICML)
//! - Conmy et al. (2023) "Towards Automated Circuit Discovery" (NeurIPS)
//!
//! ## Validated primitives
//!
//! - [`crate::hmm`] — forward/backward as belief propagation
//! - [`crate::eigh::eigh_householder_qr`] — spectral decomposition
//! - [`crate::anderson_localization::ipr`] — participation ratio

#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::needless_range_loop
)]

use crate::eigh::eigh_householder_qr;
use crate::primitives::LOG_GUARD;

/// Convert a weight matrix to a row-stochastic transition matrix.
///
/// Applies softmax normalization to each row, making each row a
/// valid conditional probability distribution. This is the PGM
/// interpretation of a DNN layer: row i represents P(output | input=i).
#[must_use]
pub fn weight_to_transition(weights: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
    let mut transition = vec![0.0; n_rows * n_cols];
    for i in 0..n_rows {
        let row_start = i * n_cols;
        let max = weights[row_start..row_start + n_cols]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for j in 0..n_cols {
            let exp_val = (weights[row_start + j] - max).exp();
            transition[row_start + j] = exp_val;
            sum += exp_val;
        }
        if sum > LOG_GUARD {
            for j in 0..n_cols {
                transition[row_start + j] /= sum;
            }
        }
    }
    transition
}

/// Belief propagation forward pass through a chain of transition matrices.
///
/// This is mathematically equivalent to the HMM forward algorithm.
/// Given an input distribution and a sequence of layer transition
/// matrices, computes the output distribution.
///
/// Returns per-layer output distributions.
#[must_use]
pub fn belief_propagation_chain(
    input_dist: &[f64],
    transition_matrices: &[&[f64]],
    layer_dims: &[usize],
) -> Vec<Vec<f64>> {
    barracuda::linalg::graph::belief_propagation_chain(input_dist, transition_matrices, layer_dims)
}

/// Compare PGM belief propagation output with neural network forward pass.
///
/// Returns the KL divergence D_KL(nn_output || pgm_output), measuring
/// how well the PGM approximation matches the neural network.
///
/// Lower KL divergence = better PGM approximation.
#[must_use]
pub fn pgm_nn_divergence(nn_output: &[f64], pgm_output: &[f64]) -> f64 {
    if nn_output.len() != pgm_output.len() || nn_output.is_empty() {
        return f64::INFINITY;
    }
    let mut kl = 0.0;
    for (i, &p) in nn_output.iter().enumerate() {
        let q = pgm_output[i].max(LOG_GUARD);
        if p > LOG_GUARD {
            kl += p * (p / q).ln();
        }
    }
    kl.max(0.0)
}

/// Spectral similarity between two weight matrices (layer comparison).
///
/// Computes the cosine similarity between sorted eigenvalue spectra
/// of the two symmetrized weight matrices. High similarity between
/// distant layers suggests "knowledge transfer" (introgression analog).
#[must_use]
pub fn layer_spectral_similarity(w1: &[f64], n1: usize, w2: &[f64], n2: usize) -> f64 {
    let s1 = symmetrize_square(w1, n1);
    let s2 = symmetrize_square(w2, n2);

    let decomp1 = eigh_householder_qr(&s1, n1);
    let decomp2 = eigh_householder_qr(&s2, n2);

    let mut ev1 = decomp1.eigenvalues;
    let mut ev2 = decomp2.eigenvalues;
    ev1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ev2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min_len = ev1.len().min(ev2.len());
    if min_len == 0 {
        return 0.0;
    }

    let ev1 = &ev1[..min_len];
    let ev2 = &ev2[..min_len];

    let dot: f64 = ev1.iter().zip(ev2.iter()).map(|(&a, &b)| a * b).sum();
    let norm1: f64 = ev1.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let norm2: f64 = ev2.iter().map(|&x| x * x).sum::<f64>().sqrt();

    if norm1 < LOG_GUARD || norm2 < LOG_GUARD {
        return 0.0;
    }
    (dot / (norm1 * norm2)).clamp(-1.0, 1.0)
}

fn symmetrize_square(m: &[f64], n: usize) -> Vec<f64> {
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            s[i * n + j] = f64::midpoint(m[i * n + j], m[j * n + i]);
        }
    }
    s
}

/// Effective rank of a weight matrix via eigenvalue entropy.
///
/// Uses the Shannon entropy of the normalized eigenvalue spectrum:
/// rank_eff = exp(H) where H = -sum(p_i * log(p_i)).
///
/// Full rank → rank_eff = n. Low rank → rank_eff << n.
/// Used for circuit discovery: layers with low effective rank
/// perform simple computations (single circuits).
///
/// Delegates to `barracuda::linalg::graph::effective_rank` (absorbed S54, H-009).
#[must_use]
pub fn effective_rank(eigenvalues: &[f64]) -> f64 {
    barracuda::linalg::effective_rank(eigenvalues)
}

/// PGM graph complexity: number of significant transition probabilities.
///
/// Counts transition matrix entries above `threshold`, normalized by
/// the total number of entries. Sparser PGMs = simpler models.
#[must_use]
pub fn pgm_complexity(transition_matrices: &[&[f64]], dims: &[usize], threshold: f64) -> f64 {
    let mut total_entries = 0usize;
    let mut significant = 0usize;

    let mut prev_dim = dims.first().copied().unwrap_or(0);
    for (idx, &trans) in transition_matrices.iter().enumerate() {
        let curr_dim = dims.get(idx + 1).copied().unwrap_or(prev_dim);
        let entries = prev_dim * curr_dim;
        for k in 0..entries.min(trans.len()) {
            total_entries += 1;
            if trans[k].abs() > threshold {
                significant += 1;
            }
        }
        prev_dim = curr_dim;
    }

    if total_entries == 0 {
        return 0.0;
    }
    significant as f64 / total_entries as f64
}

/// Run full PGM analysis on a neural network defined by weight matrices.
///
/// Converts weights to transitions, runs belief propagation, and
/// compares with direct forward pass. Returns PGM quality metrics.
#[must_use]
pub fn pgm_analysis(
    weight_matrices: &[&[f64]],
    layer_dims: &[usize],
    input_dist: &[f64],
    nn_output: &[f64],
) -> PgmAnalysisResult {
    let transitions: Vec<Vec<f64>> = weight_matrices
        .iter()
        .enumerate()
        .map(|(i, &w)| {
            let n_in = if i == 0 {
                input_dist.len()
            } else {
                layer_dims[i - 1]
            };
            let n_out = layer_dims[i];
            weight_to_transition(w, n_in, n_out)
        })
        .collect();

    let trans_refs: Vec<&[f64]> = transitions.iter().map(Vec::as_slice).collect();
    let bp_distributions = belief_propagation_chain(input_dist, &trans_refs, layer_dims);

    let pgm_output = bp_distributions.last().cloned().unwrap_or_default();
    let kl_div = pgm_nn_divergence(nn_output, &pgm_output);

    PgmAnalysisResult {
        pgm_output,
        kl_divergence: kl_div,
        per_layer_distributions: bp_distributions,
    }
}

/// Result of PGM analysis.
#[derive(Debug, Clone)]
pub struct PgmAnalysisResult {
    /// PGM belief propagation output distribution.
    pub pgm_output: Vec<f64>,
    /// KL divergence between NN output and PGM output.
    pub kl_divergence: f64,
    /// Distributions at each layer from belief propagation.
    pub per_layer_distributions: Vec<Vec<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn transition_rows_sum_to_one() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let trans = weight_to_transition(&weights, 2, 3);
        for i in 0..2 {
            let sum: f64 = (0..3).map(|j| trans[i * 3 + j]).sum();
            assert!(
                (sum - 1.0).abs() < tolerances::EXACT_F64,
                "row {i} sums to {sum}, not 1.0"
            );
        }
    }

    #[test]
    fn transition_all_positive() {
        let weights = vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0];
        let trans = weight_to_transition(&weights, 2, 3);
        assert!(
            trans.iter().all(|&v| v >= 0.0),
            "transition matrix should be non-negative"
        );
    }

    #[test]
    fn bp_preserves_normalization() {
        let input = vec![0.25, 0.25, 0.25, 0.25];
        let w1 = vec![
            1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        ];
        let t1 = weight_to_transition(&w1, 4, 4);
        let dists = belief_propagation_chain(&input, &[t1.as_slice()], &[4]);
        for dist in &dists {
            let sum: f64 = dist.iter().sum();
            assert!(
                (sum - 1.0).abs() < tolerances::CROSS_LANGUAGE,
                "distribution should sum to 1, got {sum}"
            );
        }
    }

    #[test]
    fn kl_divergence_zero_for_identical() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let kl = pgm_nn_divergence(&p, &p);
        assert!(
            kl.abs() < tolerances::EXACT_F64,
            "KL(p||p) should be 0, got {kl}"
        );
    }

    #[test]
    fn kl_divergence_positive() {
        let p = vec![0.5, 0.5];
        let q = vec![0.9, 0.1];
        let kl = pgm_nn_divergence(&p, &q);
        assert!(
            kl > 0.0,
            "KL should be positive for different distributions"
        );
    }

    #[test]
    fn effective_rank_of_identity() {
        let eigenvalues = vec![1.0; 8];
        let rank = effective_rank(&eigenvalues);
        assert!(
            (rank - 8.0).abs() < tolerances::CROSS_LANGUAGE,
            "identity-like spectrum should have full rank, got {rank}"
        );
    }

    #[test]
    fn effective_rank_of_single() {
        let mut eigenvalues = vec![0.0; 8];
        eigenvalues[0] = 1.0;
        let rank = effective_rank(&eigenvalues);
        assert!(
            (rank - 1.0).abs() < tolerances::CROSS_LANGUAGE,
            "single nonzero eigenvalue should give rank 1, got {rank}"
        );
    }

    #[test]
    fn spectral_similarity_self() {
        let w = vec![1.0, 0.5, 0.5, 1.0, 0.3, 0.7, 0.7, 0.3, 1.0];
        let sim = layer_spectral_similarity(&w, 3, &w, 3);
        assert!(
            (sim - 1.0).abs() < tolerances::SPECIAL_FUNCTION_F64,
            "self-similarity should be 1.0, got {sim}"
        );
    }

    #[test]
    fn determinism() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = weight_to_transition(&weights, 2, 3);
        let t2 = weight_to_transition(&weights, 2, 3);
        assert!(
            t1.iter()
                .zip(t2.iter())
                .all(|(a, b)| (a - b).abs() < f64::EPSILON),
            "determinism: bit-identical runs must match"
        );
    }

    #[test]
    fn kl_divergence_mismatched_lengths() {
        let p = vec![0.5, 0.5];
        let q = vec![1.0];
        assert!(pgm_nn_divergence(&p, &q).is_infinite());
    }

    #[test]
    fn kl_divergence_empty() {
        let p: Vec<f64> = vec![];
        assert!(pgm_nn_divergence(&p, &p).is_infinite());
    }

    #[test]
    fn transition_zero_row_no_panic() {
        let weights = vec![0.0; 6];
        let trans = weight_to_transition(&weights, 2, 3);
        for &v in &trans {
            assert!(v.is_finite(), "zero-weight rows must produce finite output");
        }
    }

    #[test]
    fn bp_chain_multi_layer() {
        let input = vec![0.5, 0.5];
        let w1 = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        let w2 = vec![0.3, 0.7, 0.6, 0.4, 0.5, 0.5];
        let t1 = weight_to_transition(&w1, 2, 3);
        let t2 = weight_to_transition(&w2, 3, 2);
        let dists = belief_propagation_chain(&input, &[t1.as_slice(), t2.as_slice()], &[3, 2]);
        assert_eq!(dists.len(), 3, "input + 2 layers = 3 distributions");
        for dist in &dists {
            let sum: f64 = dist.iter().sum();
            assert!(
                (sum - 1.0).abs() < tolerances::CROSS_LANGUAGE,
                "normalization at {sum}"
            );
        }
    }

    #[test]
    fn effective_rank_all_zeros() {
        let eigenvalues = vec![0.0; 8];
        let rank = effective_rank(&eigenvalues);
        assert!(
            rank.abs() < tolerances::ZERO_DETECTION,
            "all-zero eigenvalues → rank 0, got {rank}"
        );
    }

    #[test]
    fn effective_rank_two_equal() {
        let eigenvalues = vec![1.0, 1.0, 0.0, 0.0];
        let rank = effective_rank(&eigenvalues);
        assert!(
            (rank - 2.0).abs() < tolerances::CROSS_LANGUAGE,
            "two equal nonzero eigenvalues → rank 2, got {rank}"
        );
    }

    #[test]
    fn layer_spectral_similarity_different_sizes() {
        let w1 = vec![1.0, 0.5, 0.5, 1.0];
        let w2 = vec![1.0, 0.5, 0.3, 0.5, 1.0, 0.4, 0.3, 0.4, 1.0];
        let sim = layer_spectral_similarity(&w1, 2, &w2, 3);
        assert!(
            sim.is_finite(),
            "different-sized matrices should produce finite similarity"
        );
        assert!(
            (-1.0..=1.0).contains(&sim),
            "cosine similarity must be in [-1, 1], got {sim}"
        );
    }

    #[test]
    fn layer_spectral_similarity_zero_matrix() {
        let zeros = vec![0.0; 4];
        let sim = layer_spectral_similarity(&zeros, 2, &zeros, 2);
        assert!(
            sim.abs() < tolerances::ZERO_DETECTION,
            "zero matrices → 0 similarity, got {sim}"
        );
    }

    #[test]
    fn pgm_complexity_measures_sparsity() {
        let dense = vec![0.5; 12];
        let sparse = vec![
            0.001, 0.0, 0.0, 0.999, 0.0, 0.0, 0.001, 0.0, 0.0, 0.999, 0.0, 0.0,
        ];
        let dims = &[3, 4, 3];
        let c_dense = pgm_complexity(&[dense.as_slice()], dims, 0.1);
        let c_sparse = pgm_complexity(&[sparse.as_slice()], dims, 0.1);
        assert!(
            c_dense > c_sparse,
            "dense ({c_dense}) should have higher complexity than sparse ({c_sparse})"
        );
    }

    #[test]
    fn pgm_complexity_empty() {
        let c = pgm_complexity(&[], &[], 0.1);
        assert!(
            c.abs() < tolerances::ZERO_DETECTION,
            "no layers → 0 complexity, got {c}"
        );
    }

    #[test]
    fn pgm_analysis_round_trip() {
        let input = vec![0.5, 0.5];
        let w1 = vec![1.0, 0.0, 0.0, 1.0];
        let t1 = weight_to_transition(&w1, 2, 2);
        let nn_output = t1[0..2].to_vec();
        let result = pgm_analysis(&[w1.as_slice()], &[2], &input, &nn_output);
        assert!(
            result.kl_divergence.is_finite(),
            "KL divergence must be finite"
        );
        assert_eq!(
            result.per_layer_distributions.len(),
            2,
            "input + 1 layer = 2 distributions"
        );
    }

    #[test]
    fn pgm_analysis_kl_small_for_identity_like() {
        let input = vec![0.25, 0.25, 0.25, 0.25];
        let identity_like = vec![
            10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0,
        ];
        let result = pgm_analysis(&[identity_like.as_slice()], &[4], &input, &input);
        assert!(
            result.kl_divergence < tolerances::NORM_PPF_TAIL,
            "identity-like transition should give near-zero KL, got {}",
            result.kl_divergence
        );
    }
}
