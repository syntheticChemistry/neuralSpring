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
        if sum > 1e-300 {
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
    let mut distributions = Vec::with_capacity(transition_matrices.len() + 1);
    distributions.push(input_dist.to_vec());

    let mut current = input_dist.to_vec();
    for (layer_idx, &trans) in transition_matrices.iter().enumerate() {
        let n_in = current.len();
        let n_out = layer_dims[layer_idx];
        let mut next = vec![0.0; n_out];

        for j in 0..n_out {
            for i in 0..n_in {
                next[j] = current[i].mul_add(trans[i * n_out + j], next[j]);
            }
        }

        let sum: f64 = next.iter().sum();
        if sum > 1e-300 {
            for v in &mut next {
                *v /= sum;
            }
        }

        distributions.push(next.clone());
        current = next;
    }

    distributions
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
        let q = pgm_output[i].max(1e-300);
        if p > 1e-300 {
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

    if norm1 < 1e-300 || norm2 < 1e-300 {
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
#[must_use]
pub fn effective_rank(eigenvalues: &[f64]) -> f64 {
    let abs_vals: Vec<f64> = eigenvalues.iter().map(|&ev| ev.abs()).collect();
    let total: f64 = abs_vals.iter().sum();
    if total < 1e-300 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &v in &abs_vals {
        let p = v / total;
        if p > 1e-300 {
            entropy -= p * p.ln();
        }
    }
    entropy.exp()
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

    #[test]
    fn transition_rows_sum_to_one() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let trans = weight_to_transition(&weights, 2, 3);
        for i in 0..2 {
            let sum: f64 = (0..3).map(|j| trans[i * 3 + j]).sum();
            assert!((sum - 1.0).abs() < 1e-12, "row {i} sums to {sum}, not 1.0");
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
                (sum - 1.0).abs() < 1e-10,
                "distribution should sum to 1, got {sum}"
            );
        }
    }

    #[test]
    fn kl_divergence_zero_for_identical() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let kl = pgm_nn_divergence(&p, &p);
        assert!(kl.abs() < 1e-12, "KL(p||p) should be 0, got {kl}");
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
            (rank - 8.0).abs() < 1e-10,
            "identity-like spectrum should have full rank, got {rank}"
        );
    }

    #[test]
    fn effective_rank_of_single() {
        let mut eigenvalues = vec![0.0; 8];
        eigenvalues[0] = 1.0;
        let rank = effective_rank(&eigenvalues);
        assert!(
            (rank - 1.0).abs() < 1e-10,
            "single nonzero eigenvalue should give rank 1, got {rank}"
        );
    }

    #[test]
    fn spectral_similarity_self() {
        let w = vec![1.0, 0.5, 0.5, 1.0, 0.3, 0.7, 0.7, 0.3, 1.0];
        let sim = layer_spectral_similarity(&w, 3, &w, 3);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "self-similarity should be 1.0, got {sim}"
        );
    }

    #[test]
    fn determinism() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = weight_to_transition(&weights, 2, 3);
        let t2 = weight_to_transition(&weights, 2, 3);
        assert_eq!(t1, t2);
    }
}
