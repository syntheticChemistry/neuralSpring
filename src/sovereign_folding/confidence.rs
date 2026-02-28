// SPDX-License-Identifier: AGPL-3.0-or-later

//! `AlphaFold3` confidence heads: pLDDT, PAE, pDE, and ranking score.
//!
//! All heads operate on pair or single representations produced by the
//! Pairformer, using linear projections followed by sigmoid/softmax to
//! produce calibrated confidence estimates.
//!
//! Reference: Abramson et al. "Accurate structure prediction for all
//! molecules" Nature 630:493-500 (2024), Supplementary §5.9.

/// `pLDDT` confidence head: `Linear → sigmoid → [0, 1]`.
///
/// `single_repr`: `[n_res, d]`, `w`: `[d]`, `b`: scalar.
/// Returns per-residue confidence values in `[0, 1]`.
///
/// # Panics
///
/// Panics if `single_repr.len() != n_res * d` or `w.len() != d`.
#[must_use]
pub fn plddt_head(single_repr: &[f64], n_res: usize, d: usize, w: &[f64], b: f64) -> Vec<f64> {
    assert_eq!(single_repr.len(), n_res * d);
    assert_eq!(w.len(), d);

    single_repr
        .chunks_exact(d)
        .map(|row| {
            let logit: f64 = row.iter().zip(w).map(|(x, wi)| x * wi).sum::<f64>() + b;
            1.0 / (1.0 + (-logit).exp())
        })
        .collect()
}

/// PAE confidence head: pair representation → Linear → softmax → expected distance.
///
/// `pair_repr`: `[n * n * d]`, `w`: `[d, n_bins]`, `b`: `[n_bins]`.
/// Bin centers are linearly spaced from 0 to 31.75 Å.
/// Returns `(expected_distance: [n*n], probabilities: [n*n*n_bins])`.
///
/// # Panics
///
/// Panics if dimensions are inconsistent.
#[must_use]
pub fn pae_head(
    pair_repr: &[f64],
    n: usize,
    d: usize,
    w: &[f64],
    b: &[f64],
    n_bins: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_pairs = n * n;
    assert_eq!(pair_repr.len(), n_pairs * d);
    assert_eq!(w.len(), d * n_bins);
    assert_eq!(b.len(), n_bins);

    let bin_centers: Vec<f64> = (0..n_bins)
        .map(|i| 31.75 * (i as f64) / ((n_bins - 1) as f64))
        .collect();

    let mut expected = Vec::with_capacity(n_pairs);
    let mut probs = Vec::with_capacity(n_pairs * n_bins);

    for row_chunk in pair_repr.chunks_exact(d) {
        let logits: Vec<f64> = (0..n_bins)
            .map(|j| {
                row_chunk
                    .iter()
                    .enumerate()
                    .map(|(k, &x)| x * w[k * n_bins + j])
                    .sum::<f64>()
                    + b[j]
            })
            .collect();

        let (row_probs, exp_dist) = softmax_expected(&logits, &bin_centers);
        probs.extend_from_slice(&row_probs);
        expected.push(exp_dist);
    }

    (expected, probs)
}

/// pDE confidence head: pair → Linear → softmax → predicted distance error.
///
/// Similar structure to PAE but predicts absolute distance error rather than
/// alignment error. Bin centers span 0 to `max_dist` Å.
///
/// Returns `(expected_error: [n*n], probabilities: [n*n*n_bins])`.
///
/// # Panics
///
/// Panics if dimensions are inconsistent.
#[must_use]
pub fn pde_head(
    pair_repr: &[f64],
    n: usize,
    d: usize,
    w: &[f64],
    b: &[f64],
    n_bins: usize,
    max_dist: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_pairs = n * n;
    assert_eq!(pair_repr.len(), n_pairs * d);
    assert_eq!(w.len(), d * n_bins);
    assert_eq!(b.len(), n_bins);

    let bin_centers: Vec<f64> = (0..n_bins)
        .map(|i| max_dist * (i as f64) / ((n_bins - 1) as f64))
        .collect();

    let mut expected = Vec::with_capacity(n_pairs);
    let mut probs = Vec::with_capacity(n_pairs * n_bins);

    for row_chunk in pair_repr.chunks_exact(d) {
        let logits: Vec<f64> = (0..n_bins)
            .map(|j| {
                row_chunk
                    .iter()
                    .enumerate()
                    .map(|(k, &x)| x * w[k * n_bins + j])
                    .sum::<f64>()
                    + b[j]
            })
            .collect();

        let (row_probs, exp_err) = softmax_expected(&logits, &bin_centers);
        probs.extend_from_slice(&row_probs);
        expected.push(exp_err);
    }

    (expected, probs)
}

/// Ranking score: weighted combination of confidence metrics.
///
/// Computes a single scalar ranking score for a predicted structure:
/// `score = w_plddt * mean(plddt) + w_pae * (1 - mean(pae)/max_pae)
///        + w_pde * (1 - mean(pde)/max_pde)`
///
/// Higher score = better predicted quality.
#[must_use]
pub fn ranking_score(
    plddt: &[f64],
    pae_expected: &[f64],
    pde_expected: &[f64],
    weights: &RankingWeights,
) -> f64 {
    let mean_plddt = if plddt.is_empty() {
        0.0
    } else {
        plddt.iter().sum::<f64>() / plddt.len() as f64
    };

    let mean_pae = if pae_expected.is_empty() {
        0.0
    } else {
        pae_expected.iter().sum::<f64>() / pae_expected.len() as f64
    };

    let mean_pde = if pde_expected.is_empty() {
        0.0
    } else {
        pde_expected.iter().sum::<f64>() / pde_expected.len() as f64
    };

    let pae_score = (1.0 - mean_pae / weights.max_pae).max(0.0);
    let pde_score = (1.0 - mean_pde / weights.max_pde).max(0.0);

    weights
        .w_pde
        .mul_add(pde_score, weights.w_plddt.mul_add(mean_plddt, weights.w_pae * pae_score))
}

/// Weights for the ranking score computation.
pub struct RankingWeights {
    pub w_plddt: f64,
    pub w_pae: f64,
    pub w_pde: f64,
    pub max_pae: f64,
    pub max_pde: f64,
}

impl Default for RankingWeights {
    fn default() -> Self {
        Self {
            w_plddt: 0.5,
            w_pae: 0.3,
            w_pde: 0.2,
            max_pae: 31.75,
            max_pde: 30.0,
        }
    }
}

/// Shared helper: softmax over logits, then compute expected value via bin centers.
fn softmax_expected(logits: &[f64], bin_centers: &[f64]) -> (Vec<f64>, f64) {
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f64 = exps.iter().sum();

    let mut exp_val = 0.0_f64;
    let probs: Vec<f64> = exps
        .iter()
        .zip(bin_centers)
        .map(|(&e, &c)| {
            let p = e / sum;
            exp_val += p * c;
            p
        })
        .collect();

    (probs, exp_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plddt_in_range() {
        let repr = vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0, 0.0];
        let w = vec![0.1, -0.1, 0.2, -0.2];
        let vals = plddt_head(&repr, 2, 4, &w, 0.0);
        assert_eq!(vals.len(), 2);
        for v in &vals {
            assert!(*v >= 0.0 && *v <= 1.0);
        }
    }

    #[test]
    fn pae_probs_sum_to_one() {
        let repr = vec![0.1; 4 * 4 * 2];
        let w = vec![0.1; 2 * 8];
        let b = vec![0.0; 8];
        let (_, probs) = pae_head(&repr, 4, 2, &w, &b, 8);
        for row in probs.chunks_exact(8) {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn pde_probs_sum_to_one() {
        let repr = vec![0.1; 4 * 4 * 2];
        let w = vec![0.1; 2 * 6];
        let b = vec![0.0; 6];
        let (_, probs) = pde_head(&repr, 4, 2, &w, &b, 6, 30.0);
        for row in probs.chunks_exact(6) {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn pde_expected_non_negative() {
        let repr = vec![0.1; 3 * 3 * 4];
        let w = vec![0.05; 4 * 10];
        let b = vec![0.0; 10];
        let (expected, _) = pde_head(&repr, 3, 4, &w, &b, 10, 30.0);
        for &e in &expected {
            assert!(e >= 0.0, "expected distance must be non-negative");
        }
    }

    #[test]
    fn ranking_score_perfect_structure() {
        let plddt = vec![1.0; 10];
        let pae = vec![0.0; 100];
        let pde = vec![0.0; 100];
        let score = ranking_score(&plddt, &pae, &pde, &RankingWeights::default());
        assert!((score - 1.0).abs() < 1e-10, "perfect structure → score 1.0");
    }

    #[test]
    fn ranking_score_worst_structure() {
        let plddt = vec![0.0; 10];
        let pae = vec![31.75; 100]; // max PAE
        let pde = vec![30.0; 100]; // max pDE
        let score = ranking_score(&plddt, &pae, &pde, &RankingWeights::default());
        assert!(score.abs() < 1e-10, "worst structure → score ~0");
    }

    #[test]
    fn ranking_score_weights_respected() {
        let plddt = vec![0.8; 5];
        let pae = vec![5.0; 25];
        let pde = vec![3.0; 25];
        let w = RankingWeights {
            w_plddt: 1.0,
            w_pae: 0.0,
            w_pde: 0.0,
            max_pae: 31.75,
            max_pde: 30.0,
        };
        let score = ranking_score(&plddt, &pae, &pde, &w);
        assert!((score - 0.8).abs() < 1e-10, "pLDDT-only weight");
    }
}
