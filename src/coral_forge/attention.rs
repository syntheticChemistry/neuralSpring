// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    reason = "attention mechanism requires Q/K/V weight parameters and index→f64 casts for softmax"
)]

use super::activation::softmax_rows;

/// Scaled dot-product attention scores: `Q @ K^T / sqrt(d_k)`.
///
/// `query`: `[B, H, Sq, D]`, `key`: `[B, H, Skv, D]`.
/// Returns `[B, H, Sq, Skv]`.
#[must_use]
pub fn sdpa_scores(
    query: &[f64],
    key: &[f64],
    batch: usize,
    heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
) -> Vec<f64> {
    let scale = (head_dim as f64).sqrt();
    let mut scores = vec![0.0; batch * heads * q_len * kv_len];

    for b in 0..batch {
        for h in 0..heads {
            for q in 0..q_len {
                let q_base = ((b * heads + h) * q_len + q) * head_dim;
                let q_row = &query[q_base..q_base + head_dim];
                for k in 0..kv_len {
                    let k_base = ((b * heads + h) * kv_len + k) * head_dim;
                    let dot: f64 = q_row
                        .iter()
                        .zip(&key[k_base..k_base + head_dim])
                        .map(|(&qi, &ki)| qi * ki)
                        .sum();
                    scores[((b * heads + h) * q_len + q) * kv_len + k] = dot / scale;
                }
            }
        }
    }
    scores
}

/// Weighted value summation: `output[b,h,q,d] = Σ_k weights[b,h,q,k] * V[b,h,k,d]`.
#[must_use]
pub fn attention_apply(
    weights: &[f64],
    value: &[f64],
    batch: usize,
    heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; batch * heads * q_len * head_dim];

    for b in 0..batch {
        for h in 0..heads {
            for q in 0..q_len {
                let w_base = ((b * heads + h) * q_len + q) * kv_len;
                let w_row = &weights[w_base..w_base + kv_len];
                let v_head_base = (b * heads + h) * kv_len * head_dim;
                for d in 0..head_dim {
                    let acc: f64 = w_row
                        .iter()
                        .enumerate()
                        .map(|(k, &w)| w * value[v_head_base + k * head_dim + d])
                        .sum();
                    out[((b * heads + h) * q_len + q) * head_dim + d] = acc;
                }
            }
        }
    }
    out
}

/// Full scaled dot-product attention: softmax(QK^T / √d) @ V.
#[must_use]
pub fn sdpa_full(
    query: &[f64],
    key: &[f64],
    value: &[f64],
    batch: usize,
    heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
) -> Vec<f64> {
    let scores = sdpa_scores(query, key, batch, heads, q_len, kv_len, head_dim);
    let n_rows = batch * heads * q_len;
    let weights = softmax_rows(&scores, n_rows, kv_len);
    attention_apply(&weights, value, batch, heads, q_len, kv_len, head_dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;

    const EPS: f64 = tolerances::FOLDING_EPS;

    #[test]
    fn sdpa_scores_scale() {
        let d = 4;
        let q = vec![1.0; d];
        let k = vec![1.0; d];
        let scores = sdpa_scores(&q, &k, 1, 1, 1, 1, d);
        let expected = d as f64 / (d as f64).sqrt();
        assert!(
            (scores[0] - expected).abs() < EPS,
            "score = d/√d = √d = {expected}, got {}",
            scores[0]
        );
    }

    #[test]
    fn sdpa_full_preserves_magnitude() {
        let d = 4;
        let q = vec![0.5; 2 * d];
        let k = vec![0.5; 2 * d];
        let v = vec![1.0; 2 * d];
        let out = sdpa_full(&q, &k, &v, 1, 1, 2, 2, d);
        for val in &out {
            assert!(
                val.is_finite() && (val - 1.0).abs() < crate::tolerances::SDPA_PASSTHROUGH,
                "uniform query/key → uniform weights → V passthrough"
            );
        }
    }
}
