// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_arguments
)]

use super::activation::softmax_rows;

/// Outer product mean: MSA → pair representation update.
///
/// Averages the outer product of projected MSA representations over sequences.
///
/// `a`: `[N_seq, N_res, C_a]`, `b`: `[N_seq, N_res, C_b]`.
/// Returns `[N_res, N_res, C_a * C_b]`.
///
/// # Panics
///
/// Panics if slice lengths don't match declared dimensions.
#[must_use]
pub fn outer_product_mean(
    a: &[f64],
    b: &[f64],
    n_seq: usize,
    n_res: usize,
    c_a: usize,
    c_b: usize,
) -> Vec<f64> {
    assert_eq!(a.len(), n_seq * n_res * c_a);
    assert_eq!(b.len(), n_seq * n_res * c_b);

    let c_out = c_a * c_b;
    let mut out = vec![0.0; n_res * n_res * c_out];

    for s in 0..n_seq {
        for i in 0..n_res {
            for j in 0..n_res {
                for ca in 0..c_a {
                    let a_val = a[(s * n_res + i) * c_a + ca];
                    for cb in 0..c_b {
                        let b_val = b[(s * n_res + j) * c_b + cb];
                        out[(i * n_res + j) * c_out + ca * c_b + cb] =
                            a_val.mul_add(b_val, out[(i * n_res + j) * c_out + ca * c_b + cb]);
                    }
                }
            }
        }
    }

    let inv_n = 1.0 / n_seq as f64;
    for v in &mut out {
        *v *= inv_n;
    }
    out
}

/// MSA row attention scores with pair bias.
///
/// For each sequence, computes attention scores over residue positions
/// with additive pair bias from the pair representation.
///
/// `query`/`key`: `[N_seq, N_res, H, D]`, `pair_bias`: `[H, N_res, N_res]`.
/// Returns scores `[N_seq, H, N_res, N_res]`.
///
/// # Panics
///
/// Panics if slice lengths don't match declared dimensions.
#[must_use]
pub fn msa_row_attention_scores(
    query: &[f64],
    key: &[f64],
    pair_bias: &[f64],
    n_seq: usize,
    n_res: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f64> {
    let qk_len = n_seq * n_res * n_heads * head_dim;
    assert_eq!(query.len(), qk_len);
    assert_eq!(key.len(), qk_len);
    assert_eq!(pair_bias.len(), n_heads * n_res * n_res);

    let scale = (head_dim as f64).sqrt();
    let mut scores = vec![0.0; n_seq * n_heads * n_res * n_res];

    for s in 0..n_seq {
        for h in 0..n_heads {
            for i in 0..n_res {
                for j in 0..n_res {
                    let mut dot = 0.0_f64;
                    for d in 0..head_dim {
                        let qi = query[((s * n_res + i) * n_heads + h) * head_dim + d];
                        let ki = key[((s * n_res + j) * n_heads + h) * head_dim + d];
                        dot = qi.mul_add(ki, dot);
                    }
                    let bias = pair_bias[(h * n_res + i) * n_res + j];
                    scores[((s * n_heads + h) * n_res + i) * n_res + j] = dot / scale + bias;
                }
            }
        }
    }
    scores
}

/// Full MSA row attention: scores with pair bias → softmax → weighted sum.
///
/// `query`/`key`/`value`: `[N_seq, N_res, H, D]`, `pair_bias`: `[H, N_res, N_res]`.
/// Returns `[N_seq, N_res, H, D]`.
///
/// # Panics
///
/// Panics if slice lengths don't match declared dimensions.
#[must_use]
pub fn msa_row_attention(
    query: &[f64],
    key: &[f64],
    value: &[f64],
    pair_bias: &[f64],
    n_seq: usize,
    n_res: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f64> {
    let scores = msa_row_attention_scores(query, key, pair_bias, n_seq, n_res, n_heads, head_dim);
    let n_rows = n_seq * n_heads * n_res;
    let weights = softmax_rows(&scores, n_rows, n_res);

    let mut out = vec![0.0; n_seq * n_res * n_heads * head_dim];
    for s in 0..n_seq {
        for h in 0..n_heads {
            for i in 0..n_res {
                for d in 0..head_dim {
                    let mut acc = 0.0_f64;
                    for j in 0..n_res {
                        let w = weights[((s * n_heads + h) * n_res + i) * n_res + j];
                        let v = value[((s * n_res + j) * n_heads + h) * head_dim + d];
                        acc = w.mul_add(v, acc);
                    }
                    out[((s * n_res + i) * n_heads + h) * head_dim + d] = acc;
                }
            }
        }
    }
    out
}

/// MSA column attention scores (no pair bias).
///
/// For each residue position, computes attention scores across MSA sequences.
///
/// `query`/`key`: `[N_seq, N_res, H, D]`.
/// Returns scores `[N_res, H, N_seq, N_seq]`.
///
/// # Panics
///
/// Panics if slice lengths don't match declared dimensions.
#[must_use]
pub fn msa_col_attention_scores(
    query: &[f64],
    key: &[f64],
    n_seq: usize,
    n_res: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f64> {
    let qk_len = n_seq * n_res * n_heads * head_dim;
    assert_eq!(query.len(), qk_len);
    assert_eq!(key.len(), qk_len);

    let scale = (head_dim as f64).sqrt();
    let mut scores = vec![0.0; n_res * n_heads * n_seq * n_seq];

    for r in 0..n_res {
        for h in 0..n_heads {
            for si in 0..n_seq {
                for sj in 0..n_seq {
                    let mut dot = 0.0_f64;
                    for d in 0..head_dim {
                        let qi = query[((si * n_res + r) * n_heads + h) * head_dim + d];
                        let ki = key[((sj * n_res + r) * n_heads + h) * head_dim + d];
                        dot = qi.mul_add(ki, dot);
                    }
                    scores[((r * n_heads + h) * n_seq + si) * n_seq + sj] = dot / scale;
                }
            }
        }
    }
    scores
}

/// Full MSA column attention: scores → softmax → weighted sum.
///
/// `query`/`key`/`value`: `[N_seq, N_res, H, D]`.
/// Returns `[N_seq, N_res, H, D]`.
///
/// # Panics
///
/// Panics if slice lengths don't match declared dimensions.
#[must_use]
pub fn msa_col_attention(
    query: &[f64],
    key: &[f64],
    value: &[f64],
    n_seq: usize,
    n_res: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f64> {
    let scores = msa_col_attention_scores(query, key, n_seq, n_res, n_heads, head_dim);
    let n_rows = n_res * n_heads * n_seq;
    let weights = softmax_rows(&scores, n_rows, n_seq);

    let mut out = vec![0.0; n_seq * n_res * n_heads * head_dim];
    for r in 0..n_res {
        for h in 0..n_heads {
            for si in 0..n_seq {
                for d in 0..head_dim {
                    let mut acc = 0.0_f64;
                    for sj in 0..n_seq {
                        let w = weights[((r * n_heads + h) * n_seq + si) * n_seq + sj];
                        let v = value[((sj * n_res + r) * n_heads + h) * head_dim + d];
                        acc = w.mul_add(v, acc);
                    }
                    out[((si * n_res + r) * n_heads + h) * head_dim + d] = acc;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn opm_single_sequence_is_outer_product() {
        let (n_seq, n_res, c_a, c_b) = (1, 3, 2, 2);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let out = outer_product_mean(&a, &b, n_seq, n_res, c_a, c_b);
        assert_eq!(out.len(), n_res * n_res * c_a * c_b);
        let val = out[0];
        assert!((val - 0.1).abs() < EPS, "a[0,0]*b[0,0] = 0.1, got {val}");
    }

    #[test]
    fn opm_mean_over_sequences() {
        let (n_seq, n_res, c_a, c_b) = (2, 2, 1, 1);
        let a = vec![1.0, 2.0, 3.0, 4.0]; // s=0: [1,2], s=1: [3,4]
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let out = outer_product_mean(&a, &b, n_seq, n_res, c_a, c_b);
        // opm[0,0] = mean(1*1, 3*1) = 2.0
        assert!((out[0] - 2.0).abs() < EPS, "mean(1,3) = 2, got {}", out[0]);
        // opm[1,0] = mean(2*1, 4*1) = 3.0
        assert!((out[2] - 3.0).abs() < EPS, "mean(2,4) = 3, got {}", out[2]);
    }

    #[test]
    fn opm_shape() {
        let (n_seq, n_res, c_a, c_b) = (4, 3, 2, 3);
        let a = vec![0.5; n_seq * n_res * c_a];
        let b = vec![0.5; n_seq * n_res * c_b];
        let out = outer_product_mean(&a, &b, n_seq, n_res, c_a, c_b);
        assert_eq!(out.len(), n_res * n_res * c_a * c_b);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn msa_row_scores_shape() {
        let (s, n, h, d) = (4, 6, 2, 4);
        let q = vec![0.1; s * n * h * d];
        let k = vec![0.1; s * n * h * d];
        let bias = vec![0.0; h * n * n];
        let scores = msa_row_attention_scores(&q, &k, &bias, s, n, h, d);
        assert_eq!(scores.len(), s * h * n * n);
    }

    #[test]
    fn msa_row_bias_shifts_scores() {
        let (s, n, h, d) = (2, 3, 1, 2);
        let q = vec![1.0; s * n * h * d];
        let k = vec![1.0; s * n * h * d];
        let bias_zero = vec![0.0; h * n * n];
        let bias_one = vec![1.0; h * n * n];
        let s0 = msa_row_attention_scores(&q, &k, &bias_zero, s, n, h, d);
        let s1 = msa_row_attention_scores(&q, &k, &bias_one, s, n, h, d);
        for (a, b) in s0.iter().zip(s1.iter()) {
            assert!((b - a - 1.0).abs() < EPS, "bias +1 should shift by 1");
        }
    }

    #[test]
    fn msa_row_attention_output_finite() {
        let (s, n, h, d) = (2, 4, 2, 4);
        let q: Vec<f64> = (0..s * n * h * d)
            .map(|i| (i as f64).mul_add(0.01, -0.5))
            .collect();
        let k: Vec<f64> = (0..s * n * h * d)
            .map(|i| (i as f64).mul_add(0.02, -0.3))
            .collect();
        let v: Vec<f64> = (0..s * n * h * d)
            .map(|i| (i as f64).mul_add(0.03, -0.1))
            .collect();
        let bias = vec![0.0; h * n * n];
        let out = msa_row_attention(&q, &k, &v, &bias, s, n, h, d);
        assert_eq!(out.len(), s * n * h * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn msa_col_scores_shape() {
        let (s, n, h, d) = (4, 6, 2, 4);
        let q = vec![0.1; s * n * h * d];
        let k = vec![0.1; s * n * h * d];
        let scores = msa_col_attention_scores(&q, &k, s, n, h, d);
        assert_eq!(scores.len(), n * h * s * s);
    }

    #[test]
    fn msa_col_uniform_gives_equal_weights() {
        let (s, n, h, d) = (4, 2, 1, 2);
        let q = vec![1.0; s * n * h * d];
        let k = vec![1.0; s * n * h * d];
        let scores = msa_col_attention_scores(&q, &k, s, n, h, d);
        let first_row = &scores[0..s];
        for val in first_row {
            assert!(
                (val - first_row[0]).abs() < EPS,
                "uniform QK → uniform scores"
            );
        }
    }

    #[test]
    fn msa_col_attention_output_finite() {
        let (s, n, h, d) = (3, 4, 2, 4);
        let q: Vec<f64> = (0..s * n * h * d)
            .map(|i| (i as f64).mul_add(0.01, -0.5))
            .collect();
        let k: Vec<f64> = (0..s * n * h * d)
            .map(|i| (i as f64).mul_add(0.02, -0.3))
            .collect();
        let v: Vec<f64> = (0..s * n * h * d)
            .map(|i| (i as f64).mul_add(0.03, -0.1))
            .collect();
        let out = msa_col_attention(&q, &k, &v, s, n, h, d);
        assert_eq!(out.len(), s * n * h * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
