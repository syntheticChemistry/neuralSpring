// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sovereign Folding: Evoformer primitive implementations for CPU validation.
//!
//! Phase B of the sovereign folding track. Provides pure Rust f64
//! reference implementations of `AlphaFold2`'s Evoformer operations:
//!
//! - [`gelu`] — GELU activation (Hendrycks & Gimpel 2016)
//! - [`layer_norm`] — Layer normalization (Ba et al. 2016)
//! - [`softmax_rows`] — Row-wise numerically stable softmax
//! - [`sdpa_scores`] — Scaled dot-product attention scores (QKᵀ/√d)
//! - [`attention_apply`] — Weighted value summation (weights × V)
//! - [`sdpa_full`] — Complete SDPA pipeline (scores → softmax → apply)
//! - [`triangle_mul_outgoing`] — Algorithm 11 (Jumper et al. 2021)
//! - [`triangle_mul_incoming`] — Algorithm 12
//! - [`triangle_attention_scores`] — Algorithms 13-14 with pair bias
//!
//! ## References
//!
//! - Jumper et al. "Highly accurate protein structure prediction with
//!   `AlphaFold`" Nature 596:583-589 (2021)
//! - Ahdritz et al. "`OpenFold`: Retraining `AlphaFold2` yields new insights
//!   into its learning mechanisms and capacity for generalization"
//!   Nature Methods (2024)
//!
//! ## Evolution path
//!
//! ```text
//! NumPy baseline → Rust CPU → WGSL shader → sovereign pipeline
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_arguments
)]

use std::f64::consts::PI;

/// GELU activation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
#[must_use]
pub fn gelu(x: f64) -> f64 {
    let sqrt_2_over_pi = (2.0 / PI).sqrt();
    let x3 = x * x * x;
    let inner = sqrt_2_over_pi * (0.044_715_f64).mul_add(x3, x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Vectorized GELU over a slice.
#[must_use]
pub fn gelu_vec(xs: &[f64]) -> Vec<f64> {
    xs.iter().copied().map(gelu).collect()
}

/// Layer normalization along the last axis.
///
/// `x`: `[rows, dim]`, `gamma`/`beta`: `[dim]`, `eps`: stability constant.
/// Returns `gamma * (x - mean) / sqrt(var + eps) + beta` for each row.
///
/// # Panics
///
/// Panics if slice lengths don't match the declared dimensions.
#[must_use]
pub fn layer_norm(
    x: &[f64],
    rows: usize,
    dim: usize,
    gamma: &[f64],
    beta: &[f64],
    eps: f64,
) -> Vec<f64> {
    assert_eq!(x.len(), rows * dim);
    assert_eq!(gamma.len(), dim);
    assert_eq!(beta.len(), dim);

    let mut out = vec![0.0; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f64>() / dim as f64;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / dim as f64;
        let inv_std = 1.0 / (var + eps).sqrt();
        for d in 0..dim {
            out[r * dim + d] = gamma[d].mul_add((row[d] - mean) * inv_std, beta[d]);
        }
    }
    out
}

/// Row-wise numerically stable softmax.
///
/// `x`: `[rows, cols]` in row-major. Returns same shape.
///
/// # Panics
///
/// Panics if `x.len() != rows * cols`.
#[must_use]
pub fn softmax_rows(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    assert_eq!(x.len(), rows * cols);
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let max_val = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = row.iter().map(|v| (v - max_val).exp()).sum();
        for c in 0..cols {
            out[r * cols + c] = (row[c] - max_val).exp() / sum_exp;
        }
    }
    out
}

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
                for k in 0..kv_len {
                    let mut dot = 0.0_f64;
                    for d in 0..head_dim {
                        let qi = query[((b * heads + h) * q_len + q) * head_dim + d];
                        let ki = key[((b * heads + h) * kv_len + k) * head_dim + d];
                        dot = qi.mul_add(ki, dot);
                    }
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
                for d in 0..head_dim {
                    let mut acc = 0.0_f64;
                    for k in 0..kv_len {
                        let w = weights[((b * heads + h) * q_len + q) * kv_len + k];
                        let v = value[((b * heads + h) * kv_len + k) * head_dim + d];
                        acc = w.mul_add(v, acc);
                    }
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

/// Triangle multiplicative update — outgoing edges (Algorithm 11).
///
/// `proj_a[i,k,c]`, `proj_b[j,k,c]` → `output[i,j,c] = Σ_k a[i,k,c] * b[j,k,c]`.
///
/// Pre-computed gated projections, row-major `[N, N, C]`.
///
/// # Panics
///
/// Panics if projection slices have length != `n * n * channels`.
#[must_use]
pub fn triangle_mul_outgoing(
    proj_a: &[f64],
    proj_b: &[f64],
    n: usize,
    channels: usize,
) -> Vec<f64> {
    assert_eq!(proj_a.len(), n * n * channels);
    assert_eq!(proj_b.len(), n * n * channels);

    let mut out = vec![0.0; n * n * channels];
    for i in 0..n {
        for j in 0..n {
            for c in 0..channels {
                let mut acc = 0.0_f64;
                for k in 0..n {
                    acc = proj_a[(i * n + k) * channels + c]
                        .mul_add(proj_b[(j * n + k) * channels + c], acc);
                }
                out[(i * n + j) * channels + c] = acc;
            }
        }
    }
    out
}

/// Triangle multiplicative update — incoming edges (Algorithm 12).
///
/// `proj_a[k,i,c]`, `proj_b[k,j,c]` → `output[i,j,c] = Σ_k a[k,i,c] * b[k,j,c]`.
///
/// # Panics
///
/// Panics if projection slices have length != `n * n * channels`.
#[must_use]
pub fn triangle_mul_incoming(
    proj_a: &[f64],
    proj_b: &[f64],
    n: usize,
    channels: usize,
) -> Vec<f64> {
    assert_eq!(proj_a.len(), n * n * channels);
    assert_eq!(proj_b.len(), n * n * channels);

    let mut out = vec![0.0; n * n * channels];
    for i in 0..n {
        for j in 0..n {
            for c in 0..channels {
                let mut acc = 0.0_f64;
                for k in 0..n {
                    acc = proj_a[(k * n + i) * channels + c]
                        .mul_add(proj_b[(k * n + j) * channels + c], acc);
                }
                out[(i * n + j) * channels + c] = acc;
            }
        }
    }
    out
}

/// Triangle attention scores with pair bias (Algorithms 13-14).
///
/// For each row `r`: `logit[r,h,j,k] = Σ_d Q[r,j,h,d]*K[r,k,h,d]/√D + bias[h,j,k]`
///
/// `query`/`key`: `[R, N, H, D]`, `bias`: `[H, N, N]`.
/// Returns `[R, H, N, N]`.
#[must_use]
pub fn triangle_attention_scores(
    query: &[f64],
    key: &[f64],
    bias: &[f64],
    n_rows: usize,
    n_res: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f64> {
    let scale = (head_dim as f64).sqrt();
    let mut scores = vec![0.0; n_rows * n_heads * n_res * n_res];

    for row in 0..n_rows {
        for h in 0..n_heads {
            for j in 0..n_res {
                for k in 0..n_res {
                    let mut dot = 0.0_f64;
                    for d in 0..head_dim {
                        let qi = query[((row * n_res + j) * n_heads + h) * head_dim + d];
                        let ki = key[((row * n_res + k) * n_heads + h) * head_dim + d];
                        dot = qi.mul_add(ki, dot);
                    }
                    let bias_val = bias[(h * n_res + j) * n_res + k];
                    scores[((row * n_heads + h) * n_res + j) * n_res + k] = dot / scale + bias_val;
                }
            }
        }
    }
    scores
}

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

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn gelu_at_zero() {
        assert!((gelu(0.0)).abs() < EPS);
    }

    #[test]
    fn gelu_positive_monotone() {
        let a = gelu(1.0);
        let b = gelu(2.0);
        assert!(b > a, "GELU should be monotonically increasing for x > 0");
    }

    #[test]
    fn gelu_large_positive_approximates_identity() {
        let x = 5.0;
        assert!((gelu(x) - x).abs() < 0.01, "GELU(x) ≈ x for large x");
    }

    #[test]
    fn gelu_negative_region() {
        let val = gelu(-1.0);
        assert!(val < 0.0 && val > -0.2, "GELU(-1) ≈ -0.159");
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let out = layer_norm(&x, 1, 4, &gamma, &beta, 1e-5);

        let mean: f64 = out.iter().sum::<f64>() / 4.0;
        assert!(
            mean.abs() < 1e-10,
            "post-norm mean should be ≈ 0, got {mean}"
        );

        let var: f64 = out.iter().map(|v| v * v).sum::<f64>() / 4.0;
        assert!(
            (var - 1.0).abs() < 1e-4,
            "post-norm var should be ≈ 1, got {var}"
        );
    }

    #[test]
    fn layer_norm_identity_with_unit_params() {
        let x = vec![0.0, 0.0, 0.0, 0.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let out = layer_norm(&x, 1, 4, &gamma, &beta, 1e-5);
        for v in &out {
            assert!(v.abs() < 1e-10, "norm of constant = 0");
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = softmax_rows(&x, 1, 4);
        let sum: f64 = out.iter().sum();
        assert!(
            (sum - 1.0).abs() < EPS,
            "softmax should sum to 1, got {sum}"
        );
    }

    #[test]
    fn softmax_monotone() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = softmax_rows(&x, 1, 4);
        for i in 0..3 {
            assert!(out[i] < out[i + 1], "softmax preserves ordering");
        }
    }

    #[test]
    fn softmax_uniform_input() {
        let x = vec![1.0; 8];
        let out = softmax_rows(&x, 1, 8);
        for v in &out {
            assert!((v - 0.125).abs() < EPS, "uniform input → uniform output");
        }
    }

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
        let q = vec![0.5; 1 * 1 * 2 * d];
        let k = vec![0.5; 1 * 1 * 2 * d];
        let v = vec![1.0; 1 * 1 * 2 * d];
        let out = sdpa_full(&q, &k, &v, 1, 1, 2, 2, d);
        for val in &out {
            assert!(
                val.is_finite() && (val - 1.0).abs() < 1e-6,
                "uniform query/key → uniform weights → V passthrough"
            );
        }
    }

    #[test]
    fn triangle_mul_outgoing_identity_channel() {
        let n = 3;
        let c = 1;
        let a = vec![1.0; n * n * c];
        let b = vec![1.0; n * n * c];
        let out = triangle_mul_outgoing(&a, &b, n, c);
        for val in &out {
            assert!(
                (*val - n as f64).abs() < EPS,
                "all-ones: sum_k 1*1 = N = {n}, got {val}"
            );
        }
    }

    #[test]
    fn triangle_mul_incoming_identity_channel() {
        let n = 3;
        let c = 1;
        let a = vec![1.0; n * n * c];
        let b = vec![1.0; n * n * c];
        let out = triangle_mul_incoming(&a, &b, n, c);
        for val in &out {
            assert!(
                (*val - n as f64).abs() < EPS,
                "all-ones: sum_k 1*1 = N = {n}, got {val}"
            );
        }
    }

    #[test]
    fn triangle_mul_outgoing_vs_incoming_transpose() {
        let n = 4;
        let c = 2;
        let a: Vec<f64> = (0..n * n * c).map(|i| i as f64 * 0.1).collect();
        let b: Vec<f64> = (0..n * n * c).map(|i| (i as f64 + 1.0) * 0.05).collect();

        let out_o = triangle_mul_outgoing(&a, &b, n, c);
        let out_i = triangle_mul_incoming(&a, &b, n, c);

        assert!(out_o != out_i, "outgoing ≠ incoming with same inputs");
        assert!(out_o.iter().all(|v| v.is_finite()));
        assert!(out_i.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn triangle_attention_scores_shape() {
        let (r, n, h, d) = (2, 3, 2, 4);
        let q = vec![0.1; r * n * h * d];
        let k = vec![0.1; r * n * h * d];
        let bias = vec![0.0; h * n * n];
        let scores = triangle_attention_scores(&q, &k, &bias, r, n, h, d);
        assert_eq!(scores.len(), r * h * n * n);
    }

    #[test]
    fn triangle_attention_scores_bias_adds() {
        let (r, n, h, d) = (1, 2, 1, 2);
        let q = vec![1.0; r * n * h * d];
        let k = vec![1.0; r * n * h * d];
        let bias_zero = vec![0.0; h * n * n];
        let bias_one = vec![1.0; h * n * n];

        let s0 = triangle_attention_scores(&q, &k, &bias_zero, r, n, h, d);
        let s1 = triangle_attention_scores(&q, &k, &bias_one, r, n, h, d);

        for (a, b) in s0.iter().zip(s1.iter()) {
            assert!(
                (b - a - 1.0).abs() < EPS,
                "bias of 1.0 shifts scores by 1.0"
            );
        }
    }

    #[test]
    fn gelu_vec_matches_scalar() {
        let xs = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 3.0];
        let vec_result = gelu_vec(&xs);
        for (x, g) in xs.iter().zip(vec_result.iter()) {
            assert!((gelu(*x) - g).abs() < EPS);
        }
    }

    // ── Outer product mean ──────────────────────────────────────

    #[test]
    fn opm_single_sequence_is_outer_product() {
        let (n_seq, n_res, c_a, c_b) = (1, 3, 2, 2);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let out = outer_product_mean(&a, &b, n_seq, n_res, c_a, c_b);
        assert_eq!(out.len(), n_res * n_res * c_a * c_b);
        let val = out[0 * n_res * (c_a * c_b) + 0 * (c_a * c_b) + 0];
        assert!(
            (val - 1.0 * 0.1).abs() < EPS,
            "a[0,0]*b[0,0] = 0.1, got {val}"
        );
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

    // ── MSA row attention ───────────────────────────────────────

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
            .map(|i| (i as f64 * 0.01) - 0.5)
            .collect();
        let k: Vec<f64> = (0..s * n * h * d)
            .map(|i| (i as f64 * 0.02) - 0.3)
            .collect();
        let v: Vec<f64> = (0..s * n * h * d)
            .map(|i| (i as f64 * 0.03) - 0.1)
            .collect();
        let bias = vec![0.0; h * n * n];
        let out = msa_row_attention(&q, &k, &v, &bias, s, n, h, d);
        assert_eq!(out.len(), s * n * h * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── MSA column attention ────────────────────────────────────

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
            .map(|i| (i as f64 * 0.01) - 0.5)
            .collect();
        let k: Vec<f64> = (0..s * n * h * d)
            .map(|i| (i as f64 * 0.02) - 0.3)
            .collect();
        let v: Vec<f64> = (0..s * n * h * d)
            .map(|i| (i as f64 * 0.03) - 0.1)
            .collect();
        let out = msa_col_attention(&q, &k, &v, s, n, h, d);
        assert_eq!(out.len(), s * n * h * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
