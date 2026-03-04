// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(
    clippy::cast_precision_loss,
    reason = "head_dim usize→f64 for attention scale sqrt"
)]

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
                let acc: f64 = (0..n)
                    .map(|k| {
                        proj_a[(i * n + k) * channels + c] * proj_b[(j * n + k) * channels + c]
                    })
                    .sum();
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
                let acc: f64 = (0..n)
                    .map(|k| {
                        proj_a[(k * n + i) * channels + c] * proj_b[(k * n + j) * channels + c]
                    })
                    .sum();
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

#[cfg(test)]
#[expect(
    clippy::many_single_char_names,
    reason = "r,n,h,d are standard attention dimension names"
)]
mod tests {
    use super::*;
    use crate::tolerances;

    const EPS: f64 = tolerances::FOLDING_EPS;

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
}
