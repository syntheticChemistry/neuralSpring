// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(
    clippy::cast_precision_loss,
    reason = "activation index→f64 casts for normalization"
)]

/// GELU activation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
///
/// Delegates to [`primitives::gelu`](crate::primitives::gelu).
#[must_use]
pub fn gelu(x: f64) -> f64 {
    crate::primitives::gelu(x)
}

/// Vectorized GELU over a slice.
#[must_use]
pub fn gelu_vec(xs: &[f64]) -> Vec<f64> {
    xs.iter().copied().map(crate::primitives::gelu).collect()
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

    x.chunks_exact(dim)
        .flat_map(|row| {
            let mean = row.iter().sum::<f64>() / dim as f64;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / dim as f64;
            let inv_std = 1.0 / (var + eps).sqrt();
            row.iter()
                .zip(gamma.iter().zip(beta.iter()))
                .map(move |(&xd, (&g, &b))| g.mul_add((xd - mean) * inv_std, b))
        })
        .collect()
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
    x.chunks_exact(cols)
        .flat_map(|row| {
            let max_val = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = row.iter().map(|v| (v - max_val).exp()).sum();
            row.iter().map(move |&v| (v - max_val).exp() / sum_exp)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;

    const EPS: f64 = tolerances::FOLDING_EPS;

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
    fn gelu_vec_matches_scalar() {
        let xs = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 3.0];
        let vec_result = gelu_vec(&xs);
        for (x, g) in xs.iter().zip(vec_result.iter()) {
            assert!((gelu(*x) - g).abs() < EPS);
        }
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
}
