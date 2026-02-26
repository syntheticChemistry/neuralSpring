// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::suboptimal_flops
)]

//! `DeepONet` operator learning primitives for antiderivative computation.
//!
//! Port of `control/deeponet/deeponet_antideriv.py` (Study 002).
//!
//! Reproduces key results from:
//! Lu, Jin, Pang, Zhang, Karniadakis (2021)
//! "Learning nonlinear operators via `DeepONet` based on the universal
//!  approximation theorem of operators"
//! Nature Machine Intelligence, Vol 3, pp 218-229.
//!
//! ## Pure math components
//!
//! - Polynomial function generation + exact antiderivative
//! - Branch-trunk MLP inference (dot-product coupling)
//! - Operator error metrics (L2 relative, RMSE)
//!
//! ## `BarraCUDA` connection
//!
//! - Branch MLP: `barracuda::ops::batch_gemm` (sensor encoding)
//! - Trunk MLP: `barracuda::ops::batch_gemm` (location encoding)
//! - Dot product: `barracuda::ops::elementwise_mul` + `sum_reduce`
//! - Batch evaluation: embarrassingly parallel over input functions
//!
//! ## Isomorphic pattern
//!
//! ```text
//! Branch net ≈ Encoder (BERT backbone, ResNet feature extractor)
//! Trunk net  ≈ Decoder query (transformer Q projection)
//! Dot product ≈ Attention score computation
//! DeepONet IS attention between functions and locations
//! ```

use crate::rng::Rng;

/// Evaluate a polynomial u(x) = Σ aₖ xᵏ at a set of points.
#[must_use]
pub fn eval_polynomial(coeffs: &[f64], x_points: &[f64]) -> Vec<f64> {
    x_points
        .iter()
        .map(|&x| {
            let mut val = 0.0;
            let mut x_pow = 1.0;
            for &a in coeffs {
                val = a.mul_add(x_pow, val);
                x_pow *= x;
            }
            val
        })
        .collect()
}

/// Exact antiderivative G(u)(y) = Σ aₖ/(k+1) yᵏ⁺¹ at query points.
#[must_use]
pub fn exact_antiderivative(coeffs: &[f64], y_points: &[f64]) -> Vec<f64> {
    y_points
        .iter()
        .map(|&y| {
            let mut val = 0.0;
            let mut y_pow = y;
            for (k, &a) in coeffs.iter().enumerate() {
                val = (a / (k + 1) as f64).mul_add(y_pow, val);
                y_pow *= y;
            }
            val
        })
        .collect()
}

/// Generate random polynomial and its antiderivative at given points.
///
/// Returns (`u_at_sensors`, `g_at_outputs`, coefficients).
#[must_use]
pub fn random_polynomial_pair(
    rng: &mut Rng,
    max_degree: usize,
    x_sensors: &[f64],
    y_outputs: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let degree = 1 + rng.usize(max_degree);
    let coeffs: Vec<f64> = (0..degree).map(|k| rng.normal() / (k + 1) as f64).collect();
    let u_vals = eval_polynomial(&coeffs, x_sensors);
    let g_vals = exact_antiderivative(&coeffs, y_outputs);
    (u_vals, g_vals, coeffs)
}

/// Generate a dataset of random polynomials and antiderivatives.
///
/// Returns (`U_sensors`, `G_outputs`) both row-major (`n_funcs` × `n_points`).
#[must_use]
pub fn generate_dataset(
    n_funcs: usize,
    x_sensors: &[f64],
    y_outputs: &[f64],
    max_degree: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let n_sensors = x_sensors.len();
    let n_outputs = y_outputs.len();
    let mut u_flat = Vec::with_capacity(n_funcs * n_sensors);
    let mut g_flat = Vec::with_capacity(n_funcs * n_outputs);

    for _ in 0..n_funcs {
        let (u, g, _) = random_polynomial_pair(&mut rng, max_degree, x_sensors, y_outputs);
        u_flat.extend_from_slice(&u);
        g_flat.extend_from_slice(&g);
    }
    (u_flat, g_flat)
}

/// Dense layer: out = tanh(W·x + b).
///
/// `weights`: row-major (`out_dim` × `in_dim`)
#[must_use]
pub fn dense_tanh(weights: &[f64], bias: &[f64], input: &[f64], out_dim: usize) -> Vec<f64> {
    let in_dim = input.len();
    (0..out_dim)
        .map(|i| {
            let mut sum = bias[i];
            for j in 0..in_dim {
                sum = weights[i * in_dim + j].mul_add(input[j], sum);
            }
            sum.tanh()
        })
        .collect()
}

/// Dense layer: out = W·x + b (no activation).
#[must_use]
pub fn dense_linear(weights: &[f64], bias: &[f64], input: &[f64], out_dim: usize) -> Vec<f64> {
    let in_dim = input.len();
    (0..out_dim)
        .map(|i| {
            let mut sum = bias[i];
            for j in 0..in_dim {
                sum = weights[i * in_dim + j].mul_add(input[j], sum);
            }
            sum
        })
        .collect()
}

/// MLP forward pass (tanh hidden, linear output).
#[must_use]
pub fn mlp_forward(input: &[f64], layers: &[(&[f64], &[f64], usize)]) -> Vec<f64> {
    let n_layers = layers.len();
    let mut current = input.to_vec();
    for (i, &(weights, bias, out_dim)) in layers.iter().enumerate() {
        current = if i < n_layers - 1 {
            dense_tanh(weights, bias, &current, out_dim)
        } else {
            dense_linear(weights, bias, &current, out_dim)
        };
    }
    current
}

/// Branch-trunk dot product: <`branch_out`, `trunk_out`> + bias.
#[must_use]
pub fn branch_trunk_dot(branch_out: &[f64], trunk_out: &[f64], bias: f64) -> f64 {
    branch_out
        .iter()
        .zip(trunk_out.iter())
        .map(|(&b, &t)| b * t)
        .sum::<f64>()
        + bias
}

/// L2 relative error between predicted and exact operator outputs.
#[must_use]
pub fn l2_relative_error(predicted: &[f64], exact: &[f64]) -> f64 {
    let num: f64 = predicted
        .iter()
        .zip(exact.iter())
        .map(|(&p, &e)| (p - e).powi(2))
        .sum();
    let den: f64 = exact.iter().map(|&e| e * e).sum();
    (num / (den + 1e-30)).sqrt()
}

/// RMSE between predicted and exact values.
#[must_use]
pub fn rmse(predicted: &[f64], exact: &[f64]) -> f64 {
    let mse: f64 = predicted
        .iter()
        .zip(exact.iter())
        .map(|(&p, &e)| (p - e).powi(2))
        .sum::<f64>()
        / predicted.len() as f64;
    mse.sqrt()
}

/// Linspace: evenly spaced values in [start, end].
#[must_use]
pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + i as f64 * step).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::tolerances;

    #[test]
    fn polynomial_constant() {
        let coeffs = [3.0];
        let x = linspace(0.0, 1.0, 10);
        let vals = eval_polynomial(&coeffs, &x);
        for &v in &vals {
            assert!((v - 3.0).abs() < tolerances::EXACT_F64);
        }
    }

    #[test]
    fn polynomial_linear() {
        let coeffs = [1.0, 2.0]; // 1 + 2x
        let vals = eval_polynomial(&coeffs, &[0.0, 0.5, 1.0]);
        assert!((vals[0] - 1.0).abs() < tolerances::EXACT_F64);
        assert!((vals[1] - 2.0).abs() < tolerances::EXACT_F64);
        assert!((vals[2] - 3.0).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn antiderivative_constant() {
        // u(x) = 1 → G(u)(y) = y
        let coeffs = [1.0];
        let y = linspace(0.0, 1.0, 5);
        let g = exact_antiderivative(&coeffs, &y);
        for (i, &v) in g.iter().enumerate() {
            assert!(
                (v - y[i]).abs() < tolerances::EXACT_F64,
                "G(1)(y={}) = {v}, expected {}",
                y[i],
                y[i]
            );
        }
    }

    #[test]
    fn antiderivative_linear() {
        // u(x) = x → G(u)(y) = y²/2
        let coeffs = [0.0, 1.0];
        let y = linspace(0.0, 1.0, 5);
        let g = exact_antiderivative(&coeffs, &y);
        for (i, &v) in g.iter().enumerate() {
            let expected = y[i].powi(2) / 2.0;
            assert!((v - expected).abs() < tolerances::EXACT_F64);
        }
    }

    #[test]
    fn antiderivative_quadratic() {
        // u(x) = x² → G(u)(y) = y³/3
        let coeffs = [0.0, 0.0, 1.0];
        let y = linspace(0.0, 1.0, 5);
        let g = exact_antiderivative(&coeffs, &y);
        for (i, &v) in g.iter().enumerate() {
            let expected = y[i].powi(3) / 3.0;
            assert!((v - expected).abs() < tolerances::EXACT_F64);
        }
    }

    #[test]
    fn random_polynomial_pair_consistency() {
        let mut rng = Rng::new(42);
        let x = linspace(0.0, 1.0, 50);
        let y = linspace(0.0, 1.0, 50);
        let (u_vals, g_vals, coeffs) = random_polynomial_pair(&mut rng, 5, &x, &y);
        assert_eq!(u_vals.len(), 50);
        assert_eq!(g_vals.len(), 50);
        assert!(!coeffs.is_empty());

        let u_check = eval_polynomial(&coeffs, &x);
        let g_check = exact_antiderivative(&coeffs, &y);
        for i in 0..50 {
            assert!((u_vals[i] - u_check[i]).abs() < tolerances::EXACT_F64);
            assert!((g_vals[i] - g_check[i]).abs() < tolerances::EXACT_F64);
        }
    }

    #[test]
    fn dataset_correct_shape() {
        let x = linspace(0.0, 1.0, 20);
        let y = linspace(0.0, 1.0, 15);
        let (u, g) = generate_dataset(100, &x, &y, 5, 42);
        assert_eq!(u.len(), 100 * 20);
        assert_eq!(g.len(), 100 * 15);
    }

    #[test]
    fn branch_trunk_dot_identity() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 1.0, 1.0];
        let result = branch_trunk_dot(&a, &b, 0.0);
        assert!((result - 6.0).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn l2_relative_exact_match() {
        let a = [1.0, 2.0, 3.0];
        assert!(l2_relative_error(&a, &a) < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn rmse_exact_match() {
        let a = [1.0, 2.0, 3.0];
        assert!(rmse(&a, &a) < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn linspace_endpoints() {
        let v = linspace(0.0, 10.0, 11);
        assert_eq!(v.len(), 11);
        assert!((v[0]).abs() < tolerances::ZERO_DETECTION);
        assert!((v[10] - 10.0).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn mlp_forward_passthrough() {
        let w = [1.0, 0.0, 0.0, 1.0];
        let b = [0.0, 0.0];
        let result = mlp_forward(&[0.5, -0.3], &[(&w, &b, 2)]);
        assert!((result[0] - 0.5).abs() < tolerances::EXACT_F64);
        assert!((result[1] - (-0.3)).abs() < tolerances::EXACT_F64);
    }
}
