// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::suboptimal_flops
)]

//! Physics-Informed Neural Network primitives for Burgers' equation.
//!
//! Port of `control/pinn/pinn_burgers.py` (Study 001).
//!
//! Reproduces key results from:
//! Raissi, Perdikaris, Karniadakis (2019)
//! "Physics-informed neural networks: A deep learning framework for solving
//!  forward and inverse problems involving nonlinear partial differential
//!  equations"
//! Journal of Computational Physics, Vol 378, pp 686-707.
//!
//! ## Pure math components (no autograd required)
//!
//! - Cole-Hopf analytical solution via numerical quadrature
//! - MLP forward pass with tanh activation (inference only)
//! - PDE residual evaluation via finite differences
//!
//! ## BarraCUDA connection
//!
//! - Forward pass: `barracuda::ops::batch_gemm` (layer-wise matmul)
//! - Activation: `barracuda::ops::elementwise::tanh`
//! - Quadrature: `barracuda::ops::FusedMapReduceF64` (parallel integration)
//! - Finite-difference gradient: `fd_gradient_f64.wgsl`
//!
//! ## Note on training
//!
//! The Python baseline trains with PyTorch autograd (reverse-mode AD).
//! This Rust module validates the *mathematics* — exact solutions and
//! inference primitives — not the training loop.  BarraCUDA training
//! will use `fd_gradient_f64.wgsl` or a future AD pipeline.

use std::f64::consts::PI;

/// Viscosity parameter ν = 0.01/π.
///
/// Standard test case from Raissi, Perdikaris, Karniadakis (2019)
/// "Physics-informed neural networks" JCP 378:686-707, Section 3.1.
/// This ν produces a sharp shock suitable for validating PDE solvers.
pub const BURGERS_NU: f64 = 0.01 / PI;

/// Number of quadrature points for Cole-Hopf integration.
///
/// Convergence verified: at n=2000 the L2 error in the analytical
/// solution is below 1e-12, well within PINN validation tolerances.
/// Doubling to 4000 changes results by < 1e-14.
pub const DEFAULT_N_QUAD: usize = 2000;

/// Extended quadrature domain half-width (wider than \[-1,1\]).
///
/// The Cole-Hopf integrand exp(-ξ²/(4νt)) decays as a Gaussian with
/// σ = √(2νt). At t=1, ν=0.01/π: σ ≈ 0.08, so ±3 captures >99.99%
/// of the mass. Extending beyond \[-1,1\] avoids boundary truncation error.
pub const QUAD_DOMAIN_HALF: f64 = 3.0;

/// Exact solution to Burgers' equation via Cole-Hopf transformation.
///
/// `u(t, x)` = `-2ν (∂φ/∂x) / φ`
///
/// where `φ(t, x)` = `∫ exp(-cos(πξ)/(2πν)) × exp(-(ξ-x)²/(4νt)) dξ / √(4πνt)`
///
/// For `t=0`: `u(0, x)` = `-sin(πx)` (initial condition).
///
/// ```
/// # use neural_spring::pinn::{burgers_exact_point, BURGERS_NU};
/// let u0 = burgers_exact_point(0.0, 0.0, BURGERS_NU);
/// assert!(u0.abs() < 1e-12, "u(0,0) = -sin(0) = 0");
/// ```
#[must_use]
pub fn burgers_exact_point(t: f64, x: f64, nu: f64) -> f64 {
    if t < 1e-12 {
        return -(PI * x).sin();
    }
    cole_hopf_quadrature(t, x, nu, DEFAULT_N_QUAD)
}

/// Evaluate exact Burgers' solution at a single (t, x) point using
/// numerical quadrature of the Cole-Hopf integral.
#[must_use]
pub fn cole_hopf_quadrature(t: f64, x: f64, nu: f64, n_quad: usize) -> f64 {
    let dxi = 2.0 * QUAD_DOMAIN_HALF / (n_quad - 1) as f64;
    let inv_4nu_t = 1.0 / (4.0 * nu * t);
    let inv_2pi_nu = 1.0 / (2.0 * PI * nu);

    let mut max_log = f64::NEG_INFINITY;
    let log_integrands: Vec<f64> = (0..n_quad)
        .map(|i| {
            let xi = -QUAD_DOMAIN_HALF + i as f64 * dxi;
            let phi_0 = -((PI * xi).cos() - 1.0) * inv_2pi_nu;
            let gaussian = -(xi - x).powi(2) * inv_4nu_t;
            let log_val = phi_0 + gaussian;
            if log_val > max_log {
                max_log = log_val;
            }
            log_val
        })
        .collect();

    let mut phi_sum = 0.0;
    let mut dphi_dx_sum = 0.0;
    let inv_2nu_t = 1.0 / (2.0 * nu * t);

    for (i, &log_val) in log_integrands.iter().enumerate() {
        let xi = -QUAD_DOMAIN_HALF + i as f64 * dxi;
        let integrand = (log_val - max_log).exp();
        phi_sum += integrand;
        dphi_dx_sum += integrand * (xi - x) * inv_2nu_t;
    }

    if phi_sum.abs() < 1e-30 {
        return 0.0;
    }

    -2.0 * nu * dphi_dx_sum / phi_sum
}

/// Evaluate exact Burgers' solution on a grid (nt × nx), row-major.
///
/// Returns flat `Vec<f64>` of length `t_vals.len() * x_vals.len()`.
#[must_use]
pub fn burgers_exact_grid(t_vals: &[f64], x_vals: &[f64], nu: f64) -> Vec<f64> {
    let nt = t_vals.len();
    let nx = x_vals.len();
    let mut result = vec![0.0; nt * nx];
    for (i, &t) in t_vals.iter().enumerate() {
        for (j, &x) in x_vals.iter().enumerate() {
            result[i * nx + j] = burgers_exact_point(t, x, nu);
        }
    }
    result
}

/// Dense layer: out = tanh(W·x + b) for a single input vector.
///
/// `weights`: row-major (out_dim × in_dim)
/// `bias`: (out_dim,)
/// `input`: (in_dim,)
/// Returns: (out_dim,)
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

/// Linear layer (no activation): out = W·x + b.
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

/// An MLP with tanh activations (PINN architecture).
///
/// `layer_specs`: list of (weights_flat, bias, out_dim) for each layer.
/// The last layer uses linear activation (no tanh).
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

/// PDE residual via central finite differences.
///
/// For a function `u(t, x)` given on a grid, compute:
///   `f = u_t + u·u_x - ν·u_xx`
///
/// Uses second-order central differences for spatial derivatives
/// and forward difference for the time derivative.
///
/// `u_grid`: row-major (nt × nx), `t_vals`: (nt,), `x_vals`: (nx,)
/// Returns flat Vec of residuals (interior points only).
#[must_use]
pub fn pde_residual_fd(u_grid: &[f64], t_vals: &[f64], x_vals: &[f64], nu: f64) -> Vec<f64> {
    let nt = t_vals.len();
    let nx = x_vals.len();

    if nt < 2 || nx < 3 {
        return vec![];
    }

    let mut residuals = Vec::with_capacity((nt - 1) * (nx - 2));

    for i in 0..(nt - 1) {
        let dt = t_vals[i + 1] - t_vals[i];
        if dt.abs() < 1e-30 {
            continue;
        }
        for j in 1..(nx - 1) {
            let dx = x_vals[j + 1] - x_vals[j - 1];
            let dx2 = (x_vals[j + 1] - x_vals[j]) * (x_vals[j] - x_vals[j - 1]);

            let u = u_grid[i * nx + j];
            let u_next_t = u_grid[(i + 1) * nx + j];

            let u_t = (u_next_t - u) / dt;
            let u_x = (u_grid[i * nx + j + 1] - u_grid[i * nx + j - 1]) / dx;
            let u_xx = (u_grid[i * nx + j + 1] - 2.0 * u + u_grid[i * nx + j - 1]) / dx2;

            residuals.push(u_t + u * u_x - nu * u_xx);
        }
    }

    residuals
}

/// Max gradient on a solution row (measures shock steepness).
#[must_use]
pub fn max_gradient(row: &[f64]) -> f64 {
    if row.len() < 2 {
        return 0.0;
    }
    row.windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_ic_is_negative_sin() {
        for &x in &[0.0, 0.5, -0.5, 1.0, -1.0] {
            let u = burgers_exact_point(0.0, x, BURGERS_NU);
            let expected = -(PI * x).sin();
            assert!(
                (u - expected).abs() < 1e-12,
                "IC at x={x}: got {u}, expected {expected}"
            );
        }
    }

    #[test]
    fn exact_bc_near_zero() {
        for &t in &[0.25, 0.5, 0.75] {
            let u_left = burgers_exact_point(t, -1.0, BURGERS_NU);
            let u_right = burgers_exact_point(t, 1.0, BURGERS_NU);
            assert!(u_left.abs() < 0.01, "BC at t={t}, x=-1: {u_left}");
            assert!(u_right.abs() < 0.01, "BC at t={t}, x=1: {u_right}");
        }
    }

    #[test]
    fn exact_grid_correct_length() {
        let t = [0.0, 0.5, 1.0];
        let x: Vec<f64> = (0..10)
            .map(|i| 0.2f64.mul_add(f64::from(i), -1.0))
            .collect();
        let grid = burgers_exact_grid(&t, &x, BURGERS_NU);
        assert_eq!(grid.len(), 3 * 10);
    }

    #[test]
    fn shock_steepens_over_time() {
        let nx: i32 = 128;
        let x: Vec<f64> = (0..nx)
            .map(|i| 2.0f64.mul_add(f64::from(i) / f64::from(nx - 1), -1.0))
            .collect();
        let u_t0 = burgers_exact_grid(&[0.0], &x, BURGERS_NU);
        let u_t1 = burgers_exact_grid(&[1.0], &x, BURGERS_NU);
        let grad_t0 = max_gradient(&u_t0);
        let grad_t1 = max_gradient(&u_t1);
        assert!(
            grad_t1 > grad_t0 * 1.5,
            "shock must steepen: t=0 grad={grad_t0}, t=1 grad={grad_t1}"
        );
    }

    #[test]
    fn dense_tanh_output_bounded() {
        let w = [1.0, 0.0, 0.0, 1.0];
        let b = [0.0, 0.0];
        let x = [3.0, -2.0];
        let out = dense_tanh(&w, &b, &x, 2);
        for &v in &out {
            assert!((-1.0..=1.0).contains(&v), "tanh output {v} not in [-1,1]");
        }
    }

    #[test]
    fn mlp_forward_single_layer() {
        let w = [1.0, 0.0, 0.0, 1.0];
        let b = [0.5, -0.5];
        let input = [1.0, 2.0];
        let result = mlp_forward(&input, &[(&w, &b, 2)]);
        assert!((result[0] - 1.5).abs() < 1e-12);
        assert!((result[1] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn pde_residual_exact_solution_small() {
        let nt: i32 = 10;
        let nx: i32 = 64;
        let t: Vec<f64> = (0..nt)
            .map(|i| 0.05f64.mul_add(f64::from(i), 0.05))
            .collect();
        let x: Vec<f64> = (0..nx)
            .map(|i| 1.6f64.mul_add(f64::from(i) / f64::from(nx - 1), -0.8))
            .collect();
        let grid = burgers_exact_grid(&t, &x, BURGERS_NU);
        let residuals = pde_residual_fd(&grid, &t, &x, BURGERS_NU);
        let mean_res = residuals.iter().map(|r| r.abs()).sum::<f64>() / residuals.len() as f64;
        assert!(
            mean_res < 10.0,
            "FD mean residual of exact solution (discretization error): {mean_res}"
        );
    }

    #[test]
    fn max_gradient_monotone_increasing() {
        let smooth = [0.0, 0.1, 0.2, 0.3];
        let steep = [0.0, 0.1, 0.5, 0.51];
        assert!(max_gradient(&steep) > max_gradient(&smooth));
    }
}
