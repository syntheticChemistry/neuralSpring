// SPDX-License-Identifier: AGPL-3.0-or-later

//! Loss landscape characterization via energy landscape methods.
//!
//! baseCamp Sub-thesis 03: Loss Landscapes as Energy Landscapes.
//!
//! Applies the EL4ML (Energy Landscapes for Machine Learning) framework
//! from chemical physics to neural network loss surfaces. Saddle points
//! are transition states, local minima are metastable configurations.
//!
//! ## Grounding papers
//!
//! - Ballard, Das, Martiniani, Wales (2024) Digital Discovery 3, RSC
//! - Pittorino et al. (2025) "Boltzmann Entropy and NN Generalization"
//! - Liu et al. (2024) "Loss Landscape Characterization"
//!
//! ## Validated primitives
//!
//! - [`crate::eigh::eigh_householder_qr`] — Hessian eigendecomposition
//! - [`crate::rng::Rng`] — deterministic PRNG for Boltzmann sampling

#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::needless_range_loop
)]

use crate::eigh::eigh_householder_qr;
use crate::primitives::LOG_GUARD;
use crate::rng::Rng;

pub use barracuda::sample::BoltzmannResult;

/// Compute numerical Hessian of a loss function at given parameters.
///
/// Uses central finite differences with step `epsilon`.
/// `H(i,j) ≈ (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4ε²)`.
///
/// Returns flat row-major n×n Hessian matrix.
#[must_use]
pub fn numerical_hessian(
    loss_fn: &dyn Fn(&[f64]) -> f64,
    params: &[f64],
    epsilon: f64,
) -> Vec<f64> {
    barracuda::numerical::numerical_hessian(loss_fn, params, epsilon)
}

/// Compute eigenvalues of the Hessian matrix.
///
/// Returns sorted eigenvalues (ascending) characterizing the local
/// curvature of the loss surface.
#[must_use]
pub fn hessian_spectrum(hessian: &[f64], n: usize) -> Vec<f64> {
    let decomp = eigh_householder_qr(hessian, n);
    let mut eigenvalues = decomp.eigenvalues;
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

/// Landscape flatness: fraction of Hessian eigenvalues near zero.
///
/// Flat minima (many near-zero eigenvalues) correlate with
/// generalization. Sharp minima (large eigenvalues) correlate
/// with memorization.
///
/// `threshold` defines "near-zero" (typically 1e-3).
#[must_use]
pub fn landscape_flatness(eigenvalues: &[f64], threshold: f64) -> f64 {
    if eigenvalues.is_empty() {
        return 0.0;
    }
    let flat = eigenvalues
        .iter()
        .filter(|&&ev| ev.abs() < threshold)
        .count();
    flat as f64 / eigenvalues.len() as f64
}

/// Landscape sharpness: max absolute eigenvalue of the Hessian.
///
/// Higher sharpness = sharper minimum = worse generalization prediction.
#[must_use]
pub fn landscape_sharpness(eigenvalues: &[f64]) -> f64 {
    eigenvalues
        .iter()
        .map(|&ev| ev.abs())
        .fold(0.0_f64, f64::max)
}

/// Number of negative eigenvalues (saddle point index).
///
/// 0 = local minimum, 1 = first-order saddle, k = k-th order saddle.
#[must_use]
pub fn saddle_index(eigenvalues: &[f64]) -> usize {
    eigenvalues
        .iter()
        .filter(|&&ev| ev < crate::tolerances::SADDLE_EIGENVALUE_THRESHOLD)
        .count()
}

/// Boltzmann weight sampling: single Metropolis step.
///
/// Proposes a perturbation to `params`, accepts if it reduces loss
/// or with probability exp(-ΔL/T). Returns (new_params, accepted).
#[must_use]
pub fn metropolis_step(
    loss_fn: &dyn Fn(&[f64]) -> f64,
    params: &[f64],
    current_loss: f64,
    temperature: f64,
    step_size: f64,
    rng: &mut Rng,
) -> (Vec<f64>, bool) {
    let mut proposed = params.to_vec();
    for p in &mut proposed {
        *p += step_size * rng.normal();
    }

    let proposed_loss = loss_fn(&proposed);
    let delta = proposed_loss - current_loss;

    let accept =
        delta < 0.0 || (temperature > LOG_GUARD && rng.uniform() < (-delta / temperature).exp());

    if accept {
        (proposed, true)
    } else {
        (params.to_vec(), false)
    }
}

/// Run Boltzmann sampling: MCMC chain at given temperature.
///
/// Delegates to [`barracuda::sample::boltzmann_sampling`] (absorbed S56).
/// Takes a `seed` for deterministic PRNG initialization.
#[must_use]
pub fn boltzmann_sampling(
    loss_fn: &dyn Fn(&[f64]) -> f64,
    initial_params: &[f64],
    temperature: f64,
    step_size: f64,
    n_steps: usize,
    seed: u64,
) -> BoltzmannResult {
    barracuda::sample::boltzmann_sampling(
        loss_fn,
        initial_params,
        temperature,
        step_size,
        n_steps,
        seed,
    )
}

/// Transition barrier estimate between two minima.
///
/// Given loss at two minima and loss at the saddle point between them,
/// returns the activation barrier (max of the two barriers).
/// This is the EL4ML transition state theory analog.
#[must_use]
pub fn transition_barrier(loss_min1: f64, loss_min2: f64, loss_saddle: f64) -> f64 {
    let barrier1 = loss_saddle - loss_min1;
    let barrier2 = loss_saddle - loss_min2;
    barrier1.max(barrier2)
}

/// Spectral gap: difference between largest and second-largest Hessian eigenvalue.
///
/// Large spectral gap indicates a dominant curvature direction.
#[must_use]
pub fn spectral_gap(eigenvalues: &[f64]) -> f64 {
    if eigenvalues.len() < 2 {
        return 0.0;
    }
    let n = eigenvalues.len();
    eigenvalues[n - 1] - eigenvalues[n - 2]
}

/// Compute full loss landscape characterization at a point.
#[must_use]
pub fn landscape_analysis(
    loss_fn: &dyn Fn(&[f64]) -> f64,
    params: &[f64],
    epsilon: f64,
    flatness_threshold: f64,
) -> LandscapeResult {
    let hessian = numerical_hessian(loss_fn, params, epsilon);
    let n = params.len();
    let spectrum = hessian_spectrum(&hessian, n);
    let loss = loss_fn(params);

    LandscapeResult {
        loss,
        flatness: landscape_flatness(&spectrum, flatness_threshold),
        sharpness: landscape_sharpness(&spectrum),
        saddle_index: saddle_index(&spectrum),
        spectral_gap: spectral_gap(&spectrum),
        hessian_eigenvalues: spectrum,
    }
}

/// Result of loss landscape analysis at a point.
#[derive(Debug, Clone)]
pub struct LandscapeResult {
    /// Loss value at the analysis point.
    pub loss: f64,
    /// Fraction of near-zero Hessian eigenvalues.
    pub flatness: f64,
    /// Maximum absolute Hessian eigenvalue.
    pub sharpness: f64,
    /// Number of negative eigenvalues (0 = minimum, >0 = saddle).
    pub saddle_index: usize,
    /// Gap between top two eigenvalues.
    pub spectral_gap: f64,
    /// Full sorted Hessian spectrum.
    pub hessian_eigenvalues: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;

    fn quadratic_loss(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    fn rosenbrock_loss(x: &[f64]) -> f64 {
        if x.len() < 2 {
            return 0.0;
        }
        let dx = 1.0 - x[0];
        let dy = x[0].mul_add(-x[0], x[1]);
        dx.mul_add(dx, 100.0 * dy * dy)
    }

    #[test]
    fn quadratic_hessian_is_identity() {
        let params = vec![0.0; 4];
        let hessian = numerical_hessian(&quadratic_loss, &params, tolerances::HESSIAN_FD_STEP);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 2.0 } else { 0.0 };
                assert!(
                    (hessian[i * 4 + j] - expected).abs() < tolerances::OPTIMIZER_VALUE_AT_MIN,
                    "H[{i},{j}] = {}, expected {expected}",
                    hessian[i * 4 + j]
                );
            }
        }
    }

    #[test]
    fn quadratic_at_minimum() {
        let params = vec![0.0; 4];
        let result = landscape_analysis(&quadratic_loss, &params, tolerances::HESSIAN_FD_STEP, 0.1);
        assert!(
            result.loss.abs() < tolerances::EXACT_F64,
            "loss at origin should be 0"
        );
        assert_eq!(result.saddle_index, 0, "origin is a minimum, not a saddle");
        assert!(
            result.sharpness > 1.0,
            "quadratic should have positive curvature"
        );
    }

    #[test]
    fn rosenbrock_at_minimum() {
        let params = vec![1.0, 1.0];
        let result =
            landscape_analysis(&rosenbrock_loss, &params, tolerances::HESSIAN_FD_STEP, 0.1);
        assert!(
            result.loss < tolerances::SPECIAL_FUNCTION_F64,
            "loss at (1,1) should be ~0"
        );
        assert_eq!(result.saddle_index, 0, "(1,1) is a minimum");
    }

    #[test]
    fn metropolis_preserves_dimension() {
        let mut rng = Rng::new(42);
        let params = vec![1.0, 2.0, 3.0];
        let loss = quadratic_loss(&params);
        let (new_params, _) = metropolis_step(&quadratic_loss, &params, loss, 1.0, 0.1, &mut rng);
        assert_eq!(new_params.len(), params.len());
    }

    #[test]
    fn boltzmann_high_temp_accepts_most() {
        let params = vec![1.0, 1.0];
        let result = boltzmann_sampling(&quadratic_loss, &params, 100.0, 0.1, 500, 42);
        assert!(
            result.acceptance_rate > 0.3,
            "high temperature should accept >30%, got {}",
            result.acceptance_rate
        );
    }

    #[test]
    fn transition_barrier_symmetric() {
        let barrier = transition_barrier(0.0, 0.0, 1.0);
        assert!((barrier - 1.0).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn determinism() {
        let params = vec![1.0, 2.0];
        let h1 = numerical_hessian(&quadratic_loss, &params, tolerances::HESSIAN_FD_STEP);
        let h2 = numerical_hessian(&quadratic_loss, &params, tolerances::HESSIAN_FD_STEP);
        assert_eq!(h1, h2);
    }
}
