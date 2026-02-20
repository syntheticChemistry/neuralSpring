// SPDX-License-Identifier: AGPL-3.0-only

//! Benchmark functions and surrogate validation against Python baselines.
//!
//! Provides Rastrigin, Rosenbrock, and Ackley benchmark functions validated
//! against `NumPy` 2.2.6 reference values.  MLP forward-pass inference is
//! implemented in the fused GPU pipeline (`evolved::fused_mlp`); this module
//! contains the analytical functions that the surrogate models approximate.
//!
//! ## Python Baseline Provenance
//!
//! | Metric | Value | Source |
//! |--------|-------|--------|
//! | Rastrigin RBF R² | ≥0.40 | `control/surrogate/surrogate_validation.py`, seed=42, 2026-02-16 |
//! | Rosenbrock RBF R² | ≥0.95 | same |
//! | FAO-56 MLP R² | ≥0.95 | same |
//! | FAO-56 MLP RMSE | ≤0.15 mm/day | same |

use std::f64::consts::{E, PI};

/// Benchmark function: Rastrigin 2-D.
///
/// f(x,y) = 20 + x² − 10 cos(2πx) + y² − 10 cos(2πy)
///
/// ```
/// # use neural_spring::surrogate::rastrigin_2d;
/// assert!((rastrigin_2d(0.0, 0.0) - 0.0).abs() < 1e-12);
/// assert!((rastrigin_2d(1.0, 1.0) - 2.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn rastrigin_2d(x: f64, y: f64) -> f64 {
    let term_x = x.mul_add(x, 10.0f64.mul_add(-(2.0 * PI * x).cos(), 0.0));
    let term_y = y.mul_add(y, 10.0f64.mul_add(-(2.0 * PI * y).cos(), 0.0));
    20.0 + term_x + term_y
}

/// Benchmark function: Rosenbrock 2-D.
///
/// ```
/// # use neural_spring::surrogate::rosenbrock_2d;
/// assert!((rosenbrock_2d(1.0, 1.0) - 0.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn rosenbrock_2d(x: f64, y: f64) -> f64 {
    let dx = 1.0 - x;
    let dy = x.mul_add(-x, y);
    dx.mul_add(dx, 100.0 * dy * dy)
}

/// Benchmark function: Ackley 2-D.
///
/// ```
/// # use neural_spring::surrogate::ackley_2d;
/// assert!((ackley_2d(0.0, 0.0) - 0.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn ackley_2d(x: f64, y: f64) -> f64 {
    let amplitude = 20.0_f64;
    let decay = 0.2_f64;
    let freq = 2.0 * PI;
    let mean_sq = f64::midpoint(x * x, y * y);
    let mean_cos = f64::midpoint((freq * x).cos(), (freq * y).cos());
    (-amplitude).mul_add((-decay * mean_sq.sqrt()).exp(), -mean_cos.exp()) + amplitude + E
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rastrigin_global_min() {
        assert_relative_eq!(rastrigin_2d(0.0, 0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn rosenbrock_global_min() {
        assert_relative_eq!(rosenbrock_2d(1.0, 1.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn ackley_global_min() {
        assert_relative_eq!(ackley_2d(0.0, 0.0), 0.0, epsilon = 1e-12);
    }

    // Cross-validation against Python: control/surrogate/surrogate_validation.py
    // Values computed with NumPy 2.2.6, seed irrelevant (pure math).

    #[test]
    fn rastrigin_cross_python() {
        assert_relative_eq!(rastrigin_2d(1.0, 1.0), 2.0, epsilon = 1e-14);
        assert_relative_eq!(
            rastrigin_2d(2.5, -1.3),
            4.103_016_994_374_947e1,
            epsilon = 1e-10
        );
        assert_relative_eq!(rastrigin_2d(0.5, 0.5), 4.05e1, epsilon = 1e-14);
        assert_relative_eq!(rastrigin_2d(-3.0, 2.0), 13.0, epsilon = 1e-14);
    }

    #[test]
    fn rosenbrock_cross_python() {
        assert_relative_eq!(rosenbrock_2d(1.0, 1.0), 0.0, epsilon = 1e-14);
        assert_relative_eq!(rosenbrock_2d(2.5, -1.3), 5702.5, epsilon = 1e-10);
        assert_relative_eq!(rosenbrock_2d(0.5, 0.5), 6.5, epsilon = 1e-14);
        assert_relative_eq!(rosenbrock_2d(-3.0, 2.0), 4916.0, epsilon = 1e-10);
    }

    #[test]
    fn ackley_cross_python() {
        assert_relative_eq!(ackley_2d(1.0, 1.0), 3.625_384_938_440_363, epsilon = 1e-12);
        assert_relative_eq!(ackley_2d(2.5, -1.3), 8.772_020_879_614_113, epsilon = 1e-12);
        assert_relative_eq!(ackley_2d(0.5, 0.5), 4.253_654_026_568_412, epsilon = 1e-12);
        assert_relative_eq!(ackley_2d(-3.0, 2.0), 7.988_910_810_518_7, epsilon = 1e-12);
    }

    #[test]
    fn rastrigin_deterministic() {
        let points = [(1.0, 1.0), (2.5, -1.3), (0.5, 0.5), (-3.0, 2.0)];
        let run1: Vec<f64> = points.iter().map(|&(x, y)| rastrigin_2d(x, y)).collect();
        let run2: Vec<f64> = points.iter().map(|&(x, y)| rastrigin_2d(x, y)).collect();
        assert_eq!(run1, run2, "rastrigin must be bit-identical across runs");
    }

    #[test]
    fn rosenbrock_deterministic() {
        let points = [(1.0, 1.0), (2.5, -1.3), (0.5, 0.5), (-3.0, 2.0)];
        let run1: Vec<f64> = points.iter().map(|&(x, y)| rosenbrock_2d(x, y)).collect();
        let run2: Vec<f64> = points.iter().map(|&(x, y)| rosenbrock_2d(x, y)).collect();
        assert_eq!(run1, run2, "rosenbrock must be bit-identical across runs");
    }

    #[test]
    fn ackley_deterministic() {
        let points = [(1.0, 1.0), (2.5, -1.3), (0.5, 0.5), (-3.0, 2.0)];
        let run1: Vec<f64> = points.iter().map(|&(x, y)| ackley_2d(x, y)).collect();
        let run2: Vec<f64> = points.iter().map(|&(x, y)| ackley_2d(x, y)).collect();
        assert_eq!(run1, run2, "ackley must be bit-identical across runs");
    }
}
