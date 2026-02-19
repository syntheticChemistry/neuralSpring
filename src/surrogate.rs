//! MLP surrogate validation against Python baselines.
//!
//! Phase 1 stub — will implement forward-pass MLP using `BarraCUDA`'s
//! `gemm_f64` and `nn::ReLU` primitives.
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
#[must_use]
pub fn rastrigin_2d(x: f64, y: f64) -> f64 {
    let term_x = x.mul_add(x, 10.0f64.mul_add(-(2.0 * PI * x).cos(), 0.0));
    let term_y = y.mul_add(y, 10.0f64.mul_add(-(2.0 * PI * y).cos(), 0.0));
    20.0 + term_x + term_y
}

/// Benchmark function: Rosenbrock 2-D.
#[must_use]
pub fn rosenbrock_2d(x: f64, y: f64) -> f64 {
    let dx = 1.0 - x;
    let dy = x.mul_add(-x, y);
    dx.mul_add(dx, 100.0 * dy * dy)
}

/// Benchmark function: Ackley 2-D.
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
        assert_relative_eq!(ackley_2d(-3.0, 2.0), 7.988_910_810_518_700, epsilon = 1e-12);
    }
}
