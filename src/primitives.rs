// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(
    clippy::cast_precision_loss,
    reason = "array/slice lengths → f64 for statistical computations"
)]

//! CPU-side mathematical primitives shared across science modules.
//!
//! This module is the **CPU reference path**.  For GPU throughput, callers
//! should use the barracuda equivalents listed below.  These CPU functions
//! remain as: (a) independent validation references for GPU correctness,
//! (b) fallback for environments without GPU, (c) small-N fast-path where
//! GPU launch overhead dominates.
//!
//! ## `BarraCUDA` equivalents
//!
//! | CPU primitive (this module) | GPU equivalent (`barracuda`) | Notes |
//! |----------------------------|------------------------------|-------|
//! | [`shannon_entropy`] | `FusedMapReduceF64::shannon_entropy` | GPU batch reduction |
//! | [`sigmoid`] | `Tensor::sigmoid()` / `ops::sigmoid::Sigmoid` | GPU elementwise |
//! | [`hill_activation`] | `ops::bio::HillFunctionF64` / `hill_gate.wgsl` | GPU activation only |
//! | [`hill_repression`] | Not yet exposed upstream | Pending barracuda |
//! | [`rk4_step`] | `ops::ode::BatchedOdeRK4F64` / `rk4_parallel.wgsl` | GPU batch ODE |
//! | [`LOG_GUARD`], [`HILL_EPS`], [`DIVISION_GUARD`] | N/A | Numerical guards — stay here |
//!
//! ## Ownership rule
//!
//! This module **does not reimplement** barracuda math.  Each function here
//! uses a self-contained formula (no barracuda calls) so it serves as an
//! independent reference.  When barracuda absorbs `hill_repression`, this
//! module becomes a thin re-export layer with the guards remaining.

// ═══════════════════════════════════════════════════════════════════
// Numerical safety constants
// ═══════════════════════════════════════════════════════════════════

/// Guard against log(0) in entropy and probability computations.
///
/// Smallest f64 that avoids -inf from `ln()` while being negligible
/// compared to any real probability. `f64::MIN_POSITIVE ≈ 2.2e-308`,
/// so `1e-300` is safe and well above subnormal territory.
///
/// Derivation: IEEE 754 f64 subnormals start at ~5e-324. We stay
/// ~24 orders of magnitude above subnormal to avoid gradual underflow
/// penalties while remaining negligible for any real probability (p > 1e-20).
/// Matches `NumPy`'s internal `_LOGGUARD` sentinel.
pub const LOG_GUARD: f64 = 1e-300;

/// Epsilon for Hill function denominators to prevent division by zero.
///
/// `K^n + x^n` can be zero when both K and x are zero. Adding this
/// epsilon keeps the result finite without affecting the kinetics
/// at biologically relevant concentrations (x, K > 1e-6).
///
/// Derivation: must be (a) small enough that `HILL_EPS / K^n < machine_eps`
/// for typical K ∈ \[0.1, 10\] and n ∈ \[1, 4\], (b) large enough to stay
/// well above subnormal. 1e-20 satisfies both: `1e-20 / 0.1^4 = 1e-16 < ε_f64`.
pub const HILL_EPS: f64 = 1e-20;

/// Guard for denominators in statistical ratios (FST, Pearson, regression).
///
/// When the denominator of a ratio is below this, the result is
/// defined as 0.0 rather than risking amplified floating-point noise.
///
/// Derivation: f64 machine epsilon ≈ 2.2e-16. We set the guard one order
/// of magnitude above to account for accumulated rounding in sums.
/// Any variance or covariance below 1e-15 represents a degenerate
/// (constant) population where the ratio is undefined.
pub const DIVISION_GUARD: f64 = 1e-15;

/// Floor for quantization scale factors to prevent division by zero.
///
/// Used in INT8/INT4 symmetric quantization when `max(|tensor|)` is
/// zero or near-zero. Must be above subnormal territory (5e-324) but
/// small enough to never affect real tensor magnitudes (> 1e-10 in
/// practice). 1e-30 satisfies both constraints: `1e-30 / 127 ≈ 8e-33`
/// is a valid scale, and `tensor / 8e-33` for any real tensor clamps
/// safely to the quantization range.
pub const QUANTIZATION_FLOOR: f64 = 1e-30;

/// Floor for generated probability values to ensure non-zero.
///
/// Applied to `rng.uniform().max(PROBABILITY_FLOOR)` when generating
/// probability distributions for validation and benchmarking. Must be
/// positive to avoid log(0) but small enough to not bias the distribution.
/// 1e-8 keeps values well above zero without distorting the range
/// (uniform draws are in \[0,1) — 1e-8 trims only the bottom 10ppb).
pub const PROBABILITY_FLOOR: f64 = 1e-8;

/// Floor for ratio denominators to prevent inf in diagnostic prints.
///
/// Used when computing improvement ratios like `error_a / error_b.max(RATIO_GUARD)`.
/// Prevents division by zero when the denominator is effectively machine zero.
pub const RATIO_GUARD: f64 = 1e-300;

// ═══════════════════════════════════════════════════════════════════
// Shannon entropy
// ═══════════════════════════════════════════════════════════════════

/// Shannon entropy `H` = -`sum`(`p_i` * `ln`(`p_i`)) from pre-computed frequencies.
///
/// Input `frequencies` should be non-negative and sum to ~1.0.
/// Zero-frequency bins are skipped (0 * ln(0) = 0 by convention).
///
/// This is the core computation; callers are responsible for computing
/// the frequency distribution from their domain-specific data.
///
/// Delegates to `barracuda::stats::shannon_from_frequencies` (absorbed
/// from wetSpring via `ToadStool` S64, now in `BarraCUDA`).
#[must_use]
pub fn shannon_entropy(frequencies: &[f64]) -> f64 {
    barracuda::stats::shannon_from_frequencies(frequencies)
}

/// Shannon equitability `H`/`H_max` where `H_max` = ln(S).
///
/// Returns 0.0 for populations with 0 or 1 types. Returns a value in
/// \[0, 1\] for well-behaved distributions (1.0 = perfectly uniform).
#[must_use]
pub fn shannon_equitability(frequencies: &[f64]) -> f64 {
    let n_nonzero = frequencies.iter().filter(|&&p| p > DIVISION_GUARD).count();
    if n_nonzero <= 1 {
        return 0.0;
    }
    let h = shannon_entropy(frequencies);
    let h_max = (n_nonzero as f64).ln();
    if h_max <= 0.0 {
        return 0.0;
    }
    h / h_max
}

/// Convert raw counts to frequencies, then compute Shannon entropy.
///
/// Delegates to `barracuda::stats::shannon` (absorbed from wetSpring in S64).
#[must_use]
pub fn shannon_entropy_from_counts(counts: &[f64]) -> f64 {
    barracuda::stats::shannon(counts)
}

// ═══════════════════════════════════════════════════════════════════
// Hill kinetics
// ═══════════════════════════════════════════════════════════════════

/// Hill activation: `a * x^n / (K^n + x^n)`.
///
/// Standard activating Hill function from enzyme kinetics.
/// Returns 0 when x <= 0, approaches `a` as x → ∞.
/// Core computation delegates to `barracuda::stats::hill` (absorbed from
/// wetSpring/hotSpring gene regulatory networks via `ToadStool` S64, now in `BarraCUDA`).
#[must_use]
pub fn hill_activation(x: f64, amplitude: f64, k: f64, n: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    amplitude * barracuda::stats::hill(x, k, n)
}

/// Hill repression: `a * K^n / (K^n + x^n)`.
///
/// Standard repressing Hill function. Returns `a` when x = 0,
/// approaches 0 as x → ∞. Uses `barracuda::stats::hill` for the
/// core computation (repression = 1 - activation).
#[must_use]
pub fn hill_repression(x: f64, amplitude: f64, k: f64, n: f64) -> f64 {
    if x <= 0.0 {
        return amplitude;
    }
    amplitude * (1.0 - barracuda::stats::hill(x, k, n))
}

// ═══════════════════════════════════════════════════════════════════
// Sigmoid
// ═══════════════════════════════════════════════════════════════════

/// Numerically stable sigmoid: σ(x) = 1 / (1 + exp(-x)).
///
/// Uses the split formula to avoid overflow: for x >= 0 compute
/// directly, for x < 0 use exp(x)/(1+exp(x)).
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Generic RK4 integration
// ═══════════════════════════════════════════════════════════════════

/// Generic 4th-order Runge-Kutta step for fixed-size ODE systems.
///
/// Advances state by one timestep `dt` using the RHS function `rhs`.
/// The closure signature `Fn(&[f64; N]) -> [f64; N]` lets callers
/// capture any parameters (environment signal, noise source, etc.).
///
/// # Example
///
/// ```
/// use neural_spring::primitives::rk4_step;
///
/// let state = [1.0, 0.0];
/// let next = rk4_step(&state, 0.01, |y| [-y[1], y[0]]); // harmonic oscillator
/// assert!((next[0] * next[0] + next[1] * next[1] - 1.0).abs() < 1e-6);
/// ```
#[must_use]
pub fn rk4_step<const N: usize>(
    state: &[f64; N],
    dt: f64,
    mut rhs: impl FnMut(&[f64; N]) -> [f64; N],
) -> [f64; N] {
    let k1 = rhs(state);

    let half_dt = 0.5 * dt;
    let y2: [f64; N] = std::array::from_fn(|i| half_dt.mul_add(k1[i], state[i]));
    let k2 = rhs(&y2);

    let y3: [f64; N] = std::array::from_fn(|i| half_dt.mul_add(k2[i], state[i]));
    let k3 = rhs(&y3);

    let y4: [f64; N] = std::array::from_fn(|i| dt.mul_add(k3[i], state[i]));
    let k4 = rhs(&y4);

    let dt6 = dt / 6.0;
    std::array::from_fn(|i| {
        let sum_k = 2.0f64.mul_add(k3[i], 2.0f64.mul_add(k2[i], k1[i]) + k4[i]);
        dt6.mul_add(sum_k, state[i])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn entropy_uniform_distribution() {
        let p = [0.25, 0.25, 0.25, 0.25];
        let h = shannon_entropy(&p);
        let expected = (4.0_f64).ln();
        assert!((h - expected).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn entropy_singleton() {
        let p = [1.0, 0.0, 0.0];
        assert!(shannon_entropy(&p).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn equitability_uniform_is_one() {
        let p = [0.25, 0.25, 0.25, 0.25];
        assert!((shannon_equitability(&p) - 1.0).abs() < tolerances::CROSS_LANGUAGE);
    }

    #[test]
    fn equitability_singleton_is_zero() {
        let p = [1.0, 0.0, 0.0];
        assert!(shannon_equitability(&p).abs() < tolerances::CROSS_LANGUAGE);
    }

    #[test]
    fn entropy_from_counts_matches_frequencies() {
        let counts = [10.0, 20.0, 30.0, 40.0];
        let total: f64 = counts.iter().sum();
        let freqs: Vec<f64> = counts.iter().map(|&c| c / total).collect();
        let h1 = shannon_entropy_from_counts(&counts);
        let h2 = shannon_entropy(&freqs);
        assert!((h1 - h2).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn hill_activation_monotonic() {
        let a = hill_activation(0.0, 1.0, 0.5, 2.0);
        let b = hill_activation(0.5, 1.0, 0.5, 2.0);
        let c = hill_activation(1.0, 1.0, 0.5, 2.0);
        assert!(a < b && b < c);
    }

    #[test]
    fn hill_repression_decreasing() {
        let a = hill_repression(0.1, 1.0, 0.5, 2.0);
        let b = hill_repression(1.0, 1.0, 0.5, 2.0);
        assert!(a > b);
    }

    #[test]
    fn hill_activation_at_k_is_half() {
        let v = hill_activation(0.5, 1.0, 0.5, 2.0);
        assert!((v - 0.5).abs() < tolerances::CROSS_LANGUAGE);
    }

    #[test]
    fn sigmoid_at_zero() {
        assert!((sigmoid(0.0) - 0.5).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn sigmoid_symmetry() {
        for &x in &[0.5, 1.0, 2.5, 10.0, 50.0] {
            assert!((sigmoid(x) + sigmoid(-x) - 1.0).abs() < tolerances::ZERO_DETECTION);
        }
    }

    #[test]
    fn sigmoid_stable_at_extremes() {
        assert!(sigmoid(1000.0) > 0.999);
        assert!(sigmoid(-1000.0) < 0.001);
        assert!(sigmoid(1000.0).is_finite());
        assert!(sigmoid(-1000.0).is_finite());
    }

    #[test]
    fn rk4_harmonic_oscillator() {
        let mut state = [1.0, 0.0];
        let dt = 0.001;
        for _ in 0..6283 {
            state = rk4_step(&state, dt, |y| [-y[1], y[0]]);
        }
        let energy = state[0].mul_add(state[0], state[1] * state[1]);
        assert!(
            (energy - 1.0).abs() < tolerances::SPECIAL_FUNCTION_F64,
            "energy conservation: {energy}"
        );
    }

    #[test]
    fn rk4_exponential_decay() {
        let mut state = [1.0];
        let dt = 0.01;
        for _ in 0..100 {
            state = rk4_step(&state, dt, |y| [-y[0]]);
        }
        let expected = (-1.0_f64).exp();
        assert!((state[0] - expected).abs() < tolerances::HMM_POSTERIOR_SUM);
    }

    #[test]
    fn log_guard_prevents_negative_infinity() {
        assert!(LOG_GUARD.ln().is_finite());
    }
}
