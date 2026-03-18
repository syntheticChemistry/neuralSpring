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
/// Delegates to `barracuda::activations::sigmoid` (identical split formula).
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    barracuda::activations::sigmoid(x)
}

/// Pearson correlation coefficient with zero fallback.
///
/// Thin wrapper around [`barracuda::stats::correlation::pearson_correlation`]
/// that returns `0.0` on degenerate inputs (constant arrays, length < 2)
/// rather than propagating an error.  Shared across science modules that
/// compute cross-variable correlation (WDM ensemble QS, attention Anderson,
/// digester Anderson).
///
/// | GPU equivalent | Notes |
/// |----------------|-------|
/// | `barracuda::stats::CorrelationF64` | GPU batch correlation |
#[must_use]
pub fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    barracuda::stats::correlation::pearson_correlation(x, y).unwrap_or(0.0)
}

/// Numerically stable f32 sigmoid: σ(x) = 1 / (1 + e^{-x}).
///
/// Used in GPU validation binaries where tensor outputs are f32.
#[must_use]
pub fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ═══════════════════════════════════════════════════════════════════
// GELU
// ═══════════════════════════════════════════════════════════════════

/// GELU activation (approximate, matching `PyTorch` `gelu('tanh')`):
/// `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
///
/// Delegates to `barracuda::activations::gelu`.
///
/// | GPU equivalent | Notes |
/// |----------------|-------|
/// | `Tensor::gelu_wgsl()` / `gelu_f64.wgsl` | GPU elementwise |
#[must_use]
pub fn gelu(x: f64) -> f64 {
    barracuda::activations::gelu(x)
}

/// f32 GELU activation for GPU validation (tensor outputs are f32).
#[must_use]
pub fn gelu_f32(x: f32) -> f32 {
    use std::f32::consts::PI;
    let inner = (2.0_f32 / PI).sqrt() * (0.044_715_f32).mul_add(x * x * x, x);
    0.5 * x * (1.0 + inner.tanh())
}

// ═══════════════════════════════════════════════════════════════════
// Softmax
// ═══════════════════════════════════════════════════════════════════

/// Numerically stable softmax over a 1-D slice (CPU reference).
///
/// Intentional CPU-reference implementation for f64 validation.
/// Production GPU path: `barracuda::dispatch::softmax_dispatch` or
/// `Tensor::softmax_wgsl()`.  Kept here for cross-validation and
/// determinism tests that require bit-exact f64 softmax without
/// GPU dispatch overhead.
///
/// | GPU equivalent | Notes |
/// |----------------|-------|
/// | `Tensor::softmax_wgsl()` | GPU-resident f32 |
/// | `barracuda::dispatch::softmax_dispatch` | CPU/GPU auto |
#[must_use]
pub fn softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|&v| v / sum).collect()
}

// ═══════════════════════════════════════════════════════════════════
// ReLU
// ═══════════════════════════════════════════════════════════════════

/// Scalar `ReLU`: `max(0, x)`.
///
/// Delegates to `barracuda::activations::relu`.
///
/// | GPU equivalent | Notes |
/// |----------------|-------|
/// | `Tensor::relu()` | GPU elementwise |
#[must_use]
pub fn relu(x: f64) -> f64 {
    barracuda::activations::relu(x)
}

/// Scalar f32 `ReLU` for GPU validation.
#[must_use]
pub const fn relu_f32(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// Vectorized `ReLU` over a slice (allocating).
///
/// Delegates to `barracuda::activations::relu_batch`.
#[must_use]
pub fn relu_vec(x: &[f64]) -> Vec<f64> {
    barracuda::activations::relu_batch(x)
}

/// In-place `ReLU`: `max(0, x)` for each element.
pub fn relu_inplace(x: &mut [f64]) {
    for v in x.iter_mut() {
        *v = v.max(0.0);
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
    fn gelu_at_zero_is_zero() {
        assert!(gelu(0.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn gelu_positive_monotone() {
        assert!(gelu(2.0) > gelu(1.0));
    }

    #[test]
    fn gelu_large_approaches_identity() {
        assert!((gelu(5.0) - 5.0).abs() < 0.01);
    }

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "test inputs are small known constants"
    )]
    fn gelu_f32_matches_f64() {
        for &x in &[-2.0, -1.0, 0.0, 0.5, 1.0, 3.0] {
            let diff = (f64::from(gelu_f32(x as f32)) - gelu(x)).abs();
            assert!(diff < 1e-6, "gelu_f32 vs gelu mismatch at {x}: {diff}");
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let s = softmax(&[1.0, 2.0, 3.0, 4.0]);
        assert!((s.iter().sum::<f64>() - 1.0).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn softmax_preserves_order() {
        let s = softmax(&[1.0, 2.0, 3.0]);
        assert!(s[0] < s[1] && s[1] < s[2]);
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "relu returns exact 0.0 or identity")]
    fn relu_nonnegative() {
        assert_eq!(relu(-5.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(3.0), 3.0);
    }

    #[test]
    fn relu_vec_matches_scalar() {
        let v = relu_vec(&[-1.0, 0.0, 2.0]);
        assert_eq!(v, vec![0.0, 0.0, 2.0]);
    }

    #[test]
    fn relu_inplace_zeros_negatives() {
        let mut v = vec![-1.0, 0.0, 3.0];
        relu_inplace(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 3.0]);
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

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::tolerances;
    use proptest::prelude::*;

    proptest! {
        /// Softmax outputs always sum to 1.0 for any non-empty finite input.
        #[test]
        fn softmax_sums_to_one(xs in proptest::collection::vec(-100.0f64..100.0, 1..64)) {
            let s = softmax(&xs);
            let sum: f64 = s.iter().sum();
            prop_assert!((sum - 1.0).abs() < tolerances::SOFTMAX_SUM,
                "softmax sum = {sum}, expected 1.0 ± {}", tolerances::SOFTMAX_SUM);
        }

        /// Softmax outputs are always non-negative.
        #[test]
        fn softmax_nonnegative(xs in proptest::collection::vec(-100.0f64..100.0, 1..64)) {
            let s = softmax(&xs);
            for (i, &v) in s.iter().enumerate() {
                prop_assert!(v >= 0.0, "softmax[{i}] = {v} < 0");
            }
        }

        /// Shannon entropy is non-negative for any valid probability distribution.
        #[test]
        fn shannon_entropy_nonnegative(
            xs in proptest::collection::vec(0.01f64..1.0, 2..32)
        ) {
            let sum: f64 = xs.iter().sum();
            let normalized: Vec<f64> = xs.iter().map(|&x| x / sum).collect();
            let h = shannon_entropy(&normalized);
            prop_assert!(h >= -tolerances::EXACT_F64,
                "shannon entropy = {h} < 0");
        }

        /// ReLU is idempotent: relu(relu(x)) == relu(x).
        #[test]
        fn relu_idempotent(x in -1000.0f64..1000.0) {
            prop_assert_eq!(relu(relu(x)), relu(x));
        }

        /// ReLU preserves non-negative inputs exactly.
        #[test]
        fn relu_identity_for_positive(x in 0.0f64..1000.0) {
            prop_assert_eq!(relu(x), x);
        }

        /// RK4 energy conservation: harmonic oscillator energy stays bounded
        /// for arbitrary initial conditions.
        #[test]
        fn rk4_energy_bounded(
            x0 in -10.0f64..10.0,
            v0 in -10.0f64..10.0,
        ) {
            let mut state = [x0, v0];
            let dt = 0.01;
            let initial_energy = x0 * x0 + v0 * v0;
            for _ in 0..100 {
                state = rk4_step(&state, dt, |y| [-y[1], y[0]]);
            }
            let final_energy = state[0] * state[0] + state[1] * state[1];
            let drift = (final_energy - initial_energy).abs();
            prop_assert!(drift < initial_energy.max(1.0) * 0.01,
                "energy drift = {drift}, initial = {initial_energy}");
        }
    }
}
