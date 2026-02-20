// SPDX-License-Identifier: AGPL-3.0-only

//! Transformer primitive validation against Python baselines.
//!
//! Provides CPU-reference softmax and GELU implementations validated against
//! `NumPy` 2.2.6.  Full transformer inference (SDPA, `LayerNorm`, MHA, FFN)
//! is implemented in the fused GPU pipeline (`evolved::fused_transformer`)
//! and validated end-to-end by `validate_barracuda_ml_inference`.
//!
//! ## Python Baseline Provenance
//!
//! | Check | Tolerance | Rationale |
//! |-------|-----------|-----------|
//! | NumPy vs `PyTorch` softmax | 1e-10 | IEEE-754 f64 summation order only |
//! | NumPy vs `PyTorch` SDPA | 1e-10 | same |
//! | Causal mask leak | 1e-6 | exp(-1e9) ≈ 0; any leak is a bug |
//! | `LayerNorm` mean≈0 | 1e-5 | matches LN eps parameter |
//! | `LayerNorm` var≈1 | 1e-3 | accumulated f64 error over `d_model` |

/// Numerically stable softmax over a slice.
///
/// ```
/// # use neural_spring::transformer::softmax;
/// let s = softmax(&[1.0, 2.0, 3.0]);
/// assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// assert!(s.iter().all(|&v| v >= 0.0));
/// ```
#[must_use]
pub fn softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|&v| v / sum).collect()
}

/// GELU activation (approximate, matching `PyTorch` `gelu('tanh')`).
///
/// ```
/// # use neural_spring::transformer::gelu;
/// assert!((gelu(0.0) - 0.0).abs() < 1e-12);
/// assert!(gelu(5.0) > 4.9);
/// ```
#[must_use]
pub fn gelu(x: f64) -> f64 {
    use std::f64::consts::PI;
    let inner = (2.0 / PI).sqrt() * 0.044_715f64.mul_add(x.powi(3), x);
    0.5 * x * (1.0 + inner.tanh())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn softmax_sums_to_one() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let s = softmax(&x);
        let sum: f64 = s.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn softmax_all_positive() {
        let x = vec![-10.0, 0.0, 10.0];
        let s = softmax(&x);
        assert!(s.iter().all(|&v| v >= 0.0));
    }

    // Cross-validation: Python transformer_inference.softmax([1,2,3,4,5])
    #[test]
    fn softmax_cross_python() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = softmax(&x);
        let expected = [
            1.165_623_095_603_961e-2,
            3.168_492_079_612_427e-2,
            8.612_854_443_626_87e-2,
            2.341_216_572_527_366e-1,
            6.364_086_465_588_308e-1,
        ];
        for (got, want) in s.iter().zip(&expected) {
            assert_relative_eq!(got, want, epsilon = 1e-14);
        }
    }

    #[test]
    fn softmax_numerically_stable() {
        let x = vec![1e10, 1e10 + 1.0, 1e10 + 2.0];
        let s = softmax(&x);
        assert!(s.iter().all(|v| v.is_finite()));
        let sum: f64 = s.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn gelu_at_zero() {
        assert_relative_eq!(gelu(0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn gelu_positive_for_large_input() {
        assert!(gelu(5.0) > 4.9);
    }

    // Cross-validation: Python transformer_inference.gelu_numpy
    #[test]
    fn gelu_cross_python() {
        let cases = [
            (-2.0, -4.540_230_591_222_494e-2),
            (-1.0, -1.588_080_093_917_233e-1),
            (0.0, 0.0),
            (0.5, 3.457_140_098_251_439e-1),
            (1.0, 8.411_919_906_082_768e-1),
            (3.0, 2.996_362_607_918_227),
        ];
        for (x, expected) in &cases {
            assert_relative_eq!(gelu(*x), *expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn softmax_deterministic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let run1 = softmax(&x);
        let run2 = softmax(&x);
        assert_eq!(run1, run2, "softmax must be bit-identical across runs");
    }

    #[test]
    fn gelu_deterministic() {
        let inputs = [-2.0, -1.0, 0.0, 0.5, 1.0, 3.0];
        let run1: Vec<f64> = inputs.iter().map(|&x| gelu(x)).collect();
        let run2: Vec<f64> = inputs.iter().map(|&x| gelu(x)).collect();
        assert_eq!(run1, run2, "gelu must be bit-identical across runs");
    }
}
