// SPDX-License-Identifier: AGPL-3.0-only

//! Centralized validation tolerances with mathematical justification.
//!
//! Every tolerance threshold used in validation binaries is defined here.
//! No ad-hoc magic numbers in binaries. Imitates the hotSpring pattern.
//!
//! # Tolerance categories
//!
//! | Category | Basis | Example |
//! |----------|-------|---------|
//! | Machine precision | IEEE 754 f64 | 1e-12 for exact arithmetic |
//! | Cross-language | Python/Rust agreement | 1e-10 for same formula |
//! | Training variance | Stochastic optimization | 5% for trained models |
//! | Literature | Published uncertainty | Paper-reported accuracy |

// ═══════════════════════════════════════════════════════════════════
// Machine-precision tolerances (IEEE 754 f64)
// ═══════════════════════════════════════════════════════════════════

/// Tolerance for operations that should be exact in f64 arithmetic.
///
/// f64 has ~15.9 significant digits. 1e-12 allows ~3 digits of accumulated
/// rounding in compositions of exact operations (benchmark function evaluation,
/// softmax, GELU at single points).
pub const EXACT_F64: f64 = 1e-12;

/// Tolerance for cross-language validation (Rust vs Python/NumPy).
///
/// Both use IEEE 754 f64 but may differ in operation ordering, FMA usage,
/// and library implementations of transcendentals (sin, cos, exp, tanh).
/// 1e-10 accounts for these differences.
pub const CROSS_LANGUAGE: f64 = 1e-10;

// ═══════════════════════════════════════════════════════════════════
// Benchmark function tolerances
// ═══════════════════════════════════════════════════════════════════

/// Benchmark function global minimum: should be exactly 0.
///
/// Rastrigin(0,0) = 0, Rosenbrock(1,1) = 0, Ackley(0,0) = 0.
/// Pure arithmetic with no iteration, so machine precision applies.
pub const BENCHMARK_GLOBAL_MIN: f64 = EXACT_F64;

/// Benchmark function at non-trivial points: Rust vs Python.
///
/// Cross-validated against `NumPy` 2.2.6 at 4 reference points per function.
/// Differences arise from transcendental evaluation (cos, exp, sqrt).
pub const BENCHMARK_CROSS_PYTHON: f64 = CROSS_LANGUAGE;

// ═══════════════════════════════════════════════════════════════════
// Transformer primitive tolerances
// ═══════════════════════════════════════════════════════════════════

/// Softmax: sum-to-one property.
///
/// After exp and division, accumulated rounding produces residual
/// |sum - 1.0| of O(n * eps) where n is the vector length.
/// For n ≤ 64, this is well within 1e-12.
pub const SOFTMAX_SUM: f64 = EXACT_F64;

/// Softmax: Rust vs Python element-wise agreement.
///
/// Numerically stable softmax (subtract max) may differ by operation
/// ordering between Rust iterators and `NumPy` vectorized ops.
pub const SOFTMAX_CROSS_PYTHON: f64 = 1e-14;

/// GELU: Rust vs Python at reference points.
///
/// The tanh approximation involves sqrt(2/pi) and x^3 terms.
/// `NumPy`'s tanh may use different polynomial coefficients than Rust's libm.
pub const GELU_CROSS_PYTHON: f64 = EXACT_F64;

/// GELU: large input approximation (GELU(x) ≈ x for x >> 0).
///
/// At x = 10, GELU(10) ≈ 10.0 to within 1e-6.
pub const GELU_LARGE_INPUT: f64 = 1e-6;

// ═══════════════════════════════════════════════════════════════════
// Metric tolerances
// ═══════════════════════════════════════════════════════════════════

/// Metrics (R², RMSE, MAE): exact arithmetic on known inputs.
///
/// No iteration, no transcendentals — just arithmetic on small arrays.
/// Machine precision applies.
pub const METRIC_EXACT: f64 = 1e-14;

// ═══════════════════════════════════════════════════════════════════
// Training / model tolerances (Python baselines)
// ═══════════════════════════════════════════════════════════════════

/// MLP surrogate R² threshold (minimum acceptable).
///
/// FAO-56 ET₀ MLP achieves R² > 0.95 consistently with seed=42.
/// Benchmark functions (Rastrigin) may be lower due to multimodality.
pub const SURROGATE_R2_MIN: f64 = 0.40;

/// Transformer `NumPy` vs `PyTorch`: max absolute difference.
///
/// IEEE 754 f64 summation order is the only source of difference.
/// Measured max: 2.22e-16. Tolerance: 1e-10.
///
/// Provenance: `control/transformer/transformer_inference.py`, 2026-02-16
pub const TRANSFORMER_NUMPY_VS_PYTORCH: f64 = 1e-10;

/// Causal mask leak: max attention weight on future tokens.
///
/// exp(-1e9) ≈ 0. Any nonzero leak is a bug. 1e-6 catches
/// implementations that use insufficiently negative mask values.
pub const CAUSAL_MASK_LEAK: f64 = 1e-6;

/// LSTM/GRU R² threshold for 1-day weather forecast.
///
/// Persistence baseline is ~0.94; neural models should be competitive.
pub const SEQUENCE_R2_MIN: f64 = 0.80;

/// PINN L2 relative error threshold (Adam-only, no L-BFGS).
///
/// Paper achieves 0.06% with L-BFGS. Adam-only reaches ~5%.
/// 15% is the acceptance threshold.
pub const PINN_L2_ERROR_MAX: f64 = 0.15;

/// INT8 quantization: max R² degradation from FP32.
///
/// Measured: 0.017%. Threshold: 1%.
pub const QUANT_INT8_DEGRADATION: f64 = 0.01;

/// INT4 quantization: max R² degradation from FP32.
///
/// Measured: 0.79%. Threshold: 5%.
pub const QUANT_INT4_DEGRADATION: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════
// Tensor / WGSL shader tolerances (f32 compute)
// ═══════════════════════════════════════════════════════════════════

/// Exact f32 operations via WGSL (`ReLU`, add, sub, mul, scalar mul).
///
/// f32 has ~7.2 significant digits. For operations that are exact
/// in real arithmetic (e.g., max(0, x), a + b), WGSL produces
/// bit-exact results. 1e-6 allows one digit of slack.
pub const TENSOR_EXACT_F32: f64 = 1e-6;

/// Transcendental f32 ops via WGSL (GELU, sigmoid, softmax, exp).
///
/// WGSL implementations of erf/exp/tanh may use polynomial
/// approximations that differ from CPU libm. 1e-3 is conservative
/// for f32 transcendentals.
pub const TENSOR_TRANSCENDENTAL_F32: f64 = 1e-3;

/// `MatMul` and reduction ops via WGSL (f32 accumulation).
///
/// Dot products accumulate rounding errors proportional to √n for
/// random inputs. For small matrices (n ≤ 64), 1e-2 is generous;
/// for larger matrices, relative error checks are preferred.
pub const TENSOR_MATMUL_F32: f64 = 1e-2;

/// Normalization ops via WGSL (layer norm, RMS norm, batch norm).
///
/// Involves mean, variance, and division — multiple f32 reductions.
/// 1e-3 accounts for accumulated f32 rounding in the norm pipeline.
pub const TENSOR_NORM_F32: f64 = 1e-3;

// ═══════════════════════════════════════════════════════════════════
// GPU f64 shader tolerances (SHADER_F64 path)
// ═══════════════════════════════════════════════════════════════════

/// Exact f64 operations via WGSL `SHADER_F64` (sum, dot, add, mul).
///
/// GPU f64 arithmetic is IEEE 754 compliant on NVIDIA.  For exact
/// operations the only error is accumulation order (non-deterministic
/// parallel reduction).  1e-10 is conservative for arrays up to 10k.
pub const GPU_F64_EXACT: f64 = 1e-10;

/// Transcendental f64 ops via WGSL `SHADER_F64` (exp, log, sqrt, erf).
///
/// GPU implementations may use polynomial approximations that differ
/// from CPU libm.  On NVIDIA (`sm_89`+) these are accurate to ~1 ULP;
/// on NVK/older drivers the `exp_f64`/`log_f64` workarounds may cost
/// a few extra ULPs.  1e-8 accommodates all validated hardware.
pub const GPU_F64_TRANSCENDENTAL: f64 = 1e-8;

/// Statistical reductions via f64 WGSL (variance, std, correlation).
///
/// Two-pass algorithms (mean then variance) compound reduction error.
/// 1e-9 for arrays up to 10k elements.
pub const GPU_F64_STATS: f64 = 1e-9;

// ═══════════════════════════════════════════════════════════════════
// ML inference pipeline tolerances (f32 multi-op chains)
// ═══════════════════════════════════════════════════════════════════

/// Multi-op ML pipeline (MLP: 3× matmul + 2× `ReLU` + softmax).
///
/// Each f32 matmul accumulates O(√k) rounding for inner dimension k.
/// Chaining 3 linear layers with non-linearities compounds error.
/// 1e-2 matches the single-`matmul` tolerance since the MLP is small
/// (max inner dim = 64).
pub const ML_MLP_F32: f64 = TENSOR_MATMUL_F32;

/// Multi-op transformer pipeline (6+ `matmul` stages, `LayerNorm`, GELU).
///
/// A transformer encoder block chains `LayerNorm` → QKV projections →
/// scaled dot-product attention → output projection → FFN (2× matmul).
/// Each stage compounds f32 error. 0.05 is conservative for small
/// configs (`d_model`=32, `seq_len`=8).
pub const ML_TRANSFORMER_F32: f64 = 0.05;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerance_ordering() {
        assert!(EXACT_F64 < CROSS_LANGUAGE);
        assert!(SOFTMAX_CROSS_PYTHON < CROSS_LANGUAGE);
    }

    #[test]
    fn all_positive() {
        let tols = [
            EXACT_F64,
            CROSS_LANGUAGE,
            BENCHMARK_GLOBAL_MIN,
            BENCHMARK_CROSS_PYTHON,
            SOFTMAX_SUM,
            SOFTMAX_CROSS_PYTHON,
            GELU_CROSS_PYTHON,
            GELU_LARGE_INPUT,
            METRIC_EXACT,
            SURROGATE_R2_MIN,
            TRANSFORMER_NUMPY_VS_PYTORCH,
            CAUSAL_MASK_LEAK,
            SEQUENCE_R2_MIN,
            PINN_L2_ERROR_MAX,
            QUANT_INT8_DEGRADATION,
            QUANT_INT4_DEGRADATION,
            TENSOR_EXACT_F32,
            TENSOR_TRANSCENDENTAL_F32,
            TENSOR_MATMUL_F32,
            TENSOR_NORM_F32,
            GPU_F64_EXACT,
            GPU_F64_TRANSCENDENTAL,
            GPU_F64_STATS,
            ML_MLP_F32,
            ML_TRANSFORMER_F32,
        ];
        for (i, &t) in tols.iter().enumerate() {
            assert!(t > 0.0, "tolerance index {i} must be positive, got {t}");
        }
    }
}
