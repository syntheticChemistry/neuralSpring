// SPDX-License-Identifier: AGPL-3.0-or-later

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
// Phase 0++ evolutionary / stochastic algorithm tolerances
// ═══════════════════════════════════════════════════════════════════

/// Counterdiabatic protocol: max gap between CD and naive for the
/// comparison to be considered "comparable" (not a strict improvement).
///
/// 0.01 L1 distance is a tight threshold — the mean-field Wright-Fisher
/// operates in a 32-dimensional simplex where total variation is at most 2.
pub const CD_COMPARABLE_DIST: f64 = 0.01;

/// Adiabaticity gap: max excess KL for CD over naive before failing.
///
/// CD should stay closer to equilibrium; 0.05 nats allows for numerical
/// noise in the Fisher information discretization (1000 grid points).
pub const ADIABATIC_KL_GAP: f64 = 0.05;

/// HMM posterior: row-sum tolerance (should sum to 1.0).
///
/// Forward-backward accumulates rounding from T matrix-vector products.
/// For T ≤ 5000 with scaling, 1e-8 is conservative.
pub const HMM_POSTERIOR_SUM: f64 = 1e-8;

/// QS cooperation variance: max variance in late cooperation frequency.
///
/// A stabilized QS model should not oscillate beyond this level.
pub const QS_VARIANCE_MAX: f64 = 0.05;

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
// FFT tolerances (Cooley-Tukey radix-2 via WGSL)
// ═══════════════════════════════════════════════════════════════════

/// FFT inverse round-trip: `IFFT(FFT(x)) == x` (f32).
///
/// The butterfly has O(log N) stages, each accumulating f32 rounding.
/// For N ≤ 1024, max element-wise error is well within 1e-3.
pub const FFT_INVERSE_F32: f64 = 1e-3;

/// FFT inverse round-trip (f64 precision).
///
/// Double-precision butterflies lose ~1 ULP per stage.  For N ≤ 1024
/// (10 stages), 1e-10 is conservative.
pub const FFT_INVERSE_F64: f64 = 1e-10;

/// FFT Parseval's theorem: `||x||² == ||X||²/N` (f32).
///
/// Parseval equality relates time-domain and frequency-domain energy.
/// The ratio should be 1.0; accumulated f32 rounding for N ≤ 1024
/// stays within 1e-3.
pub const FFT_PARSEVAL_F32: f64 = 1e-3;

/// FFT known DFT pair: delta→constant, constant→delta (f32).
///
/// Analytically exact transforms verified element-wise.  Single butterfly
/// pass for delta/constant; 1e-5 accommodates f32 twiddle factor precision.
pub const FFT_KNOWN_PAIR_F32: f64 = 1e-5;

/// FFT cosine energy concentration (f32).
///
/// A pure cosine at frequency f should have energy only at bins f and N-f.
/// Leakage into other bins should be below this threshold.
pub const FFT_SPECTRAL_LEAKAGE_F32: f64 = 1e-4;

/// FFT Parseval's theorem (f64).
///
/// Energy conservation in double-precision butterflies.  For N ≤ 1024
/// the ratio `freq_energy / time_energy` should be 1.0 within 1e-10.
pub const FFT_PARSEVAL_F64: f64 = 1e-10;

/// FFT known DFT pair (f64): delta→constant, constant→delta.
///
/// Analytically exact transforms verified element-wise at f64 precision.
pub const FFT_KNOWN_PAIR_F64: f64 = 1e-10;

/// FFT cosine energy concentration (f64).
///
/// Leakage into off-peak bins for a pure cosine input (f64).
pub const FFT_SPECTRAL_LEAKAGE_F64: f64 = 1e-10;

/// RFFT output shape: N real inputs → N/2+1 complex outputs.
///
/// The RFFT exploits conjugate symmetry of real-valued signals.
/// Shape validation is exact (boolean check), but energy tolerance
/// for the compacted spectrum uses f32 FFT thresholds since `Rfft`
/// delegates to `Fft1D` (f32) internally.
pub const RFFT_DC_COMPONENT_F32: f64 = 1e-3;

// ═══════════════════════════════════════════════════════════════════
// GPU shader tolerances (metalForge Phase 3c shaders)
// ═══════════════════════════════════════════════════════════════════

/// GPU HMM forward: log-likelihood agreement with CPU reference (f32).
///
/// The GPU shader uses f32 logsumexp (max-subtract trick).  The CPU
/// reference uses f64 scaled forward.  For T ≤ 100 observations and
/// N ≤ 10 states, the f32 → f64 gap plus log-domain rounding stays
/// within 0.5 (log-likelihood absolute difference).
pub const GPU_HMM_LOG_LIKELIHOOD_F32: f64 = 0.5;

/// GPU HMM forward: per-state alpha agreement with CPU (f32).
///
/// Each timestep accumulates f32 logsumexp error.  For T ≤ 100,
/// max per-element absolute difference is within 0.1.
pub const GPU_HMM_ALPHA_F32: f64 = 0.1;

/// GPU batch fitness: linear fitness dot-product (f32).
///
/// Population fitness = genotype · weights.  For `genome_len` ≤ 64,
/// f32 dot-product rounding is within 1e-4.
pub const GPU_FITNESS_F32: f64 = 1e-4;

/// GPU RK4: ODE integration agreement with CPU (f32).
///
/// Multi-step RK4 accumulates rounding per step.  For `n_steps` ≤ 1000
/// with dt ≤ 0.01, the per-variable error stays within 1e-2.
pub const GPU_RK4_F32: f64 = 1e-2;

/// GPU Jaccard distance: pairwise binary set distance (f32).
///
/// Jaccard = 1 - |A∩B|/|A∪B|. The numerator and denominator are
/// integer counts, but f32 accumulation of 0/1 values for `n_genes` ≤ 1000
/// produces exact sums. The division is the only rounding source.
pub const GPU_JACCARD_F32: f64 = 1e-4;

/// GPU locus variance: per-locus allele frequency variance (f32).
///
/// One-pass Welford variance for `n_pops` ≤ 64 populations.  The f32
/// mean/variance computation loses ~2 digits vs f64.  1e-3 accommodates.
pub const GPU_LOCUS_VARIANCE_F32: f64 = 1e-3;

/// GPU spatial payoff: PD stencil fitness (f32).
///
/// Payoff accumulates 8 neighbor contributions using f32 arithmetic.
/// Since `b` and `c` are encoded as integer×1000 and divided, the only
/// rounding is in the division and sum of 8 terms.  1e-3 accommodates.
pub const GPU_SPATIAL_PAYOFF_F32: f64 = 1e-3;

/// GPU batch IPR: inverse participation ratio (f32).
///
/// `IPR = sum(|ψ_i|^4)` accumulates `dim` fourth-power terms in f32.
/// For `dim` ≤ 64, f32 sum is accurate to ~4 digits.  1e-3 accommodates.
pub const GPU_BATCH_IPR_F32: f64 = 1e-3;

/// GPU pairwise Hamming: proportional Hamming distance (f32).
///
/// Hamming = `diff_count` / `seq_len`.  The count is exact (integer);
/// the division is the only f32 rounding source.  1e-6 is conservative.
pub const GPU_HAMMING_F32: f64 = 1e-6;

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
    #[allow(clippy::assertions_on_constants)]
    fn tolerance_ordering() {
        assert!(
            EXACT_F64 < CROSS_LANGUAGE,
            "exact should be tighter than cross-language"
        );
        assert!(
            SOFTMAX_CROSS_PYTHON < CROSS_LANGUAGE,
            "softmax should be tighter than cross-language"
        );
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
            CD_COMPARABLE_DIST,
            ADIABATIC_KL_GAP,
            HMM_POSTERIOR_SUM,
            QS_VARIANCE_MAX,
            GPU_F64_EXACT,
            GPU_F64_TRANSCENDENTAL,
            GPU_F64_STATS,
            ML_MLP_F32,
            ML_TRANSFORMER_F32,
            FFT_INVERSE_F32,
            FFT_INVERSE_F64,
            FFT_PARSEVAL_F32,
            FFT_PARSEVAL_F64,
            FFT_KNOWN_PAIR_F32,
            FFT_KNOWN_PAIR_F64,
            FFT_SPECTRAL_LEAKAGE_F32,
            FFT_SPECTRAL_LEAKAGE_F64,
            RFFT_DC_COMPONENT_F32,
            GPU_HMM_LOG_LIKELIHOOD_F32,
            GPU_HMM_ALPHA_F32,
            GPU_FITNESS_F32,
            GPU_RK4_F32,
            GPU_JACCARD_F32,
            GPU_LOCUS_VARIANCE_F32,
            GPU_SPATIAL_PAYOFF_F32,
            GPU_BATCH_IPR_F32,
            GPU_HAMMING_F32,
        ];
        for (i, &t) in tols.iter().enumerate() {
            assert!(t > 0.0, "tolerance index {i} must be positive, got {t}");
        }
    }
}
