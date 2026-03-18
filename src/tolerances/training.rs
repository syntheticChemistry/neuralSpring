// SPDX-License-Identifier: AGPL-3.0-or-later

//! Training, model, and quantization tolerances.
//!
//! Covers MLP surrogate thresholds, transformer cross-framework agreement,
//! PINN/DeepONet convergence, quantization degradation bounds, and
//! training-monitor policy knobs (spectral early-stopping).

// ═══════════════════════════════════════════════════════════════════
// Training / model tolerances (Python baselines)
// ═══════════════════════════════════════════════════════════════════

/// MLP surrogate R² threshold (minimum acceptable).
///
/// FAO-56 ET₀ MLP achieves R² > 0.95 consistently with seed=42.
/// Benchmark functions (Rastrigin) may be lower due to multimodality.
///
/// Provenance: `control/surrogate/surrogate_validation.py` (seed=42, commit `BASELINE_COMMIT`)
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
///
/// Provenance: `control/sequence/sequence_forecasting.py` (seed=42, commit `BASELINE_COMMIT`)
pub const SEQUENCE_R2_MIN: f64 = 0.80;

/// PINN L2 relative error threshold (Adam-only, no L-BFGS).
///
/// Paper achieves 0.06% with L-BFGS. Adam-only reaches ~5%.
/// 15% is the acceptance threshold.
pub const PINN_L2_ERROR_MAX: f64 = 0.15;

/// PINN Cole-Hopf initial condition: u(0,x) = -sin(πx).
///
/// The Cole-Hopf transformation at t=0 reduces to the IC analytically.
/// Machine precision applies (no iteration or quadrature).
pub const PINN_IC_EXACT: f64 = super::EXACT_F64;

/// PINN boundary condition: u(t,±1) ≈ 0.
///
/// For ν = 0.01/π, the boundary values are exponentially small.
/// 0.01 catches implementation errors in the Cole-Hopf quadrature.
pub const PINN_BC_TOLERANCE: f64 = 0.01;

/// PINN shock steepening ratio: gradient(t=1) / gradient(t=0).
///
/// Burgers' equation steepens by construction. A ratio below 1.5
/// would indicate the solution failed to develop a shock front.
pub const PINN_SHOCK_RATIO_MIN: f64 = 1.5;

/// `DeepONet` antiderivative: max error for known analytical operators.
///
/// Tests u=1→y, u=x→y²/2, u=x²→y³/3. These are exact integrations
/// of low-degree polynomials, so machine precision applies.
pub const DEEPONET_EXACT_ANTIDERIV: f64 = super::EXACT_F64;

/// `DeepONet` dataset generation: polynomial evaluation consistency.
///
/// Verifying that eval + antiderivative round-trip is exact.
pub const DEEPONET_POLYNOMIAL_EXACT: f64 = super::EXACT_F64;

/// INT8 quantization: max R² degradation from FP32.
///
/// Measured: 0.017%. Threshold: 1%.
pub const QUANT_INT8_DEGRADATION: f64 = 0.01;

/// INT4 quantization: max R² degradation from FP32.
///
/// Measured: 0.79%. Threshold: 5%.
pub const QUANT_INT4_DEGRADATION: f64 = 0.05;

/// INT8 dequantization: max per-element error.
///
/// 256 quantization levels in `[-128, 127]` produce at most 0.5
/// quantization step error per element.
pub const QUANT_Q8_ELEMENT_ERROR: f64 = 0.5;

/// INT4 dequantization: max per-element error.
///
/// 16 quantization levels produce at most 1.0 quantization step
/// error per element.
pub const QUANT_Q4_ELEMENT_ERROR: f64 = 1.0;

// ═══════════════════════════════════════════════════════════════════
// Training monitor policy thresholds
// ═══════════════════════════════════════════════════════════════════
// Training-loop policy knobs rather than validation tolerances,
// centralized here for runtime introspection via the registry.

/// Bandwidth growth factor that triggers Yellow attention.
///
/// When consecutive-epoch bandwidth ratio exceeds this threshold,
/// the monitor escalates from Green to Yellow. 2.0 means the spectral
/// bandwidth doubled in one epoch — a sign of rapidly changing weight
/// structure. Adapted from hotSpring `BrainInterrupt` thresholds.
pub const TRAINING_BANDWIDTH_GROWTH: f64 = 2.0;

/// Loss stall detection: max loss range over the stall window.
///
/// When the loss range over the last `LOSS_STALL_WINDOW` epochs falls
/// below this threshold, training is considered stalled. 1e-6 is tight
/// enough to catch genuine plateaus while ignoring f64 rounding noise.
pub const TRAINING_LOSS_STALL: f64 = 1e-6;

/// Bandwidth explosion factor that triggers Red attention.
///
/// 5× bandwidth increase in one epoch indicates weight matrix instability
/// (e.g., exploding gradients manifesting spectrally).
pub const TRAINING_BANDWIDTH_EXPLOSION: f64 = 5.0;

/// IPR collapse threshold that triggers Red attention and early stop.
///
/// IPR < 0.01 means eigenstates are fully localized — the network is
/// memorizing rather than generalizing. For n=64 weight matrices, the
/// extended baseline is IPR ≈ 1/64 ≈ 0.016; 0.01 is slightly below.
pub const TRAINING_IPR_COLLAPSE: f64 = 0.01;

/// Loss divergence factor that triggers Red attention and early stop.
///
/// If current loss exceeds 10× the previous epoch's loss, training has
/// diverged (e.g., learning rate too high, NaN propagation beginning).
pub const TRAINING_LOSS_DIVERGENCE: f64 = 10.0;

/// Learning rate reduction factor for corrective Yellow/Red actions.
///
/// Halving the LR is a conservative intervention that reduces
/// gradient magnitude without requiring a full restart.
pub const TRAINING_LR_REDUCTION: f64 = 0.5;
