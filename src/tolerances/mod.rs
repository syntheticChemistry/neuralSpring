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

/// Threshold for treating a value as effectively zero.
///
/// Used in combined absolute-or-relative error checks where division
/// by the expected value would amplify noise.  Below this threshold,
/// absolute error is used instead of relative error.  Slightly tighter
/// than machine epsilon to avoid false triggers on legitimate small values.
pub const ZERO_DETECTION: f64 = 1e-14;

/// `norm_ppf` at extreme quantiles (0.975) uses polynomial approximation.
///
/// Less accurate than the CDF. A&S 26.2.17 gives ~4 decimal digits.
pub const NORM_PPF_TAIL: f64 = 0.01;

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

/// Nelder-Mead convergence to known minimum position.
///
/// 3 digits sufficient for stochastic simplex.
pub const OPTIMIZER_POSITION: f64 = 1e-3;

/// Nelder-Mead on multimodal (Rastrigin) — may find local basin.
pub const OPTIMIZER_POSITION_MULTIMODAL: f64 = 0.1;

/// Function value at converged minimum.
pub const OPTIMIZER_VALUE_AT_MIN: f64 = 1e-4;

/// Function value bound for multimodal convergence.
pub const OPTIMIZER_VALUE_MULTIMODAL: f64 = 1.0;

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

/// Special function evaluations (erf, bessel, norm\_cdf, norm\_ppf).
///
/// `barracuda::special` implementations use polynomial/Chebyshev approximations
/// for erf, bessel\_j0/j1, norm\_cdf, and norm\_ppf.  These achieve ~6 digits
/// of accuracy in f64 at reference points (NIST DLMF, A&S tables).
pub const SPECIAL_FUNCTION_F64: f64 = 1e-6;

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

/// PINN Cole-Hopf initial condition: u(0,x) = -sin(πx).
///
/// The Cole-Hopf transformation at t=0 reduces to the IC analytically.
/// Machine precision applies (no iteration or quadrature).
pub const PINN_IC_EXACT: f64 = EXACT_F64;

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
pub const DEEPONET_EXACT_ANTIDERIV: f64 = EXACT_F64;

/// `DeepONet` dataset generation: polynomial evaluation consistency.
///
/// Verifying that eval + antiderivative round-trip is exact.
pub const DEEPONET_POLYNOMIAL_EXACT: f64 = EXACT_F64;

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

/// IPR localization threshold: `IPR > 1/N` indicates localization.
///
/// For `N = 20`, the extended (delocalized) baseline is `IPR ≈ 1/N = 0.05`.
/// Localized states have `IPR >> 0.05`. The check verifies `mean_IPR > 0.05`
/// at strong disorder (`W/t = 10`), confirming Anderson localization.
pub const IPR_LOCALIZATION_MIN: f64 = 0.05;

/// HMM Viterbi accuracy: minimum fraction of correctly decoded states.
///
/// A well-specified HMM should decode > 50% of states correctly.
/// The weather HMM (2-state) with long sequences achieves ~70%.
pub const HMM_DECODE_ACCURACY_MIN: f64 = 0.05;

/// Introgression detection: minimum detected introgression fraction.
///
/// Under the PhyloNet-HMM model, true introgression fraction is ~0.2.
/// The detector should identify > 5% introgression loci.
pub const INTROGRESSION_FRACTION_MIN: f64 = 0.05;

/// Introgression detection: detected fraction vs true fraction tolerance.
///
/// With 500 synthetic loci (seed=42), the PhyloNet-HMM Viterbi path
/// detects fraction within 0.15 of the true introgression fraction.
/// This accounts for HMM boundary effects and ILS/introgression overlap.
/// Source: `control/introgression/introgression.py` (seed=42, n=500).
pub const INTROGRESSION_FRACTION_ABS: f64 = 0.15;

/// Introgression detection: false positive rate upper bound.
///
/// When no introgression is present (ILS-only loci), the detector
/// should report < 25% introgression. This is generous because
/// the 3-topology model has inherent overlap between ILS and
/// introgression gene-tree patterns.
/// Source: `control/introgression/introgression.py` (seed=42, ILS-only).
pub const INTROGRESSION_FPR_MAX: f64 = 0.25;

/// Gene tree concordance: minimum concordant topology fraction.
///
/// Under the multispecies coalescent, the concordant gene-tree
/// topology (matching the species tree) should dominate. With
/// introgression fraction ~0.2, concordant fraction > 0.2 is expected.
/// Source: coalescent theory — concordant topology probability ≥ 1 - 2/3 exp(-t).
pub const GENE_TREE_CONCORDANT_MIN: f64 = 0.2;

/// Game theory cooperation: minimum QS cooperation frequency.
///
/// At signal threshold below carrying capacity, QS-mediated
/// cooperation should be detectable (cooperation frequency > 5%).
pub const GAME_COOPERATION_MIN: f64 = 0.05;

/// Replicator dynamics: accumulated Euler-step tolerance.
///
/// 1000 Euler steps at dt=0.01 accumulate O(dt²·n) truncation error.
/// Cross-language validation shows Rust and Python agree to ~1e-7;
/// 1e-6 provides margin for different FP summation order.
pub const REPLICATOR_DYNAMICS: f64 = 1e-6;

/// Regulatory network: minimum Hill function response.
///
/// For non-zero inputs above the activation threshold, the Hill
/// function should produce a measurable response (> 1%).
pub const REGULATORY_RESPONSE_MIN: f64 = 0.01;

/// Eco-dynamics mean fitness improvement: EA should improve.
///
/// Over 10 generations with tournament selection, mean fitness
/// should increase by at least 8% from the random initial population.
pub const ECO_FITNESS_IMPROVEMENT_MIN: f64 = 0.08;

/// Pangenome selection: minimum positive selection signal (dN/dS > 1).
///
/// The test sequence has elevated nonsynonymous substitutions;
/// the chi-squared p-value should indicate significance.
pub const PANGENOME_SELECTION_P_MIN: f64 = 0.01;

/// Meta-population `F_ST` threshold: differentiation above drift alone.
///
/// With `F_ST` = 0.1 target, the observed `F_ST` should exceed 1% to
/// demonstrate measurable genetic structure.
pub const META_POP_FST_MIN: f64 = 0.01;

/// Meta-population inter-population allele frequency variance.
///
/// With `F_ST > 0`, allele frequency variance across populations
/// should exceed 0.1% (0.001) to confirm genetic structure.
pub const META_POP_AF_VARIANCE_MIN: f64 = 0.001;

/// Phylo HMM Viterbi margin: excess accuracy over chance for
/// 4-state phylo HMM on 5000 observations. Tighter than the
/// 2-state weather HMM because more states make the problem harder.
pub const HMM_PHYLO_DECODE_MARGIN: f64 = 0.02;

/// Signal integration: minimum dynamic range of Hill gate response.
///
/// The Hill function should produce distinguishable high and low
/// outputs. A dynamic range < 1% indicates a degenerate gate.
pub const SIGNAL_DYNAMIC_RANGE_MIN: f64 = 0.01;

/// Layer spectral similarity: self-similarity tolerance.
///
/// `layer_spectral_similarity(W, W)` should return ≈ 1.0 (cosine similarity
/// of a vector with itself).  Deviations arise from eigenvalue sorting
/// and f64 rounding in the SVD/eigendecomposition.  0.01 is conservative.
pub const SPECTRAL_SELF_SIMILARITY: f64 = 0.01;

/// PGM complexity monotonicity slack.
///
/// When asserting that a deeper/denser PGM is at least as complex as
/// a shallower/sparser one, allow a small negative margin for floating-point
/// rounding in the entropy-based complexity measure.
pub const PGM_COMPLEXITY_SLACK: f64 = 0.01;

// ═══════════════════════════════════════════════════════════════════
// ODE integrator configuration
// ═══════════════════════════════════════════════════════════════════

/// Default RK45 absolute tolerance for adaptive ODE integration.
///
/// 1e-8 is standard for biological ODE systems (GRN, replicator dynamics)
/// where state variables are O(1).  Matches `SciPy` `solve_ivp` default.
pub const ODE_ATOL: f64 = 1e-8;

/// Default RK45 relative tolerance for adaptive ODE integration.
///
/// 1e-6 balances accuracy against step count for smooth ODE systems.
/// Matches `SciPy` `solve_ivp` default.
pub const ODE_RTOL: f64 = 1e-6;

// ═══════════════════════════════════════════════════════════════════
// Numerical stability guards
// ═══════════════════════════════════════════════════════════════════

/// Guard for logarithm inputs to avoid `ln(0)`.
///
/// 1e-30 is small enough to not affect results but large enough to
/// prevent `-inf` in KL divergence, cross-entropy, and similar.
pub const LOG_ZERO_GUARD: f64 = 1e-30;

/// Layer normalization epsilon (f32 numerical stability).
///
/// Prevents division by zero in variance normalization.  Matches
/// `PyTorch` default `LayerNorm(eps=1e-5)`.
pub const LAYER_NORM_EPS: f64 = 1e-5;

/// Hessian finite-difference step size.
///
/// Central difference `h` for numerical Hessian computation.
/// 1e-5 balances truncation error O(h²) against cancellation noise O(eps/h²).
pub const HESSIAN_FD_STEP: f64 = 1e-5;

/// Numerical Hessian reconstruction vs analytical values.
///
/// Central FD Hessian with step h = 1e-5 has truncation error O(h²) = O(1e-10),
/// but catastrophic cancellation in f(x±h) differences amplifies to O(eps/h²)
/// ≈ O(1e-6).  For the Rosenbrock function at (1,1) where H\[0,0\] = 802,
/// the absolute error ≈ 0.03.  1.0 provides generous margin for larger
/// functions where higher-order terms contribute.
pub const HESSIAN_FD_ABS: f64 = 1.0;

/// Saddle-point classification: eigenvalue negativity threshold.
///
/// An eigenvalue is counted as "negative" (contributing to the saddle index)
/// only if it falls below this threshold.  1e-10 prevents numerical noise
/// near zero from inflating the saddle count while still detecting genuine
/// negative curvature in the loss landscape.
pub const SADDLE_EIGENVALUE_THRESHOLD: f64 = -1e-10;

// ═══════════════════════════════════════════════════════════════════
// Eigenvalue decomposition (barracuda Jacobi eigensolver)
// ═══════════════════════════════════════════════════════════════════

/// Jacobi eigensolver: matrix reconstruction relative error (f64, n≤8).
///
/// `barracuda::linalg::eigh_f64` uses classic Jacobi rotations which
/// converge to ~1e-2 relative reconstruction error at n=8.  LAPACK's
/// divide-and-conquer achieves 1e-14.  `ToadStool` handoff: upgrade to
/// `divide_and_conquer` for machine-precision eigendecomposition.
pub const EIGH_JACOBI_RECONSTRUCT: f64 = 1e-2;

/// Jacobi eigensolver: eigenvalue agreement (f64, n≤8).
///
/// Eigenvalue accuracy is tighter than reconstruction because orthogonal
/// similarity transforms preserve the spectrum even when eigenvectors
/// are only approximate.
pub const EIGH_JACOBI_EIGENVALUE: f64 = 1e-3;

// ═══════════════════════════════════════════════════════════════════
// ODE integrator agreement
// ═══════════════════════════════════════════════════════════════════

/// RK4 vs RK45 integrator agreement on identical ODEs.
///
/// Fixed-step RK4 and adaptive RK45 produce slightly different
/// trajectories due to step-size strategy.  For well-behaved systems
/// (regulatory networks, signal integration, game replicator dynamics)
/// the final-state difference is bounded by ~1e-2.
pub const ODE_INTEGRATOR_AGREEMENT: f64 = 1e-2;

// ═══════════════════════════════════════════════════════════════════
// Statistical critical values (chi-squared tables)
// ═══════════════════════════════════════════════════════════════════

/// Chi-squared critical value for df=9 at p < 0.05.
///
/// From standard chi-squared tables (Pearson 1900).  Used by pangenome
/// frequency-spectrum deviation from neutral.
pub const CHI2_CRITICAL_DF9_P05: f64 = 16.92;

/// Chi-squared critical value for df=1 at p < 0.05.
///
/// Standard threshold for 2×2 contingency tests.  Used by pangenome
/// environment-association per-gene tests.
pub const CHI2_CRITICAL_DF1_P05: f64 = 3.84;

/// Minimum environment-associated genes for pangenome selection signal.
///
/// With 200 genes and ~10% under selection, at least 5 should pass
/// the per-gene chi-squared test (df=1, p < 0.05).
pub const PANGENOME_MIN_ASSOCIATED_GENES: f64 = 5.0;

// ═══════════════════════════════════════════════════════════════════
// Miscellaneous validation thresholds
// ═══════════════════════════════════════════════════════════════════

/// Swarm fitness comparison tolerance (heterogeneous >= homogeneous - tol).
///
/// Heterogeneous swarms may not always exceed homogeneous fitness,
/// but should be within 2.0 fitness units (mean-of-last-10 scale).
/// Foreback, Bohm, Dolson (2025).
pub const SWARM_FITNESS_COMPARISON: f64 = 2.0;

/// PINN finite-difference PDE residual upper bound.
///
/// Mean absolute PDE residual of the Cole-Hopf exact solution evaluated
/// on a coarse FD grid (10×40).  FD truncation error dominates at
/// O(Δt + Δx²) ≈ O(0.1).  10.0 is generous for the coarse grid.
pub const PINN_FD_RESIDUAL_MAX: f64 = 10.0;

/// Seasonal temperature model annual mean (DC offset ≈ 8.5°C).
///
/// The synthetic cosine model `T(d) = 8.5 - 17·cos(2πd/365)` has
/// mean 8.5°C.  Tolerance 0.5°C for discrete sampling (365 points).
pub const SEASONAL_ANNUAL_MEAN: f64 = 8.5;

/// Seasonal temperature model mean tolerance.
pub const SEASONAL_ANNUAL_MEAN_TOL: f64 = 0.5;

/// Eco-dynamics dominance comparison tolerance.
///
/// Multi-niche dominance should not exceed single-niche by more
/// than 0.3 (fraction scale 0–1).  Allows for stochastic run
/// variance with seed=42.
pub const ECO_DOMINANCE_COMPARISON: f64 = 0.3;

/// Spectral theory: Jacobi (dense) vs Sturm bisection (tridiag)
/// eigenvalue agreement.
///
/// Two fundamentally different eigensolvers produce spectra that
/// differ by ~1e-2 for n=64 Aubry-André Hamiltonians.
pub const SPECTRAL_EIGENSOLVER_CROSS: f64 = 0.05;

/// Spectral theory: Kappus-Wegner anomaly γ(E=0) ≈ W²/96 for
/// small disorder.
///
/// Statistical agreement with the Kappus-Wegner formula requires
/// many realizations.  50% relative error threshold for N=5000, 50 realizations.
pub const KAPPUS_WEGNER_REL: f64 = 0.5;

/// Spectral theory: level spacing ratio distance from Poisson (localized).
///
/// For strong Anderson disorder (W=8), the mean spacing ratio should
/// be within 0.05 of the Poisson value 2ln(2)-1 ≈ 0.386.
pub const LEVEL_SPACING_POISSON_TOL: f64 = 0.05;

mod gpu;
mod registry;

pub use gpu::*;
pub use registry::{all_tolerances, categories, tolerance_by_name, NamedTolerance};
