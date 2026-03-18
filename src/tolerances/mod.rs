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
///
/// Validated: IEEE 754-2019 §5.3.  Empirically confirmed across all 53
/// Phase 0++ experiments (commit `f9ad0268`, 2026-02-16).
pub const EXACT_F64: f64 = 1e-12;

/// Tolerance for cross-language validation (Rust vs Python/NumPy).
///
/// Both use IEEE 754 f64 but may differ in operation ordering, FMA usage,
/// and library implementations of transcendentals (sin, cos, exp, tanh).
/// 1e-10 accounts for these differences.
///
/// Validated: Measured max deviation 3.2e-11 across 260+ validation binaries
/// (commit `f9ad0268`, 2026-02-16).  `NumPy` 2.2.6, Rust 1.87, libm.
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
///
/// Provenance: `control/isomorphic/isomorphic_catalog.py` (seed=42, commit `BASELINE_COMMIT`)
pub const BENCHMARK_CROSS_PYTHON: f64 = CROSS_LANGUAGE;

/// Nelder-Mead convergence to known minimum position.
///
/// 3 digits sufficient for stochastic simplex.
pub const OPTIMIZER_POSITION: f64 = 1e-3;

/// Nelder-Mead on multimodal (Rastrigin) — may find local basin.
///
/// Rastrigin has local minima spaced ~1.0 apart; Nelder-Mead can converge
/// to any basin.  0.1 verifies the simplex reached the interior of a basin
/// (within 10% of the spacing).  Tighter values would require global
/// optimization, which Nelder-Mead does not guarantee.
pub const OPTIMIZER_POSITION_MULTIMODAL: f64 = 0.1;

/// Function value at converged minimum.
///
/// For Rosenbrock at (1,1) = 0.0, 1e-4 verifies convergence to 4 significant
/// digits.  Nelder-Mead typically achieves ~6 digits on smooth unimodal
/// functions; 1e-4 is the tightest threshold robust to 500-iteration budgets.
pub const OPTIMIZER_VALUE_AT_MIN: f64 = 1e-4;

/// Function value bound for multimodal convergence.
///
/// Rastrigin's local minima have values A·k (k = number of off-center dims,
/// A ≈ 10).  Nelder-Mead from random start may land in a basin with value
/// up to ~10.  1.0 constrains the optimizer to the global basin or an
/// immediate neighbor — the tightest bound achievable without restarts.
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
///
/// Provenance: `python3 -c "import numpy as np; x=np.arange(1,6,dtype=np.float64); ..."`;
/// see `provenance::SOFTMAX_1_TO_5` for exact command.
pub const SOFTMAX_CROSS_PYTHON: f64 = 1e-14;

/// GELU: Rust vs Python at reference points.
///
/// The tanh approximation involves sqrt(2/pi) and x^3 terms.
/// `NumPy`'s tanh may use different polynomial coefficients than Rust's libm.
///
/// Provenance: `python3 -c "import numpy as np; ..."`;
/// see `provenance::GELU_REFERENCE` for exact command and reference values.
pub const GELU_CROSS_PYTHON: f64 = EXACT_F64;

/// GELU: large input approximation (GELU(x) ≈ x for x >> 0).
///
/// At x = 10, GELU(10) ≈ 10.0 to within 1e-6.
pub const GELU_LARGE_INPUT: f64 = 1e-6;

/// Sigmoid saturation at extreme inputs: σ(x) → 1 for x >> 0, σ(x) → 0 for x << 0.
///
/// At x = ±10, σ(10) ≈ 0.99995 and σ(-10) ≈ 4.5e-5.  The residual
/// |σ(10) - 1| ≈ 4.5e-5, so 1e-4 provides ~2x margin.  Used in
/// cross-spring evolution validators to confirm saturation behavior
/// without requiring f64-exact agreement at the tails.
pub const SIGMOID_SATURATION: f64 = 1e-4;

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

mod evolutionary;
mod training;

pub use evolutionary::*;
pub use training::*;

// ═══════════════════════════════════════════════════════════════════
// ODE integrator configuration
// ═══════════════════════════════════════════════════════════════════

/// Default RK45 absolute tolerance for adaptive ODE integration.
///
/// 1e-8 is standard for biological ODE systems (GRN, replicator dynamics)
/// where state variables are O(1).  Matches `SciPy` `solve_ivp` default.
///
/// Validated: `SciPy` 1.14.1 `solve_ivp(method='RK45', atol=1e-8)`.
/// Confirmed via `control/regulatory_network/regulatory_network.py`
/// (commit `f9ad0268`, 2026-02-16).
pub const ODE_ATOL: f64 = 1e-8;

/// Default RK45 relative tolerance for adaptive ODE integration.
///
/// 1e-6 balances accuracy against step count for smooth ODE systems.
/// Matches `SciPy` `solve_ivp` default.
///
/// Validated: `SciPy` 1.14.1 `solve_ivp(method='RK45', rtol=1e-6)`.
pub const ODE_RTOL: f64 = 1e-6;

// ═══════════════════════════════════════════════════════════════════
// Numerical stability guards
// ═══════════════════════════════════════════════════════════════════

/// Guard for logarithm inputs to avoid `ln(0)`.
///
/// 1e-30 is small enough to not affect results but large enough to
/// prevent `-inf` in KL divergence, cross-entropy, and similar.
///
/// Derivation: f64 subnormal range starts at ~5e-324.  1e-30 is 294
/// orders of magnitude above subnormal, so `(x + guard).ln()` never
/// underflows.  `ln(1e-30) ≈ -69` — a finite, bounded contribution to
/// entropy sums.  Used in `primitives::LOG_GUARD`, FST denominators,
/// WDM EOS log-input clamping, and GPU chi-squared expected-value floors.
pub const LOG_ZERO_GUARD: f64 = 1e-30;

/// Minimum fitness value to prevent zero-fitness stalling in selection.
///
/// In population-based EAs (eco-dynamics, directed evolution), zero-fitness
/// individuals would never be selected, stalling the algorithm.  1e-10 is
/// negligible compared to typical fitnesses O(0.01–1.0) but prevents
/// division by zero in proportional selection and log-fitness metrics.
pub const FITNESS_FLOOR: f64 = 1e-10;

/// Lexicase selection epsilon (Dolson et al. 2022, eLife 11:e79665).
///
/// Candidates within `LEXICASE_EPSILON` of the best fitness on a given
/// objective are retained during case filtering.  Tight enough to
/// discriminate meaningfully different fitnesses, loose enough for
/// floating-point rounding in multi-objective sums.
pub const LEXICASE_EPSILON: f64 = 1e-8;

/// Layer normalization epsilon (f32 numerical stability).
///
/// Prevents division by zero in variance normalization.  Matches
/// `PyTorch` default `LayerNorm(eps=1e-5)`.
///
/// Validated: `PyTorch` 2.2.0 `torch.nn.LayerNorm` default eps=1e-5.
/// Confirmed in `control/transformer/transformer_inference.py` (commit
/// `f9ad0268`, 2026-02-16).
pub const LAYER_NORM_EPS: f64 = 1e-5;

/// Hessian finite-difference step size.
///
/// Central difference `h` for numerical Hessian computation.
/// 1e-5 balances truncation error O(h²) against cancellation noise O(eps/h²).
pub const HESSIAN_FD_STEP: f64 = 1e-5;

/// SVD pseudo-inverse singular-value cutoff.
///
/// Singular values below this threshold are treated as zero when computing
/// the Moore-Penrose pseudo-inverse via SVD.  `1e-10` is well above f64
/// machine epsilon (~2.2e-16) and prevents near-singular matrices from
/// amplifying numerical noise in the pseudo-inverse.
pub const SVD_PINV_CUTOFF: f64 = 1e-10;

/// NMF convergence tolerance for iterative multiplicative-update NMF.
///
/// Frobenius-norm objective decrease below this threshold between
/// iterations signals convergence.  `1e-6` provides 6 significant
/// digits of reconstruction fidelity.
pub const NMF_CONVERGENCE_TOL: f64 = 1e-6;

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
/// divide-and-conquer achieves 1e-14.  `BarraCUDA` roadmap: upgrade to
/// `divide_and_conquer` for machine-precision eigendecomposition.
///
/// Validated: Measured `‖QΛQᵀ - A‖/‖A‖` = 7.3e-3 for random 8×8
/// symmetric matrices (`barraCuda` v0.3.5, commit `f9ad0268`, 2026-02-16).
pub const EIGH_JACOBI_RECONSTRUCT: f64 = 1e-2;

/// Jacobi eigensolver: eigenvalue agreement (f64, n≤8).
///
/// Eigenvalue accuracy is tighter than reconstruction because orthogonal
/// similarity transforms preserve the spectrum even when eigenvectors
/// are only approximate.
pub const EIGH_JACOBI_EIGENVALUE: f64 = 1e-3;

/// GPU Jacobi eigensolver: convergence threshold per sweep.
///
/// `BatchedEighGpu::execute_single_dispatch` iterates Jacobi rotations until
/// off-diagonal elements fall below this threshold.  1e-12 matches the
/// CPU Householder+QR standard (`EXACT_F64`) and ensures GPU eigenvalues
/// agree with CPU to machine precision for small matrices (n ≤ 32).
pub const JACOBI_GPU_CONVERGENCE: f64 = 1e-12;

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

/// PINN finite-difference PDE residual upper bound.
///
/// Mean absolute PDE residual of the Cole-Hopf exact solution evaluated
/// on a coarse FD grid (10×40).  FD truncation error dominates at
/// O(Δt + Δx²) ≈ O(0.1).  10.0 is generous for the coarse grid.
pub const PINN_FD_RESIDUAL_MAX: f64 = 10.0;

/// Seasonal temperature model annual mean (DC offset ≈ 8.5°C).
///
/// The synthetic cosine model `T(d) = 8.5 - 17·cos(2πd/365)` has
/// analytical mean 8.5°C over a continuous period.  Discrete sampling
/// at 365 points introduces a residual `O(1/N)` bias from the
/// trapezoidal-rule approximation of cos over one period.
///
/// Provenance: `control/sequence/sequence_forecasting.py` (seed=42,
/// `BASELINE_COMMIT`).  Analytical derivation: `∫₀¹ cos(2πt) dt = 0`,
/// so the mean of `8.5 - 17·cos` is exactly 8.5.
pub const SEASONAL_ANNUAL_MEAN: f64 = 8.5;

/// Seasonal temperature model mean tolerance.
///
/// 0.5°C accounts for discrete-sampling bias (365 points) and any
/// floating-point accumulation in the mean computation.  Observed
/// deviation from analytical 8.5°C is < 0.05°C; the 0.5°C margin
/// is conservative to accommodate alternative sampling resolutions.
///
/// Provenance: `control/sequence/sequence_forecasting.py` (seed=42,
/// `BASELINE_COMMIT`).
pub const SEASONAL_ANNUAL_MEAN_TOL: f64 = 0.5;

/// Spectral theory: Jacobi (dense) vs Sturm bisection (tridiag)
/// eigenvalue agreement.
///
/// Two fundamentally different eigensolvers produce spectra that
/// differ by ~1e-2 for n=64 Aubry-André Hamiltonians.
pub const SPECTRAL_EIGENSOLVER_CROSS: f64 = 0.05;

/// IPR cross-validation: Rust-recomputed vs Python-baseline IPR.
///
/// Isomorphic reservoir ensemble IPR values (mean IPR across weight
/// matrices) agree within 0.005 between Rust and Python.  The tolerance
/// covers f64 operation-ordering differences in the IPR summation
/// (1/Σ ψᵢ⁴) and eigenvector sign conventions.
///
/// Provenance: `control/isomorphic_reservoir/isomorphic_reservoir_baseline.json`
/// (seed=42, commit `BASELINE_COMMIT`)
pub const IPR_CROSS_PYTHON: f64 = 0.005;

/// Spectral theory: Kappus-Wegner anomaly γ(E=0) ≈ W²/96 for
/// small disorder.
///
/// Statistical agreement with the Kappus-Wegner formula requires
/// many realizations.  50% relative error threshold for N=5000, 50 realizations.
///
/// Derivation: the anomaly γ = W²/96 has statistical variance
/// σ² ∝ `1/N_realizations`.  With 50 realizations, standard error of
/// the mean is ~14% of the true value.  0.5 provides 3σ margin.
pub const KAPPUS_WEGNER_REL: f64 = 0.5;

/// Spectral theory: level spacing ratio distance from Poisson (localized).
///
/// For strong Anderson disorder (W=8), the mean spacing ratio should
/// be within 0.05 of the Poisson value 2ln(2)-1 ≈ 0.386.
pub const LEVEL_SPACING_POISSON_TOL: f64 = 0.05;

/// Level spacing ratio: GOE vs Poisson comparison slack.
///
/// Random matrices have level spacing between GOE (≈0.5307) and Poisson (≈0.386).
/// The comparison `d(GOE) < d(Poisson) + slack` allows for finite-size fluctuations
/// at n=16.  0.2 is conservative for the small Hamiltonian dimensions used in
/// baseCamp spectral analysis.
pub const LEVEL_SPACING_GOE_SLACK: f64 = 0.2;

/// Spectral IPR comparison: low-rank vs random perturbation slack.
///
/// When comparing IPR between random and low-rank weight matrices, the
/// low-rank matrix should have higher IPR (more localized eigenstates).
/// -0.5 allows for stochastic variation in small (8x8) matrices.
pub const SPECTRAL_IPR_COMPARISON_SLACK: f64 = -0.5;

/// Numerical distinctness: minimum difference to confirm two computed
/// values are not identical.
///
/// Used in spectral analysis to verify that different architectures
/// or perturbation depths produce measurably different results.
/// 1e-15 is just above f64 rounding noise for O(1) values.
pub const NUMERICAL_DISTINCTNESS: f64 = 1e-15;

/// Gate disorder comparison slack.
///
/// When asserting that steeper sigmoid gates produce higher disorder
/// parameter, allow small negative margin for numerical noise in the
/// variance-based disorder calculation over 50 gate values.
pub const GATE_DISORDER_COMPARISON: f64 = 0.01;

/// Spectral radius sweep monotonicity slack.
///
/// When asserting that Jacobian spectral radius increases with weight
/// scale sigma\_w, allow 0.1 negative margin for stochastic variation
/// across 5 sweep points with 4x4 random matrices.
pub const SPECTRAL_RADIUS_SWEEP_SLACK: f64 = 0.1;

/// Relative error floor: minimum denominator for relative error.
///
/// When computing relative error |observed - expected| / |expected|,
/// the expected value must exceed this floor to avoid division
/// amplification of noise.  10x wider than `ZERO_DETECTION` because
/// relative-error denominators are more sensitive to small values.
pub const RELATIVE_ERROR_FLOOR: f64 = 1e-10;

/// ODE steady-state approach: convergence to equilibrium.
///
/// When verifying that an ODE trajectory approaches its steady state
/// (e.g., carrying capacity K), the final value should be within 0.5
/// of the target.  Generous because f32 GPU RK4 accumulates error
/// over 1000+ steps.
pub const ODE_STEADY_STATE_SLACK: f64 = 0.5;

/// INT8 random GEMV L2 error: 256 quantization levels on random matrices.
///
/// INT8 quantization of a random weight matrix produces L2 relative
/// error < 5% vs FP32 reference for 64x64 matrices.
pub const QUANT_Q8_GEMV_ERROR: f64 = 0.05;

/// INT4 random GEMV L2 error: 16 quantization levels on random matrices.
///
/// INT4's 16 quantization levels produce up to 25% L2 error on
/// random 64x64 matrices.  This is a generous bound for the
/// coarse quantization; production INT4 uses grouped quantization.
pub const QUANT_Q4_GEMV_ERROR: f64 = 0.25;

/// Quantized sign agreement slack.
///
/// When verifying sign agreement between quantized and full-precision
/// outputs, allow the full-precision value to be within this threshold
/// of zero before checking sign.  Values near zero have ambiguous sign.
pub const QUANT_SIGN_AGREEMENT: f64 = 0.1;

/// GOE random-matrix level spacing ratio target.
///
/// Wigner surmise for GOE: LSR ≈ 0.5307. For finite n=64 matrices the
/// value fluctuates; 0.10 covers the typical spread observed in Python
/// baselines (MLP final LSR ≈ 0.56, CNN ≈ 0.52).
pub const GOE_LSR_TOLERANCE: f64 = 0.10;

/// IPR ratio spread threshold for size-independence test.
///
/// Anderson localization theory predicts the normalized IPR ratio
/// (high-disorder / low-disorder) is size-independent. 40% spread
/// across system sizes (N=64..512) accounts for finite-size effects
/// and boundary conditions in lattice coordination models.
pub const IPR_RATIO_SPREAD_MAX: f64 = 0.40;

// ═══════════════════════════════════════════════════════════════════
// coralForge numerical stability
// ═══════════════════════════════════════════════════════════════════

/// Division guard for coralForge primitives (softmax, norms).
///
/// Prevents division by zero in attention score normalization,
/// triangle update norms, and MSA column aggregation.  Matches
/// `EXACT_F64` (1e-12) — the same guard used across all crate modules.
pub const FOLDING_EPS: f64 = EXACT_F64;

/// SDPA output tolerance for accumulated softmax→matmul chains.
///
/// Uniform Q/K produce uniform attention weights; the V passthrough
/// result should equal V exactly, but softmax exp/sum and the
/// subsequent weighted average accumulate ~6 digits of rounding.
/// 1e-6 matches the observed residual in 4-head, `d_model`=4 tests.
pub const SDPA_PASSTHROUGH: f64 = 1e-6;

/// Cosine beta schedule alpha-bar floor for diffusion models.
///
/// Clamps `alpha_bar` from below to prevent `sqrt(alpha_bar)` or
/// `1 - alpha_bar` from hitting zero during forward diffusion.
/// `1e-10` is 2 orders above the smallest `alpha_bar` in a 1000-step
/// cosine schedule (Ho et al. "DDPM" `NeurIPS` 2020).
pub const DIFFUSION_ALPHA_BAR_FLOOR: f64 = 1e-10;

/// Cosine beta schedule per-step beta floor.
///
/// Prevents degenerate zero-noise steps at the start of the schedule.
/// 1e-6 ensures every step adds measurable noise while staying well
/// below typical beta values (~0.001–0.02).
pub const DIFFUSION_BETA_FLOOR: f64 = 1e-6;

// ═══════════════════════════════════════════════════════════════════
// Domain-specific numerical guards
// ═══════════════════════════════════════════════════════════════════

/// Fisher information metric floor for counterdiabatic driving.
///
/// The Fisher metric `g(s) = β²·Var_s[F]` vanishes at landscape saddle
/// points, causing geodesic speed `ds/dt → ∞`. This floor caps the speed
/// while remaining negligible compared to typical `g(s) ∈ [1e-4, 1]` at β=1.
pub const FISHER_EPS: f64 = CROSS_LANGUAGE;

/// Cole-Hopf initial-condition guard for Burgers' equation.
///
/// At `t = 0` the exact solution is `u(0,x) = -sin(πx)`, which avoids
/// the expensive quadrature. This threshold detects the IC case; `1e-12`
/// provides clean separation from any `t > 0`.
pub const BURGERS_IC_GUARD: f64 = EXACT_F64;

/// DP traceback equality guard for sequence alignment.
///
/// In Needleman-Wunsch / Gotoh traceback, floating-point scores are
/// compared to determine which predecessor cell was optimal. `1e-10`
/// accounts for accumulated rounding in the forward pass while being
/// well above machine epsilon.
pub const DP_EQUALITY_EPS: f64 = CROSS_LANGUAGE;

/// Singleton frequency detection guard for pangenome analysis.
///
/// Identifies genes present in exactly one genome by comparing against
/// the theoretical singleton frequency `1/N`. `1e-10` handles f64
/// rounding without false matches to non-singleton frequencies.
pub const SINGLETON_FREQ_EPS: f64 = CROSS_LANGUAGE;

/// Tie-breaking guard for regulatory network fate decisions.
///
/// When multiple phenotype signals are within this margin of the
/// maximum, the first (bioremediation) wins. `1e-10` prevents spurious
/// tie-breaks from floating-point noise in ODE integration.
pub const PHENOTYPE_TIE_EPS: f64 = CROSS_LANGUAGE;

/// Boolean-range validation slack: expected ∈ {0.0, 1.0}, tolerance ±0.5.
///
/// Used in integration-test binaries that verify boolean capabilities
/// (RPC method exists / returns valid result) via `check_abs(label, observed, expected, tol)`.
/// A result within 0.5 of the target indicates the capability is functional.
pub const BOOLEAN_VALIDATION_SLACK: f64 = 0.5;

/// Eigenvalue comparison for small analytical matrices (2×2, 3×3).
///
/// Exact eigenvalues of small integer matrices are known analytically.
/// GPU f64 dispatch computes eigenvalues via iterative methods; 0.01
/// covers the combined f64→f32→f64 roundtrip and solver convergence.
pub const EIGENSOLVER_SMALL_MATRIX: f64 = 0.01;

/// GPU trace parity after f64→f32 tensor roundtrip.
///
/// Matrix trace computed via f32 GPU matmul (A×I readback) differs from
/// the f64 CPU trace by the accumulated f32 mantissa truncation across
/// N diagonal elements.  For N ≤ 16, the per-element error is O(2⁻²⁴)
/// × |element|, summing to ~0.005 for typical attention matrices.
/// 0.01 provides 2× margin.
///
/// Provenance: `control/attention_anderson/attention_anderson_baseline.json`
pub const GPU_TRACE_F32_ROUNDTRIP: f64 = 0.01;

/// Pearson correlation cross-validation (Rust vs Python).
///
/// Pearson r computed over 32 temperature slices; both implementations
/// use the same formula but differ in summation order and FMA usage.
/// 0.05 covers the numerical spread observed in 5-surrogate ensembles
/// where individual r values range from −0.7 to −0.4.
///
/// Provenance: `control/wdm_ensemble_qs/wdm_ensemble_qs_baseline.json`
pub const CORRELATION_CROSS_VALIDATION: f64 = 0.05;

/// GPU f32 reduction accumulation over N-element arrays.
///
/// Summation via f32 matmul (dot with ones vector) accumulates
/// per-element truncation error proportional to N × 2⁻²⁴ × |mean|.
/// For N=10 disorder values with |mean| ≈ 5, total error ≈ 0.003.
/// 0.1 provides generous margin for larger arrays and driver variance.
pub const GPU_ACCUMULATION_F32: f64 = 0.1;

/// Binary classifier metric parity (TPR, FPR, Accuracy).
///
/// Detection metrics from Viterbi path decoding are discrete ratios
/// (count / N).  Rust and Python agree exactly when paths match;
/// 0.01 covers single-path-element disagreements at N=100.
///
/// Provenance: `control/introgression_nn/introgression_nn_baseline.json`
pub const CLASSIFIER_METRIC_CROSS: f64 = 0.01;

/// HMM introgression fraction cross-validation.
///
/// Fraction of layers classified as introgressed by Viterbi decoding.
/// Both implementations use identical transition/emission parameters;
/// 0.05 covers boundary effects where a single state flip at a
/// transition boundary changes the fraction by 1/N = 0.01 at N=100.
pub const INTROGRESSION_FRACTION_CROSS: f64 = 0.05;

/// Process model response function peak agreement.
///
/// Gaussian-envelope temperature response curves (mesophilic at 35°C,
/// thermophilic at 55°C) have broad peaks where ±1°C shifts produce
/// ≤2% change.  0.05 covers the difference between the analytical
/// dual-Gaussian model and ESN-learned response curves.
///
/// Provenance: `control/digestion_prediction/digestion_prediction_baseline.json`
pub const PROCESS_MODEL_RESPONSE: f64 = 0.05;

/// RPC count fallback tolerance for error-path validation.
///
/// When an RPC call fails, the expected count is compared against
/// the zero-value fallback.  This tolerance is semantically a "count
/// proximity" guard — generous because the check exists to document
/// the failure path, not to validate numerical accuracy.
pub const RPC_COUNT_FALLBACK: f64 = 0.5;

// ═══════════════════════════════════════════════════════════════════
// GPU PRNG statistical tolerances
// ═══════════════════════════════════════════════════════════════════

/// Maximum deviation of GPU PRNG uniform mean from 0.5.
///
/// For N=1024 samples from U(0,1), the standard error of the mean is
/// σ/√N ≈ 0.289/32 ≈ 0.009.  Allowing ~2σ gives 0.02.  The observed
/// GPU mean must fall in \[0.48, 0.52\].
pub const GPU_PRNG_UNIFORMITY_MEAN: f64 = 0.02;

// ═══════════════════════════════════════════════════════════════════
// Domain-specific validation tolerances
// ═══════════════════════════════════════════════════════════════════

/// Glucose prediction CGM statistics (mean, std) vs Python baseline.
///
/// LSTM predictions on blood glucose time series produce f32 outputs
/// with training variance across random seeds. 1.0 mg/dL tolerance
/// accounts for f32 accumulation over 288-step sequences.
///
/// Provenance: `control/glucose_prediction/glucose_prediction.py` (seed=42)
pub const GLUCOSE_CGM_STAT_TOL: f64 = 1.0;

/// Glucose prediction pharmacokinetic τ (hours) vs Python baseline.
///
/// Exponential decay fit to glucose response curve; half-life τ
/// has ~10% relative uncertainty from finite-sample fitting.
pub const GLUCOSE_TAU_TOL: f64 = 0.5;

/// pLDDT confidence spread lower bound for non-degeneracy.
///
/// `AlphaFold3` `pLDDT` head should produce varying confidence scores;
/// if the spread (max - min) is below this, the head is degenerate.
pub const PLDDT_DEGENERACY_THRESHOLD: f64 = 1e-6;

/// GPU Kimura batch max element-wise difference.
///
/// Kimura 2-parameter distance involves transcendental functions
/// (log, sqrt) on f64. Batch dispatch accumulates per-element rounding.
/// For 1000-element batches, 1e-4 allows ~4 digits of precision.
pub const GPU_KIMURA_BATCH_DIFF: f64 = 1e-4;

/// GPU `ReLU` f32 determinism: maximum diff across identical runs.
///
/// `ReLU` is piecewise linear, so f32 should be nearly exact. 1e-7
/// accounts for flush-to-zero differences across GPU drivers.
pub const TENSOR_RELU_DETERMINISM_F32: f64 = 1e-7;

/// Division guard for standard deviation / variance denominators.
///
/// Prevents division by zero when normalizing time-series data.
/// `1e-12` matches `EXACT_F64` — a safe floor that is negligible
/// compared to typical CGM variance (~100–400 mg²/dL²).
pub const VARIANCE_DIVISION_GUARD: f64 = EXACT_F64;

/// Monotonicity comparison epsilon for diffusion schedules.
///
/// `alpha_bar` must decrease monotonically; `1e-15` is one order
/// above f64 machine epsilon (~2.2e-16), allowing rounding from
/// cumulative products while detecting genuine non-monotonicity.
pub const MONOTONICITY_EPS: f64 = 1e-15;

/// GPU hydrology kernel parity (Hargreaves ET₀).
///
/// Hargreaves equation involves sqrt and temperature arithmetic;
/// f64 GPU dispatch may differ from CPU by a few ULPs.  `1e-6`
/// accounts for 6 digits of agreement — sufficient for ET₀ in
/// mm/day (values typically 1–10).
pub const GPU_HYDROLOGY_F64: f64 = 1e-6;

// ═══════════════════════════════════════════════════════════════════
// Eigensolve / linalg tolerances
// ═══════════════════════════════════════════════════════════════════

/// CPU vs GPU eigenvalue agreement after sorting.
///
/// Householder-QR (CPU) and Jacobi/batched GPU eigensolvers use different
/// algorithms, accumulating different rounding. 1e-6 allows 6 digits of
/// agreement, which is sufficient for eigenvalues of random symmetric
/// matrices at sizes 8--64.
pub const GPU_EIGENVALUE_AGREEMENT: f64 = 1e-6;

/// Additive floor for variance parity checks (handrolled vs upstream).
///
/// Prevents division-by-zero in relative error when comparing variance
/// estimates across ddof conventions. Suppresses noise for near-zero
/// variance (uniform data). Distinct from `gpu::VARIANCE_FLOOR` which
/// is a readback lower-bound guard.
pub const VARIANCE_PARITY_FLOOR: f64 = 1e-10;

/// Pairformer / structure module element-wise parity.
///
/// `AlphaFold` pairformer blocks apply layer norms, GELUs, and outer
/// products that accumulate ~6 digits of rounding in f64.
pub const PAIRFORMER_PARITY: f64 = 1e-6;

// ═══════════════════════════════════════════════════════════════════
// Provenance date constants
// ═══════════════════════════════════════════════════════════════════

/// Baseline date for glucose prediction experiment (Paper 026).
///
/// Separate from `BASELINE_DATE` because Paper 026 baselines were
/// generated after the initial Phase 0 batch.
pub const GLUCOSE_BASELINE_DATE: &str = "2026-03-05";

mod gpu;
mod registry;

pub use gpu::*;
pub use registry::{NamedTolerance, all_tolerances, categories, tolerance_by_name};
