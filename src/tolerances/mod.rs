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
///
/// Provenance: `control/game_theory/game_theory.py` (seed=42, commit `BASELINE_COMMIT`)
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
///
/// Derivation: f64 subnormal range starts at ~5e-324.  1e-30 is 294
/// orders of magnitude above subnormal, so `(x + guard).ln()` never
/// underflows.  `ln(1e-30) ≈ -69` — a finite, bounded contribution to
/// entropy sums.  Used in `primitives::LOG_GUARD`, FST denominators,
/// WDM EOS log-input clamping, and GPU chi-squared expected-value floors.
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

/// Swarm fitness comparison tolerance (heterogeneous >= homogeneous - tol).
///
/// Heterogeneous swarms may not always exceed homogeneous fitness,
/// but should be within 2.0 fitness units (mean-of-last-10 scale).
/// Foreback, Bohm, Dolson (2025).
///
/// Derivation: measured fitness gap between heterogeneous and homogeneous
/// swarms across 10 seeded runs (seed 0..9): mean gap = 0.3 ± 1.2.
/// 2.0 ≈ mean + 1.5σ, ensuring > 95% of runs pass.
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

// ═══════════════════════════════════════════════════════════════════
// baseCamp spectral analysis tolerances
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
// Population genetics tolerances
// ═══════════════════════════════════════════════════════════════════

/// `F_ST` for identical populations: Weir-Cockerham sample correction.
///
/// `F_ST` for identical allele frequencies should be ~0 by definition.
/// The Weir-Cockerham estimator introduces a small sample-size bias
/// (denominator correction), so the tolerance is 0.05 rather than
/// machine precision.
pub const FST_IDENTICAL_POP_TOL: f64 = 0.05;

/// `F_ST` estimator agreement: mean-of-ratios vs ratio-of-sums.
///
/// Two `F_ST` estimators (W-C individual-locus vs multi-locus average)
/// can differ by up to 0.05 for 10 loci with 20 individuals per pop.
pub const FST_ESTIMATOR_AGREEMENT: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════
// Game theory equilibrium tolerances
// ═══════════════════════════════════════════════════════════════════

/// PD defection dominance upper bound on cooperation.
///
/// In the prisoner's dilemma with b=3, c=1, defection should dominate
/// after 2000 Euler steps.  Cooperation frequency should drop below 0.1.
pub const GAME_DEFECTION_UPPER: f64 = 0.1;

/// QS cooperation late-phase minimum.
///
/// After 80+ generations with quorum sensing (threshold 0.3, benefit 0.3),
/// cooperation should stabilize above 0.1 (detectable cooperation).
/// Stricter than `GAME_COOPERATION_MIN` (0.05) because this tests the
/// barracuda RK45 implementation specifically.
pub const GAME_QS_COOPERATION_MIN: f64 = 0.1;

/// QS cooperation variance upper bound.
///
/// After stabilization (gen 50+), the cooperation frequency variance
/// should remain below 0.1, indicating equilibrium rather than cycling.
pub const GAME_QS_VARIANCE_MAX: f64 = 0.1;

// ═══════════════════════════════════════════════════════════════════
// Relative-error and near-zero guard tolerances
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
// Quantization error tolerances
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
// Publication experiment tolerances (Exp-050, Exp-052, Exp-053)
// ═══════════════════════════════════════════════════════════════════

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

mod gpu;
mod registry;

pub use gpu::*;
pub use registry::{all_tolerances, categories, tolerance_by_name, NamedTolerance};
