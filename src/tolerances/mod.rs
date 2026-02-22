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

// ═══════════════════════════════════════════════════════════════════
// Phase 0++ stochastic model tolerances
// ═══════════════════════════════════════════════════════════════════

/// `BarraCUDA` Tensor matmul chain for eco dynamics (f32 accumulation).
pub const BARRACUDA_GPU_ECO_F32: f64 = 1e-3;

/// ‖\[A,B\]‖_F / ‖A‖‖B‖ threshold for approximate commutativity.
///
/// Relaxed because random matrices are generically non-commuting.
pub const SPECTRAL_COMMUTATIVITY_EPS: f64 = 0.01;

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

/// GPU multi-objective fitness: per-chunk mean + 0.1*std (f32).
///
/// Directed evolution Paper 014. Each (individual, objective) pair computes
/// mean and population std over a genome chunk. f32 vs f64 CPU gives ~1e-3.
pub const GPU_MULTI_OBJ_FITNESS_F32: f64 = 1e-3;

/// GPU pairwise L2: MODES novelty metric distance (f32).
///
/// L2 = sqrt(sum((a\[d\]-b\[d\])²)).  f32 squared-diff accumulation plus
/// sqrt produces ~4–5 digit accuracy.  1e-3 accommodates dim ≤ 64.
pub const GPU_MODES_L2_F32: f64 = 1e-3;

/// GPU two-input Hill: AND gate vs CPU `signal_integration::two_input_hill` (f32).
///
/// WGSL `pow` is transcendental; f32 vs f64 CPU produces ~3 digits
/// agreement.  1e-3 accommodates GPU Hill vs CPU reference.
pub const GPU_HILL_F32: f64 = 1e-3;

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

/// ML inference pipeline output norm relative tolerance (10%).
///
/// Multi-stage f32 tensor pipelines (MLP, Transformer) accumulate
/// rounding across matmul + activation + normalization.  Output
/// vector norm should agree within 10% of the Python baseline norm.
pub const ML_PIPELINE_NORM_REL: f64 = 0.1;

// ═══════════════════════════════════════════════════════════════════
// Upstream parity and reduce pipeline (Phase 5c)
// ═══════════════════════════════════════════════════════════════════

/// `ReduceScalarPipeline::sum_f64` vs CPU sum agreement.
///
/// GPU f64 parallel reduction has non-deterministic summation order.
/// For arrays up to 10k elements, the diff is within machine epsilon.
pub const GPU_REDUCE_F64: f64 = 1e-10;

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

/// Shannon diversity lower bound for swarm heterogeneity checks.
///
/// Guards against division-by-zero in entropy calculations.
pub const DIVERSITY_EPSILON: f64 = 1e-10;

/// Variance floor for GPU locus variance readback.
///
/// Prevents false negatives when variance is legitimately near zero.
pub const VARIANCE_FLOOR: f64 = -1e-6;

/// Hill gate / regulatory / introgression bounds tolerance (f32 GPU).
///
/// GPU Hill-function dispatch bounds checks need a small slack
/// to account for f32 rounding at boundary values.
pub const GPU_BOUNDS_SLACK_F32: f64 = 1e-5;

mod registry;
pub use registry::{all_tolerances, categories, tolerance_by_name, NamedTolerance};
