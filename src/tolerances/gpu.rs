// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU, tensor, shader, FFT, and dispatch tolerance constants.
//!
//! Extracted from the main tolerances module for domain focus.
//! All constants here govern GPU compute, f32/f64 shader precision,
//! FFT butterfly rounding, and CPU↔GPU dispatch parity.

use super::{CROSS_LANGUAGE, EXACT_F64, ZERO_DETECTION};

// ═══════════════════════════════════════════════════════════════════
// Phase 0++ stochastic model tolerances
// ═══════════════════════════════════════════════════════════════════

/// `BarraCUDA` Tensor matmul chain for eco dynamics (f32 accumulation).
///
/// Eco-dynamics fitness evaluation chains `matmul` → `ReLU` → `matmul` on f32
/// GPU tensors.  Each matmul (inner dim ≤ 32) accumulates `~√32·eps_f32 ≈ 7e-7`
/// per element, and chaining two layers doubles the error budget.
/// 1e-3 provides margin for future larger inner dimensions.
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

/// GPU HMM forward: f64 log-likelihood agreement with CPU (upstream `HmmBatchForwardF64`).
///
/// The f64 shader is strictly tighter than the f32 path: with double-precision
/// log-domain arithmetic the accumulated rounding over T ≤ 100 timesteps and
/// N ≤ 10 states stays within 0.05 (10× tighter than f32).
pub const GPU_HMM_LOG_LIKELIHOOD_F64: f64 = GPU_HMM_LOG_LIKELIHOOD_F32 * 0.1;

/// GPU HMM forward: f32 log-likelihood for extended sequences (T > 100, N > 4).
///
/// Longer sequences and larger state spaces accumulate more logsumexp rounding.
/// 2× the base tolerance covers T ≤ 1000 with N ≤ 10 states.
pub const GPU_HMM_LOG_LIKELIHOOD_F32_EXTENDED: f64 = GPU_HMM_LOG_LIKELIHOOD_F32 * 2.0;

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

/// GPU multi-objective fitness: Bessel correction gap GPU↔CPU (f64 buffers).
///
/// `barracuda::ops::bio::MultiObjFitnessGpu` uses sample std (n-1) while
/// CPU `multi_objective_fitness` uses population std (n).  For genome chunks
/// of length 10 (`genome_len`=40 / `n_objectives`=4), this algorithmic
/// difference yields ~2e-3 observed absolute gap.  3e-3 provides 50% margin.
///
/// Validated: RTX 4070 Vulkan + llvmpipe, seed=42/77/123, commit `f9ad0268`.
pub const GPU_MULTI_OBJ_BESSEL_F64: f64 = 3e-3;

/// Upstream parity: local `metalForge` vs `BarraCuda` `MultiObjFitnessGpu` (f32).
///
/// Upstream uses Bessel correction (n-1) for std; local uses population n.
/// Observed diff ~2e-3, so 5e-3 accommodates the known algorithmic difference.
pub const GPU_UPSTREAM_MULTI_OBJ_PARITY_F32: f64 = 5e-3;

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

/// `HillGateGpu` f64 grid: two-input Hill function computed in f64 on GPU.
/// Tighter than f32 path but still needs transcendental tolerance for `pow`.
pub const GPU_HILL_GATE_F64: f64 = 1e-6;

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

/// LSTM glucose reservoir: GPU `Tensor::matmul` gate projection (Paper 026).
///
/// Each LSTM step chains 2× matmul + 3× add + sigmoid/tanh (CPU-side).
/// Over 12 steps the f32 rounding compounds, especially through the
/// sigmoid/tanh transcendentals.  0.05 accommodates the multi-step
/// chain while still catching gross errors.
pub const GPU_LSTM_GLUCOSE_F32: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════
// Upstream parity and reduce pipeline (Phase 5c)
// ═══════════════════════════════════════════════════════════════════

/// `ReduceScalarPipeline::sum_f64` vs CPU sum agreement.
///
/// GPU f64 parallel reduction has non-deterministic summation order.
/// For arrays up to 10k elements, the diff is within machine epsilon.
pub const GPU_REDUCE_F64: f64 = 1e-10;

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

/// GPU logsumexp f32 tolerance.
///
/// Logsumexp accumulates via max-subtract + exp + sum + log; f32
/// precision limits the result to ~7 significant digits.  1e-4
/// tolerates the exp/log round-trip.
pub const GPU_LOGSUMEXP_F32: f64 = 1e-4;

/// GPU RK45 adaptive integrator f32 tolerance.
///
/// Dormand-Prince 5(4) has 6 stages of f32 multiply-add.  The
/// accumulated rounding from the Butcher tableau coefficients
/// requires ~5e-4 tolerance vs exact CPU f64 reference.
pub const GPU_RK45_F32: f64 = 5e-4;

// ═══════════════════════════════════════════════════════════════════
// Cross-dispatch f64 parity (upstream barracuda dispatch vs CPU ref)
// ═══════════════════════════════════════════════════════════════════

/// Cross-dispatch matmul: upstream vs CPU reference (f64, n=64).
///
/// GPU parallel reduction accumulates dot products in different order than
/// sequential CPU; for 64-element inner products the rounding difference
/// reaches O(n * eps * ||A|| * ||B||) ≈ 1e-3 on typical inputs.
pub const DISPATCH_MATMUL_F64: f64 = 1e-3;

/// Cross-dispatch Frobenius norm: upstream vs CPU reference (f64, n=1024).
///
/// Single parallel reduction of 1024 squared values; accumulation order
/// difference yields ~6 digits of agreement.
pub const DISPATCH_FROBENIUS_F64: f64 = 1e-6;

/// Cross-dispatch transpose: upstream vs CPU reference (f64, n=32).
///
/// Pure data movement — no arithmetic difference.  Tolerance equals
/// machine precision for exact operations.
pub const DISPATCH_TRANSPOSE_F64: f64 = EXACT_F64;

/// Cross-dispatch element-wise ops: softmax, gelu, `hmm_forward`, mean (f64).
///
/// These involve exp/log/div chains where implementation ordering differs;
/// f64 agreement at 10 digits is typical.
pub const DISPATCH_ELEMENTWISE_F64: f64 = CROSS_LANGUAGE;

/// Cross-dispatch two-pass statistics: variance, `l2_distance` (f64).
///
/// Two-pass algorithms (mean → residuals → mean) compound rounding from
/// both passes; 8 digits of agreement is typical for n ≤ 4096.
pub const DISPATCH_TWOPASS_F64: f64 = 1e-8;

/// Cross-dispatch near-zero analytical value (e.g. gelu(0) ≈ 0).
pub const DISPATCH_NEAR_ZERO_F64: f64 = ZERO_DETECTION;

/// Cross-dispatch f32 Tensor API round-trip: `softmax_dim`, `argmax_dim`.
///
/// f64 → f32 upload → Tensor op → f32 readback → f64 conversion.
/// f32 mantissa gives ~7 decimal digits; for softmax with 8–256 elements
/// the row-sum deviates by ~5e-8 from unity. 1e-6 is conservative.
pub const DISPATCH_F32_ROUNDTRIP: f64 = 1e-6;

/// Cross-dispatch Viterbi f32 accumulated: GPU scores in f32 + argmax.
///
/// The Viterbi log-probability accumulates over T timesteps; each
/// f32 max-reduction introduces ~`eps_f32` per step. For T=10, observed
/// deviation from exact f64 reference is ~1e-7. 1e-5 is conservative.
pub const DISPATCH_VITERBI_F32: f64 = 1e-5;

// ═══════════════════════════════════════════════════════════════════
// GPU promotion dispatch parity (CPU → GPU round-trip)
// ═══════════════════════════════════════════════════════════════════

/// GPU matmul vs CPU max element-wise difference (identity product).
///
/// f64 CPU → f32 GPU → f64 readback: identity product should be exact
/// to ~7 decimal digits (f32 mantissa).  0.01 allows for accumulated
/// rounding in the tile-based matmul kernel.
pub const GPU_MATMUL_IDENTITY_F32: f64 = 0.01;

/// GPU matmul vs CPU max element-wise difference (random product).
///
/// Random 8×8 f32 matmul accumulates 8 multiply-add rounding errors;
/// worst-case element ~0.03 observed on llvmpipe/Vulkan.
pub const GPU_MATMUL_RANDOM_F32: f64 = 0.05;

/// GPU transpose vs CPU max element-wise difference.
///
/// Pure data movement — rounding only from f64→f32→f64 conversion.
pub const GPU_TRANSPOSE_F32: f64 = 0.01;

/// GPU Frobenius norm vs CPU absolute tolerance.
///
/// Single-pass norm reduction on f32, compared to f64 CPU reference.
pub const GPU_FROBENIUS_F32: f64 = 0.01;

/// GPU commutator `[A,B]` vs CPU max element-wise difference.
///
/// Two matmuls plus subtraction on f32; error accumulates from both
/// products.  0.1 covers worst-case 4×4 random matrices.
pub const GPU_COMMUTATOR_F32: f64 = 0.1;

/// GPU `distance_to_normal` upper bound for known-symmetric matrices.
///
/// Symmetric matrix should have zero commutator; f32 rounding and
/// two matmul round-trips give residual ≤ 0.05.
pub const GPU_NORMAL_DISTANCE_SYMMETRIC_F32: f64 = 0.05;

/// CPU `distance_to_normal` upper bound for known-symmetric matrices.
///
/// f64 CPU should give essentially zero for symmetric input.
pub const CPU_NORMAL_DISTANCE_SYMMETRIC_F64: f64 = 1e-6;

/// GPU softmax vs CPU max element-wise difference.
///
/// exp/sum/div chain on f32 vs f64 reference.
pub const GPU_SOFTMAX_DISPATCH_F32: f64 = 0.01;

/// GPU softmax sum-to-one tolerance.
///
/// After exp/sum/div on f32 GPU, the row sum deviates from 1.0 by
/// `O(n·eps_f32)` where n is the vector length.  For n ≤ 256 this is ~3e-5,
/// but the f64→f32→f64 round-trip widens the gap.  Observed max
/// deviation ~1e-3 on llvmpipe/NVK backends; 5e-3 provides 5× margin.
pub const GPU_SOFTMAX_SUM_F32: f64 = 5e-3;

/// GPU Boltzmann distribution vs CPU max element-wise difference.
///
/// Pre-scaled softmax; additional `mul_add` rounding vs CPU f64.
pub const GPU_BOLTZMANN_F32: f64 = 0.05;

/// GPU L2 distance vs CPU absolute tolerance.
///
/// L2 = sqrt(sum((a-b)²)).  The squared-difference accumulation on f32
/// loses ~2 digits vs f64 for dim ≤ 64, and the final sqrt introduces
/// one additional rounding step.  0.01 covers observed max diff ~5e-3.
pub const GPU_L2_DISPATCH_F32: f64 = 0.01;

/// GPU mean reduction vs CPU absolute tolerance.
///
/// Single-pass sum/N on f32 GPU.  For n ≤ 256 elements of O(1) magnitude,
/// the f32 summation error is `~n·eps_f32·max(|x|) ≈ 3e-5`, well within
/// 0.01.  The wider tolerance accommodates driver-dependent reduction order.
pub const GPU_MEAN_DISPATCH_F32: f64 = 0.01;

/// GPU variance vs CPU absolute tolerance.
///
/// Two-pass (mean → residual → mean) on f32; 0.1 covers the
/// catastrophic-cancellation rounding in small datasets.
pub const GPU_VARIANCE_DISPATCH_F32: f64 = 0.1;

/// GPU Shannon entropy vs CPU absolute tolerance.
///
/// log + mul + sum chain on f32; uniform(4) entropy ≈ 1.386.
pub const GPU_ENTROPY_F32: f64 = 0.05;

/// GPU Pearson correlation vs exact known value.
///
/// Perfect linear relationship: f32 rounding in variance chain.
pub const GPU_PEARSON_F32: f64 = 0.05;

/// GPU chi-squared statistic vs CPU absolute tolerance.
///
/// Large expected values (25) dampen the division rounding;
/// 0.5 covers worst-case f32 error in the 4-bin test.
pub const GPU_CHI_SQUARED_F32: f64 = 0.5;

/// GPU GELU activation absolute tolerance.
///
/// GELU(x) = x·Φ(x) uses tanh approximation on f32 GPU.  WGSL `tanh`
/// polynomial coefficients differ from CPU libm, yielding ~3 digits of
/// agreement.  0.01 covers the observed max element-wise diff ~5e-3
/// across the input range [-3, 3] on validated hardware.
pub const GPU_GELU_F32: f64 = 0.01;

/// GPU HMM forward step normalization tolerance.
///
/// Each HMM forward step multiplies the alpha vector by a transition matrix
/// and normalizes.  On f32 GPU, the matrix-vector product plus normalization
/// loses ~2 digits vs f64 CPU per step.  For single-step validation (T=1),
/// 0.01 matches the matmul tolerance.
pub const GPU_HMM_STEP_F32: f64 = 0.01;

/// GPU sum reduction absolute tolerance.
///
/// Parallel f32 sum reduction has non-deterministic addition order.  For
/// n ≤ 256 elements of magnitude O(1..10), the worst-case absolute
/// deviation from sequential f64 sum reaches ~0.05.  0.1 provides margin
/// for larger arrays and mixed-sign cancellation.
pub const GPU_SUM_DISPATCH_F32: f64 = 0.1;

/// GPU max reduction absolute tolerance.
///
/// Max reduction on f32 should be exact (no accumulation), but the
/// f64→f32→f64 round-trip truncates mantissa bits.  For values of
/// magnitude O(1..10), the round-trip error is `~eps_f32·|x| ≈ 1e-6`.
/// 0.1 is generous to accommodate edge cases with large magnitudes.
pub const GPU_MAX_DISPATCH_F32: f64 = 0.1;

/// GPU KL divergence absolute tolerance.
///
/// KL(P||Q) = Σ p·ln(p/q) involves log and multiply-accumulate on f32.
/// The log introduces ~1 ULP error per element, and the sum accumulates
/// `O(n·eps_f32)` additional rounding.  For n ≤ 64 with distributions
/// bounded away from zero, the observed diff is ~1e-3.
pub const GPU_KL_DISPATCH_F32: f64 = 0.01;

/// Multi-objective fitness GPU vs CPU max element-wise difference.
///
/// Batch fitness evaluation on f64 GPU buffers; rounding from
/// the multi-objective sum-of-squares pattern on 12-gene genotypes.
pub const GPU_MULTI_OBJ_FITNESS_F64: f64 = 1e-2;

/// GPU inter-population allele frequency variance vs CPU (f32).
///
/// Allele frequency variance across populations is computed per-locus
/// on f32 GPU, then averaged.  The two-pass mean/variance pattern on
/// f32 introduces ~2 digits of rounding vs f64 reference.  0.02
/// accommodates 2–8 populations × 10–50 loci.
pub const GPU_AF_VARIANCE_F32: f64 = 0.02;

/// GPU HMM Viterbi log-probability gap (4-state, T≤50, f64 dispatch).
///
/// Viterbi path log-probability computed via stepwise GPU dispatch
/// accumulates logsumexp-equivalent rounding per timestep.  For 4-state
/// HMMs with T≤50, the gap between GPU-reconstructed and CPU reference
/// stays within 2.0 nats.
pub const GPU_HMM_VITERBI_LOGPROB_F64: f64 = 2.0;

/// GPU Viterbi long-sequence path agreement: minimum fraction of
/// states matching the CPU reference path.
///
/// Over 200+ timesteps, f32 accumulation can shift argmax at boundary
/// states where two paths have near-equal log-probabilities.  90%
/// agreement ensures the GPU path is structurally correct while
/// allowing boundary-state disagreements.
pub const GPU_VITERBI_PATH_AGREEMENT_MIN: f64 = 0.90;

/// GPU pairwise FST: f32 allele-frequency intermediary widens the
/// FST gap vs f64 CPU reference.
///
/// Pairwise FST computes per-locus allele frequencies in f32 then
/// averages heterozygosity ratios.  Observed gap ~0.05 for 10–20
/// loci and 10 individuals per population.  0.1 provides margin.
pub const GPU_FST_PAIRWISE_F32: f64 = 0.1;

// ═══════════════════════════════════════════════════════════════════
// baseCamp dispatch parity (f64 GPU typed ops)
// ═══════════════════════════════════════════════════════════════════

/// GPU f64 variance dispatch parity: `VarianceF64` vs CPU.
///
/// One-pass parallel variance accumulation on f64 GPU buffers introduces
/// rounding from the per-workgroup partial-sum pattern.  Observed diff
/// < 1e-9 for n ≤ 4096.  1e-8 provides margin for larger workloads.
pub const GPU_VARIANCE_F64: f64 = 1e-8;

/// GPU f64 Pearson correlation dispatch parity: `CorrelationF64` vs CPU.
///
/// GPU correlation requires two variance reductions plus a covariance
/// reduction, each introducing independent rounding.  Observed diff
/// < 1e-7 for n ≤ 256.
pub const GPU_PEARSON_F64: f64 = 1e-6;

/// GPU f64 Shannon entropy dispatch parity: `FusedMapReduceF64` vs CPU.
///
/// Entropy involves `ln()` transcendentals on GPU; f64 polyfill
/// introduces ~4 digits less precision than native libm.  Observed
/// diff < 5e-5 for n ≤ 100.
pub const GPU_ENTROPY_F64: f64 = 1e-4;

/// GPU Jacobi eigenvalue agreement in dispatch validation (f64).
///
/// Jacobi eigensolver eigenvalue accuracy (0.1 absolute) for the
/// 16×16 Hamiltonians used in `validate_compute_dispatch` and
/// `validate_basecamp_gpu`.  Wider than `EIGH_JACOBI_EIGENVALUE`
/// (1e-3) because the dispatch path exercises larger matrices.
pub const GPU_EIGH_DISPATCH_F64: f64 = 0.1;

/// BP / PGM normalization sum: belief propagation output sums to 1.
///
/// Row-stochastic matrix multiplication preserves normalization to
/// machine precision.  Multi-layer BP chains compound rounding but
/// stay within 1e-8 for ≤ 10 layers at dim ≤ 64.
pub const PGM_NORMALIZATION_SUM: f64 = 1e-8;

/// ML inference pipeline output norm relative tolerance (10%).
///
/// Multi-stage f32 tensor pipelines (MLP, Transformer) accumulate
/// rounding across matmul + activation + normalization.  Output
/// vector norm should agree within 10% of the Python baseline norm.
pub const ML_PIPELINE_NORM_REL: f64 = 0.1;

// ═══════════════════════════════════════════════════════════════════
// GPU commutator tolerances (f64 spectral dispatch)
// ═══════════════════════════════════════════════════════════════════

/// GPU commutator near-zero (f64): norm for near-commuting matrices.
///
/// For diagonal or nearly-diagonal matrices, the commutator norm
/// should be effectively zero.  f64 GPU parallel reduction accumulates
/// rounding from two matmuls; 1e-12 is tight for small (8x8) matrices.
pub const GPU_COMMUTATOR_NEAR_ZERO_F64: f64 = 1e-12;

/// GPU commutator residual (f64): upper bound for random matrices.
///
/// Random matrices are generically non-commuting.  The normalized
/// commutator for random nxn matrices has mean that grows with n.
/// 0.5 bounds small (8x8) random matrix commutators.
pub const GPU_COMMUTATOR_RESIDUAL_F64: f64 = 0.5;

// ═══════════════════════════════════════════════════════════════════
// Hardware dispatch cost bounds (metalForge probes)
// ═══════════════════════════════════════════════════════════════════

/// `PCIe` bridge latency floor (microseconds).
///
/// Even fast `PCIe` 4.0 bridges have measurable latency.  Below this
/// indicates measurement error or driver shortcutting.
pub const BRIDGE_COST_MIN_US: f64 = 100.0;

/// `PCIe` bridge latency ceiling (microseconds).
///
/// Bridge latency above this for lightweight probes indicates
/// driver or device issues.
pub const BRIDGE_COST_MAX_US: f64 = 5000.0;

/// Chained bridge overhead ratio: chained-cost / single-cost.
///
/// Chaining two bridge operations should cost less than 3x a single
/// operation due to driver batching and connection reuse.
pub const BRIDGE_CHAIN_OVERHEAD_MAX: f64 = 3.0;

/// Minimum bridge probe latency (microseconds).
///
/// `PCIe` bridge probes measure at least 5 us of round-trip latency.
pub const BRIDGE_PROBE_MIN_US: f64 = 5.0;

/// 1 MB transfer latency floor (microseconds).
///
/// Transferring 1 MB over `PCIe` 4.0 x16 (~25 GB/s theoretical)
/// takes at least ~40 us including overhead.  30 us provides margin.
pub const TRANSFER_1MB_MIN_US: f64 = 30.0;

/// 1 MB transfer latency ceiling (microseconds).
pub const TRANSFER_1MB_MAX_US: f64 = 50.0;

/// CPU/GPU dispatch cost ratio lower bound.
///
/// For small workloads, CPU cost should be at least 80% of GPU cost.
pub const DISPATCH_COST_RATIO_MIN: f64 = 0.8;

/// CPU/GPU dispatch cost ratio upper bound.
pub const DISPATCH_COST_RATIO_MAX: f64 = 1.3;

// ═══════════════════════════════════════════════════════════════════
// coralForge df64 core streaming (f64 buffer I/O → df64 compute → f64 output)
// ═══════════════════════════════════════════════════════════════════

/// df64 arithmetic tolerance — dot products, matrix ops, accumulations.
///
/// df64 (double-float emulated on FP32 cores) provides ~15 digits of
/// precision for basic arithmetic.  Dot products and matrix multiplications
/// accumulate rounding proportional to problem size; for Evoformer-scale
/// tensors (N ≤ 256, C ≤ 64), 1e-6 accommodates the df64 accumulation error.
pub const GPU_DF64_ARITHMETIC: f64 = 1e-6;

/// df64 transcendental tolerance — `exp_df64`, `tanh_df64`, softmax, GELU.
///
/// df64 transcendentals use polynomial approximations that lose ~3 digits
/// vs native f64 libm.  For Evoformer operations involving exp/tanh chains
/// (softmax rows, GELU activation, SDPA pipeline), 5e-4 covers the
/// observed max diff on RTX 4070 Hybrid FP64 strategy.
pub const GPU_DF64_TRANSCENDENTAL: f64 = 5e-4;
