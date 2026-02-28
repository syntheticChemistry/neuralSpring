// SPDX-License-Identifier: AGPL-3.0-or-later

//! WGSL shader source catalog — single source of truth for validation.
//!
//! ## Upstream-sourced (absorbed by `ToadStool`)
//!
//! `ToadStool` S68 evolved all shaders to f64 canonical with runtime
//! downcast via `LazyLock<String>`. Some constants became private;
//! those use local shader copies instead of re-exports.
//!
//! | Constant | Source |
//! |----------|--------|
//! | [`HMM_FORWARD_LOG`] | `barracuda::ops::bio::hmm::WGSL_HMM_FORWARD_LOG_F32` (still pub) |
//! | [`BATCH_FITNESS_EVAL`] | `barracuda::ops::bio::batch_fitness::WGSL_BATCH_FITNESS_EVAL` (still pub) |
//! | [`RK4_PARALLEL`] | `include_str!(rk4_parallel_f64.wgsl)` (upstream f64 canonical) |
//! | [`PAIRWISE_JACCARD`] | Local copy (upstream now private `LazyLock`) |
//! | [`LOCUS_VARIANCE`] | `barracuda::ops::bio::locus_variance::WGSL_LOCUS_VARIANCE_F64` |
//! | [`SPATIAL_PAYOFF`] | Local copy (upstream now private `LazyLock`) |
//! | [`BATCH_IPR`] | Local copy (upstream now `LazyLock<String>`) |
//! | [`PAIRWISE_HAMMING`] | Local copy (upstream now private `LazyLock`) |
//!
//! ## Upstream-sourced (generalized variants — absorbed by `ToadStool` `d45fdfb3`)
//!
//! Local copies retained for validation compatibility (different binding layouts).
//!
//! | Constant | Upstream | Difference |
//! |----------|----------|------------|
//! | [`PAIRWISE_L2`] | `barracuda::shaders::math::pairwise_l2` | Closed-form pair decode |
//! | [`MULTI_OBJ_FITNESS`] | `barracuda::shaders::bio::multi_obj_fitness` | Bessel correction |
//! | [`SWARM_NN_FORWARD`] | `barracuda::shaders::bio::swarm_nn_forward` | Generic MLP dims |
//! | [`HILL_GATE`] | `barracuda::shaders::bio::hill_gate` | Mode 0/1 generalization |
//! | [`MEAN_REDUCE`] | `barracuda::shaders::reduce::mean_reduce` | Effectively identical |
//!
//! ## Still local (4 shaders — no upstream equivalent)
//!
//! | Constant | Domain | Papers |
//! |----------|--------|--------|
//! | [`HEAD_SPLIT`] | MHA | — |
//! | [`HEAD_CONCAT`] | MHA | — |
//! | [`XOSHIRO128SS`] | PRNG | — |
//! | [`SWARM_NN_SCORES`] | Swarm scores | 015 |

// ── Upstream-sourced (re-exported from barracuda) ───────────────────

/// HMM forward pass in log-domain (Papers 016–018). Absorbed by `ToadStool`.
pub use barracuda::ops::bio::hmm::WGSL_HMM_FORWARD_LOG_F32 as HMM_FORWARD_LOG;

/// Batch linear fitness evaluation (Papers 011–015). Absorbed by `ToadStool`.
pub use barracuda::ops::bio::batch_fitness::WGSL_BATCH_FITNESS_EVAL as BATCH_FITNESS_EVAL;

/// Parallel RK4 ODE integration (Papers 020–021).
///
/// Absorbed by `ToadStool` S68. Upstream evolved to f64 canonical
/// (`rk4_parallel_f64.wgsl`); f64 version requires Sovereign Compiler
/// polyfill injection. Local f32 copy retained for direct wgpu validation.
pub const RK4_PARALLEL: &str = include_str!("../../shaders/rk4_parallel.wgsl");

/// Pairwise Jaccard distance (Paper 024). Absorbed by `ToadStool`.
///
/// `ToadStool` S68 evolved the f32 const to private `LazyLock<String>`
/// (f64 canonical + runtime downcast). Local copy used for validation.
pub const PAIRWISE_JACCARD: &str = include_str!("../../shaders/pairwise_jaccard.wgsl");

/// Per-locus allele frequency variance (Paper 025). Absorbed by `ToadStool`.
///
/// Upstream provides `WGSL_LOCUS_VARIANCE_F64` (S68 f64 canonical).
pub use barracuda::ops::bio::locus_variance::WGSL_LOCUS_VARIANCE_F64 as LOCUS_VARIANCE;

/// Spatial payoff on grid (Paper 019). Absorbed by `ToadStool`.
///
/// `ToadStool` S68 evolved to private `LazyLock<String>`. Local copy.
pub const SPATIAL_PAYOFF: &str = include_str!("../../shaders/spatial_payoff.wgsl");

/// Batch IPR (Papers 022–023). Absorbed by `ToadStool`.
pub const BATCH_IPR: &str = include_str!("../../shaders/batch_ipr.wgsl");

/// Pairwise Hamming distance (Paper 017). Absorbed by `ToadStool`.
///
/// `ToadStool` S68 evolved to private `LazyLock<String>`. Local copy.
pub const PAIRWISE_HAMMING: &str = include_str!("../../shaders/pairwise_hamming.wgsl");

// ── Still local (pending absorption) ────────────────────────────────

/// Scalar mean reduction.
pub const MEAN_REDUCE: &str = include_str!("../../shaders/mean_reduce.wgsl");

/// Pairwise L2 (Euclidean) distance (Paper 012 — MODES novelty).
///
/// One thread per pair. `d = sqrt(sum((a_i - b_i)^2))` in feature space.
pub const PAIRWISE_L2: &str = include_str!("../../shaders/pairwise_l2.wgsl");

/// Multi-objective fitness evaluation (Paper 014 — directed evolution).
///
/// One thread per (individual, objective). Computes mean and std across
/// evaluation chunks for multi-objective selection.
pub const MULTI_OBJ_FITNESS: &str = include_str!("../../shaders/multi_obj_fitness.wgsl");

/// Batch neural network forward pass (Paper 015 — swarm robotics).
///
/// One thread per (controller, evaluation). Runs a 1→4→5 MLP with sigmoid
/// activation for heterogeneous swarm controllers.
pub const SWARM_NN_FORWARD: &str = include_str!("../../shaders/swarm_nn_forward.wgsl");

/// Two-input Hill function AND gate (Paper 021 — signal integration).
///
/// One thread per (cdg, ai2) grid point. Computes the combined Hill response
/// modeling a biological AND gate for quorum sensing.
pub const HILL_GATE: &str = include_str!("../../shaders/hill_gate.wgsl");

/// GPU head split for multi-head attention (S-03b workaround).
///
/// Reshapes `[batch, seq, d_model]` → `[batch, n_heads, seq, d_head]`.
pub const HEAD_SPLIT: &str = include_str!("../../shaders/head_split.wgsl");

/// GPU head concatenation for multi-head attention (S-03b workaround).
///
/// Reshapes `[batch, n_heads, seq, d_head]` → `[batch, seq, d_model]`.
pub const HEAD_CONCAT: &str = include_str!("../../shaders/head_concat.wgsl");

/// GPU-parallel PRNG using Xoshiro128** (all stochastic algorithms).
///
/// Each thread maintains independent 4×u32 state seeded via `SplitMix32`.
/// Generates uniform f32 in `[0, 1)`. State persists across dispatches.
pub const XOSHIRO128SS: &str = include_str!("../../shaders/xoshiro128ss.wgsl");

/// Swarm NN max-activation scores (Paper 015 — swarm robotics pipeline).
///
/// Extracts per-controller max activation for `mean_reduce` chaining.
pub const SWARM_NN_SCORES: &str = include_str!("../../shaders/swarm_nn_scores.wgsl");

// ── Phase 2a: New local shaders (Session 43 — evolving for ToadStool) ───

/// Batched numerically-stable log-sum-exp reduction (Papers 016–018).
///
/// Each thread computes `logsumexp` over one row of a `[batch × width]` matrix.
/// Uses the max-subtract trick: `max(x) + log(Σ exp(x_i - max(x)))`.
pub const LOGSUMEXP_REDUCE: &str = include_str!("../../shaders/logsumexp_reduce.wgsl");

/// Fermi imitation dynamics stencil update (Paper 019 — game theory).
///
/// Each thread updates one cell's strategy by comparing fitness with a
/// Moore neighbor via the Fermi function `P = 1/(1+exp((f_self-f_nb)/κ))`.
pub const STENCIL_COOPERATION: &str = include_str!("../../shaders/stencil_cooperation.wgsl");

/// Adaptive Dormand-Prince RK45 single step (Papers 020–021).
///
/// One thread per ODE system. Computes 5th-order solution and embedded
/// 4th-order error estimate for Hill-function regulatory network kinetics.
pub const RK45_ADAPTIVE: &str = include_str!("../../shaders/rk45_adaptive.wgsl");

/// Wright-Fisher drift + selection step (Papers 024–025).
///
/// Each thread handles one (population, locus) pair. Applies selection
/// via fitness weighting, then binomial drift using inline xoshiro128**.
pub const WRIGHT_FISHER_STEP: &str = include_str!("../../shaders/wright_fisher_step.wgsl");

// ── Write phase: new extensions for ToadStool absorption (S64) ──────

/// Fused chi-squared statistic (f64): `sum((o-e)²/e)` in a single dispatch.
///
/// Replaces the CPU elementwise loop + GPU sum pipeline currently used in
/// `gpu_ops::reduction::chi_squared_gpu`. Two input arrays (observed, expected),
/// workgroup-parallel reduction.
///
/// ## Absorption target: `barracuda::ops::fused_chi_squared_f64`
pub const CHI_SQUARED_F64: &str = include_str!("../../shaders/chi_squared_f64.wgsl");

/// Fused KL divergence (f64): `sum(p * ln(p/q))` in a single dispatch.
///
/// Replaces the CPU log-ratio + GPU sum pipeline currently used in
/// `gpu_ops::reduction::kl_divergence_gpu`. Two input arrays (p, q),
/// workgroup-parallel reduction with guard against zero.
///
/// ## Absorption target: `barracuda::ops::fused_kl_divergence_f64`
pub const KL_DIVERGENCE_F64: &str = include_str!("../../shaders/kl_divergence_f64.wgsl");

// ── Phase 4: New shaders for GPU-resident pipelines ─────────────────

/// HMM backward pass in log-domain (Papers 016–018).
///
/// Per-timestep dispatch. Each thread computes `log_beta[i]` for one state
/// via logsumexp over predecessors. Host iterates t = T-2..0.
///
/// ## Absorption target: `barracuda::ops::bio::hmm`
pub const HMM_BACKWARD_LOG: &str = include_str!("../../shaders/hmm_backward_log.wgsl");

/// HMM Viterbi decoding in log-domain (Papers 016–018).
///
/// Per-timestep dispatch. Each thread computes the best predecessor
/// (argmax) and score for one state. Host handles backtracing.
///
/// ## Absorption target: `barracuda::ops::bio::hmm`
pub const HMM_VITERBI: &str = include_str!("../../shaders/hmm_viterbi.wgsl");

/// Pearson correlation between upper triangles of two N×N matrices (Paper 025).
///
/// Workgroup-parallel reduction of the 5 sufficient statistics
/// (Σa, Σb, Σab, Σa², Σb²). Host finalizes r from partials.
///
/// ## Absorption target: `barracuda::stats::matrix_correlation`
pub const MATRIX_CORRELATION: &str = include_str!("../../shaders/matrix_correlation.wgsl");

/// Simple linear regression via normal equations (Paper 012 — MODES).
///
/// Workgroup-parallel reduction for (Sx, Sy, Sxx, Sxy, N).
/// Host computes `a = (N·Sxy - Sx·Sy) / (N·Sxx - Sx²)`.
///
/// ## Absorption target: `barracuda::stats::linear_regression_gpu`
pub const LINEAR_REGRESSION: &str = include_str!("../../shaders/linear_regression.wgsl");

// ── coralForge: Evoformer primitives (Phase B) ────────────────
//
// AlphaFold2 Evoformer operations with df64 emulation for f64-class
// precision on consumer GPUs. All require df64_core.wgsl injection
// via `compile_shader_df64`.

/// GELU activation (Evoformer FFN, baseCamp Sub-02).
///
/// Pointwise `GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))`.
/// No df64 needed (pointwise op).
pub const GELU_F64: &str = include_str!("../../shaders/gelu_f64.wgsl");

/// Sigmoid activation (Evoformer gating).
pub const SIGMOID_F64: &str = include_str!("../../shaders/sigmoid_f64.wgsl");

/// Layer normalization with df64 reduction (Evoformer, transformers).
///
/// Workgroup-parallel mean/variance via df64 accumulation.
pub const LAYER_NORM_F64: &str = include_str!("../../shaders/layer_norm_f64.wgsl");

/// Row-wise softmax with df64 accumulation (SDPA pass 2).
///
/// Numerically stable: max-subtract + df64 sum of exponentials.
pub const SOFTMAX_F64: &str = include_str!("../../shaders/softmax_f64.wgsl");

/// Scaled dot-product attention scores with df64 (SDPA pass 1).
///
/// `scores[b,h,q,k] = Σ_d Q[b,h,q,d]*K[b,h,k,d] / √d_k`
pub const SDPA_SCORES_F64: &str = include_str!("../../shaders/sdpa_scores_f64.wgsl");

/// Weighted value summation with df64 (SDPA pass 3).
///
/// `output[b,h,q,d] = Σ_k weights[b,h,q,k] * V[b,h,k,d]`
pub const ATTENTION_APPLY_F64: &str = include_str!("../../shaders/attention_apply_f64.wgsl");

/// Triangle multiplicative update — outgoing edges (Algorithm 11).
///
/// `output[i,j,c] = Σ_k proj_a[i,k,c] * proj_b[j,k,c]` with df64.
pub const TRIANGLE_MUL_OUTGOING_F64: &str =
    include_str!("../../shaders/triangle_mul_outgoing_f64.wgsl");

/// Triangle multiplicative update — incoming edges (Algorithm 12).
///
/// `output[i,j,c] = Σ_k proj_a[k,i,c] * proj_b[k,j,c]` with df64.
pub const TRIANGLE_MUL_INCOMING_F64: &str =
    include_str!("../../shaders/triangle_mul_incoming_f64.wgsl");

/// Triangle self-attention scores with pair bias (Algorithms 13-14).
///
/// `logit[r,h,j,k] = Σ_d Q[r,j,h,d]*K[r,k,h,d]/√D + bias[h,j,k]` with df64.
pub const TRIANGLE_ATTENTION_F64: &str = include_str!("../../shaders/triangle_attention_f64.wgsl");

/// Outer product mean: MSA → pair representation (Evoformer).
///
/// `output[i,j,ca*cb] = mean_s(a[s,i,ca] * b[s,j,cb])` with df64 accumulation.
/// Converts evolutionary covariance (MSA) to structural contacts (pair).
pub const OUTER_PRODUCT_MEAN_F64: &str = include_str!("../../shaders/outer_product_mean_f64.wgsl");

/// MSA row attention scores with pair bias (Evoformer).
///
/// `scores[s,h,i,j] = Σ_d Q[s,i,h,d]*K[s,j,h,d]/√d + bias[h,i,j]` with df64.
/// Per-sequence attention over residue positions. Pair bias injects structure.
pub const MSA_ROW_ATTENTION_SCORES_F64: &str =
    include_str!("../../shaders/msa_row_attention_scores_f64.wgsl");

/// MSA column attention scores (Evoformer).
///
/// `scores[r,h,si,sj] = Σ_d Q[si,r,h,d]*K[sj,r,h,d]/√d` with df64.
/// Per-position attention across MSA sequences (no pair bias).
pub const MSA_COL_ATTENTION_SCORES_F64: &str =
    include_str!("../../shaders/msa_col_attention_scores_f64.wgsl");

// ── Structure Module: IPA + backbone (Phase B.3) ─────────────────────

/// Invariant Point Attention scores (Algorithm 22).
///
/// Three-term IPA logit: scalar QK, pair bias, and SE(3)-equivariant
/// point distance through backbone frames. df64 for both dot products
/// and distance accumulation.
pub const IPA_SCORES_F64: &str = include_str!("../../shaders/ipa_scores_f64.wgsl");

/// Backbone frame composition (Structure Module iteration).
///
/// Composes current frames with predicted delta transforms (quaternion
/// + translation). df64 for rotation matrix multiplication.
pub const BACKBONE_UPDATE_F64: &str = include_str!("../../shaders/backbone_update_f64.wgsl");

/// Torsion angle prediction (Structure Module).
///
/// Fused `ResNet` + unit circle normalization kernel. Predicts 7
/// torsion angles (sin, cos) per residue from the single representation.
/// df64 for all matrix multiplications.
pub const TORSION_ANGLES_F64: &str = include_str!("../../shaders/torsion_angles_f64.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_shaders_non_empty() {
        let shaders: &[(&str, &str)] = &[
            ("HMM_FORWARD_LOG", HMM_FORWARD_LOG),
            ("BATCH_FITNESS_EVAL", BATCH_FITNESS_EVAL),
            ("RK4_PARALLEL", RK4_PARALLEL),
            ("MEAN_REDUCE", MEAN_REDUCE),
            ("PAIRWISE_JACCARD", PAIRWISE_JACCARD),
            ("LOCUS_VARIANCE", LOCUS_VARIANCE),
            ("SPATIAL_PAYOFF", SPATIAL_PAYOFF),
            ("BATCH_IPR", BATCH_IPR),
            ("PAIRWISE_HAMMING", PAIRWISE_HAMMING),
            ("PAIRWISE_L2", PAIRWISE_L2),
            ("MULTI_OBJ_FITNESS", MULTI_OBJ_FITNESS),
            ("SWARM_NN_FORWARD", SWARM_NN_FORWARD),
            ("HILL_GATE", HILL_GATE),
            ("HEAD_SPLIT", HEAD_SPLIT),
            ("HEAD_CONCAT", HEAD_CONCAT),
            ("XOSHIRO128SS", XOSHIRO128SS),
            ("SWARM_NN_SCORES", SWARM_NN_SCORES),
            ("LOGSUMEXP_REDUCE", LOGSUMEXP_REDUCE),
            ("STENCIL_COOPERATION", STENCIL_COOPERATION),
            ("RK45_ADAPTIVE", RK45_ADAPTIVE),
            ("WRIGHT_FISHER_STEP", WRIGHT_FISHER_STEP),
            ("CHI_SQUARED_F64", CHI_SQUARED_F64),
            ("KL_DIVERGENCE_F64", KL_DIVERGENCE_F64),
            ("GELU_F64", GELU_F64),
            ("SIGMOID_F64", SIGMOID_F64),
            ("LAYER_NORM_F64", LAYER_NORM_F64),
            ("SOFTMAX_F64", SOFTMAX_F64),
            ("SDPA_SCORES_F64", SDPA_SCORES_F64),
            ("ATTENTION_APPLY_F64", ATTENTION_APPLY_F64),
            ("TRIANGLE_MUL_OUTGOING_F64", TRIANGLE_MUL_OUTGOING_F64),
            ("TRIANGLE_MUL_INCOMING_F64", TRIANGLE_MUL_INCOMING_F64),
            ("TRIANGLE_ATTENTION_F64", TRIANGLE_ATTENTION_F64),
            ("OUTER_PRODUCT_MEAN_F64", OUTER_PRODUCT_MEAN_F64),
            ("MSA_ROW_ATTENTION_SCORES_F64", MSA_ROW_ATTENTION_SCORES_F64),
            ("MSA_COL_ATTENTION_SCORES_F64", MSA_COL_ATTENTION_SCORES_F64),
            ("IPA_SCORES_F64", IPA_SCORES_F64),
            ("BACKBONE_UPDATE_F64", BACKBONE_UPDATE_F64),
            ("TORSION_ANGLES_F64", TORSION_ANGLES_F64),
            ("HMM_BACKWARD_LOG", HMM_BACKWARD_LOG),
            ("HMM_VITERBI", HMM_VITERBI),
            ("MATRIX_CORRELATION", MATRIX_CORRELATION),
            ("LINEAR_REGRESSION", LINEAR_REGRESSION),
        ];
        for (name, src) in shaders {
            assert!(
                !src.is_empty(),
                "shader {name} is empty — missing .wgsl file?"
            );
            assert!(
                src.contains("@compute"),
                "shader {name} missing @compute entry point"
            );
        }
    }

    #[test]
    fn shader_count_is_42() {
        assert_eq!(
            42,
            [
                HMM_FORWARD_LOG,
                BATCH_FITNESS_EVAL,
                RK4_PARALLEL,
                MEAN_REDUCE,
                PAIRWISE_JACCARD,
                LOCUS_VARIANCE,
                SPATIAL_PAYOFF,
                BATCH_IPR,
                PAIRWISE_HAMMING,
                PAIRWISE_L2,
                MULTI_OBJ_FITNESS,
                SWARM_NN_FORWARD,
                HILL_GATE,
                HEAD_SPLIT,
                HEAD_CONCAT,
                XOSHIRO128SS,
                SWARM_NN_SCORES,
                LOGSUMEXP_REDUCE,
                STENCIL_COOPERATION,
                RK45_ADAPTIVE,
                WRIGHT_FISHER_STEP,
                CHI_SQUARED_F64,
                KL_DIVERGENCE_F64,
                GELU_F64,
                SIGMOID_F64,
                LAYER_NORM_F64,
                SOFTMAX_F64,
                SDPA_SCORES_F64,
                ATTENTION_APPLY_F64,
                TRIANGLE_MUL_OUTGOING_F64,
                TRIANGLE_MUL_INCOMING_F64,
                TRIANGLE_ATTENTION_F64,
                OUTER_PRODUCT_MEAN_F64,
                MSA_ROW_ATTENTION_SCORES_F64,
                MSA_COL_ATTENTION_SCORES_F64,
                IPA_SCORES_F64,
                BACKBONE_UPDATE_F64,
                TORSION_ANGLES_F64,
                HMM_BACKWARD_LOG,
                HMM_VITERBI,
                MATRIX_CORRELATION,
                LINEAR_REGRESSION,
            ]
            .len()
        );
    }
}
