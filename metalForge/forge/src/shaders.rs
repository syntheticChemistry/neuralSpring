// SPDX-License-Identifier: AGPL-3.0-or-later

//! WGSL shader source catalog — single source of truth for validation.
//!
//! ## Upstream-sourced (identical — absorbed by ToadStool `77f70b2e`)
//!
//! | Constant | Upstream path |
//! |----------|---------------|
//! | [`HMM_FORWARD_LOG`] | `barracuda::ops::bio::hmm::WGSL_HMM_FORWARD_LOG_F32` |
//! | [`BATCH_FITNESS_EVAL`] | `barracuda::ops::bio::batch_fitness::WGSL_BATCH_FITNESS_EVAL` |
//! | [`RK4_PARALLEL`] | `barracuda::ops::rk_stage::WGSL_RK4_PARALLEL` |
//! | [`PAIRWISE_JACCARD`] | `barracuda::ops::bio::pairwise_jaccard::WGSL_PAIRWISE_JACCARD` |
//! | [`LOCUS_VARIANCE`] | `barracuda::ops::bio::locus_variance::WGSL_LOCUS_VARIANCE` |
//! | [`SPATIAL_PAYOFF`] | `barracuda::ops::bio::spatial_payoff::WGSL_SPATIAL_PAYOFF` |
//! | [`BATCH_IPR`] | `barracuda::spectral::batch_ipr::WGSL_BATCH_IPR` |
//! | [`PAIRWISE_HAMMING`] | `barracuda::ops::bio::pairwise_hamming::WGSL_PAIRWISE_HAMMING` |
//!
//! ## Upstream-sourced (generalized variants — absorbed by ToadStool `d45fdfb3`)
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

/// Parallel RK4 ODE integration (Papers 020–021). Absorbed by `ToadStool`.
pub use barracuda::ops::rk_stage::WGSL_RK4_PARALLEL as RK4_PARALLEL;

/// Pairwise Jaccard distance (Paper 024). Absorbed by `ToadStool`.
pub use barracuda::ops::bio::pairwise_jaccard::WGSL_PAIRWISE_JACCARD as PAIRWISE_JACCARD;

/// Per-locus allele frequency variance (Paper 025). Absorbed by `ToadStool`.
pub use barracuda::ops::bio::locus_variance::WGSL_LOCUS_VARIANCE as LOCUS_VARIANCE;

/// Spatial payoff on grid (Paper 019). Absorbed by `ToadStool`.
pub use barracuda::ops::bio::spatial_payoff::WGSL_SPATIAL_PAYOFF as SPATIAL_PAYOFF;

/// Batch IPR (Papers 022–023). Absorbed by `ToadStool`.
pub use barracuda::spectral::batch_ipr::WGSL_BATCH_IPR as BATCH_IPR;

/// Pairwise Hamming distance (Paper 017). Absorbed by `ToadStool`.
pub use barracuda::ops::bio::pairwise_hamming::WGSL_PAIRWISE_HAMMING as PAIRWISE_HAMMING;

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
    fn shader_count_is_17() {
        assert_eq!(
            17,
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
            ]
            .len()
        );
    }
}
