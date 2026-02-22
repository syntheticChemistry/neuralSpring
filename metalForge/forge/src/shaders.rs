// SPDX-License-Identifier: AGPL-3.0-or-later

//! WGSL shader source catalog — single source of truth for `ToadStool` absorption.
//!
//! Each constant contains the full WGSL source for one shader, loaded via
//! `include_str!` from `metalForge/shaders/`. `ToadStool` can absorb any shader
//! by copying the constant and its corresponding [`super::bindings`] layout.
//!
//! ## Domain map
//!
//! | Shader | Domain | Papers | Absorption target |
//! |--------|--------|--------|-------------------|
//! | [`HMM_FORWARD_LOG`] | Phylogenetics | 016–018 | `barracuda::ops::hmm` |
//! | [`BATCH_FITNESS_EVAL`] | Evolution | 011–015 | `barracuda::ops::batch_gemm` |
//! | [`RK4_PARALLEL`] | ODE integration | 020–021 | `barracuda::ops::ode` |
//! | [`MEAN_REDUCE`] | Aggregation | — | `barracuda::pipeline::ReduceScalarPipeline` |
//! | [`PAIRWISE_JACCARD`] | Pangenome | 024 | `barracuda::ops::pairwise_distance` |
//! | [`LOCUS_VARIANCE`] | Meta-population | 025 | `barracuda::ops::VarianceReduceF64` |
//! | [`SPATIAL_PAYOFF`] | Game theory | 019 | `barracuda::ops::stencil` |
//! | [`BATCH_IPR`] | Spectral | 022–023 | `barracuda::ops::batch_reduce` |
//! | [`PAIRWISE_HAMMING`] | Alignment | 017 | `barracuda::ops::pairwise_distance` |
//! | [`PAIRWISE_L2`] | MODES novelty | 012 | `barracuda::ops::pairwise_distance` |
//! | [`MULTI_OBJ_FITNESS`] | Directed evo | 014 | `barracuda::ops::batch_gemm` |
//! | [`SWARM_NN_FORWARD`] | Swarm robotics | 015 | `barracuda::ops::batch_gemm` |
//! | [`HILL_GATE`] | Signal | 021 | `barracuda::ops::elementwise` |
//! | [`HEAD_SPLIT`] | MHA | — | `barracuda::ops::mha` |
//! | [`HEAD_CONCAT`] | MHA | — | `barracuda::ops::mha` |
//! | [`XOSHIRO128SS`] | PRNG | — | `barracuda::ops::prng` |

/// HMM forward pass in log-domain (Papers 016–018).
///
/// Computes `alpha_curr[s] = log_emit[s] + logsumexp_j(alpha_prev[j] + log_trans[j][s])`.
/// One thread per HMM state.
pub const HMM_FORWARD_LOG: &str = include_str!("../../shaders/hmm_forward_log.wgsl");

/// Batch linear fitness evaluation (Papers 011–015).
///
/// `fitness[i] = dot(population[i], weights)` for each individual.
/// One thread per individual in the population.
pub const BATCH_FITNESS_EVAL: &str = include_str!("../../shaders/batch_fitness_eval.wgsl");

/// Parallel 4th-order Runge-Kutta ODE integration (Papers 020–021).
///
/// Each thread integrates one independent ODE system for `n_steps` steps.
/// Uses a scratch buffer for intermediate RK4 stages.
pub const RK4_PARALLEL: &str = include_str!("../../shaders/rk4_parallel.wgsl");

/// Scalar mean reduction.
///
/// Single-workgroup reduction: `result[0] = mean(values)`.
/// Designed to chain after domain shaders in a pipeline.
pub const MEAN_REDUCE: &str = include_str!("../../shaders/mean_reduce.wgsl");

/// Pairwise Jaccard distance (Paper 024 — pangenome selection).
///
/// One thread per pair. `distance = 1 - |A ∩ B| / |A ∪ B|` over binary gene
/// presence/absence vectors.
pub const PAIRWISE_JACCARD: &str = include_str!("../../shaders/pairwise_jaccard.wgsl");

/// Per-locus allele frequency variance (Paper 025 — meta-population).
///
/// One thread per locus. Computes variance of allele frequencies across
/// sub-populations for `F_ST` estimation.
pub const LOCUS_VARIANCE: &str = include_str!("../../shaders/locus_variance.wgsl");

/// Spatial payoff computation on a grid (Paper 019 — game theory).
///
/// One thread per cell. Sums payoffs from Moore neighborhood interactions
/// using the prisoner's dilemma payoff matrix.
pub const SPATIAL_PAYOFF: &str = include_str!("../../shaders/spatial_payoff.wgsl");

/// Batch inverse participation ratio (Papers 022–023 — spectral analysis).
///
/// One thread per eigenvector. `IPR = sum(|psi_i|^4)` measures localization.
pub const BATCH_IPR: &str = include_str!("../../shaders/batch_ipr.wgsl");

/// Pairwise Hamming distance (Paper 017 — `SATé` alignment).
///
/// One thread per pair. Counts mismatches between aligned sequences.
pub const PAIRWISE_HAMMING: &str = include_str!("../../shaders/pairwise_hamming.wgsl");

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
    fn shader_count_is_16() {
        assert_eq!(
            16,
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
            ]
            .len()
        );
    }
}
