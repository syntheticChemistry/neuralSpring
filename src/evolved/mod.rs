// SPDX-License-Identifier: AGPL-3.0-or-later

//! Locally evolved GPU-resident ops and WGSL shader exports.
//!
//! ## Absorption status (Feb 21, 2026 — `ToadStool` `dc540afd`)
//!
//! S-01 through S-11 absorbed by `ToadStool`. Deprecated workarounds
//! fossilized in `metalForge/fossils/evolved_s01_s11/`.
//!
//! ## Active Rust evolutions
//!
//! | Module | Why active | Path to absorption |
//! |--------|-----------|-------------------|
//! | `mha` | Native projection shaders hang (S-03b) — GPU `head_split`/`head_concat` shaders ready | `ToadStool`: replace fused projection with `matmul` + `head_split.wgsl` / `head_concat.wgsl` |
//! | `hmm_forward_gpu` | No `BarraCUDA` equivalent | Candidate for `ops::hmm` |
//!
//! ## WGSL shader inventory (absorption-ready)
//!
//! Following the hotSpring pattern, each WGSL shader is exported as a
//! `pub const` from its parent library module. The `ToadStool`/`BarraCUDA`
//! team can absorb these by copying the WGSL source directly.
//!
//! | WGSL Export | Library Module | Binary | Checks | Absorption Target |
//! |-------------|---------------|--------|--------|-------------------|
//! | [`hmm::WGSL_HMM_FORWARD_LOG`] | `hmm` | `validate_gpu_hmm_forward` | 13/13 | `barracuda::ops::hmm` |
//! | [`pangenome_selection::WGSL_PAIRWISE_JACCARD`] | `pangenome_selection` | `validate_gpu_pangenome` | 6/6 | `barracuda::ops::pairwise_distance` |
//! | [`meta_population::WGSL_LOCUS_VARIANCE`] | `meta_population` | `validate_gpu_meta_pop` | 7/7 | `barracuda::ops::VarianceReduceF64` |
//! | [`game_theory::WGSL_SPATIAL_PAYOFF`] | `game_theory` | `validate_gpu_game_theory` | 5/5 | `barracuda::ops::stencil` |
//! | [`anderson_localization::WGSL_BATCH_IPR`] | `anderson_localization` | `validate_gpu_anderson` | 5/5 | `barracuda::ops::batch_reduce` |
//! | [`sate_alignment::WGSL_PAIRWISE_HAMMING`] | `sate_alignment` | `validate_gpu_sate` | 5/5 | `barracuda::ops::pairwise_distance` |
//! | `WGSL_BATCH_FITNESS_EVAL` | (multi-paper) | `validate_gpu_batch_fitness` | 20/20 | `barracuda::ops::batch_gemm` |
//! | `WGSL_RK4_PARALLEL` | (multi-paper) | `validate_gpu_rk4` | 8/8 | `barracuda::ops::ode` |
//! | `WGSL_MEAN_REDUCE` | (aggregation) | `validate_gpu_pure_workload` | 7/7 | `barracuda::pipeline::ReduceScalarPipeline` |
//!
//! [`hmm::WGSL_HMM_FORWARD_LOG`]: crate::hmm::WGSL_HMM_FORWARD_LOG
//! [`pangenome_selection::WGSL_PAIRWISE_JACCARD`]: crate::pangenome_selection::WGSL_PAIRWISE_JACCARD
//! [`meta_population::WGSL_LOCUS_VARIANCE`]: crate::meta_population::WGSL_LOCUS_VARIANCE
//! [`game_theory::WGSL_SPATIAL_PAYOFF`]: crate::game_theory::WGSL_SPATIAL_PAYOFF
//! [`anderson_localization::WGSL_BATCH_IPR`]: crate::anderson_localization::WGSL_BATCH_IPR
//! [`sate_alignment::WGSL_PAIRWISE_HAMMING`]: crate::sate_alignment::WGSL_PAIRWISE_HAMMING

pub mod hmm_forward_gpu;
pub mod mha;

/// WGSL shader: GPU-resident head split for MHA.
///
/// Reindexes `[B, S, D]` → `[B, H, S, D/H]` in a single dispatch.
/// Replaces the CPU head-split in `evolved::mha` and the fused
/// projection+head-split in native `BarraCUDA` MHA (S-03b hang).
///
/// Absorption target: `barracuda::ops::mha`.
/// Validated: `validate_mha_gpu` (10/10 PASS).
pub const WGSL_HEAD_SPLIT: &str = include_str!("../../metalForge/shaders/head_split.wgsl");

/// WGSL shader: GPU-resident head concatenation for MHA.
///
/// Reindexes `[B, H, S, D/H]` → `[B, S, D]` in a single dispatch.
/// Replaces the CPU head-concat in `evolved::mha` and the fused
/// concat+projection in native `BarraCUDA` MHA (S-03b hang).
///
/// Absorption target: `barracuda::ops::mha`.
/// Validated: `validate_mha_gpu` (10/10 PASS).
pub const WGSL_HEAD_CONCAT: &str = include_str!("../../metalForge/shaders/head_concat.wgsl");

/// WGSL shader: parallel EA population fitness evaluation.
///
/// One thread per individual, dot-product fitness against target vector.
/// Spans Papers 011–015 (evolution workloads).
///
/// Absorption target: `barracuda::ops::batch_gemm`.
/// Validated: `validate_gpu_batch_fitness` (20/20 PASS).
pub const WGSL_BATCH_FITNESS_EVAL: &str =
    include_str!("../../metalForge/shaders/batch_fitness_eval.wgsl");

/// WGSL shader: parallel RK4 ODE integration.
///
/// One thread per system, full RK4 stepping with configurable step count.
/// Spans Papers 020–021 (regulatory network / signal integration).
///
/// Absorption target: `barracuda::ops::ode`.
/// Validated: `validate_gpu_rk4` (8/8 PASS).
pub const WGSL_RK4_PARALLEL: &str = include_str!("../../metalForge/shaders/rk4_parallel.wgsl");

/// WGSL shader: scalar mean reduction (chained after fitness evaluation).
///
/// Absorption target: `barracuda::pipeline::ReduceScalarPipeline`.
/// Validated: `validate_gpu_pure_workload` (7/7 PASS).
pub const WGSL_MEAN_REDUCE: &str = include_str!("../../metalForge/shaders/mean_reduce.wgsl");
