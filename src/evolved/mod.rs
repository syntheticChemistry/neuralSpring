// SPDX-License-Identifier: AGPL-3.0-or-later

//! Locally evolved GPU-resident ops and WGSL shader exports.
//!
//! ## Absorption status (Feb 22, 2026 — `ToadStool` `77f70b2e`)
//!
//! S-01 through S-12 absorbed by `ToadStool` (`77f70b2e`). 8 of 16 WGSL
//! shaders now sourced from upstream
//! `barracuda` (`ops::bio::*`, `ops::rk_stage`, `spectral::batch_ipr`).
//!
//! ## Active Rust evolutions
//!
//! | Module | Why active | Path to absorption |
//! |--------|-----------|-------------------|
//! | `mha` | Native projection shaders hang (S-03b) — GPU `head_split`/`head_concat` shaders ready | `ToadStool`: replace fused projection with `matmul` + `head_split.wgsl` / `head_concat.wgsl` |
//! | `hmm_forward_gpu` | Shader absorbed; local dispatch wrapper pending retirement | Migrate callers to `barracuda::ops::bio::HmmBatchForwardF64` |
//! | `tensor_sync` | S-13: `PooledBuffer` drop-before-completion race — `gpu_fence`, `fenced_matmul`, `materialize` | `ToadStool`: add `device.poll(Wait)` in `PooledBuffer::drop` before returning to pool |
//!
//! ## Retirement checklist
//!
//! A module can be retired (moved to `metalForge/fossils/`) when **all** of:
//!
//! 1. Upstream fix is merged into `BarraCUDA` or `ToadStool` main branch.
//! 2. `neuralSpring` validation binary passes using the upstream API.
//! 3. No downstream code imports from the evolved module.
//! 4. A fossil record commit documents the retirement (what, why, upstream SHA).
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
//! | [`modes::WGSL_PAIRWISE_L2`] | `modes` | `validate_gpu_modes` | 15/15 | `barracuda::ops::pairwise_distance` |
//! | [`directed_evolution::WGSL_MULTI_OBJ_FITNESS`] | `directed_evolution` | `validate_gpu_directed` | 6/6 | `barracuda::ops::batch_gemm` |
//! | [`swarm_robotics::WGSL_SWARM_NN_FORWARD`] | `swarm_robotics` | `validate_gpu_swarm` | 9/9 | `barracuda::ops::batch_gemm` |
//! | [`signal_integration::WGSL_HILL_GATE`] | `signal_integration` | `validate_gpu_signal` | 9/9 | `barracuda::ops::elementwise` |
//!
//! [`hmm::WGSL_HMM_FORWARD_LOG`]: crate::hmm::WGSL_HMM_FORWARD_LOG
//! [`pangenome_selection::WGSL_PAIRWISE_JACCARD`]: crate::pangenome_selection::WGSL_PAIRWISE_JACCARD
//! [`meta_population::WGSL_LOCUS_VARIANCE`]: crate::meta_population::WGSL_LOCUS_VARIANCE
//! [`game_theory::WGSL_SPATIAL_PAYOFF`]: crate::game_theory::WGSL_SPATIAL_PAYOFF
//! [`anderson_localization::WGSL_BATCH_IPR`]: crate::anderson_localization::WGSL_BATCH_IPR
//! [`sate_alignment::WGSL_PAIRWISE_HAMMING`]: crate::sate_alignment::WGSL_PAIRWISE_HAMMING
//! [`modes::WGSL_PAIRWISE_L2`]: crate::modes::WGSL_PAIRWISE_L2
//! [`directed_evolution::WGSL_MULTI_OBJ_FITNESS`]: crate::directed_evolution::WGSL_MULTI_OBJ_FITNESS
//! [`swarm_robotics::WGSL_SWARM_NN_FORWARD`]: crate::swarm_robotics::WGSL_SWARM_NN_FORWARD
//! [`signal_integration::WGSL_HILL_GATE`]: crate::signal_integration::WGSL_HILL_GATE

pub mod hmm_forward_gpu;
pub mod mha;
pub mod tensor_sync;

/// WGSL shader: GPU-resident head split for MHA.
///
/// Reindexes `[B, S, D]` → `[B, H, S, D/H]` in a single dispatch.
/// Replaces the CPU head-split in `evolved::mha` and the fused
/// projection+head-split in native `BarraCUDA` MHA (S-03b hang).
///
/// Absorption target: `barracuda::ops::mha`.
/// Validated: `validate_mha_gpu` (10/10 PASS).
pub use neural_spring_forge::shaders::HEAD_SPLIT as WGSL_HEAD_SPLIT;

/// WGSL shader: GPU-resident head concatenation for MHA.
///
/// Reindexes `[B, H, S, D/H]` → `[B, S, D]` in a single dispatch.
/// Replaces the CPU head-concat in `evolved::mha` and the fused
/// concat+projection in native `BarraCUDA` MHA (S-03b hang).
///
/// Absorption target: `barracuda::ops::mha`.
/// Validated: `validate_mha_gpu` (10/10 PASS).
pub use neural_spring_forge::shaders::HEAD_CONCAT as WGSL_HEAD_CONCAT;

/// WGSL shader: parallel EA population fitness evaluation.
///
/// One thread per individual, dot-product fitness against target vector.
/// Spans Papers 011–015 (evolution workloads).
///
/// Absorption target: `barracuda::ops::batch_gemm`.
/// Validated: `validate_gpu_batch_fitness` (20/20 PASS).
pub use neural_spring_forge::shaders::BATCH_FITNESS_EVAL as WGSL_BATCH_FITNESS_EVAL;

/// WGSL shader: parallel RK4 ODE integration.
///
/// One thread per system, full RK4 stepping with configurable step count.
/// Spans Papers 020–021 (regulatory network / signal integration).
///
/// Absorption target: `barracuda::ops::ode`.
/// Validated: `validate_gpu_rk4` (8/8 PASS).
pub use neural_spring_forge::shaders::RK4_PARALLEL as WGSL_RK4_PARALLEL;

/// WGSL shader: scalar mean reduction (chained after fitness evaluation).
///
/// Absorption target: `barracuda::pipeline::ReduceScalarPipeline`.
/// Validated: `validate_gpu_pure_workload` (7/7 PASS).
pub use neural_spring_forge::shaders::MEAN_REDUCE as WGSL_MEAN_REDUCE;
