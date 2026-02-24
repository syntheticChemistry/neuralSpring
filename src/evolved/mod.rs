// SPDX-License-Identifier: AGPL-3.0-or-later

//! Locally evolved GPU-resident ops and WGSL shader exports.
//!
//! ## Absorption status (Feb 24, 2026 — `ToadStool` S59 `9404fdb4`)
//!
//! S-01 through S-12 absorbed by `ToadStool`. All 16 WGSL shaders now have
//! upstream equivalents in `barracuda` (`ops::bio::*`, `ops::rk_stage`,
//! `spectral::batch_ipr`, `ops::prng`, `ops::LogsumexpWgsl`, typed bio ops).
//! Only `head_split`/`head_concat` remain truly local (MHA S-03b workaround).
//!
//! WGSL constants below are **absorbed upstream but retained** for raw shader
//! validation — our GPU pipeline validators depend on local binding layouts.
//! See `metalForge/shaders/ABSORPTION_TRACKER.md` for the full status.
//!
//! ## Session 47: MHA S-03b fixed upstream (`ToadStool` `fe573095`)
//!
//! The MHA projection under-dispatch bug (z-dimension `div_ceil(16)` → `seq_len`)
//! was fixed in `ToadStool` S46. The `mha` module is kept temporarily for backward
//! compatibility until upstream MHA is validated end-to-end here.
//!
//! ## Session 47: `hmm_forward_gpu` retired
//!
//! The local f32 HMM forward dispatch has been retired. All callers now use
//! upstream `barracuda::ops::bio::HmmBatchForwardF64` (f64, batch, wetSpring origin).
//! Local module moved to `metalForge/fossils/evolved_hmm_forward_gpu/`.
//!
//! ## Active Rust evolutions
//!
//! | Module | Why active | Path to absorption |
//! |--------|-----------|-------------------|
//! | `mha` | S-03b fixed upstream (`fe573095`); upstream `ops::mha` exists (S52+) but projection shaders need RTX 4070 validation | Validate upstream MHA at production sizes, then retire |
//!
//! ## Fossilized (Session 40)
//!
//! | Module | Reason | Location |
//! |--------|--------|----------|
//! | `tensor_sync` | S-13 **FIXED** upstream at `d45fdfb3` — zero callers remain | `metalForge/fossils/evolved_s13/` |
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
//! | ~~`WGSL_BATCH_FITNESS_EVAL`~~ | (removed S59) | rewired to `barracuda::ops::batch_gemm` | — | absorbed |
//! | ~~`WGSL_RK4_PARALLEL`~~ | (removed S59) | rewired to `barracuda::ops::rk_stage` | — | absorbed |
//! | ~~`WGSL_MEAN_REDUCE`~~ | (removed S59) | rewired to `barracuda::pipeline::ReduceScalarPipeline` | — | absorbed |
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

pub mod mha;

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

// WGSL_BATCH_FITNESS_EVAL, WGSL_RK4_PARALLEL, WGSL_MEAN_REDUCE removed (S59 sync):
// all callers have rewired to upstream `barracuda::ops::*` typed APIs.
// Shader source still available via `neural_spring_forge::shaders::*` if needed.
