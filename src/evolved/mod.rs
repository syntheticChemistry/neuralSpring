// SPDX-License-Identifier: AGPL-3.0-or-later

//! Locally evolved GPU-resident ops and WGSL shader exports.
//!
//! ## Absorption status (Mar 4, 2026 — barraCuda v0.3.1 standalone)
//!
//! **All 21 WGSL shaders absorbed upstream.** S-01 through S-12 absorbed by
//! `ToadStool` `77f70b2e`. `head_split.wgsl` and `head_concat.wgsl` absorbed
//! by `ToadStool` `0c998992` (S60–S61) as part of the MHA decomposition that
//! resolved S-03b. `barraCuda` extracted from `ToadStool` at S89; neuralSpring
//! now depends on standalone `../barraCuda/crates/barracuda` v0.3.1.
//!
//! The `mha` module provides a 2D→3D→2D adapter over upstream
//! `barracuda::ops::mha::MultiHeadAttention` for science callers that
//! work with `[seq, d_model]` matrices rather than `[batch, seq, d_model]`.
//!
//! WGSL constants below are **absorbed upstream but retained** for raw shader
//! validation — our GPU pipeline validators depend on local binding layouts.
//! See `metalForge/shaders/ABSORPTION_TRACKER.md` for the full status.
//!
//! ## Session 62 sync: S-03b fully resolved
//!
//! `ToadStool` `0c998992` decomposed the fused MHA projection into
//! `Tensor::matmul` + `head_split.wgsl` / `head_concat.wgsl` —
//! exactly the approach neuralSpring evolved locally. The upstream
//! MHA now works at production sizes (B=4, S=128, H=8, d=512).
//!
//! ## Active Rust evolutions
//!
//! **None.** All evolved modules now delegate to upstream `BarraCUDA`.
//!
//! | Module | Status | Upstream |
//! |--------|--------|----------|
//! | `mha` | Thin wrapper (delegates to upstream) | `barracuda::ops::mha::MultiHeadAttention` |
//!
//! ## Fossilized (Session 40)
//!
//! | Module | Reason | Location |
//! |--------|--------|----------|
//! | `tensor_sync` | S-13 **FIXED** upstream at `d45fdfb3` — zero callers remain | `metalForge/fossils/evolved_s13/` |
//!
//! ## WGSL shader inventory (all absorbed)
//!
//! All 21 WGSL shaders from neuralSpring have been absorbed into upstream
//! `BarraCUDA`. Local shader constants are retained for raw shader validation.
//!
//! | WGSL Export | Library Module | Absorption Status |
//! |-------------|---------------|-------------------|
//! | [`hmm::WGSL_HMM_FORWARD_LOG`] | `hmm` | Absorbed |
//! | [`pangenome_selection::WGSL_PAIRWISE_JACCARD`] | `pangenome_selection` | Absorbed |
//! | [`meta_population::WGSL_LOCUS_VARIANCE`] | `meta_population` | Absorbed |
//! | [`game_theory::WGSL_SPATIAL_PAYOFF`] | `game_theory` | Absorbed |
//! | [`anderson_localization::WGSL_BATCH_IPR`] | `anderson_localization` | Absorbed |
//! | [`sate_alignment::WGSL_PAIRWISE_HAMMING`] | `sate_alignment` | Absorbed |
//! | [`modes::WGSL_PAIRWISE_L2`] | `modes` | Absorbed |
//! | [`directed_evolution::WGSL_MULTI_OBJ_FITNESS`] | `directed_evolution` | Absorbed |
//! | [`swarm_robotics::WGSL_SWARM_NN_FORWARD`] | `swarm_robotics` | Absorbed |
//! | [`signal_integration::WGSL_HILL_GATE`] | `signal_integration` | Absorbed |
//! | `head_split.wgsl` | MHA | Absorbed S60–S61 (`0c998992`) |
//! | `head_concat.wgsl` | MHA | Absorbed S60–S61 (`0c998992`) |
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
/// **Absorbed upstream**: `ToadStool` `0c998992` (S60–S61).
/// Retained for local validation via `validate_mha_gpu`.
pub use neural_spring_forge::shaders::HEAD_SPLIT as WGSL_HEAD_SPLIT;

/// WGSL shader: GPU-resident head concatenation for MHA.
///
/// Reindexes `[B, H, S, D/H]` → `[B, S, D]` in a single dispatch.
/// **Absorbed upstream**: `ToadStool` `0c998992` (S60–S61).
/// Retained for local validation via `validate_mha_gpu`.
pub use neural_spring_forge::shaders::HEAD_CONCAT as WGSL_HEAD_CONCAT;
