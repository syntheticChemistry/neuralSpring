// SPDX-License-Identifier: AGPL-3.0-or-later

//! coralForge: sovereign structure prediction engine.
//!
//! Pure Rust f64 reference implementations of structure prediction
//! primitives, validated against `AlphaFold2`/`AlphaFold3` baselines
//! and accelerated via `BarraCUDA`.
//!
//! ## Primitives
//!
//! - [`gelu`], [`layer_norm`], [`softmax_rows`] — activations
//! - [`sdpa_scores`], [`sdpa_full`] — scaled dot-product attention
//! - [`triangle_mul_outgoing`], [`triangle_mul_incoming`] — Algorithms 11-12
//! - [`triangle_attention_scores`] — Algorithms 13-14 with pair bias
//! - [`msa_row_attention`], [`msa_col_attention`] — MSA attention
//! - [`structure`] — IPA, backbone frames, torsion angles
//! - [`diffusion`] — DDPM/DDIM noise schedules, SE(3) equivariance
//! - [`pairformer`] — Pairformer block with timestep conditioning
//! - [`confidence`] — pLDDT, PAE, pDE confidence heads
//!
//! ## Evolution path
//!
//! ```text
//! Python baseline → Rust CPU → BarraCUDA → WGSL shader → sovereign pipeline
//! ```

mod activation;
mod attention;
#[expect(
    clippy::similar_names,
    clippy::cast_precision_loss,
    reason = "confidence metrics use short variable names and count→f64 casts"
)]
pub mod confidence;
pub mod diffusion;
mod msa;
pub mod pairformer;
pub mod structure;
mod triangle;

// Re-export all public API items for backward compatibility.
pub use activation::{gelu, gelu_vec, layer_norm, softmax_rows};
pub use attention::{attention_apply, sdpa_full, sdpa_scores};
pub use confidence::{RankingWeights, pae_head, pde_head, plddt_head, ranking_score};
pub use diffusion::{
    NoiseSchedule, cosine_beta_schedule, ddim_reverse_step, ddpm_reverse_step, forward_diffusion,
    linear_beta_schedule, pair_transition_ffn, remove_center_of_mass, se3_equivariant_noise,
};
pub use msa::{
    msa_col_attention, msa_col_attention_scores, msa_row_attention, msa_row_attention_scores,
    outer_product_mean,
};
pub use pairformer::{
    PairformerWeights, condition_pair_with_timestep, pairformer_block, sinusoidal_embedding,
};
pub use triangle::{triangle_attention_scores, triangle_mul_incoming, triangle_mul_outgoing};
