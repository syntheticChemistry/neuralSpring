// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sovereign Folding: Evoformer primitive implementations for CPU validation.
//!
//! Phase B of the sovereign folding track. Provides pure Rust f64
//! reference implementations of `AlphaFold2`'s Evoformer operations:
//!
//! - [`gelu`] — GELU activation (Hendrycks & Gimpel 2016)
//! - [`layer_norm`] — Layer normalization (Ba et al. 2016)
//! - [`softmax_rows`] — Row-wise numerically stable softmax
//! - [`sdpa_scores`] — Scaled dot-product attention scores (QKᵀ/√d)
//! - [`attention_apply`] — Weighted value summation (weights × V)
//! - [`sdpa_full`] — Complete SDPA pipeline (scores → softmax → apply)
//! - [`triangle_mul_outgoing`] — Algorithm 11 (Jumper et al. 2021)
//! - [`triangle_mul_incoming`] — Algorithm 12
//! - [`triangle_attention_scores`] — Algorithms 13-14 with pair bias
//!
//! ## References
//!
//! - Jumper et al. "Highly accurate protein structure prediction with
//!   `AlphaFold`" Nature 596:583-589 (2021)
//! - Ahdritz et al. "`OpenFold`: Retraining `AlphaFold2` yields new insights
//!   into its learning mechanisms and capacity for generalization"
//!   Nature Methods (2024)
//!
//! ## Evolution path
//!
//! ```text
//! NumPy baseline → Rust CPU → WGSL shader → sovereign pipeline
//! ```

mod activation;
mod attention;
#[allow(clippy::similar_names, clippy::cast_precision_loss)]
pub mod confidence;
pub mod diffusion;
mod msa;
pub mod pairformer;
mod triangle;

// Re-export all public API items for backward compatibility.
pub use activation::{gelu, gelu_vec, layer_norm, softmax_rows};
pub use attention::{attention_apply, sdpa_full, sdpa_scores};
pub use confidence::{pae_head, pde_head, plddt_head, ranking_score, RankingWeights};
pub use diffusion::{
    cosine_beta_schedule, ddim_reverse_step, ddpm_reverse_step, forward_diffusion,
    linear_beta_schedule, pair_transition_ffn, remove_center_of_mass, se3_equivariant_noise,
    NoiseSchedule,
};
pub use pairformer::{
    condition_pair_with_timestep, pairformer_block, sinusoidal_embedding, PairformerWeights,
};
pub use msa::{
    msa_col_attention, msa_col_attention_scores, msa_row_attention, msa_row_attention_scores,
    outer_product_mean,
};
pub use triangle::{triangle_attention_scores, triangle_mul_incoming, triangle_mul_outgoing};
