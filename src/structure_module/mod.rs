// SPDX-License-Identifier: AGPL-3.0-or-later

//! Structure Module: rigid-body frame operations and Invariant Point Attention.
//!
//! Phase B.3 of the sovereign folding track. Implements the Structure Module
//! from `AlphaFold2` (Jumper et al. 2021, Algorithm 22):
//!
//! - Frame operations: quaternion→rotation, frame application/inversion
//! - [`ipa_scores`] — Invariant Point Attention score computation
//! - [`ipa_apply`] — IPA weighted value summation (scalar + point outputs)
//! - [`backbone_update`] — Frame composition from predicted updates
//!
//! ## Frame representation
//!
//! Each residue frame is stored as 12 f64 values: rotation matrix (9, row-major)
//! followed by translation vector (3). This avoids quaternion singularities
//! during GPU computation.
//!
//! ## IPA attention score (Algorithm 22)
//!
//! ```text
//! a[h,i,j] = w_L * Q·K/√c
//!          + w_C * pair_bias[h,i,j]
//!          + w_P * (-γ/2) * Σ_p ||T_i(q_p) - T_j(k_p)||²
//! ```
//!
//! The point distance term makes attention SE(3)-equivariant: scores depend
//! on 3D proximity of query/key points projected through backbone frames.
//!
//! ## References
//!
//! - Jumper et al. "Highly accurate protein structure prediction with
//!   `AlphaFold`" Nature 596:583-589 (2021), Algorithm 22
//! - Ahdritz et al. "`OpenFold`" Nature Methods (2024)

mod backbone;
mod frame;
mod ipa;

// Re-exports: public API
pub use backbone::{backbone_update, torsion_angles};
pub use frame::{apply_frame, compose_frames, invert_frame, quat_to_rotation};
pub use ipa::{ipa_apply, ipa_scores, IpaConfig};
