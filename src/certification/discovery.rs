// SPDX-License-Identifier: AGPL-3.0-or-later

//! Layer 1 (Discovery) — primal discovery via `CompositionContext`.
//!
//! Validates that `CompositionContext::from_live_discovery_with_fallback()`
//! finds the required capability families: tensor, security, compute, ai.

use log::{info, warn};
use primalspring::composition::{CompositionContext, validate_liveness};
use primalspring::validation::ValidationResult;

/// Required capability families for neuralSpring NUCLEUS.
pub const REQUIRED_CAPABILITIES: &[&str] = &["tensor", "security", "compute", "ai"];

/// Run discovery + liveness validation (L1).
///
/// Returns the `CompositionContext` for downstream layers and the count
/// of alive primals. If `alive == 0`, the caller should exit with code 2
/// (bare-only mode).
pub fn validate(v: &mut ValidationResult) -> (CompositionContext, usize) {
    let mut ctx = CompositionContext::from_live_discovery_with_fallback();

    let family_id = std::env::var("FAMILY_ID").ok();
    if let Some(ref fid) = family_id {
        info!("FAMILY_ID={fid} — family-isolated socket discovery");
    }

    let alive = validate_liveness(&mut ctx, v, REQUIRED_CAPABILITIES);

    if alive == 0 {
        warn!("No NUCLEUS primals discovered — bare guideStone only");
        info!("Deploy from plasmidBin and rerun for full certification");
    }

    (ctx, alive)
}
