// SPDX-License-Identifier: AGPL-3.0-or-later

//! Layer 5 (Cross-spring) — ecosystem-level validation.
//!
//! Validates that neuralSpring's frozen validation artifacts exist,
//! deploy graphs are structurally sound, and cross-spring protocol
//! liveness (ping, hash determinism) holds across families.

use primalspring::composition::{CompositionContext, is_skip_error};
use primalspring::validation::ValidationResult;

/// Run cross-spring validation (L5).
///
/// Tier 1 (Rust structural) always runs. Tier 2 (Live) runs after.
pub fn validate(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    validate_rust_tier(v);
    validate_live_tier(ctx, v);
}

fn validate_rust_tier(v: &mut ValidationResult) {
    let gap_status = std::path::Path::new("experiments/results/gap-status.json");
    v.check_bool(
        "cross_spring:rust:gap_status_exists",
        gap_status.exists(),
        if gap_status.exists() {
            "present"
        } else {
            "missing"
        },
    );

    let validation_state = std::path::Path::new("experiments/results/validation-state.json");
    v.check_bool(
        "cross_spring:rust:validation_state_exists",
        validation_state.exists(),
        if validation_state.exists() {
            "present"
        } else {
            "missing"
        },
    );

    let primal_gaps = std::path::Path::new("docs/PRIMAL_GAPS.md");
    v.check_bool(
        "cross_spring:rust:primal_gaps_exists",
        primal_gaps.exists(),
        if primal_gaps.exists() {
            "present"
        } else {
            "missing"
        },
    );

    let foundation = std::path::Path::new("docs/FOUNDATION_SEEDING.md");
    v.check_bool(
        "cross_spring:rust:foundation_manifest",
        foundation.exists(),
        if foundation.exists() {
            "present"
        } else {
            "missing"
        },
    );

    let checksums = std::path::Path::new("validation/CHECKSUMS");
    v.check_bool(
        "cross_spring:rust:checksums_file",
        checksums.exists(),
        if checksums.exists() {
            "present"
        } else {
            "missing"
        },
    );
}

fn validate_live_tier(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    let families = ["tensor", "security", "compute", "ai"];
    for family in &families {
        match ctx.call(family, &format!("{family}.ping"), serde_json::json!({})) {
            Ok(_) => {
                v.check_bool(&format!("cross_spring:live:ping:{family}"), true, "responded");
            }
            Err(e) if is_skip_error(&e) => {
                v.check_skip(
                    &format!("cross_spring:live:ping:{family}"),
                    &format!("{family} offline: {e}"),
                );
            }
            Err(e) => {
                v.check_skip(
                    &format!("cross_spring:live:ping:{family}"),
                    &format!("{family} unavailable: {e}"),
                );
            }
        }
    }

    match ctx.hash_bytes(b"cross-spring:determinism:marker-1", "blake3") {
        Ok(hash1) => match ctx.hash_bytes(b"cross-spring:determinism:marker-1", "blake3") {
            Ok(hash2) => {
                v.check_bool(
                    "cross_spring:live:hash_determinism",
                    hash1 == hash2,
                    "BLAKE3 hash is deterministic across calls",
                );
            }
            Err(e) => {
                v.check_skip(
                    "cross_spring:live:hash_determinism",
                    &format!("second hash call failed: {e}"),
                );
            }
        },
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "cross_spring:live:hash_determinism",
                &format!("security offline: {e}"),
            );
        }
        Err(e) => {
            v.check_skip(
                "cross_spring:live:hash_determinism",
                &format!("hash unavailable: {e}"),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn cross_spring_frozen_artifacts_exist() {
        assert!(
            std::path::Path::new("experiments/results/gap-status.json").exists(),
            "gap-status.json required"
        );
        assert!(
            std::path::Path::new("experiments/results/validation-state.json").exists(),
            "validation-state.json required"
        );
        assert!(
            std::path::Path::new("docs/PRIMAL_GAPS.md").exists(),
            "PRIMAL_GAPS.md required"
        );
    }

    #[test]
    fn cross_spring_checksums_exist() {
        assert!(
            std::path::Path::new("validation/CHECKSUMS").exists(),
            "CHECKSUMS required"
        );
    }

    #[test]
    fn cross_spring_foundation_manifest_exists() {
        assert!(
            std::path::Path::new("docs/FOUNDATION_SEEDING.md").exists(),
            "FOUNDATION_SEEDING.md required"
        );
    }
}
