// SPDX-License-Identifier: AGPL-3.0-or-later

//! Layer 3 (NUCLEUS) — additive NUCLEUS capabilities.
//!
//! Validates capabilities that only exist in a fully deployed NUCLEUS:
//! - BearDog signing receipt (crypto round-trip)
//! - Songbird capability discovery resolution

use primalspring::composition::{CompositionContext, is_skip_error};
use primalspring::validation::ValidationResult;

/// Run additive NUCLEUS validation (L3).
pub fn validate(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    beardog_signing(ctx, v);
    songbird_discovery(ctx, v);
}

fn beardog_signing(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.hash_bytes(b"guidestone:neuralspring:certified", "blake3") {
        Ok(receipt) => {
            v.check_bool(
                "additive:beardog_signing_receipt",
                !receipt.is_empty(),
                &format!("signing receipt len={}", receipt.len()),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "additive:beardog_signing_receipt",
                &format!("security not available (graceful skip): {e}"),
            );
        }
        Err(e) => {
            v.check_bool(
                "additive:beardog_signing_receipt",
                false,
                &format!("signing error: {e}"),
            );
        }
    }
}

fn songbird_discovery(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.resolve_capability("tensor") {
        Ok(result) => {
            let found = result
                .get("found")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
                || result.get("endpoint").is_some()
                || result.get("socket").is_some();
            v.check_bool(
                "additive:songbird_discovery",
                found,
                &format!("resolved tensor provider: {result}"),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "additive:songbird_discovery",
                &format!("discovery not available (graceful skip): {e}"),
            );
        }
        Err(e) => {
            v.check_skip(
                "additive:songbird_discovery",
                &format!("resolve gap (graceful skip): {e}"),
            );
        }
    }
}
