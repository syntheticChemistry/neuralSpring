// SPDX-License-Identifier: AGPL-3.0-or-later

//! Certification organelle — neuralSpring's eukaryotic self-validation.
//!
//! Absorbed from `neuralspring_guidestone` binary during the interstadial
//! eukaryotic evolution (May 2026). Six layers, each additive:
//!
//! | Layer | Module           | Requires primals? | Description |
//! |-------|------------------|-------------------|-------------|
//! | L0    | [`bare`]         | No                | 5 certified properties |
//! | L1    | [`discovery`]    | Yes               | `CompositionContext` liveness |
//! | L2    | [`parity`]       | Yes               | Domain science parity (7 capabilities) |
//! | L3    | [`nucleus`]      | Yes               | Additive NUCLEUS (signing, discovery) |
//! | L4    | [`composition`]  | Yes               | NUCLEUS composition (graphs, registry, families) |
//! | L5    | [`cross_spring`] | Yes               | Cross-spring validation (artifacts, ping, determinism) |
//!
//! ## Usage
//!
//! ```ignore
//! use neural_spring::certification;
//! let result = certification::certify(5); // run all layers
//! std::process::exit(result.exit_code_skip_aware());
//! ```

pub mod bare;
pub mod composition;
pub mod cross_spring;
pub mod discovery;
pub mod nucleus;
pub mod parity;

use log::{info, warn};
use primalspring::validation::ValidationResult;

/// Maximum supported certification layer.
pub const MAX_LAYER: u8 = 5;

const SPRING_NAME: &str = crate::config::PRIMAL_DISPLAY_NAME;
const GUIDESTONE_VERSION: &str = "0.4.0";

/// Run certification up to `max_layer` (0-5) and return a [`ValidationResult`].
///
/// - L0: bare properties (determinism, traceability, checksums, env-agnostic, tolerances)
/// - L1: primal discovery + liveness
/// - L2: domain science parity (7 capabilities)
/// - L3: additive NUCLEUS (`BearDog` signing, Songbird discovery)
/// - L4: NUCLEUS composition (deploy graphs, capability registry, family calls)
/// - L5: cross-spring validation (frozen artifacts, protocol liveness, hash determinism)
///
/// Early-exits when `max_layer` is reached or when L1 discovers zero primals.
#[must_use]
pub fn certify(max_layer: u8) -> ValidationResult {
    let layer = max_layer.min(MAX_LAYER);

    ValidationResult::print_banner(&format!(
        "{SPRING_NAME} guideStone v{GUIDESTONE_VERSION} — Level {layer}"
    ));

    let mut v = ValidationResult::new(&format!("{SPRING_NAME} guideStone v{GUIDESTONE_VERSION}"));

    // L0: Bare Properties
    v.section("Layer 0: Bare Properties");
    bare::validate(&mut v);

    if layer == 0 {
        v.finish();
        return v;
    }

    // L1: Discovery + Liveness
    v.section("Layer 1: Discovery + Liveness");
    let (mut ctx, alive) = discovery::validate(&mut v);

    if alive == 0 {
        v.finish();
        return v;
    }

    if layer == 1 {
        v.finish();
        return v;
    }

    // L2: Domain Science Parity
    v.section("Layer 2: Domain Science Parity");
    parity::validate(&mut ctx, &mut v);

    if layer == 2 {
        v.finish();
        return v;
    }

    // L3: Additive NUCLEUS
    v.section("Layer 3: Additive NUCLEUS");
    nucleus::validate(&mut ctx, &mut v);

    if layer == 3 {
        v.finish();
        return v;
    }

    // L4: NUCLEUS Composition
    v.section("Layer 4: NUCLEUS Composition");
    composition::validate(&mut ctx, &mut v);

    if layer == 4 {
        v.finish();
        return v;
    }

    // L5: Cross-Spring Validation
    v.section("Layer 5: Cross-Spring Validation");
    cross_spring::validate(&mut ctx, &mut v);

    v.finish();

    let code = v.exit_code_skip_aware();
    match code {
        0 => info!("CERTIFIED: {SPRING_NAME} guideStone L{layer} — all checks passed"),
        1 => warn!("FAILED: {SPRING_NAME} guideStone L{layer} — regression detected"),
        2 => info!("BARE ONLY: {SPRING_NAME} guideStone — no NUCLEUS available"),
        _ => {}
    }

    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use primalspring::validation::CheckOutcome;

    #[test]
    fn max_layer_constant() {
        assert_eq!(MAX_LAYER, 5);
    }

    #[test]
    fn certify_bare_layer_returns_result() {
        let result = certify(0);
        let code = result.exit_code_skip_aware();
        assert!(
            code <= 2,
            "L0 exit code must be 0 (pass), 1 (fail/stale checksums), or 2 (skip), got {code}"
        );
    }

    #[test]
    fn certify_clamps_above_max() {
        let result = certify(255);
        let code = result.exit_code_skip_aware();
        assert!(code <= 2, "exit code must be 0, 1, or 2, got {code}");
    }

    #[test]
    fn certify_l0_runs_bare_property_checks_only() {
        let result = certify(0);
        assert!(
            result.experiment.contains("neuralSpring"),
            "experiment name should identify the spring"
        );
        assert!(
            result.experiment.contains("guideStone"),
            "experiment name should identify guideStone"
        );
        assert!(
            result.checks.iter().any(|c| c.name.starts_with("P1:")),
            "L0 must include deterministic RNG checks"
        );
        assert!(
            result.checks.iter().any(|c| c.name.starts_with("P2:")),
            "L0 must include provenance checks"
        );
        assert!(
            result.checks.iter().all(|c| !c.name.ends_with(".liveness")),
            "L0 must not run primal liveness checks"
        );
    }

    #[test]
    fn certify_l0_passes_when_checksums_valid() {
        let result = certify(0);
        if result.failed == 0 {
            assert!(result.all_passed());
            assert_eq!(result.exit_code_skip_aware(), 0);
        }
    }

    #[test]
    fn certify_without_primals_exits_after_discovery() {
        let l0 = certify(0);
        let l1 = certify(1);
        assert!(
            l1.checks.len() >= l0.checks.len(),
            "L1 should add discovery/liveness checks on top of L0"
        );
        let liveness_checks: Vec<_> = l1
            .checks
            .iter()
            .filter(|c| c.name.ends_with(".liveness"))
            .collect();
        assert!(
            !liveness_checks.is_empty(),
            "L1 should record liveness checks even without primals"
        );
        assert!(
            liveness_checks
                .iter()
                .any(|c| matches!(c.outcome, CheckOutcome::Skip)),
            "unreachable primals should be recorded as skips, not failures"
        );
        assert!(l1.exit_code_skip_aware() <= 2);
    }

    #[test]
    fn certify_layer_clamp_matches_max_layer() {
        let at_max = certify(MAX_LAYER);
        let above_max = certify(MAX_LAYER + 1);
        assert_eq!(
            at_max.exit_code_skip_aware(),
            above_max.exit_code_skip_aware(),
            "layers above MAX_LAYER should clamp to MAX_LAYER"
        );
    }
}
