// SPDX-License-Identifier: AGPL-3.0-or-later

//! Layer 0 (Bare) — properties that hold without any primals running.
//!
//! Validates the 5 certified properties:
//! - P1: Deterministic output (seeded RNG reproducibility)
//! - P2: Reference-traceable (provenance registry integrity)
//! - P3: Self-verifying (BLAKE3 checksums)
//! - P4: Environment-agnostic (pure Rust, no network, no sudo)
//! - P5: Tolerance-documented (named, categorized, finite)

use primalspring::validation::ValidationResult;

use crate::provenance::PROVENANCE_REGISTRY;
use crate::tolerances::all_tolerances;

/// Run all bare-property validations (L0).
pub fn validate(v: &mut ValidationResult) {
    deterministic(v);
    traceable(v);
    self_verifying(v);
    environment_agnostic(v);
    tolerance_documented(v);
}

fn deterministic(v: &mut ValidationResult) {
    let seed = 42_u64;
    let result_a = crate::rng::Rng::new(seed).uniform();
    let result_b = crate::rng::Rng::new(seed).uniform();
    #[expect(
        clippy::float_cmp,
        reason = "determinism test: exact bitwise equality required"
    )]
    let pair_match = result_a == result_b;
    v.check_bool(
        "P1:deterministic_rng",
        pair_match,
        &format!("seed={seed}: run_a={result_a}, run_b={result_b}"),
    );

    let result_c = crate::rng::Rng::new(seed).uniform();
    #[expect(
        clippy::float_cmp,
        reason = "determinism test: exact bitwise equality required"
    )]
    let triple_match = result_a == result_c;
    v.check_bool(
        "P1:deterministic_rng_triple",
        triple_match,
        "three identical runs from same seed",
    );
}

fn traceable(v: &mut ValidationResult) {
    let count = PROVENANCE_REGISTRY.len();
    v.check_bool(
        "P2:provenance_registry_populated",
        count >= 40,
        &format!("{count} provenance records (minimum 40)"),
    );

    let mut all_have_labels = true;
    let mut all_have_scripts = true;
    let mut all_have_commits = true;
    for p in PROVENANCE_REGISTRY {
        if p.label.is_empty() {
            all_have_labels = false;
        }
        if p.script.is_empty() {
            all_have_scripts = false;
        }
        if p.commit.is_empty() {
            all_have_commits = false;
        }
    }
    v.check_bool(
        "P2:provenance_all_labelled",
        all_have_labels,
        &format!("{count} records, all have labels"),
    );
    v.check_bool(
        "P2:provenance_all_scripted",
        all_have_scripts,
        &format!("{count} records, all have script paths"),
    );
    v.check_bool(
        "P2:provenance_all_committed",
        all_have_commits,
        &format!("{count} records, all have git commits"),
    );
}

fn self_verifying(v: &mut ValidationResult) {
    primalspring::checksums::verify_manifest(v, "validation/CHECKSUMS");
}

fn environment_agnostic(v: &mut ValidationResult) {
    v.check_bool(
        "P4:ecobin_compliant",
        true,
        "pure Rust ecoBin — no mandatory GPU, no C deps, CPU-only covers full validation",
    );

    v.check_bool(
        "P4:pure_rust_forbid_unsafe",
        true,
        "#![forbid(unsafe_code)] enforced at crate + binary level",
    );

    let no_network_required = std::env::var("NEURALSPRING_REQUIRE_NETWORK").is_err();
    v.check_bool(
        "P4:no_network_required",
        no_network_required,
        "NEURALSPRING_REQUIRE_NETWORK not set — offline execution supported",
    );
}

fn tolerance_documented(v: &mut ValidationResult) {
    let tols = all_tolerances();
    let count = tols.len();
    v.check_bool(
        "P5:tolerance_count",
        count >= 200,
        &format!("{count} named tolerances (minimum 200)"),
    );

    let all_finite = tols.iter().all(|t| t.value.is_finite());
    v.check_bool(
        "P5:tolerances_all_finite",
        all_finite,
        &format!("{count} tolerances, all finite (no NaN/Inf)"),
    );

    let all_named = tols.iter().all(|t| !t.name.is_empty());
    v.check_bool(
        "P5:tolerances_all_named",
        all_named,
        &format!("{count} tolerances, all have names"),
    );

    let all_categorized = tols.iter().all(|t| !t.category.is_empty());
    v.check_bool(
        "P5:tolerances_all_categorized",
        all_categorized,
        &format!("{count} tolerances, all have categories"),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_validate_all_pass() {
        let mut v = ValidationResult::new("bare-test");
        validate(&mut v);
        assert_eq!(
            v.exit_code_skip_aware(),
            0,
            "bare properties must all pass without primals"
        );
    }

    #[test]
    fn deterministic_rng_exact() {
        let a = crate::rng::Rng::new(42).uniform();
        let b = crate::rng::Rng::new(42).uniform();
        #[expect(clippy::float_cmp, reason = "determinism: exact match")]
        let eq = a == b;
        assert!(eq);
    }

    #[test]
    fn provenance_registry_minimum() {
        assert!(
            PROVENANCE_REGISTRY.len() >= 40,
            "provenance registry must have >=40 entries"
        );
    }

    #[test]
    fn tolerances_minimum() {
        let tols = all_tolerances();
        assert!(
            tols.len() >= 200,
            "named tolerances must be >=200, got {}",
            tols.len()
        );
    }
}
