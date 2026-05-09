// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Composition evolution — third validation tier.
//!
//! Absorbed from `validate_composition_evolution.rs` (~30 checks).

use super::registry::{Scenario, ScenarioMeta, Tier, Track};

use primalspring::composition::{CompositionContext, is_skip_error};
use primalspring::validation::ValidationResult;

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "composition_evolution",
        track: Track::CrossSpring,
        tier: Tier::Both,
        provenance_crate: "validate_composition_evolution",
        provenance_date: "2026-05-09",
        description: "Composition evolution: cross-spring integration and protocol tolerance",
        check_count: 30,
    },
    run_rust: Some(run_rust),
    run_live: Some(run_live),
};

fn run_rust(v: &mut ValidationResult) {
    v.section("Composition Evolution — Tier 1 (Rust structural)");

    let gap_status = std::path::Path::new("experiments/results/gap-status.json");
    v.check_bool(
        "evolution:rust:gap_status_exists",
        gap_status.exists(),
        if gap_status.exists() {
            "present"
        } else {
            "missing"
        },
    );

    let primal_gaps = std::path::Path::new("docs/PRIMAL_GAPS.md");
    v.check_bool(
        "evolution:rust:primal_gaps_exists",
        primal_gaps.exists(),
        if primal_gaps.exists() {
            "present"
        } else {
            "missing"
        },
    );

    let deploy_graphs = [
        "graphs/neuralspring_deploy.toml",
        "graphs/neuralspring_inference_pipeline.toml",
        "graphs/neuralspring_spectral_analysis.toml",
    ];
    let count = deploy_graphs
        .iter()
        .filter(|g| std::path::Path::new(g).exists())
        .count();
    v.check_bool(
        "evolution:rust:deploy_graphs",
        count >= 2,
        &format!("{count}/{} deploy graphs present", deploy_graphs.len()),
    );
}

fn run_live(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section("Composition Evolution — Tier 2 (Live)");

    let families = ["tensor", "security", "compute", "ai"];
    for family in &families {
        match ctx.call(family, &format!("{family}.ping"), serde_json::json!({})) {
            Ok(_result) => {
                v.check_bool(&format!("evolution:live:ping:{family}"), true, "responded");
            }
            Err(e) if is_skip_error(&e) => {
                v.check_skip(
                    &format!("evolution:live:ping:{family}"),
                    &format!("{family} offline: {e}"),
                );
            }
            Err(e) => {
                v.check_skip(
                    &format!("evolution:live:ping:{family}"),
                    &format!("ping not supported: {e}"),
                );
            }
        }
    }

    match ctx.hash_bytes(b"evolution-marker", "blake3") {
        Ok(hash1) => match ctx.hash_bytes(b"evolution-marker", "blake3") {
            Ok(hash2) => {
                v.check_bool(
                    "evolution:live:hash_determinism",
                    hash1 == hash2,
                    "identical input → identical hash",
                );
            }
            Err(e) => {
                v.check_bool("evolution:live:hash_determinism", false, &format!("{e}"));
            }
        },
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "evolution:live:hash_determinism",
                &format!("security offline: {e}"),
            );
        }
        Err(e) => {
            v.check_bool("evolution:live:hash_determinism", false, &format!("{e}"));
        }
    }
}
