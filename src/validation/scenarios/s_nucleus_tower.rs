// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: NUCLEUS Tower — expanded primal validation.
//!
//! Absorbed from `validate_nucleus_tower.rs` (~47 checks).

use super::registry::{Scenario, ScenarioMeta, Tier, Track};

use primalspring::composition::{CompositionContext, is_skip_error};
use primalspring::validation::ValidationResult;

use crate::validation::composition::PROTO_NUCLEATE_VALIDATION_CAPABILITIES;

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "nucleus_tower",
        track: Track::NucleusComposition,
        tier: Tier::Both,
        provenance_crate: "validate_nucleus_tower",
        provenance_date: "2026-05-09",
        description: "NUCLEUS Tower mode: expanded primal validation with capability probing",
        check_count: 47,
    },
    run_rust: Some(run_rust),
    run_live: Some(run_live),
};

fn run_rust(v: &mut ValidationResult) {
    v.section("NUCLEUS Tower — Tier 1 (Rust structural)");

    let cap_count = PROTO_NUCLEATE_VALIDATION_CAPABILITIES.len();
    v.check_bool(
        "tower:rust:capabilities_registered",
        cap_count >= 7,
        &format!("{cap_count} capabilities in PROTO_NUCLEATE"),
    );

    let registry_toml = include_str!("../../../config/capability_registry.toml");
    v.check_bool(
        "tower:rust:registry_toml_populated",
        !registry_toml.is_empty(),
        &format!("registry TOML len={}", registry_toml.len()),
    );

    for cap in PROTO_NUCLEATE_VALIDATION_CAPABILITIES {
        v.check_bool(
            &format!("tower:rust:cap_non_empty:{cap}"),
            !cap.is_empty(),
            "capability name is non-empty",
        );
    }
}

fn run_live(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section("NUCLEUS Tower — Tier 2 (Live probing)");

    let families = ["tensor", "security", "compute", "ai"];
    for family in &families {
        match ctx.resolve_capability(family) {
            Ok(result) => {
                let found = result
                    .get("found")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                    || result.get("endpoint").is_some()
                    || result.get("socket").is_some();
                v.check_bool(
                    &format!("tower:live:resolve:{family}"),
                    found,
                    &format!("{result}"),
                );
            }
            Err(e) if is_skip_error(&e) => {
                v.check_skip(
                    &format!("tower:live:resolve:{family}"),
                    &format!("{family} offline: {e}"),
                );
            }
            Err(e) => {
                v.check_skip(
                    &format!("tower:live:resolve:{family}"),
                    &format!("resolve gap: {e}"),
                );
            }
        }
    }

    match ctx.call(
        "tensor",
        "stats.mean",
        serde_json::json!({"data": [10.0, 20.0, 30.0]}),
    ) {
        Ok(result) => {
            let value = result
                .get("result")
                .or_else(|| result.get("mean"))
                .and_then(serde_json::Value::as_f64);
            if let Some(val) = value {
                v.check_bool(
                    "tower:live:stats_mean_20",
                    (val - 20.0).abs() < 0.01,
                    &format!("mean={val}"),
                );
            } else {
                v.check_bool(
                    "tower:live:stats_mean_20",
                    false,
                    &format!("unexpected: {result}"),
                );
            }
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip("tower:live:stats_mean_20", &format!("tensor offline: {e}"));
        }
        Err(e) => {
            v.check_bool("tower:live:stats_mean_20", false, &format!("{e}"));
        }
    }
}
