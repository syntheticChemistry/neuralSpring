// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Schema standard validation — Wave 20 envelope compliance.
//!
//! Validates that neuralSpring's API responses conform to the ecosystem
//! schema standards:
//! - `capability.list` returns `{ primal, capabilities, count }` envelope
//! - `primal.info` returns canonical shape
//! - Live probe of biomeOS `primal.list` when available

use super::registry::{Scenario, ScenarioMeta, Tier, Track};

use primalspring::composition::{CompositionContext, is_skip_error};
use primalspring::validation::ValidationResult;

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "schema_standard",
        track: Track::CrossSpring,
        tier: Tier::Both,
        provenance_crate: "neuralspring_schema_validation",
        provenance_date: "2026-05-16",
        description: "Wave 20 schema: capability.list envelope, primal.list probe",
        check_count: 9,
    },
    run_rust: Some(run_rust),
    run_live: Some(run_live),
};

fn run_rust(v: &mut ValidationResult) {
    v.section("Schema Standard — Tier 1 (Rust structural)");

    let handler_src = include_str!("../../bin/neuralspring_primal/handlers.rs");
    v.check_bool(
        "schema:rust:capability_list_has_count",
        handler_src.contains("ALL_CAPABILITIES.len()"),
        "capability.list handler includes count field",
    );
    v.check_bool(
        "schema:rust:capability_list_has_primal",
        handler_src.contains("\"primal\": PRIMAL_NAME")
            || handler_src.contains("\"primal\": PRIMAL_NAME,"),
        "capability.list handler includes primal field",
    );
    v.check_bool(
        "schema:rust:capability_list_has_capabilities",
        handler_src.contains("\"capabilities\": ALL_CAPABILITIES"),
        "capability.list handler includes capabilities array",
    );

    let caps = crate::config::ALL_CAPABILITIES;
    v.check_bool(
        "schema:rust:all_caps_non_empty",
        !caps.is_empty(),
        &format!("ALL_CAPABILITIES has {} entries", caps.len()),
    );
    v.check_bool(
        "schema:rust:all_caps_have_dots",
        caps.iter().all(|c| c.contains('.')),
        "all capabilities follow domain.verb naming convention",
    );

    let config_toml = include_str!("../../../config/capability_registry.toml");
    let has_primal_list = config_toml.contains("primal.list");
    v.check_bool(
        "schema:rust:primal_list_in_registry",
        has_primal_list,
        "primal.list in local capability_registry.toml (Wave 20)",
    );
}

fn run_live(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section("Schema Standard — Tier 2 (Live)");

    match ctx.call("orchestration", "primal.list", serde_json::json!({})) {
        Ok(result) => {
            let has_primals_key = result.get("primals").is_some()
                || result.get("primal_ids").is_some()
                || result.is_array();
            v.check_bool(
                "schema:live:primal_list_responds",
                true,
                &format!("primal.list responded with {}", summarize_keys(&result)),
            );
            v.check_bool(
                "schema:live:primal_list_shape",
                has_primals_key,
                "primal.list response contains primals collection",
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_bool(
                "schema:live:primal_list_responds",
                true,
                "SKIP — biomeOS orchestrator not running",
            );
            v.check_bool(
                "schema:live:primal_list_shape",
                true,
                "SKIP — depends on primal.list (skipped)",
            );
        }
        Err(e) => {
            v.check_bool(
                "schema:live:primal_list_responds",
                false,
                &format!("primal.list failed: {e}"),
            );
            v.check_bool(
                "schema:live:primal_list_shape",
                false,
                "primal.list response unavailable",
            );
        }
    }

    match ctx.call(
        "orchestration",
        "capability.list",
        serde_json::json!({"primal": "neuralspring"}),
    ) {
        Ok(result) => {
            let has_count = result.get("count").is_some();
            v.check_bool(
                "schema:live:capability_list_has_count",
                has_count,
                &format!(
                    "capability.list count={}",
                    result
                        .get("count")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                ),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_bool(
                "schema:live:capability_list_has_count",
                true,
                "SKIP — biomeOS not available for capability.list probe",
            );
        }
        Err(e) => {
            v.check_bool(
                "schema:live:capability_list_has_count",
                false,
                &format!("capability.list failed: {e}"),
            );
        }
    }
}

fn summarize_keys(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => {
            let keys: Vec<&String> = map.keys().take(5).collect();
            format!(
                "keys: [{}]",
                keys.iter()
                    .map(|k| k.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        serde_json::Value::Array(arr) => format!("array[{}]", arr.len()),
        other => format!("{other}"),
    }
}
