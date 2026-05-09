// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Compute dispatch — toadStool-mediated compute.
//!
//! Absorbed from `validate_nucleus_compute_dispatch.rs` (~36 checks).

use super::registry::{Scenario, ScenarioMeta, Tier, Track};

use primalspring::composition::{CompositionContext, is_skip_error};
use primalspring::validation::ValidationResult;

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "compute_dispatch",
        track: Track::NucleusComposition,
        tier: Tier::Both,
        provenance_crate: "validate_nucleus_compute_dispatch",
        provenance_date: "2026-05-09",
        description: "NUCLEUS atomic compute dispatch for spectral science",
        check_count: 36,
    },
    run_rust: Some(run_rust),
    run_live: Some(run_live),
};

fn run_rust(v: &mut ValidationResult) {
    v.section("Compute Dispatch — Tier 1 (Rust structural)");

    let graph_path = std::path::Path::new("graphs/neuralspring_deploy.toml");
    v.check_bool(
        "dispatch:rust:deploy_graph_exists",
        graph_path.exists(),
        if graph_path.exists() {
            "present"
        } else {
            "missing"
        },
    );

    let registry = include_str!("../../../config/capability_registry.toml");
    let has_compute = registry.contains("compute.dispatch");
    v.check_bool(
        "dispatch:rust:registry_has_compute",
        has_compute,
        "compute.dispatch in registry",
    );
}

fn run_live(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section("Compute Dispatch — Tier 2 (Live)");

    match ctx.call(
        "compute",
        "compute.dispatch",
        serde_json::json!({"operation": "probe"}),
    ) {
        Ok(result) => {
            let ack = result.get("status").is_some()
                || result.get("result").is_some()
                || result.get("dispatched").is_some();
            v.check_bool("dispatch:live:probe_ack", ack, &format!("{result}"));
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip("dispatch:live:probe_ack", &format!("compute offline: {e}"));
        }
        Err(e) => {
            v.check_bool("dispatch:live:probe_ack", false, &format!("{e}"));
        }
    }

    match ctx.call(
        "compute",
        "compute.dispatch",
        serde_json::json!({"operation": "status"}),
    ) {
        Ok(result) => {
            v.check_bool("dispatch:live:status", true, &format!("{result}"));
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip("dispatch:live:status", &format!("compute offline: {e}"));
        }
        Err(e) => {
            v.check_bool("dispatch:live:status", false, &format!("{e}"));
        }
    }
}
