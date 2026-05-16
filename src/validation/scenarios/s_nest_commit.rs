// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Nest commit provenance — live data chain validation.
//!
//! Validates that neuralSpring can drive a full provenance chain through
//! `nest.store` (content persistence) and `nest.commit` (session finalization)
//! signal dispatch. Structural checks verify signal graph awareness;
//! live checks exercise the actual dispatch path through biomeOS.

use super::registry::{Scenario, ScenarioMeta, Tier, Track};

use primalspring::composition::{CompositionContext, is_skip_error};
use primalspring::validation::ValidationResult;

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "nest_commit_provenance",
        track: Track::Provenance,
        tier: Tier::Both,
        provenance_crate: "neuralspring_live_provenance",
        provenance_date: "2026-05-16",
        description: "Live provenance chain: nest.store + nest.commit signal dispatch",
        check_count: 10,
    },
    run_rust: Some(run_rust),
    run_live: Some(run_live),
};

fn run_rust(v: &mut ValidationResult) {
    v.section("Nest Commit Provenance — Tier 1 (Rust structural)");

    let signal_tools = include_str!("../../../../primalSpring/config/signal_tools.toml");
    v.check_bool(
        "provenance:rust:nest_store_signal_exists",
        signal_tools.contains("nest.store"),
        "nest.store signal defined in primalSpring signal_tools.toml",
    );
    v.check_bool(
        "provenance:rust:nest_commit_signal_exists",
        signal_tools.contains("nest.commit"),
        "nest.commit signal defined in primalSpring signal_tools.toml",
    );

    let caps = crate::config::ALL_CAPABILITIES;
    v.check_bool(
        "provenance:rust:has_security_audit_log",
        caps.contains(&"security.audit_log"),
        "security.audit_log in capabilities (skunkBat provenance)",
    );

    let weight_loader_src = include_str!("../../weight_loader.rs");
    v.check_bool(
        "provenance:rust:commit_session_signal_wired",
        weight_loader_src.contains("nest.commit"),
        "nest.commit dispatch wired in weight_loader",
    );
    v.check_bool(
        "provenance:rust:store_science_result_wired",
        weight_loader_src.contains("store_science_result"),
        "store_science_result provenance wrapper in weight_loader",
    );

    let deploy = include_str!("../../../graphs/neuralspring_deploy.toml");
    v.check_bool(
        "provenance:rust:deploy_has_nest_atomic",
        deploy.contains("nest_atomic"),
        "deploy graph includes nest_atomic fragment",
    );
}

fn run_live(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section("Nest Commit Provenance — Tier 2 (Live)");

    match ctx.dispatch(
        "nest.store",
        serde_json::json!({
            "content": "eyJzY2VuYXJpbyI6InByb3ZlbmFuY2VfdGVzdCJ9",
            "content_type": "application/json",
            "author": "neuralspring:provenance_validation",
            "metadata": {
                "method": "validation.nest_commit_scenario",
                "domain": "science",
                "spring": "neuralSpring",
            },
        }),
    ) {
        Ok(result) => {
            v.check_bool(
                "provenance:live:nest_store_dispatch",
                true,
                &format!("nest.store dispatch succeeded: {result}"),
            );

            let session_id = result
                .get("session_commit")
                .and_then(|s| s.as_str())
                .or_else(|| result.get("dag_event").and_then(|s| s.as_str()))
                .unwrap_or("validation-session");

            match ctx.dispatch(
                "nest.commit",
                serde_json::json!({ "session_id": session_id }),
            ) {
                Ok(commit_result) => {
                    v.check_bool(
                        "provenance:live:nest_commit_dispatch",
                        true,
                        &format!("nest.commit dispatch succeeded: {commit_result}"),
                    );
                }
                Err(e) if is_skip_error(&e) => {
                    v.check_bool(
                        "provenance:live:nest_commit_dispatch",
                        true,
                        "SKIP — nest.commit not available (pre-v3.57 biomeOS or missing primals)",
                    );
                }
                Err(e) => {
                    v.check_bool(
                        "provenance:live:nest_commit_dispatch",
                        false,
                        &format!("nest.commit dispatch failed: {e}"),
                    );
                }
            }
        }
        Err(e) if is_skip_error(&e) => {
            v.check_bool(
                "provenance:live:nest_store_dispatch",
                true,
                "SKIP — NestGate/composition not available",
            );
            v.check_bool(
                "provenance:live:nest_commit_dispatch",
                true,
                "SKIP — depends on nest.store (skipped)",
            );
        }
        Err(e) => {
            v.check_bool(
                "provenance:live:nest_store_dispatch",
                false,
                &format!("nest.store dispatch failed: {e}"),
            );
            v.check_bool(
                "provenance:live:nest_commit_dispatch",
                false,
                "SKIP — depends on nest.store (failed)",
            );
        }
    }

    match crate::weight_loader::store_science_result(
        ctx,
        "validation.provenance_check",
        &serde_json::json!({"check": "science_result_provenance", "status": "pass"}),
        "neuralspring:provenance_validation",
    ) {
        Ok(_) => {
            v.check_bool(
                "provenance:live:science_result_store",
                true,
                "store_science_result() dispatched successfully",
            );
        }
        Err(ref e) if {
            let msg = format!("{e}");
            msg.contains("SocketNotFound") || msg.contains("not available") || msg.contains("Connection refused")
        } => {
            v.check_bool(
                "provenance:live:science_result_store",
                true,
                "SKIP — composition not available for science result storage",
            );
        }
        Err(e) => {
            v.check_bool(
                "provenance:live:science_result_store",
                false,
                &format!("store_science_result failed: {e}"),
            );
        }
    }
}
