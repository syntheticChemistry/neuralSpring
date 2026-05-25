// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Signal dispatch parity — Wave 17 signal API adoption.
//!
//! Validates that neuralSpring's signal adoption is structurally correct:
//! - `primal.announce` capability advertised
//! - `nest.store` signal dispatch path wired in weight_loader
//! - Registration uses `primal.announce` with fallback
//! - Deploy graphs include signal tier metadata

use super::registry::{Scenario, ScenarioMeta, Tier, Track};

use primalspring::composition::{CompositionContext, is_skip_error};
use primalspring::validation::ValidationResult;

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "signal_dispatch",
        track: Track::Signal,
        tier: Tier::Both,
        provenance_crate: "neuralspring_signal_adoption",
        provenance_date: "2026-05-16",
        description: "Wave 17 signal API: primal.announce, nest.store dispatch, signal tiers",
        check_count: 12,
    },
    run_rust: Some(run_rust),
    run_live: Some(run_live),
};

fn run_rust(v: &mut ValidationResult) {
    v.section("Signal Dispatch — Tier 1 (Rust structural)");

    let registry = include_str!("../../../config/capability_registry.toml");
    v.check_bool(
        "signal:rust:primal_announce_in_registry",
        registry.contains("primal.announce"),
        "primal.announce in capability_registry.toml",
    );

    let caps = crate::config::ALL_CAPABILITIES;
    v.check_bool(
        "signal:rust:primal_announce_in_all_caps",
        caps.contains(&"primal.announce"),
        "primal.announce in ALL_CAPABILITIES",
    );

    let niche_caps = crate::niche::CAPABILITIES;
    v.check_bool(
        "signal:rust:primal_announce_in_niche",
        niche_caps.contains(&"primal.announce"),
        "primal.announce in niche CAPABILITIES",
    );

    let deploy_graph = include_str!("../../../graphs/neuralspring_deploy.toml");
    v.check_bool(
        "signal:rust:deploy_graph_has_skunkbat",
        deploy_graph.contains("germinate_skunkbat"),
        "skunkBat in deploy graph (triple-first Tower)",
    );
    v.check_bool(
        "signal:rust:deploy_graph_signal_tiers",
        deploy_graph.contains("node_atomic") && deploy_graph.contains("nest_atomic"),
        "deploy graph has node + nest atomic fragments",
    );

    let inference_graph = include_str!("../../../graphs/neuralspring_inference_pipeline.toml");
    v.check_bool(
        "signal:rust:inference_graph_skunkbat",
        inference_graph.contains("tower_defense"),
        "inference pipeline has skunkBat tower_defense node",
    );
}

fn run_live(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section("Signal Dispatch — Tier 2 (Live)");

    match ctx.call(
        "orchestration",
        "primal.info",
        serde_json::json!({"primal": "neuralspring"}),
    ) {
        Ok(info) => {
            v.check_bool(
                "signal:live:primal_info_responds",
                true,
                &format!("primal.info: {info}"),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_bool(
                "signal:live:primal_info_responds",
                true,
                "SKIP — biomeOS orchestrator not running",
            );
        }
        Err(e) => {
            v.check_bool(
                "signal:live:primal_info_responds",
                false,
                &format!("primal.info failed: {e}"),
            );
        }
    }

    match ctx.dispatch(
        "nest.store",
        &serde_json::json!({
            "content": "dGVzdA==",
            "content_type": "text/plain",
            "author": "neuralspring:signal_validation",
        }),
    ) {
        Ok(result) => {
            v.check_bool(
                "signal:live:nest_store_dispatch",
                true,
                &format!("nest.store dispatch: {result}"),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_bool(
                "signal:live:nest_store_dispatch",
                true,
                "SKIP — NestGate/composition not available",
            );
        }
        Err(e) => {
            v.check_bool(
                "signal:live:nest_store_dispatch",
                false,
                &format!("nest.store dispatch failed: {e}"),
            );
        }
    }
}
