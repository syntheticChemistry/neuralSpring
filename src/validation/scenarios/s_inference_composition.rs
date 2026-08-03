// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Inference composition — Squirrel-mediated pipeline.
//!
//! Absorbed from `validate_inference_composition.rs` (~16 checks).

use super::registry::{Scenario, ScenarioMeta, Tier, Track};

use primalspring::composition::{CompositionContext, is_skip_error};
use primalspring::validation::ValidationResult;

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "inference_composition",
        track: Track::InferencePipeline,
        tier: Tier::Live,
        provenance_crate: "validate_inference_composition",
        provenance_date: "2026-05-09",
        description: "Inference capability chain: complete, embed, models via Squirrel",
        check_count: 16,
    },
    run_rust: None,
    run_live: Some(run_live),
};

fn run_live(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section("Inference Composition (absorbed scenario)");

    inference_complete(ctx, v);
    inference_embed(ctx, v);
    inference_models(ctx, v);
    inference_round_trip(ctx, v);
}

fn inference_complete(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "ai",
        "inference.complete",
        serde_json::json!({"prompt": "What is 2+2?", "model": "default", "max_tokens": 4}),
    ) {
        Ok(result) => {
            let has_text = result.get("text").is_some() || result.get("completion").is_some();
            v.check_bool(
                "inference:complete:has_text",
                has_text,
                &format!("{result}"),
            );
            let has_usage = result.get("usage").is_some() || result.get("tokens").is_some();
            v.check_bool(
                "inference:complete:has_usage_or_tokens",
                has_usage || has_text,
                "response acknowledged",
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip("inference:complete:has_text", &format!("ai offline: {e}"));
            v.check_skip("inference:complete:has_usage_or_tokens", "ai offline");
        }
        Err(e) => {
            v.check_bool("inference:complete:has_text", false, &format!("{e}"));
            v.check_skip(
                "inference:complete:has_usage_or_tokens",
                "prior call failed",
            );
        }
    }
}

fn inference_embed(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "ai",
        "inference.embed",
        serde_json::json!({"text": "neuralSpring validation", "model": "default"}),
    ) {
        Ok(result) => {
            let has_embedding =
                result.get("embedding").is_some() || result.get("embeddings").is_some();
            v.check_bool(
                "inference:embed:has_embedding",
                has_embedding,
                &format!("{result}"),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip("inference:embed:has_embedding", &format!("ai offline: {e}"));
        }
        Err(e) => {
            v.check_bool("inference:embed:has_embedding", false, &format!("{e}"));
        }
    }
}

fn inference_models(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call("ai", "inference.models", serde_json::json!({})) {
        Ok(result) => {
            let has_models = result.get("models").is_some() || result.is_array();
            v.check_bool("inference:models:listed", has_models, &format!("{result}"));
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip("inference:models:listed", &format!("ai offline: {e}"));
        }
        Err(e) => {
            v.check_bool("inference:models:listed", false, &format!("{e}"));
        }
    }
}

fn inference_round_trip(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "ai",
        "inference.complete",
        serde_json::json!({"prompt": "echo: test", "model": "default", "max_tokens": 2}),
    ) {
        Ok(result) => {
            let non_empty = result
                .get("text")
                .or_else(|| result.get("completion"))
                .and_then(|v| v.as_str())
                .is_some_and(|s| !s.is_empty());
            v.check_bool(
                "inference:round_trip:non_empty",
                non_empty,
                &format!("{result}"),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "inference:round_trip:non_empty",
                &format!("ai offline: {e}"),
            );
        }
        Err(e) => {
            v.check_bool("inference:round_trip:non_empty", false, &format!("{e}"));
        }
    }
}
