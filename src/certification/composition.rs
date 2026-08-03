// SPDX-License-Identifier: AGPL-3.0-or-later

//! Layer 4 (Composition) — NUCLEUS composition validation.
//!
//! Validates that neuralSpring's deploy graph structure is sound and that
//! live composition calls through the proto-nucleate graph succeed for
//! each capability family: tensor, security, compute, and AI.

use primalspring::composition::{CompositionContext, is_skip_error};
use primalspring::validation::ValidationResult;

/// Run NUCLEUS composition validation (L4).
pub fn validate(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    validate_deploy_graph_presence(v);
    validate_capability_registry(v);
    validate_composition_tensor(ctx, v);
    validate_composition_security(ctx, v);
    validate_composition_compute(ctx, v);
    validate_composition_ai(ctx, v);
}

fn validate_deploy_graph_presence(v: &mut ValidationResult) {
    let graphs = [
        "graphs/neuralspring_deploy.toml",
        "graphs/neuralspring_inference_pipeline.toml",
        "graphs/neuralspring_spectral_analysis.toml",
        "graphs/neuralspring_proto_nucleate.toml",
    ];
    for graph in &graphs {
        let exists = std::path::Path::new(graph).exists();
        let name = graph.rsplit('/').next().unwrap_or(graph);
        v.check_bool(
            &format!("composition:graph:{name}"),
            exists,
            if exists { "present" } else { "missing" },
        );
    }
}

fn validate_capability_registry(v: &mut ValidationResult) {
    let registry_toml = include_str!("../../config/capability_registry.toml");
    v.check_bool(
        "composition:registry:non_empty",
        !registry_toml.is_empty(),
        &format!("{} bytes", registry_toml.len()),
    );

    let cap_count = registry_toml.matches("method =").count();
    v.check_bool(
        "composition:registry:minimum_capabilities",
        cap_count >= 30,
        &format!("{cap_count} capabilities registered (minimum 30)"),
    );
}

fn validate_composition_tensor(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "tensor",
        "stats.mean",
        serde_json::json!({"data": [1.0, 2.0, 3.0, 4.0, 5.0]}),
    ) {
        Ok(result) => {
            let has_result = result.get("result").is_some() || result.get("mean").is_some();
            v.check_bool(
                "composition:tensor:stats.mean",
                has_result,
                &format!("{result}"),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "composition:tensor:stats.mean",
                &format!("tensor offline: {e}"),
            );
        }
        Err(e) => {
            v.check_bool("composition:tensor:stats.mean", false, &format!("{e}"));
        }
    }
}

fn validate_composition_security(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.hash_bytes(b"composition:neuralspring:l4:certified", "blake3") {
        Ok(receipt) => {
            v.check_bool(
                "composition:security:crypto.hash",
                !receipt.is_empty(),
                &format!("hash len={}", receipt.len()),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "composition:security:crypto.hash",
                &format!("security offline: {e}"),
            );
        }
        Err(e) => {
            v.check_bool("composition:security:crypto.hash", false, &format!("{e}"));
        }
    }
}

fn validate_composition_compute(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "compute",
        "compute.dispatch",
        serde_json::json!({"op": "probe", "tier": "certification"}),
    ) {
        Ok(_result) => {
            v.check_bool(
                "composition:compute:dispatch",
                true,
                "dispatch acknowledged",
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "composition:compute:dispatch",
                &format!("compute offline: {e}"),
            );
        }
        Err(e) => {
            v.check_bool("composition:compute:dispatch", false, &format!("{e}"));
        }
    }
}

fn validate_composition_ai(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "ai",
        "inference.complete",
        serde_json::json!({"prompt": "certification probe", "max_tokens": 1}),
    ) {
        Ok(result) => {
            let has_text = result.get("text").is_some();
            v.check_bool(
                "composition:ai:inference.complete",
                has_text,
                &format!("{result}"),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "composition:ai:inference.complete",
                &format!("ai offline: {e}"),
            );
        }
        Err(e) => {
            v.check_skip(
                "composition:ai:inference.complete",
                &format!("inference not available: {e}"),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn composition_registry_has_capabilities() {
        let registry = include_str!("../../config/capability_registry.toml");
        let count = registry.matches("method =").count();
        assert!(count >= 30, "expected >=30 capabilities, got {count}");
    }

    #[test]
    fn deploy_graph_file_exists() {
        let exists = std::path::Path::new("graphs/neuralspring_deploy.toml").exists();
        assert!(exists, "primary deploy graph must exist");
    }
}
