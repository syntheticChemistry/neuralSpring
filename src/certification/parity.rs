// SPDX-License-Identifier: AGPL-3.0-or-later

//! Layer 2 (Parity) — domain science parity via IPC.
//!
//! Validates 7 `PROTO_NUCLEATE` capabilities against known Python/analytical
//! reference values:
//! - `stats.mean`, `tensor.matmul`, `tensor.create`
//! - `compute.dispatch`, `crypto.hash`
//! - `inference.complete`, `inference.embed`

use primalspring::composition::{self, CompositionContext, is_skip_error, validate_parity};
use primalspring::tolerances;
use primalspring::validation::ValidationResult;

use crate::validation::composition::PROTO_NUCLEATE_VALIDATION_CAPABILITIES;

/// Run domain-science parity validation (L2).
pub fn validate(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section(&format!(
        "Parity ({} capabilities)",
        PROTO_NUCLEATE_VALIDATION_CAPABILITIES.len()
    ));

    stats_mean(ctx, v);
    tensor_matmul(ctx, v);
    tensor_create(ctx, v);
    compute_dispatch(ctx, v);
    crypto_hash(ctx, v);
    inference_complete(ctx, v);
    inference_embed(ctx, v);
}

fn stats_mean(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    validate_parity(
        ctx,
        v,
        "stats_mean_5elem",
        "tensor",
        "stats.mean",
        serde_json::json!({"data": [1.0, 2.0, 3.0, 4.0, 5.0]}),
        "result",
        3.0,
        tolerances::IPC_ROUND_TRIP_TOL,
    );
}

fn tensor_matmul(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    let params = serde_json::json!({
        "a": [[1.0, 2.0], [3.0, 4.0]],
        "b": [[5.0, 6.0], [7.0, 8.0]],
        "rows_a": 2, "cols_a": 2, "cols_b": 2,
    });
    let expected = &[19.0, 22.0, 43.0, 50.0];
    composition::validate_parity_vec(
        ctx,
        v,
        "tensor_matmul_2x2",
        "tensor",
        "tensor.matmul",
        params,
        "data",
        expected,
        tolerances::IPC_ROUND_TRIP_TOL,
    );
}

fn tensor_create(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "tensor",
        "tensor.create",
        serde_json::json!({"shape": [2, 3], "fill": "zeros"}),
    ) {
        Ok(result) => {
            let has_shape = result.get("shape").is_some() || result.get("dimensions").is_some();
            v.check_bool(
                "tensor_create_has_shape",
                has_shape,
                &format!("response: {result}"),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "tensor_create_has_shape",
                &format!("tensor not available: {e}"),
            );
        }
        Err(e) => {
            v.check_bool(
                "tensor_create_has_shape",
                false,
                &format!("composition error: {e}"),
            );
        }
    }
}

fn compute_dispatch(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "compute",
        "compute.dispatch",
        serde_json::json!({"operation": "probe"}),
    ) {
        Ok(result) => {
            let acknowledged = result.get("status").is_some()
                || result.get("result").is_some()
                || result.get("dispatched").is_some();
            v.check_bool(
                "compute_dispatch_ack",
                acknowledged,
                &format!("response: {result}"),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "compute_dispatch_ack",
                &format!("compute not available: {e}"),
            );
        }
        Err(e) => {
            v.check_bool(
                "compute_dispatch_ack",
                false,
                &format!("composition error: {e}"),
            );
        }
    }
}

fn crypto_hash(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.hash_bytes(b"neuralspring-guidestone-parity", "blake3") {
        Ok(hash) => {
            v.check_bool(
                "crypto_hash_nonempty",
                !hash.is_empty(),
                &format!("BLAKE3 hash len={}", hash.len()),
            );
            match ctx.hash_bytes(b"neuralspring-guidestone-parity", "blake3") {
                Ok(hash2) => {
                    v.check_bool(
                        "crypto_hash_deterministic",
                        hash == hash2,
                        "identical input → identical hash",
                    );
                }
                Err(e) => {
                    v.check_bool(
                        "crypto_hash_deterministic",
                        false,
                        &format!("second hash call failed: {e}"),
                    );
                }
            }
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "crypto_hash_nonempty",
                &format!("security not available: {e}"),
            );
            v.check_skip("crypto_hash_deterministic", "security not available");
        }
        Err(e) => {
            v.check_bool("crypto_hash_nonempty", false, &format!("hash error: {e}"));
            v.check_skip("crypto_hash_deterministic", "prior hash call failed");
        }
    }
}

fn inference_complete(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "ai",
        "inference.complete",
        serde_json::json!({"prompt": "test", "model": "default", "max_tokens": 1}),
    ) {
        Ok(result) => {
            let has_text = result.get("text").is_some() || result.get("completion").is_some();
            v.check_bool(
                "inference_complete_has_text",
                has_text,
                &format!("response: {result}"),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "inference_complete_has_text",
                &format!("ai not available: {e}"),
            );
        }
        Err(e) => {
            v.check_bool(
                "inference_complete_has_text",
                false,
                &format!("composition error: {e}"),
            );
        }
    }
}

fn inference_embed(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "ai",
        "inference.embed",
        serde_json::json!({"text": "test", "model": "default"}),
    ) {
        Ok(result) => {
            let has_embedding =
                result.get("embedding").is_some() || result.get("embeddings").is_some();
            v.check_bool(
                "inference_embed_has_embedding",
                has_embedding,
                &format!("response: {result}"),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "inference_embed_has_embedding",
                &format!("ai not available: {e}"),
            );
        }
        Err(e) => {
            v.check_bool(
                "inference_embed_has_embedding",
                false,
                &format!("composition error: {e}"),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::validation::composition::PROTO_NUCLEATE_VALIDATION_CAPABILITIES;

    #[test]
    fn proto_nucleate_capabilities_nonempty() {
        assert!(
            !PROTO_NUCLEATE_VALIDATION_CAPABILITIES.is_empty(),
            "proto-nucleate must define validation capabilities"
        );
    }

    #[test]
    fn proto_nucleate_includes_core_methods() {
        let caps = PROTO_NUCLEATE_VALIDATION_CAPABILITIES;
        assert!(caps.contains(&"stats.mean"), "must include stats.mean");
        assert!(
            caps.contains(&"tensor.matmul"),
            "must include tensor.matmul"
        );
        assert!(caps.contains(&"crypto.hash"), "must include crypto.hash");
    }
}
