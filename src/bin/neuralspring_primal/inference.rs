// SPDX-License-Identifier: AGPL-3.0-or-later

//! Inference domain handlers — `inference.complete`, `inference.embed`,
//! `inference.models`.
//!
//! Implements the ecoPrimal vendor-agnostic inference wire standard.
//! neuralSpring exposes its native Rust models (via BarraCUDA/wgpu)
//! through these methods, allowing consumers to discover it as a
//! provider alongside Ollama or remote APIs.

use super::PrimalState;
use super::rpc::{self, JsonRpcResponse};

/// Handle `inference.complete` — text completion via native Rust models.
///
/// Currently returns a stub response indicating the method is recognized
/// but no models are loaded. Once neuralSpring's transformer engine is
/// wired to the primal binary, this will dispatch to loaded models.
pub fn handle_complete(
    id: serde_json::Value,
    params: &serde_json::Value,
    state: &PrimalState,
) -> JsonRpcResponse {
    let prompt = params
        .get("prompt")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");

    let model = params
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("neuralspring-default");

    if prompt.is_empty() && params.get("messages").is_none() {
        return JsonRpcResponse::error(
            id,
            rpc::error_code::INVALID_PARAMS,
            "inference.complete requires 'prompt' or 'messages'".into(),
        );
    }

    let has_gpu = state.dispatcher.has_gpu();
    let backend = format!("{}", state.dispatcher.backend());

    // Stub: acknowledge the request with provider metadata.
    // Real inference will be wired when model loading is integrated.
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "text": format!(
                "[neuralSpring] inference.complete stub — model '{model}' not yet loaded. \
                 GPU: {has_gpu}, backend: {backend}"
            ),
            "model": model,
            "provider": "neuralspring",
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }),
    )
}

/// Handle `inference.embed` — embedding generation via native models.
///
/// Stub: returns a zero-vector embedding until a native embedding
/// model is loaded.
pub fn handle_embed(
    id: serde_json::Value,
    params: &serde_json::Value,
    _state: &PrimalState,
) -> JsonRpcResponse {
    let dim = 384; // typical small embedding dimension

    let input_count = match params.get("input") {
        Some(serde_json::Value::String(_)) => 1,
        Some(serde_json::Value::Array(arr)) => arr.len(),
        _ => {
            return JsonRpcResponse::error(
                id,
                rpc::error_code::INVALID_PARAMS,
                "inference.embed requires 'input' (string or array of strings)".into(),
            );
        }
    };

    let zero_vec: Vec<f32> = vec![0.0; dim];
    let embeddings: Vec<&Vec<f32>> = vec![&zero_vec; input_count];

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "embeddings": embeddings,
            "model": "neuralspring-embed-stub"
        }),
    )
}

/// Handle `inference.models` — list available models.
///
/// Returns metadata about what neuralSpring can serve. Currently
/// advertises a stub entry; real model enumeration will come from
/// the model registry once transformer weights are loadable.
pub fn handle_models(id: serde_json::Value, state: &PrimalState) -> JsonRpcResponse {
    let has_gpu = state.dispatcher.has_gpu();
    let backend = format!("{}", state.dispatcher.backend());

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "models": [
                {
                    "id": "neuralspring-native",
                    "name": "neuralSpring Native (stub)",
                    "supports_completion": true,
                    "supports_embedding": true,
                    "context_length": 2048,
                    "metadata": {
                        "gpu_available": has_gpu,
                        "backend": backend,
                        "status": "stub — model loading not yet wired"
                    }
                }
            ]
        }),
    )
}
