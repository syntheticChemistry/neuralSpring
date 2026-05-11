// SPDX-License-Identifier: AGPL-3.0-or-later

//! JSON-RPC method handlers for neuralSpring primal capabilities.

use super::discovery::{discover_data_primal_and_forward, forward_to_primal};
use super::rpc::{self, error_code, JsonRpcResponse};
use super::{PRIMAL_NAME, PrimalState};

use neural_spring::config::ALL_CAPABILITIES;
use neural_spring::niche;

/// Kubernetes-style liveness probe: is the process alive and able to
/// handle requests?  Returns `{"status": "alive"}` per Semantic Method
/// Naming Standard v2.1.
pub fn handle_liveness(id: serde_json::Value) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "status": "alive",
            "primal": PRIMAL_NAME,
        }),
    )
}

/// Kubernetes-style readiness probe: is the primal fully initialized
/// and ready to serve science requests?  Checks that the GPU dispatcher
/// is constructed (it initializes asynchronously at startup).
pub fn handle_readiness(id: serde_json::Value, state: &PrimalState) -> JsonRpcResponse {
    let gpu_ready = state.dispatcher.has_gpu() || {
        // CPU-only mode is also "ready" — GPU is optional
        true
    };
    let uptime = state.start_time.elapsed().as_secs();
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "ready": gpu_ready,
            "primal": PRIMAL_NAME,
            "version": env!("CARGO_PKG_VERSION"),
            "subsystems": {
                "dispatcher": true,
                "gpu": state.dispatcher.has_gpu(),
                "backend": format!("{}", state.dispatcher.backend()),
            },
            "uptime_seconds": uptime,
        }),
    )
}

/// Combined health check (`DEPLOYMENT_VALIDATION_STANDARD` triad).
/// Returns liveness + readiness in a single response for benchScale and
/// plasmidBin smoke tests.
pub fn handle_health_check(id: serde_json::Value, state: &PrimalState) -> JsonRpcResponse {
    let uptime = state.start_time.elapsed().as_secs();
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "status": "ok",
            "alive": true,
            "ready": true,
            "primal": PRIMAL_NAME,
            "version": env!("CARGO_PKG_VERSION"),
            "subsystems": {
                "dispatcher": true,
                "gpu": state.dispatcher.has_gpu(),
                "backend": format!("{}", state.dispatcher.backend()),
            },
            "uptime_seconds": uptime,
        }),
    )
}

/// Primal identity for T4 discovery (`ECOSYSTEM_COMPLIANCE_MATRIX`).
pub fn handle_identity_get(id: serde_json::Value) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": PRIMAL_NAME,
            "niche": niche::NICHE_NAME,
            "version": env!("CARGO_PKG_VERSION"),
            "domain": neural_spring::config::PRIMAL_DOMAIN,
            "license": "AGPL-3.0-or-later",
            "capabilities": ALL_CAPABILITIES,
        }),
    )
}

/// MCP tool listing — hotSpring composition pattern (`mcp.tools.list` on the
/// primal surface).  Returns each capability as a discoverable tool with its
/// domain parsed from the `domain.verb` naming convention.
pub fn handle_mcp_tools_list(id: serde_json::Value) -> JsonRpcResponse {
    let tools: Vec<serde_json::Value> = ALL_CAPABILITIES
        .iter()
        .map(|cap| {
            let domain = cap.split('.').next().unwrap_or("unknown");
            serde_json::json!({
                "name": cap,
                "domain": domain,
            })
        })
        .collect();
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": PRIMAL_NAME,
            "tools": tools,
            "count": tools.len(),
        }),
    )
}

pub fn handle_capability_list(id: serde_json::Value) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": PRIMAL_NAME,
            "capabilities": ALL_CAPABILITIES,
        }),
    )
}

pub fn handle_cross_spring_provenance(id: serde_json::Value) -> JsonRpcResponse {
    use barracuda::shaders::provenance;

    let shaders = provenance::cross_spring_shaders();
    let matrix = provenance::cross_spring_matrix();

    let shader_entries: Vec<serde_json::Value> = shaders
        .iter()
        .map(|s| {
            serde_json::json!({
                "path": s.path,
                "origin": format!("{}", s.origin),
                "consumers": s.consumers.iter().map(|c| format!("{c}")).collect::<Vec<_>>(),
                "category": format!("{}", s.category),
                "evolution_note": s.evolution_note,
                "created": s.created,
                "absorbed": s.absorbed,
            })
        })
        .collect();

    let matrix_entries: Vec<serde_json::Value> = matrix
        .iter()
        .map(|((from, to), count)| {
            serde_json::json!({
                "from": format!("{from}"),
                "to": format!("{to}"),
                "shared_shaders": count,
            })
        })
        .collect();

    let report = provenance::evolution_report();

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "total_shaders": shaders.len(),
            "cross_spring_edges": matrix.len(),
            "shaders": shader_entries,
            "dependency_matrix": matrix_entries,
            "evolution_report": report,
        }),
    )
}

pub fn handle_cross_spring_benchmark(
    id: serde_json::Value,
    state: &PrimalState,
) -> JsonRpcResponse {
    use std::time::Instant;

    let data: Vec<f64> = (0..1024).map(|i| f64::from(i) * 0.001).collect();
    let mat: Vec<f64> = (0..16 * 16).map(|i| f64::from(i) * 0.01).collect();

    let t0 = Instant::now();
    let var = state.dispatcher.variance(&data);
    let variance_us = t0.elapsed().as_micros();

    let t0 = Instant::now();
    let mean = state.dispatcher.mean(&data);
    let mean_us = t0.elapsed().as_micros();

    let t0 = Instant::now();
    let sm = state.dispatcher.softmax(&data);
    let softmax_us = t0.elapsed().as_micros();

    let t0 = Instant::now();
    let gelu = state.dispatcher.gelu(&data);
    let gelu_us = t0.elapsed().as_micros();

    let t0 = Instant::now();
    let (evals, _) = state.dispatcher.eigh(&mat, 16);
    let eigh_us = t0.elapsed().as_micros();

    let t0 = Instant::now();
    let shannon = state.dispatcher.shannon_entropy(&data[..256]);
    let shannon_us = t0.elapsed().as_micros();

    let t0 = Instant::now();
    let pearson = state
        .dispatcher
        .pearson_correlation(&data[..512], &data[512..]);
    let pearson_us = t0.elapsed().as_micros();

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "backend": format!("{}", state.dispatcher.backend()),
            "shared_memory_f64_safe": state.dispatcher.shared_memory_f64_safe(),
            "benchmarks": {
                "variance_1024": {"result": var, "us": variance_us, "origin": "stats.variance → dispatch"},
                "mean_1024": {"result": mean, "us": mean_us, "origin": "stats.mean → dispatch"},
                "softmax_1024": {"len": sm.len(), "us": softmax_us, "origin": "nn.softmax → dispatch"},
                "gelu_1024": {"len": gelu.len(), "us": gelu_us, "origin": "nn.gelu → dispatch"},
                "eigh_16x16": {"n_eigenvalues": evals.len(), "us": eigh_us, "origin": "linalg.eigh → dispatch"},
                "shannon_256": {"result": shannon, "us": shannon_us, "origin": "stats.shannon → dispatch"},
                "pearson_512": {"result": pearson, "us": pearson_us, "origin": "stats.pearson → dispatch"},
            },
            "provenance": {
                "total_tracked_shaders": barracuda::shaders::provenance::cross_spring_shaders().len(),
                "cross_spring_edges": barracuda::shaders::provenance::cross_spring_matrix().len(),
            }
        }),
    )
}

/// biomeOS provenance trio RPC surface (begin / record / complete / status).
/// Acknowledges the call on this niche; full DAG lifecycle is composed via biomeOS graphs.
pub fn handle_provenance(
    id: serde_json::Value,
    method: &str,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": PRIMAL_NAME,
            "niche": niche::NICHE_NAME,
            "method": method,
            "params": params,
        }),
    )
}

/// Advertises this niche's capability surface for cross-primal discovery.
pub fn handle_primal_discover(id: serde_json::Value) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": PRIMAL_NAME,
            "niche": niche::NICHE_NAME,
            "capabilities": niche::CAPABILITIES,
        }),
    )
}

/// Discover Squirrel's socket and forward an inference request.
///
/// Returns `None` if Squirrel is not running or the call fails, allowing
/// the caller to return a `SERVICE_UNAVAILABLE` error.
fn try_squirrel_route(method: &str, params: &serde_json::Value) -> Option<serde_json::Value> {
    use neural_spring::primal_names;
    use neural_spring::validation::composition;
    use std::time::Duration;

    let socket = match composition::discover_primal_socket(primal_names::SQUIRREL) {
        composition::DiscoveryResult::Found(path) => path,
        composition::DiscoveryResult::NotFound { .. } => return None,
    };
    composition::json_rpc_call(&socket, method, params, Duration::from_secs(30)).ok()
}

/// Inference completion — routes through Squirrel when available.
pub fn handle_inference_complete(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    if let Some(result) = try_squirrel_route("inference.complete", params) {
        return JsonRpcResponse::success(id, result);
    }

    JsonRpcResponse::error(
        id,
        error_code::SERVICE_UNAVAILABLE,
        "inference.complete: Squirrel primal not discovered — \
         no inference provider available"
            .to_string(),
    )
}

/// Inference embedding — routes through Squirrel when available.
pub fn handle_inference_embed(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    if let Some(result) = try_squirrel_route("inference.embed", params) {
        return JsonRpcResponse::success(id, result);
    }

    JsonRpcResponse::error(
        id,
        error_code::SERVICE_UNAVAILABLE,
        "inference.embed: Squirrel primal not discovered — \
         no embedding provider available"
            .to_string(),
    )
}

/// List available inference models — routes through Squirrel when available.
pub fn handle_inference_models(id: serde_json::Value) -> JsonRpcResponse {
    if let Some(result) = try_squirrel_route("inference.models", &serde_json::json!({})) {
        return JsonRpcResponse::success(id, result);
    }

    JsonRpcResponse::error(
        id,
        error_code::SERVICE_UNAVAILABLE,
        "inference.models: Squirrel primal not discovered — \
         no inference provider available"
            .to_string(),
    )
}

/// Node Atomic compute offload hook: reports dispatcher readiness and echoes request params.
pub fn handle_compute_offload(
    id: serde_json::Value,
    params: &serde_json::Value,
    state: &PrimalState,
) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": PRIMAL_NAME,
            "niche": niche::NICHE_NAME,
            "params": params,
            "gpu_available": state.dispatcher.has_gpu(),
            "backend": format!("{}", state.dispatcher.backend()),
        }),
    )
}

pub fn handle_precision_routing(id: serde_json::Value, state: &PrimalState) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "fp64_strategy": format!("{:?}", state.dispatcher.fp64_strategy()),
            "precision_routing": format!("{:?}", state.dispatcher.precision_routing()),
            "shared_memory_f64_safe": state.dispatcher.shared_memory_f64_safe(),
            "bandwidth_tier": format!("{:?}", state.dispatcher.bandwidth_tier()),
            "needs_pow_workaround": state.dispatcher.needs_pow_workaround(),
            "gpu_available": state.dispatcher.has_gpu(),
            "adapter": state.dispatcher.adapter_name(),
        }),
    )
}

pub async fn handle_forward(id: serde_json::Value, params: &serde_json::Value) -> JsonRpcResponse {
    let primal = match params.get("primal").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => {
            return JsonRpcResponse::error(
                id,
                error_code::INVALID_PARAMS,
                "Missing 'primal' parameter".to_string(),
            );
        }
    };
    let method = match params.get("method").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => {
            return JsonRpcResponse::error(
                id,
                error_code::INVALID_PARAMS,
                "Missing 'method' parameter".to_string(),
            );
        }
    };
    let inner_params = params
        .get("params")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    match forward_to_primal(&primal, &method, &inner_params).await {
        Ok(resp) => JsonRpcResponse::success(id, resp),
        Err(e) => JsonRpcResponse::error(
            id,
            error_code::INTERNAL_ERROR,
            format!("Forward failed: {e}"),
        ),
    }
}

pub async fn dispatch_async(request: &rpc::JsonRpcRequest) -> JsonRpcResponse {
    let id = request.id.clone();
    let params = &request.params;

    match request.method.as_str() {
        "primal.forward" => handle_forward(id, params).await,
        method if method.starts_with("data.") => {
            match discover_data_primal_and_forward(method, params).await {
                Ok(resp) => JsonRpcResponse::success(id, resp),
                Err(e) => JsonRpcResponse::error(
                    id,
                    error_code::INTERNAL_ERROR,
                    format!("data.* forward failed: {e}"),
                ),
            }
        }
        _ => JsonRpcResponse::error(
            id,
            error_code::METHOD_NOT_FOUND,
            format!("Method not found: {}", request.method),
        ),
    }
}
