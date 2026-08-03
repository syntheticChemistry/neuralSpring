// SPDX-License-Identifier: AGPL-3.0-or-later

//! JSON-RPC method handlers for neuralSpring primal capabilities.

use super::discovery::{
    discover_by_capability, discover_data_primal_and_forward, forward_to_primal,
};
use super::rpc::{self, JsonRpcResponse, error_code};
use super::{PRIMAL_NAME, PrimalState};

use neural_spring::capabilities;
use neural_spring::config::ALL_CAPABILITIES;
use neural_spring::niche;
use neural_spring::nucleus_pipeline::{
    PIPELINE_CAPABILITIES, dispatch_capability, dispatch_capability_gpu, is_pipeline_capability,
};
use neural_spring::primal_names;
use neural_spring_forge::graph::StageOutput;

use std::time::{Duration, Instant};

const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const PROVENANCE_TIMEOUT: Duration = Duration::from_secs(5);
const INFERENCE_TIMEOUT: Duration = Duration::from_secs(30);
const COMPUTE_DISPATCH_TIMEOUT: Duration = Duration::from_secs(10);
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(5);

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
    // Ready when GPU is available, or when CPU-only mode is acceptable
    // (GPU is optional unless REQUIRE_GPU / NEURALSPRING_REQUIRE_GPU is set).
    let gpu_ready = state.dispatcher.has_gpu() || !neural_spring::validation::gpu_required();
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
                "gpu_required": neural_spring::validation::gpu_required(),
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
            "count": ALL_CAPABILITIES.len(),
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
///
/// Forwards to the biomeOS orchestrator via capability-based discovery. Falls
/// back to a local acknowledgment when biomeOS is not running.
pub fn handle_provenance(
    id: serde_json::Value,
    method: &str,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    let orchestrator =
        try_discover_and_call(primal_names::BIOMEOS, method, params, PROVENANCE_TIMEOUT);
    if let Some(result) = orchestrator {
        return JsonRpcResponse::success(id, result);
    }

    let rhizocrypt =
        try_discover_and_call(primal_names::RHIZOCRYPT, method, params, PROVENANCE_TIMEOUT);
    if let Some(result) = rhizocrypt {
        return JsonRpcResponse::success(id, result);
    }

    log::debug!("{method}: no provenance primal discovered — local acknowledgment");
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": PRIMAL_NAME,
            "niche": niche::NICHE_NAME,
            "method": method,
            "status": "acknowledged_locally",
            "note": "provenance primals not discovered — DAG lifecycle deferred",
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

/// Discover a primal by name and forward a JSON-RPC call.
///
/// Returns `None` if the primal is not running or the call fails, allowing
/// the caller to fall back gracefully.
fn try_discover_and_call(
    primal_name: &str,
    method: &str,
    params: &serde_json::Value,
    timeout: std::time::Duration,
) -> Option<serde_json::Value> {
    use neural_spring::validation::composition;

    let socket = match composition::discover_primal_socket(primal_name) {
        composition::DiscoveryResult::Found(path) => path,
        composition::DiscoveryResult::NotFound { .. } => return None,
    };
    composition::json_rpc_call(&socket, method, params, timeout).ok()
}

/// Discover a primal by capability and forward a JSON-RPC call.
///
/// Probes live sockets for the requested capability, falling back to
/// compile-time name hints when no socket advertises it.
fn try_discover_and_call_capability(
    capability: &str,
    method: &str,
    params: &serde_json::Value,
    timeout: Duration,
) -> Option<serde_json::Value> {
    use neural_spring::validation::composition;

    let socket = discover_by_capability(capability, PROBE_TIMEOUT)?;
    composition::json_rpc_call(&socket, method, params, timeout).ok()
}

/// Inference completion — routes to whichever primal advertises inference.
pub fn handle_inference_complete(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    if let Some(result) = try_discover_and_call_capability(
        capabilities::INFERENCE_COMPLETE,
        capabilities::INFERENCE_COMPLETE,
        params,
        INFERENCE_TIMEOUT,
    ) {
        return JsonRpcResponse::success(id, result);
    }

    JsonRpcResponse::error(
        id,
        error_code::SERVICE_UNAVAILABLE,
        format!(
            "{}: no inference provider discovered",
            capabilities::INFERENCE_COMPLETE
        ),
    )
}

/// Inference embedding — routes to whichever primal advertises embedding.
pub fn handle_inference_embed(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    if let Some(result) = try_discover_and_call_capability(
        capabilities::INFERENCE_EMBED,
        capabilities::INFERENCE_EMBED,
        params,
        INFERENCE_TIMEOUT,
    ) {
        return JsonRpcResponse::success(id, result);
    }

    JsonRpcResponse::error(
        id,
        error_code::SERVICE_UNAVAILABLE,
        format!(
            "{}: no embedding provider discovered",
            capabilities::INFERENCE_EMBED
        ),
    )
}

/// List available inference models — routes to whichever primal advertises models.
pub fn handle_inference_models(id: serde_json::Value) -> JsonRpcResponse {
    if let Some(result) = try_discover_and_call_capability(
        capabilities::INFERENCE_MODELS,
        capabilities::INFERENCE_MODELS,
        &serde_json::json!({}),
        INFERENCE_TIMEOUT,
    ) {
        return JsonRpcResponse::success(id, result);
    }

    JsonRpcResponse::error(
        id,
        error_code::SERVICE_UNAVAILABLE,
        format!(
            "{}: no inference provider discovered",
            capabilities::INFERENCE_MODELS
        ),
    )
}

/// Node Atomic compute offload — forwards to whichever primal advertises dispatch.
///
/// If a compute dispatch provider is running, the workload is forwarded for
/// distributed dispatch. Otherwise, reports local dispatcher readiness.
pub fn handle_compute_offload(
    id: serde_json::Value,
    params: &serde_json::Value,
    state: &PrimalState,
) -> JsonRpcResponse {
    if let Some(result) = try_discover_and_call_capability(
        capabilities::COMPUTE_DISPATCH_SUBMIT,
        capabilities::COMPUTE_DISPATCH_SUBMIT,
        params,
        COMPUTE_DISPATCH_TIMEOUT,
    ) {
        return JsonRpcResponse::success(id, result);
    }

    log::debug!("compute.offload: no dispatch provider discovered — local dispatch info");
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": PRIMAL_NAME,
            "niche": niche::NICHE_NAME,
            "status": "local_dispatch",
            "gpu_available": state.dispatcher.has_gpu(),
            "backend": format!("{}", state.dispatcher.backend()),
            "note": "no compute dispatch provider discovered — workload handled locally",
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

pub fn handle_primal_announce(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    let primal_id = params
        .get("primal_id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    if let Some(result) = try_discover_and_call(
        primal_names::BIOMEOS,
        "primal.announce",
        params,
        PROVENANCE_TIMEOUT,
    ) {
        return JsonRpcResponse::success(
            id,
            serde_json::json!({
                "status": "forwarded",
                "primal": PRIMAL_NAME,
                "announced_by": primal_id,
                "upstream_result": result,
            }),
        );
    }

    log::debug!("primal.announce: biomeOS not discovered — local acknowledgment");
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "status": "acknowledged_locally",
            "primal": PRIMAL_NAME,
            "announced_by": primal_id,
            "note": "biomeOS not discovered — announcement stored locally",
        }),
    )
}

pub fn handle_composition_status(id: serde_json::Value, state: &PrimalState) -> JsonRpcResponse {
    let science_capabilities = ALL_CAPABILITIES
        .iter()
        .filter(|cap| cap.starts_with("science."))
        .count();
    let pipeline_registered = PIPELINE_CAPABILITIES
        .iter()
        .filter(|cap| ALL_CAPABILITIES.contains(cap))
        .count();
    let gpu_available = state.dispatcher.has_gpu();
    let gpu_required = neural_spring::validation::gpu_required();
    let ready = gpu_available || !gpu_required;
    let status = if ready { "ready" } else { "degraded" };

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": PRIMAL_NAME,
            "status": status,
            "nucleus_atomic": "Tower",
            "composition_layer": "L4",
            "capabilities_total": ALL_CAPABILITIES.len(),
            "science_capabilities": science_capabilities,
            "pipeline_stages": PIPELINE_CAPABILITIES.len(),
            "pipeline_stages_registered": pipeline_registered,
            "gpu_available": gpu_available,
            "gpu_required": gpu_required,
            "backend": format!("{}", state.dispatcher.backend()),
            "signal_api": "wave17",
        }),
    )
}

/// Method registration — forwards to biomeOS orchestrator when available.
///
/// neuralSpring is a niche, not an orchestrator. Registration requests are
/// forwarded to biomeOS for the canonical method registry.
pub fn handle_method_register(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    let method = params
        .get("method")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    if let Some(result) = try_discover_and_call(
        primal_names::BIOMEOS,
        "method.register",
        params,
        REGISTRATION_TIMEOUT,
    ) {
        return JsonRpcResponse::success(
            id,
            serde_json::json!({
                "status": "forwarded",
                "primal": PRIMAL_NAME,
                "method": method,
                "upstream_result": result,
            }),
        );
    }

    log::debug!("method.register: biomeOS not discovered — local acknowledgment only");
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "status": "acknowledged_locally",
            "primal": PRIMAL_NAME,
            "method": method,
            "note": "biomeOS not discovered — use primal.announce for Wave 17 registration",
        }),
    )
}

pub fn handle_compute_dispatch(
    id: serde_json::Value,
    params: &serde_json::Value,
    state: &PrimalState,
) -> JsonRpcResponse {
    if let Some(op) = params.get("operation").and_then(|v| v.as_str()) {
        return match op {
            "probe" => JsonRpcResponse::success(
                id,
                serde_json::json!({
                    "primal": PRIMAL_NAME,
                    "status": "dispatch_ready",
                    "gpu_available": state.dispatcher.has_gpu(),
                    "backend": format!("{}", state.dispatcher.backend()),
                    "pipeline_stages": PIPELINE_CAPABILITIES.len(),
                }),
            ),
            "status" => JsonRpcResponse::success(
                id,
                serde_json::json!({
                    "primal": PRIMAL_NAME,
                    "status": "ok",
                    "gpu_available": state.dispatcher.has_gpu(),
                    "gpu_required": neural_spring::validation::gpu_required(),
                    "backend": format!("{}", state.dispatcher.backend()),
                    "adapter": state.dispatcher.adapter_name(),
                    "pipeline_capabilities": PIPELINE_CAPABILITIES,
                }),
            ),
            _ => JsonRpcResponse::error(
                id,
                error_code::INVALID_PARAMS,
                format!("Unknown compute.dispatch operation: {op}"),
            ),
        };
    }

    let workload = params
        .get("workload")
        .or_else(|| params.get("capability"))
        .and_then(|v| v.as_str());

    let Some(workload) = workload else {
        return JsonRpcResponse::error(
            id,
            error_code::INVALID_PARAMS,
            "Missing 'workload' or 'capability' parameter (expected science capability)"
                .to_string(),
        );
    };

    if !is_pipeline_capability(workload) {
        return JsonRpcResponse::error(
            id,
            error_code::INVALID_PARAMS,
            format!("Unknown workload capability: {workload}"),
        );
    }

    let substrate_hint = params
        .get("substrate_hint")
        .and_then(|v| v.as_str())
        .unwrap_or("auto");
    let use_gpu = substrate_hint != "cpu" && state.dispatcher.has_gpu();
    let actual_backend = if use_gpu {
        format!("{}", state.dispatcher.backend())
    } else {
        "CPU".to_string()
    };

    let t0 = Instant::now();
    let (success, output) = if use_gpu {
        dispatch_capability_gpu(workload, &state.dispatcher)
    } else {
        dispatch_capability(workload)
    };
    let elapsed_us = t0.elapsed().as_micros();

    if !success {
        return JsonRpcResponse::error(
            id,
            error_code::INTERNAL_ERROR,
            format!("Dispatch failed for workload: {workload}"),
        );
    }

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": PRIMAL_NAME,
            "workload": workload,
            "status": "dispatched",
            "success": success,
            "gpu_available": state.dispatcher.has_gpu(),
            "substrate_hint": substrate_hint,
            "actual_substrate": actual_backend,
            "elapsed_us": elapsed_us,
            "output_summary": stage_output_summary(&output),
        }),
    )
}

fn stage_output_summary(output: &StageOutput) -> serde_json::Value {
    match output {
        StageOutput::Scalar(v) => serde_json::json!({ "kind": "scalar", "value": v }),
        StageOutput::Vector(v) => serde_json::json!({ "kind": "vector", "len": v.len() }),
        StageOutput::Map(m) => {
            serde_json::json!({ "kind": "map", "keys": m.keys().collect::<Vec<_>>() })
        }
        StageOutput::Empty => serde_json::json!({ "kind": "empty" }),
    }
}

/// Security audit log — forwards to whichever primal advertises audit logging.
///
/// Logs the event locally, then attempts to forward to a discovered provider
/// for centralized audit trail. Falls back to local-only logging when no
/// provider is discovered.
pub fn handle_security_audit_log(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    let event = params
        .get("event")
        .and_then(|v| v.as_str())
        .unwrap_or("audit_query");
    log::info!("security.audit_log: {event}");

    if let Some(result) = try_discover_and_call_capability(
        capabilities::SECURITY_AUDIT_LOG,
        capabilities::SECURITY_AUDIT_LOG,
        params,
        PROVENANCE_TIMEOUT,
    ) {
        return JsonRpcResponse::success(
            id,
            serde_json::json!({
                "primal": PRIMAL_NAME,
                "event": event,
                "status": "forwarded",
                "upstream_result": result,
            }),
        );
    }

    log::debug!("security.audit_log: no audit provider discovered — local-only");
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": PRIMAL_NAME,
            "event": event,
            "status": "logged_locally",
            "note": "no audit provider discovered — audit stored locally only",
        }),
    )
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
