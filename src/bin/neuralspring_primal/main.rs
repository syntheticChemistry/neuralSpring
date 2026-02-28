// SPDX-License-Identifier: AGPL-3.0-or-later

//! neuralSpring biomeOS Primal — Tower Mode
//!
//! JSON-RPC 2.0 server exposing neuralSpring's spectral analysis AND
//! coralForge capabilities to the biomeOS ecosystem.
//!
//! ## Capability domains
//!
//! **Spectral analysis** (baseCamp):
//!   `science.ipr`, `science.disorder_sweep`, `science.spectral_analysis`,
//!   `science.anderson_localization`, `science.hessian_eigen`,
//!   `science.agent_coordination`, `science.training_trajectory`
//!
//! **coralForge** (nF-01/02):
//!   `science.evoformer_block`, `science.structure_module`,
//!   `science.folding_health`
//!
//! **GPU dispatch**:
//!   `science.gpu_dispatch` — route arbitrary Dispatcher operations
//!
//! ## biomeOS integration
//!
//! On startup, probes for a biomeOS orchestrator socket and registers
//! capabilities via `lifecycle.register` + `capability.register`.
//! Sends heartbeats every 30s. Deregisters on SIGTERM.
//!
//! Socket: `$XDG_RUNTIME_DIR/biomeos/neuralspring-{family_id}.sock`

#![allow(
    clippy::pedantic,
    clippy::nursery,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]

mod folding;
mod spectral;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Semaphore;

use neural_spring::gpu_dispatch::Dispatcher;

// ═══════════════════════════════════════════════════════════════════
// JSON-RPC 2.0 types
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[serde(rename = "jsonrpc")]
    _jsonrpc: String,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
    id: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
    id: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

impl JsonRpcResponse {
    pub fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0",
            result: Some(result),
            error: None,
            id,
        }
    }

    pub fn error(id: serde_json::Value, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0",
            result: None,
            error: Some(JsonRpcError { code, message }),
            id,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Shared primal state
// ═══════════════════════════════════════════════════════════════════

pub struct PrimalState {
    pub dispatcher: Dispatcher,
    pub start_time: Instant,
    pub requests_served: AtomicU64,
}

pub const ALL_CAPABILITIES: &[&str] = &[
    "science.spectral_analysis",
    "science.anderson_localization",
    "science.hessian_eigen",
    "science.agent_coordination",
    "science.ipr",
    "science.disorder_sweep",
    "science.training_trajectory",
    "science.evoformer_block",
    "science.structure_module",
    "science.folding_health",
    "science.gpu_dispatch",
];

// ═══════════════════════════════════════════════════════════════════
// Request dispatcher
// ═══════════════════════════════════════════════════════════════════

fn dispatch_sync(request: &JsonRpcRequest, state: &PrimalState) -> Option<JsonRpcResponse> {
    let id = request.id.clone();
    let params = &request.params;

    Some(match request.method.as_str() {
        "health" => spectral::handle_health(id, state),
        "science.ipr" => spectral::handle_ipr(id, params),
        "science.disorder_sweep" => spectral::handle_disorder_sweep(id, params),
        "science.spectral_analysis" => spectral::handle_spectral_analysis(id, params),
        "science.anderson_localization" => spectral::handle_anderson_localization(id, params),
        "science.hessian_eigen" => spectral::handle_hessian_eigen(id, params),
        "science.agent_coordination" => spectral::handle_agent_coordination(id, params),
        "science.training_trajectory" => spectral::handle_training_trajectory(id, params),
        "science.evoformer_block" => folding::handle_evoformer_block(id, params),
        "science.structure_module" => folding::handle_structure_module(id, params),
        "science.folding_health" => folding::handle_folding_health(id, state),
        "science.gpu_dispatch" => folding::handle_gpu_dispatch(id, params, state),
        "primal.forward" | "data.ncbi_search" | "data.ncbi_fetch" | "data.pdb_search"
        | "data.pdb_fetch" => return None,
        _ => JsonRpcResponse::error(id, -32601, format!("Method not found: {}", request.method)),
    })
}

async fn dispatch_async(request: &JsonRpcRequest) -> JsonRpcResponse {
    let id = request.id.clone();
    let params = &request.params;

    match request.method.as_str() {
        "primal.forward" => handle_forward(id, params).await,
        method if method.starts_with("data.") => {
            match forward_to_primal("nestgate", method, params).await {
                Ok(resp) => JsonRpcResponse::success(id, resp),
                Err(e) => {
                    JsonRpcResponse::error(id, -32000, format!("NestGate forward failed: {e}"))
                }
            }
        }
        _ => JsonRpcResponse::error(id, -32601, format!("Method not found: {}", request.method)),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Cross-primal forwarding (capability-based discovery)
// ═══════════════════════════════════════════════════════════════════

async fn forward_to_primal(
    primal_name: &str,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value> {
    let socket = discover_primal_socket(primal_name)?;
    let stream = UnixStream::connect(&socket)
        .await
        .with_context(|| format!("connecting to {primal_name} at {}", socket.display()))?;

    let (reader, mut writer) = stream.into_split();

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    });

    let mut payload = serde_json::to_vec(&request)?;
    payload.push(b'\n');
    writer.write_all(&payload).await?;
    writer.flush().await?;

    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();
    buf_reader.read_line(&mut line).await?;

    let resp: serde_json::Value = serde_json::from_str(line.trim())?;
    Ok(resp)
}

fn discover_primal_socket(primal_name: &str) -> Result<PathBuf> {
    let socket_dir = resolve_socket_dir();
    let family_id = get_family_id();

    let with_family = socket_dir.join(format!("{primal_name}-{family_id}.sock"));
    if with_family.exists() {
        return Ok(with_family);
    }

    let without_family = socket_dir.join(format!("{primal_name}.sock"));
    if without_family.exists() {
        return Ok(without_family);
    }

    if let Ok(entries) = std::fs::read_dir(&socket_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(primal_name) && name_str.ends_with(".sock") {
                return Ok(entry.path());
            }
        }
    }

    anyhow::bail!(
        "No socket found for primal '{primal_name}' in {}",
        socket_dir.display()
    )
}

async fn handle_forward(id: serde_json::Value, params: &serde_json::Value) -> JsonRpcResponse {
    let primal = match params.get("primal").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => {
            return JsonRpcResponse::error(id, -32602, "Missing 'primal' parameter".to_string())
        }
    };
    let method = match params.get("method").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => {
            return JsonRpcResponse::error(id, -32602, "Missing 'method' parameter".to_string())
        }
    };
    let inner_params = params
        .get("params")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    match forward_to_primal(&primal, &method, &inner_params).await {
        Ok(resp) => JsonRpcResponse::success(id, resp),
        Err(e) => JsonRpcResponse::error(id, -32000, format!("Forward failed: {e}")),
    }
}

// ═══════════════════════════════════════════════════════════════════
// biomeOS registration
// ═══════════════════════════════════════════════════════════════════

async fn register_with_biomeos(our_socket: &std::path::Path) {
    let biomeos_socket = resolve_socket_dir().join("biomeOS.sock");
    if !biomeos_socket.exists() {
        eprintln!(
            "[biomeos] No orchestrator found at {}, running standalone",
            biomeos_socket.display()
        );
        return;
    }

    let reg_result = forward_to_primal_raw(
        &biomeos_socket,
        "lifecycle.register",
        &serde_json::json!({
            "name": "neuralspring",
            "socket_path": our_socket.to_string_lossy(),
            "pid": std::process::id(),
        }),
    )
    .await;

    match reg_result {
        Ok(_) => eprintln!("[biomeos] Registered with lifecycle manager"),
        Err(e) => eprintln!("[biomeos] lifecycle.register failed (non-fatal): {e}"),
    }

    for cap in ALL_CAPABILITIES {
        let cap_result = forward_to_primal_raw(
            &biomeos_socket,
            "capability.register",
            &serde_json::json!({
                "primal": "neuralspring",
                "capability": cap,
                "socket_path": our_socket.to_string_lossy(),
            }),
        )
        .await;

        if let Err(e) = cap_result {
            eprintln!("[biomeos] capability.register({cap}) failed (non-fatal): {e}");
        }
    }

    eprintln!(
        "[biomeos] All {} capabilities registered",
        ALL_CAPABILITIES.len()
    );
}

async fn forward_to_primal_raw(
    socket_path: &std::path::Path,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value> {
    let stream = UnixStream::connect(socket_path).await?;
    let (reader, mut writer) = stream.into_split();

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    });

    let mut payload = serde_json::to_vec(&request)?;
    payload.push(b'\n');
    writer.write_all(&payload).await?;
    writer.flush().await?;

    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        buf_reader.read_line(&mut line),
    )
    .await
    .with_context(|| "timeout waiting for biomeOS response")??;

    let resp: serde_json::Value = serde_json::from_str(line.trim())?;
    Ok(resp)
}

async fn heartbeat_loop(our_socket: PathBuf) {
    let biomeos_socket = resolve_socket_dir().join("biomeOS.sock");
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));

    loop {
        interval.tick().await;

        if !biomeos_socket.exists() {
            continue;
        }

        let _ = forward_to_primal_raw(
            &biomeos_socket,
            "lifecycle.status",
            &serde_json::json!({
                "name": "neuralspring",
                "socket_path": our_socket.to_string_lossy(),
                "status": "healthy",
            }),
        )
        .await;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Socket resolution (capability-based, no hardcoded paths)
// ═══════════════════════════════════════════════════════════════════

fn resolve_socket_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("BIOMEOS_SOCKET_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(xdg).join("biomeos");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let Ok(meta) = std::fs::metadata("/proc/self") {
            let uid = meta.uid();
            let dir = PathBuf::from(format!("/run/user/{uid}/biomeos"));
            if dir.parent().is_some_and(std::path::Path::exists) {
                return dir;
            }
        }
    }
    std::env::temp_dir().join("biomeos")
}

fn resolve_socket_path(family_id: &str) -> PathBuf {
    resolve_socket_dir().join(format!("neuralspring-{family_id}.sock"))
}

fn get_family_id() -> String {
    if let Ok(id) = std::env::var("FAMILY_ID") {
        return id;
    }
    if let Ok(id) = std::env::var("BIOMEOS_FAMILY_ID") {
        return id;
    }
    "default".to_string()
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let family_id = get_family_id();
    let socket_path = resolve_socket_path(&family_id);

    if let Some(parent) = socket_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .context("creating socket directory")?;
    }

    if socket_path.exists() {
        tokio::fs::remove_file(&socket_path)
            .await
            .context("removing stale socket")?;
    }

    eprintln!("[init] Initializing GPU dispatcher...");
    let dispatcher = Dispatcher::new().await;
    eprintln!(
        "[init] Dispatcher ready: backend={}, gpu={}",
        dispatcher.backend(),
        dispatcher.has_gpu()
    );

    let state = Arc::new(PrimalState {
        dispatcher,
        start_time: Instant::now(),
        requests_served: AtomicU64::new(0),
    });

    let listener = UnixListener::bind(&socket_path)
        .context(format!("binding to {}", socket_path.display()))?;

    eprintln!("neuralSpring primal listening on {}", socket_path.display());
    eprintln!("  Family ID: {family_id}");
    eprintln!("  Mode: Tower (local Eastgate)");
    eprintln!("  Capabilities ({}):", ALL_CAPABILITIES.len());
    for cap in ALL_CAPABILITIES {
        eprintln!("    - {cap}");
    }

    register_with_biomeos(&socket_path).await;

    let heartbeat_socket = socket_path.clone();
    tokio::spawn(heartbeat_loop(heartbeat_socket));

    let shutdown_socket = socket_path.clone();
    tokio::spawn(async move {
        if let Ok(()) = tokio::signal::ctrl_c().await {
            eprintln!("\n[shutdown] SIGINT received, cleaning up...");
            let _ = tokio::fs::remove_file(&shutdown_socket).await;
            std::process::exit(0);
        }
    });

    let concurrency = Arc::new(Semaphore::new(4));

    loop {
        let (stream, _addr) = listener.accept().await?;
        let permit = concurrency.clone().acquire_owned().await?;
        let state = state.clone();

        tokio::spawn(async move {
            let (reader, mut writer) = stream.into_split();
            let mut buf_reader = BufReader::new(reader);
            let mut line = String::new();

            while buf_reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    line.clear();
                    continue;
                }

                let response = match serde_json::from_str::<JsonRpcRequest>(trimmed) {
                    Ok(req) => {
                        state.requests_served.fetch_add(1, Ordering::Relaxed);
                        match dispatch_sync(&req, &state) {
                            Some(resp) => resp,
                            None => dispatch_async(&req).await,
                        }
                    }
                    Err(e) => JsonRpcResponse::error(
                        serde_json::Value::Null,
                        -32700,
                        format!("Parse error: {e}"),
                    ),
                };

                let resp_json = serde_json::to_vec(&response).unwrap_or_default();
                let _ = writer.write_all(&resp_json).await;
                let _ = writer.write_all(b"\n").await;
                let _ = writer.flush().await;

                line.clear();
            }

            drop(permit);
        });
    }
}
