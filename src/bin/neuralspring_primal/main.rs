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
//! capabilities via `nucleus.register` + `capability.register`.
//! Sends heartbeats every 30s via `nucleus.heartbeat`.
//! Deregisters on SIGINT/SIGTERM via `nucleus.deregister`.
//!
//! Socket: `$XDG_RUNTIME_DIR/biomeos/neuralspring-{family_id}.sock`

#![expect(
    clippy::similar_names,
    reason = "cli / state / reader / writer naming mirrors JSON-RPC protocol terms"
)]

mod biomeos;
mod discovery;
mod folding;
mod handlers;
mod rpc;
mod spectral;
mod tower;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, UnixListener};
use tokio::sync::Semaphore;
use tokio::sync::watch;

use serde_json::error::Category;

use neural_spring::gpu_dispatch::Dispatcher;

use rpc::JsonRpcResponse;

// ═══════════════════════════════════════════════════════════════════
// UniBin CLI (wateringHole UNIBIN_ARCHITECTURE_STANDARD)
// ═══════════════════════════════════════════════════════════════════

#[derive(Parser)]
#[command(
    name = "neuralspring",
    version,
    about = "neuralSpring — spectral analysis & structure prediction primal for biomeOS",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the JSON-RPC server (Tower mode, default)
    Serve {
        /// Override family ID (default: $`FAMILY_ID` or "default")
        #[arg(long)]
        family_id: Option<String>,
    },
    /// Print health / version info and exit
    Health,
    /// List all advertised capabilities
    Capabilities,
}

// ═══════════════════════════════════════════════════════════════════
// Shared primal state and config
// ═══════════════════════════════════════════════════════════════════

pub struct PrimalState {
    pub dispatcher: Dispatcher,
    pub start_time: Instant,
    pub requests_served: AtomicU64,
}

const PRIMAL_NAME: &str = env!("CARGO_PKG_NAME");

const DEFAULT_IPC_TIMEOUT_SECS: u64 = 5;
const DEFAULT_HEARTBEAT_SECS: u64 = 30;
const DEFAULT_MAX_CONCURRENT: usize = 4;

fn orchestrator_socket() -> String {
    if let Ok(s) = std::env::var(neural_spring::config::ENV_BIOMEOS_ORCHESTRATOR) {
        return s;
    }
    if let Ok(xdg) = std::env::var(neural_spring::config::ENV_XDG_RUNTIME_DIR) {
        let path = std::path::PathBuf::from(xdg)
            .join(neural_spring::config::BIOMEOS_SOCKET_SUBDIR)
            .join(neural_spring::config::BIOMEOS_ORCHESTRATOR_SOCKET);
        if path.exists() {
            return path.to_string_lossy().into_owned();
        }
    }
    let tmp = std::env::temp_dir()
        .join(neural_spring::config::BIOMEOS_SOCKET_SUBDIR)
        .join(neural_spring::config::BIOMEOS_ORCHESTRATOR_SOCKET);
    tmp.to_string_lossy().into_owned()
}

fn ipc_response_timeout_secs() -> u64 {
    std::env::var(neural_spring::config::ENV_IPC_TIMEOUT)
        .or_else(|_| std::env::var(neural_spring::config::ENV_IPC_TIMEOUT_SPRING))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_IPC_TIMEOUT_SECS)
}

fn heartbeat_interval_secs() -> u64 {
    std::env::var(neural_spring::config::ENV_HEARTBEAT_SECS)
        .or_else(|_| std::env::var(neural_spring::config::ENV_HEARTBEAT_SECS_SPRING))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_HEARTBEAT_SECS)
}

pub use neural_spring::config::ALL_CAPABILITIES;

// ═══════════════════════════════════════════════════════════════════
// Request dispatcher
// ═══════════════════════════════════════════════════════════════════

fn dispatch_sync(request: &rpc::JsonRpcRequest, state: &PrimalState) -> Option<JsonRpcResponse> {
    let id = request.id.clone();
    let params = &request.params;
    let method = rpc::normalize_method(&request.method);

    Some(match method {
        "health" => spectral::handle_health(id, state),
        "health.check" => handlers::handle_health_check(id, state),
        "identity.get" => handlers::handle_identity_get(id),
        "mcp.tools.list" => handlers::handle_mcp_tools_list(id),
        "capabilities.list" | "capability.list" | "primal.capabilities" => {
            handlers::handle_capability_list(id)
        }
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
        "science.cross_spring_provenance" => handlers::handle_cross_spring_provenance(id),
        "science.cross_spring_benchmark" => handlers::handle_cross_spring_benchmark(id, state),
        "science.precision_routing" => handlers::handle_precision_routing(id, state),
        "inference.complete" => handlers::handle_inference_complete(id, params),
        "inference.embed" => handlers::handle_inference_embed(id, params),
        "inference.models" => handlers::handle_inference_models(id),
        "health.liveness" => handlers::handle_liveness(id),
        "health.readiness" => handlers::handle_readiness(id, state),
        "provenance.begin" | "provenance.record" | "provenance.complete" | "provenance.status" => {
            handlers::handle_provenance(id, method, params)
        }
        "primal.discover" => handlers::handle_primal_discover(id),
        "compute.offload" => handlers::handle_compute_offload(id, params, state),
        "primal.forward" | "data.ncbi_search" | "data.ncbi_fetch" | "data.pdb_search"
        | "data.pdb_fetch" => return None,
        _ => JsonRpcResponse::error(
            id,
            rpc::error_code::METHOD_NOT_FOUND,
            format!("Method not found: {method}"),
        ),
    })
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Health) => {
            println!(
                "{} {} (AGPL-3.0-or-later)",
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
            );
            println!("status: ready");
            return Ok(());
        }
        Some(Commands::Capabilities) => {
            for cap in ALL_CAPABILITIES {
                println!("{cap}");
            }
            return Ok(());
        }
        Some(Commands::Serve { .. }) | None => {}
    }

    let cli_family = match &cli.command {
        Some(Commands::Serve {
            family_id: Some(id),
        }) => Some(id.clone()),
        _ => None,
    };
    let family_id = cli_family.unwrap_or_else(discovery::get_family_id);
    let socket_path = discovery::resolve_socket_path(&family_id);

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

    log::info!("Initializing GPU dispatcher...");
    let dispatcher = Dispatcher::new().await;
    log::info!(
        "Dispatcher ready: backend={}, gpu={}",
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

    let tcp_port: u16 = std::env::var(neural_spring::config::ENV_TCP_PORT)
        .or_else(|_| std::env::var(neural_spring::config::ENV_TCP_PORT_SPRING))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let tcp_listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, tcp_port))
        .await
        .context("binding TCP fallback")?;
    let tcp_addr = tcp_listener.local_addr().context("TCP local address")?;
    log::info!("TCP fallback listening on {tcp_addr}");

    log::info!("neuralSpring primal listening on {}", socket_path.display());
    log::info!("Family ID: {family_id}");
    log::info!("Mode: Tower (biomeOS niche)");
    log::info!("Capabilities ({}):", ALL_CAPABILITIES.len());
    for cap in ALL_CAPABILITIES {
        log::debug!("  {cap}");
    }

    tower::probe_tower_atomic();

    biomeos::register_with_biomeos(&socket_path).await;

    if std::env::var(neural_spring::config::ENV_VISUALIZATION_PUSH)
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    {
        push_petaltongue_scenario(&family_id);
    }

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    spawn_lifecycle_tasks(&socket_path, shutdown_tx);

    accept_loop(listener, tcp_listener, state, shutdown_rx).await
}

fn push_petaltongue_scenario(family_id: &str) {
    match neural_spring::visualization::PetalTonguePushClient::discover() {
        Ok(client) => {
            let (scenario, edges) = neural_spring::visualization::full_study();
            let mut merged = scenario;
            merged.edges.extend_from_slice(&edges);
            match client.push_render(
                &format!("neuralspring-{family_id}"),
                "neuralSpring Full Study",
                &merged,
            ) {
                Ok(()) => log::info!("petalTongue: pushed full study scenario"),
                Err(e) => log::warn!("petalTongue: push failed (non-fatal): {e}"),
            }
        }
        Err(_) => {
            log::debug!("petalTongue not found (optional, skipping visualization push)");
        }
    }
}

fn spawn_lifecycle_tasks(socket_path: &std::path::Path, shutdown_tx: watch::Sender<bool>) {
    let heartbeat_socket = socket_path.to_path_buf();
    tokio::spawn(biomeos::heartbeat_loop(heartbeat_socket));

    let shutdown_socket = socket_path.to_path_buf();
    let shutdown_tx_int = shutdown_tx.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        log::info!("SIGINT received, deregistering...");
        biomeos::deregister_from_nucleus(&shutdown_socket).await;
        let _ = tokio::fs::remove_file(&shutdown_socket).await;
        let _ = shutdown_tx_int.send(true);
    });

    #[cfg(unix)]
    {
        let sigterm_socket = socket_path.to_path_buf();
        let shutdown_tx_term = shutdown_tx.clone();
        tokio::spawn(async move {
            let Ok(mut sig) =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            else {
                return;
            };
            sig.recv().await;
            log::info!("SIGTERM received, deregistering...");
            biomeos::deregister_from_nucleus(&sigterm_socket).await;
            let _ = tokio::fs::remove_file(&sigterm_socket).await;
            let _ = shutdown_tx_term.send(true);
        });
    }
    drop(shutdown_tx);
}

async fn handle_connection<R, W>(
    reader: R,
    mut writer: W,
    state: Arc<PrimalState>,
    permit: tokio::sync::OwnedSemaphorePermit,
) where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
    W: tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();

    while buf_reader.read_line(&mut line).await.unwrap_or(0) > 0 {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            line.clear();
            continue;
        }

        let response = match serde_json::from_str::<serde_json::Value>(trimmed) {
            Err(e) => {
                let msg = format!("Parse error: {e}");
                match e.classify() {
                    Category::Syntax | Category::Eof | Category::Io => JsonRpcResponse::error(
                        serde_json::Value::Null,
                        rpc::error_code::PARSE_ERROR,
                        msg,
                    ),
                    Category::Data => JsonRpcResponse::error(
                        serde_json::Value::Null,
                        rpc::error_code::INVALID_REQUEST,
                        msg,
                    ),
                }
            }
            Ok(v) => {
                let id_fallback = v.get("id").cloned().unwrap_or(serde_json::Value::Null);
                match serde_json::from_value::<rpc::JsonRpcRequest>(v) {
                    Err(e) => JsonRpcResponse::error(
                        id_fallback,
                        rpc::error_code::INVALID_REQUEST,
                        format!("Invalid request: {e}"),
                    ),
                    Ok(req) => {
                        if req.jsonrpc_version != "2.0" {
                            JsonRpcResponse::error(
                                req.id.clone(),
                                rpc::error_code::INVALID_REQUEST,
                                "jsonrpc must be \"2.0\"".to_string(),
                            )
                        } else if req.method.is_empty() {
                            JsonRpcResponse::error(
                                req.id.clone(),
                                rpc::error_code::INVALID_REQUEST,
                                "method must not be empty".to_string(),
                            )
                        } else {
                            state.requests_served.fetch_add(1, Ordering::Relaxed);
                            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                dispatch_sync(&req, &state)
                            })) {
                                Err(_) => JsonRpcResponse::error(
                                    req.id.clone(),
                                    rpc::error_code::INTERNAL_ERROR,
                                    "Internal error: handler panicked".to_string(),
                                ),
                                Ok(Some(resp)) => resp,
                                Ok(None) => handlers::dispatch_async(&req).await,
                            }
                        }
                    }
                }
            }
        };

        let resp_json = serde_json::to_vec(&response).unwrap_or_default();
        let _ = writer.write_all(&resp_json).await;
        let _ = writer.write_all(b"\n").await;
        let _ = writer.flush().await;

        line.clear();
    }

    drop(permit);
}

async fn accept_loop(
    unix_listener: UnixListener,
    tcp_listener: TcpListener,
    state: Arc<PrimalState>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<()> {
    let max_concurrent: usize = std::env::var("PRIMAL_MAX_CONCURRENT")
        .or_else(|_| std::env::var("NEURALSPRING_MAX_CONCURRENT"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_CONCURRENT);
    let concurrency = Arc::new(Semaphore::new(max_concurrent));

    loop {
        tokio::select! {
            result = unix_listener.accept() => {
                let (stream, _addr) = result?;
                let permit = concurrency.clone().acquire_owned().await?;
                let state = state.clone();

                tokio::spawn(async move {
                    let (reader, writer) = stream.into_split();
                    handle_connection(reader, writer, state, permit).await;
                });
            }
            result = tcp_listener.accept() => {
                let (stream, addr) = result?;
                log::debug!("TCP connection from {addr}");
                let permit = concurrency.clone().acquire_owned().await?;
                let state = state.clone();

                tokio::spawn(async move {
                    let (reader, writer) = stream.into_split();
                    handle_connection(reader, writer, state, permit).await;
                });
            }
            result = shutdown_rx.changed() => {
                if result.is_err() {
                    log::info!("Shutting down accept loop (shutdown channel closed)");
                    break;
                }
                if *shutdown_rx.borrow() {
                    log::info!("Shutting down accept loop");
                    break;
                }
            }
        }
    }
    Ok(())
}
