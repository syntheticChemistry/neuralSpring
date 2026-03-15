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
    clippy::pedantic,
    clippy::nursery,
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    reason = "validation binary"
)]

mod biomeos;
mod discovery;
mod folding;
mod handlers;
mod rpc;
mod spectral;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tokio::sync::Semaphore;

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
        /// Override family ID (default: $FAMILY_ID or "default")
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

fn orchestrator_socket() -> String {
    if let Ok(s) = std::env::var("BIOMEOS_ORCHESTRATOR_SOCKET") {
        return s;
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let path = format!("{xdg}/biomeos/biomeos.sock");
        if std::path::Path::new(&path).exists() {
            return path;
        }
    }
    let tmp = std::env::temp_dir().join("biomeos/biomeos.sock");
    tmp.to_string_lossy().into_owned()
}

fn ipc_response_timeout_secs() -> u64 {
    std::env::var("PRIMAL_IPC_TIMEOUT_SECS")
        .or_else(|_| std::env::var("NEURALSPRING_IPC_TIMEOUT_SECS"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
}

fn heartbeat_interval_secs() -> u64 {
    std::env::var("PRIMAL_HEARTBEAT_SECS")
        .or_else(|_| std::env::var("NEURALSPRING_HEARTBEAT_SECS"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30)
}

pub use neural_spring::config::ALL_CAPABILITIES;

// ═══════════════════════════════════════════════════════════════════
// Request dispatcher
// ═══════════════════════════════════════════════════════════════════

fn dispatch_sync(request: &rpc::JsonRpcRequest, state: &PrimalState) -> Option<JsonRpcResponse> {
    let id = request.id.clone();
    let params = &request.params;

    Some(match request.method.as_str() {
        "health" => spectral::handle_health(id, state),
        "capability.list" => handlers::handle_capability_list(id),
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
        "primal.forward" | "data.ncbi_search" | "data.ncbi_fetch" | "data.pdb_search"
        | "data.pdb_fetch" => return None,
        _ => JsonRpcResponse::error(
            id,
            rpc::error_code::METHOD_NOT_FOUND,
            format!("Method not found: {}", request.method),
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
        Some(Commands::Serve {
            family_id: Some(id),
        }) => {
            std::env::set_var("FAMILY_ID", id);
        }
        Some(Commands::Serve { family_id: None }) | None => {}
    }

    let family_id = discovery::get_family_id();
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

    biomeos::register_with_biomeos(&socket_path).await;

    // Optional petalTongue integration: push a full-study scenario on startup.
    // If petalTongue is unavailable, this silently skips (no compile-time dep).
    match neural_spring::visualization::PetalTonguePushClient::discover() {
        Ok(client) => {
            let (scenario, edges) = neural_spring::visualization::full_study();
            let mut merged = scenario.clone();
            merged.edges.extend_from_slice(&edges);
            match client.push_render(
                &format!("neuralspring-{family_id}"),
                "neuralSpring Full Study",
                &merged,
            ) {
                Ok(()) => eprintln!("[petaltongue] Pushed full study scenario"),
                Err(e) => eprintln!("[petaltongue] Push failed (non-fatal): {e}"),
            }
        }
        Err(_) => {
            eprintln!("[petaltongue] Not found (optional, skipping visualization push)");
        }
    }

    let heartbeat_socket = socket_path.clone();
    tokio::spawn(biomeos::heartbeat_loop(heartbeat_socket));

    let shutdown_socket = socket_path.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        eprintln!("\n[shutdown] SIGINT received, deregistering...");
        biomeos::deregister_from_nucleus(&shutdown_socket).await;
        let _ = tokio::fs::remove_file(&shutdown_socket).await;
        std::process::exit(0);
    });

    #[cfg(unix)]
    {
        let sigterm_socket = socket_path.clone();
        tokio::spawn(async move {
            let mut sig = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect(
                    "SIGTERM handler registration requires a tokio runtime with signal support",
                );
            sig.recv().await;
            eprintln!("\n[shutdown] SIGTERM received, deregistering...");
            biomeos::deregister_from_nucleus(&sigterm_socket).await;
            let _ = tokio::fs::remove_file(&sigterm_socket).await;
            std::process::exit(0);
        });
    }

    let max_concurrent: usize = std::env::var("PRIMAL_MAX_CONCURRENT")
        .or_else(|_| std::env::var("NEURALSPRING_MAX_CONCURRENT"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let concurrency = Arc::new(Semaphore::new(max_concurrent));

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

                let response = match serde_json::from_str::<rpc::JsonRpcRequest>(trimmed) {
                    Ok(req) => {
                        state.requests_served.fetch_add(1, Ordering::Relaxed);
                        match dispatch_sync(&req, &state) {
                            Some(resp) => resp,
                            None => handlers::dispatch_async(&req).await,
                        }
                    }
                    Err(e) => JsonRpcResponse::error(
                        serde_json::Value::Null,
                        rpc::error_code::PARSE_ERROR,
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
