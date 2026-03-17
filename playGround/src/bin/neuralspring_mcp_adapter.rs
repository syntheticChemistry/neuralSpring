// SPDX-License-Identifier: AGPL-3.0-or-later

//! MCP Adapter: bridges Squirrel MCP and the neuralSpring primal.
//!
//! Discovers both sockets, registers neuralSpring's 14 science
//! capabilities with Squirrel, then listens for incoming tool calls
//! from Squirrel and forwards them to the neuralSpring primal.

#![expect(clippy::nursery, reason = "playground binary — iterating rapidly")]

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;

use neuralspring_playground::ipc_client;
use neuralspring_playground::mcp_tools;
use neuralspring_playground::primal_client::PrimalClient;
use neuralspring_playground::squirrel_client::SquirrelClient;

#[derive(Parser)]
#[command(
    name = "neuralspring-mcp-adapter",
    about = "Bridge between Squirrel MCP and neuralSpring primal"
)]
struct Cli {
    /// Override neuralSpring primal socket path
    #[arg(long)]
    primal_socket: Option<PathBuf>,

    /// Override Squirrel socket path
    #[arg(long)]
    squirrel_socket: Option<PathBuf>,

    /// Listen on this socket for forwarded tool.execute calls from Squirrel
    #[arg(long)]
    listen: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    log::info!("neuralSpring MCP Adapter starting...");

    // --- Discover neuralSpring primal ---
    let primal = match &cli.primal_socket {
        Some(path) => PrimalClient::new(path.clone()),
        None => PrimalClient::discover().context(
            "Could not find neuralSpring primal socket. \
             Is neuralspring_primal running? Use --primal-socket to specify.",
        )?,
    };

    match primal.health().await {
        Ok(h) => log::info!("neuralSpring primal: healthy — {h}"),
        Err(e) => {
            log::warn!("neuralSpring health check failed: {e}");
            log::warn!("Continuing anyway — primal may start later.");
        }
    }

    // --- Discover Squirrel ---
    let squirrel = match &cli.squirrel_socket {
        Some(path) => SquirrelClient::new(path.clone()),
        None => match SquirrelClient::discover() {
            Ok(s) => s,
            Err(e) => {
                log::warn!("Squirrel not found: {e}");
                log::warn!("Running in standalone mode (no AI routing).");
                log::warn!("Set --squirrel-socket or start Squirrel to enable.");
                run_standalone_listener(&cli, &primal).await?;
                return Ok(());
            }
        },
    };

    match squirrel.health().await {
        Ok(h) => log::info!("Squirrel: {} (uptime {}s)", h.status, h.uptime_secs),
        Err(e) => {
            log::warn!("Squirrel health check failed: {e}");
        }
    }

    // --- Register capabilities with Squirrel ---
    let tools = mcp_tools::tool_definitions();
    let cap_names: Vec<&str> = tools.iter().map(|t| t.name).collect();

    let adapter_socket = adapter_socket_path(&cli);
    let socket_str = adapter_socket.to_string_lossy().to_string();

    match squirrel
        .announce_capabilities("neuralspring-playground", &cap_names, &socket_str)
        .await
    {
        Ok(_) => log::info!("Registered {} capabilities with Squirrel", cap_names.len()),
        Err(e) => {
            log::warn!("capability.announce failed: {e}");
            log::warn!("Tools may not be discoverable via Squirrel.");
        }
    }

    log::info!("Registered tools:");
    for tool in &tools {
        log::info!(
            "  - {} ({})",
            tool.name,
            tool.description.split(':').next().unwrap_or("")
        );
    }

    // --- Listen for tool.execute calls ---
    run_listener(&cli, &primal).await
}

fn adapter_socket_path(cli: &Cli) -> PathBuf {
    if let Some(path) = &cli.listen {
        return path.clone();
    }
    let dir = ipc_client::resolve_socket_dir();
    dir.join("neuralspring-mcp-adapter.sock")
}

async fn run_standalone_listener(cli: &Cli, primal: &PrimalClient) -> Result<()> {
    log::info!("Standalone mode: forwarding JSON-RPC to neuralSpring primal.");
    run_listener(cli, primal).await
}

async fn run_listener(cli: &Cli, primal: &PrimalClient) -> Result<()> {
    let socket_path = adapter_socket_path(cli);

    if let Some(parent) = socket_path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }
    if socket_path.exists() {
        tokio::fs::remove_file(&socket_path).await.ok();
    }

    let listener = UnixListener::bind(&socket_path)
        .context(format!("binding to {}", socket_path.display()))?;

    log::info!("Listening on {}", socket_path.display());
    log::info!("Ready to bridge tool.execute -> neuralSpring primal");

    let shutdown_path = socket_path.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        log::info!("Shutting down...");
        let _ = tokio::fs::remove_file(&shutdown_path).await;
        std::process::exit(0);
    });

    loop {
        let (stream, _) = listener.accept().await?;
        let (reader, mut writer) = stream.into_split();
        let mut buf_reader = BufReader::new(reader);
        let mut line = String::new();

        while buf_reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                line.clear();
                continue;
            }

            let response = handle_request(trimmed, primal).await;

            let resp_bytes = serde_json::to_vec(&response).unwrap_or_default();
            let _ = writer.write_all(&resp_bytes).await;
            let _ = writer.write_all(b"\n").await;
            let _ = writer.flush().await;

            line.clear();
        }
    }
}

async fn handle_request(raw: &str, primal: &PrimalClient) -> serde_json::Value {
    let request: serde_json::Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(e) => {
            return serde_json::json!({
                "jsonrpc": "2.0",
                "error": { "code": -32700, "message": format!("Parse error: {e}") },
                "id": null
            });
        }
    };

    let id = request
        .get("id")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let method = request.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let params = request
        .get("params")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    match method {
        "tool.execute" => {
            let tool_name = params.get("tool").and_then(|t| t.as_str()).unwrap_or("");
            let args = params.get("args").cloned().unwrap_or(serde_json::json!({}));

            if !mcp_tools::ALL_CAPABILITIES.contains(&tool_name) {
                return serde_json::json!({
                    "jsonrpc": "2.0",
                    "error": { "code": -32601, "message": format!("Unknown tool: {tool_name}") },
                    "id": id
                });
            }

            match primal.call_capability(tool_name, &args).await {
                Ok(result) => serde_json::json!({
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": id
                }),
                Err(e) => serde_json::json!({
                    "jsonrpc": "2.0",
                    "error": { "code": -32000, "message": format!("{e}") },
                    "id": id
                }),
            }
        }

        "tool.list" => {
            let tools: Vec<_> = mcp_tools::tool_definitions()
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    })
                })
                .collect();
            serde_json::json!({
                "jsonrpc": "2.0",
                "result": { "tools": tools },
                "id": id
            })
        }

        "health" | "system.health" => {
            serde_json::json!({
                "jsonrpc": "2.0",
                "result": { "status": "ok", "adapter": "neuralspring-mcp-adapter" },
                "id": id
            })
        }

        m if m.starts_with("science.") => match primal.call_capability(m, &params).await {
            Ok(result) => serde_json::json!({
                "jsonrpc": "2.0",
                "result": result,
                "id": id
            }),
            Err(e) => serde_json::json!({
                "jsonrpc": "2.0",
                "error": { "code": -32000, "message": format!("{e}") },
                "id": id
            }),
        },

        _ => {
            serde_json::json!({
                "jsonrpc": "2.0",
                "error": { "code": -32601, "message": format!("Method not found: {method}") },
                "id": id
            })
        }
    }
}
