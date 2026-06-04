// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed client for the Squirrel MCP primal.
//!
//! Wraps [`crate::ipc_client`] with Squirrel-specific JSON-RPC methods:
//! `ai.query`, `tool.execute`, `capability.announce`, `system.health`,
//! and `ai.list_providers`.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::ipc_client;

/// Typed JSON-RPC client for the Squirrel MCP / AI gateway primal.
pub struct SquirrelClient {
    /// Path to Squirrel's Unix domain socket.
    socket: PathBuf,
    /// Per-RPC timeout for IPC calls.
    timeout: Duration,
}

/// Parsed `ai.query` reply: model output and usage metadata from Squirrel.
#[derive(Debug, Deserialize)]
pub struct AiResponse {
    /// Completion text returned by the provider.
    #[serde(default)]
    pub response: String,
    /// Provider id that served the request (e.g. openai, anthropic).
    #[serde(default)]
    pub provider: String,
    /// Model id or name used for the call.
    #[serde(default)]
    pub model: String,
    /// Total tokens billed or reported by the provider.
    #[serde(default)]
    pub tokens_used: u64,
    /// End-to-end latency for the query (milliseconds).
    #[serde(default)]
    pub latency_ms: u64,
    /// Whether Squirrel reported the call as successful.
    #[serde(default)]
    pub success: bool,
}

/// `system.health` snapshot from Squirrel.
#[derive(Debug, Deserialize)]
pub struct HealthStatus {
    /// High-level status string (e.g. ok, degraded).
    #[serde(default)]
    pub status: String,
    /// Process uptime in seconds.
    #[serde(default)]
    pub uptime_secs: u64,
}

/// Squirrel's primary capability for capability-based discovery.
const SQUIRREL_CAPABILITY: &str = "ai.query";

impl SquirrelClient {
    /// Connect to Squirrel via capability-based discovery with name fallback.
    #[allow(deprecated)]
    pub fn discover() -> Result<Self> {
        let socket = ipc_client::discover_by_capability(
            SQUIRREL_CAPABILITY,
            neural_spring::primal_names::SQUIRREL,
        )
        .context("discovering Squirrel socket")?;
        Ok(Self {
            socket,
            timeout: ipc_client::ipc_timeout(),
        })
    }

    /// Connect to Squirrel at a specific socket path.
    #[must_use]
    pub fn new(socket: PathBuf) -> Self {
        Self {
            socket,
            timeout: ipc_client::ipc_timeout(),
        }
    }

    /// Send a prompt to an AI provider via Squirrel's `ai.query`.
    pub async fn ai_query(
        &self,
        prompt: &str,
        model: Option<&str>,
        max_tokens: Option<u32>,
        temperature: Option<f64>,
    ) -> Result<AiResponse> {
        let mut params = serde_json::json!({ "prompt": prompt });
        if let Some(m) = model {
            params["model"] = serde_json::Value::String(m.to_owned());
        }
        if let Some(t) = max_tokens {
            params["max_tokens"] = serde_json::json!(t);
        }
        if let Some(t) = temperature {
            params["temperature"] = serde_json::json!(t);
        }

        let result = ipc_client::call(&self.socket, "ai.query", &params, self.timeout).await?;
        let resp: AiResponse = serde_json::from_value(result)?;
        Ok(resp)
    }

    /// Announce capabilities to Squirrel via `capability.announce`.
    pub async fn announce_capabilities(
        &self,
        primal_name: &str,
        capabilities: &[&str],
        socket_path: &str,
    ) -> Result<serde_json::Value> {
        let params = serde_json::json!({
            "primal": primal_name,
            "capabilities": capabilities,
            "socket": socket_path,
        });
        ipc_client::call(&self.socket, "capability.announce", &params, self.timeout).await
    }

    /// Execute a tool via Squirrel's `tool.execute`.
    pub async fn execute_tool(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let params = serde_json::json!({
            "tool": tool_name,
            "args": args,
        });
        ipc_client::call(&self.socket, "tool.execute", &params, self.timeout).await
    }

    /// List available AI providers.
    pub async fn list_providers(&self) -> Result<serde_json::Value> {
        ipc_client::call(
            &self.socket,
            "ai.list_providers",
            &serde_json::json!({}),
            self.timeout,
        )
        .await
    }

    /// Health check.
    pub async fn health(&self) -> Result<HealthStatus> {
        let result = ipc_client::call(
            &self.socket,
            "system.health",
            &serde_json::json!({}),
            self.timeout,
        )
        .await?;
        let status: HealthStatus = serde_json::from_value(result)?;
        Ok(status)
    }
}
