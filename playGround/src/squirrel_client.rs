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

pub struct SquirrelClient {
    socket: PathBuf,
    timeout: Duration,
}

#[derive(Debug, Deserialize)]
pub struct AiResponse {
    #[serde(default)]
    pub response: String,
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub tokens_used: u64,
    #[serde(default)]
    pub latency_ms: u64,
    #[serde(default)]
    pub success: bool,
}

#[derive(Debug, Deserialize)]
pub struct HealthStatus {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub uptime_secs: u64,
}

impl SquirrelClient {
    /// Connect to Squirrel via automatic socket discovery.
    pub fn discover() -> Result<Self> {
        let socket =
            ipc_client::discover_socket("squirrel").context("discovering Squirrel socket")?;
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
