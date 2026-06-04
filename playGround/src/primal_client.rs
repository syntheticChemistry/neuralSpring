// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed client for the neuralSpring primal.
//!
//! Wraps [`crate::ipc_client`] with convenience methods for each of the
//! 14 `science.*` capabilities and general primal queries (`health`,
//! `capability.list`).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::ipc_client;

/// Primal socket hint for name-based fallback discovery.
///
/// Capability-based discovery (`science.spectral_analysis`) is the primary
/// path; this hint is only used when no running primal advertises the
/// required capability.
const PRIMAL_SOCKET_HINT: &str = neural_spring::niche::NICHE_NAME;

/// Primary capability used to discover a neuralSpring primal instance.
const DISCOVERY_CAPABILITY: &str = "science.spectral_analysis";

/// Typed JSON-RPC client for the neuralSpring primal (science capabilities over Unix IPC).
#[deprecated(
    since = "0.2.0",
    note = "use CompositionContext::from_live_discovery_with_fallback() and ctx.call() instead"
)]
pub struct PrimalClient {
    /// Path to the primal's Unix domain socket.
    socket: PathBuf,
    /// Per-RPC timeout for IPC calls.
    timeout: Duration,
}

#[allow(deprecated)]
impl PrimalClient {
    /// Connect to neuralSpring via capability-based socket discovery.
    ///
    /// Probes the biomeOS socket directory for any primal advertising
    /// `science.spectral_analysis`, falling back to name-based discovery
    /// if no capability probe succeeds.
    pub fn discover() -> Result<Self> {
        let socket = ipc_client::discover_by_capability(DISCOVERY_CAPABILITY, PRIMAL_SOCKET_HINT)
            .context("discovering neuralSpring socket")?;
        Ok(Self {
            socket,
            timeout: ipc_client::ipc_timeout(),
        })
    }

    /// Connect to neuralSpring at a specific socket path.
    #[must_use]
    pub fn new(socket: PathBuf) -> Self {
        Self {
            socket,
            timeout: ipc_client::ipc_timeout(),
        }
    }

    /// Generic capability call.
    pub async fn call_capability(
        &self,
        capability: &str,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        ipc_client::call(&self.socket, capability, params, self.timeout).await
    }

    /// Health check.
    pub async fn health(&self) -> Result<serde_json::Value> {
        ipc_client::call(&self.socket, "health", &serde_json::json!({}), self.timeout).await
    }

    /// List advertised capabilities.
    pub async fn capability_list(&self) -> Result<Vec<String>> {
        let result = ipc_client::call(
            &self.socket,
            "capability.list",
            &serde_json::json!({}),
            self.timeout,
        )
        .await?;
        let caps = result
            .get("capabilities")
            .and_then(|c| c.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        Ok(caps)
    }

    /// Run spectral analysis on a weight matrix (`science.spectral_analysis`).
    pub async fn spectral_analysis(&self, params: &serde_json::Value) -> Result<serde_json::Value> {
        self.call_capability("science.spectral_analysis", params)
            .await
    }

    /// Anderson localization study on a tight-binding Hamiltonian (`science.anderson_localization`).
    pub async fn anderson_localization(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        self.call_capability("science.anderson_localization", params)
            .await
    }

    /// Hessian eigen-spectrum for loss-landscape diagnostics (`science.hessian_eigen`).
    pub async fn hessian_eigen(&self, params: &serde_json::Value) -> Result<serde_json::Value> {
        self.call_capability("science.hessian_eigen", params).await
    }

    /// Multi-agent coordination metrics from an interaction graph (`science.agent_coordination`).
    pub async fn agent_coordination(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        self.call_capability("science.agent_coordination", params)
            .await
    }

    /// Inverse participation ratio and related localization metrics (`science.ipr`).
    pub async fn ipr(&self, params: &serde_json::Value) -> Result<serde_json::Value> {
        self.call_capability("science.ipr", params).await
    }

    /// Disorder-strength sweep across a lattice (`science.disorder_sweep`).
    pub async fn disorder_sweep(&self, params: &serde_json::Value) -> Result<serde_json::Value> {
        self.call_capability("science.disorder_sweep", params).await
    }

    /// Training trajectory / dynamics summary (`science.training_trajectory`).
    pub async fn training_trajectory(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        self.call_capability("science.training_trajectory", params)
            .await
    }

    /// GPU shader dispatch workload (`science.gpu_dispatch`).
    pub async fn gpu_dispatch(&self, params: &serde_json::Value) -> Result<serde_json::Value> {
        self.call_capability("science.gpu_dispatch", params).await
    }
}
