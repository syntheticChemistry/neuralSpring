// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed client for the biomeOS orchestrator.
//!
//! Wraps [`crate::ipc_client`] with biomeOS lifecycle methods:
//! `primal.announce` (Wave 17, preferred), `nucleus.register`,
//! `nucleus.deregister`, `nucleus.heartbeat`,
//! `capability.register`, and `capability.resolve`.
//!
//! Replaces ad-hoc `forward_to_primal_raw` calls with a proper typed
//! interface and consistent request ID generation.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::ipc_client;

/// Typed JSON-RPC client for the biomeOS NUCLEUS orchestrator over Unix sockets.
pub struct BiomeOsClient {
    /// Path to the orchestrator's Unix domain socket.
    socket: PathBuf,
    /// Per-call timeout for IPC round-trips.
    timeout: Duration,
}

/// Result of `capability.resolve`: which primal hosts a requested capability.
#[derive(Debug, Deserialize)]
pub struct ResolveResult {
    /// Advertised primal name that owns the capability.
    #[serde(default)]
    pub primal: String,
    /// Unix socket path where that primal accepts JSON-RPC.
    #[serde(default)]
    pub socket_path: String,
}

impl BiomeOsClient {
    /// Discover the biomeOS orchestrator socket via standard resolution.
    ///
    /// Looks for `biomeos.sock` in the biomeOS socket directory.
    /// Returns `None` if no orchestrator is running.
    #[must_use]
    pub fn discover() -> Option<Self> {
        let socket_dir = ipc_client::resolve_socket_dir();
        let orchestrator = socket_dir.join(neural_spring::config::BIOMEOS_ORCHESTRATOR_SOCKET);
        if orchestrator.exists() {
            Some(Self {
                socket: orchestrator,
                timeout: ipc_client::ipc_timeout(),
            })
        } else {
            None
        }
    }

    /// Connect to a specific orchestrator socket.
    #[must_use]
    pub fn new(socket: PathBuf) -> Self {
        Self {
            socket,
            timeout: ipc_client::ipc_timeout(),
        }
    }

    /// Announce this primal to the NUCLEUS orchestrator (Wave 17 signal API).
    ///
    /// Uses `primal.announce` to register primal identity, capabilities, and
    /// socket in a single call. Falls back to legacy `nucleus.register` if
    /// the orchestrator doesn't support announce.
    pub async fn announce(
        &self,
        primal_name: &str,
        methods: &[&str],
        our_socket: &Path,
    ) -> Result<serde_json::Value> {
        let result = ipc_client::call(
            &self.socket,
            "primal.announce",
            &serde_json::json!({
                "primal_id": primal_name,
                "transport": our_socket.to_string_lossy(),
                "methods": methods,
                "lifecycle": { "state": "running" },
                "pid": std::process::id(),
            }),
            self.timeout,
        )
        .await;

        match result {
            Ok(v) => Ok(v),
            Err(_) => self.register(primal_name, our_socket).await,
        }
    }

    /// Register this primal with the NUCLEUS orchestrator (legacy pattern).
    pub async fn register(
        &self,
        primal_name: &str,
        our_socket: &Path,
    ) -> Result<serde_json::Value> {
        ipc_client::call(
            &self.socket,
            "nucleus.register",
            &serde_json::json!({
                "name": primal_name,
                "socket_path": our_socket.to_string_lossy(),
                "pid": std::process::id(),
            }),
            self.timeout,
        )
        .await
    }

    /// Deregister this primal from the NUCLEUS orchestrator.
    pub async fn deregister(
        &self,
        primal_name: &str,
        our_socket: &Path,
    ) -> Result<serde_json::Value> {
        ipc_client::call(
            &self.socket,
            "nucleus.deregister",
            &serde_json::json!({
                "name": primal_name,
                "socket_path": our_socket.to_string_lossy(),
            }),
            self.timeout,
        )
        .await
    }

    /// Send a heartbeat to the NUCLEUS orchestrator.
    pub async fn heartbeat(
        &self,
        primal_name: &str,
        our_socket: &Path,
        status: &str,
    ) -> Result<serde_json::Value> {
        ipc_client::call(
            &self.socket,
            "nucleus.heartbeat",
            &serde_json::json!({
                "name": primal_name,
                "socket_path": our_socket.to_string_lossy(),
                "status": status,
            }),
            self.timeout,
        )
        .await
    }

    /// Register a single capability with the orchestrator.
    pub async fn register_capability(
        &self,
        primal_name: &str,
        capability: &str,
        our_socket: &Path,
    ) -> Result<serde_json::Value> {
        ipc_client::call(
            &self.socket,
            "capability.register",
            &serde_json::json!({
                "primal": primal_name,
                "capability": capability,
                "socket_path": our_socket.to_string_lossy(),
            }),
            self.timeout,
        )
        .await
    }

    /// Register all capabilities for a primal. Logs failures but does not
    /// bail — capability registration is best-effort during startup.
    pub async fn register_all_capabilities(
        &self,
        primal_name: &str,
        capabilities: &[&str],
        our_socket: &Path,
    ) {
        for cap in capabilities {
            if let Err(e) = self.register_capability(primal_name, cap, our_socket).await {
                log::warn!("capability.register({cap}) failed (non-fatal): {e}");
            }
        }
    }

    /// Resolve which primal provides a given capability.
    pub async fn resolve_capability(&self, capability: &str) -> Result<ResolveResult> {
        let result = ipc_client::call(
            &self.socket,
            "capability.resolve",
            &serde_json::json!({ "capability": capability }),
            self.timeout,
        )
        .await?;
        serde_json::from_value(result).context("parsing capability.resolve response")
    }

    /// The orchestrator socket path.
    #[must_use]
    pub fn socket_path(&self) -> &Path {
        &self.socket
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test temp paths are always valid UTF-8")]
mod tests {
    use super::*;

    #[test]
    fn discover_returns_none_when_no_orchestrator() {
        let missing = std::env::temp_dir().join("ns_biomeos_client_test_nonexistent");
        let missing_str = missing.to_str().expect("temp_dir is valid UTF-8");
        temp_env::with_var("BIOMEOS_SOCKET_DIR", Some(missing_str), || {
            assert!(BiomeOsClient::discover().is_none());
        });
    }

    #[test]
    fn new_sets_socket_path() {
        let sock = std::env::temp_dir().join("ns_test.sock");
        let client = BiomeOsClient::new(sock.clone());
        assert_eq!(client.socket_path(), sock.as_path());
    }
}
