// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed client for the biomeOS orchestrator.
//!
//! Wraps [`crate::ipc_client`] with biomeOS lifecycle methods:
//! `nucleus.register`, `nucleus.deregister`, `nucleus.heartbeat`,
//! `capability.register`, and `capability.resolve`.
//!
//! Replaces ad-hoc `forward_to_primal_raw` calls with a proper typed
//! interface and consistent request ID generation.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::ipc_client;

pub struct BiomeOsClient {
    socket: PathBuf,
    timeout: Duration,
}

#[derive(Debug, Deserialize)]
pub struct ResolveResult {
    #[serde(default)]
    pub primal: String,
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

    /// Register this primal with the NUCLEUS orchestrator.
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
mod tests {
    use super::*;

    #[test]
    fn discover_returns_none_when_no_orchestrator() {
        temp_env::with_var(
            "BIOMEOS_SOCKET_DIR",
            Some("/tmp/biomeos_client_test_nonexistent"),
            || {
                assert!(BiomeOsClient::discover().is_none());
            },
        );
    }

    #[test]
    fn new_sets_socket_path() {
        let client = BiomeOsClient::new(PathBuf::from("/tmp/test.sock"));
        assert_eq!(client.socket_path(), Path::new("/tmp/test.sock"));
    }
}
