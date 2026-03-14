// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed client for the `ToadStool` compute primal.
//!
//! Wraps [`crate::ipc_client`] with `ToadStool`-specific JSON-RPC methods for
//! compute submission, GPU discovery, and workload management.
//!
//! `ToadStool` is the compute orchestration layer of the ecoPrimals compute
//! triangle (`coralReef` → `ToadStool` → `barraCuda`).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::ipc_client;

pub struct ToadStoolClient {
    socket: PathBuf,
    timeout: Duration,
}

#[derive(Debug, Deserialize)]
pub struct GpuInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub vendor: String,
    #[serde(default)]
    pub driver: String,
    #[serde(default)]
    pub memory_mb: u64,
    #[serde(default)]
    pub compute_units: u32,
    #[serde(default)]
    pub supports_f64: bool,
}

#[derive(Debug, Deserialize)]
pub struct ComputeCapabilities {
    #[serde(default)]
    pub gpu_available: bool,
    #[serde(default)]
    pub npu_available: bool,
    #[serde(default)]
    pub substrates: Vec<String>,
    #[serde(default)]
    pub shader_compiler: bool,
}

#[derive(Debug, Deserialize)]
pub struct JobStatus {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub progress: f64,
    #[serde(default)]
    pub elapsed_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct JobResult {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub result: Option<serde_json::Value>,
    #[serde(default)]
    pub elapsed_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct HealthStatus {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub uptime_secs: u64,
    #[serde(default)]
    pub version: String,
}

impl ToadStoolClient {
    /// Connect to `ToadStool` via automatic socket discovery.
    pub fn discover() -> Result<Self> {
        let socket =
            ipc_client::discover_socket("toadstool").context("discovering ToadStool socket")?;
        Ok(Self {
            socket,
            timeout: Duration::from_secs(30),
        })
    }

    /// Connect to `ToadStool` at a specific socket path.
    #[must_use]
    pub fn new(socket: PathBuf) -> Self {
        Self {
            socket,
            timeout: Duration::from_secs(30),
        }
    }

    /// Query GPU hardware info via `gpu.info`.
    pub async fn gpu_info(&self) -> Result<GpuInfo> {
        let result = ipc_client::call(
            &self.socket,
            "gpu.info",
            &serde_json::json!({}),
            self.timeout,
        )
        .await?;
        let info: GpuInfo = serde_json::from_value(result)?;
        Ok(info)
    }

    /// Query GPU memory via `gpu.memory`.
    pub async fn gpu_memory(&self) -> Result<serde_json::Value> {
        ipc_client::call(
            &self.socket,
            "gpu.memory",
            &serde_json::json!({}),
            self.timeout,
        )
        .await
    }

    /// Query compute capabilities via `compute.capabilities`.
    pub async fn capabilities(&self) -> Result<ComputeCapabilities> {
        let result = ipc_client::call(
            &self.socket,
            "compute.capabilities",
            &serde_json::json!({}),
            self.timeout,
        )
        .await?;
        let caps: ComputeCapabilities = serde_json::from_value(result)?;
        Ok(caps)
    }

    /// Submit a compute job via `compute.submit`.
    ///
    /// Returns the job ID for status polling.
    pub async fn submit(&self, workload_type: &str, params: &serde_json::Value) -> Result<String> {
        let submit_params = serde_json::json!({
            "workload_type": workload_type,
            "params": params,
        });
        let result =
            ipc_client::call(&self.socket, "compute.submit", &submit_params, self.timeout).await?;
        let id = result["id"].as_str().unwrap_or_default().to_owned();
        Ok(id)
    }

    /// Poll job status via `compute.status`.
    pub async fn status(&self, job_id: &str) -> Result<JobStatus> {
        let params = serde_json::json!({ "id": job_id });
        let result =
            ipc_client::call(&self.socket, "compute.status", &params, self.timeout).await?;
        let status: JobStatus = serde_json::from_value(result)?;
        Ok(status)
    }

    /// Retrieve job result via `compute.result`.
    pub async fn result(&self, job_id: &str) -> Result<JobResult> {
        let params = serde_json::json!({ "id": job_id });
        let result =
            ipc_client::call(&self.socket, "compute.result", &params, self.timeout).await?;
        let jr: JobResult = serde_json::from_value(result)?;
        Ok(jr)
    }

    /// Cancel a job via `compute.cancel`.
    pub async fn cancel(&self, job_id: &str) -> Result<serde_json::Value> {
        let params = serde_json::json!({ "id": job_id });
        ipc_client::call(&self.socket, "compute.cancel", &params, self.timeout).await
    }

    /// List active jobs via `compute.list`.
    pub async fn list_jobs(&self) -> Result<serde_json::Value> {
        ipc_client::call(
            &self.socket,
            "compute.list",
            &serde_json::json!({}),
            self.timeout,
        )
        .await
    }

    /// Dispatch a GPU workload via `science.gpu.dispatch`.
    pub async fn gpu_dispatch(
        &self,
        shader_source: &str,
        workgroup_dims: [u32; 3],
        buffers: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let params = serde_json::json!({
            "shader": shader_source,
            "workgroup_dims": workgroup_dims,
            "buffers": buffers,
        });
        ipc_client::call(&self.socket, "science.gpu.dispatch", &params, self.timeout).await
    }

    /// Discover available substrates via `science.substrate.discover`.
    pub async fn discover_substrates(&self) -> Result<serde_json::Value> {
        ipc_client::call(
            &self.socket,
            "science.substrate.discover",
            &serde_json::json!({}),
            self.timeout,
        )
        .await
    }

    /// Health check via `toadstool.health`.
    pub async fn health(&self) -> Result<HealthStatus> {
        let result = ipc_client::call(
            &self.socket,
            "toadstool.health",
            &serde_json::json!({}),
            self.timeout,
        )
        .await?;
        let status: HealthStatus = serde_json::from_value(result)?;
        Ok(status)
    }
}
