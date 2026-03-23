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

/// Typed JSON-RPC client for the `ToadStool` compute orchestration primal.
pub struct ToadStoolClient {
    /// Path to `ToadStool`'s Unix domain socket.
    socket: PathBuf,
    /// Per-RPC timeout for compute and GPU IPC calls.
    timeout: Duration,
}

/// `gpu.info` payload: devices and driver/backends visible to `ToadStool`.
#[derive(Debug, Deserialize)]
pub struct GpuInfo {
    /// Discovered GPU devices (ids, backends, ordering).
    #[serde(default)]
    pub devices: Vec<GpuDevice>,
    /// Driver or runtime version string from the GPU stack.
    #[serde(default)]
    pub driver: String,
    /// Backend identifiers available for compute (e.g. wgpu, CUDA).
    #[serde(default)]
    pub compute_backends: Vec<String>,
}

/// One entry in the `gpu.info` device list.
#[derive(Debug, Deserialize)]
pub struct GpuDevice {
    /// Stable device id or name from the primal.
    #[serde(default)]
    pub id: String,
    /// Backend used for this device (e.g. Vulkan, CUDA).
    #[serde(default)]
    pub backend: String,
    /// Ordinal index within the backend enumeration.
    #[serde(default)]
    pub index: u32,
}

/// `compute.capabilities` reply: supported workload kinds and scheduling metadata.
#[derive(Debug, Deserialize)]
pub struct ComputeCapabilities {
    /// Service or primal instance id for this endpoint.
    #[serde(default)]
    pub service_id: String,
    /// Workload type strings accepted by `compute.submit`.
    #[serde(default)]
    pub supported_workload_types: Vec<String>,
    /// Opaque per-unit descriptors (scheduler-dependent JSON).
    #[serde(default)]
    pub compute_units: Vec<serde_json::Value>,
}

/// `compute.status` poll result for a submitted job.
#[derive(Debug, Deserialize)]
pub struct JobStatus {
    /// Job id returned by `compute.submit`.
    #[serde(default)]
    pub id: String,
    /// Coarse lifecycle state (queued, running, completed, failed, …).
    #[serde(default)]
    pub status: String,
    /// Normalized progress in \[0, 1\] when reported.
    #[serde(default)]
    pub progress: f64,
    /// Elapsed wall time since submission (milliseconds).
    #[serde(default)]
    pub elapsed_ms: u64,
}

/// `compute.result` payload when a job finishes.
#[derive(Debug, Deserialize)]
pub struct JobResult {
    /// Job id matching `compute.submit` / status polls.
    #[serde(default)]
    pub id: String,
    /// Final lifecycle state string.
    #[serde(default)]
    pub status: String,
    /// Structured result payload from the workload when successful.
    #[serde(default)]
    pub result: Option<serde_json::Value>,
    /// Total elapsed time for the job (milliseconds).
    #[serde(default)]
    pub elapsed_ms: u64,
}

/// `toadstool.health` summary for orchestrator observability.
#[derive(Debug, Deserialize)]
pub struct HealthStatus {
    /// Whether the primal considers itself healthy.
    #[serde(default)]
    pub healthy: bool,
    /// Process uptime in seconds.
    #[serde(default)]
    pub uptime_secs: u64,
    /// Build or protocol version string.
    #[serde(default)]
    pub version: String,
    /// Number of in-flight workloads currently executing.
    #[serde(default)]
    pub active_workloads: u32,
    /// Number of workloads waiting in the scheduler queue.
    #[serde(default)]
    pub queued_workloads: u32,
}

impl ToadStoolClient {
    /// Connect to `ToadStool` via capability-based discovery.
    ///
    /// Discovers the compute orchestration primal by probing for the
    /// `compute.submit` capability on all sockets in the biomeOS directory.
    /// Falls back to name-based discovery (`toadstool`) if no capability
    /// probe succeeds.
    pub fn discover() -> Result<Self> {
        let socket = ipc_client::discover_by_capability(
            "compute.submit",
            neural_spring::config::TOADSTOOL_NAME_HINT,
        )
        .context("discovering compute orchestration primal")?;
        Ok(Self {
            socket,
            timeout: Duration::from_secs(30),
        })
    }

    /// Connect to `ToadStool` at a specific socket path.
    #[must_use]
    pub const fn new(socket: PathBuf) -> Self {
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

    // ═══════════════════════════════════════════════════════════════
    // compute.dispatch protocol (wetSpring V124 / healthSpring V31)
    // ═══════════════════════════════════════════════════════════════

    /// Submit a dispatch operation via `compute.dispatch.submit`.
    ///
    /// Returns a [`DispatchHandle`] for retrieving the result.
    pub async fn dispatch_submit(
        &self,
        operation: &str,
        input: &serde_json::Value,
    ) -> Result<DispatchHandle> {
        let params = serde_json::json!({
            "operation": operation,
            "input": input,
        });
        let result = ipc_client::call(
            &self.socket,
            "compute.dispatch.submit",
            &params,
            self.timeout,
        )
        .await?;
        let handle: DispatchHandle = serde_json::from_value(result)?;
        Ok(handle)
    }

    /// Retrieve the result of a dispatch operation via `compute.dispatch.result`.
    pub async fn dispatch_result(&self, dispatch_id: &str) -> Result<DispatchResult> {
        let params = serde_json::json!({ "dispatch_id": dispatch_id });
        let result = ipc_client::call(
            &self.socket,
            "compute.dispatch.result",
            &params,
            self.timeout,
        )
        .await?;
        let dr: DispatchResult = serde_json::from_value(result)?;
        Ok(dr)
    }

    /// Query available dispatch capabilities via `compute.dispatch.capabilities`.
    pub async fn dispatch_capabilities(&self) -> Result<Vec<String>> {
        let result = ipc_client::call(
            &self.socket,
            "compute.dispatch.capabilities",
            &serde_json::json!({}),
            self.timeout,
        )
        .await?;
        let caps: Vec<String> = serde_json::from_value(result)?;
        Ok(caps)
    }
}

/// Handle returned by `compute.dispatch.submit` for result retrieval.
#[derive(Debug, Deserialize)]
pub struct DispatchHandle {
    /// Opaque id for `compute.dispatch.result` polling.
    pub dispatch_id: String,
    /// Initial or last-known dispatch state from the submit response.
    #[serde(default)]
    pub status: String,
}

/// Result of a completed dispatch operation.
#[derive(Debug, Deserialize)]
pub struct DispatchResult {
    /// Dispatch id matching the submit handle.
    #[serde(default)]
    pub dispatch_id: String,
    /// Terminal state of the dispatch (success, error, cancelled, …).
    #[serde(default)]
    pub status: String,
    /// Structured output from the dispatched operation when available.
    #[serde(default)]
    pub output: Option<serde_json::Value>,
    /// Wall time for the dispatch (milliseconds).
    #[serde(default)]
    pub elapsed_ms: u64,
}
