// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed client for the coralReef sovereign shader compiler primal.
//!
//! Wraps [`crate::ipc_client`] with coralReef-specific JSON-RPC methods for
//! WGSL → native binary compilation and compiler status queries.
//!
//! coralReef compiles WGSL/SPIR-V shaders to native GPU binaries (NVIDIA SASS,
//! AMD GFX) and dispatches them via DRM ioctl, bypassing the wgpu/Vulkan stack.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::ipc_client;

/// Typed JSON-RPC client for the coralReef sovereign shader compiler primal.
pub struct CoralReefClient {
    /// Path to coralReef's Unix domain socket.
    socket: PathBuf,
    /// Per-RPC timeout (compile calls may be long-running).
    timeout: Duration,
}

/// Parsed `shader.compile.*` response: native binary metadata from the IPC reply.
#[derive(Debug, Deserialize)]
pub struct CompileResult {
    /// Size of the compiled native binary in bytes.
    #[serde(default)]
    pub binary_size: usize,
    /// Wall-clock compile time reported by the compiler (milliseconds).
    #[serde(default)]
    pub compile_time_ms: f64,
    /// GPU ISA or backend identifier (e.g. SASS, GFX).
    #[serde(default)]
    pub target_arch: String,
    /// General-purpose register usage reported for the kernel.
    #[serde(default)]
    pub gpr_count: u32,
    /// Static shared memory required by the kernel (bytes).
    #[serde(default)]
    pub shared_mem_bytes: u32,
    /// Launch grid local size (x, y, z) for the compiled kernel.
    #[serde(default)]
    pub local_size: [u32; 3],
    /// Base64-encoded native binary (for dispatch via coral-driver)
    #[serde(default)]
    pub binary_b64: String,
}

/// `shader.compile.capabilities` payload: supported targets and feature flags.
#[derive(Debug, Deserialize)]
pub struct CompilerCapabilities {
    /// NVIDIA target triples or chip names the compiler can emit.
    #[serde(default)]
    pub nvidia_targets: Vec<String>,
    /// AMD GFX / ISA targets the compiler can emit.
    #[serde(default)]
    pub amd_targets: Vec<String>,
    /// Whether double-precision is supported in generated kernels.
    #[serde(default)]
    pub supports_f64: bool,
    /// Whether SPIR-V ingestion is supported.
    #[serde(default)]
    pub supports_spirv: bool,
    /// Fused multiply-add policy string from the compiler.
    #[serde(default)]
    pub fma_policy: String,
}

/// `shader.compile.status` snapshot: uptime and cache statistics.
#[derive(Debug, Deserialize)]
pub struct CompilerStatus {
    /// High-level compiler state (e.g. ready, busy).
    #[serde(default)]
    pub status: String,
    /// Compiler build or protocol version string.
    #[serde(default)]
    pub version: String,
    /// Total shaders compiled since process start (if reported).
    #[serde(default)]
    pub shaders_compiled: u64,
    /// Number of entries in the on-disk or in-memory shader cache.
    #[serde(default)]
    pub cache_entries: u64,
}

impl CoralReefClient {
    /// Connect to coralReef via capability-based discovery.
    ///
    /// Discovers the sovereign shader compiler by probing for the
    /// `shader.compile.wgsl` capability on all sockets in the biomeOS
    /// directory.  Falls back to name-based discovery (`coralreef`) if no
    /// capability probe succeeds.
    #[allow(deprecated)]
    pub fn discover() -> Result<Self> {
        let socket = ipc_client::discover_by_capability(
            "shader.compile.wgsl",
            neural_spring::primal_names::CORALREEF,
        )
        .context("discovering shader compiler primal")?;
        Ok(Self {
            socket,
            timeout: Duration::from_secs(60),
        })
    }

    /// Connect to coralReef at a specific socket path.
    #[must_use]
    pub const fn new(socket: PathBuf) -> Self {
        Self {
            socket,
            timeout: Duration::from_secs(60),
        }
    }

    /// Compile a single WGSL shader to a native GPU binary.
    ///
    /// Returns the compiled binary metadata and base64-encoded bytes.
    pub async fn compile_wgsl(
        &self,
        source: &str,
        entry_point: &str,
        target: Option<&str>,
    ) -> Result<CompileResult> {
        let mut params = serde_json::json!({
            "source": source,
            "entry_point": entry_point,
        });
        if let Some(t) = target {
            params["target"] = serde_json::Value::String(t.to_owned());
        }
        let result =
            ipc_client::call(&self.socket, "shader.compile.wgsl", &params, self.timeout).await?;
        let cr: CompileResult = serde_json::from_value(result)?;
        Ok(cr)
    }

    /// Compile a WGSL shader for multiple targets simultaneously.
    pub async fn compile_wgsl_multi(
        &self,
        source: &str,
        entry_point: &str,
        targets: &[&str],
    ) -> Result<Vec<CompileResult>> {
        let params = serde_json::json!({
            "source": source,
            "entry_point": entry_point,
            "targets": targets,
        });
        let result = ipc_client::call(
            &self.socket,
            "shader.compile.wgsl.multi",
            &params,
            self.timeout,
        )
        .await?;
        let results: Vec<CompileResult> = serde_json::from_value(result)?;
        Ok(results)
    }

    /// Compile a SPIR-V binary to native GPU code.
    pub async fn compile_spirv(
        &self,
        spirv_b64: &str,
        target: Option<&str>,
    ) -> Result<CompileResult> {
        let mut params = serde_json::json!({ "spirv_b64": spirv_b64 });
        if let Some(t) = target {
            params["target"] = serde_json::Value::String(t.to_owned());
        }
        let result =
            ipc_client::call(&self.socket, "shader.compile.spirv", &params, self.timeout).await?;
        let cr: CompileResult = serde_json::from_value(result)?;
        Ok(cr)
    }

    /// Query compiler status via `shader.compile.status`.
    pub async fn status(&self) -> Result<CompilerStatus> {
        let result = ipc_client::call(
            &self.socket,
            "shader.compile.status",
            &serde_json::json!({}),
            self.timeout,
        )
        .await?;
        let status: CompilerStatus = serde_json::from_value(result)?;
        Ok(status)
    }

    /// Query compiler capabilities via `shader.compile.capabilities`.
    pub async fn capabilities(&self) -> Result<CompilerCapabilities> {
        let result = ipc_client::call(
            &self.socket,
            "shader.compile.capabilities",
            &serde_json::json!({}),
            self.timeout,
        )
        .await?;
        let caps: CompilerCapabilities = serde_json::from_value(result)?;
        Ok(caps)
    }
}
