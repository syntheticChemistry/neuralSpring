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

pub struct CoralReefClient {
    socket: PathBuf,
    timeout: Duration,
}

#[derive(Debug, Deserialize)]
pub struct CompileResult {
    #[serde(default)]
    pub binary_size: usize,
    #[serde(default)]
    pub compile_time_ms: f64,
    #[serde(default)]
    pub target_arch: String,
    #[serde(default)]
    pub gpr_count: u32,
    #[serde(default)]
    pub shared_mem_bytes: u32,
    #[serde(default)]
    pub local_size: [u32; 3],
    /// Base64-encoded native binary (for dispatch via coral-driver)
    #[serde(default)]
    pub binary_b64: String,
}

#[derive(Debug, Deserialize)]
pub struct CompilerCapabilities {
    #[serde(default)]
    pub nvidia_targets: Vec<String>,
    #[serde(default)]
    pub amd_targets: Vec<String>,
    #[serde(default)]
    pub supports_f64: bool,
    #[serde(default)]
    pub supports_spirv: bool,
    #[serde(default)]
    pub fma_policy: String,
}

#[derive(Debug, Deserialize)]
pub struct CompilerStatus {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub shaders_compiled: u64,
    #[serde(default)]
    pub cache_entries: u64,
}

impl CoralReefClient {
    /// Connect to coralReef via automatic socket discovery.
    pub fn discover() -> Result<Self> {
        let socket =
            ipc_client::discover_socket("coralreef").context("discovering coralReef socket")?;
        Ok(Self {
            socket,
            timeout: Duration::from_secs(60),
        })
    }

    /// Connect to coralReef at a specific socket path.
    #[must_use]
    pub fn new(socket: PathBuf) -> Self {
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
