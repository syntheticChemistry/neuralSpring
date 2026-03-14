// SPDX-License-Identifier: AGPL-3.0-or-later

//! Probe the ecoPrimals compute triangle: barraCuda (local), ToadStool, coralReef.
//!
//! Reports availability, capabilities, and latency of each tier:
//! - Tier 0: barraCuda (direct wgpu GPU access)
//! - Tier 1: ToadStool (compute orchestration via IPC)
//! - Tier 2: coralReef (sovereign shader compiler via IPC)
//!
//! Use this to understand what compute paths are available before benchmarking.

#![expect(clippy::pedantic, reason = "diagnostic binary")]

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use barracuda::prelude::{TensorSession, WgpuDevice};

use neuralspring_playground::coralreef_client::CoralReefClient;
use neuralspring_playground::toadstool_client::ToadStoolClient;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("\n{}", "=".repeat(60));
    println!("ecoPrimals Compute Triangle — Probe");
    println!("{}\n", "=".repeat(60));

    // ── Tier 0: barraCuda (direct GPU) ──────────────────────────
    println!("Tier 0: barraCuda (local wgpu)");
    match WgpuDevice::new().await {
        Ok(device) => {
            let info = device.adapter_info();
            println!("  Status:  AVAILABLE");
            println!("  GPU:     {} ({:?})", info.name, info.backend);

            let device = Arc::new(device);
            let t0 = Instant::now();
            let mut session = TensorSession::with_device(device.clone());
            let pipeline_ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!("  Pipelines: {pipeline_ms:.1}ms (17 shaders compiled)");

            let data: Vec<f32> = vec![1.0; 1024];
            let t0 = Instant::now();
            let a = session.tensor_with_shape(&data, &[32, 32])?;
            let b = session.tensor_with_shape(&data, &[32, 32])?;
            let _c = session.matmul(&a, &b)?;
            session.run()?;
            let dispatch_ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!("  Dispatch:  {dispatch_ms:.1}ms (matmul 32x32, first call)");

            session.reset();
            let t0 = Instant::now();
            let a = session.tensor_with_shape(&data, &[32, 32])?;
            let b = session.tensor_with_shape(&data, &[32, 32])?;
            let _c = session.matmul(&a, &b)?;
            session.run()?;
            let dispatch2_ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!("  Dispatch:  {dispatch2_ms:.1}ms (matmul 32x32, second call / hot)");
        }
        Err(e) => {
            println!("  Status:  UNAVAILABLE");
            println!("  Error:   {e}");
        }
    }

    println!();

    // ── Tier 1: ToadStool (compute orchestration) ───────────────
    println!("Tier 1: ToadStool (compute orchestration)");
    match ToadStoolClient::discover() {
        Ok(client) => {
            let t0 = Instant::now();
            match client.health().await {
                Ok(health) => {
                    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    println!("  Status:  AVAILABLE");
                    println!("  Version: {}", health.version);
                    println!("  Uptime:  {}s", health.uptime_secs);
                    println!("  Latency: {latency_ms:.1}ms (health check)");

                    if let Ok(caps) = client.capabilities().await {
                        println!("  GPU:     {}", caps.gpu_available);
                        println!("  NPU:     {}", caps.npu_available);
                        println!("  Shader:  {}", caps.shader_compiler);
                        if !caps.substrates.is_empty() {
                            println!("  Substrates: {}", caps.substrates.join(", "));
                        }
                    }
                }
                Err(e) => {
                    println!("  Status:  SOCKET FOUND, NOT RESPONDING");
                    println!("  Error:   {e}");
                }
            }
        }
        Err(_) => {
            println!("  Status:  NOT DISCOVERED");
            println!("  Hint:    Start ToadStool or set BIOMEOS_SOCKET_DIR");
        }
    }

    println!();

    // ── Tier 2: coralReef (sovereign shader compiler) ───────────
    println!("Tier 2: coralReef (sovereign shader compiler)");
    match CoralReefClient::discover() {
        Ok(client) => {
            let t0 = Instant::now();
            match client.status().await {
                Ok(status) => {
                    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    println!("  Status:  AVAILABLE");
                    println!("  Version: {}", status.version);
                    println!("  Compiled: {} shaders", status.shaders_compiled);
                    println!("  Cache:   {} entries", status.cache_entries);
                    println!("  Latency: {latency_ms:.1}ms (status check)");

                    if let Ok(caps) = client.capabilities().await {
                        if !caps.nvidia_targets.is_empty() {
                            println!("  NVIDIA:  {}", caps.nvidia_targets.join(", "));
                        }
                        if !caps.amd_targets.is_empty() {
                            println!("  AMD:     {}", caps.amd_targets.join(", "));
                        }
                        println!("  f64:     {}", caps.supports_f64);
                        println!("  FMA:     {}", caps.fma_policy);
                    }
                }
                Err(e) => {
                    println!("  Status:  SOCKET FOUND, NOT RESPONDING");
                    println!("  Error:   {e}");
                }
            }
        }
        Err(_) => {
            println!("  Status:  NOT DISCOVERED");
            println!("  Hint:    Start coralReef or set BIOMEOS_SOCKET_DIR");
        }
    }

    println!();

    // ── Summary ─────────────────────────────────────────────────
    println!("{}", "-".repeat(60));
    println!("Compute paths available for playGround benchmarks:");
    println!("  --hot      Reuse TensorSession (Tier 0, pre-compiled pipelines)");
    println!("  (default)  Cold dispatch (Tier 0, per-call pipeline creation)");
    println!("  --toadstool  Route via ToadStool (Tier 1, when available)");
    println!("  --sovereign  coralReef native binary (Tier 2, when available)");
    println!();

    Ok(())
}
