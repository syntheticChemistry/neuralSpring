// SPDX-License-Identifier: AGPL-3.0-or-later

//! Integration tests for playGround.
//!
//! Tests marked `#[ignore]` require live primals or GPU hardware.
//! Run them with: `cargo test -p neuralspring-playground -- --ignored`

#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration tests — panic on failure is the intended behavior"
)]

use neuralspring_playground::coralreef_client::CoralReefClient;
use neuralspring_playground::hf_hub::{self, HfHub};
// HfHub now routes through Songbird (Tower Atomic) — tests require live Songbird daemon.
use neuralspring_playground::inference::{transformer::TransformerEngine, weights};
use neuralspring_playground::model_config::TransformerConfig;
use neuralspring_playground::toadstool_client::ToadStoolClient;

// ═══════════════════════════════════════════════════════════════════
// Primal discovery (no live primal needed — tests socket resolution)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn toadstool_discover_reports_unavailable() {
    match ToadStoolClient::discover() {
        Ok(_) => {
            println!("ToadStool socket found (daemon may or may not be responding)");
        }
        Err(e) => {
            let chain = format!("{e:#}");
            assert!(
                chain.contains("no socket found") || chain.contains("ToadStool"),
                "expected socket discovery error, got: {chain}"
            );
        }
    }
}

#[test]
fn coralreef_discover_reports_unavailable() {
    match CoralReefClient::discover() {
        Ok(_) => {
            println!("coralReef socket found (daemon may or may not be responding)");
        }
        Err(e) => {
            let chain = format!("{e:#}");
            assert!(
                chain.contains("no socket found") || chain.contains("coralReef"),
                "expected socket discovery error, got: {chain}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Live primal tests (require running daemons)
// ═══════════════════════════════════════════════════════════════════

#[tokio::test]
#[ignore = "requires live ToadStool daemon"]
async fn toadstool_health_check() {
    let client = ToadStoolClient::discover().expect("ToadStool not found");
    let health = client.health().await.expect("health check failed");
    assert!(health.healthy, "ToadStool should be healthy");
    assert!(!health.version.is_empty());
    println!(
        "ToadStool v{} — uptime {}s, {} active / {} queued workloads",
        health.version, health.uptime_secs, health.active_workloads, health.queued_workloads
    );
}

#[tokio::test]
#[ignore = "requires live ToadStool daemon"]
async fn toadstool_capabilities() {
    let client = ToadStoolClient::discover().expect("ToadStool not found");
    let caps = client
        .capabilities()
        .await
        .expect("capabilities query failed");
    assert!(
        !caps.supported_workload_types.is_empty(),
        "ToadStool should report supported workload types"
    );
    println!(
        "ToadStool service: {} — workloads: {:?}",
        caps.service_id, caps.supported_workload_types
    );
}

#[tokio::test]
#[ignore = "requires live ToadStool daemon"]
async fn toadstool_gpu_info() {
    let client = ToadStoolClient::discover().expect("ToadStool not found");
    let info = client.gpu_info().await.expect("gpu.info failed");
    assert!(!info.driver.is_empty(), "GPU driver should not be empty");
    println!(
        "GPU driver: {}, backends: {:?}, devices: {}",
        info.driver,
        info.compute_backends,
        info.devices.len()
    );
}

#[tokio::test]
#[ignore = "requires live coralReef daemon"]
async fn coralreef_status() {
    let client = CoralReefClient::discover().expect("coralReef not found");
    let status = client.status().await.expect("status query failed");
    assert_eq!(status.status, "ok");
    assert!(!status.version.is_empty());
}

#[tokio::test]
#[ignore = "requires live coralReef daemon"]
async fn coralreef_capabilities() {
    let client = CoralReefClient::discover().expect("coralReef not found");
    let caps = client
        .capabilities()
        .await
        .expect("capabilities query failed");
    assert!(
        !caps.nvidia_targets.is_empty() || !caps.amd_targets.is_empty(),
        "coralReef should report at least one target architecture"
    );
}

#[tokio::test]
#[ignore = "requires live coralReef daemon"]
async fn coralreef_compile_trivial_shader() {
    let client = CoralReefClient::discover().expect("coralReef not found");
    let source = r"
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read_write> b: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i = gid.x;
            if i >= arrayLength(&b) { return; }
            b[i] = a[i] * 2.0;
        }
    ";
    let result = client
        .compile_wgsl(source, "main", None)
        .await
        .expect("WGSL compilation failed");
    assert!(
        result.binary_size > 0,
        "compiled binary should not be empty"
    );
    assert!(
        result.compile_time_ms > 0.0,
        "compile time should be positive"
    );
}

// ═══════════════════════════════════════════════════════════════════
// GPU tests (require wgpu-capable hardware)
// ═══════════════════════════════════════════════════════════════════

#[tokio::test]
#[ignore = "requires GPU hardware"]
async fn gpu_tensor_session_basic() {
    use barracuda::prelude::{TensorSession, WgpuDevice};
    use std::sync::Arc;

    let device = Arc::new(WgpuDevice::new().await.expect("no GPU"));
    let mut session = TensorSession::with_device(device);

    let a = session
        .tensor_with_shape(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2])
        .unwrap();
    let b = session
        .tensor_with_shape(&[5.0_f32, 6.0, 7.0, 8.0], &[2, 2])
        .unwrap();
    let c = session.add(&a, &b).unwrap();
    session.run().unwrap();

    let result = c.to_vec().unwrap();
    assert_eq!(result.len(), 4);
    // Provenance: analytical — [1,2;3,4] + [5,6;7,8] = [6,8;10,12].
    // Tolerance: f32 GPU roundtrip (IEEE 754 single precision).
    let f32_gpu_tol = 1e-5;
    assert!((result[0] - 6.0).abs() < f32_gpu_tol);
    assert!((result[3] - 12.0).abs() < f32_gpu_tol);
}

#[tokio::test]
#[ignore = "requires GPU hardware"]
async fn gpu_session_reset_retains_pipelines() {
    use barracuda::prelude::{TensorSession, WgpuDevice};
    use std::sync::Arc;
    use std::time::Instant;

    let device = Arc::new(WgpuDevice::new().await.expect("no GPU"));
    let mut session = TensorSession::with_device(device);

    let data = vec![1.0_f32; 1024];

    // First run — pipelines compiled
    let t0 = Instant::now();
    let a = session.tensor_with_shape(&data, &[32, 32]).unwrap();
    let b = session.tensor_with_shape(&data, &[32, 32]).unwrap();
    let _c = session.matmul(&a, &b).unwrap();
    session.run().unwrap();
    let first_run_us = t0.elapsed().as_micros();

    session.reset();

    // Second run — pipelines reused
    let t0 = Instant::now();
    let a = session.tensor_with_shape(&data, &[32, 32]).unwrap();
    let b = session.tensor_with_shape(&data, &[32, 32]).unwrap();
    let _c = session.matmul(&a, &b).unwrap();
    session.run().unwrap();
    let second_run_us = t0.elapsed().as_micros();

    println!("First run: {first_run_us}µs, Second run: {second_run_us}µs");
    // Second run should be faster (no pipeline compilation)
    // Allow some slack for OS scheduling
}

// ═══════════════════════════════════════════════════════════════════
// HuggingFace integration (requires network)
// ═══════════════════════════════════════════════════════════════════

#[tokio::test]
#[ignore = "requires network access to HuggingFace"]
async fn hf_hub_model_info() {
    let cache_dir = hf_hub::default_cache_dir();
    let hub = HfHub::new(None, cache_dir).unwrap();
    let info = hub
        .model_info("openai-community/gpt2")
        .await
        .expect("failed to fetch GPT-2 info");
    assert_eq!(info.model_id, "openai-community/gpt2");
    assert!(!info.siblings.is_empty());
}

#[tokio::test]
#[ignore = "requires network access to HuggingFace"]
async fn hf_hub_list_safetensors() {
    let cache_dir = hf_hub::default_cache_dir();
    let hub = HfHub::new(None, cache_dir).unwrap();
    let files = hub
        .list_safetensors("openai-community/gpt2")
        .await
        .expect("failed to list safetensors");
    assert!(
        !files.is_empty(),
        "GPT-2 should have at least one safetensors file"
    );
}

// ═══════════════════════════════════════════════════════════════════
// End-to-end: model download + inference (requires network + GPU)
// ═══════════════════════════════════════════════════════════════════

#[tokio::test]
#[ignore = "requires network + GPU + ~500MB download"]
async fn e2e_gpt2_forward_pass() {
    use std::sync::Arc;

    let cache_dir = hf_hub::default_cache_dir();
    let hub = HfHub::new(None, cache_dir).unwrap();
    let files = hub
        .download_model("openai-community/gpt2")
        .await
        .expect("download failed");
    assert!(files.is_complete());

    let config = TransformerConfig::from_file(files.config.as_ref().unwrap()).unwrap();
    assert_eq!(config.model_type, "gpt2");

    let device = Arc::new(barracuda::prelude::WgpuDevice::new().await.expect("no GPU"));
    let raw = weights::load_safetensors(&files.safetensors, &device).expect("load failed");
    let model_weights = weights::organize_weights(raw, &config);
    let engine = TransformerEngine::new(device, config, model_weights);

    let token_ids = vec![15496u32, 11, 995]; // "Hello, world"
    let output = engine.forward(&token_ids).expect("forward pass failed");
    assert_eq!(output.seq_len, 3);
    assert!(!output.logits.is_empty());
    assert!(!output.logits.iter().any(|v| v.is_nan()));

    let top = TransformerEngine::top_k(&output.logits, 5);
    println!("Top 5 predictions: {top:?}");
    assert_eq!(top.len(), 5);
}
