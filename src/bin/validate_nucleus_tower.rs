// SPDX-License-Identifier: AGPL-3.0-or-later

//! NUCLEUS Tower Mode Validator
//!
//! Validates the expanded neuralspring_primal with:
//!   1. GPU-aware health check (hardware info, uptime, counters)
//!   2. Evoformer block RPC (folding pipeline via JSON-RPC)
//!   3. Structure Module RPC (IPA + backbone + torsion)
//!   4. Folding health report
//!   5. GPU dispatch operations (mat_mul, softmax, eigh)
//!   6. Cross-primal discovery (graceful failure without NestGate)
//!   7. Concurrent request handling
//!   8. Error handling for unknown methods

#![allow(
    clippy::pedantic,
    clippy::nursery,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::time::sleep;

use neural_spring::validation::ValidationHarness;

// ═══════════════════════════════════════════════════════════════════════════════
// JSON-RPC helpers
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    method: String,
    params: serde_json::Value,
    id: u64,
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[serde(rename = "jsonrpc")]
    _jsonrpc: String,
    result: Option<serde_json::Value>,
    error: Option<serde_json::Value>,
    #[serde(rename = "id")]
    _id: u64,
}

async fn rpc_call(
    socket: &PathBuf,
    method: &str,
    params: serde_json::Value,
    id: u64,
) -> Result<serde_json::Value> {
    let stream = UnixStream::connect(socket)
        .await
        .context("connecting to neuralspring socket")?;

    let (reader, mut writer) = stream.into_split();

    let req = JsonRpcRequest {
        jsonrpc: "2.0",
        method: method.to_string(),
        params,
        id,
    };

    let req_json = serde_json::to_vec(&req)?;
    writer.write_all(&req_json).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    writer.shutdown().await?;

    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();
    buf_reader.read_line(&mut line).await?;

    let resp: JsonRpcResponse = serde_json::from_str(line.trim())?;

    if let Some(err) = resp.error {
        anyhow::bail!("RPC error: {err}");
    }

    resp.result
        .ok_or_else(|| anyhow::anyhow!("no result in response"))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("NUCLEUS Tower Mode Integration");

    let socket_dir = std::env::temp_dir().join("biomeos-tower-test");
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");
    let socket_path = socket_dir.join("neuralspring-test.sock");

    if socket_path.exists() {
        std::fs::remove_file(&socket_path).expect("remove stale socket");
    }

    let primal_bin = std::env::current_exe()
        .expect("current_exe")
        .parent()
        .expect("exe parent")
        .join("neuralspring_primal");

    if !primal_bin.exists() {
        eprintln!(
            "  neuralspring_primal binary not found at {}",
            primal_bin.display()
        );
        eprintln!("  Build with: cargo build --features primal --bin neuralspring_primal");
        eprintln!("  Skipping NUCLEUS Tower tests (primal binary not available)");
        h.finish();
    }

    let mut child = tokio::process::Command::new(&primal_bin)
        .env("BIOMEOS_SOCKET_DIR", &socket_dir)
        .env("FAMILY_ID", "test")
        .env("NEURALSPRING_BACKEND", "cpu")
        .kill_on_drop(true)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawning neuralspring_primal");

    // Wait for socket with Dispatcher init time
    let mut retries = 0;
    while !socket_path.exists() && retries < 100 {
        sleep(Duration::from_millis(100)).await;
        retries += 1;
    }

    if !socket_path.exists() {
        eprintln!("  Socket did not appear after 10s. Primal may have failed to start.");
        child.kill().await.ok();
        h.finish();
    }

    sleep(Duration::from_millis(500)).await;

    // ═══════════════════════════════════════════════════════════════════
    // Test 1-2: GPU-aware health check
    // ═══════════════════════════════════════════════════════════════════
    let health = rpc_call(&socket_path, "health", serde_json::json!({}), 1).await;
    match &health {
        Ok(val) => {
            let status = val.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");
            h.check_abs(
                "health.status == healthy",
                if status == "healthy" { 1.0 } else { 0.0 },
                1.0,
                0.5,
            );

            let caps = val
                .get("capabilities")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            h.check_abs("health.capabilities >= 11", if caps >= 11 { 1.0 } else { 0.0 }, 1.0, 0.5);

            let has_hardware = val.get("hardware").is_some();
            h.check_abs("health.hardware present", if has_hardware { 1.0 } else { 0.0 }, 1.0, 0.5);

            let has_stats = val.get("stats").is_some();
            h.check_abs("health.stats present", if has_stats { 1.0 } else { 0.0 }, 1.0, 0.5);
        }
        Err(e) => {
            eprintln!("  Health check failed: {e}");
            h.check_abs("health.reachable", 0.0, 1.0, 0.5);
            h.check_abs("health.capabilities", 0.0, 1.0, 0.5);
            h.check_abs("health.hardware", 0.0, 1.0, 0.5);
            h.check_abs("health.stats", 0.0, 1.0, 0.5);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 3-5: Evoformer block RPC
    // ═══════════════════════════════════════════════════════════════════
    let evo = rpc_call(
        &socket_path,
        "science.evoformer_block",
        serde_json::json!({
            "n_seq": 4, "n_res": 6, "n_heads": 2, "head_dim": 4,
            "c_pair": 4, "seed": 42,
        }),
        10,
    ).await;

    match &evo {
        Ok(val) => {
            let msa_finite = val.get("msa_finite").and_then(|v| v.as_bool()).unwrap_or(false);
            h.check_abs("evoformer.msa_finite", if msa_finite { 1.0 } else { 0.0 }, 1.0, 0.5);

            let pair_finite = val.get("pair_finite").and_then(|v| v.as_bool()).unwrap_or(false);
            h.check_abs("evoformer.pair_finite", if pair_finite { 1.0 } else { 0.0 }, 1.0, 0.5);

            let tri_finite = val.get("tri_attn_finite").and_then(|v| v.as_bool()).unwrap_or(false);
            h.check_abs("evoformer.tri_attn_finite", if tri_finite { 1.0 } else { 0.0 }, 1.0, 0.5);

            let msa_changed = val.get("msa_changed").and_then(|v| v.as_bool()).unwrap_or(false);
            h.check_abs("evoformer.msa_changed", if msa_changed { 1.0 } else { 0.0 }, 1.0, 0.5);

            let pair_changed = val.get("pair_changed").and_then(|v| v.as_bool()).unwrap_or(false);
            h.check_abs("evoformer.pair_changed", if pair_changed { 1.0 } else { 0.0 }, 1.0, 0.5);
        }
        Err(e) => {
            eprintln!("  science.evoformer_block failed: {e}");
            h.check_abs("evoformer.reachable", 0.0, 1.0, 0.5);
            h.check_abs("evoformer.pair", 0.0, 1.0, 0.5);
            h.check_abs("evoformer.tri", 0.0, 1.0, 0.5);
            h.check_abs("evoformer.msa_changed", 0.0, 1.0, 0.5);
            h.check_abs("evoformer.pair_changed", 0.0, 1.0, 0.5);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 6-8: Structure Module RPC
    // ═══════════════════════════════════════════════════════════════════
    let sm = rpc_call(
        &socket_path,
        "science.structure_module",
        serde_json::json!({
            "n_res": 6, "c_single": 8, "c_pair": 4,
            "n_heads": 2, "head_dim": 4, "n_points": 2, "seed": 42,
        }),
        20,
    ).await;

    match &sm {
        Ok(val) => {
            let ipa_finite = val.get("ipa_scores_finite").and_then(|v| v.as_bool()).unwrap_or(false);
            h.check_abs("structure.ipa_finite", if ipa_finite { 1.0 } else { 0.0 }, 1.0, 0.5);

            let frames_finite = val.get("frames_finite").and_then(|v| v.as_bool()).unwrap_or(false);
            h.check_abs("structure.frames_finite", if frames_finite { 1.0 } else { 0.0 }, 1.0, 0.5);

            let torsion_finite = val.get("torsion_finite").and_then(|v| v.as_bool()).unwrap_or(false);
            h.check_abs("structure.torsion_finite", if torsion_finite { 1.0 } else { 0.0 }, 1.0, 0.5);

            let torsion_count = val.get("torsion_count").and_then(|v| v.as_u64()).unwrap_or(0);
            h.check_abs("structure.torsion_count", torsion_count as f64, (6 * 7 * 2) as f64, 0.5);
        }
        Err(e) => {
            eprintln!("  science.structure_module failed: {e}");
            h.check_abs("structure.ipa", 0.0, 1.0, 0.5);
            h.check_abs("structure.frames", 0.0, 1.0, 0.5);
            h.check_abs("structure.torsion", 0.0, 1.0, 0.5);
            h.check_abs("structure.torsion_count", 0.0, 1.0, 0.5);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 9: Folding health report
    // ═══════════════════════════════════════════════════════════════════
    let fh = rpc_call(&socket_path, "science.folding_health", serde_json::json!({}), 30).await;
    match &fh {
        Ok(val) => {
            let primitives = val.get("folding_primitives").and_then(|v| v.as_object());
            if let Some(p) = primitives {
                let all_true = p.values().all(|v| v.as_bool().unwrap_or(false));
                h.check_abs("folding_health.all_primitives", if all_true { 1.0 } else { 0.0 }, 1.0, 0.5);
                h.check_abs("folding_health.primitive_count", p.len() as f64, 14.0, 0.5);
            } else {
                h.check_abs("folding_health.primitives_present", 0.0, 1.0, 0.5);
                h.check_abs("folding_health.count", 0.0, 14.0, 0.5);
            }
        }
        Err(e) => {
            eprintln!("  science.folding_health failed: {e}");
            h.check_abs("folding_health.reachable", 0.0, 1.0, 0.5);
            h.check_abs("folding_health.count", 0.0, 14.0, 0.5);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 10-11: GPU dispatch operations
    // ═══════════════════════════════════════════════════════════════════
    let mm = rpc_call(
        &socket_path,
        "science.gpu_dispatch",
        serde_json::json!({
            "op": "softmax",
            "x": [1.0, 2.0, 3.0, 4.0],
        }),
        40,
    ).await;

    match &mm {
        Ok(val) => {
            let result: Vec<f64> = val.get("result")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            let sum: f64 = result.iter().sum();
            h.check_abs("gpu_dispatch.softmax_sum", sum, 1.0, 1e-10);

            let backend = val.get("backend").and_then(|v| v.as_str()).unwrap_or("");
            let has_backend = !backend.is_empty();
            h.check_abs("gpu_dispatch.backend_reported", if has_backend { 1.0 } else { 0.0 }, 1.0, 0.5);
        }
        Err(e) => {
            eprintln!("  science.gpu_dispatch softmax failed: {e}");
            h.check_abs("gpu_dispatch.softmax", 0.0, 1.0, 0.5);
            h.check_abs("gpu_dispatch.backend", 0.0, 1.0, 0.5);
        }
    }

    let mean_result = rpc_call(
        &socket_path,
        "science.gpu_dispatch",
        serde_json::json!({
            "op": "mean",
            "data": [2.0, 4.0, 6.0, 8.0],
        }),
        41,
    ).await;

    match &mean_result {
        Ok(val) => {
            let result = val.get("result").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
            h.check_abs("gpu_dispatch.mean", result, 5.0, 1e-10);
        }
        Err(e) => {
            eprintln!("  science.gpu_dispatch mean failed: {e}");
            h.check_abs("gpu_dispatch.mean", 0.0, 5.0, 0.5);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 12: Cross-primal forward (graceful failure)
    // ═══════════════════════════════════════════════════════════════════
    let forward = rpc_call(
        &socket_path,
        "data.pdb_fetch",
        serde_json::json!({ "pdb_id": "1CRN" }),
        50,
    ).await;

    let forward_failed_gracefully = forward.is_err();
    h.check_abs(
        "cross_primal.nestgate_unavailable_graceful",
        if forward_failed_gracefully { 1.0 } else { 0.0 },
        1.0,
        0.5,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Test 13-14: Concurrent requests
    // ═══════════════════════════════════════════════════════════════════
    let sp = socket_path.clone();
    let handles: Vec<_> = (0..4).map(|i| {
        let sp = sp.clone();
        tokio::spawn(async move {
            rpc_call(
                &sp,
                "science.ipr",
                serde_json::json!({ "wavefunction": [0.5, 0.5, 0.5, 0.5] }),
                100 + i,
            ).await
        })
    }).collect();

    let mut all_ok = true;
    for handle in handles {
        match handle.await {
            Ok(Ok(_)) => {}
            _ => { all_ok = false; }
        }
    }
    h.check_abs("concurrent.4_requests", if all_ok { 1.0 } else { 0.0 }, 1.0, 0.5);

    // ═══════════════════════════════════════════════════════════════════
    // Test 15: Request counter incremented
    // ═══════════════════════════════════════════════════════════════════
    let health2 = rpc_call(&socket_path, "health", serde_json::json!({}), 200).await;
    match &health2 {
        Ok(val) => {
            let served = val.get("stats")
                .and_then(|s| s.get("requests_served"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            h.check_abs("stats.requests_served > 0", if served > 0 { 1.0 } else { 0.0 }, 1.0, 0.5);
        }
        Err(e) => {
            eprintln!("  Health check 2 failed: {e}");
            h.check_abs("stats.requests_served", 0.0, 1.0, 0.5);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 16: Unknown method returns proper error
    // ═══════════════════════════════════════════════════════════════════
    let unknown = rpc_call(
        &socket_path,
        "science.nonexistent_method",
        serde_json::json!({}),
        300,
    ).await;
    h.check_abs(
        "rpc.unknown_method_error",
        if unknown.is_err() { 1.0 } else { 0.0 },
        1.0,
        0.5,
    );

    // Clean up
    child.kill().await.ok();
    let _ = std::fs::remove_file(&socket_path);
    let _ = std::fs::remove_dir(&socket_dir);

    h.finish();
}
