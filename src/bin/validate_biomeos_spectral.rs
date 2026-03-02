// SPDX-License-Identifier: AGPL-3.0-or-later

//! biomeOS Integration Validator: neuralSpring Spectral Capabilities
//!
//! Validates end-to-end biomeOS capability routing by:
//!   1. Starting the `neuralspring_primal` JSON-RPC server
//!   2. Calling each capability via the Unix socket
//!   3. Comparing results to direct Rust function calls (CPU reference)
//!   4. Verifying JSON-RPC protocol compliance
//!
//! This proves neuralSpring's spectral analysis is accessible through
//! the biomeOS capability routing infrastructure.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::suboptimal_flops,
    clippy::option_if_let_else
)]

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::time::sleep;

use neural_spring::anderson_localization::disorder_sweep;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
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
    let mut h = ValidationHarness::new("biomeOS Spectral Integration");

    let socket_dir = std::env::temp_dir().join("biomeos-test");
    if let Err(e) = std::fs::create_dir_all(&socket_dir) {
        eprintln!("  Failed to create socket directory: {e}");
        h.check_abs("setup.socket_dir", 0.0, 1.0, 0.5);
        h.finish();
    }
    let socket_path = socket_dir.join("neural-spring-test.sock");

    if socket_path.exists() {
        let _ = std::fs::remove_file(&socket_path);
    }

    let primal_bin = match std::env::current_exe() {
        Ok(exe) => {
            if let Some(dir) = exe.parent() {
                dir.join("neuralspring_primal")
            } else {
                eprintln!("  current_exe has no parent directory");
                h.check_abs("setup.primal_bin", 0.0, 1.0, 0.5);
                h.finish();
            }
        }
        Err(e) => {
            eprintln!("  Failed to resolve current_exe: {e}");
            h.check_abs("setup.primal_bin", 0.0, 1.0, 0.5);
            h.finish();
        }
    };

    if !primal_bin.exists() {
        eprintln!(
            "  neuralspring_primal binary not found at {}",
            primal_bin.display()
        );
        eprintln!("  Build with: cargo build --features primal --bin neuralspring_primal");
        eprintln!("  Skipping biomeOS integration tests (primal binary not available)");
        h.finish();
    }

    let mut child = match tokio::process::Command::new(&primal_bin)
        .env("BIOMEOS_SOCKET_DIR", &socket_dir)
        .env("FAMILY_ID", "test")
        .kill_on_drop(true)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  Failed to spawn neuralspring_primal: {e}");
            h.check_abs("setup.spawn_primal", 0.0, 1.0, 0.5);
            h.finish();
        }
    };

    // Wait for socket to appear
    let mut retries = 0;
    while !socket_path.exists() && retries < 50 {
        sleep(Duration::from_millis(100)).await;
        retries += 1;
    }

    if !socket_path.exists() {
        eprintln!("  Socket did not appear after 5s. Primal may have failed to start.");
        child.kill().await.ok();
        h.finish();
    }

    // Give the server a moment to be fully ready
    sleep(Duration::from_millis(200)).await;

    // ═══════════════════════════════════════════════════════════════════
    // Test 1: Health check
    // ═══════════════════════════════════════════════════════════════════
    let health = rpc_call(&socket_path, "health", serde_json::json!({}), 1).await;
    match &health {
        Ok(val) => {
            let status = val
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            h.check_abs(
                "health.status == healthy",
                if status == "healthy" { 1.0 } else { 0.0 },
                1.0,
                0.5,
            );
            let caps = val
                .get("capabilities")
                .and_then(|v| v.as_array())
                .map_or(0, std::vec::Vec::len);
            h.check_abs(
                "health.capabilities count >= 7",
                if caps >= 7 { 1.0 } else { 0.0 },
                1.0,
                0.5,
            );
        }
        Err(e) => {
            eprintln!("  Health check failed: {e}");
            h.check_abs("health.reachable", 0.0, 1.0, 0.5);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 2: IPR computation — compare RPC vs direct call
    // ═══════════════════════════════════════════════════════════════════
    let wavefunction = vec![0.5, 0.5, 0.5, 0.5];
    let cpu_ipr = neural_spring::anderson_localization::ipr(&wavefunction);

    let rpc_result = rpc_call(
        &socket_path,
        "science.ipr",
        serde_json::json!({ "wavefunction": wavefunction }),
        2,
    )
    .await;

    match rpc_result {
        Ok(val) => {
            let rpc_ipr = val
                .get("ipr")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(f64::NAN);
            h.check_abs(
                "science.ipr parity",
                rpc_ipr,
                cpu_ipr,
                tolerances::EXACT_F64,
            );
        }
        Err(e) => {
            eprintln!("  science.ipr failed: {e}");
            h.check_abs("science.ipr reachable", 0.0, 1.0, 0.5);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 3: Disorder sweep — compare RPC vs direct call
    // ═══════════════════════════════════════════════════════════════════
    let n = 16;
    let seed = 42u64;
    let w_vals = vec![1.0, 4.0, 16.0];

    let mut rng_cpu = Rng::new(seed);
    let cpu_iprs = disorder_sweep(n, 1.0, &w_vals, &mut rng_cpu);

    let rpc_result = rpc_call(
        &socket_path,
        "science.disorder_sweep",
        serde_json::json!({
            "lattice_size": n,
            "hopping": 1.0,
            "disorder_values": w_vals,
            "seed": seed,
        }),
        3,
    )
    .await;

    match rpc_result {
        Ok(val) => {
            let rpc_iprs: Vec<f64> = val
                .get("ipr_values")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();

            for (i, (&cpu, &rpc)) in cpu_iprs.iter().zip(rpc_iprs.iter()).enumerate() {
                h.check_abs(
                    &format!("disorder_sweep[W={}] parity", w_vals[i]),
                    rpc,
                    cpu,
                    tolerances::EXACT_F64,
                );
            }
        }
        Err(e) => {
            eprintln!("  science.disorder_sweep failed: {e}");
            h.check_abs("science.disorder_sweep reachable", 0.0, 1.0, 0.5);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 4: Spectral analysis — verify result structure and physics
    // ═══════════════════════════════════════════════════════════════════
    let rpc_result = rpc_call(
        &socket_path,
        "science.spectral_analysis",
        serde_json::json!({ "dim": 16, "disorder": 2.0, "seed": 42 }),
        4,
    )
    .await;

    match rpc_result {
        Ok(val) => {
            let evals: Vec<f64> = val
                .get("eigenvalues")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            h.check_abs("spectral.eigenvalue_count", evals.len() as f64, 16.0, 0.5);

            let ipr_val = val
                .get("mean_ipr")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            let ipr_above_zero = if ipr_val > 0.0 { 1.0 } else { 0.0 };
            h.check_abs("spectral.ipr > 0", ipr_above_zero, 1.0, 0.5);

            let lsr = val
                .get("level_spacing_ratio")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            let lsr_valid = if (0.0..=1.0).contains(&lsr) { 1.0 } else { 0.0 };
            h.check_abs("spectral.lsr in [0,1]", lsr_valid, 1.0, 0.5);
        }
        Err(e) => {
            eprintln!("  science.spectral_analysis failed: {e}");
            h.check_abs("spectral reachable", 0.0, 1.0, 0.5);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 5: Anderson localization — verify disorder sweep curve
    // ═══════════════════════════════════════════════════════════════════
    let rpc_result = rpc_call(
        &socket_path,
        "science.anderson_localization",
        serde_json::json!({
            "lattice_size": 12,
            "hopping": 1.0,
            "disorder_values": [0.5, 2.0, 8.0],
            "seed": 42,
        }),
        5,
    )
    .await;

    match rpc_result {
        Ok(val) => {
            let results = val.get("results").and_then(|v| v.as_array());
            if let Some(r) = results {
                h.check_abs("anderson.result_count", r.len() as f64, 3.0, 0.5);

                let iprs: Vec<f64> = r
                    .iter()
                    .filter_map(|v| v.get("mean_ipr").and_then(serde_json::Value::as_f64))
                    .collect();

                if iprs.len() == 3 {
                    let ipr_increases = if iprs[2] > iprs[0] { 1.0 } else { 0.0 };
                    h.check_abs(
                        "anderson.ipr increases with disorder",
                        ipr_increases,
                        1.0,
                        0.5,
                    );
                }
            } else {
                h.check_abs("anderson.results array", 0.0, 1.0, 0.5);
            }
        }
        Err(e) => {
            eprintln!("  science.anderson_localization failed: {e}");
            h.check_abs("anderson reachable", 0.0, 1.0, 0.5);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 6: Hessian eigenanalysis — verify known diagonal Hessian
    // ═══════════════════════════════════════════════════════════════════
    let rpc_result = rpc_call(
        &socket_path,
        "science.hessian_eigen",
        serde_json::json!({ "dim": 10, "surface_type": "quadratic" }),
        6,
    )
    .await;

    match rpc_result {
        Ok(val) => {
            let evals: Vec<f64> = val
                .get("eigenvalues")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();

            h.check_abs("hessian.eigenvalue_count", evals.len() as f64, 10.0, 0.5);

            // For quadratic surface with diagonal [1..10], eigenvalues should be 1..10
            if evals.len() == 10 {
                for (i, &eval) in evals.iter().enumerate() {
                    h.check_abs(
                        &format!("hessian.eval[{i}]"),
                        eval,
                        (i + 1) as f64,
                        tolerances::CROSS_LANGUAGE,
                    );
                }
            }

            let trace = val
                .get("trace")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            h.check_abs("hessian.trace", trace, 55.0, tolerances::CROSS_LANGUAGE);
        }
        Err(e) => {
            eprintln!("  science.hessian_eigen failed: {e}");
            h.check_abs("hessian reachable", 0.0, 1.0, 0.5);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 7: Agent coordination — verify result structure
    // ═══════════════════════════════════════════════════════════════════
    let rpc_result = rpc_call(
        &socket_path,
        "science.agent_coordination",
        serde_json::json!({
            "n_agents": 4,
            "dimensions": 2,
            "comm_range": 5.0,
            "disorder_values": [0.0, 1.0],
            "seed": 42,
        }),
        7,
    )
    .await;

    match rpc_result {
        Ok(val) => {
            let results = val.get("results").and_then(|v| v.as_array());
            if let Some(r) = results {
                h.check_abs("coordination.result_count", r.len() as f64, 2.0, 0.5);

                for entry in r {
                    let ipr = entry
                        .get("mean_ipr")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(-1.0);
                    let ipr_valid = if ipr > 0.0 { 1.0 } else { 0.0 };
                    let w = entry
                        .get("disorder")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(-1.0);
                    h.check_abs(&format!("coordination.ipr[W={w}] > 0"), ipr_valid, 1.0, 0.5);
                }
            } else {
                h.check_abs("coordination.results array", 0.0, 1.0, 0.5);
            }
        }
        Err(e) => {
            eprintln!("  science.agent_coordination failed: {e}");
            h.check_abs("coordination reachable", 0.0, 1.0, 0.5);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 8: Training trajectory — verify epoch count and physics
    // ═══════════════════════════════════════════════════════════════════
    let rpc_result = rpc_call(
        &socket_path,
        "science.training_trajectory",
        serde_json::json!({ "dim": 8, "n_epochs": 10, "seed": 42 }),
        8,
    )
    .await;

    match rpc_result {
        Ok(val) => {
            let trajectory = val.get("trajectory").and_then(|v| v.as_array());
            if let Some(t) = trajectory {
                h.check_abs("trajectory.epoch_count", t.len() as f64, 11.0, 0.5);

                let first_ipr = t
                    .first()
                    .and_then(|v| v.get("mean_ipr"))
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0);
                let first_valid = if first_ipr > 0.0 { 1.0 } else { 0.0 };
                h.check_abs("trajectory.first_ipr > 0", first_valid, 1.0, 0.5);
            } else {
                h.check_abs("trajectory.array present", 0.0, 1.0, 0.5);
            }
        }
        Err(e) => {
            eprintln!("  science.training_trajectory failed: {e}");
            h.check_abs("trajectory reachable", 0.0, 1.0, 0.5);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 9: Method not found — verify proper error handling
    // ═══════════════════════════════════════════════════════════════════
    let rpc_result = rpc_call(
        &socket_path,
        "science.nonexistent",
        serde_json::json!({}),
        9,
    )
    .await;

    let got_error = rpc_result.is_err();
    h.check_abs(
        "rpc.method_not_found returns error",
        if got_error { 1.0 } else { 0.0 },
        1.0,
        0.5,
    );

    // Clean up
    child.kill().await.ok();
    let _ = std::fs::remove_file(&socket_path);
    let _ = std::fs::remove_dir(&socket_dir);

    h.finish();
}
