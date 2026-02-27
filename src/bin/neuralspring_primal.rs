// SPDX-License-Identifier: AGPL-3.0-or-later

//! neuralSpring biomeOS Primal
//!
//! JSON-RPC 2.0 server exposing neuralSpring's spectral analysis
//! capabilities to the biomeOS ecosystem via capability-based discovery.
//!
//! Capabilities:
//!   science.spectral_analysis    — full spectral decomposition (eigensolve → IPR/LSR)
//!   science.anderson_localization — disorder sweep with Anderson Hamiltonians
//!   science.hessian_eigen         — Hessian eigenanalysis for loss landscapes
//!   science.agent_coordination    — multi-agent coordination spectral analysis
//!   science.ipr                   — inverse participation ratio computation
//!   science.disorder_sweep        — IPR vs disorder strength curve
//!   science.training_trajectory   — spectral evolution over training epochs
//!
//! Socket: $XDG_RUNTIME_DIR/biomeos/neuralspring-{family_id}.sock

#![allow(clippy::pedantic, clippy::nursery)]

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tokio::sync::Semaphore;

use neural_spring::agent_coordination::{coordination_spectral_analysis, generate_lattice_agents};
use neural_spring::anderson_localization::{
    anderson_hamiltonian_random, disorder_sweep, ipr, jacobi_eigh, mean_ipr,
};
use neural_spring::eigh::eigh_householder_qr;
use neural_spring::rng::Rng;

// ═══════════════════════════════════════════════════════════════════════════════
// JSON-RPC 2.0 types
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
    id: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
    id: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

impl JsonRpcResponse {
    fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0",
            result: Some(result),
            error: None,
            id,
        }
    }

    fn error(id: serde_json::Value, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0",
            result: None,
            error: Some(JsonRpcError { code, message }),
            id,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Capability handlers
// ═══════════════════════════════════════════════════════════════════════════════

fn handle_health(id: serde_json::Value) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "status": "healthy",
            "primal": "neuralspring",
            "version": env!("CARGO_PKG_VERSION"),
            "capabilities": [
                "science.spectral_analysis",
                "science.anderson_localization",
                "science.hessian_eigen",
                "science.agent_coordination",
                "science.ipr",
                "science.disorder_sweep",
                "science.training_trajectory"
            ]
        }),
    )
}

fn handle_ipr(id: serde_json::Value, params: &serde_json::Value) -> JsonRpcResponse {
    let wavefunction: Vec<f64> = match serde_json::from_value(
        params
            .get("wavefunction")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    ) {
        Ok(v) => v,
        Err(e) => return JsonRpcResponse::error(id, -32602, format!("Invalid params: {e}")),
    };
    let result = ipr(&wavefunction);
    JsonRpcResponse::success(id, serde_json::json!({ "ipr": result }))
}

fn handle_disorder_sweep(id: serde_json::Value, params: &serde_json::Value) -> JsonRpcResponse {
    let n = params
        .get("lattice_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(20) as usize;
    let t = params
        .get("hopping")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let w_vals: Vec<f64> = params
        .get("disorder_values")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_else(|| vec![0.5, 1.0, 2.0, 4.0, 8.0, 16.0]);
    let seed = params.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);

    let mut rng = Rng::new(seed);
    let iprs = disorder_sweep(n, t, &w_vals, &mut rng);

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "disorder_values": w_vals,
            "ipr_values": iprs,
            "lattice_size": n,
            "hopping": t,
        }),
    )
}

fn handle_spectral_analysis(id: serde_json::Value, params: &serde_json::Value) -> JsonRpcResponse {
    let n = params.get("dim").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let w = params
        .get("disorder")
        .and_then(|v| v.as_f64())
        .unwrap_or(2.0);
    let seed = params.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);

    let mut rng = Rng::new(seed);
    let h = anderson_hamiltonian_random(n, 1.0, w, &mut rng);
    let decomp = eigh_householder_qr(&h, n);

    let ipr_val = mean_ipr(&decomp.eigenvectors, n);
    let mut evals = decomp.eigenvalues.clone();
    evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lsr = neural_spring::weight_spectral::level_spacing_ratio(&evals);

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "eigenvalues": evals,
            "mean_ipr": ipr_val,
            "level_spacing_ratio": lsr,
            "dim": n,
            "disorder": w,
        }),
    )
}

fn handle_anderson_localization(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    let n = params
        .get("lattice_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(20) as usize;
    let t = params
        .get("hopping")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let w_vals: Vec<f64> = params
        .get("disorder_values")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_else(|| vec![0.5, 1.0, 2.0, 4.0, 8.0, 16.0]);
    let seed = params.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);

    let mut rng = Rng::new(seed);
    let mut results = Vec::new();

    for &w in &w_vals {
        let h = anderson_hamiltonian_random(n, t, w, &mut rng);
        let (eigenvalues, eigenvectors) = jacobi_eigh(&h, n);
        let ipr_val = mean_ipr(&eigenvectors, n);
        let mut sorted_evals = eigenvalues.clone();
        sorted_evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lsr = neural_spring::weight_spectral::level_spacing_ratio(&sorted_evals);

        results.push(serde_json::json!({
            "disorder": w,
            "mean_ipr": ipr_val,
            "level_spacing_ratio": lsr,
            "eigenvalue_range": [sorted_evals.first(), sorted_evals.last()],
        }));
    }

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "results": results,
            "lattice_size": n,
            "hopping": t,
        }),
    )
}

fn handle_hessian_eigen(id: serde_json::Value, params: &serde_json::Value) -> JsonRpcResponse {
    let n = params.get("dim").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
    let surface = params
        .get("surface_type")
        .and_then(|v| v.as_str())
        .unwrap_or("quadratic");

    let hessian: Vec<f64> = match surface {
        "quadratic" => {
            let mut h = vec![0.0; n * n];
            for i in 0..n {
                h[i * n + i] = (i + 1) as f64;
            }
            h
        }
        "rosenbrock" => {
            let mut h = vec![0.0; n * n];
            for i in 0..n {
                h[i * n + i] = 200.0 + 2.0;
                if i + 1 < n {
                    h[i * n + i + 1] = -200.0;
                    h[(i + 1) * n + i] = -200.0;
                }
            }
            h
        }
        _ => {
            let mut h = vec![0.0; n * n];
            for i in 0..n {
                h[i * n + i] = (i + 1) as f64;
            }
            h
        }
    };

    let decomp = eigh_householder_qr(&hessian, n);
    let mut evals = decomp.eigenvalues.clone();
    evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let entropy = neural_spring::primitives::shannon_entropy(&evals);
    let trace: f64 = evals.iter().sum();

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "eigenvalues": evals,
            "spectral_entropy": entropy,
            "trace": trace,
            "condition_number": evals.last().unwrap_or(&1.0) / evals.first().unwrap_or(&1.0).max(1e-15),
            "dim": n,
            "surface_type": surface,
        }),
    )
}

fn handle_agent_coordination(id: serde_json::Value, params: &serde_json::Value) -> JsonRpcResponse {
    let n = params
        .get("n_agents")
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as usize;
    let dim = params
        .get("dimensions")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as usize;
    let comm = params
        .get("comm_range")
        .and_then(|v| v.as_f64())
        .unwrap_or(3.0);
    let disorder_vals: Vec<f64> = params
        .get("disorder_values")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_else(|| vec![0.0, 0.5, 1.0, 2.0]);
    let seed = params.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);

    let mut rng = Rng::new(seed);
    let cap_var = params
        .get("capability_variance")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let agents = generate_lattice_agents(n, dim, cap_var, &mut rng);

    let mut results = Vec::new();
    for &w in &disorder_vals {
        let cr = coordination_spectral_analysis(&agents, comm, w);
        results.push(serde_json::json!({
            "disorder": w,
            "mean_ipr": cr.mean_ipr,
            "level_spacing_ratio": cr.level_spacing_ratio,
            "algebraic_connectivity": cr.algebraic_connectivity,
        }));
    }

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "results": results,
            "n_agents": n,
            "dimensions": dim,
            "comm_range": comm,
        }),
    )
}

fn handle_training_trajectory(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    let dim = params.get("dim").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let n_epochs = params
        .get("n_epochs")
        .and_then(|v| v.as_u64())
        .unwrap_or(20) as usize;
    let seed = params.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);

    let mut rng = Rng::new(seed);

    let mut w_start = vec![0.0f64; dim * dim];
    let mut w_end = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            w_start[i * dim + j] = rng.uniform() - 0.5;
            w_end[i * dim + j] = rng.uniform() - 0.5;
        }
    }
    // Symmetrize
    for i in 0..dim {
        for j in (i + 1)..dim {
            w_start[j * dim + i] = w_start[i * dim + j];
            w_end[j * dim + i] = w_end[i * dim + j];
        }
    }

    let mut trajectory = Vec::new();
    for epoch in 0..=n_epochs {
        let alpha = epoch as f64 / n_epochs as f64;
        let w: Vec<f64> = w_start
            .iter()
            .zip(&w_end)
            .map(|(&s, &e)| s + alpha * (e - s))
            .collect();

        let decomp = eigh_householder_qr(&w, dim);
        let ipr_val = mean_ipr(&decomp.eigenvectors, dim);
        let mut evals = decomp.eigenvalues.clone();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let entropy = neural_spring::primitives::shannon_entropy(&evals);

        trajectory.push(serde_json::json!({
            "epoch": epoch,
            "alpha": alpha,
            "mean_ipr": ipr_val,
            "spectral_entropy": entropy,
        }));
    }

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "trajectory": trajectory,
            "dim": dim,
            "n_epochs": n_epochs,
        }),
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Request dispatcher
// ═══════════════════════════════════════════════════════════════════════════════

fn dispatch(request: JsonRpcRequest) -> JsonRpcResponse {
    match request.method.as_str() {
        "health" => handle_health(request.id),
        "science.ipr" => handle_ipr(request.id, &request.params),
        "science.disorder_sweep" => handle_disorder_sweep(request.id, &request.params),
        "science.spectral_analysis" => handle_spectral_analysis(request.id, &request.params),
        "science.anderson_localization" => {
            handle_anderson_localization(request.id, &request.params)
        }
        "science.hessian_eigen" => handle_hessian_eigen(request.id, &request.params),
        "science.agent_coordination" => handle_agent_coordination(request.id, &request.params),
        "science.training_trajectory" => handle_training_trajectory(request.id, &request.params),
        _ => JsonRpcResponse::error(
            request.id,
            -32601,
            format!("Method not found: {}", request.method),
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Socket resolution (follows biomeOS 5-tier standard)
// ═══════════════════════════════════════════════════════════════════════════════

/// 4-tier socket resolution (biomeOS standard):
///   1. `BIOMEOS_SOCKET_DIR` env var (explicit override)
///   2. `XDG_RUNTIME_DIR`/biomeos (freedesktop)
///   3. /run/user/{uid}/biomeos (systemd)
///   4. `TMPDIR` or /tmp fallback
fn resolve_socket_path(family_id: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("BIOMEOS_SOCKET_DIR") {
        return PathBuf::from(dir).join(format!("neuralspring-{family_id}.sock"));
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(xdg)
            .join("biomeos")
            .join(format!("neuralspring-{family_id}.sock"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let Ok(meta) = std::fs::metadata("/proc/self") {
            let uid = meta.uid();
            let dir = PathBuf::from(format!("/run/user/{uid}/biomeos"));
            if dir.parent().is_some_and(std::path::Path::exists) {
                return dir.join(format!("neuralspring-{family_id}.sock"));
            }
        }
    }
    let tmp = std::env::var("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"));
    tmp.join("biomeos")
        .join(format!("neuralspring-{family_id}.sock"))
}

fn get_family_id() -> String {
    if let Ok(id) = std::env::var("FAMILY_ID") {
        return id;
    }
    if let Ok(id) = std::env::var("BIOMEOS_FAMILY_ID") {
        return id;
    }
    "default".to_string()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let family_id = get_family_id();
    let socket_path = resolve_socket_path(&family_id);

    if let Some(parent) = socket_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .context("creating socket directory")?;
    }

    if socket_path.exists() {
        tokio::fs::remove_file(&socket_path)
            .await
            .context("removing stale socket")?;
    }

    let listener = UnixListener::bind(&socket_path)
        .context(format!("binding to {}", socket_path.display()))?;

    eprintln!("neuralSpring primal listening on {}", socket_path.display());
    eprintln!("  Family ID: {family_id}");
    eprintln!("  Capabilities: science.spectral_analysis, science.anderson_localization,");
    eprintln!("                science.hessian_eigen, science.agent_coordination,");
    eprintln!("                science.ipr, science.disorder_sweep, science.training_trajectory");

    let concurrency = Arc::new(Semaphore::new(4));

    loop {
        let (stream, _addr) = listener.accept().await?;
        let permit = concurrency.clone().acquire_owned().await?;

        tokio::spawn(async move {
            let (reader, mut writer) = stream.into_split();
            let mut buf_reader = BufReader::new(reader);
            let mut line = String::new();

            while buf_reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    line.clear();
                    continue;
                }

                let response = match serde_json::from_str::<JsonRpcRequest>(trimmed) {
                    Ok(req) => dispatch(req),
                    Err(e) => JsonRpcResponse::error(
                        serde_json::Value::Null,
                        -32700,
                        format!("Parse error: {e}"),
                    ),
                };

                let resp_json = serde_json::to_vec(&response).unwrap_or_default();
                let _ = writer.write_all(&resp_json).await;
                let _ = writer.write_all(b"\n").await;
                let _ = writer.flush().await;

                line.clear();
            }

            drop(permit);
        });
    }
}
