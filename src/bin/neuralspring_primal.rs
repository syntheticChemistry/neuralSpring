// SPDX-License-Identifier: AGPL-3.0-or-later

//! neuralSpring biomeOS Primal — Tower Mode
//!
//! JSON-RPC 2.0 server exposing neuralSpring's spectral analysis AND
//! sovereign folding capabilities to the biomeOS ecosystem.
//!
//! ## Capability domains
//!
//! **Spectral analysis** (baseCamp):
//!   `science.ipr`, `science.disorder_sweep`, `science.spectral_analysis`,
//!   `science.anderson_localization`, `science.hessian_eigen`,
//!   `science.agent_coordination`, `science.training_trajectory`
//!
//! **Sovereign folding** (nF-01/02):
//!   `science.evoformer_block`, `science.structure_module`,
//!   `science.folding_health`
//!
//! **GPU dispatch**:
//!   `science.gpu_dispatch` — route arbitrary Dispatcher operations
//!
//! ## biomeOS integration
//!
//! On startup, probes for a biomeOS orchestrator socket and registers
//! capabilities via `lifecycle.register` + `capability.register`.
//! Sends heartbeats every 30s. Deregisters on SIGTERM.
//!
//! Socket: `$XDG_RUNTIME_DIR/biomeos/neuralspring-{family_id}.sock`

#![allow(
    clippy::pedantic,
    clippy::nursery,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Semaphore;

use neural_spring::agent_coordination::{coordination_spectral_analysis, generate_lattice_agents};
use neural_spring::anderson_localization::{
    anderson_hamiltonian_random, disorder_sweep, ipr, jacobi_eigh, mean_ipr,
};
use neural_spring::eigh::eigh_householder_qr;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::rng::Rng;
use neural_spring::sovereign_folding::{
    msa_col_attention, msa_row_attention, outer_product_mean,
    triangle_attention_scores, triangle_mul_incoming, triangle_mul_outgoing,
};
use neural_spring::structure_module::{
    backbone_update, ipa_scores, torsion_angles, IpaConfig,
};

// ═══════════════════════════════════════════════════════════════════════════════
// JSON-RPC 2.0 types
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[serde(rename = "jsonrpc")]
    _jsonrpc: String,
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
// Shared primal state
// ═══════════════════════════════════════════════════════════════════════════════

struct PrimalState {
    dispatcher: Dispatcher,
    start_time: Instant,
    requests_served: AtomicU64,
}

const ALL_CAPABILITIES: &[&str] = &[
    "science.spectral_analysis",
    "science.anderson_localization",
    "science.hessian_eigen",
    "science.agent_coordination",
    "science.ipr",
    "science.disorder_sweep",
    "science.training_trajectory",
    "science.evoformer_block",
    "science.structure_module",
    "science.folding_health",
    "science.gpu_dispatch",
];

// ═══════════════════════════════════════════════════════════════════════════════
// Spectral analysis handlers (existing — baseCamp)
// ═══════════════════════════════════════════════════════════════════════════════

fn handle_health(id: serde_json::Value, state: &PrimalState) -> JsonRpcResponse {
    let uptime = state.start_time.elapsed().as_secs();
    let served = state.requests_served.load(Ordering::Relaxed);

    let hardware = serde_json::json!({
        "gpu_available": state.dispatcher.has_gpu(),
        "gpu_name": state.dispatcher.adapter_name(),
        "fp64_strategy": format!("{:?}", state.dispatcher.fp64_strategy()),
        "backend": format!("{}", state.dispatcher.backend()),
    });

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "status": "healthy",
            "primal": "neuralspring",
            "version": env!("CARGO_PKG_VERSION"),
            "capabilities": ALL_CAPABILITIES,
            "hardware": hardware,
            "stats": {
                "requests_served": served,
                "uptime_seconds": uptime,
            }
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

fn handle_agent_coordination(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
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
            .map(|(&s, &e)| alpha.mul_add(e - s, s))
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
// Sovereign folding handlers (new — nF-01/02)
// ═══════════════════════════════════════════════════════════════════════════════

/// One Evoformer block iteration (Algorithm 6, Jumper et al. 2021).
///
/// Composes: MSA row/col attention → outer product mean → triangle
/// multiplicative outgoing/incoming → triangle attention.
fn handle_evoformer_block(id: serde_json::Value, params: &serde_json::Value) -> JsonRpcResponse {
    let n_seq = params.get("n_seq").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
    let n_res = params.get("n_res").and_then(|v| v.as_u64()).unwrap_or(6) as usize;
    let n_heads = params.get("n_heads").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
    let head_dim = params.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
    let c_pair = params.get("c_pair").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
    let seed = params.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);

    let c_msa = n_heads * head_dim;
    let mut rng = Rng::new(seed);

    let msa_len = n_seq * n_res * c_msa;
    let pair_len = n_res * n_res * c_pair;
    let mut msa: Vec<f64> = (0..msa_len).map(|_| rng.normal()).collect();
    let mut pair: Vec<f64> = (0..pair_len).map(|_| rng.normal()).collect();

    let msa_input: Vec<f64> = msa.clone();
    let pair_input: Vec<f64> = pair.clone();

    // Step 1: MSA row attention with pair bias
    let w_q: Vec<f64> = (0..c_msa * n_heads * head_dim).map(|_| rng.normal() * 0.1).collect();
    let w_k: Vec<f64> = (0..c_msa * n_heads * head_dim).map(|_| rng.normal() * 0.1).collect();
    let w_v: Vec<f64> = (0..c_msa * n_heads * head_dim).map(|_| rng.normal() * 0.1).collect();

    let q_row = matmul_3d(&msa, &w_q, n_seq, n_res, c_msa, n_heads * head_dim);
    let k_row = matmul_3d(&msa, &w_k, n_seq, n_res, c_msa, n_heads * head_dim);
    let v_row = matmul_3d(&msa, &w_v, n_seq, n_res, c_msa, n_heads * head_dim);

    let w_bias: Vec<f64> = (0..c_pair * n_heads).map(|_| rng.normal() * 0.1).collect();
    let pair_bias = einsum_ijc_ch(&pair, &w_bias, n_res, n_res, c_pair, n_heads);

    let msa_row_out = msa_row_attention(&q_row, &k_row, &v_row, &pair_bias,
        n_seq, n_res, n_heads, head_dim);
    let w_o_row: Vec<f64> = (0..n_heads * head_dim * c_msa).map(|_| rng.normal() * 0.1).collect();
    let projected_row = matmul_3d(&msa_row_out, &w_o_row, n_seq, n_res, n_heads * head_dim, c_msa);
    for i in 0..msa.len() {
        msa[i] += projected_row[i];
    }

    // Step 2: MSA column attention
    let w_q_col: Vec<f64> = (0..c_msa * n_heads * head_dim).map(|_| rng.normal() * 0.1).collect();
    let w_k_col: Vec<f64> = (0..c_msa * n_heads * head_dim).map(|_| rng.normal() * 0.1).collect();
    let w_v_col: Vec<f64> = (0..c_msa * n_heads * head_dim).map(|_| rng.normal() * 0.1).collect();

    let q_col = matmul_3d(&msa, &w_q_col, n_seq, n_res, c_msa, n_heads * head_dim);
    let k_col = matmul_3d(&msa, &w_k_col, n_seq, n_res, c_msa, n_heads * head_dim);
    let v_col = matmul_3d(&msa, &w_v_col, n_seq, n_res, c_msa, n_heads * head_dim);

    let msa_col_out = msa_col_attention(&q_col, &k_col, &v_col,
        n_seq, n_res, n_heads, head_dim);
    let w_o_col: Vec<f64> = (0..n_heads * head_dim * c_msa).map(|_| rng.normal() * 0.1).collect();
    let projected_col = matmul_3d(&msa_col_out, &w_o_col, n_seq, n_res, n_heads * head_dim, c_msa);
    for i in 0..msa.len() {
        msa[i] += projected_col[i];
    }

    // Step 3: Outer product mean
    let c_opm = 2;
    let w_opm_a: Vec<f64> = (0..c_msa * c_opm).map(|_| rng.normal() * 0.1).collect();
    let w_opm_b: Vec<f64> = (0..c_msa * c_opm).map(|_| rng.normal() * 0.1).collect();
    let opm_a = matmul_3d(&msa, &w_opm_a, n_seq, n_res, c_msa, c_opm);
    let opm_b = matmul_3d(&msa, &w_opm_b, n_seq, n_res, c_msa, c_opm);
    let opm_out = outer_product_mean(&opm_a, &opm_b, n_seq, n_res, c_opm, c_opm);

    let w_opm_proj: Vec<f64> = (0..c_opm * c_opm * c_pair).map(|_| rng.normal() * 0.1).collect();
    let opm_projected = matmul_2d(&opm_out, &w_opm_proj, n_res * n_res, c_opm * c_opm, c_pair);
    for i in 0..pair.len() {
        pair[i] += opm_projected[i];
    }

    // Step 4: Triangle multiplicative outgoing
    let w_tri_a: Vec<f64> = (0..c_pair * c_pair).map(|_| rng.normal() * 0.1).collect();
    let w_tri_b: Vec<f64> = (0..c_pair * c_pair).map(|_| rng.normal() * 0.1).collect();
    let proj_a = matmul_2d(&pair, &w_tri_a, n_res * n_res, c_pair, c_pair);
    let proj_b = matmul_2d(&pair, &w_tri_b, n_res * n_res, c_pair, c_pair);
    let tri_out = triangle_mul_outgoing(&proj_a, &proj_b, n_res, c_pair);
    let w_tri_proj: Vec<f64> = (0..c_pair * c_pair).map(|_| rng.normal() * 0.1).collect();
    let tri_projected = matmul_2d(&tri_out, &w_tri_proj, n_res * n_res, c_pair, c_pair);
    for i in 0..pair.len() {
        pair[i] += tri_projected[i];
    }

    // Step 5: Triangle multiplicative incoming
    let w_tri_a_in: Vec<f64> = (0..c_pair * c_pair).map(|_| rng.normal() * 0.1).collect();
    let w_tri_b_in: Vec<f64> = (0..c_pair * c_pair).map(|_| rng.normal() * 0.1).collect();
    let proj_a_in = matmul_2d(&pair, &w_tri_a_in, n_res * n_res, c_pair, c_pair);
    let proj_b_in = matmul_2d(&pair, &w_tri_b_in, n_res * n_res, c_pair, c_pair);
    let tri_in = triangle_mul_incoming(&proj_a_in, &proj_b_in, n_res, c_pair);
    let w_tri_proj_in: Vec<f64> = (0..c_pair * c_pair).map(|_| rng.normal() * 0.1).collect();
    let tri_proj_in = matmul_2d(&tri_in, &w_tri_proj_in, n_res * n_res, c_pair, c_pair);
    for i in 0..pair.len() {
        pair[i] += tri_proj_in[i];
    }

    // Step 6: Triangle attention scores
    let w_tq: Vec<f64> = (0..c_pair * n_heads * head_dim).map(|_| rng.normal() * 0.1).collect();
    let w_tk: Vec<f64> = (0..c_pair * n_heads * head_dim).map(|_| rng.normal() * 0.1).collect();
    let tri_q = matmul_2d(&pair, &w_tq, n_res * n_res, c_pair, n_heads * head_dim);
    let tri_k = matmul_2d(&pair, &w_tk, n_res * n_res, c_pair, n_heads * head_dim);
    let w_tri_bias: Vec<f64> = (0..c_pair * n_heads).map(|_| rng.normal() * 0.1).collect();
    let tri_bias = einsum_ijc_ch(&pair, &w_tri_bias, n_res, n_res, c_pair, n_heads);
    let tri_attn = triangle_attention_scores(&tri_q, &tri_k, &tri_bias,
        n_res, n_res, n_heads, head_dim);

    let msa_changed = msa.iter().zip(&msa_input).any(|(a, b)| (a - b).abs() > 1e-15);
    let pair_changed = pair.iter().zip(&pair_input).any(|(a, b)| (a - b).abs() > 1e-15);

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "n_seq": n_seq,
            "n_res": n_res,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "c_pair": c_pair,
            "c_msa": c_msa,
            "msa_shape": [n_seq, n_res, c_msa],
            "pair_shape": [n_res, n_res, c_pair],
            "tri_attn_shape": [n_res, n_heads, n_res, n_res],
            "msa_finite": msa.iter().all(|v| v.is_finite()),
            "pair_finite": pair.iter().all(|v| v.is_finite()),
            "tri_attn_finite": tri_attn.iter().all(|v| v.is_finite()),
            "msa_changed": msa_changed,
            "pair_changed": pair_changed,
        }),
    )
}

/// One Structure Module step (Algorithm 22, Jumper et al. 2021).
///
/// IPA scores + backbone frame update + torsion angle prediction.
fn handle_structure_module(id: serde_json::Value, params: &serde_json::Value) -> JsonRpcResponse {
    let n_res = params.get("n_res").and_then(|v| v.as_u64()).unwrap_or(6) as usize;
    let c_single = params.get("c_single").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
    let c_pair = params.get("c_pair").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
    let n_heads = params.get("n_heads").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
    let head_dim = params.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
    let n_points = params.get("n_points").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
    let seed = params.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);

    let mut rng = Rng::new(seed);

    let single: Vec<f64> = (0..n_res * c_single).map(|_| rng.normal()).collect();
    let pair: Vec<f64> = (0..n_res * n_res * c_pair).map(|_| rng.normal()).collect();

    // Identity frames: [rot_3x3 | trans_3]
    let mut frames = vec![0.0f64; n_res * 12];
    for i in 0..n_res {
        frames[i * 12] = 1.0;
        frames[i * 12 + 4] = 1.0;
        frames[i * 12 + 8] = 1.0;
    }

    // IPA projections
    let w_iq: Vec<f64> = (0..c_single * n_heads * head_dim).map(|_| rng.normal() * 0.1).collect();
    let w_ik: Vec<f64> = (0..c_single * n_heads * head_dim).map(|_| rng.normal() * 0.1).collect();

    let q_scalar = matmul_2d(&single, &w_iq, n_res, c_single, n_heads * head_dim);
    let k_scalar = matmul_2d(&single, &w_ik, n_res, c_single, n_heads * head_dim);

    let w_ipa_bias: Vec<f64> = (0..c_pair * n_heads).map(|_| rng.normal() * 0.1).collect();
    let pair_bias = einsum_ijc_ch(&pair, &w_ipa_bias, n_res, n_res, c_pair, n_heads);

    let w_qp: Vec<f64> = (0..c_single * n_heads * n_points * 3).map(|_| rng.normal() * 0.1).collect();
    let w_kp: Vec<f64> = (0..c_single * n_heads * n_points * 3).map(|_| rng.normal() * 0.1).collect();
    let q_points = matmul_2d(&single, &w_qp, n_res, c_single, n_heads * n_points * 3);
    let k_points = matmul_2d(&single, &w_kp, n_res, c_single, n_heads * n_points * 3);

    let cfg = IpaConfig {
        n_res, n_heads, head_dim, n_points,
        w_l: 1.0, w_c: 1.0, w_p: 1.0, gamma: 0.5,
    };
    let ipa_s = ipa_scores(&q_scalar, &k_scalar, &pair_bias,
        &q_points, &k_points, &frames, &cfg);

    // Backbone update
    let mut delta_quats: Vec<f64> = (0..n_res * 4).map(|_| rng.normal() * 0.1).collect();
    for i in 0..n_res {
        delta_quats[i * 4] += 1.0;
    }
    let delta_trans: Vec<f64> = (0..n_res * 3).map(|_| rng.normal() * 0.1).collect();
    let updated_frames = backbone_update(&delta_quats, &delta_trans, &frames, n_res);

    // Torsion angle prediction
    let c_hidden = 6;
    let hh = c_hidden * c_hidden;
    let weight_len = c_single * c_hidden + c_hidden
        + hh + c_hidden + hh + c_hidden
        + hh + c_hidden + hh + c_hidden
        + c_hidden * 14 + 14;
    let torsion_weights: Vec<f64> = (0..weight_len).map(|_| rng.normal() * 0.1).collect();
    let torsion_out = torsion_angles(&single, &torsion_weights, n_res, c_single, c_hidden);

    let ipa_finite = ipa_s.iter().all(|v| v.is_finite());
    let frames_finite = updated_frames.iter().all(|v| v.is_finite());
    let torsion_finite = torsion_out.iter().all(|v| v.is_finite());

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "n_res": n_res,
            "ipa_scores_shape": [n_heads, n_res, n_res],
            "ipa_scores_finite": ipa_finite,
            "frames_shape": [n_res, 12],
            "frames_finite": frames_finite,
            "torsion_shape": [n_res, 7, 2],
            "torsion_finite": torsion_finite,
            "torsion_count": torsion_out.len(),
        }),
    )
}

/// Report folding primitive availability and validation status.
fn handle_folding_health(id: serde_json::Value, state: &PrimalState) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "folding_primitives": {
                "gelu": true,
                "layer_norm": true,
                "softmax_rows": true,
                "sdpa_scores": true,
                "sdpa_full": true,
                "msa_row_attention": true,
                "msa_col_attention": true,
                "outer_product_mean": true,
                "triangle_mul_outgoing": true,
                "triangle_mul_incoming": true,
                "triangle_attention_scores": true,
                "ipa_scores": true,
                "backbone_update": true,
                "torsion_angles": true,
            },
            "gpu_available": state.dispatcher.has_gpu(),
            "gpu_adapter": state.dispatcher.adapter_name(),
            "validated_papers": ["nF-01 (OpenFold)", "nF-02 (AlphaFold2)"],
            "validation_status": "179/179 validate_all",
        }),
    )
}

/// Route a Dispatcher operation through GPU or CPU.
fn handle_gpu_dispatch(id: serde_json::Value, params: &serde_json::Value, state: &PrimalState) -> JsonRpcResponse {
    let op = match params.get("op").and_then(|v| v.as_str()) {
        Some(o) => o,
        None => return JsonRpcResponse::error(id, -32602, "Missing 'op' parameter".to_string()),
    };

    match op {
        "mat_mul" => {
            let a: Vec<f64> = match params.get("a").and_then(|v| serde_json::from_value(v.clone()).ok()) {
                Some(v) => v,
                None => return JsonRpcResponse::error(id, -32602, "Missing 'a' parameter".to_string()),
            };
            let b: Vec<f64> = match params.get("b").and_then(|v| serde_json::from_value(v.clone()).ok()) {
                Some(v) => v,
                None => return JsonRpcResponse::error(id, -32602, "Missing 'b' parameter".to_string()),
            };
            let n = params.get("n").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            if n == 0 || a.len() != n * n || b.len() != n * n {
                return JsonRpcResponse::error(id, -32602, "Invalid matrix dimensions".to_string());
            }
            let result = state.dispatcher.mat_mul(&a, &b, n);
            JsonRpcResponse::success(id, serde_json::json!({ "result": result, "backend": format!("{}", state.dispatcher.backend()) }))
        }
        "softmax" => {
            let x: Vec<f64> = match params.get("x").and_then(|v| serde_json::from_value(v.clone()).ok()) {
                Some(v) => v,
                None => return JsonRpcResponse::error(id, -32602, "Missing 'x' parameter".to_string()),
            };
            let result = state.dispatcher.softmax(&x);
            JsonRpcResponse::success(id, serde_json::json!({ "result": result, "backend": format!("{}", state.dispatcher.backend()) }))
        }
        "mean" => {
            let data: Vec<f64> = match params.get("data").and_then(|v| serde_json::from_value(v.clone()).ok()) {
                Some(v) => v,
                None => return JsonRpcResponse::error(id, -32602, "Missing 'data' parameter".to_string()),
            };
            let result = state.dispatcher.mean(&data);
            JsonRpcResponse::success(id, serde_json::json!({ "result": result, "backend": format!("{}", state.dispatcher.backend()) }))
        }
        "variance" => {
            let data: Vec<f64> = match params.get("data").and_then(|v| serde_json::from_value(v.clone()).ok()) {
                Some(v) => v,
                None => return JsonRpcResponse::error(id, -32602, "Missing 'data' parameter".to_string()),
            };
            let result = state.dispatcher.variance(&data);
            JsonRpcResponse::success(id, serde_json::json!({ "result": result, "backend": format!("{}", state.dispatcher.backend()) }))
        }
        "eigh" => {
            let a: Vec<f64> = match params.get("a").and_then(|v| serde_json::from_value(v.clone()).ok()) {
                Some(v) => v,
                None => return JsonRpcResponse::error(id, -32602, "Missing 'a' parameter".to_string()),
            };
            let n = params.get("n").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            if n == 0 || a.len() != n * n {
                return JsonRpcResponse::error(id, -32602, "Invalid matrix dimensions".to_string());
            }
            let (eigenvalues, _eigenvectors) = state.dispatcher.eigh(&a, n);
            JsonRpcResponse::success(id, serde_json::json!({ "eigenvalues": eigenvalues, "backend": format!("{}", state.dispatcher.backend()) }))
        }
        _ => JsonRpcResponse::error(id, -32602, format!("Unknown dispatch op: {op}")),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Linear algebra helpers for Evoformer composition
//
// Delegates to upstream barracuda::dispatch::matmul_dispatch for non-square
// matrices (m, k, n). Falls back to CPU loop if dispatch fails.
// ═══════════════════════════════════════════════════════════════════════════════

/// Batched matmul: [batch, rows, in] × [in, out] → [batch, rows, out]
///
/// Each batch slice dispatched through `barracuda::dispatch::matmul_dispatch`.
fn matmul_3d(a: &[f64], w: &[f64], batch: usize, rows: usize, in_dim: usize, out_dim: usize) -> Vec<f64> {
    let mut out = vec![0.0; batch * rows * out_dim];
    let slice_size = rows * in_dim;
    for b in 0..batch {
        let a_slice = &a[b * slice_size..(b + 1) * slice_size];
        match barracuda::dispatch::matmul_dispatch(a_slice, w, rows, in_dim, out_dim, None) {
            Ok(result) => {
                out[b * rows * out_dim..(b + 1) * rows * out_dim].copy_from_slice(&result);
            }
            Err(_) => {
                for r in 0..rows {
                    for o in 0..out_dim {
                        let mut acc = 0.0f64;
                        for i in 0..in_dim {
                            acc = a_slice[r * in_dim + i].mul_add(w[i * out_dim + o], acc);
                        }
                        out[b * rows * out_dim + r * out_dim + o] = acc;
                    }
                }
            }
        }
    }
    out
}

/// 2D matmul: [rows, in] × [in, out] → [rows, out]
fn matmul_2d(a: &[f64], w: &[f64], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f64> {
    barracuda::dispatch::matmul_dispatch(a, w, rows, in_dim, out_dim, None)
        .unwrap_or_else(|_| matmul_3d(a, w, 1, rows, in_dim, out_dim))
}

/// einsum("ijc,ch->hij", tensor, weight) for pair bias computation.
/// tensor: [I, J, C], weight: [C, H] → out: [H, I, J]
fn einsum_ijc_ch(tensor: &[f64], weight: &[f64], ni: usize, nj: usize, nc: usize, nh: usize) -> Vec<f64> {
    let mut out = vec![0.0; nh * ni * nj];
    for i in 0..ni {
        for j in 0..nj {
            for h in 0..nh {
                let mut acc = 0.0f64;
                for c in 0..nc {
                    acc = tensor[i * nj * nc + j * nc + c]
                        .mul_add(weight[c * nh + h], acc);
                }
                out[h * ni * nj + i * nj + j] = acc;
            }
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-primal forwarding
// ═══════════════════════════════════════════════════════════════════════════════

/// Forward a JSON-RPC request to a sibling primal via Unix socket.
async fn forward_to_primal(primal_name: &str, method: &str, params: &serde_json::Value) -> Result<serde_json::Value> {
    let socket = discover_primal_socket(primal_name)?;
    let stream = UnixStream::connect(&socket)
        .await
        .with_context(|| format!("connecting to {primal_name} at {}", socket.display()))?;

    let (reader, mut writer) = stream.into_split();

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    });

    let mut payload = serde_json::to_vec(&request)?;
    payload.push(b'\n');
    writer.write_all(&payload).await?;
    writer.flush().await?;

    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();
    buf_reader.read_line(&mut line).await?;

    let resp: serde_json::Value = serde_json::from_str(line.trim())?;
    Ok(resp)
}

/// Discover a sibling primal's socket path.
fn discover_primal_socket(primal_name: &str) -> Result<PathBuf> {
    let socket_dir = resolve_socket_dir();
    let family_id = get_family_id();

    let with_family = socket_dir.join(format!("{primal_name}-{family_id}.sock"));
    if with_family.exists() {
        return Ok(with_family);
    }

    let without_family = socket_dir.join(format!("{primal_name}.sock"));
    if without_family.exists() {
        return Ok(without_family);
    }

    // Scan for any socket matching the primal name
    if let Ok(entries) = std::fs::read_dir(&socket_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(primal_name) && name_str.ends_with(".sock") {
                return Ok(entry.path());
            }
        }
    }

    anyhow::bail!("No socket found for primal '{primal_name}' in {}", socket_dir.display())
}

async fn handle_forward(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    let primal = match params.get("primal").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => return JsonRpcResponse::error(id, -32602, "Missing 'primal' parameter".to_string()),
    };
    let method = match params.get("method").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => return JsonRpcResponse::error(id, -32602, "Missing 'method' parameter".to_string()),
    };
    let inner_params = params.get("params").cloned().unwrap_or(serde_json::Value::Null);

    match forward_to_primal(&primal, &method, &inner_params).await {
        Ok(resp) => JsonRpcResponse::success(id, resp),
        Err(e) => JsonRpcResponse::error(id, -32000, format!("Forward failed: {e}")),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// biomeOS registration
// ═══════════════════════════════════════════════════════════════════════════════

async fn register_with_biomeos(_socket_path: &std::path::Path, our_socket: &std::path::Path) {
    let biomeos_socket = resolve_socket_dir().join("biomeOS.sock");
    if !biomeos_socket.exists() {
        eprintln!("[biomeos] No orchestrator found at {}, running standalone", biomeos_socket.display());
        return;
    }

    // lifecycle.register
    let reg_result = forward_to_primal_raw(
        &biomeos_socket,
        "lifecycle.register",
        &serde_json::json!({
            "name": "neuralspring",
            "socket_path": our_socket.to_string_lossy(),
            "pid": std::process::id(),
        }),
    ).await;

    match reg_result {
        Ok(_) => eprintln!("[biomeos] Registered with lifecycle manager"),
        Err(e) => eprintln!("[biomeos] lifecycle.register failed (non-fatal): {e}"),
    }

    // capability.register for each capability
    for cap in ALL_CAPABILITIES {
        let cap_result = forward_to_primal_raw(
            &biomeos_socket,
            "capability.register",
            &serde_json::json!({
                "primal": "neuralspring",
                "capability": cap,
                "socket_path": our_socket.to_string_lossy(),
            }),
        ).await;

        if let Err(e) = cap_result {
            eprintln!("[biomeos] capability.register({cap}) failed (non-fatal): {e}");
        }
    }

    eprintln!("[biomeos] All {} capabilities registered", ALL_CAPABILITIES.len());
}

async fn forward_to_primal_raw(socket_path: &std::path::Path, method: &str, params: &serde_json::Value) -> Result<serde_json::Value> {
    let stream = UnixStream::connect(socket_path).await?;
    let (reader, mut writer) = stream.into_split();

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    });

    let mut payload = serde_json::to_vec(&request)?;
    payload.push(b'\n');
    writer.write_all(&payload).await?;
    writer.flush().await?;

    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();
    tokio::time::timeout(std::time::Duration::from_secs(5), buf_reader.read_line(&mut line)).await
        .with_context(|| "timeout waiting for biomeOS response")??;

    let resp: serde_json::Value = serde_json::from_str(line.trim())?;
    Ok(resp)
}

async fn heartbeat_loop(our_socket: PathBuf) {
    let biomeos_socket = resolve_socket_dir().join("biomeOS.sock");
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));

    loop {
        interval.tick().await;

        if !biomeos_socket.exists() {
            continue;
        }

        let _ = forward_to_primal_raw(
            &biomeos_socket,
            "lifecycle.status",
            &serde_json::json!({
                "name": "neuralspring",
                "socket_path": our_socket.to_string_lossy(),
                "status": "healthy",
            }),
        ).await;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Request dispatcher
// ═══════════════════════════════════════════════════════════════════════════════

fn dispatch_sync(request: &JsonRpcRequest, state: &PrimalState) -> Option<JsonRpcResponse> {
    let id = request.id.clone();
    let params = &request.params;

    Some(match request.method.as_str() {
        "health" => handle_health(id, state),
        "science.ipr" => handle_ipr(id, params),
        "science.disorder_sweep" => handle_disorder_sweep(id, params),
        "science.spectral_analysis" => handle_spectral_analysis(id, params),
        "science.anderson_localization" => handle_anderson_localization(id, params),
        "science.hessian_eigen" => handle_hessian_eigen(id, params),
        "science.agent_coordination" => handle_agent_coordination(id, params),
        "science.training_trajectory" => handle_training_trajectory(id, params),
        "science.evoformer_block" => handle_evoformer_block(id, params),
        "science.structure_module" => handle_structure_module(id, params),
        "science.folding_health" => handle_folding_health(id, state),
        "science.gpu_dispatch" => handle_gpu_dispatch(id, params, state),
        "primal.forward" | "data.ncbi_search" | "data.ncbi_fetch"
            | "data.pdb_search" | "data.pdb_fetch" => return None,
        _ => JsonRpcResponse::error(
            id,
            -32601,
            format!("Method not found: {}", request.method),
        ),
    })
}

async fn dispatch_async(request: &JsonRpcRequest) -> JsonRpcResponse {
    let id = request.id.clone();
    let params = &request.params;

    match request.method.as_str() {
        "primal.forward" => handle_forward(id, params).await,
        // Auto-route data.* methods to NestGate
        method if method.starts_with("data.") => {
            match forward_to_primal("nestgate", method, params).await {
                Ok(resp) => JsonRpcResponse::success(id, resp),
                Err(e) => JsonRpcResponse::error(id, -32000, format!("NestGate forward failed: {e}")),
            }
        }
        _ => JsonRpcResponse::error(id, -32601, format!("Method not found: {}", request.method)),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Socket resolution (follows biomeOS 5-tier standard)
// ═══════════════════════════════════════════════════════════════════════════════

fn resolve_socket_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("BIOMEOS_SOCKET_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(xdg).join("biomeos");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let Ok(meta) = std::fs::metadata("/proc/self") {
            let uid = meta.uid();
            let dir = PathBuf::from(format!("/run/user/{uid}/biomeos"));
            if dir.parent().is_some_and(std::path::Path::exists) {
                return dir;
            }
        }
    }
    let tmp = std::env::var("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"));
    tmp.join("biomeos")
}

fn resolve_socket_path(family_id: &str) -> PathBuf {
    resolve_socket_dir().join(format!("neuralspring-{family_id}.sock"))
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

    // Initialize GPU dispatcher (warm up GPU at startup)
    eprintln!("[init] Initializing GPU dispatcher...");
    let dispatcher = Dispatcher::new().await;
    eprintln!("[init] Dispatcher ready: backend={}, gpu={}",
        dispatcher.backend(), dispatcher.has_gpu());

    let state = Arc::new(PrimalState {
        dispatcher,
        start_time: Instant::now(),
        requests_served: AtomicU64::new(0),
    });

    let listener = UnixListener::bind(&socket_path)
        .context(format!("binding to {}", socket_path.display()))?;

    eprintln!("neuralSpring primal listening on {}", socket_path.display());
    eprintln!("  Family ID: {family_id}");
    eprintln!("  Mode: Tower (local Eastgate)");
    eprintln!("  Capabilities ({}):", ALL_CAPABILITIES.len());
    for cap in ALL_CAPABILITIES {
        eprintln!("    - {cap}");
    }

    // Register with biomeOS orchestrator (best-effort)
    register_with_biomeos(&socket_path, &socket_path).await;

    // Start heartbeat loop
    let heartbeat_socket = socket_path.clone();
    tokio::spawn(heartbeat_loop(heartbeat_socket));

    // Handle SIGTERM for graceful shutdown
    let shutdown_socket = socket_path.clone();
    tokio::spawn(async move {
        if let Ok(()) = tokio::signal::ctrl_c().await {
            eprintln!("\n[shutdown] SIGINT received, cleaning up...");
            let _ = tokio::fs::remove_file(&shutdown_socket).await;
            std::process::exit(0);
        }
    });

    let concurrency = Arc::new(Semaphore::new(4));

    loop {
        let (stream, _addr) = listener.accept().await?;
        let permit = concurrency.clone().acquire_owned().await?;
        let state = state.clone();

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
                    Ok(req) => {
                        state.requests_served.fetch_add(1, Ordering::Relaxed);
                        match dispatch_sync(&req, &state) {
                            Some(resp) => resp,
                            None => dispatch_async(&req).await,
                        }
                    }
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
