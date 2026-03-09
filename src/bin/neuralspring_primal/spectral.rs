// SPDX-License-Identifier: AGPL-3.0-or-later

//! baseCamp spectral analysis RPC handlers.
//!
//! Each handler takes a JSON-RPC `id` + `params` and returns a
//! [`JsonRpcResponse`].  Computation is fully self-contained —
//! no cross-primal calls needed.

use neural_spring::agent_coordination::{coordination_spectral_analysis, generate_lattice_agents};
use neural_spring::anderson_localization::{
    anderson_hamiltonian_random, disorder_sweep, ipr, jacobi_eigh, mean_ipr,
};
use neural_spring::eigh::eigh_householder_qr;
use neural_spring::rng::Rng;

use super::rpc::JsonRpcResponse;
use super::PrimalState;

pub fn handle_health(id: serde_json::Value, state: &PrimalState) -> JsonRpcResponse {
    use std::sync::atomic::Ordering;

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
            "primal": env!("CARGO_PKG_NAME"),
            "version": env!("CARGO_PKG_VERSION"),
            "capabilities": super::ALL_CAPABILITIES,
            "hardware": hardware,
            "stats": {
                "requests_served": served,
                "uptime_seconds": uptime,
            }
        }),
    )
}

pub fn handle_ipr(id: serde_json::Value, params: &serde_json::Value) -> JsonRpcResponse {
    let wavefunction: Vec<f64> = match serde_json::from_value(
        params
            .get("wavefunction")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    ) {
        Ok(v) => v,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                super::rpc::error_code::INVALID_PARAMS,
                format!("Invalid params: {e}"),
            )
        }
    };
    let result = ipr(&wavefunction);
    JsonRpcResponse::success(id, serde_json::json!({ "ipr": result }))
}

pub fn handle_disorder_sweep(id: serde_json::Value, params: &serde_json::Value) -> JsonRpcResponse {
    let n = params_usize(params, "lattice_size", 20);
    let t = params_f64(params, "hopping", 1.0);
    let w_vals = params_f64_vec(params, "disorder_values", &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0]);
    let seed = params_u64(params, "seed", 42);

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

pub fn handle_spectral_analysis(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    let n = params_usize(params, "dim", 16);
    let w = params_f64(params, "disorder", 2.0);
    let seed = params_u64(params, "seed", 42);

    let mut rng = Rng::new(seed);
    let h = anderson_hamiltonian_random(n, 1.0, w, &mut rng);
    let decomp = eigh_householder_qr(&h, n);

    let ipr_val = mean_ipr(&decomp.eigenvectors, n);
    let mut evals = decomp.eigenvalues.clone();
    evals.sort_by(|a, b| a.total_cmp(b));
    let lsr = neural_spring::weight_spectral::level_spacing_ratio(&evals);
    let bw = neural_spring::weight_spectral::spectral_bandwidth(&evals);
    let cond = neural_spring::weight_spectral::spectral_condition_number(&evals);
    let phase = neural_spring::weight_spectral::classify_phase(lsr);

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "eigenvalues": evals,
            "mean_ipr": ipr_val,
            "level_spacing_ratio": lsr,
            "bandwidth": bw,
            "condition_number": cond,
            "phase": format!("{phase}"),
            "dim": n,
            "disorder": w,
        }),
    )
}

pub fn handle_anderson_localization(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    let n = params_usize(params, "lattice_size", 20);
    let t = params_f64(params, "hopping", 1.0);
    let w_vals = params_f64_vec(params, "disorder_values", &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0]);
    let seed = params_u64(params, "seed", 42);

    let mut rng = Rng::new(seed);
    let mut results = Vec::new();

    for &w in &w_vals {
        let h = anderson_hamiltonian_random(n, t, w, &mut rng);
        let (eigenvalues, eigenvectors) = jacobi_eigh(&h, n);
        let ipr_val = mean_ipr(&eigenvectors, n);
        let mut sorted_evals = eigenvalues.clone();
        sorted_evals.sort_by(|a, b| a.total_cmp(b));
        let lsr = neural_spring::weight_spectral::level_spacing_ratio(&sorted_evals);

        let bw = neural_spring::weight_spectral::spectral_bandwidth(&sorted_evals);
        let cond = neural_spring::weight_spectral::spectral_condition_number(&sorted_evals);
        let phase = neural_spring::weight_spectral::classify_phase(lsr);

        results.push(serde_json::json!({
            "disorder": w,
            "mean_ipr": ipr_val,
            "level_spacing_ratio": lsr,
            "bandwidth": bw,
            "condition_number": cond,
            "phase": format!("{phase}"),
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

#[expect(clippy::cast_precision_loss, reason = "validation binary")]
pub fn handle_hessian_eigen(id: serde_json::Value, params: &serde_json::Value) -> JsonRpcResponse {
    let n = params_usize(params, "dim", 20);
    let surface = params
        .get("surface_type")
        .and_then(|v| v.as_str())
        .unwrap_or("quadratic");

    let hessian: Vec<f64> = match surface {
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
    evals.sort_by(|a, b| a.total_cmp(b));
    let entropy = neural_spring::primitives::shannon_entropy(&evals);
    let trace: f64 = evals.iter().sum();
    let cond = neural_spring::weight_spectral::spectral_condition_number(&evals);
    let phase = neural_spring::weight_spectral::classify_phase(
        neural_spring::weight_spectral::level_spacing_ratio(&evals),
    );

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "eigenvalues": evals,
            "spectral_entropy": entropy,
            "trace": trace,
            "condition_number": cond,
            "bandwidth": neural_spring::weight_spectral::spectral_bandwidth(&evals),
            "phase": format!("{phase}"),
            "dim": n,
            "surface_type": surface,
        }),
    )
}

pub fn handle_agent_coordination(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    let n = params_usize(params, "n_agents", 16);
    let dim = params_usize(params, "dimensions", 2);
    let comm = params_f64(params, "comm_range", 3.0);
    let disorder_vals = params_f64_vec(params, "disorder_values", &[0.0, 0.5, 1.0, 2.0]);
    let seed = params_u64(params, "seed", 42);
    let cap_var = params_f64(params, "capability_variance", 1.0);

    let mut rng = Rng::new(seed);
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

#[expect(clippy::cast_precision_loss, reason = "validation binary")]
pub fn handle_training_trajectory(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    let dim = params_usize(params, "dim", 16);
    let n_epochs = params_usize(params, "n_epochs", 20);
    let seed = params_u64(params, "seed", 42);

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
        evals.sort_by(|a, b| a.total_cmp(b));
        let entropy = neural_spring::primitives::shannon_entropy(&evals);
        let lsr = neural_spring::weight_spectral::level_spacing_ratio(&evals);
        let phase = neural_spring::weight_spectral::classify_phase(lsr);

        trajectory.push(serde_json::json!({
            "epoch": epoch,
            "alpha": alpha,
            "mean_ipr": ipr_val,
            "spectral_entropy": entropy,
            "level_spacing_ratio": lsr,
            "bandwidth": neural_spring::weight_spectral::spectral_bandwidth(&evals),
            "condition_number": neural_spring::weight_spectral::spectral_condition_number(&evals),
            "phase": format!("{phase}"),
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

// ═══════════════════════════════════════════════════════════════════
// Typed parameter extraction (replaces raw unwrap_or chains)
// ═══════════════════════════════════════════════════════════════════

fn params_usize(params: &serde_json::Value, key: &str, default: u64) -> usize {
    params.get(key).and_then(|v| v.as_u64()).unwrap_or(default) as usize
}

fn params_f64(params: &serde_json::Value, key: &str, default: f64) -> f64 {
    params.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

fn params_u64(params: &serde_json::Value, key: &str, default: u64) -> u64 {
    params.get(key).and_then(|v| v.as_u64()).unwrap_or(default)
}

fn params_f64_vec(params: &serde_json::Value, key: &str, default: &[f64]) -> Vec<f64> {
    params
        .get(key)
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_else(|| default.to_vec())
}
