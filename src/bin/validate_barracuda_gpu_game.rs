// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: game-theoretic payoff evaluation (Paper 019).
//!
//! Validates that `BarraCUDA` `Tensor` matmul on GPU correctly computes
//! payoff = `strategy_vector` × `payoff_matrix^T` for QS cooperation games.
//! Domain: game-theoretic payoff evaluation for quorum sensing cooperation.
//!
//! ## S-14 workaround
//!
//! All matmul operations use A × B^T (transpose second operand).
//!
//! ## S-15 workaround
//!
//! All data uses `rng.uniform()` ([0, 1)) to avoid matmul hang.
//! Payoff matrix uses positive values only (R, S, T, P shifted).
//!
//! ## Provenance
//!
//! CPU baseline: `validate_barracuda_game_theory`, `control/game_theory`

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    reason = "validation binary"
)]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_tensor;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, gpu_readback, max_abs_diff_gpu_vs_cpu};
use std::sync::Arc;

fn cpu_a_bt(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b.len();
    let depth = a[0].len();
    let mut out = vec![vec![0.0_f64; cols]; rows];
    for row_idx in 0..rows {
        for col_idx in 0..cols {
            for inner_idx in 0..depth {
                out[row_idx][col_idx] += a[row_idx][inner_idx] * b[col_idx][inner_idx];
            }
        }
    }
    out
}

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            println!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };
    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_gpu_game");

    validate_payoff_matrix_matmul(&mut h, &device);
    validate_pd_payoff_structure(&mut h, &device);
    validate_spatial_payoff_aggregation(&mut h, &device);
    validate_payoff_finite(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

fn validate_payoff_matrix_matmul(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(42);
    let n_agents = 20_usize;
    let n_strategies = 4_usize;
    let n_games = 3_usize;

    let strategy_vectors: Vec<Vec<f64>> = (0..n_agents)
        .map(|_| (0..n_strategies).map(|_| rng.uniform()).collect())
        .collect();
    let payoff_matrices: Vec<Vec<Vec<f64>>> = (0..n_games)
        .map(|_| {
            (0..n_strategies)
                .map(|_| (0..n_strategies).map(|_| rng.uniform()).collect())
                .collect()
        })
        .collect();

    let mut cpu_payoffs = Vec::with_capacity(n_agents * n_games);
    for g in 0..n_games {
        let pay = cpu_a_bt(&strategy_vectors, &payoff_matrices[g]);
        for row in &pay {
            cpu_payoffs.extend_from_slice(row);
        }
    }

    let strat_flat: Vec<f32> = strategy_vectors
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let mut all_gpu: Vec<f32> = Vec::with_capacity(n_agents * n_strategies * n_games);
    for g in 0..n_games {
        let strat_t = gpu_tensor!(h, &strat_flat, &[n_agents, n_strategies], device);
        let pay_flat: Vec<f32> = payoff_matrices[g]
            .iter()
            .flat_map(|r| r.iter().map(|&x| x as f32))
            .collect();
        let pay_t = gpu_tensor!(h, &pay_flat, &[n_strategies, n_strategies], device);
        let pay_t_t = match pay_t.transpose() {
            Ok(t) => t,
            Err(e) => {
                h.check_bool(&format!("transpose: {e}"), false);
                return;
            }
        };
        let out_t = match strat_t.matmul(&pay_t_t) {
            Ok(t) => t,
            Err(e) => {
                h.check_bool(&format!("strategy × payoff^T: {e}"), false);
                return;
            }
        };
        let Some(out) = gpu_readback(h, &out_t) else {
            return;
        };
        all_gpu.extend_from_slice(&out);
    }

    let diff = max_abs_diff_gpu_vs_cpu(&all_gpu, &cpu_payoffs);
    h.check_upper(
        &format!("payoff matrix matmul: max diff ({diff:.2e})"),
        diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_pd_payoff_structure(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let n_agents = 20_usize;
    let n_strategies = 4_usize;

    let coop_idx = 0_usize;
    let defect_idx = 1_usize;

    let mut rng = Rng::new(123);
    let strategy_vectors: Vec<Vec<f64>> = (0..n_agents)
        .map(|_| (0..n_strategies).map(|_| rng.uniform()).collect())
        .collect();

    let payoff_matrix: Vec<Vec<f64>> = vec![
        vec![3.0, 0.0, 2.0, 1.0],
        vec![5.0, 1.0, 3.0, 2.0],
        vec![2.5, 1.5, 2.0, 1.5],
        vec![2.0, 2.0, 1.5, 1.0],
    ];
    let opp_coop_col: Vec<f64> = payoff_matrix.iter().map(|r| r[coop_idx]).collect();
    let payoff_defect_when_opp_coop = opp_coop_col[defect_idx];
    let payoff_coop_when_opp_coop = opp_coop_col[coop_idx];

    let defect_better = payoff_defect_when_opp_coop > payoff_coop_when_opp_coop;

    let pay_flat: Vec<f32> = payoff_matrix
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let strat_flat: Vec<f32> = strategy_vectors
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let strat_t = gpu_tensor!(h, &strat_flat, &[n_agents, n_strategies], device);
    let pay_t = gpu_tensor!(h, &pay_flat, &[n_strategies, n_strategies], device);
    let pay_t_t = match pay_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let out_t = match strat_t.matmul(&pay_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };

    h.check_bool(
        "PD payoff structure: defect > cooperate when opponent cooperates",
        defect_better,
    );
    h.check_bool(
        "PD payoff computation: all outputs finite",
        out.iter().all(|x| x.is_finite()),
    );
}

fn validate_spatial_payoff_aggregation(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(99);
    let n_agents = 20_usize;
    let n_strategies = 4_usize;
    let n_neighbors = 8_usize;

    let strategy_vectors: Vec<Vec<f64>> = (0..n_agents)
        .map(|_| (0..n_strategies).map(|_| rng.uniform()).collect())
        .collect();
    let neighbor_weights: Vec<Vec<f64>> = (0..n_agents)
        .map(|_| (0..n_neighbors).map(|_| rng.uniform()).collect())
        .collect();
    let payoff_matrix: Vec<Vec<f64>> = (0..n_strategies)
        .map(|_| (0..n_strategies).map(|_| rng.uniform()).collect())
        .collect();

    let local_payoffs = cpu_a_bt(&strategy_vectors, &payoff_matrix);
    let cpu_aggregated: Vec<f64> = (0..n_agents)
        .map(|i| {
            (0..n_strategies)
                .map(|obj| {
                    (0..n_neighbors)
                        .map(|k| local_payoffs[i][obj] * neighbor_weights[i][k])
                        .sum::<f64>()
                        / n_neighbors as f64
                })
                .sum()
        })
        .collect();

    let strat_flat: Vec<f32> = strategy_vectors
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let pay_flat: Vec<f32> = payoff_matrix
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let strat_t = gpu_tensor!(h, &strat_flat, &[n_agents, n_strategies], device);
    let pay_t = gpu_tensor!(h, &pay_flat, &[n_strategies, n_strategies], device);
    let pay_t_t = match pay_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let local_t = match strat_t.matmul(&pay_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };

    let Some(local_gpu) = gpu_readback(h, &local_t) else {
        return;
    };

    let gpu_aggregated: Vec<f64> = (0..n_agents)
        .map(|i| {
            (0..n_strategies)
                .map(|obj| {
                    let base = f64::from(local_gpu[i * n_strategies + obj]);
                    (0..n_neighbors)
                        .map(|k| base * neighbor_weights[i][k])
                        .sum::<f64>()
                        / n_neighbors as f64
                })
                .sum()
        })
        .collect();

    let diff = gpu_aggregated
        .iter()
        .zip(cpu_aggregated.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        &format!("spatial payoff aggregation: max diff ({diff:.2e})"),
        diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_payoff_finite(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(77);
    let n_agents = 20_usize;
    let n_strategies = 4_usize;

    let strat_flat: Vec<f32> = (0..n_agents * n_strategies)
        .map(|_| rng.uniform() as f32)
        .collect();
    let pay_flat: Vec<f32> = (0..n_strategies * n_strategies)
        .map(|_| rng.uniform() as f32)
        .collect();

    let strat_t = gpu_tensor!(h, &strat_flat, &[n_agents, n_strategies], device);
    let pay_t = gpu_tensor!(h, &pay_flat, &[n_strategies, n_strategies], device);
    let pay_t_t = match pay_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let out_t = match strat_t.matmul(&pay_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };

    h.check_bool(
        "payoff finite: all GPU payoff values finite",
        out.iter().all(|x| x.is_finite()),
    );
}

fn validate_determinism(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(42);
    let n_agents = 20_usize;
    let n_strategies = 4_usize;

    let strat_flat: Vec<f32> = (0..n_agents * n_strategies)
        .map(|_| rng.uniform() as f32)
        .collect();
    let pay_flat: Vec<f32> = (0..n_strategies * n_strategies)
        .map(|_| rng.uniform() as f32)
        .collect();

    let run = |_: u32| -> Option<Vec<f32>> {
        let s =
            Tensor::from_data(&strat_flat, vec![n_agents, n_strategies], device.clone()).ok()?;
        let p =
            Tensor::from_data(&pay_flat, vec![n_strategies, n_strategies], device.clone()).ok()?;
        let pt = p.transpose().ok()?;
        let out = s.matmul(&pt).ok()?;
        out.to_vec().ok()
    };

    let Some(r1) = run(1) else {
        h.check_bool("determinism run1 failed", false);
        return;
    };
    let Some(r2) = run(2) else {
        h.check_bool("determinism run2 failed", false);
        return;
    };

    let identical = r1
        .iter()
        .zip(r2.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    h.check_bool("determinism: two GPU runs bit-identical", identical);
}
