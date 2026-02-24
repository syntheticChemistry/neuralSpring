// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: HMM transition matrix chain (Papers 016–018).
//!
//! Validates GPU `Tensor::matmul` for HMM forward-pass operations:
//! batch alpha × transition, emission scoring, and stationary
//! distribution dot products.  All shapes are non-square to avoid S-14.
//!
//! ## S-14 workaround
//!
//! HMM state counts are small (N < 32).  This validator uses batched
//! alpha vectors (T×N with T ≠ N) so the matmul operands are never
//! both square.
//!
//! ## Provenance
//!
//! CPU baseline: `validate_barracuda_hmm` (14 checks, hmm.rs forward/backward)
//! GPU: `Tensor::matmul` for transition chain.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::needless_range_loop
)]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_tensor;
use neural_spring::tolerances;
use neural_spring::validation::{gpu_readback, ValidationHarness};
use std::sync::Arc;

type Dev = Arc<barracuda::device::WgpuDevice>;

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };
    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_gpu_hmm");

    validate_batch_alpha_transition(&mut h, &device);
    validate_emission_scoring(&mut h, &device);
    validate_stationary_dot(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

/// Batch forward: `alpha_batch` (T×N) × A^T (N×N) where T=50, N=3.
fn validate_batch_alpha_transition(h: &mut ValidationHarness, device: &Dev) {
    let n_time = 50_usize;
    let n_states = 3_usize;

    let trans = [0.7_f64, 0.2, 0.1, 0.1, 0.6, 0.3, 0.2, 0.2, 0.6];

    let mut alpha_f64 = vec![0.0_f64; n_time * n_states];
    for t in 0..n_time {
        for j in 0..n_states {
            alpha_f64[t * n_states + j] = ((t * n_states + j + 1) as f64) * 0.01;
        }
    }

    let mut cpu_result = vec![0.0_f64; n_time * n_states];
    for t in 0..n_time {
        for j in 0..n_states {
            for k in 0..n_states {
                cpu_result[t * n_states + j] +=
                    alpha_f64[t * n_states + k] * trans[j * n_states + k];
            }
        }
    }

    let alpha_f32: Vec<f32> = alpha_f64.iter().map(|&x| x as f32).collect();
    let trans_f32: Vec<f32> = trans.iter().map(|&x| x as f32).collect();

    let alpha_t = gpu_tensor!(h, &alpha_f32, &[n_time, n_states], device);
    let trans_t = gpu_tensor!(h, &trans_f32, &[n_states, n_states], device);
    let trans_t_t = match trans_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };
    let out_t = match alpha_t.matmul(&trans_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul alpha×A^T: {e}"), false);
            return;
        }
    };
    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };

    let max_diff: f64 = out
        .iter()
        .zip(cpu_result.iter())
        .map(|(&g, &c)| (f64::from(g) - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        &format!("batch alpha×A^T: max diff GPU vs CPU ({max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );

    let sum_first_row: f64 = (0..n_states).map(|j| f64::from(out[j])).sum();
    h.check_bool(
        &format!("forward product row sum finite ({sum_first_row:.4})"),
        sum_first_row.is_finite() && sum_first_row > 0.0,
    );
}

/// Emission scoring: `obs_features` (T×M) × `emission_weights` (M×N).
fn validate_emission_scoring(h: &mut ValidationHarness, device: &Dev) {
    let n_time = 40_usize;
    let n_features = 5_usize;
    let n_states = 3_usize;

    let obs_f64: Vec<f64> = (0..n_time * n_features)
        .map(|i| ((i % 7) as f64) * 0.1)
        .collect();
    let weights_f64: Vec<f64> = (0..n_features * n_states)
        .map(|i| ((i + 1) as f64) * 0.2)
        .collect();

    let mut cpu_scores = vec![0.0_f64; n_time * n_states];
    for t in 0..n_time {
        for j in 0..n_states {
            for k in 0..n_features {
                cpu_scores[t * n_states + j] +=
                    obs_f64[t * n_features + k] * weights_f64[k * n_states + j];
            }
        }
    }

    let obs_f32: Vec<f32> = obs_f64.iter().map(|&x| x as f32).collect();
    let wt_f32: Vec<f32> = weights_f64.iter().map(|&x| x as f32).collect();

    let obs_t = gpu_tensor!(h, &obs_f32, &[n_time, n_features], device);
    let wt_t = gpu_tensor!(h, &wt_f32, &[n_features, n_states], device);
    let out_t = match obs_t.matmul(&wt_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul obs×weights: {e}"), false);
            return;
        }
    };
    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };

    let max_diff: f64 = out
        .iter()
        .zip(cpu_scores.iter())
        .map(|(&g, &c)| (f64::from(g) - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        &format!("emission scoring: max diff ({max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

/// Stationary distribution: pi (1×N) × A (N×N) ≈ pi.
fn validate_stationary_dot(h: &mut ValidationHarness, device: &Dev) {
    let pi: Vec<f32> = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let trans: Vec<f32> = vec![0.7, 0.2, 0.1, 0.1, 0.6, 0.3, 0.2, 0.2, 0.6];

    let pi_batch: Vec<f32> = pi.iter().cycle().take(3 * 10).copied().collect();
    let pi_t = gpu_tensor!(h, &pi_batch, &[10, 3], device);
    let a_t = gpu_tensor!(h, &trans, &[3, 3], device);

    let out_t = match pi_t.matmul(&a_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul pi×A: {e}"), false);
            return;
        }
    };
    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };

    let max_diff: f64 = out[..3]
        .iter()
        .zip(pi.iter())
        .map(|(&g, &p)| (f64::from(g) - f64::from(p)).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        &format!("πA ≈ π (max diff {max_diff:.2e})"),
        max_diff,
        tolerances::TENSOR_MATMUL_F32,
    );
}

fn validate_determinism(h: &mut ValidationHarness, device: &Dev) {
    let n_time = 50_usize;
    let n_states = 3_usize;
    let alpha: Vec<f32> = (0..n_time * n_states).map(|i| (i as f32) * 0.01).collect();
    let trans: Vec<f32> = vec![0.7, 0.2, 0.1, 0.1, 0.6, 0.3, 0.2, 0.2, 0.6];

    let run = |_run_id: u32| -> Option<Vec<f32>> {
        let a = Tensor::from_data(&alpha, vec![n_time, n_states], device.clone()).ok()?;
        let t = Tensor::from_data(&trans, vec![n_states, n_states], device.clone()).ok()?;
        let tt = t.transpose().ok()?;
        let out = a.matmul(&tt).ok()?;
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
