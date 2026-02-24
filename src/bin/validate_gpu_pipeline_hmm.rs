// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU pipeline validation: `HmmBatchForwardF64` (`BarraCUDA`) + CPU mean (Papers 016-018).
//!
//! Replaces raw wgpu pipeline with typed `BarraCUDA` op: `barracuda::ops::bio::HmmBatchForwardF64`.
//! Stage 1: HmmBatchForwardF64.dispatch → `log_lik_out[n_seqs]` (f64).
//! Stage 2: CPU mean over `log_lik_out`.
//!
//! ## Pipeline
//!
//! ```text
//! Upload log_trans, log_emit, log_pi, observations (once)
//!   ↓
//! HmmBatchForwardF64.dispatch → log_lik_out[n_seqs]
//!   ↓
//! CPU mean(log_lik_out) → scalar
//! ```
//!
//! ## Provenance
//!
//! Typed op: `barracuda::ops::bio::HmmBatchForwardF64` (f64).
//! Validates: `BarraCUDA` HMM forward API with mean log-likelihood summary.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::explicit_iter_loop
)]

use barracuda::ops::bio::HmmBatchForwardF64;
use neural_spring::gpu::Gpu;
use neural_spring::hmm::Hmm;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use wgpu::util::DeviceExt;

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

    let mut h = ValidationHarness::new("gpu_pipeline_hmm");

    validate_small(&mut h, &gpu);
    validate_larger(&mut h, &gpu);
    validate_single_state(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_log_lik(hmm: &Hmm, obs_batch: &[Vec<usize>]) -> f64 {
    let mut sum = 0.0_f64;
    for obs in obs_batch {
        let (_, log_lik) = hmm.forward(obs);
        sum += log_lik;
    }
    if obs_batch.is_empty() {
        0.0
    } else {
        sum / obs_batch.len() as f64
    }
}

// ── BarraCUDA typed op + CPU mean ──────────────────────────────────

struct HmmF64Params {
    n_states: u32,
    n_symbols: u32,
    log_trans: Vec<f64>,
    log_emit: Vec<f64>,
    log_pi: Vec<f64>,
}

fn hmm_to_f64_params(hmm: &Hmm) -> HmmF64Params {
    let n_states = hmm.num_states();
    let n_symbols = hmm.num_symbols();
    let log_trans: Vec<f64> = hmm.transition.iter().map(|&p| p.ln()).collect();
    let log_emit: Vec<f64> = hmm.emission.iter().map(|&p| p.ln()).collect();
    let log_pi: Vec<f64> = hmm.initial.iter().map(|&p| p.ln()).collect();
    HmmF64Params {
        n_states: n_states as u32,
        n_symbols: n_symbols as u32,
        log_trans,
        log_emit,
        log_pi,
    }
}

fn gpu_hmm_mean_log_lik(
    gpu: &Gpu,
    op: &HmmBatchForwardF64,
    params: &HmmF64Params,
    obs_batch: &[Vec<usize>],
) -> Result<f64, String> {
    let device = gpu.device();
    let n_seqs = obs_batch.len() as u32;
    let n_steps = obs_batch.iter().map(Vec::len).max().unwrap_or(0) as u32;

    let mut obs_flat: Vec<u32> = Vec::with_capacity((n_seqs * n_steps) as usize);
    for seq in obs_batch {
        for &o in seq {
            obs_flat.push(o as u32);
        }
        let pad_len = (n_steps as usize).saturating_sub(seq.len());
        obs_flat.extend(std::iter::repeat_n(0, pad_len));
    }

    let log_trans_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_log_trans"),
        contents: bytemuck::cast_slice(&params.log_trans),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let log_emit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_log_emit"),
        contents: bytemuck::cast_slice(&params.log_emit),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let log_pi_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_log_pi"),
        contents: bytemuck::cast_slice(&params.log_pi),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let obs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_observations"),
        contents: bytemuck::cast_slice(&obs_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let alpha_size = u64::from(n_seqs) * u64::from(n_steps) * u64::from(params.n_states) * 8;
    let log_alpha_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_log_alpha"),
        size: alpha_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let log_lik_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_log_lik"),
        size: u64::from(n_seqs) * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(
        params.n_states,
        params.n_symbols,
        n_steps,
        n_seqs,
        &log_trans_buf,
        &log_emit_buf,
        &log_pi_buf,
        &obs_buf,
        &log_alpha_buf,
        &log_lik_buf,
    )
    .map_err(|e| format!("HMM dispatch: {e}"))?;

    let log_lik = gpu.read_buffer_f64(&log_lik_buf, n_seqs as usize)?;
    let mean = log_lik.iter().sum::<f64>() / log_lik.len() as f64;
    Ok(mean)
}

// ── Validation functions ───────────────────────────────────────────

fn make_hmm_small(rng: &mut Rng) -> (Hmm, Vec<Vec<usize>>) {
    let hmm = Hmm::new(
        vec![
            vec![0.7, 0.2, 0.1],
            vec![0.2, 0.6, 0.2],
            vec![0.1, 0.2, 0.7],
        ],
        vec![
            vec![0.4, 0.3, 0.3],
            vec![0.2, 0.5, 0.3],
            vec![0.3, 0.3, 0.4],
        ],
        vec![0.33, 0.34, 0.33],
    );
    let mut obs_batch = Vec::with_capacity(4);
    for _ in 0..4 {
        let (_, obs) = hmm.generate_sequence(5, rng);
        obs_batch.push(obs);
    }
    (hmm, obs_batch)
}

fn validate_small(h: &mut ValidationHarness, gpu: &Gpu) {
    let dev = Arc::clone(gpu.wgpu_device());
    let op = match HmmBatchForwardF64::new(dev) {
        Ok(o) => o,
        Err(e) => {
            h.check_bool(
                &format!("HMM small: HmmBatchForwardF64::new failed — {e}"),
                false,
            );
            return;
        }
    };

    let mut rng = Rng::new(42);
    let (hmm, obs_batch) = make_hmm_small(&mut rng);
    let params = hmm_to_f64_params(&hmm);
    let cpu_mean = cpu_mean_log_lik(&hmm, &obs_batch);

    match gpu_hmm_mean_log_lik(gpu, &op, &params, &obs_batch) {
        Ok(gpu_mean) => {
            h.check_bool(
                &format!("HMM small: GPU mean finite ({gpu_mean:.6})"),
                gpu_mean.is_finite(),
            );
            h.check_abs(
                &format!("HMM small: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_HMM_ALPHA_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("HMM small: dispatch failed — {e}"), false);
        }
    }
}

fn make_hmm_larger(rng: &mut Rng) -> (Hmm, Vec<Vec<usize>>) {
    let hmm = Hmm::new(
        vec![
            vec![0.8, 0.1, 0.05, 0.05],
            vec![0.05, 0.85, 0.05, 0.05],
            vec![0.05, 0.05, 0.8, 0.1],
            vec![0.05, 0.05, 0.1, 0.8],
        ],
        vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.3, 0.3, 0.2, 0.2],
            vec![0.2, 0.3, 0.3, 0.2],
            vec![0.25, 0.25, 0.25, 0.25],
        ],
        vec![0.25, 0.25, 0.25, 0.25],
    );
    let mut obs_batch = Vec::with_capacity(8);
    for _ in 0..8 {
        let (_, obs) = hmm.generate_sequence(20, rng);
        obs_batch.push(obs);
    }
    (hmm, obs_batch)
}

fn validate_larger(h: &mut ValidationHarness, gpu: &Gpu) {
    let dev = Arc::clone(gpu.wgpu_device());
    let op = match HmmBatchForwardF64::new(dev) {
        Ok(o) => o,
        Err(e) => {
            h.check_bool(
                &format!("HMM larger: HmmBatchForwardF64::new failed — {e}"),
                false,
            );
            return;
        }
    };

    let mut rng = Rng::new(777);
    let (hmm, obs_batch) = make_hmm_larger(&mut rng);
    let params = hmm_to_f64_params(&hmm);
    let cpu_mean = cpu_mean_log_lik(&hmm, &obs_batch);

    match gpu_hmm_mean_log_lik(gpu, &op, &params, &obs_batch) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("HMM larger: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_HMM_ALPHA_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("HMM larger: dispatch failed — {e}"), false);
        }
    }
}

fn validate_single_state(h: &mut ValidationHarness, gpu: &Gpu) {
    let dev = Arc::clone(gpu.wgpu_device());
    let op = match HmmBatchForwardF64::new(dev) {
        Ok(o) => o,
        Err(e) => {
            h.check_bool(
                &format!("HMM single state: HmmBatchForwardF64::new failed — {e}"),
                false,
            );
            return;
        }
    };

    let hmm = Hmm::new(vec![vec![1.0]], vec![vec![1.0]], vec![1.0]);
    let obs_batch: Vec<Vec<usize>> = (0..10).map(|_| vec![0; 10]).collect();
    let params = hmm_to_f64_params(&hmm);
    let cpu_mean = cpu_mean_log_lik(&hmm, &obs_batch);

    match gpu_hmm_mean_log_lik(gpu, &op, &params, &obs_batch) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("HMM single state: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_HMM_ALPHA_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("HMM single state: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let dev = Arc::clone(gpu.wgpu_device());
    let op = match HmmBatchForwardF64::new(dev) {
        Ok(o) => o,
        Err(e) => {
            h.check_bool(
                &format!("HMM determinism: HmmBatchForwardF64::new failed — {e}"),
                false,
            );
            return;
        }
    };

    let mut rng = Rng::new(123);
    let (hmm, obs_batch) = make_hmm_small(&mut rng);
    let params = hmm_to_f64_params(&hmm);

    let r1 = gpu_hmm_mean_log_lik(gpu, &op, &params, &obs_batch);
    let r2 = gpu_hmm_mean_log_lik(gpu, &op, &params, &obs_batch);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("HMM determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f64::EPSILON,
            );
        }
        _ => {
            h.check_bool("HMM determinism: dispatch failed", false);
        }
    }
}
