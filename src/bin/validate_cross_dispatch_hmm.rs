// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-dispatch validation: HMM forward log-likelihood (Papers 016, 018).
//!
//! Uses `BarraCUDA` typed op `HmmBatchForwardF64` to validate GPU ↔ CPU parity
//! for the HMM forward algorithm vs the `neural_spring::hmm::Hmm` CPU reference.
//!
//! ## Buffer layout (`HmmBatchForwardF64`)
//!
//! - `log_trans`: \[`n_states` × `n_states`\] f64 row-major
//! - `log_emit`: \[`n_states` × `n_symbols`\] f64 row-major
//! - `log_pi`: \[`n_states`\] f64
//! - `observations`: \[`n_seqs` × `n_steps`\] u32
//! - `log_alpha_out`: \[`n_seqs` × `n_steps` × `n_states`\] f64
//! - `log_lik_out`: \[`n_seqs`\] f64
//!
//! ## Provenance
//!
//! | Baseline | Source |
//! |----------|--------|
//! | CPU reference | `neural_spring::hmm::Hmm` (Papers 016, 018) |
//! | GPU kernel | `barracuda::ops::bio::HmmBatchForwardF64` |

#![expect(clippy::cast_possible_truncation, reason = "validation binary")]

use std::sync::Arc;

use barracuda::dispatch::{dispatch_for, DispatchTarget};
use barracuda::ops::bio::HmmBatchForwardF64;
use neural_spring::gpu::Gpu;
use neural_spring::hmm::Hmm;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
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

    let mut h = ValidationHarness::new("cross_dispatch_hmm");
    validate_dispatch_routing(&mut h);
    validate_hmm_parity(&mut h, &gpu);
    h.finish();
}

// ── Dispatch routing ─────────────────────────────────────────────

fn validate_dispatch_routing(h: &mut ValidationHarness) {
    let small = dispatch_for("hmm_forward", 100);
    let large = dispatch_for("hmm_forward", 10_000);

    h.check_bool(
        &format!("dispatch: hmm_forward(100) → {small:?}"),
        matches!(small, DispatchTarget::Cpu),
    );
    h.check_bool(
        &format!("dispatch: hmm_forward(10k) → {large:?}"),
        matches!(large, DispatchTarget::Gpu),
    );
}

// ── HMM parity: GPU vs CPU ───────────────────────────────────────

struct HmmForwardParams<'a> {
    log_trans: &'a [f64],
    log_emit: &'a [f64],
    log_pi: &'a [f64],
    observations: &'a [u32],
    n_states: u32,
    n_symbols: u32,
    n_steps: u32,
    n_seqs: u32,
}

fn gpu_hmm_forward(gpu: &Gpu, params: &HmmForwardParams<'_>) -> Result<f64, String> {
    let device = gpu.device();
    let op = HmmBatchForwardF64::new(Arc::clone(gpu.wgpu_device())).map_err(|e| e.to_string())?;

    let log_trans_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_log_trans"),
        contents: bytemuck::cast_slice(params.log_trans),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let log_emit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_log_emit"),
        contents: bytemuck::cast_slice(params.log_emit),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let log_pi_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_log_pi"),
        contents: bytemuck::cast_slice(params.log_pi),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let obs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_observations"),
        contents: bytemuck::cast_slice(params.observations),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let log_alpha_size = (params.n_seqs * params.n_steps * params.n_states) as usize;
    let log_alpha_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xd_log_alpha"),
        size: (log_alpha_size * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let log_lik_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xd_log_lik"),
        size: (params.n_seqs as usize * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    op.dispatch(&barracuda::ops::bio::hmm::HmmForwardArgs {
        n_states: params.n_states,
        n_symbols: params.n_symbols,
        n_steps: params.n_steps,
        n_seqs: params.n_seqs,
        log_trans: &log_trans_buf,
        log_emit: &log_emit_buf,
        log_pi: &log_pi_buf,
        observations: &obs_buf,
        log_alpha_out: &log_alpha_buf,
        log_lik_out: &log_lik_buf,
    })
    .map_err(|e| e.to_string())?;

    let log_lik = gpu.read_buffer_f64(&log_lik_buf, params.n_seqs as usize)?;
    Ok(log_lik[0])
}

fn hmm_to_log_f64(hmm: &Hmm) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let log_initial: Vec<f64> = hmm.initial.iter().map(|&p| p.ln()).collect();
    let log_trans: Vec<f64> = hmm.transition.iter().map(|&p| p.ln()).collect();
    let log_emit: Vec<f64> = hmm.emission.iter().map(|&p| p.ln()).collect();
    (log_initial, log_trans, log_emit)
}

fn validate_hmm_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let mut rng = Rng::new(42);
    let n_states = 3_usize;
    let n_symbols = 4_usize;
    let seq_len = 20_usize;

    let transition: Vec<Vec<f64>> = (0..n_states)
        .map(|_| {
            let row: Vec<f64> = (0..n_states).map(|_| rng.uniform()).collect();
            let sum: f64 = row.iter().sum();
            row.into_iter().map(|x| x / sum).collect()
        })
        .collect();

    let emission: Vec<Vec<f64>> = (0..n_states)
        .map(|_| {
            let row: Vec<f64> = (0..n_symbols).map(|_| rng.uniform()).collect();
            let sum: f64 = row.iter().sum();
            row.into_iter().map(|x| x / sum).collect()
        })
        .collect();

    let mut initial: Vec<f64> = (0..n_states).map(|_| rng.uniform()).collect();
    let sum_init: f64 = initial.iter().sum();
    for x in &mut initial {
        *x /= sum_init;
    }

    let hmm = Hmm::new(transition, emission, initial);
    let obs: Vec<usize> = (0..seq_len).map(|_| rng.usize(n_symbols)).collect();

    let (_, cpu_ll) = hmm.forward(&obs);
    let (log_pi, log_trans, log_emit) = hmm_to_log_f64(&hmm);
    let observations: Vec<u32> = obs.iter().map(|&o| (o.min(n_symbols - 1)) as u32).collect();

    match gpu_hmm_forward(
        gpu,
        &HmmForwardParams {
            log_trans: &log_trans,
            log_emit: &log_emit,
            log_pi: &log_pi,
            observations: &observations,
            n_states: n_states as u32,
            n_symbols: n_symbols as u32,
            n_steps: seq_len as u32,
            n_seqs: 1,
        },
    ) {
        Ok(gpu_ll) => {
            let diff = (gpu_ll - cpu_ll).abs();
            h.check_upper(
                &format!("HMM log-lik parity: GPU={gpu_ll:.4} vs CPU={cpu_ll:.4}, diff={diff:.2e}"),
                diff,
                tolerances::GPU_HMM_LOG_LIKELIHOOD_F32,
            );
            h.check_bool("HMM log-lik negative", cpu_ll < 0.0);
        }
        Err(e) => {
            h.check_bool(&format!("HMM parity: GPU failed — {e}"), false);
        }
    }
}
