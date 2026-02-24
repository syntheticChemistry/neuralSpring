// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: HMM forward pass via upstream `HmmBatchForwardF64`.
//!
//! Validates the upstream `BarraCUDA` HMM forward pass (f64 batch GPU shader)
//! against the CPU reference in `src/hmm.rs`. Replaces the local f32 evolved
//! dispatch with the upstream f64 API — strictly better precision.
//!
//! Evolution path:
//! ```text
//! Python (hmmlearn) → Rust CPU (hmm.rs) → local f32 GPU (retired)
//!   → upstream HmmBatchForwardF64 (wetSpring f64, absorbed by ToadStool)
//! ```
//!
//! ## Papers validated
//!
//! - Paper 016: HMM Forward/Backward/Viterbi (Liu et al., 2014)
//! - Paper 017: `SATé` Alignment (Liu et al., 2009)
//! - Paper 018: Introgression Detection (Liu et al., 2015)
//!
//! ## Provenance
//!
//! CPU reference: `hmm::Hmm::forward` (seed=42, 2-state 20-obs).
//! GPU shader: `barracuda::ops::bio::hmm::HmmBatchForwardF64` (wetSpring origin)

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_lines
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

    let mut h = ValidationHarness::new("gpu_hmm_forward");

    validate_2state_weather(&mut h, &gpu);
    validate_3state_genomic(&mut h, &gpu);
    validate_log_likelihood_sign(&mut h, &gpu);
    validate_alpha_sum_property(&mut h, &gpu);
    validate_longer_sequence(&mut h, &gpu);

    h.finish();
}

fn weather_hmm() -> Hmm {
    Hmm::new(
        vec![vec![0.7, 0.3], vec![0.4, 0.6]],
        vec![vec![0.1, 0.4, 0.5], vec![0.6, 0.3, 0.1]],
        vec![0.6, 0.4],
    )
}

fn genomic_hmm() -> Hmm {
    Hmm::new(
        vec![
            vec![0.8, 0.1, 0.1],
            vec![0.05, 0.9, 0.05],
            vec![0.1, 0.1, 0.8],
        ],
        vec![
            vec![0.4, 0.3, 0.2, 0.1],
            vec![0.1, 0.4, 0.3, 0.2],
            vec![0.2, 0.1, 0.4, 0.3],
        ],
        vec![0.33, 0.34, 0.33],
    )
}

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

fn dispatch_hmm_f64(
    gpu: &Gpu,
    op: &HmmBatchForwardF64,
    params: &HmmF64Params,
    obs_batch: &[Vec<usize>],
) -> Result<(Vec<f64>, Vec<f64>), String> {
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
        label: Some("log_trans"),
        contents: bytemuck::cast_slice(&params.log_trans),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let log_emit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("log_emit"),
        contents: bytemuck::cast_slice(&params.log_emit),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let log_pi_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("log_pi"),
        contents: bytemuck::cast_slice(&params.log_pi),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let obs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("observations"),
        contents: bytemuck::cast_slice(&obs_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let alpha_size = u64::from(n_seqs) * u64::from(n_steps) * u64::from(params.n_states) * 8;
    let log_alpha_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("log_alpha"),
        size: alpha_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let log_lik_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("log_lik"),
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

    let alpha_count = (n_seqs * n_steps * params.n_states) as usize;
    let alpha = gpu.read_buffer_f64(&log_alpha_buf, alpha_count)?;
    let log_lik = gpu.read_buffer_f64(&log_lik_buf, n_seqs as usize)?;
    Ok((alpha, log_lik))
}

fn validate_2state_weather(h: &mut ValidationHarness, gpu: &Gpu) {
    let dev = Arc::clone(gpu.wgpu_device());
    let op = match HmmBatchForwardF64::new(dev) {
        Ok(op) => op,
        Err(e) => {
            h.check_bool(&format!("HmmBatchForwardF64::new failed — {e}"), false);
            return;
        }
    };

    let hmm = weather_hmm();
    let params = hmm_to_f64_params(&hmm);
    let mut rng = Rng::new(42);
    let (_, obs) = hmm.generate_sequence(20, &mut rng);
    let (cpu_alpha, cpu_ll) = hmm.forward(&obs);

    match dispatch_hmm_f64(gpu, &op, &params, &[obs]) {
        Ok((alpha, log_lik)) => {
            let gpu_ll = log_lik[0];
            let all_finite = alpha.iter().all(|v| v.is_finite());
            h.check_bool("2-state weather: GPU alpha all finite", all_finite);

            h.check_bool(
                &format!("2-state weather: GPU LL finite ({gpu_ll:.6})"),
                gpu_ll.is_finite(),
            );
            h.check_bool(
                "2-state weather: GPU LL negative (probability < 1)",
                gpu_ll < 0.0,
            );
            h.check_abs(
                &format!("2-state weather: GPU LL ≈ CPU LL ({gpu_ll:.6} vs {cpu_ll:.6})"),
                gpu_ll,
                cpu_ll,
                tolerances::GPU_HMM_LOG_LIKELIHOOD_F32 * 0.1,
            );
            h.check_bool(
                &format!("2-state weather: CPU LL finite ({cpu_ll:.6})"),
                cpu_ll.is_finite(),
            );
            let _ = cpu_alpha;
        }
        Err(e) => {
            h.check_bool(
                &format!("2-state weather: GPU dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_3state_genomic(h: &mut ValidationHarness, gpu: &Gpu) {
    let dev = Arc::clone(gpu.wgpu_device());
    let op = match HmmBatchForwardF64::new(dev) {
        Ok(op) => op,
        Err(e) => {
            h.check_bool(&format!("HmmBatchForwardF64::new failed — {e}"), false);
            return;
        }
    };

    let hmm = genomic_hmm();
    let params = hmm_to_f64_params(&hmm);
    let mut rng = Rng::new(123);
    let (_, obs) = hmm.generate_sequence(30, &mut rng);
    let (_, cpu_ll) = hmm.forward(&obs);

    match dispatch_hmm_f64(gpu, &op, &params, &[obs]) {
        Ok((_, log_lik)) => {
            let gpu_ll = log_lik[0];
            h.check_bool(
                &format!("3-state genomic: GPU LL finite ({gpu_ll:.6})"),
                gpu_ll.is_finite(),
            );
            h.check_abs(
                &format!("3-state genomic: GPU LL ≈ CPU LL ({gpu_ll:.6} vs {cpu_ll:.6})"),
                gpu_ll,
                cpu_ll,
                tolerances::GPU_HMM_LOG_LIKELIHOOD_F32 * 0.1,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("3-state genomic: GPU dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_log_likelihood_sign(h: &mut ValidationHarness, gpu: &Gpu) {
    let dev = Arc::clone(gpu.wgpu_device());
    let op = match HmmBatchForwardF64::new(dev) {
        Ok(op) => op,
        Err(e) => {
            h.check_bool(&format!("HmmBatchForwardF64::new failed — {e}"), false);
            return;
        }
    };

    let hmm = weather_hmm();
    let params = hmm_to_f64_params(&hmm);
    let obs_short: Vec<usize> = vec![0, 1, 2, 0, 1];
    let obs_long: Vec<usize> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];

    let max_len = obs_long.len();
    let mut short_padded = obs_short;
    short_padded.resize(max_len, 0);

    match dispatch_hmm_f64(gpu, &op, &params, &[short_padded, obs_long]) {
        Ok((_, log_lik)) => {
            h.check_bool(
                &format!(
                    "LL sign: short > long ({:.4} > {:.4})",
                    log_lik[0], log_lik[1]
                ),
                log_lik[0] > log_lik[1],
            );
        }
        Err(e) => h.check_bool(&format!("LL sign: dispatch failed — {e}"), false),
    }
}

fn validate_alpha_sum_property(h: &mut ValidationHarness, gpu: &Gpu) {
    let dev = Arc::clone(gpu.wgpu_device());
    let op = match HmmBatchForwardF64::new(dev) {
        Ok(op) => op,
        Err(e) => {
            h.check_bool(&format!("HmmBatchForwardF64::new failed — {e}"), false);
            return;
        }
    };

    let hmm = weather_hmm();
    let params = hmm_to_f64_params(&hmm);
    let obs: Vec<usize> = vec![0, 1, 0, 2, 1];

    match dispatch_hmm_f64(gpu, &op, &params, &[obs]) {
        Ok((alpha, _)) => {
            let n = params.n_states as usize;
            let n_steps = 5;
            let final_alpha: Vec<f64> = alpha[(n_steps - 1) * n..n_steps * n].to_vec();
            let max_a = final_alpha
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let lse = max_a
                + final_alpha
                    .iter()
                    .map(|&a| (a - max_a).exp())
                    .sum::<f64>()
                    .ln();
            h.check_bool(
                &format!("alpha sum property: logsumexp finite ({lse:.6})"),
                lse.is_finite(),
            );
            h.check_bool(
                "alpha sum property: logsumexp negative (prob < 1)",
                lse < 0.0,
            );
        }
        Err(e) => {
            h.check_bool(&format!("alpha sum: dispatch failed — {e}"), false);
        }
    }
}

fn validate_longer_sequence(h: &mut ValidationHarness, gpu: &Gpu) {
    let dev = Arc::clone(gpu.wgpu_device());
    let op = match HmmBatchForwardF64::new(dev) {
        Ok(op) => op,
        Err(e) => {
            h.check_bool(&format!("HmmBatchForwardF64::new failed — {e}"), false);
            return;
        }
    };

    let hmm = genomic_hmm();
    let params = hmm_to_f64_params(&hmm);
    let mut rng = Rng::new(999);
    let (_, obs) = hmm.generate_sequence(100, &mut rng);
    let (_, cpu_ll) = hmm.forward(&obs);

    match dispatch_hmm_f64(gpu, &op, &params, &[obs]) {
        Ok((_, log_lik)) => {
            let gpu_ll = log_lik[0];
            h.check_abs(
                &format!("100-obs genomic: GPU LL ≈ CPU LL ({gpu_ll:.6} vs {cpu_ll:.6})"),
                gpu_ll,
                cpu_ll,
                tolerances::GPU_HMM_LOG_LIKELIHOOD_F32 * 0.1,
            );
            h.check_bool(
                &format!("100-obs genomic: GPU LL finite ({gpu_ll:.8})"),
                gpu_ll.is_finite(),
            );
        }
        Err(e) => {
            h.check_bool(&format!("100-obs: dispatch failed — {e}"), false);
        }
    }
}
