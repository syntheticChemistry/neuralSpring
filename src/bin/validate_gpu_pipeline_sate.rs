// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline validation: `pairwise_hamming` → mean (Paper 017).
//!
//! Uses `BarraCUDA` typed op `PairwiseHammingGpu` (f32) with CPU mean reduction.
//! Replaces raw wgpu chain (`pairwise_hamming` + `mean_reduce`) for validation.
//!
//! ## Pipeline
//!
//! ```text
//! Upload sequences (once)
//!   ↓
//! PairwiseHammingGpu.dispatch() → distances[n_pairs] (f32)
//!   ↓
//! CPU mean(distances)
//! ```
//!
//! ## Provenance
//!
//! GPU op: `barracuda::ops::bio::PairwiseHammingGpu` (f32 pipeline)
//! Validates: `SATé` alignment mean pairwise distance (Liu et al., 2009).

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "validation binary"
)]

use barracuda::ops::bio::PairwiseHammingGpu;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use wgpu::util::DeviceExt;

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

    let mut h = ValidationHarness::new("gpu_pipeline_sate");

    validate_sate_small(&mut h, &gpu);
    validate_sate_larger(&mut h, &gpu);
    validate_sate_identical(&mut h, &gpu);
    validate_sate_all_differ(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_pairwise_hamming(sequences: &[u32], n_seqs: usize, seq_len: usize) -> f32 {
    let n_pairs = n_seqs * (n_seqs - 1) / 2;
    if n_pairs == 0 {
        return 0.0;
    }
    let mut total = 0.0_f32;
    for i in 0..n_seqs {
        for j in (i + 1)..n_seqs {
            let mut diff = 0_u32;
            for s in 0..seq_len {
                if sequences[i * seq_len + s] != sequences[j * seq_len + s] {
                    diff += 1;
                }
            }
            total += diff as f32 / seq_len as f32;
        }
    }
    total / n_pairs as f32
}

// ── GPU via BarraCUDA typed op ──────────────────────────────────────

fn gpu_mean_pairwise_hamming(
    gpu: &Gpu,
    sequences: &[u32],
    n_seqs: u32,
    seq_len: u32,
) -> Result<f32, String> {
    let op = PairwiseHammingGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let n_pairs = (n_seqs * (n_seqs - 1) / 2) as usize;

    let seq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_sate_sequences"),
        contents: bytemuck::cast_slice(sequences),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_sate_distances"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&seq_buf, &dist_buf, n_seqs, seq_len);

    let distances = gpu
        .read_buffer_f32(&dist_buf, n_pairs)
        .map_err(|e| e.to_string())?;
    let mean = distances.iter().sum::<f32>() / distances.len() as f32;
    Ok(mean)
}

// ── Validation functions ───────────────────────────────────────────

fn generate_sequences(n_seqs: usize, seq_len: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let mut flat = Vec::with_capacity(n_seqs * seq_len);
    for _ in 0..(n_seqs * seq_len) {
        flat.push(rng.usize(4) as u32);
    }
    flat
}

fn validate_sate_small(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 8_usize;
    let seq_len = 20_usize;
    let sequences = generate_sequences(n_seqs, seq_len, 42);

    let cpu_mean = cpu_mean_pairwise_hamming(&sequences, n_seqs, seq_len);

    match gpu_mean_pairwise_hamming(gpu, &sequences, n_seqs as u32, seq_len as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("sate small 8×20: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_HAMMING_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("sate small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_sate_larger(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 16_usize;
    let seq_len = 50_usize;
    let sequences = generate_sequences(n_seqs, seq_len, 777);

    let cpu_mean = cpu_mean_pairwise_hamming(&sequences, n_seqs, seq_len);

    match gpu_mean_pairwise_hamming(gpu, &sequences, n_seqs as u32, seq_len as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("sate larger 16×50: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_HAMMING_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("sate larger: dispatch failed — {e}"), false);
        }
    }
}

fn validate_sate_identical(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 4_usize;
    let seq_len = 10_usize;
    let base_seq: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1];
    let mut sequences = vec![0_u32; n_seqs * seq_len];
    for i in 0..n_seqs {
        sequences[i * seq_len..(i + 1) * seq_len].copy_from_slice(&base_seq);
    }

    match gpu_mean_pairwise_hamming(gpu, &sequences, n_seqs as u32, seq_len as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("sate identical: mean distance={gpu_mean:.6} vs 0"),
                f64::from(gpu_mean),
                0.0,
                tolerances::GPU_HAMMING_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("sate identical: dispatch failed — {e}"), false);
        }
    }
}

fn validate_sate_all_differ(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 4_usize;
    let seq_len = 8_usize;
    let mut sequences = vec![0_u32; n_seqs * seq_len];
    for i in 0..n_seqs {
        for s in 0..seq_len {
            sequences[i * seq_len + s] = ((i + s) % 4) as u32;
        }
    }

    let cpu_mean = cpu_mean_pairwise_hamming(&sequences, n_seqs, seq_len);

    match gpu_mean_pairwise_hamming(gpu, &sequences, n_seqs as u32, seq_len as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("sate all differ: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_HAMMING_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("sate all differ: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 6_usize;
    let seq_len = 12_usize;
    let sequences = generate_sequences(n_seqs, seq_len, 99);

    let r1 = gpu_mean_pairwise_hamming(gpu, &sequences, n_seqs as u32, seq_len as u32);
    let r2 = gpu_mean_pairwise_hamming(gpu, &sequences, n_seqs as u32, seq_len as u32);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("sate determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("sate determinism: dispatch failed", false);
        }
    }
}
