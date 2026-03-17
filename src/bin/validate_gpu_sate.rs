// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: pairwise Hamming distance via `BarraCUDA` upstream API.
//!
//! Validates `barracuda::ops::bio::PairwiseHammingGpu` against CPU
//! Hamming distance computation from `sate_alignment.rs`.  The typed op
//! evaluates all n*(n-1)/2 pairwise distances in a single dispatch.
//!
//! ## Papers validated
//!
//! - Paper 017: `SATé` Alignment (Liu et al., 2009)
//!
//! ## Provenance
//!
//! CPU reference: `sate_alignment::pairwise_distance_matrix` (seed=42, `n_seqs=8` `seq_len=50`).
//! Upstream API: `barracuda::ops::bio::PairwiseHammingGpu`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![expect(clippy::cast_possible_truncation, reason = "validation binary")]

use barracuda::ops::bio::PairwiseHammingGpu;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::sate_alignment::pairwise_distance_matrix;
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

    let op = PairwiseHammingGpu::new(Arc::clone(gpu.wgpu_device()));
    let mut h = ValidationHarness::new("gpu_sate");

    validate_small(&mut h, &gpu, &op);
    validate_larger(&mut h, &gpu, &op);
    validate_identical_sequences(&mut h, &gpu, &op);
    validate_determinism(&mut h, &gpu, &op);

    h.finish();
}

fn generate_test_sequences(n_seqs: usize, seq_len: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let mut flat = Vec::with_capacity(n_seqs * seq_len);
    for _ in 0..(n_seqs * seq_len) {
        flat.push(rng.usize(4) as u32);
    }
    flat
}

fn gpu_pairwise_hamming(
    op: &PairwiseHammingGpu,
    gpu: &Gpu,
    sequences: &[u32],
    n_seqs: u32,
    seq_len: u32,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let n_pairs = (n_seqs * (n_seqs - 1) / 2) as usize;

    let sequences_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sequences"),
        contents: bytemuck::cast_slice(sequences),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let distances_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("distances"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&sequences_buf, &distances_buf, n_seqs, seq_len);

    gpu.read_buffer_f32(&distances_buf, n_pairs)
}

fn validate_small(h: &mut ValidationHarness, gpu: &Gpu, op: &PairwiseHammingGpu) {
    let n_seqs = 8_usize;
    let seq_len = 50_usize;
    let flat = generate_test_sequences(n_seqs, seq_len, 42);
    let seqs_u8: Vec<u8> = flat.iter().map(|&v| v as u8).collect();

    let cpu_matrix = pairwise_distance_matrix(&seqs_u8, n_seqs, seq_len, false);
    let mut cpu_upper = Vec::new();
    for i in 0..n_seqs {
        for j in (i + 1)..n_seqs {
            cpu_upper.push(cpu_matrix[i * n_seqs + j]);
        }
    }

    match gpu_pairwise_hamming(op, gpu, &flat, n_seqs as u32, seq_len as u32) {
        Ok(gpu_dist) => {
            h.check_bool(
                &format!("small: correct pair count ({})", gpu_dist.len()),
                gpu_dist.len() == cpu_upper.len(),
            );

            let max_diff: f64 = gpu_dist
                .iter()
                .zip(cpu_upper.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("small: max GPU-CPU diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_HAMMING_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_larger(h: &mut ValidationHarness, gpu: &Gpu, op: &PairwiseHammingGpu) {
    let n_seqs = 20_usize;
    let seq_len = 200_usize;
    let flat = generate_test_sequences(n_seqs, seq_len, 77);
    let seqs_u8: Vec<u8> = flat.iter().map(|&v| v as u8).collect();

    let cpu_matrix = pairwise_distance_matrix(&seqs_u8, n_seqs, seq_len, false);
    let mut cpu_upper = Vec::new();
    for i in 0..n_seqs {
        for j in (i + 1)..n_seqs {
            cpu_upper.push(cpu_matrix[i * n_seqs + j]);
        }
    }

    match gpu_pairwise_hamming(op, gpu, &flat, n_seqs as u32, seq_len as u32) {
        Ok(gpu_dist) => {
            let max_diff: f64 = gpu_dist
                .iter()
                .zip(cpu_upper.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "20×200: max GPU-CPU diff ({max_diff:.2e}), {} pairs",
                    gpu_dist.len()
                ),
                max_diff,
                tolerances::GPU_HAMMING_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("20×200: dispatch failed — {e}"), false);
        }
    }
}

fn validate_identical_sequences(h: &mut ValidationHarness, gpu: &Gpu, op: &PairwiseHammingGpu) {
    let n_seqs = 10_u32;
    let seq_len = 64_u32;
    let template: Vec<u32> = vec![2; seq_len as usize];
    let flat: Vec<u32> = template
        .iter()
        .cycle()
        .take((n_seqs * seq_len) as usize)
        .copied()
        .collect();

    match gpu_pairwise_hamming(op, gpu, &flat, n_seqs, seq_len) {
        Ok(gpu_dist) => {
            let max_dist = gpu_dist.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let all_zero = gpu_dist
                .iter()
                .all(|&d| d.abs() < tolerances::GPU_HAMMING_F32 as f32);
            h.check_bool(
                &format!("identical sequences: all Hamming=0 (max={max_dist:.2e})"),
                all_zero,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("identical sequences: dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu, op: &PairwiseHammingGpu) {
    let n_seqs = 8_u32;
    let seq_len = 50_u32;
    let flat = generate_test_sequences(n_seqs as usize, seq_len as usize, 123);

    let run1 = gpu_pairwise_hamming(op, gpu, &flat, n_seqs, seq_len);
    let run2 = gpu_pairwise_hamming(op, gpu, &flat, n_seqs, seq_len);

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let bit_identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
            h.check_bool("determinism: two Hamming runs bit-identical", bit_identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}
