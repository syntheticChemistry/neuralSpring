// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: pairwise L2 distance via `BarraCUDA` `PairwiseL2Gpu`.
//!
//! Validates `barracuda::ops::bio::PairwiseL2Gpu` against CPU
//! L2 distance computation from `modes.rs`.  The GPU op computes
//! all pairwise L2 distances in a single dispatch.
//!
//! ## Papers validated
//!
//! - Paper 012: MODES (novelty metric via pairwise L2 distance)
//!
//! ## Provenance
//!
//! CPU reference: `modes::l2_distance` (seed=0, 5×3 pairwise features).
//! GPU op: `barracuda::ops::bio::PairwiseL2Gpu`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]

use barracuda::ops::bio::PairwiseL2Gpu;
use neural_spring::gpu::Gpu;
use neural_spring::modes::l2_distance;
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
        Err(e) => {
            eprintln!("  SKIP: {e} — no GPU/CPU adapter available");
            eprintln!("  0/0 checks — skipping gracefully");
            std::process::exit(0);
        }
    };

    let op = PairwiseL2Gpu::new(Arc::clone(gpu.wgpu_device()));
    let mut h = ValidationHarness::new("gpu_modes");

    validate_small_features(&mut h, &gpu, &op);
    validate_known_distances(&mut h, &gpu, &op);
    validate_determinism(&mut h, &gpu, &op);

    h.finish();
}

/// CPU reference: all pairwise L2 distances (upper triangle, row-major).
fn cpu_pairwise_l2(features: &[Vec<f64>]) -> Vec<f64> {
    let n = features.len();
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            out.push(l2_distance(&features[i], &features[j]));
        }
    }
    out
}

fn gpu_pairwise_l2(
    gpu: &Gpu,
    op: &PairwiseL2Gpu,
    features_flat: &[f32],
    n: u32,
    dim: u32,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let n_pairs = (n * (n - 1) / 2) as usize;

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("features"),
        contents: bytemuck::cast_slice(features_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("distances"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&input_buf, &output_buf, n, dim);

    gpu.read_buffer_f32(&output_buf, n_pairs)
}

fn validate_small_features(h: &mut ValidationHarness, gpu: &Gpu, op: &PairwiseL2Gpu) {
    let features: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ];
    let n = 5_usize;
    let dim = 3_usize;

    let cpu = cpu_pairwise_l2(&features);
    let flat: Vec<f32> = features
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f32))
        .collect();

    match gpu_pairwise_l2(gpu, op, &flat, n as u32, dim as u32) {
        Ok(gpu_dist) => {
            h.check_bool(
                &format!("small features: correct pair count ({})", gpu_dist.len()),
                gpu_dist.len() == cpu.len(),
            );

            for (idx, (&g, &c)) in gpu_dist.iter().zip(cpu.iter()).enumerate() {
                h.check_abs(
                    &format!("small features[{idx}]: GPU ≈ CPU ({g:.6} vs {c:.6})"),
                    f64::from(g),
                    c,
                    tolerances::GPU_MODES_L2_F32,
                );
            }
        }
        Err(e) => {
            h.check_bool(&format!("small features: dispatch failed — {e}"), false);
        }
    }
}

fn validate_known_distances(h: &mut ValidationHarness, gpu: &Gpu, op: &PairwiseL2Gpu) {
    // (0,0,0) vs (1,0,0) = 1.0
    // (0,0,0) vs (1,1,1) = sqrt(3) ≈ 1.732
    let features: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0],
    ];
    let flat: Vec<f32> = features
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f32))
        .collect();

    match gpu_pairwise_l2(gpu, op, &flat, 3, 3) {
        Ok(gpu_dist) => {
            let d_01 = gpu_dist[0]; // (0,0,0) vs (1,0,0)
            let d_02 = gpu_dist[1]; // (0,0,0) vs (1,1,1)
            let d_12 = gpu_dist[2]; // (1,0,0) vs (1,1,1) = sqrt(2) ≈ 1.414

            h.check_abs(
                "known: (0,0,0) vs (1,0,0) = 1.0",
                f64::from(d_01),
                1.0,
                tolerances::GPU_MODES_L2_F32,
            );
            h.check_abs(
                "known: (0,0,0) vs (1,1,1) = √3",
                f64::from(d_02),
                3_f64.sqrt(),
                tolerances::GPU_MODES_L2_F32,
            );
            h.check_abs(
                "known: (1,0,0) vs (1,1,1) = √2",
                f64::from(d_12),
                2_f64.sqrt(),
                tolerances::GPU_MODES_L2_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("known distances: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu, op: &PairwiseL2Gpu) {
    let features: Vec<Vec<f64>> = vec![
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
        vec![0.7, 0.8, 0.9],
    ];
    let flat: Vec<f32> = features
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f32))
        .collect();

    let run1 = gpu_pairwise_l2(gpu, op, &flat, 3, 3);
    let run2 = gpu_pairwise_l2(gpu, op, &flat, 3, 3);

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(&a, &b)| (a - b).abs() < f32::EPSILON);
            h.check_bool("determinism: two runs identical", identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}
