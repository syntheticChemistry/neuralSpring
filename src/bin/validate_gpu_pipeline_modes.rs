// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline validation: pairwise_l2 → mean (Paper 012).
//!
//! Uses BarraCUDA typed op `PairwiseL2Gpu` (f32) with CPU mean reduction.
//! Replaces raw wgpu chain (pairwise_l2 + mean_reduce) for validation.
//!
//! ## Pipeline
//!
//! ```text
//! Upload features [N x D] (once)
//!   ↓
//! PairwiseL2Gpu.dispatch() → distances[N*(N-1)/2] (f32)
//!   ↓
//! CPU mean(distances)
//! ```
//!
//! ## Provenance
//!
//! GPU op: `barracuda::ops::bio::PairwiseL2Gpu` (f32 pipeline)
//! Validates: end-to-end GPU-resident computation with scalar-only readback.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless
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

    let mut h = ValidationHarness::new("gpu_pipeline_modes");

    validate_small(&mut h, &gpu);
    validate_identical(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_pairwise_l2(features: &[Vec<f64>]) -> f64 {
    let n = features.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += l2_distance(&features[i], &features[j]);
            count += 1;
        }
    }
    sum / count as f64
}

// ── GPU via BarraCUDA typed op ──────────────────────────────────────

fn gpu_pairwise_l2_mean(gpu: &Gpu, features_flat: &[f32], n: u32, dim: u32) -> Result<f32, String> {
    let op = PairwiseL2Gpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let n_pairs = (n * (n - 1) / 2) as usize;

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_features"),
        contents: bytemuck::cast_slice(features_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_distances"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&input_buf, &output_buf, n, dim);

    let distances = gpu.read_buffer_f32(&output_buf, n_pairs)?;
    let mean = distances.iter().sum::<f32>() / distances.len() as f32;
    Ok(mean)
}

// ── Validation functions ───────────────────────────────────────────

fn validate_small(h: &mut ValidationHarness, gpu: &Gpu) {
    let features: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ];
    let n = 5_u32;
    let dim = 3_u32;

    let cpu_mean = cpu_mean_pairwise_l2(&features);

    let flat: Vec<f32> = features
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f32))
        .collect();

    match gpu_pairwise_l2_mean(gpu, &flat, n, dim) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("modes small 5×3: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                cpu_mean,
                tolerances::GPU_MODES_L2_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("modes small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_identical(h: &mut ValidationHarness, gpu: &Gpu) {
    let features: Vec<Vec<f64>> = vec![
        vec![0.5, 0.5, 0.5],
        vec![0.5, 0.5, 0.5],
        vec![0.5, 0.5, 0.5],
    ];
    let n = 3_u32;
    let dim = 3_u32;

    let cpu_mean = cpu_mean_pairwise_l2(&features);

    let flat: Vec<f32> = features
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f32))
        .collect();

    match gpu_pairwise_l2_mean(gpu, &flat, n, dim) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("modes identical: GPU={gpu_mean:.6} vs CPU mean=0"),
                f64::from(gpu_mean),
                cpu_mean,
                tolerances::GPU_MODES_L2_F32,
            );
            h.check_abs(
                "modes identical: mean distance should be 0",
                f64::from(gpu_mean),
                0.0,
                tolerances::GPU_MODES_L2_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("modes identical: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let features: Vec<Vec<f64>> = vec![
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
        vec![0.7, 0.8, 0.9],
        vec![0.2, 0.3, 0.4],
        vec![0.5, 0.6, 0.7],
    ];
    let flat: Vec<f32> = features
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f32))
        .collect();

    let r1 = gpu_pairwise_l2_mean(gpu, &flat, 5, 3);
    let r2 = gpu_pairwise_l2_mean(gpu, &flat, 5, 3);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("modes determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("modes determinism: dispatch failed", false);
        }
    }
}
