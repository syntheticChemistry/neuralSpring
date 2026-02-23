// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU pipeline validation: `BatchIprGpu` (`BarraCUDA`) + CPU mean (Papers 022-023).
//!
//! Replaces raw wgpu pipeline with typed `BarraCUDA` op: `barracuda::spectral::BatchIprGpu`.
//! Stage 1: BatchIprGpu.dispatch → `ipr_out[n_vectors]` (f32).
//! Stage 2: CPU mean over `ipr_out`.
//!
//! ## Pipeline
//!
//! ```text
//! eigenvectors (upload once)
//!   ↓
//! BatchIprGpu.dispatch → ipr_out[n_vectors]
//!   ↓
//! CPU mean(ipr_out) → scalar
//! ```
//!
//! ## Provenance
//!
//! Typed op: `barracuda::spectral::BatchIprGpu` (f32).
//! Validates: `BarraCUDA` spectral API with scalar-only readback.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use barracuda::spectral::BatchIprGpu;
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

    let mut h = ValidationHarness::new("gpu_pipeline_spectral");

    validate_uniform_vectors(&mut h, &gpu);
    validate_localized_vectors(&mut h, &gpu);
    validate_random_vectors(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_ipr(eigenvectors: &[f32], dim: usize, n_vectors: usize) -> f32 {
    let mut total = 0.0f32;
    for v in 0..n_vectors {
        let mut sum_p4 = 0.0f32;
        for i in 0..dim {
            let val = eigenvectors[v * dim + i];
            let p2 = val * val;
            sum_p4 += p2 * p2;
        }
        total += sum_p4;
    }
    total / n_vectors as f32
}

// ── BarraCUDA typed op + CPU mean ────────────────────────────────────

fn gpu_chained_mean_ipr(
    gpu: &Gpu,
    eigenvectors: &[f32],
    dim: u32,
    n_vectors: u32,
) -> Result<f32, String> {
    let device = gpu.device();
    let op = BatchIprGpu::new(Arc::clone(gpu.wgpu_device()));

    let eigenvectors_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_eigenvectors"),
        contents: bytemuck::cast_slice(eigenvectors),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let ipr_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_ipr_out"),
        size: u64::from(n_vectors) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&eigenvectors_buf, &ipr_buf, dim, n_vectors);

    let ipr_out = gpu.read_buffer_f32(&ipr_buf, n_vectors as usize)?;
    let mean = ipr_out.iter().sum::<f32>() / ipr_out.len() as f32;
    Ok(mean)
}

// ── Validation functions ───────────────────────────────────────────

fn validate_uniform_vectors(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 8_usize;
    let n_vectors = 4_usize;
    let val = 1.0f32 / (dim as f32).sqrt();
    let eigenvectors: Vec<f32> = vec![val; dim * n_vectors];
    let expected = 1.0f32 / dim as f32;

    match gpu_chained_mean_ipr(gpu, &eigenvectors, dim as u32, n_vectors as u32) {
        Ok(gpu_mean) => {
            h.check_bool(
                &format!("uniform: GPU mean finite ({gpu_mean:.6})"),
                gpu_mean.is_finite(),
            );
            h.check_abs(
                &format!("uniform: GPU={gpu_mean:.6} vs expected={expected:.6} (1/dim)"),
                f64::from(gpu_mean),
                f64::from(expected),
                tolerances::GPU_BATCH_IPR_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("uniform: dispatch failed — {e}"), false);
        }
    }
}

fn validate_localized_vectors(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 8_usize;
    let n_vectors = 4_usize;
    let mut eigenvectors = vec![0.0f32; dim * n_vectors];
    for v in 0..n_vectors {
        eigenvectors[v * dim + v % dim] = 1.0f32;
    }

    match gpu_chained_mean_ipr(gpu, &eigenvectors, dim as u32, n_vectors as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("localized: GPU={gpu_mean:.6} vs expected=1.0"),
                f64::from(gpu_mean),
                1.0,
                tolerances::GPU_BATCH_IPR_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("localized: dispatch failed — {e}"), false);
        }
    }
}

fn validate_random_vectors(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 16_usize;
    let n_vectors = 8_usize;
    let mut rng = Rng::new(42);
    let eigenvectors: Vec<f32> = (0..dim * n_vectors).map(|_| rng.uniform() as f32).collect();

    let cpu_mean = cpu_mean_ipr(&eigenvectors, dim, n_vectors);

    match gpu_chained_mean_ipr(gpu, &eigenvectors, dim as u32, n_vectors as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("random: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_BATCH_IPR_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("random: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 16_usize;
    let n_vectors = 8_usize;
    let mut rng = Rng::new(42);
    let eigenvectors: Vec<f32> = (0..dim * n_vectors).map(|_| rng.uniform() as f32).collect();

    let r1 = gpu_chained_mean_ipr(gpu, &eigenvectors, dim as u32, n_vectors as u32);
    let r2 = gpu_chained_mean_ipr(gpu, &eigenvectors, dim as u32, n_vectors as u32);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}
