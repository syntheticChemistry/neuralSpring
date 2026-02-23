// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline validation: spatial payoff → mean (Paper 019).
//!
//! Uses `BarraCUDA` typed op `SpatialPayoffGpu` (f32) with CPU mean reduction.
//! Replaces raw wgpu chain (`spatial_payoff` + `mean_reduce`) for validation.
//!
//! ## Pipeline
//!
//! ```text
//! Upload grid (once)
//!   ↓
//! SpatialPayoffGpu.dispatch() → fitness[n²] (f32)
//!   ↓
//! CPU mean(fitness)
//! ```
//!
//! ## Provenance
//!
//! GPU op: `barracuda::ops::bio::SpatialPayoffGpu` (f32 pipeline)
//! Validates: end-to-end GPU-resident computation with scalar-only readback.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

use barracuda::ops::bio::SpatialPayoffGpu;
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

    let mut h = ValidationHarness::new("gpu_pipeline_ecology");

    validate_small_grid(&mut h, &gpu);
    validate_larger_grid(&mut h, &gpu);
    validate_all_cooperators(&mut h, &gpu);
    validate_all_defectors(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_spatial_mean_fitness(grid: &[u32], n: usize, b: f32, c: f32) -> f32 {
    let n_i = n as i32;
    let mut total = 0.0_f32;
    for idx in 0..(n * n) {
        let i = idx / n;
        let j = idx % n;
        let me = grid[idx];
        let mut fit = 0.0_f32;
        for di in -1..=1 {
            for dj in -1..=1 {
                if di == 0 && dj == 0 {
                    continue;
                }
                let ni = (i as i32 + di + n_i).rem_euclid(n_i) as usize;
                let nj = (j as i32 + dj + n_i).rem_euclid(n_i) as usize;
                let other = grid[ni * n + nj];
                if me == 1 && other == 1 {
                    fit += b - c;
                } else if me == 1 && other == 0 {
                    fit -= c;
                } else if me == 0 && other == 1 {
                    fit += b;
                }
            }
        }
        total += fit;
    }
    total / (n * n) as f32
}

// ── GPU via BarraCUDA typed op ──────────────────────────────────────

fn gpu_spatial_mean_fitness(
    gpu: &Gpu,
    grid: &[u32],
    n: u32,
    b: f32,
    c: f32,
) -> Result<f32, String> {
    let op = SpatialPayoffGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let nn = (n * n) as usize;

    let grid_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_grid"),
        contents: bytemuck::cast_slice(grid),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_fitness_out"),
        size: u64::from(n) * u64::from(n) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&grid_buf, &fitness_buf, n, b, c);

    let fitness = gpu.read_buffer_f32(&fitness_buf, nn)?;
    let mean = fitness.iter().sum::<f32>() / fitness.len() as f32;
    Ok(mean)
}

// ── Validation functions ───────────────────────────────────────────

fn validate_small_grid(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 4_usize;
    let b = 0.5_f32;
    let c = 0.3_f32;
    let mut rng = Rng::new(42);
    let grid: Vec<u32> = (0..n * n).map(|_| u32::from(rng.uniform() > 0.5)).collect();

    let cpu_mean = cpu_spatial_mean_fitness(&grid, n, b, c);

    match gpu_spatial_mean_fitness(gpu, &grid, n as u32, b, c) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("ecology small 4×4: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_SPATIAL_PAYOFF_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("ecology small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_larger_grid(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 16_usize;
    let b = 0.5_f32;
    let c = 0.3_f32;
    let mut rng = Rng::new(777);
    let grid: Vec<u32> = (0..n * n).map(|_| u32::from(rng.uniform() > 0.5)).collect();

    let cpu_mean = cpu_spatial_mean_fitness(&grid, n, b, c);

    match gpu_spatial_mean_fitness(gpu, &grid, n as u32, b, c) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("ecology larger 16×16: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_SPATIAL_PAYOFF_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("ecology larger: dispatch failed — {e}"), false);
        }
    }
}

fn validate_all_cooperators(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 4_usize;
    let b = 0.5_f32;
    let c = 0.3_f32;
    let grid: Vec<u32> = vec![1; n * n];

    let cpu_mean = cpu_spatial_mean_fitness(&grid, n, b, c);

    match gpu_spatial_mean_fitness(gpu, &grid, n as u32, b, c) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("ecology all cooperators: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_SPATIAL_PAYOFF_F32,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("ecology all cooperators: dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_all_defectors(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 4_usize;
    let b = 0.5_f32;
    let c = 0.3_f32;
    let grid: Vec<u32> = vec![0; n * n];

    match gpu_spatial_mean_fitness(gpu, &grid, n as u32, b, c) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("ecology all defectors: mean fitness={gpu_mean:.6} vs 0"),
                f64::from(gpu_mean),
                0.0,
                tolerances::GPU_SPATIAL_PAYOFF_F32,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("ecology all defectors: dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 8_usize;
    let b = 0.5_f32;
    let c = 0.3_f32;
    let mut rng = Rng::new(42);
    let grid: Vec<u32> = (0..n * n).map(|_| u32::from(rng.uniform() > 0.5)).collect();

    let r1 = gpu_spatial_mean_fitness(gpu, &grid, n as u32, b, c);
    let r2 = gpu_spatial_mean_fitness(gpu, &grid, n as u32, b, c);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("ecology determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("ecology determinism: dispatch failed", false);
        }
    }
}
