// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: spatial PD payoff via `BarraCUDA` `SpatialPayoffGpu`.
//!
//! Validates `barracuda::ops::bio::SpatialPayoffGpu` against CPU PD payoff
//! stencil from `game_theory.rs`. The GPU op computes fitness for each
//! cell in a 2D grid using Moore neighborhood (8 neighbors) with periodic
//! boundary.
//!
//! ## Papers validated
//!
//! - Paper 019: Game Theory (spatial prisoner's dilemma)
//!
//! ## Provenance
//!
//! CPU reference: `game_theory::spatial_cooperation` (seed=42, 10×10 grid).
//! GPU op: `barracuda::ops::bio::SpatialPayoffGpu`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "validation binary"
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
            println!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let op = SpatialPayoffGpu::new(Arc::clone(gpu.wgpu_device()));
    let mut h = ValidationHarness::new("gpu_game_theory");

    validate_small_grid(&mut h, &gpu, &op);
    validate_larger_grid(&mut h, &gpu, &op);
    validate_determinism(&mut h, &gpu, &op);
    validate_all_cooperators(&mut h, &gpu, &op);

    h.finish();
}

/// CPU reference: PD payoff stencil over Moore neighborhood (8 neighbors) with periodic boundary.
/// (1,1) → b-c, (1,0) → -c, (0,1) → b, (0,0) → 0.0. Sum all neighbor payoffs into fitness.
#[must_use]
fn cpu_spatial_fitness(grid: &[u32], grid_size: usize, b: f32, c: f32) -> Vec<f32> {
    let n = grid_size as i32;
    let neighbors: [(i32, i32); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];

    let mut fitness = Vec::with_capacity(grid_size * grid_size);
    for i in 0..grid_size {
        for j in 0..grid_size {
            let me = grid[i * grid_size + j];
            let mut total = 0.0_f32;
            for (di, dj) in &neighbors {
                let ni = ((i as i32 + di).rem_euclid(n)) as usize;
                let nj = ((j as i32 + dj).rem_euclid(n)) as usize;
                let other = grid[ni * grid_size + nj];
                total += match (me, other) {
                    (1, 1) => b - c,
                    (1, 0) => -c,
                    (0, 1) => b,
                    _ => 0.0,
                };
            }
            fitness.push(total);
        }
    }
    fitness
}

fn gpu_spatial_payoff(
    gpu: &Gpu,
    op: &SpatialPayoffGpu,
    grid: &[u32],
    grid_size: u32,
    b: f32,
    c: f32,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let n_cells = (grid_size * grid_size) as usize;

    let grid_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grid"),
        contents: bytemuck::cast_slice(grid),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fitness"),
        size: (n_cells * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&grid_buf, &fitness_buf, grid_size, b, c);

    gpu.read_buffer_f32(&fitness_buf, n_cells)
}

fn make_grid(grid_size: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    (0..grid_size * grid_size)
        .map(|_| u32::from(rng.uniform() >= 0.5))
        .collect()
}

fn validate_small_grid(h: &mut ValidationHarness, gpu: &Gpu, op: &SpatialPayoffGpu) {
    let grid_size = 10_usize;
    let seed = 42_u64;
    let b = 3.0_f32;
    let c = 1.0_f32;

    let grid = make_grid(grid_size, seed);
    let cpu_fitness = cpu_spatial_fitness(&grid, grid_size, b, c);

    match gpu_spatial_payoff(gpu, op, &grid, grid_size as u32, b, c) {
        Ok(gpu_fitness) => {
            h.check_bool(
                &format!("small grid: correct cell count ({})", gpu_fitness.len()),
                gpu_fitness.len() == cpu_fitness.len(),
            );

            let max_diff: f64 = gpu_fitness
                .iter()
                .zip(cpu_fitness.iter())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("small grid: max GPU-CPU diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_SPATIAL_PAYOFF_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("small grid: dispatch failed — {e}"), false);
        }
    }
}

fn validate_larger_grid(h: &mut ValidationHarness, gpu: &Gpu, op: &SpatialPayoffGpu) {
    let grid_size = 20_usize;
    let seed = 77_u64;
    let b = 3.0_f32;
    let c = 1.0_f32;

    let grid = make_grid(grid_size, seed);
    let cpu_fitness = cpu_spatial_fitness(&grid, grid_size, b, c);

    match gpu_spatial_payoff(gpu, op, &grid, grid_size as u32, b, c) {
        Ok(gpu_fitness) => {
            let max_diff: f64 = gpu_fitness
                .iter()
                .zip(cpu_fitness.iter())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "20×20 grid: max GPU-CPU diff ({max_diff:.2e}), {} cells",
                    gpu_fitness.len()
                ),
                max_diff,
                tolerances::GPU_SPATIAL_PAYOFF_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("20×20 grid: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu, op: &SpatialPayoffGpu) {
    let grid_size = 10_usize;
    let seed = 42_u64;
    let b = 3.0_f32;
    let c = 1.0_f32;

    let grid = make_grid(grid_size, seed);

    let run1 = gpu_spatial_payoff(gpu, op, &grid, grid_size as u32, b, c);
    let run2 = gpu_spatial_payoff(gpu, op, &grid, grid_size as u32, b, c);

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(&a, &b)| (a - b).abs() < f32::EPSILON);
            h.check_bool("determinism: two spatial_payoff runs identical", identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}

fn validate_all_cooperators(h: &mut ValidationHarness, gpu: &Gpu, op: &SpatialPayoffGpu) {
    let grid_size = 10_usize;
    let b = 3.0_f32;
    let c = 1.0_f32;
    let expected_fitness = 8.0 * (b - c); // 8 * 2.0 = 16.0

    let grid: Vec<u32> = vec![1; grid_size * grid_size];

    match gpu_spatial_payoff(gpu, op, &grid, grid_size as u32, b, c) {
        Ok(gpu_fitness) => {
            let max_diff: f64 = gpu_fitness
                .iter()
                .map(|&g| (f64::from(g) - f64::from(expected_fitness)).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("all cooperators: max diff from 8*(b-c)=16 ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_SPATIAL_PAYOFF_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("all cooperators: dispatch failed — {e}"), false);
        }
    }
}
