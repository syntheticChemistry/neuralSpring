// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: Fermi imitation stencil via `BarraCUDA` `StencilCooperationGpu` API.
//!
//! Validates `barracuda::ops::bio::StencilCooperationGpu` against CPU reference.
//! The GPU shader updates each cell's strategy by comparing fitness with a
//! deterministic neighbor; adoption probability follows the Fermi function.
//!
//! ## Papers validated
//!
//! - Paper 019: Game Theory (spatial evolutionary PD, imitation dynamics)
//!
//! ## Provenance
//!
//! Upstream: `barracuda::ops::bio::StencilCooperationGpu` (f64 pipeline)
//! CPU reference: inline `cpu_stencil_update`, `cpu_pd_fitness`

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "validation binary"
)]

use barracuda::ops::bio::StencilCooperationGpu;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// PD payoff stencil over Moore neighborhood (8 neighbors) with periodic boundary.
/// (1,1)→b-c, (1,0)→-c, (0,1)→b, (0,0)→0.0.
#[must_use]
fn cpu_pd_fitness(grid: &[u32], grid_size: usize, b: f64, c: f64) -> Vec<f64> {
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
            let mut total = 0.0_f64;
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

fn cpu_stencil_update(
    strategies: &[u32],
    fitness: &[f64],
    grid_size: usize,
    kappa: f64,
    step: u32,
) -> Vec<u32> {
    let n = grid_size as i32;
    let offsets: [(i32, i32); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];
    (0..grid_size * grid_size)
        .map(|idx| {
            let i = (idx / grid_size) as i32;
            let j = (idx % grid_size) as i32;
            let neighbor_select = (idx as u32 + step) % 8;
            let (di, dj) = offsets[neighbor_select as usize];
            let nb_i = ((i + di + n) % n) as usize;
            let nb_j = ((j + dj + n) % n) as usize;
            let nb_idx = nb_i * grid_size + nb_j;
            let f_self = fitness[idx];
            let f_nb = fitness[nb_idx];
            let p_adopt = 1.0 / (1.0 + ((f_self - f_nb) / kappa).exp());
            if p_adopt > 0.5 {
                strategies[nb_idx]
            } else {
                strategies[idx]
            }
        })
        .collect()
}

fn read_buffer_u32(gpu: &Gpu, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<u32>, String> {
    let staging = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (count * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 4) as u64);
    gpu.queue().submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    let _ = gpu.device().poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| format!("{e:?}"))?;
    let data = slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    Ok(result)
}

fn gpu_stencil(
    gpu: &Gpu,
    strategies: &[u32],
    fitness: &[f64],
    grid_size: u32,
    kappa: f64,
    step: u32,
) -> Result<Vec<u32>, String> {
    let device = gpu.device();
    let op = StencilCooperationGpu::new(Arc::clone(gpu.wgpu_device()));

    let strategies_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("strategies"),
        contents: bytemuck::cast_slice(strategies),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fitness_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fitness"),
        contents: bytemuck::cast_slice(fitness),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let n_cells = (grid_size * grid_size) as usize;
    let new_strategies_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("new_strategies"),
        size: (n_cells * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(
        &strategies_buf,
        &fitness_buf,
        &new_strategies_buf,
        grid_size,
        kappa,
        step,
    );

    read_buffer_u32(gpu, &new_strategies_buf, n_cells)
}

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

    let mut h = ValidationHarness::new("gpu_stencil");

    validate_basic_update(&mut h, &gpu);
    validate_all_cooperators(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

fn validate_basic_update(h: &mut ValidationHarness, gpu: &Gpu) {
    let grid_size = 6_usize;
    let kappa = 0.1_f64;
    let step = 0_u32;
    let b = 3.0_f64;
    let c = 1.0_f64;

    let mut rng = Rng::new(42);
    let strategies: Vec<u32> = (0..grid_size * grid_size)
        .map(|_| u32::from(rng.uniform() >= 0.5))
        .collect();

    let fitness = cpu_pd_fitness(&strategies, grid_size, b, c);

    let cpu_new = cpu_stencil_update(&strategies, &fitness, grid_size, kappa, step);

    match gpu_stencil(gpu, &strategies, &fitness, grid_size as u32, kappa, step) {
        Ok(gpu_new) => {
            let all_match = gpu_new.iter().zip(cpu_new.iter()).all(|(&g, &c)| g == c);
            h.check_bool(
                "basic update 6×6: all cells match CPU (strategies from Rng 42, PD b=3 c=1)",
                all_match,
            );
        }
        Err(e) => {
            h.check_bool(&format!("basic update: dispatch failed — {e}"), false);
        }
    }
}

fn validate_all_cooperators(h: &mut ValidationHarness, gpu: &Gpu) {
    let grid_size = 6_usize;
    let kappa = 0.1_f64;
    let step = 0_u32;
    let b = 3.0_f64;
    let c = 1.0_f64;

    let strategies: Vec<u32> = vec![1; grid_size * grid_size];
    let fitness = cpu_pd_fitness(&strategies, grid_size, b, c);

    match gpu_stencil(gpu, &strategies, &fitness, grid_size as u32, kappa, step) {
        Ok(gpu_new) => {
            let all_cooperators = gpu_new.iter().all(|&s| s == 1);
            h.check_bool(
                "all cooperators: strategies unchanged (equal fitness → p_adopt=0.5, not >0.5)",
                all_cooperators,
            );
        }
        Err(e) => {
            h.check_bool(&format!("all cooperators: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let grid_size = 6_usize;
    let kappa = 0.1_f64;
    let step = 0_u32;

    let mut rng = Rng::new(42);
    let strategies: Vec<u32> = (0..grid_size * grid_size)
        .map(|_| u32::from(rng.uniform() >= 0.5))
        .collect();
    let b = 3.0_f64;
    let c = 1.0_f64;
    let fitness = cpu_pd_fitness(&strategies, grid_size, b, c);

    let run1 = gpu_stencil(gpu, &strategies, &fitness, grid_size as u32, kappa, step);
    let run2 = gpu_stencil(gpu, &strategies, &fitness, grid_size as u32, kappa, step);

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let identical = r1.iter().zip(r2.iter()).all(|(a, b)| *a == *b);
            h.check_bool("determinism: two stencil_update runs identical", identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}
