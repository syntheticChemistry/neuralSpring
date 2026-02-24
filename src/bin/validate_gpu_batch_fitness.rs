// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: batch fitness evaluation via `BarraCUDA` `BatchFitnessGpu` API.
//!
//! Validates `barracuda::ops::bio::BatchFitnessGpu` (f64 pipeline) against CPU
//! dot-product fitness computation.
//!
//! Evolution path:
//! ```text
//! Python (numpy.dot) → Rust CPU (loop) → BarraCUDA CPU (variance)
//!   → GPU WGSL shader (batch_fitness_eval_f64.wgsl) → ToadStool absorption
//! ```
//!
//! ## Papers validated
//!
//! - Paper 011: Counterdiabatic Evolution (Iram/Dolson, 2020)
//! - Paper 012: MODES Toolbox (Dolson et al., 2019)
//! - Paper 013: Ecological Dynamics (Dolson & Ofria, 2018)
//! - Paper 014: Directed Evolution (Dolson et al., 2022)
//! - Paper 015: Swarm Robotics (Foreback/Dolson, 2025)
//!
//! ## Provenance
//!
//! CPU reference: `directed_evolution::multi_objective_fitness` (linear dot-product).
//! GPU: `barracuda::ops::bio::BatchFitnessGpu` (f64 pipeline)
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use barracuda::ops::bio::BatchFitnessGpu;
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
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let mut h = ValidationHarness::new("gpu_batch_fitness");

    let op = BatchFitnessGpu::new(Arc::clone(gpu.wgpu_device()));

    validate_small_population(&mut h, &gpu, &op);
    validate_uniform_weights(&mut h, &gpu, &op);
    validate_zero_genotype(&mut h, &gpu, &op);
    validate_larger_population(&mut h, &gpu, &op);
    validate_determinism(&mut h, &gpu, &op);

    h.finish();
}

fn cpu_batch_fitness(
    population: &[f64],
    weights: &[f64],
    pop_size: usize,
    genome_len: usize,
) -> Vec<f64> {
    (0..pop_size)
        .map(|i| {
            let base = i * genome_len;
            (0..genome_len)
                .map(|g| population[base + g] * weights[g])
                .sum()
        })
        .collect()
}

fn gpu_batch_fitness(
    op: &BatchFitnessGpu,
    gpu: &Gpu,
    population: &[f64],
    weights: &[f64],
    pop_size: u32,
    genome_len: u32,
) -> Result<Vec<f64>, String> {
    let device = gpu.device();

    let pop_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("population"),
        contents: bytemuck::cast_slice(population),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let weight_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weights"),
        contents: bytemuck::cast_slice(weights),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fitness"),
        size: u64::from(pop_size) * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&pop_buf, &weight_buf, &fitness_buf, pop_size, genome_len);

    gpu.read_buffer_f64(&fitness_buf, pop_size as usize)
}

fn validate_small_population(h: &mut ValidationHarness, gpu: &Gpu, op: &BatchFitnessGpu) {
    let pop_size = 8_u32;
    let genome_len = 4_u32;
    let mut rng = Rng::new(42);

    let population: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..genome_len).map(|_| rng.uniform()).collect();

    let cpu = cpu_batch_fitness(
        &population,
        &weights,
        pop_size as usize,
        genome_len as usize,
    );

    match gpu_batch_fitness(op, gpu, &population, &weights, pop_size, genome_len) {
        Ok(gpu_fitness) => {
            h.check_bool(
                &format!("small pop: correct count ({})", gpu_fitness.len()),
                gpu_fitness.len() == pop_size as usize,
            );

            for (i, (&g, &c)) in gpu_fitness.iter().zip(cpu.iter()).enumerate() {
                h.check_abs(
                    &format!("small pop[{i}]: GPU ≈ CPU ({g:.6} vs {c:.6})"),
                    g,
                    c,
                    tolerances::GPU_FITNESS_F32,
                );
            }
        }
        Err(e) => {
            h.check_bool(&format!("small pop: dispatch failed — {e}"), false);
        }
    }
}

fn validate_uniform_weights(h: &mut ValidationHarness, gpu: &Gpu, op: &BatchFitnessGpu) {
    let pop_size = 4_u32;
    let genome_len = 8_u32;
    let weights: Vec<f64> = vec![1.0; genome_len as usize];
    let population: Vec<f64> = (0..pop_size * genome_len)
        .map(|i| f64::from(i % genome_len))
        .collect();

    let expected_sum: f64 = (0..genome_len).map(f64::from).sum();

    match gpu_batch_fitness(op, gpu, &population, &weights, pop_size, genome_len) {
        Ok(gpu_fitness) => {
            for (i, &g) in gpu_fitness.iter().enumerate() {
                h.check_abs(
                    &format!("uniform weights[{i}]: sum={g:.2} vs {expected_sum:.2}"),
                    g,
                    expected_sum,
                    tolerances::GPU_FITNESS_F32,
                );
            }
        }
        Err(e) => {
            h.check_bool(&format!("uniform weights: failed — {e}"), false);
        }
    }
}

fn validate_zero_genotype(h: &mut ValidationHarness, gpu: &Gpu, op: &BatchFitnessGpu) {
    let pop_size = 4_u32;
    let genome_len = 8_u32;
    let weights: Vec<f64> = vec![1.0; genome_len as usize];
    let population: Vec<f64> = vec![0.0; (pop_size * genome_len) as usize];

    match gpu_batch_fitness(op, gpu, &population, &weights, pop_size, genome_len) {
        Ok(gpu_fitness) => {
            for (i, &g) in gpu_fitness.iter().enumerate() {
                h.check_abs(
                    &format!("zero genotype[{i}]: fitness={g:.6} vs 0.0"),
                    g,
                    0.0,
                    tolerances::GPU_FITNESS_F32,
                );
            }
        }
        Err(e) => {
            h.check_bool(&format!("zero genotype: failed — {e}"), false);
        }
    }
}

fn validate_larger_population(h: &mut ValidationHarness, gpu: &Gpu, op: &BatchFitnessGpu) {
    let pop_size = 512_u32;
    let genome_len = 16_u32;
    let mut rng = Rng::new(777);

    let population: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..genome_len).map(|_| rng.uniform()).collect();

    let cpu = cpu_batch_fitness(
        &population,
        &weights,
        pop_size as usize,
        genome_len as usize,
    );

    match gpu_batch_fitness(op, gpu, &population, &weights, pop_size, genome_len) {
        Ok(gpu_fitness) => {
            let max_diff: f64 = gpu_fitness
                .iter()
                .zip(cpu.iter())
                .map(|(&g, &c)| (g - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_bool(
                &format!("512 individuals: max diff {max_diff:.2e} < tol"),
                max_diff < tolerances::GPU_FITNESS_F32,
            );

            h.check_bool(
                &format!("512 individuals: correct count ({})", gpu_fitness.len()),
                gpu_fitness.len() == pop_size as usize,
            );
        }
        Err(e) => {
            h.check_bool(&format!("512 pop: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu, op: &BatchFitnessGpu) {
    let pop_size = 32_u32;
    let genome_len = 8_u32;
    let mut rng = Rng::new(42);

    let population: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..genome_len).map(|_| rng.uniform()).collect();

    let run1 = gpu_batch_fitness(op, gpu, &population, &weights, pop_size, genome_len);
    let run2 = gpu_batch_fitness(op, gpu, &population, &weights, pop_size, genome_len);

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(&a, &b)| (a - b).abs() < f64::EPSILON);
            h.check_bool("determinism: two runs identical", identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}
