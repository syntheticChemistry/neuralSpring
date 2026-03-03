// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: multi-objective fitness via `BarraCUDA` `MultiObjFitnessGpu`.
//!
//! Validates `barracuda::ops::bio::MultiObjFitnessGpu` against CPU
//! `directed_evolution::multi_objective_fitness`. The GPU op computes
//! per-chunk mean + 0.1*std for each (individual, objective) pair.
//!
//! ## Papers validated
//!
//! - Paper 014: Directed Evolution (multi-objective fitness)
//!
//! ## Provenance
//!
//! CPU reference: `directed_evolution::multi_objective_fitness` (seed=42, `pop_size`=10 `n_objectives`=4).
//! GPU op: `barracuda::ops::bio::MultiObjFitnessGpu`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

use barracuda::ops::bio::MultiObjFitnessGpu;
use neural_spring::directed_evolution::multi_objective_fitness;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use wgpu::util::DeviceExt;

fn gpu_multi_obj_fitness(
    gpu: &Gpu,
    op: &MultiObjFitnessGpu,
    genotypes: &[f64],
    pop: u32,
    genome_len: u32,
    n_obj: u32,
) -> Result<Vec<f64>, String> {
    let device = gpu.device();
    let n_fitness = (pop * n_obj) as usize;

    let genotypes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("genotypes"),
        contents: bytemuck::cast_slice(genotypes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fitness"),
        size: (n_fitness * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&genotypes_buf, &fitness_buf, pop, genome_len, n_obj);

    gpu.read_buffer_f64(&fitness_buf, n_fitness)
}

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

    let op = MultiObjFitnessGpu::new(Arc::clone(gpu.wgpu_device()));
    let mut h = ValidationHarness::new("gpu_directed");

    validate_single_genotype(&mut h, &gpu, &op);
    validate_batch(&mut h, &gpu, &op);
    validate_determinism(&mut h, &gpu, &op);
    validate_uniform_genotype(&mut h, &gpu, &op);

    h.finish();
}

fn validate_single_genotype(h: &mut ValidationHarness, gpu: &Gpu, op: &MultiObjFitnessGpu) {
    let pop_size = 1_u32;
    let genome_len = 40_u32;
    let n_objectives = 4_u32;

    let mut rng = Rng::new(42);
    let genotype_f64: Vec<f64> = (0..genome_len as usize).map(|_| rng.uniform()).collect();

    let cpu_fitness = multi_objective_fitness(&genotype_f64, n_objectives as usize);

    match gpu_multi_obj_fitness(gpu, op, &genotype_f64, pop_size, genome_len, n_objectives) {
        Ok(gpu_fitness) => {
            h.check_bool(
                &format!("single genotype: correct count ({})", gpu_fitness.len()),
                gpu_fitness.len() == n_objectives as usize,
            );

            let max_diff: f64 = gpu_fitness
                .iter()
                .zip(cpu_fitness.iter())
                .map(|(&g, &c)| (g - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("single genotype: max GPU-CPU diff ({max_diff:.2e})"),
                max_diff,
                2e-3,
            );
        }
        Err(e) => {
            h.check_bool(&format!("single genotype: dispatch failed — {e}"), false);
        }
    }
}

fn validate_batch(h: &mut ValidationHarness, gpu: &Gpu, op: &MultiObjFitnessGpu) {
    let pop_size = 10_u32;
    let genome_len = 40_u32;
    let n_objectives = 4_u32;

    let mut rng = Rng::new(77);
    let genotypes_f64: Vec<Vec<f64>> = (0..pop_size as usize)
        .map(|_| (0..genome_len as usize).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_fitness: Vec<f64> = genotypes_f64
        .iter()
        .flat_map(|g| multi_objective_fitness(g, n_objectives as usize))
        .collect();

    let genotypes_flat: Vec<f64> = genotypes_f64
        .iter()
        .flat_map(|g| g.iter().copied())
        .collect();

    match gpu_multi_obj_fitness(gpu, op, &genotypes_flat, pop_size, genome_len, n_objectives) {
        Ok(gpu_fitness) => {
            h.check_bool(
                &format!("batch: correct count ({})", gpu_fitness.len()),
                gpu_fitness.len() == (pop_size * n_objectives) as usize,
            );

            let max_diff: f64 = gpu_fitness
                .iter()
                .zip(cpu_fitness.iter())
                .map(|(&g, &c)| (g - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("batch: max GPU-CPU diff ({max_diff:.2e})"),
                max_diff,
                2e-3,
            );
        }
        Err(e) => {
            h.check_bool(&format!("batch: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu, op: &MultiObjFitnessGpu) {
    let pop_size = 5_u32;
    let genome_len = 40_u32;
    let n_objectives = 4_u32;

    let mut rng = Rng::new(123);
    let genotypes_f64: Vec<f64> = (0..(pop_size * genome_len) as usize)
        .map(|_| rng.uniform())
        .collect();

    let run1 = gpu_multi_obj_fitness(gpu, op, &genotypes_f64, pop_size, genome_len, n_objectives);
    let run2 = gpu_multi_obj_fitness(gpu, op, &genotypes_f64, pop_size, genome_len, n_objectives);

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

fn validate_uniform_genotype(h: &mut ValidationHarness, gpu: &Gpu, op: &MultiObjFitnessGpu) {
    let pop_size = 1_u32;
    let genome_len = 40_u32;
    let n_objectives = 4_u32;

    let genotypes_f64: Vec<f64> = vec![0.5_f64; (pop_size * genome_len) as usize];

    match gpu_multi_obj_fitness(gpu, op, &genotypes_f64, pop_size, genome_len, n_objectives) {
        Ok(gpu_fitness) => {
            let expected = 0.5_f64;
            let max_diff: f64 = gpu_fitness
                .iter()
                .map(|&g| (g - expected).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("uniform genotype 0.5: all fitness≈0.5 (max diff {max_diff:.2e})"),
                max_diff,
                2e-3,
            );
        }
        Err(e) => {
            h.check_bool(&format!("uniform genotype: dispatch failed — {e}"), false);
        }
    }
}
