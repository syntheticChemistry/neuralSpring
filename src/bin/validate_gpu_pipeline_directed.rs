// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline validation: `multi_obj_fitness` → mean (Paper 014).
//!
//! Uses `BarraCUDA` typed op `MultiObjFitnessGpu` (f64) with CPU mean reduction.
//! Replaces raw wgpu chain (`multi_obj_fitness` + `mean_reduce`) for validation.
//!
//! ## Pipeline
//!
//! ```text
//! Upload genotypes [pop_size x genome_len] (once)
//!   ↓
//! MultiObjFitnessGpu.dispatch() → fitness[pop_size * n_objectives] (f64)
//!   ↓
//! CPU mean(fitness)
//! ```
//!
//! ## Provenance
//!
//! GPU op: `barracuda::ops::bio::MultiObjFitnessGpu` (f64 pipeline)
//! Validates: end-to-end GPU-resident computation with scalar-only readback.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use barracuda::ops::bio::MultiObjFitnessGpu;
use neural_spring::directed_evolution::multi_objective_fitness;
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

    let mut h = ValidationHarness::new("gpu_pipeline_directed");

    validate_single(&mut h, &gpu);
    validate_batch(&mut h, &gpu);
    validate_uniform(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_fitness(genotypes: &[Vec<f64>], n_objectives: usize) -> f64 {
    let fitnesses: Vec<f64> = genotypes
        .iter()
        .flat_map(|g| multi_objective_fitness(g, n_objectives))
        .collect();
    if fitnesses.is_empty() {
        return 0.0;
    }
    fitnesses.iter().sum::<f64>() / fitnesses.len() as f64
}

// ── GPU via BarraCUDA typed op ──────────────────────────────────────

fn gpu_multi_obj_mean(
    gpu: &Gpu,
    genotypes: &[f64],
    pop_size: u32,
    genome_len: u32,
    n_objectives: u32,
) -> Result<f64, String> {
    let op = MultiObjFitnessGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let n_fitness = (pop_size * n_objectives) as usize;

    let genotypes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_genotypes"),
        contents: bytemuck::cast_slice(genotypes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_fitness_out"),
        size: (n_fitness * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(
        &genotypes_buf,
        &fitness_buf,
        pop_size,
        genome_len,
        n_objectives,
    );

    let fitness = gpu.read_buffer_f64(&fitness_buf, n_fitness)?;
    let mean = fitness.iter().sum::<f64>() / fitness.len() as f64;
    Ok(mean)
}

// ── Validation functions ───────────────────────────────────────────

fn validate_single(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 1_u32;
    let genome_len = 40_u32;
    let n_objectives = 4_u32;

    let mut rng = Rng::new(42);
    let genotype_f64: Vec<f64> = (0..genome_len as usize).map(|_| rng.uniform()).collect();

    let genotypes: Vec<Vec<f64>> = vec![genotype_f64.clone()];
    let cpu_mean = cpu_mean_fitness(&genotypes, n_objectives as usize);

    match gpu_multi_obj_mean(gpu, &genotype_f64, pop_size, genome_len, n_objectives) {
        Ok(gpu_mean) => {
            h.check_upper(
                &format!("directed single 1×4: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                (gpu_mean - cpu_mean).abs(),
                tolerances::GPU_MULTI_OBJ_BESSEL_F64,
            );
        }
        Err(e) => {
            h.check_bool(&format!("directed single: dispatch failed — {e}"), false);
        }
    }
}

fn validate_batch(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 10_u32;
    let genome_len = 40_u32;
    let n_objectives = 4_u32;

    let mut rng = Rng::new(77);
    let genotypes_f64: Vec<Vec<f64>> = (0..pop_size as usize)
        .map(|_| (0..genome_len as usize).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_mean = cpu_mean_fitness(&genotypes_f64, n_objectives as usize);

    let genotypes_flat: Vec<f64> = genotypes_f64
        .iter()
        .flat_map(|g| g.iter().copied())
        .collect();

    match gpu_multi_obj_mean(gpu, &genotypes_flat, pop_size, genome_len, n_objectives) {
        Ok(gpu_mean) => {
            h.check_upper(
                &format!("directed batch 10×4: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                (gpu_mean - cpu_mean).abs(),
                tolerances::GPU_MULTI_OBJ_BESSEL_F64,
            );
        }
        Err(e) => {
            h.check_bool(&format!("directed batch: dispatch failed — {e}"), false);
        }
    }
}

fn validate_uniform(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 1_u32;
    let genome_len = 40_u32;
    let n_objectives = 4_u32;

    let genotypes_flat: Vec<f64> = vec![0.5; (pop_size * genome_len) as usize];

    match gpu_multi_obj_mean(gpu, &genotypes_flat, pop_size, genome_len, n_objectives) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("directed uniform 0.5: mean≈0.5, GPU={gpu_mean:.6}"),
                gpu_mean,
                0.5,
                tolerances::GPU_MULTI_OBJ_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("directed uniform: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 5_u32;
    let genome_len = 40_u32;
    let n_objectives = 4_u32;

    let mut rng = Rng::new(123);
    let genotypes_flat: Vec<f64> = (0..(pop_size * genome_len) as usize)
        .map(|_| rng.uniform())
        .collect();

    let r1 = gpu_multi_obj_mean(gpu, &genotypes_flat, pop_size, genome_len, n_objectives);
    let r2 = gpu_multi_obj_mean(gpu, &genotypes_flat, pop_size, genome_len, n_objectives);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("directed determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f64::EPSILON,
            );
        }
        _ => {
            h.check_bool("directed determinism: dispatch failed", false);
        }
    }
}
