// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline validation: batch fitness → mean (Paper 011).
//!
//! Uses BarraCUDA typed op `BatchFitnessGpu` (f64) with CPU mean reduction.
//! Replaces raw wgpu chain (batch_fitness_eval + mean_reduce) for validation.
//!
//! ## Pipeline
//!
//! ```text
//! Upload population + weights (once)
//!   ↓
//! BatchFitnessGpu.dispatch() → fitness[pop_size] (f64)
//!   ↓
//! CPU mean(fitness)
//! ```
//!
//! ## Provenance
//!
//! GPU op: `barracuda::ops::bio::BatchFitnessGpu` (f64 pipeline)
//! Validates: end-to-end GPU-resident computation for Paper 011 (Counterdiabatic).

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
        Err(e) => {
            eprintln!("  SKIP: {e} — no GPU/CPU adapter available");
            eprintln!("  0/0 checks — skipping gracefully");
            std::process::exit(0);
        }
    };

    let mut h = ValidationHarness::new("gpu_pipeline_fitness");

    validate_small_population(&mut h, &gpu);
    validate_larger_population(&mut h, &gpu);
    validate_uniform_weights(&mut h, &gpu);
    validate_zero_genotype(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_fitness(genotypes: &[f64], weights: &[f64], pop_size: usize, genome_len: usize) -> f64 {
    let total: f64 = (0..pop_size)
        .map(|i| {
            let base = i * genome_len;
            (0..genome_len)
                .map(|g| genotypes[base + g] * weights[g])
                .sum::<f64>()
        })
        .sum();
    total / pop_size as f64
}

// ── GPU via BarraCUDA typed op ──────────────────────────────────────

fn gpu_mean_fitness(
    gpu: &Gpu,
    genotypes: &[f64],
    weights: &[f64],
    pop_size: u32,
    genome_len: u32,
) -> Result<f64, String> {
    let op = BatchFitnessGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();

    let geno_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_genotypes"),
        contents: bytemuck::cast_slice(genotypes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let weight_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_weights"),
        contents: bytemuck::cast_slice(weights),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_fitness_out"),
        size: u64::from(pop_size) * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&geno_buf, &weight_buf, &fitness_buf, pop_size, genome_len);

    let fitness = gpu.read_buffer_f64(&fitness_buf, pop_size as usize)?;
    let mean = fitness.iter().sum::<f64>() / fitness.len() as f64;
    Ok(mean)
}

// ── Validation functions ───────────────────────────────────────────

fn validate_small_population(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 8_usize;
    let genome_len = 16_usize;
    let mut rng = Rng::new(42);
    let genotypes: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..genome_len).map(|_| rng.uniform()).collect();

    let cpu_mean = cpu_mean_fitness(&genotypes, &weights, pop_size, genome_len);

    match gpu_mean_fitness(
        gpu,
        &genotypes,
        &weights,
        pop_size as u32,
        genome_len as u32,
    ) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("fitness small 8×16: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("fitness small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_larger_population(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 64_usize;
    let genome_len = 32_usize;
    let mut rng = Rng::new(777);
    let genotypes: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..genome_len).map(|_| rng.uniform()).collect();

    let cpu_mean = cpu_mean_fitness(&genotypes, &weights, pop_size, genome_len);

    match gpu_mean_fitness(
        gpu,
        &genotypes,
        &weights,
        pop_size as u32,
        genome_len as u32,
    ) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("fitness larger 64×32: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("fitness larger: dispatch failed — {e}"), false);
        }
    }
}

fn validate_uniform_weights(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 16_usize;
    let genome_len = 8_usize;
    let mut rng = Rng::new(123);
    let genotypes: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = vec![1.0; genome_len];

    let cpu_mean = cpu_mean_fitness(&genotypes, &weights, pop_size, genome_len);

    match gpu_mean_fitness(
        gpu,
        &genotypes,
        &weights,
        pop_size as u32,
        genome_len as u32,
    ) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("fitness uniform weights: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("fitness uniform weights: dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_zero_genotype(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 4_usize;
    let genome_len = 4_usize;
    let genotypes: Vec<f64> = vec![0.0; pop_size * genome_len];
    let weights: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

    match gpu_mean_fitness(
        gpu,
        &genotypes,
        &weights,
        pop_size as u32,
        genome_len as u32,
    ) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("fitness zero genotype: mean={gpu_mean:.6} vs 0"),
                gpu_mean,
                0.0,
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("fitness zero genotype: dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 32_usize;
    let genome_len = 12_usize;
    let mut rng = Rng::new(99);
    let genotypes: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..genome_len).map(|_| rng.uniform()).collect();

    let r1 = gpu_mean_fitness(
        gpu,
        &genotypes,
        &weights,
        pop_size as u32,
        genome_len as u32,
    );
    let r2 = gpu_mean_fitness(
        gpu,
        &genotypes,
        &weights,
        pop_size as u32,
        genome_len as u32,
    );

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("fitness determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f64::EPSILON,
            );
        }
        _ => {
            h.check_bool("fitness determinism: dispatch failed", false);
        }
    }
}
