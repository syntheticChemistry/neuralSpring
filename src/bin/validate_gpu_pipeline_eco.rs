// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline validation: batch fitness → mean (Paper 013).
//!
//! Uses `BarraCUDA` typed op `BatchFitnessGpu` (f64) with CPU mean reduction.
//! Replaces raw wgpu chain (`batch_fitness_eval` + `mean_reduce`) for validation.
//! Eco dynamics uses the same `batch_fitness` op as Paper 011; validates mean
//! fitness across ecological niches.
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
//! Validates: ecological dynamics mean fitness (Dolson & Ofria, 2018).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::many_single_char_names,
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

    let mut h = ValidationHarness::new("gpu_pipeline_eco");

    validate_eco_small(&mut h, &gpu);
    validate_eco_larger(&mut h, &gpu);
    validate_eco_extreme_niche(&mut h, &gpu);
    validate_eco_diverse_genotypes(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_batch_fitness(
    genotypes: &[f64],
    weights: &[f64],
    pop_size: usize,
    genome_len: usize,
) -> f64 {
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

fn gpu_mean_batch_fitness(
    gpu: &Gpu,
    genotypes: &[f64],
    weights: &[f64],
    pop_size: u32,
    genome_len: u32,
) -> Result<f64, String> {
    let op = BatchFitnessGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();

    let geno_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_eco_genotypes"),
        contents: bytemuck::cast_slice(genotypes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let weight_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_eco_weights"),
        contents: bytemuck::cast_slice(weights),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_eco_fitness_out"),
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

fn validate_eco_small(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 12_usize;
    let genome_len = 10_usize;
    let mut rng = Rng::new(42);
    let genotypes: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..genome_len)
        .map(|i| (i as f64 + 0.5) / genome_len as f64)
        .collect();

    let cpu_mean = cpu_mean_batch_fitness(&genotypes, &weights, pop_size, genome_len);

    match gpu_mean_batch_fitness(
        gpu,
        &genotypes,
        &weights,
        pop_size as u32,
        genome_len as u32,
    ) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("eco small 12×10: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("eco small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_eco_larger(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 48_usize;
    let genome_len = 24_usize;
    let mut rng = Rng::new(100);
    let genotypes: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..genome_len).map(|_| rng.uniform()).collect();

    let cpu_mean = cpu_mean_batch_fitness(&genotypes, &weights, pop_size, genome_len);

    match gpu_mean_batch_fitness(
        gpu,
        &genotypes,
        &weights,
        pop_size as u32,
        genome_len as u32,
    ) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("eco larger 48×24: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("eco larger: dispatch failed — {e}"), false);
        }
    }
}

fn validate_eco_extreme_niche(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 8_usize;
    let genome_len = 8_usize;
    let genotypes: Vec<f64> = vec![1.0; pop_size * genome_len];
    let weights: Vec<f64> = vec![0.1; genome_len];

    let cpu_mean = cpu_mean_batch_fitness(&genotypes, &weights, pop_size, genome_len);

    match gpu_mean_batch_fitness(
        gpu,
        &genotypes,
        &weights,
        pop_size as u32,
        genome_len as u32,
    ) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("eco extreme niche: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("eco extreme niche: dispatch failed — {e}"), false);
        }
    }
}

fn validate_eco_diverse_genotypes(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 16_usize;
    let genome_len = 6_usize;
    let mut genotypes = vec![0.0_f64; pop_size * genome_len];
    for i in 0..pop_size {
        for g in 0..genome_len {
            genotypes[i * genome_len + g] = if (i + g) % 2 == 0 { 1.0 } else { 0.0 };
        }
    }
    let weights: Vec<f64> = vec![1.0, -0.5, 0.5, -0.25, 0.25, 0.0];

    let cpu_mean = cpu_mean_batch_fitness(&genotypes, &weights, pop_size, genome_len);

    match gpu_mean_batch_fitness(
        gpu,
        &genotypes,
        &weights,
        pop_size as u32,
        genome_len as u32,
    ) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("eco diverse genotypes: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("eco diverse: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 24_usize;
    let genome_len = 14_usize;
    let mut rng = Rng::new(333);
    let genotypes: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..genome_len).map(|_| rng.uniform()).collect();

    let r1 = gpu_mean_batch_fitness(
        gpu,
        &genotypes,
        &weights,
        pop_size as u32,
        genome_len as u32,
    );
    let r2 = gpu_mean_batch_fitness(
        gpu,
        &genotypes,
        &weights,
        pop_size as u32,
        genome_len as u32,
    );

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("eco determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f64::EPSILON,
            );
        }
        _ => {
            h.check_bool("eco determinism: dispatch failed", false);
        }
    }
}
