// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: pure GPU Jaccard→mean pipeline (Paper 024).
//!
//! Uses `BarraCUDA` typed op `PairwiseJaccardGpu` (f32) with CPU mean reduction.
//! Replaces raw wgpu chain (`pairwise_jaccard` + `mean_reduce`) for validation.
//!
//! ## Pipeline
//!
//! ```text
//! PA matrix (upload once)
//!   ↓
//! PairwiseJaccardGpu.dispatch() → distances[] (f32)
//!   ↓
//! CPU mean(distances)
//! ```
//!
//! ## Provenance
//!
//! GPU op: `barracuda::ops::bio::PairwiseJaccardGpu` (f32 pipeline)
//! Validates: end-to-end GPU-resident computation with scalar-only readback.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).
//! Note: PA is column-major: pa[gene * `n_genomes` + genome]

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "validation binary"
)]

use barracuda::ops::bio::PairwiseJaccardGpu;
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

    let mut h = ValidationHarness::new("gpu_pipeline_genomics");

    validate_identical_genomes(&mut h, &gpu);
    validate_disjoint_genomes(&mut h, &gpu);
    validate_random_pa_small(&mut h, &gpu);
    validate_random_pa_larger(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_jaccard(pa: &[f32], n_genes: usize, n_genomes: usize) -> f32 {
    let n_pairs = n_genomes * (n_genomes - 1) / 2;
    let mut total = 0.0f32;
    for i in 0..n_genomes {
        for j in (i + 1)..n_genomes {
            let mut inter = 0.0f32;
            let mut union_c = 0.0f32;
            for g in 0..n_genes {
                let a = pa[g * n_genomes + i];
                let b = pa[g * n_genomes + j];
                inter += a * b;
                union_c += a.max(b);
            }
            let dist = if union_c > 0.0 {
                1.0 - inter / union_c
            } else {
                1.0
            };
            total += dist;
        }
    }
    total / n_pairs as f32
}

// ── GPU via BarraCUDA typed op ──────────────────────────────────────

fn gpu_chained_mean_jaccard(
    gpu: &Gpu,
    pa: &[f32],
    n_genomes: u32,
    n_genes: u32,
) -> Result<f32, String> {
    let op = PairwiseJaccardGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let n_pairs = (n_genomes as usize) * (n_genomes as usize - 1) / 2;

    let pa_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_pa"),
        contents: bytemuck::cast_slice(pa),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let distances_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_distances_out"),
        size: (n_pairs as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&pa_buf, &distances_buf, n_genomes, n_genes);

    let distances = gpu.read_buffer_f32(&distances_buf, n_pairs)?;
    let mean = distances.iter().sum::<f32>() / distances.len() as f32;
    Ok(mean)
}

// ── Validation functions ───────────────────────────────────────────

fn validate_identical_genomes(h: &mut ValidationHarness, gpu: &Gpu) {
    // All genomes identical → all distances = 0.0, mean = 0.0
    let n_genomes = 4_usize;
    let n_genes = 8_usize;
    let mut pa = vec![0.0f32; n_genes * n_genomes];
    for g in 0..n_genes {
        for _genome in 0..n_genomes {
            pa[g * n_genomes + _genome] = (g % 2) as f32; // same column for all
        }
    }

    match gpu_chained_mean_jaccard(gpu, &pa, n_genomes as u32, n_genes as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("identical: GPU={gpu_mean:.6} vs expected=0.0"),
                f64::from(gpu_mean),
                0.0,
                tolerances::GPU_JACCARD_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("identical: dispatch failed — {e}"), false);
        }
    }
}

fn validate_disjoint_genomes(h: &mut ValidationHarness, gpu: &Gpu) {
    // Disjoint genomes: each gene present in exactly one genome
    // All pairs have intersection=0, union>0 → distance = 1.0, mean = 1.0
    let n_genomes = 4_usize;
    let n_genes = 8_usize; // 2 genes per genome, disjoint
    let mut pa = vec![0.0f32; n_genes * n_genomes];
    for g in 0..n_genes {
        let genome = g % n_genomes;
        pa[g * n_genomes + genome] = 1.0f32;
    }

    match gpu_chained_mean_jaccard(gpu, &pa, n_genomes as u32, n_genes as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("disjoint: GPU={gpu_mean:.6} vs expected=1.0"),
                f64::from(gpu_mean),
                1.0,
                tolerances::GPU_JACCARD_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("disjoint: dispatch failed — {e}"), false);
        }
    }
}

fn validate_random_pa_small(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_genomes = 8_usize;
    let n_genes = 50_usize;
    let mut rng = Rng::new(42);
    let pa: Vec<f32> = (0..n_genes * n_genomes)
        .map(|_| if rng.uniform() < 0.5 { 0.0 } else { 1.0 })
        .collect();

    let cpu_mean = cpu_mean_jaccard(&pa, n_genes, n_genomes);

    match gpu_chained_mean_jaccard(gpu, &pa, n_genomes as u32, n_genes as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("random 8×50: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_JACCARD_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("random 8×50: dispatch failed — {e}"), false);
        }
    }
}

fn validate_random_pa_larger(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_genomes = 16_usize;
    let n_genes = 100_usize;
    let mut rng = Rng::new(777);
    let pa: Vec<f32> = (0..n_genes * n_genomes)
        .map(|_| if rng.uniform() < 0.5 { 0.0 } else { 1.0 })
        .collect();

    let cpu_mean = cpu_mean_jaccard(&pa, n_genes, n_genomes);

    match gpu_chained_mean_jaccard(gpu, &pa, n_genomes as u32, n_genes as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("random 16×100: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_JACCARD_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("random 16×100: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_genomes = 8_usize;
    let n_genes = 50_usize;
    let mut rng = Rng::new(42);
    let pa: Vec<f32> = (0..n_genes * n_genomes)
        .map(|_| if rng.uniform() < 0.5 { 0.0 } else { 1.0 })
        .collect();

    let r1 = gpu_chained_mean_jaccard(gpu, &pa, n_genomes as u32, n_genes as u32);
    let r2 = gpu_chained_mean_jaccard(gpu, &pa, n_genomes as u32, n_genes as u32);

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
