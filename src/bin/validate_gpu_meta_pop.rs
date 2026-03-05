// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: per-locus allele frequency variance via `BarraCUDA` upstream API.
//!
//! Validates `barracuda::ops::bio::LocusVarianceGpu` against CPU
//! variance computation from `meta_population.rs`.  The typed op
//! computes per-locus variance in a single dispatch (one thread per locus).
//!
//! Evolution path:
//! ```text
//! Python (numpy.var) → Rust CPU (loop) → BarraCUDA CPU (stats::variance)
//!   → GPU WGSL shader (locus_variance.wgsl) → `BarraCUDA` absorption
//!   → barracuda::ops::bio::LocusVarianceGpu
//! ```
//!
//! ## Papers validated
//!
//! - Paper 025: Meta-Population Differentiation (Anderson, 2024)
//!
//! ## Provenance
//!
//! CPU reference: `meta_population::inter_population_af_variance` (per-locus variance).
//! Upstream API: `barracuda::ops::bio::LocusVarianceGpu`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "validation binary"
)]

use barracuda::ops::bio::LocusVarianceGpu;
use neural_spring::gpu::Gpu;
use neural_spring::meta_population::{allele_frequencies, generate_population};
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

    let op = LocusVarianceGpu::new(Arc::clone(gpu.wgpu_device()));
    let mut h = ValidationHarness::new("gpu_meta_pop");

    validate_small_variance(&mut h, &gpu, &op);
    validate_larger_variance(&mut h, &gpu, &op);
    validate_uniform_pops(&mut h, &gpu, &op);
    validate_determinism(&mut h, &gpu, &op);

    h.finish();
}

fn gpu_locus_variance(
    op: &LocusVarianceGpu,
    gpu: &Gpu,
    allele_freqs: &[f64],
    n_pops: u32,
    n_loci: u32,
) -> Result<Vec<f64>, String> {
    let device = gpu.device();

    let allele_freqs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("allele_freqs"),
        contents: bytemuck::cast_slice(allele_freqs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("per_locus_var"),
        size: u64::from(n_loci) * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&allele_freqs_buf, &output_buf, n_pops, n_loci);

    gpu.read_buffer_f64(&output_buf, n_loci as usize)
}

fn cpu_locus_variance(all_freqs: &[Vec<f64>], n_loci: usize) -> Vec<f64> {
    let n_pops = all_freqs.len();
    (0..n_loci)
        .map(|j| {
            let mean: f64 = all_freqs.iter().map(|af| af[j]).sum::<f64>() / n_pops as f64;
            all_freqs
                .iter()
                .map(|af| (af[j] - mean).powi(2))
                .sum::<f64>()
                / n_pops as f64
        })
        .collect()
}

fn make_test_data(seed: u64) -> (Vec<Vec<f64>>, usize, usize, usize) {
    let mut rng = Rng::new(seed);
    let n_pops = 6_usize;
    let n_loci = 100_usize;
    let n_individuals = 20_usize;
    let fst_target = 0.15;
    let temperatures = [65.0, 72.0, 78.0, 85.0, 70.0, 90.0];
    let temp_min = 65.0;
    let temp_max = 90.0;
    let n_thermal = n_loci / 5;

    let ancestral: Vec<f64> = (0..n_loci).map(|_| rng.beta(2.0, 2.0)).collect();
    let populations: Vec<Vec<f64>> = (0..n_pops)
        .map(|i| {
            generate_population(
                n_individuals,
                n_loci,
                &ancestral,
                fst_target,
                temperatures[i],
                temp_min,
                temp_max,
                n_thermal,
                &mut rng,
            )
        })
        .collect();
    (populations, n_pops, n_loci, n_individuals)
}

fn validate_small_variance(h: &mut ValidationHarness, gpu: &Gpu, op: &LocusVarianceGpu) {
    let (populations, n_pops, n_loci, n_individuals) = make_test_data(42);

    let all_freqs: Vec<Vec<f64>> = populations
        .iter()
        .map(|pop| allele_frequencies(pop, n_individuals, n_loci))
        .collect();
    let cpu_var = cpu_locus_variance(&all_freqs, n_loci);

    // Flatten to row-major f64: af[pop * n_loci + locus]
    let af_f64: Vec<f64> = all_freqs.iter().flat_map(|af| af.iter().copied()).collect();

    match gpu_locus_variance(op, gpu, &af_f64, n_pops as u32, n_loci as u32) {
        Ok(gpu_var) => {
            h.check_bool(
                &format!("6×100: correct count ({})", gpu_var.len()),
                gpu_var.len() == n_loci,
            );

            let max_diff: f64 = gpu_var
                .iter()
                .zip(cpu_var.iter())
                .map(|(&g, &c)| (g - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("6×100: max GPU-CPU var diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );

            let gpu_mean: f64 = gpu_var.iter().copied().sum::<f64>() / gpu_var.len() as f64;
            h.check_lower(
                &format!("6×100: mean locus variance > 0 ({gpu_mean:.6})"),
                gpu_mean,
                0.0,
            );
        }
        Err(e) => {
            h.check_bool(&format!("6×100: dispatch failed — {e}"), false);
        }
    }
}

fn validate_larger_variance(h: &mut ValidationHarness, gpu: &Gpu, op: &LocusVarianceGpu) {
    let (populations, n_pops, n_loci, n_individuals) = make_test_data(77);

    let all_freqs: Vec<Vec<f64>> = populations
        .iter()
        .map(|pop| allele_frequencies(pop, n_individuals, n_loci))
        .collect();
    let cpu_var = cpu_locus_variance(&all_freqs, n_loci);

    let af_f64: Vec<f64> = all_freqs.iter().flat_map(|af| af.iter().copied()).collect();

    match gpu_locus_variance(op, gpu, &af_f64, n_pops as u32, n_loci as u32) {
        Ok(gpu_var) => {
            let max_diff: f64 = gpu_var
                .iter()
                .zip(cpu_var.iter())
                .map(|(&g, &c)| (g - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("seed=77: max GPU-CPU var diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );

            let cpu_mean: f64 = cpu_var.iter().sum::<f64>() / cpu_var.len() as f64;
            let gpu_mean: f64 = gpu_var.iter().copied().sum::<f64>() / gpu_var.len() as f64;
            h.check_abs(
                &format!("seed=77: mean var GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("seed=77: dispatch failed — {e}"), false);
        }
    }
}

fn validate_uniform_pops(h: &mut ValidationHarness, gpu: &Gpu, op: &LocusVarianceGpu) {
    let n_pops = 4_u32;
    let n_loci = 16_u32;
    let af_f64: Vec<f64> = vec![0.5; (n_pops * n_loci) as usize];

    match gpu_locus_variance(op, gpu, &af_f64, n_pops, n_loci) {
        Ok(gpu_var) => {
            let all_zero = gpu_var
                .iter()
                .all(|&v| v.abs() < tolerances::GPU_LOCUS_VARIANCE_F32);
            h.check_bool(
                &format!(
                    "uniform AF=0.5: all variance≈0 (max={:.2e})",
                    gpu_var.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
                ),
                all_zero,
            );
        }
        Err(e) => {
            h.check_bool(&format!("uniform: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu, op: &LocusVarianceGpu) {
    let (populations, n_pops, n_loci, n_individuals) = make_test_data(42);

    let all_freqs: Vec<Vec<f64>> = populations
        .iter()
        .map(|pop| allele_frequencies(pop, n_individuals, n_loci))
        .collect();
    let af_f64: Vec<f64> = all_freqs.iter().flat_map(|af| af.iter().copied()).collect();

    let run1 = gpu_locus_variance(op, gpu, &af_f64, n_pops as u32, n_loci as u32);
    let run2 = gpu_locus_variance(op, gpu, &af_f64, n_pops as u32, n_loci as u32);

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(&a, &b)| (a - b).abs() < f64::EPSILON);
            h.check_bool("determinism: two variance runs identical", identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}
