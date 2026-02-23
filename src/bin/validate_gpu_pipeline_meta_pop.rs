// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline validation: `locus_variance` → mean (Paper 025).
//!
//! Uses `BarraCUDA` typed op `LocusVarianceGpu` (f64) with CPU mean reduction.
//! Replaces raw wgpu chain (`locus_variance` + `mean_reduce`) for validation.
//!
//! ## Pipeline
//!
//! ```text
//! Upload allele frequencies (once)
//!   ↓
//! LocusVarianceGpu.dispatch() → per_locus_var[n_loci] (f64)
//!   ↓
//! CPU mean(per_locus_var)
//! ```
//!
//! ## Provenance
//!
//! GPU op: `barracuda::ops::bio::LocusVarianceGpu` (f64 pipeline)
//! Validates: meta-population mean per-locus variance (Campbell, Anderson et al., 2017).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::needless_range_loop
)]

use barracuda::ops::bio::LocusVarianceGpu;
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

    let mut h = ValidationHarness::new("gpu_pipeline_meta_pop");

    validate_meta_pop_small(&mut h, &gpu);
    validate_meta_pop_larger(&mut h, &gpu);
    validate_meta_pop_uniform(&mut h, &gpu);
    validate_meta_pop_differentiated(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_locus_variance(allele_freqs: &[f64], n_pops: usize, n_loci: usize) -> f64 {
    let mut total = 0.0_f64;
    for locus in 0..n_loci {
        let mut sum = 0.0_f64;
        for pop in 0..n_pops {
            sum += allele_freqs[pop * n_loci + locus];
        }
        let mean = sum / n_pops as f64;
        let mut var_sum = 0.0_f64;
        for pop in 0..n_pops {
            let diff = allele_freqs[pop * n_loci + locus] - mean;
            var_sum = diff.mul_add(diff, var_sum);
        }
        total += var_sum / n_pops as f64;
    }
    total / n_loci as f64
}

// ── GPU via BarraCUDA typed op ──────────────────────────────────────

fn gpu_mean_locus_variance(
    gpu: &Gpu,
    allele_freqs: &[f64],
    n_pops: u32,
    n_loci: u32,
) -> Result<f64, String> {
    let op = LocusVarianceGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();

    let af_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_meta_allele_freqs"),
        contents: bytemuck::cast_slice(allele_freqs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let var_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_meta_variances"),
        size: u64::from(n_loci) * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&af_buf, &var_buf, n_pops, n_loci);

    let variances = gpu.read_buffer_f64(&var_buf, n_loci as usize)?;
    let mean = variances.iter().sum::<f64>() / variances.len() as f64;
    Ok(mean)
}

// ── Validation functions ───────────────────────────────────────────

fn validate_meta_pop_small(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 4_usize;
    let n_loci = 12_usize;
    let mut rng = Rng::new(42);
    let allele_freqs: Vec<f64> = (0..n_pops * n_loci).map(|_| rng.uniform()).collect();

    let cpu_mean = cpu_mean_locus_variance(&allele_freqs, n_pops, n_loci);

    match gpu_mean_locus_variance(gpu, &allele_freqs, n_pops as u32, n_loci as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("meta_pop small 4×12: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("meta_pop small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_meta_pop_larger(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 8_usize;
    let n_loci = 32_usize;
    let mut rng = Rng::new(777);
    let allele_freqs: Vec<f64> = (0..n_pops * n_loci).map(|_| rng.uniform()).collect();

    let cpu_mean = cpu_mean_locus_variance(&allele_freqs, n_pops, n_loci);

    match gpu_mean_locus_variance(gpu, &allele_freqs, n_pops as u32, n_loci as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("meta_pop larger 8×32: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("meta_pop larger: dispatch failed — {e}"), false);
        }
    }
}

fn validate_meta_pop_uniform(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 4_usize;
    let n_loci = 8_usize;
    let allele_freqs: Vec<f64> = vec![0.5; n_pops * n_loci];

    match gpu_mean_locus_variance(gpu, &allele_freqs, n_pops as u32, n_loci as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("meta_pop uniform: mean variance={gpu_mean:.6} vs 0"),
                gpu_mean,
                0.0,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("meta_pop uniform: dispatch failed — {e}"), false);
        }
    }
}

fn validate_meta_pop_differentiated(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 4_usize;
    let n_loci = 6_usize;
    let mut allele_freqs = vec![0.0_f64; n_pops * n_loci];
    for pop in 0..n_pops {
        for locus in 0..n_loci {
            allele_freqs[pop * n_loci + locus] = (pop as f64)
                .mul_add(0.2, locus as f64 * 0.05)
                .clamp(0.01, 0.99);
        }
    }

    let cpu_mean = cpu_mean_locus_variance(&allele_freqs, n_pops, n_loci);

    match gpu_mean_locus_variance(gpu, &allele_freqs, n_pops as u32, n_loci as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("meta_pop differentiated: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("meta_pop differentiated: dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 6_usize;
    let n_loci = 16_usize;
    let mut rng = Rng::new(99);
    let allele_freqs: Vec<f64> = (0..n_pops * n_loci).map(|_| rng.uniform()).collect();

    let r1 = gpu_mean_locus_variance(gpu, &allele_freqs, n_pops as u32, n_loci as u32);
    let r2 = gpu_mean_locus_variance(gpu, &allele_freqs, n_pops as u32, n_loci as u32);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("meta_pop determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f64::EPSILON,
            );
        }
        _ => {
            h.check_bool("meta_pop determinism: dispatch failed", false);
        }
    }
}
