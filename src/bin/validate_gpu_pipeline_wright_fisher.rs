// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU pipeline validation: `WrightFisherGpu` (`BarraCUDA`) + CPU mean (Papers 024-025).
//!
//! Replaces raw wgpu pipeline with typed `BarraCUDA` op: `barracuda::ops::bio::WrightFisherGpu`.
//! Stage 1: WrightFisherGpu.dispatch → `freq_out`[`n_pops` × `n_loci`] (f64).
//! Stage 2: CPU mean over `freq_out`.
//!
//! ## Pipeline
//!
//! ```text
//! Upload freq_in + selection + PRNG state (once)
//!   ↓
//! WrightFisherGpu.dispatch → freq_out[n_pops × n_loci]
//!   ↓
//! CPU mean(freq_out) → scalar
//! ```
//!
//! ## Papers validated
//!
//! - Paper 024: Pangenome Selection (Moulana, Anderson et al., 2020)
//! - Paper 025: Meta-Population Dynamics (Campbell, Anderson et al., 2017)
//!
//! ## Provenance
//!
//! Typed op: `barracuda::ops::bio::WrightFisherGpu` (f64).
//! Validates: end-to-end GPU-resident stochastic population genetics.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

use barracuda::ops::bio::WrightFisherGpu;
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const fn splitmix32(state: &mut u32) -> u32 {
    *state = state.wrapping_add(0x9e37_79b9);
    let mut z = *state;
    z = (z ^ (z >> 15)).wrapping_mul(0x85eb_ca6b);
    z = (z ^ (z >> 13)).wrapping_mul(0xc2b2_ae35);
    z ^ (z >> 16)
}

fn seed_prng(n_threads: usize, base_seed: u32) -> Vec<u32> {
    let mut result = Vec::with_capacity(n_threads * 4);
    for t in 0..n_threads {
        let mut sm = base_seed.wrapping_add(t as u32 * 1_000_003);
        for _ in 0..4 {
            result.push(splitmix32(&mut sm));
        }
    }
    result
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
        Err(e) => {
            eprintln!("  SKIP: {e} — no GPU/CPU adapter available");
            eprintln!("  0/0 checks — skipping gracefully");
            std::process::exit(0);
        }
    };

    let mut h = ValidationHarness::new("gpu_pipeline_wright_fisher");

    validate_neutral_mean(&mut h, &gpu);
    validate_selection_shifts_mean(&mut h, &gpu);
    validate_boundary_frequencies(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

/// `WrightFisherGpu` dispatch + CPU mean.
fn gpu_wf_mean(
    gpu: &Gpu,
    freq_in: &[f64],
    selection: &[f64],
    prng_state: &[u32],
    n_pops: u32,
    n_loci: u32,
    two_n: u32,
) -> Result<f64, String> {
    let device = gpu.device();
    let op = WrightFisherGpu::new(Arc::clone(gpu.wgpu_device()));
    let n_total = (n_pops * n_loci) as usize;

    let freq_in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pipe_freq_in"),
        contents: bytemuck::cast_slice(freq_in),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let selection_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pipe_selection"),
        contents: bytemuck::cast_slice(selection),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let freq_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pipe_freq_out"),
        size: (n_total as u64) * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let prng_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pipe_prng"),
        contents: bytemuck::cast_slice(prng_state),
        usage: wgpu::BufferUsages::STORAGE,
    });

    op.dispatch(
        &freq_in_buf,
        &selection_buf,
        &freq_out_buf,
        &prng_buf,
        n_pops,
        n_loci,
        two_n,
    );

    let freq_out = gpu.read_buffer_f64(&freq_out_buf, n_total)?;
    let mean = freq_out.iter().sum::<f64>() / freq_out.len() as f64;
    Ok(mean)
}

/// Neutral drift (s=0): mean frequency ≈ 0.5 ± stochastic noise.
fn validate_neutral_mean(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 1_u32;
    let n_loci = 500_u32;
    let two_n = 100_u32;
    let n_total = (n_pops * n_loci) as usize;

    let freq_in: Vec<f64> = vec![0.5; n_total];
    let selection: Vec<f64> = vec![0.0; n_loci as usize];
    let prng_state = seed_prng(n_total, 42);

    match gpu_wf_mean(
        gpu,
        &freq_in,
        &selection,
        &prng_state,
        n_pops,
        n_loci,
        two_n,
    ) {
        Ok(mean) => {
            let diff = (mean - 0.5).abs();
            h.check_upper(
                &format!("neutral pipeline: |mean - 0.5| = {diff:.4} within QS_VARIANCE_MAX"),
                diff,
                tolerances::QS_VARIANCE_MAX,
            );
        }
        Err(e) => {
            h.check_bool(&format!("neutral pipeline: dispatch failed — {e}"), false);
        }
    }
}

/// Positive selection (s=0.1): pipeline mean should exceed neutral expectation.
fn validate_selection_shifts_mean(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 1_u32;
    let n_loci = 500_u32;
    let two_n = 200_u32;
    let n_total = (n_pops * n_loci) as usize;

    let freq_in: Vec<f64> = vec![0.5; n_total];
    let selection: Vec<f64> = vec![0.1; n_loci as usize];
    let prng_state = seed_prng(n_total, 123);

    match gpu_wf_mean(
        gpu,
        &freq_in,
        &selection,
        &prng_state,
        n_pops,
        n_loci,
        two_n,
    ) {
        Ok(mean) => {
            h.check_bool(
                &format!("selection pipeline: mean={mean:.4} > 0.5 after positive selection"),
                mean > 0.5,
            );
        }
        Err(e) => {
            h.check_bool(&format!("selection pipeline: dispatch failed — {e}"), false);
        }
    }
}

/// Boundary: `freq_in` = 0 or 1 stays fixed regardless of selection.
fn validate_boundary_frequencies(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 1_u32;
    let n_loci = 100_u32;
    let two_n = 50_u32;
    let n_total = (n_pops * n_loci) as usize;

    // Half at fixation (1.0), half at loss (0.0)
    let freq_in: Vec<f64> = (0..n_total)
        .map(|i| if i < n_total / 2 { 0.0 } else { 1.0 })
        .collect();
    let selection: Vec<f64> = vec![0.05; n_loci as usize];
    let prng_state = seed_prng(n_total, 777);

    match gpu_wf_mean(
        gpu,
        &freq_in,
        &selection,
        &prng_state,
        n_pops,
        n_loci,
        two_n,
    ) {
        Ok(mean) => {
            let diff = (mean - 0.5).abs();
            h.check_upper(
                &format!("boundary pipeline: |mean - 0.5| = {diff:.4} (fixed alleles)"),
                diff,
                tolerances::QS_VARIANCE_MAX,
            );
        }
        Err(e) => {
            h.check_bool(&format!("boundary pipeline: dispatch failed — {e}"), false);
        }
    }
}

/// Same PRNG seed → identical pipeline scalar.
fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 1_u32;
    let n_loci = 200_u32;
    let two_n = 50_u32;
    let n_total = (n_pops * n_loci) as usize;

    let freq_in: Vec<f64> = vec![0.5; n_total];
    let selection: Vec<f64> = vec![0.02; n_loci as usize];
    let prng1 = seed_prng(n_total, 9999);
    let prng2 = seed_prng(n_total, 9999);

    let r1 = gpu_wf_mean(gpu, &freq_in, &selection, &prng1, n_pops, n_loci, two_n);
    let r2 = gpu_wf_mean(gpu, &freq_in, &selection, &prng2, n_pops, n_loci, two_n);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("pipeline determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f64::EPSILON,
            );
        }
        _ => {
            h.check_bool("pipeline determinism: dispatch failed", false);
        }
    }
}
