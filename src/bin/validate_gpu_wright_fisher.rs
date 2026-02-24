// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: Wright-Fisher drift + selection via `BarraCUDA` `WrightFisherGpu` API.
//!
//! Validates `barracuda::ops::bio::WrightFisherGpu` against statistical expectations.
//! Wright-Fisher is stochastic; we use statistical tests rather than exact comparison.
//!
//! ## Papers validated
//!
//! - Paper 024: Pangenome Selection
//! - Paper 025: Meta-Population Dynamics
//!
//! ## Provenance
//!
//! Upstream: `barracuda::ops::bio::WrightFisherGpu` (f64 pipeline)
//! PRNG: xoshiro128** seeded via `SplitMix32`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use barracuda::ops::bio::WrightFisherGpu;
use neural_spring::gpu::Gpu;
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

    let mut h = ValidationHarness::new("gpu_wright_fisher");

    validate_neutral_drift(&mut h, &gpu);
    validate_selection_bias(&mut h, &gpu);
    validate_fixation_boundaries(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

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

fn gpu_wright_fisher(
    gpu: &Gpu,
    freq_in: &[f64],
    selection: &[f64],
    prng_state: &[u32],
    n_pops: u32,
    n_loci: u32,
    two_n: u32,
) -> Result<Vec<f64>, String> {
    let device = gpu.device();
    let op = WrightFisherGpu::new(Arc::clone(gpu.wgpu_device()));

    let freq_in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("freq_in"),
        contents: bytemuck::cast_slice(freq_in),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let selection_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("selection"),
        contents: bytemuck::cast_slice(selection),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let n_total = (n_pops * n_loci) as usize;
    let freq_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("freq_out"),
        size: (n_total * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let prng_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("prng_state"),
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

    gpu.read_buffer_f64(&freq_out_buf, n_total)
}

fn validate_neutral_drift(h: &mut ValidationHarness, gpu: &Gpu) {
    // s=0 (neutral), p=0.5, 2N=100, n_pops=1, n_loci=1000
    let n_pops = 1_u32;
    let n_loci = 1000_u32;
    let two_n = 100_u32;
    let n_total = (n_pops * n_loci) as usize;

    let freq_in: Vec<f64> = vec![0.5; n_total];
    let selection: Vec<f64> = vec![0.0; n_loci as usize];
    let prng_state = seed_prng(n_total, 42);

    match gpu_wright_fisher(
        gpu,
        &freq_in,
        &selection,
        &prng_state,
        n_pops,
        n_loci,
        two_n,
    ) {
        Ok(freq_out) => {
            let mean: f64 = freq_out.iter().sum::<f64>() / n_total as f64;
            let diff = (mean - 0.5).abs();
            h.check_upper(
                "neutral drift: |mean - 0.5| within QS_VARIANCE_MAX",
                diff,
                tolerances::QS_VARIANCE_MAX,
            );
        }
        Err(e) => {
            h.check_bool(&format!("neutral drift: dispatch failed — {e}"), false);
        }
    }
}

fn validate_selection_bias(h: &mut ValidationHarness, gpu: &Gpu) {
    // s=0.1 (positive selection), p=0.5, 2N=200, n_pops=1, n_loci=1000
    let n_pops = 1_u32;
    let n_loci = 1000_u32;
    let two_n = 200_u32;
    let n_total = (n_pops * n_loci) as usize;

    let freq_in: Vec<f64> = vec![0.5; n_total];
    let selection: Vec<f64> = vec![0.1; n_loci as usize];
    let prng_state = seed_prng(n_total, 123);

    match gpu_wright_fisher(
        gpu,
        &freq_in,
        &selection,
        &prng_state,
        n_pops,
        n_loci,
        two_n,
    ) {
        Ok(freq_out) => {
            let mean: f64 = freq_out.iter().sum::<f64>() / n_total as f64;
            h.check_bool(
                "selection bias: mean frequency > 0.5 after positive selection",
                mean > 0.5,
            );
        }
        Err(e) => {
            h.check_bool(&format!("selection bias: dispatch failed — {e}"), false);
        }
    }
}

fn validate_fixation_boundaries(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 2_u32;
    let n_loci = 50_u32;
    let two_n = 100_u32;
    let n_total = (n_pops * n_loci) as usize;

    let freq_in: Vec<f64> = vec![0.5; n_total];
    let selection: Vec<f64> = vec![0.0; n_loci as usize];
    let prng_state = seed_prng(n_total, 999);

    match gpu_wright_fisher(
        gpu,
        &freq_in,
        &selection,
        &prng_state,
        n_pops,
        n_loci,
        two_n,
    ) {
        Ok(freq_out) => {
            let in_bounds = freq_out
                .iter()
                .all(|&p| (0.0..=1.0).contains(&p) && p.is_finite());
            h.check_bool(
                "fixation boundaries: all frequencies in [0, 1], no NaN/Inf",
                in_bounds,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("fixation boundaries: dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 1_u32;
    let n_loci = 100_u32;
    let two_n = 50_u32;
    let n_total = (n_pops * n_loci) as usize;

    let freq_in: Vec<f64> = vec![0.5; n_total];
    let selection: Vec<f64> = vec![0.05; n_loci as usize];
    let prng_state1 = seed_prng(n_total, 7777);
    let prng_state2 = seed_prng(n_total, 7777);

    let run1 = gpu_wright_fisher(
        gpu,
        &freq_in,
        &selection,
        &prng_state1,
        n_pops,
        n_loci,
        two_n,
    );
    let run2 = gpu_wright_fisher(
        gpu,
        &freq_in,
        &selection,
        &prng_state2,
        n_pops,
        n_loci,
        two_n,
    );

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(&a, &b)| (a - b).abs() < f64::EPSILON);
            h.check_bool("determinism: same PRNG seed → identical output", identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}
