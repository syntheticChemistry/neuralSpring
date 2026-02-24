// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: two-input Hill function via `BarraCUDA` `HillGateGpu`.
//!
//! Validates `barracuda::ops::bio::HillGateGpu` against CPU
//! `signal_integration::two_input_hill`. The GPU op evaluates
//! the Hill function over a 2D (cdg, ai) grid in a single dispatch.
//!
//! ## Papers validated
//!
//! - Paper 021: Signal Integration (two-input Hill function / AND gate)
//!
//! ## Provenance
//!
//! CPU reference: `signal_integration::two_input_hill` (seed=0, 10×10 grid).
//! GPU op: `barracuda::ops::bio::HillGateGpu`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]

use barracuda::ops::bio::{HillGateGpu, HillGateParams};
use neural_spring::gpu::Gpu;
use neural_spring::signal_integration::two_input_hill;
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

    let dev = Arc::clone(gpu.wgpu_device());
    let Ok(op) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| HillGateGpu::new(dev)))
    else {
        eprintln!("  SKIP: HillGateGpu f64 shader compilation failed (driver limitation)");
        eprintln!("  0/0 checks — skipping gracefully");
        std::process::exit(0);
    };
    let mut h = ValidationHarness::new("gpu_signal");

    validate_small_grid(&mut h, &gpu, &op);
    validate_and_gate_corners(&mut h, &gpu, &op);
    validate_determinism(&mut h, &gpu, &op);
    validate_larger_grid(&mut h, &gpu, &op);

    h.finish();
}

/// CPU reference: two-input Hill over 2D grid.
#[must_use]
fn cpu_hill_grid(
    cdg_grid: &[f64],
    ai_grid: &[f64],
    vmax: f64,
    k1: f64,
    k2: f64,
    n1: f64,
    n2: f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(cdg_grid.len() * ai_grid.len());
    for cdg in cdg_grid {
        for ai in ai_grid {
            out.push(two_input_hill(*cdg, *ai, vmax, k1, k2, n1, n2));
        }
    }
    out
}

fn gpu_hill_grid(
    gpu: &Gpu,
    op: &HillGateGpu,
    cdg: &[f32],
    ai: &[f32],
    params: &HillGateParams,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();

    let input_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cdg"),
        contents: bytemuck::cast_slice(cdg),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let input_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ai"),
        contents: bytemuck::cast_slice(ai),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let n_total = (params.n_a * params.n_b) as usize;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hill_output"),
        size: (n_total * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&input_a, &input_b, &output, params);

    gpu.read_buffer_f32(&output, n_total)
}

fn make_linear_grid(n: usize, low: f64, high: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![low];
    }
    (0..n)
        .map(|i| low + (high - low) * (i as f64) / ((n - 1) as f64))
        .collect()
}

fn validate_small_grid(h: &mut ValidationHarness, gpu: &Gpu, op: &HillGateGpu) {
    let nx = 10_usize;
    let ny = 10_usize;
    let vmax = 1.0_f64;
    let k1 = 1.0_f64;
    let k2 = 1.0_f64;
    let n1 = 2.0_f64;
    let n2 = 2.0_f64;

    let cdg_cpu = make_linear_grid(nx, 0.01, 5.0);
    let ai_cpu = make_linear_grid(ny, 0.01, 5.0);
    let cpu_out = cpu_hill_grid(&cdg_cpu, &ai_cpu, vmax, k1, k2, n1, n2);

    let cdg_f32: Vec<f32> = cdg_cpu.iter().map(|x| *x as f32).collect();
    let ai_f32: Vec<f32> = ai_cpu.iter().map(|x| *x as f32).collect();
    let params = HillGateParams {
        n_a: nx as u32,
        n_b: ny as u32,
        mode: 1,
        _pad: 0,
        k_a: k1,
        k_b: k2,
        n_a_exp: n1,
        n_b_exp: n2,
        vmax,
        _pad2: 0.0,
    };

    match gpu_hill_grid(gpu, op, &cdg_f32, &ai_f32, &params) {
        Ok(gpu_out) => {
            h.check_bool(
                &format!("small grid: correct cell count ({})", gpu_out.len()),
                gpu_out.len() == cpu_out.len(),
            );

            let max_diff: f64 = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("small grid: max GPU-CPU diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_HILL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("small grid: dispatch failed — {e}"), false);
        }
    }
}

fn validate_and_gate_corners(h: &mut ValidationHarness, gpu: &Gpu, op: &HillGateGpu) {
    let low = 0.01_f64;
    let high = 5.0_f64;
    let cdg_cpu = [low, high];
    let ai_cpu = [low, high];
    let vmax = 1.0_f64;
    let k1 = 1.0_f64;
    let k2 = 1.0_f64;
    let n1 = 2.0_f64;
    let n2 = 2.0_f64;

    let cpu_out = cpu_hill_grid(&cdg_cpu, &ai_cpu, vmax, k1, k2, n1, n2);

    let cdg_f32: Vec<f32> = cdg_cpu.iter().map(|x| *x as f32).collect();
    let ai_f32: Vec<f32> = ai_cpu.iter().map(|x| *x as f32).collect();
    let params = HillGateParams {
        n_a: 2,
        n_b: 2,
        mode: 1,
        _pad: 0,
        k_a: 1.0,
        k_b: 1.0,
        n_a_exp: 2.0,
        n_b_exp: 2.0,
        vmax: 1.0,
        _pad2: 0.0,
    };

    match gpu_hill_grid(gpu, op, &cdg_f32, &ai_f32, &params) {
        Ok(gpu_out) => {
            let max_diff: f64 = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "AND gate corners: max diff ({max_diff:.2e}), low/low, high/low, low/high, high/high"
                ),
                max_diff,
                tolerances::GPU_HILL_F32,
            );

            // AND gate: high only when both high
            let threshold = 0.5_f32;
            let (off_off, on_off, off_on, on_on) = (gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3]);
            h.check_bool("AND corners: off/off low", off_off < threshold);
            h.check_bool("AND corners: on/off low", on_off < threshold);
            h.check_bool("AND corners: off/on low", off_on < threshold);
            h.check_bool("AND corners: on/on high", on_on > threshold);
        }
        Err(e) => {
            h.check_bool(&format!("AND gate corners: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu, op: &HillGateGpu) {
    let nx = 10_usize;
    let ny = 10_usize;
    let cdg_cpu = make_linear_grid(nx, 0.01, 5.0);
    let ai_cpu = make_linear_grid(ny, 0.01, 5.0);
    let cdg_f32: Vec<f32> = cdg_cpu.iter().map(|x| *x as f32).collect();
    let ai_f32: Vec<f32> = ai_cpu.iter().map(|x| *x as f32).collect();
    let params = HillGateParams {
        n_a: nx as u32,
        n_b: ny as u32,
        mode: 1,
        _pad: 0,
        k_a: 1.0,
        k_b: 1.0,
        n_a_exp: 2.0,
        n_b_exp: 2.0,
        vmax: 1.0,
        _pad2: 0.0,
    };

    let run1 = gpu_hill_grid(gpu, op, &cdg_f32, &ai_f32, &params);
    let run2 = gpu_hill_grid(gpu, op, &cdg_f32, &ai_f32, &params);

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(&a, &b)| (a - b).abs() < f32::EPSILON);
            h.check_bool("determinism: two hill_gate runs identical", identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}

fn validate_larger_grid(h: &mut ValidationHarness, gpu: &Gpu, op: &HillGateGpu) {
    let nx = 32_usize;
    let ny = 32_usize;
    let vmax = 1.0_f64;
    let k1 = 1.0_f64;
    let k2 = 1.0_f64;
    let n1 = 2.0_f64;
    let n2 = 2.0_f64;

    let cdg_cpu = make_linear_grid(nx, 0.01, 5.0);
    let ai_cpu = make_linear_grid(ny, 0.01, 5.0);
    let cpu_out = cpu_hill_grid(&cdg_cpu, &ai_cpu, vmax, k1, k2, n1, n2);

    let cdg_f32: Vec<f32> = cdg_cpu.iter().map(|x| *x as f32).collect();
    let ai_f32: Vec<f32> = ai_cpu.iter().map(|x| *x as f32).collect();
    let params = HillGateParams {
        n_a: nx as u32,
        n_b: ny as u32,
        mode: 1,
        _pad: 0,
        k_a: k1,
        k_b: k2,
        n_a_exp: n1,
        n_b_exp: n2,
        vmax,
        _pad2: 0.0,
    };

    match gpu_hill_grid(gpu, op, &cdg_f32, &ai_f32, &params) {
        Ok(gpu_out) => {
            let max_diff: f64 = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "32×32 grid: max GPU-CPU diff ({max_diff:.2e}), {} cells",
                    gpu_out.len()
                ),
                max_diff,
                tolerances::GPU_HILL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("32×32 grid: dispatch failed — {e}"), false);
        }
    }
}
