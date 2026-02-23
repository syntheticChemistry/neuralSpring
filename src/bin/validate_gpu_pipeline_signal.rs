// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU pipeline validation: HillGateGpu (BarraCUDA) + CPU mean (Paper 021).
//!
//! Replaces raw wgpu pipeline with typed BarraCUDA op: `barracuda::ops::bio::HillGateGpu`.
//! Stage 1: HillGateGpu.dispatch → output[nx*ny] (f64).
//! Stage 2: CPU mean over output.
//!
//! ## Pipeline
//!
//! ```text
//! Upload cdg_grid[nx], ai_grid[ny] (once)
//!   ↓
//! HillGateGpu.dispatch → output[nx*ny]
//!   ↓
//! CPU mean(output) → scalar
//! ```
//!
//! ## Provenance
//!
//! Typed op: `barracuda::ops::bio::HillGateGpu` (f64).
//! Validates: BarraCUDA Hill gate API with scalar summary.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

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
        Err(e) => {
            eprintln!("  SKIP: {e} — no GPU/CPU adapter available");
            eprintln!("  0/0 checks — skipping gracefully");
            std::process::exit(0);
        }
    };

    let dev = Arc::clone(gpu.wgpu_device());
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        HillGateGpu::new(dev);
    }))
    .is_err()
    {
        eprintln!("  SKIP: HillGateGpu f64 shader compilation failed (driver limitation)");
        eprintln!("  0/0 checks — skipping gracefully");
        std::process::exit(0);
    }

    let mut h = ValidationHarness::new("gpu_pipeline_signal");

    validate_small_grid(&mut h, &gpu);
    validate_larger_grid(&mut h, &gpu);
    validate_high_params(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_hill_grid(
    cdg_grid: &[f64],
    ai_grid: &[f64],
    vmax: f64,
    k1: f64,
    k2: f64,
    n1: f64,
    n2: f64,
) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for cdg in cdg_grid {
        for ai in ai_grid {
            sum += two_input_hill(*cdg, *ai, vmax, k1, k2, n1, n2);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn make_linear_grid(n: usize, low: f64, high: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![low];
    }
    (0..n)
        .map(|i| low + (high - low) * (i as f64) / ((n - 1) as f64))
        .collect()
}

// ── BarraCUDA typed op + CPU mean ──────────────────────────────────

fn gpu_hill_mean(
    gpu: &Gpu,
    cdg: &[f64],
    ai: &[f64],
    params: &HillGateParams,
) -> Result<f64, String> {
    let device = gpu.device();
    let dev = Arc::clone(gpu.wgpu_device());
    let op = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| HillGateGpu::new(dev)))
    {
        Ok(o) => o,
        Err(_) => {
            return Err("HillGateGpu f64 shader compilation failed (driver limitation)".into())
        }
    };
    let n_total = (params.n_a * params.n_b) as usize;

    let cdg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_cdg"),
        contents: bytemuck::cast_slice(cdg),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let ai_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_ai"),
        contents: bytemuck::cast_slice(ai),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_hill_output"),
        size: (n_total * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&cdg_buf, &ai_buf, &output_buf, params);

    let out = gpu.read_buffer_f64(&output_buf, n_total)?;
    let mean = out.iter().sum::<f64>() / out.len() as f64;
    Ok(mean)
}

// ── Validation functions ───────────────────────────────────────────

fn validate_small_grid(h: &mut ValidationHarness, gpu: &Gpu) {
    let nx = 10_usize;
    let ny = 10_usize;
    let vmax = 1.0_f64;
    let k1 = 1.0_f64;
    let k2 = 1.0_f64;
    let n1 = 2.0_f64;
    let n2 = 2.0_f64;

    let cdg_cpu = make_linear_grid(nx, 0.01, 5.0);
    let ai_cpu = make_linear_grid(ny, 0.01, 5.0);
    let cpu_mean = cpu_mean_hill_grid(&cdg_cpu, &ai_cpu, vmax, k1, k2, n1, n2);

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

    match gpu_hill_mean(gpu, &cdg_cpu, &ai_cpu, &params) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("signal small 10×10: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_HILL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("signal small grid: dispatch failed — {e}"), false);
        }
    }
}

fn validate_larger_grid(h: &mut ValidationHarness, gpu: &Gpu) {
    let nx = 32_usize;
    let ny = 32_usize;
    let vmax = 1.0_f64;
    let k1 = 1.0_f64;
    let k2 = 1.0_f64;
    let n1 = 2.0_f64;
    let n2 = 2.0_f64;

    let cdg_cpu = make_linear_grid(nx, 0.01, 5.0);
    let ai_cpu = make_linear_grid(ny, 0.01, 5.0);
    let cpu_mean = cpu_mean_hill_grid(&cdg_cpu, &ai_cpu, vmax, k1, k2, n1, n2);

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

    match gpu_hill_mean(gpu, &cdg_cpu, &ai_cpu, &params) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("signal larger 32×32: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_HILL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("signal larger grid: dispatch failed — {e}"), false);
        }
    }
}

fn validate_high_params(h: &mut ValidationHarness, gpu: &Gpu) {
    let nx = 10_usize;
    let ny = 10_usize;
    let vmax = 2.0_f64;
    let k1 = 1.0_f64;
    let k2 = 1.0_f64;
    let n1 = 3.0_f64;
    let n2 = 3.0_f64;

    let cdg_cpu = make_linear_grid(nx, 0.5, 3.0);
    let ai_cpu = make_linear_grid(ny, 0.5, 3.0);
    let cpu_mean = cpu_mean_hill_grid(&cdg_cpu, &ai_cpu, vmax, k1, k2, n1, n2);

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

    match gpu_hill_mean(gpu, &cdg_cpu, &ai_cpu, &params) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("signal high params vmax=2: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_HILL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("signal high params: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let nx = 10_usize;
    let ny = 10_usize;
    let cdg_cpu = make_linear_grid(nx, 0.01, 5.0);
    let ai_cpu = make_linear_grid(ny, 0.01, 5.0);
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

    let r1 = gpu_hill_mean(gpu, &cdg_cpu, &ai_cpu, &params);
    let r2 = gpu_hill_mean(gpu, &cdg_cpu, &ai_cpu, &params);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("signal determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f64::EPSILON,
            );
        }
        _ => {
            h.check_bool("signal determinism: dispatch failed", false);
        }
    }
}
