// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: two-input Hill function via `metalForge` WGSL shader.
//!
//! Validates `metalForge/shaders/hill_gate.wgsl` against CPU
//! `signal_integration::two_input_hill`. The GPU shader evaluates
//! the Hill function over a 2D (cdg, ai) grid in a single dispatch.
//!
//! ## Papers validated
//!
//! - Paper 021: Signal Integration (two-input Hill function / AND gate)
//!
//! ## Provenance
//!
//! CPU reference: `signal_integration::two_input_hill` (seed=0, 10×10 grid).
//! WGSL shader: `metalForge/shaders/hill_gate.wgsl`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::similar_names
)]

use neural_spring::gpu::Gpu;
use neural_spring::signal_integration::two_input_hill;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = include_str!("../../metalForge/shaders/hill_gate.wgsl");

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

    let mut h = ValidationHarness::new("gpu_signal");

    validate_small_grid(&mut h, &gpu);
    validate_and_gate_corners(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);
    validate_larger_grid(&mut h, &gpu);

    h.finish();
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HillParams {
    nx: u32,
    ny: u32,
    vmax: f32,
    k1: f32,
    k2: f32,
    n1: f32,
    n2: f32,
    _pad: u32,
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
    cdg: &[f32],
    ai: &[f32],
    params: &HillParams,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hill_gate"),
        source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hill_gate_bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hill_gate_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hill_gate_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "hill_gate",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let cdg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cdg_grid"),
        contents: bytemuck::cast_slice(cdg),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let ai_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ai_grid"),
        contents: bytemuck::cast_slice(ai),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let n_total = (params.nx * params.ny) as usize;
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hill_output"),
        size: (n_total * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hill_params"),
        contents: bytemuck::bytes_of(params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hill_gate_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cdg_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ai_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hill_gate_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hill_gate_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(gpu.dispatch_1d(n_total as u32, 256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f32(&output_buf, n_total)
}

fn make_linear_grid(n: usize, low: f64, high: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![low];
    }
    (0..n)
        .map(|i| low + (high - low) * (i as f64) / ((n - 1) as f64))
        .collect()
}

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
    let cpu_out = cpu_hill_grid(&cdg_cpu, &ai_cpu, vmax, k1, k2, n1, n2);

    let cdg_f32: Vec<f32> = cdg_cpu.iter().map(|x| *x as f32).collect();
    let ai_f32: Vec<f32> = ai_cpu.iter().map(|x| *x as f32).collect();
    let params = HillParams {
        nx: nx as u32,
        ny: ny as u32,
        vmax: vmax as f32,
        k1: k1 as f32,
        k2: k2 as f32,
        n1: n1 as f32,
        n2: n2 as f32,
        _pad: 0,
    };

    match gpu_hill_grid(gpu, &cdg_f32, &ai_f32, &params) {
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

fn validate_and_gate_corners(h: &mut ValidationHarness, gpu: &Gpu) {
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
    let params = HillParams {
        nx: 2,
        ny: 2,
        vmax: 1.0,
        k1: 1.0,
        k2: 1.0,
        n1: 2.0,
        n2: 2.0,
        _pad: 0,
    };

    match gpu_hill_grid(gpu, &cdg_f32, &ai_f32, &params) {
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

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let nx = 10_usize;
    let ny = 10_usize;
    let cdg_cpu = make_linear_grid(nx, 0.01, 5.0);
    let ai_cpu = make_linear_grid(ny, 0.01, 5.0);
    let cdg_f32: Vec<f32> = cdg_cpu.iter().map(|x| *x as f32).collect();
    let ai_f32: Vec<f32> = ai_cpu.iter().map(|x| *x as f32).collect();
    let params = HillParams {
        nx: nx as u32,
        ny: ny as u32,
        vmax: 1.0,
        k1: 1.0,
        k2: 1.0,
        n1: 2.0,
        n2: 2.0,
        _pad: 0,
    };

    let run1 = gpu_hill_grid(gpu, &cdg_f32, &ai_f32, &params);
    let run2 = gpu_hill_grid(gpu, &cdg_f32, &ai_f32, &params);

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
    let cpu_out = cpu_hill_grid(&cdg_cpu, &ai_cpu, vmax, k1, k2, n1, n2);

    let cdg_f32: Vec<f32> = cdg_cpu.iter().map(|x| *x as f32).collect();
    let ai_f32: Vec<f32> = ai_cpu.iter().map(|x| *x as f32).collect();
    let params = HillParams {
        nx: nx as u32,
        ny: ny as u32,
        vmax: vmax as f32,
        k1: k1 as f32,
        k2: k2 as f32,
        n1: n1 as f32,
        n2: n2 as f32,
        _pad: 0,
    };

    match gpu_hill_grid(gpu, &cdg_f32, &ai_f32, &params) {
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

const fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

const fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
