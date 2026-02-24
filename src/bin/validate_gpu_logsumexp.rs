// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: batched log-sum-exp reduce via `metalForge` WGSL shader.
//!
//! Validates `metalForge/shaders/logsumexp_reduce.wgsl` against CPU reference.
//! The GPU shader computes numerically-stable logsumexp per row of a
//! [batch × width] matrix using the max-subtract trick.
//!
//! ## Papers validated
//!
//! - Papers 016–018: HMM forward/backward, phylogenetics log-likelihood
//!
//! ## Provenance
//!
//! CPU reference: inline `cpu_logsumexp`. WGSL: `metalForge/shaders/logsumexp_reduce.wgsl`.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = include_str!("../../metalForge/shaders/logsumexp_reduce.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    batch: u32,
    width: u32,
}

fn cpu_logsumexp(input: &[f32], batch: usize, width: usize) -> Vec<f32> {
    (0..batch)
        .map(|b| {
            let row = &input[b * width..(b + 1) * width];
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
            max_val + sum_exp.ln()
        })
        .collect()
}

fn gpu_logsumexp(gpu: &Gpu, input: &[f32], batch: u32, width: u32) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("logsumexp_reduce"),
        source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("logsumexp_bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("logsumexp_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("logsumexp_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "logsumexp_reduce",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: (batch as usize * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = Params { batch, width };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("logsumexp_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("logsumexp_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("logsumexp_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(gpu.dispatch_1d(batch, 256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f32(&output_buf, batch as usize)
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
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let mut h = ValidationHarness::new("gpu_logsumexp");

    validate_small_batch(&mut h, &gpu);
    validate_known_values(&mut h, &gpu);
    validate_large_batch(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

fn validate_small_batch(h: &mut ValidationHarness, gpu: &Gpu) {
    let batch = 4_usize;
    let width = 8_usize;

    let mut rng = Rng::new(42);
    let input: Vec<f32> = (0..batch * width).map(|_| rng.uniform() as f32).collect();

    let cpu_out = cpu_logsumexp(&input, batch, width);

    match gpu_logsumexp(gpu, &input, batch as u32, width as u32) {
        Ok(gpu_out) => {
            let max_diff: f64 = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "small batch 4×8: max GPU-CPU diff ({max_diff:.2e}) within logsumexp f32 tolerance"
                ),
                max_diff,
                tolerances::GPU_LOGSUMEXP_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("small batch: dispatch failed — {e}"), false);
        }
    }
}

fn validate_known_values(h: &mut ValidationHarness, gpu: &Gpu) {
    let batch = 2_usize;
    let width = 3_usize;
    let input: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0];

    let cpu_out = cpu_logsumexp(&input, batch, width);

    match gpu_logsumexp(gpu, &input, batch as u32, width as u32) {
        Ok(gpu_out) => {
            let max_diff: f64 = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "known values [[0,0,0],[1,2,3]]: max GPU-CPU diff ({max_diff:.2e}), row0≈ln(3), row1≈3.4076"
                ),
                max_diff,
                tolerances::GPU_LOGSUMEXP_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("known values: dispatch failed — {e}"), false);
        }
    }
}

fn validate_large_batch(h: &mut ValidationHarness, gpu: &Gpu) {
    let batch = 64_usize;
    let width = 128_usize;

    let mut rng = Rng::new(77);
    let input: Vec<f32> = (0..batch * width).map(|_| rng.uniform() as f32).collect();

    let cpu_out = cpu_logsumexp(&input, batch, width);

    match gpu_logsumexp(gpu, &input, batch as u32, width as u32) {
        Ok(gpu_out) => {
            let all_finite = gpu_out.iter().all(|&x| x.is_finite());
            h.check_bool("large batch 64×128: all outputs finite", all_finite);

            let max_diff: f64 = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "large batch 64×128: max GPU-CPU diff ({max_diff:.2e}) within logsumexp f32 tolerance"
                ),
                max_diff,
                tolerances::GPU_LOGSUMEXP_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("large batch: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let batch = 4_usize;
    let width = 8_usize;

    let mut rng = Rng::new(42);
    let input: Vec<f32> = (0..batch * width).map(|_| rng.uniform() as f32).collect();

    let run1 = gpu_logsumexp(gpu, &input, batch as u32, width as u32);
    let run2 = gpu_logsumexp(gpu, &input, batch as u32, width as u32);

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(&a, &b)| (a - b).abs() < f32::EPSILON);
            h.check_bool(
                "determinism: two logsumexp_reduce runs identical",
                identical,
            );
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
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
