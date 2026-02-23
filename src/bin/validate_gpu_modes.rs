// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: pairwise L2 distance via `metalForge` WGSL shader.
//!
//! Validates `metalForge/shaders/pairwise_l2.wgsl` against CPU
//! L2 distance computation from `modes.rs`.  The GPU shader computes
//! all pairwise L2 distances in a single dispatch.
//!
//! ## Papers validated
//!
//! - Paper 012: MODES (novelty metric via pairwise L2 distance)
//!
//! ## Provenance
//!
//! CPU reference: `modes::l2_distance` (seed=0, 5×3 pairwise features).
//! WGSL shader: `metalForge/shaders/pairwise_l2.wgsl`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]

use barracuda::ops::bio::PairwiseL2Gpu;
use neural_spring::gpu::Gpu;
use neural_spring::modes::l2_distance;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = include_str!("../../metalForge/shaders/pairwise_l2.wgsl");

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

    let mut h = ValidationHarness::new("gpu_modes");

    validate_small_features(&mut h, &gpu);
    validate_known_distances(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);
    validate_upstream_parity(&mut h, &gpu);

    h.finish();
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    dim: u32,
}

/// CPU reference: all pairwise L2 distances (upper triangle, row-major).
fn cpu_pairwise_l2(features: &[Vec<f64>]) -> Vec<f64> {
    let n = features.len();
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            out.push(l2_distance(&features[i], &features[j]));
        }
    }
    out
}

/// Run GPU pairwise L2 distance shader.
fn gpu_pairwise_l2(gpu: &Gpu, features_flat: &[f32], n: u32, dim: u32) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pairwise_l2"),
        source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pairwise_l2_bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pairwise_l2_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pairwise_l2_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "pairwise_l2",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let features_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("features"),
        contents: bytemuck::cast_slice(features_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let n_pairs = (n * (n - 1) / 2) as usize;
    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("distances"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = Params { n, dim };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pairwise_l2_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: features_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dist_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pairwise_l2_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pairwise_l2_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(gpu.dispatch_1d(n_pairs as u32, 256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f32(&dist_buf, n_pairs)
}

fn validate_small_features(h: &mut ValidationHarness, gpu: &Gpu) {
    let features: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ];
    let n = 5_usize;
    let dim = 3_usize;

    let cpu = cpu_pairwise_l2(&features);
    let flat: Vec<f32> = features
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f32))
        .collect();

    match gpu_pairwise_l2(gpu, &flat, n as u32, dim as u32) {
        Ok(gpu_dist) => {
            h.check_bool(
                &format!("small features: correct pair count ({})", gpu_dist.len()),
                gpu_dist.len() == cpu.len(),
            );

            for (idx, (&g, &c)) in gpu_dist.iter().zip(cpu.iter()).enumerate() {
                h.check_abs(
                    &format!("small features[{idx}]: GPU ≈ CPU ({g:.6} vs {c:.6})"),
                    f64::from(g),
                    c,
                    tolerances::GPU_MODES_L2_F32,
                );
            }
        }
        Err(e) => {
            h.check_bool(&format!("small features: dispatch failed — {e}"), false);
        }
    }
}

fn validate_known_distances(h: &mut ValidationHarness, gpu: &Gpu) {
    // (0,0,0) vs (1,0,0) = 1.0
    // (0,0,0) vs (1,1,1) = sqrt(3) ≈ 1.732
    let features: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0],
    ];
    let flat: Vec<f32> = features
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f32))
        .collect();

    match gpu_pairwise_l2(gpu, &flat, 3, 3) {
        Ok(gpu_dist) => {
            let d_01 = gpu_dist[0]; // (0,0,0) vs (1,0,0)
            let d_02 = gpu_dist[1]; // (0,0,0) vs (1,1,1)
            let d_12 = gpu_dist[2]; // (1,0,0) vs (1,1,1) = sqrt(2) ≈ 1.414

            h.check_abs(
                "known: (0,0,0) vs (1,0,0) = 1.0",
                f64::from(d_01),
                1.0,
                tolerances::GPU_MODES_L2_F32,
            );
            h.check_abs(
                "known: (0,0,0) vs (1,1,1) = √3",
                f64::from(d_02),
                3_f64.sqrt(),
                tolerances::GPU_MODES_L2_F32,
            );
            h.check_abs(
                "known: (1,0,0) vs (1,1,1) = √2",
                f64::from(d_12),
                2_f64.sqrt(),
                tolerances::GPU_MODES_L2_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("known distances: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let features: Vec<Vec<f64>> = vec![
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
        vec![0.7, 0.8, 0.9],
    ];
    let flat: Vec<f32> = features
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f32))
        .collect();

    let run1 = gpu_pairwise_l2(gpu, &flat, 3, 3);
    let run2 = gpu_pairwise_l2(gpu, &flat, 3, 3);

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(&a, &b)| (a - b).abs() < f32::EPSILON);
            h.check_bool("determinism: two runs identical", identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}

fn validate_upstream_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 5_u32;
    let dim = 3_u32;
    let features: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ];
    let flat: Vec<f32> = features
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f32))
        .collect();

    let local = gpu_pairwise_l2(gpu, &flat, n, dim);

    let dev = Arc::clone(gpu.wgpu_device());
    let device = gpu.device();
    let op = PairwiseL2Gpu::new(dev);
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("features"),
        contents: bytemuck::cast_slice(&flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let n_pairs = 5 * 4 / 2;
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("distances"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    op.dispatch(&input_buf, &output_buf, n, dim);
    let upstream = gpu.read_buffer_f32(&output_buf, n_pairs);

    match (local, upstream) {
        (Ok(l), Ok(u)) => {
            let max_diff: f64 = l
                .iter()
                .zip(u.iter())
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                .fold(0.0_f64, f64::max);
            h.check_upper(
                &format!("upstream parity: local vs PairwiseL2Gpu diff {max_diff:.2e}"),
                max_diff,
                tolerances::GPU_MODES_L2_F32,
            );
        }
        _ => h.check_bool("upstream parity: dispatch failed", false),
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
