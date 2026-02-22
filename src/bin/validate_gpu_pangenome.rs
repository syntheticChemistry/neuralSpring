// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: pairwise Jaccard distance via `metalForge` WGSL shader.
//!
//! Validates `metalForge/shaders/pairwise_jaccard.wgsl` against CPU
//! Jaccard distance computation from `pangenome_selection.rs`.  The GPU
//! shader evaluates all n*(n-1)/2 pairwise distances in a single dispatch.
//!
//! Evolution path:
//! ```text
//! Python (numpy binary ops) → Rust CPU (loop) → BarraCUDA CPU (stats)
//!   → GPU WGSL shader (pairwise_jaccard.wgsl) → ToadStool absorption
//! ```
//!
//! ## Papers validated
//!
//! - Paper 024: Pangenome Selection Dynamics (Anderson, 2024)
//!
//! ## Provenance
//!
//! CPU reference: `pangenome_selection::jaccard_distance_matrix` (upper-triangle pairwise).
//! WGSL shader: `metalForge/shaders/pairwise_jaccard.wgsl`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::similar_names
)]

use barracuda::ops::bio::PairwiseJaccardGpu;
use neural_spring::gpu::Gpu;
use neural_spring::pangenome_selection::{generate_pa_matrix, jaccard_distance_matrix};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = include_str!("../../metalForge/shaders/pairwise_jaccard.wgsl");

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

    let mut h = ValidationHarness::new("gpu_pangenome");

    validate_small_pa(&mut h, &gpu);
    validate_larger_pa(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);
    validate_identity_diagonal(&mut h, &gpu);
    validate_upstream_parity(&mut h, &gpu);

    h.finish();
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct JaccardParams {
    n_genomes: u32,
    n_genes: u32,
}

fn gpu_pairwise_jaccard(
    gpu: &Gpu,
    pa: &[f32],
    n_genomes: u32,
    n_genes: u32,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pairwise_jaccard"),
        source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("jaccard_bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("jaccard_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("jaccard_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "pairwise_jaccard",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let pa_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pa_matrix"),
        contents: bytemuck::cast_slice(pa),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let n_pairs = (n_genomes * (n_genomes - 1) / 2) as usize;
    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("distances"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = JaccardParams { n_genomes, n_genes };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("jaccard_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pa_buf.as_entire_binding(),
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
        label: Some("jaccard_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jaccard_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(gpu.dispatch_1d(n_pairs as u32, 256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f32(&dist_buf, n_pairs)
}

fn validate_small_pa(h: &mut ValidationHarness, gpu: &Gpu) {
    let mut rng = Rng::new(42);
    let n_genomes = 10_usize;
    let n_genes = 50_usize;
    let env_labels: Vec<usize> = (0..5).map(|_| 0).chain((0..5).map(|_| 1)).collect();
    let pa = generate_pa_matrix(n_genomes, n_genes, 0.25, 0.10, &mut rng, &env_labels);

    let cpu_jd = jaccard_distance_matrix(&pa, n_genes, n_genomes);
    let mut cpu_upper = Vec::new();
    for i in 0..n_genomes {
        for j in (i + 1)..n_genomes {
            cpu_upper.push(cpu_jd[i * n_genomes + j]);
        }
    }

    let pa_f32: Vec<f32> = pa.iter().map(|&v| v as f32).collect();

    match gpu_pairwise_jaccard(gpu, &pa_f32, n_genomes as u32, n_genes as u32) {
        Ok(gpu_dist) => {
            h.check_bool(
                &format!("small PA: correct pair count ({})", gpu_dist.len()),
                gpu_dist.len() == cpu_upper.len(),
            );

            let max_diff: f64 = gpu_dist
                .iter()
                .zip(cpu_upper.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("small PA: max GPU-CPU diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_JACCARD_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("small PA: dispatch failed — {e}"), false);
        }
    }
}

fn validate_larger_pa(h: &mut ValidationHarness, gpu: &Gpu) {
    let mut rng = Rng::new(77);
    let n_genomes = 30_usize;
    let n_genes = 200_usize;
    let env_labels: Vec<usize> = (0..15).map(|_| 0).chain((0..15).map(|_| 1)).collect();
    let pa = generate_pa_matrix(n_genomes, n_genes, 0.25, 0.10, &mut rng, &env_labels);

    let cpu_jd = jaccard_distance_matrix(&pa, n_genes, n_genomes);
    let mut cpu_upper = Vec::new();
    for i in 0..n_genomes {
        for j in (i + 1)..n_genomes {
            cpu_upper.push(cpu_jd[i * n_genomes + j]);
        }
    }

    let pa_f32: Vec<f32> = pa.iter().map(|&v| v as f32).collect();

    match gpu_pairwise_jaccard(gpu, &pa_f32, n_genomes as u32, n_genes as u32) {
        Ok(gpu_dist) => {
            let max_diff: f64 = gpu_dist
                .iter()
                .zip(cpu_upper.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "30×200 PA: max GPU-CPU diff ({max_diff:.2e}), {} pairs",
                    gpu_dist.len()
                ),
                max_diff,
                tolerances::GPU_JACCARD_F32,
            );

            let gpu_mean: f64 =
                gpu_dist.iter().map(|&v| f64::from(v)).sum::<f64>() / gpu_dist.len() as f64;
            h.check_lower(
                &format!("30×200 PA: mean Jaccard > 0 ({gpu_mean:.4})"),
                gpu_mean,
                0.0,
            );
        }
        Err(e) => {
            h.check_bool(&format!("30×200 PA: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let mut rng = Rng::new(42);
    let n_genomes = 10_u32;
    let n_genes = 50_u32;
    let env_labels: Vec<usize> = (0..5).map(|_| 0).chain((0..5).map(|_| 1)).collect();
    let pa = generate_pa_matrix(
        n_genomes as usize,
        n_genes as usize,
        0.25,
        0.10,
        &mut rng,
        &env_labels,
    );
    let pa_f32: Vec<f32> = pa.iter().map(|&v| v as f32).collect();

    let run1 = gpu_pairwise_jaccard(gpu, &pa_f32, n_genomes, n_genes);
    let run2 = gpu_pairwise_jaccard(gpu, &pa_f32, n_genomes, n_genes);

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(&a, &b)| (a - b).abs() < f32::EPSILON);
            h.check_bool("determinism: two Jaccard runs identical", identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}

fn validate_identity_diagonal(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_genomes = 4_u32;
    let n_genes = 8_u32;
    let pa_f32: Vec<f32> = vec![1.0; (n_genomes * n_genes) as usize];

    match gpu_pairwise_jaccard(gpu, &pa_f32, n_genomes, n_genes) {
        Ok(gpu_dist) => {
            let all_zero = gpu_dist
                .iter()
                .all(|&d| d.abs() < tolerances::GPU_JACCARD_F32 as f32);
            h.check_bool(
                &format!(
                    "identical genomes: all Jaccard=0 (max={:.2e})",
                    gpu_dist.iter().map(|v| v.abs()).fold(0.0_f32, f32::max)
                ),
                all_zero,
            );
        }
        Err(e) => {
            h.check_bool(&format!("identity: dispatch failed — {e}"), false);
        }
    }
}

fn validate_upstream_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let mut rng = Rng::new(42);
    let n_genomes = 10_u32;
    let n_genes = 50_u32;
    let env_labels: Vec<usize> = (0..5).map(|_| 0).chain((0..5).map(|_| 1)).collect();
    let pa = generate_pa_matrix(
        n_genomes as usize, n_genes as usize, 0.25, 0.10, &mut rng, &env_labels,
    );
    let pa_f32: Vec<f32> = pa.iter().map(|&v| v as f32).collect();
    let n_pairs = (n_genomes * (n_genomes - 1) / 2) as usize;

    let local = gpu_pairwise_jaccard(gpu, &pa_f32, n_genomes, n_genes);

    let dev = Arc::clone(gpu.wgpu_device());
    let device = gpu.device();
    let op = PairwiseJaccardGpu::new(dev);
    let pa_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pa"), contents: bytemuck::cast_slice(&pa_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dist"), size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    op.dispatch(&pa_buf, &dist_buf, n_genomes, n_genes);
    let upstream = gpu.read_buffer_f32(&dist_buf, n_pairs);

    match (local, upstream) {
        (Ok(l), Ok(u)) => {
            let max_diff: f64 = l.iter().zip(u.iter())
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                .fold(0.0_f64, f64::max);
            h.check_upper(
                &format!("upstream parity: local vs PairwiseJaccardGpu diff {max_diff:.2e}"),
                max_diff, tolerances::GPU_JACCARD_F32,
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
