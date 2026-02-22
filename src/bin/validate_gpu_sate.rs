// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: pairwise Hamming distance via `metalForge` WGSL shader.
//!
//! Validates `metalForge/shaders/pairwise_hamming.wgsl` against CPU
//! Hamming distance computation from `sate_alignment.rs`.  The GPU
//! shader evaluates all n*(n-1)/2 pairwise distances in a single dispatch.
//!
//! ## Papers validated
//!
//! - Paper 017: SATé Alignment (Liu et al., 2009)
//!
//! ## Provenance
//!
//! CPU reference: `sate_alignment::pairwise_distance_matrix` (seed=42, n_seqs=8 seq_len=50).
//! WGSL shader: `metalForge/shaders/pairwise_hamming.wgsl`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::doc_markdown,
    clippy::needless_range_loop
)]

use barracuda::ops::bio::PairwiseHammingGpu;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::sate_alignment::pairwise_distance_matrix;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = include_str!("../../metalForge/shaders/pairwise_hamming.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_seqs: u32,
    seq_len: u32,
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

    let mut h = ValidationHarness::new("gpu_sate");

    validate_small(&mut h, &gpu);
    validate_larger(&mut h, &gpu);
    validate_identical_sequences(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);
    validate_upstream_parity(&mut h, &gpu);

    h.finish();
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

fn generate_test_sequences(n_seqs: usize, seq_len: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let mut flat = Vec::with_capacity(n_seqs * seq_len);
    for _ in 0..(n_seqs * seq_len) {
        flat.push(rng.usize(4) as u32);
    }
    flat
}

fn gpu_pairwise_hamming(
    gpu: &Gpu,
    sequences: &[u32],
    n_seqs: u32,
    seq_len: u32,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pairwise_hamming"),
        source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hamming_bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hamming_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hamming_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "pairwise_hamming",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let seq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sequences"),
        contents: bytemuck::cast_slice(sequences),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let n_pairs = (n_seqs * (n_seqs - 1) / 2) as usize;
    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("distances"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = Params { n_seqs, seq_len };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hamming_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: seq_buf.as_entire_binding(),
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
        label: Some("hamming_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hamming_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((n_pairs as u32).div_ceil(256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f32(&dist_buf, n_pairs)
}

fn validate_small(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 8_usize;
    let seq_len = 50_usize;
    let flat = generate_test_sequences(n_seqs, seq_len, 42);
    let seqs_u8: Vec<u8> = flat.iter().map(|&v| v as u8).collect();

    let cpu_matrix = pairwise_distance_matrix(&seqs_u8, n_seqs, seq_len, false);
    let mut cpu_upper = Vec::new();
    for i in 0..n_seqs {
        for j in (i + 1)..n_seqs {
            cpu_upper.push(cpu_matrix[i * n_seqs + j]);
        }
    }

    match gpu_pairwise_hamming(gpu, &flat, n_seqs as u32, seq_len as u32) {
        Ok(gpu_dist) => {
            h.check_bool(
                &format!("small: correct pair count ({})", gpu_dist.len()),
                gpu_dist.len() == cpu_upper.len(),
            );

            let max_diff: f64 = gpu_dist
                .iter()
                .zip(cpu_upper.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("small: max GPU-CPU diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_HAMMING_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_larger(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 20_usize;
    let seq_len = 200_usize;
    let flat = generate_test_sequences(n_seqs, seq_len, 77);
    let seqs_u8: Vec<u8> = flat.iter().map(|&v| v as u8).collect();

    let cpu_matrix = pairwise_distance_matrix(&seqs_u8, n_seqs, seq_len, false);
    let mut cpu_upper = Vec::new();
    for i in 0..n_seqs {
        for j in (i + 1)..n_seqs {
            cpu_upper.push(cpu_matrix[i * n_seqs + j]);
        }
    }

    match gpu_pairwise_hamming(gpu, &flat, n_seqs as u32, seq_len as u32) {
        Ok(gpu_dist) => {
            let max_diff: f64 = gpu_dist
                .iter()
                .zip(cpu_upper.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "20×200: max GPU-CPU diff ({max_diff:.2e}), {} pairs",
                    gpu_dist.len()
                ),
                max_diff,
                tolerances::GPU_HAMMING_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("20×200: dispatch failed — {e}"), false);
        }
    }
}

fn validate_identical_sequences(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 10_u32;
    let seq_len = 64_u32;
    let template: Vec<u32> = vec![2; seq_len as usize];
    let flat: Vec<u32> = template
        .iter()
        .cycle()
        .take((n_seqs * seq_len) as usize)
        .copied()
        .collect();

    match gpu_pairwise_hamming(gpu, &flat, n_seqs, seq_len) {
        Ok(gpu_dist) => {
            let max_dist = gpu_dist.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let all_zero = gpu_dist
                .iter()
                .all(|&d| d.abs() < tolerances::GPU_HAMMING_F32 as f32);
            h.check_bool(
                &format!("identical sequences: all Hamming=0 (max={max_dist:.2e})"),
                all_zero,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("identical sequences: dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 8_u32;
    let seq_len = 50_u32;
    let flat = generate_test_sequences(n_seqs as usize, seq_len as usize, 123);

    let run1 = gpu_pairwise_hamming(gpu, &flat, n_seqs, seq_len);
    let run2 = gpu_pairwise_hamming(gpu, &flat, n_seqs, seq_len);

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let bit_identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
            h.check_bool("determinism: two Hamming runs bit-identical", bit_identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}

fn validate_upstream_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 8_u32;
    let seq_len = 50_u32;
    let flat = generate_test_sequences(n_seqs as usize, seq_len as usize, 42);
    let n_pairs = (n_seqs * (n_seqs - 1) / 2) as usize;

    let local = gpu_pairwise_hamming(gpu, &flat, n_seqs, seq_len);

    let dev = Arc::clone(gpu.wgpu_device());
    let device = gpu.device();
    let op = PairwiseHammingGpu::new(dev);
    let seq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("seqs"), contents: bytemuck::cast_slice(&flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dist"), size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    op.dispatch(&seq_buf, &dist_buf, n_seqs, seq_len);
    let upstream = gpu.read_buffer_f32(&dist_buf, n_pairs);

    match (local, upstream) {
        (Ok(l), Ok(u)) => {
            let max_diff: f64 = l.iter().zip(u.iter())
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                .fold(0.0_f64, f64::max);
            h.check_upper(
                &format!("upstream parity: local vs PairwiseHammingGpu diff {max_diff:.2e}"),
                max_diff, tolerances::GPU_HAMMING_F32,
            );
        }
        _ => h.check_bool("upstream parity: dispatch failed", false),
    }
}
