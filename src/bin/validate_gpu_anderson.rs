// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: batch inverse participation ratio via `metalForge` WGSL shader.
//!
//! Validates `metalForge/shaders/batch_ipr.wgsl` against CPU IPR computation
//! from `anderson_localization.rs`.  The GPU shader computes IPR = sum(|ψ_i|^4)
//! for each eigenvector in a single dispatch.
//!
//! ## Papers validated
//!
//! - Papers 022-023: Anderson Localization / Spectral
//!
//! ## Provenance
//!
//! CPU reference: `anderson_localization::mean_ipr` (seed=0, Aubry-André n=16).
//! WGSL shader: `metalForge/shaders/batch_ipr.wgsl`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::doc_markdown
)]

use barracuda::pipeline::ReduceScalarPipeline;
use barracuda::spectral::BatchIprGpu;
use neural_spring::anderson_localization::{
    aubry_andre_hamiltonian, ipr, jacobi_eigh, GOLDEN_RATIO,
};
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = include_str!("../../metalForge/shaders/batch_ipr.wgsl");

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

    let mut h = ValidationHarness::new("gpu_anderson");

    validate_extended_state(&mut h, &gpu);
    validate_localized_state(&mut h, &gpu);
    validate_transition(&mut h, &gpu);
    validate_uniform_vector(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);
    validate_upstream_parity(&mut h, &gpu);
    validate_reduce_pipeline_mean(&mut h, &gpu);

    h.finish();
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    dim: u32,
    n_vectors: u32,
}

fn gpu_batch_ipr(
    gpu: &Gpu,
    eigenvectors: &[f32],
    dim: u32,
    n_vectors: u32,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("batch_ipr"),
        source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("batch_ipr_bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("batch_ipr_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("batch_ipr_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "batch_ipr",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let ev_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("eigenvectors"),
        contents: bytemuck::cast_slice(eigenvectors),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let n_vectors_usize = n_vectors as usize;
    let ipr_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ipr_out"),
        size: (n_vectors_usize * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = Params { dim, n_vectors };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("batch_ipr_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ev_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ipr_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("batch_ipr_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("batch_ipr_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(gpu.dispatch_1d(n_vectors, 256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f32(&ipr_buf, n_vectors_usize)
}

fn validate_extended_state(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 16_usize;
    let t = 1.0_f64;
    let w = 1.0_f64; // below transition
    let alpha = 1.0 / GOLDEN_RATIO;
    let phi = 0.0_f64;

    let h_mat = aubry_andre_hamiltonian(n, t, w, alpha, phi);
    let (_eigvals, ev) = jacobi_eigh(&h_mat, n);

    let mut flat: Vec<f32> = Vec::with_capacity(n * n);
    for k in 0..n {
        for i in 0..n {
            flat.push(ev[i * n + k] as f32);
        }
    }

    let cpu_ipr: Vec<f64> = (0..n)
        .map(|k| {
            let col: Vec<f64> = (0..n).map(|i| ev[i * n + k]).collect();
            ipr(&col)
        })
        .collect();

    match gpu_batch_ipr(gpu, &flat, n as u32, n as u32) {
        Ok(gpu_ipr) => {
            let max_diff: f64 = gpu_ipr
                .iter()
                .zip(cpu_ipr.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("extended (w=1.0): max GPU-CPU diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_BATCH_IPR_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("extended state: dispatch failed — {e}"), false);
        }
    }
}

fn validate_localized_state(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 16_usize;
    let t = 1.0_f64;
    let w = 4.0_f64; // above transition
    let alpha = 1.0 / GOLDEN_RATIO;
    let phi = 0.0_f64;

    let h_mat = aubry_andre_hamiltonian(n, t, w, alpha, phi);
    let (_eigvals, ev) = jacobi_eigh(&h_mat, n);

    let mut flat: Vec<f32> = Vec::with_capacity(n * n);
    for k in 0..n {
        for i in 0..n {
            flat.push(ev[i * n + k] as f32);
        }
    }

    let cpu_ipr: Vec<f64> = (0..n)
        .map(|k| {
            let col: Vec<f64> = (0..n).map(|i| ev[i * n + k]).collect();
            ipr(&col)
        })
        .collect();

    match gpu_batch_ipr(gpu, &flat, n as u32, n as u32) {
        Ok(gpu_ipr) => {
            let max_diff: f64 = gpu_ipr
                .iter()
                .zip(cpu_ipr.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("localized (w=4.0): max GPU-CPU diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_BATCH_IPR_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("localized state: dispatch failed — {e}"), false);
        }
    }
}

fn validate_transition(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 16_usize;
    let t = 1.0_f64;
    let alpha = 1.0 / GOLDEN_RATIO;
    let phi = 0.0_f64;

    let h_below = aubry_andre_hamiltonian(n, t, 1.0_f64, alpha, phi);
    let h_above = aubry_andre_hamiltonian(n, t, 4.0_f64, alpha, phi);

    let (_e1, ev_below) = jacobi_eigh(&h_below, n);
    let (_e2, ev_above) = jacobi_eigh(&h_above, n);

    let mut flat_below: Vec<f32> = Vec::with_capacity(n * n);
    for k in 0..n {
        for i in 0..n {
            flat_below.push(ev_below[i * n + k] as f32);
        }
    }
    let mut flat_above: Vec<f32> = Vec::with_capacity(n * n);
    for k in 0..n {
        for i in 0..n {
            flat_above.push(ev_above[i * n + k] as f32);
        }
    }

    match (
        gpu_batch_ipr(gpu, &flat_below, n as u32, n as u32),
        gpu_batch_ipr(gpu, &flat_above, n as u32, n as u32),
    ) {
        (Ok(ipr_below), Ok(ipr_above)) => {
            let mean_below: f64 =
                ipr_below.iter().map(|&v| f64::from(v)).sum::<f64>() / ipr_below.len() as f64;
            let mean_above: f64 =
                ipr_above.iter().map(|&v| f64::from(v)).sum::<f64>() / ipr_above.len() as f64;

            h.check_bool(
                &format!(
                    "transition: mean IPR localized ({mean_above:.4}) > extended ({mean_below:.4})"
                ),
                mean_above > mean_below,
            );
        }
        _ => {
            h.check_bool("transition: dispatch failed", false);
        }
    }
}

fn validate_uniform_vector(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 64_usize;
    let scale = 1.0 / (n as f64).sqrt();
    let expected_ipr = 1.0 / (n as f64); // IPR = n * (1/n)^2 = 1/n

    let uniform: Vec<f32> = (0..n).map(|_| scale as f32).collect();

    match gpu_batch_ipr(gpu, &uniform, n as u32, 1) {
        Ok(gpu_ipr) => {
            let got = f64::from(gpu_ipr[0]);
            let diff = (got - expected_ipr).abs();
            h.check_upper(
                &format!(
                    "uniform [1/√n]: GPU IPR ({got:.2e}) vs expected 1/n ({expected_ipr:.2e}), diff={diff:.2e}"
                ),
                diff,
                tolerances::GPU_BATCH_IPR_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("uniform vector: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 16_usize;
    let t = 1.0_f64;
    let w = 2.0_f64;
    let alpha = 1.0 / GOLDEN_RATIO;
    let phi = 0.0_f64;

    let h_mat = aubry_andre_hamiltonian(n, t, w, alpha, phi);
    let (_eigvals, ev) = jacobi_eigh(&h_mat, n);

    let mut flat: Vec<f32> = Vec::with_capacity(n * n);
    for k in 0..n {
        for i in 0..n {
            flat.push(ev[i * n + k] as f32);
        }
    }

    let run1 = gpu_batch_ipr(gpu, &flat, n as u32, n as u32);
    let run2 = gpu_batch_ipr(gpu, &flat, n as u32, n as u32);

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(&a, &b)| (a - b).abs() < f32::EPSILON);
            h.check_bool("determinism: two IPR runs bit-identical", identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}

fn validate_reduce_pipeline_mean(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 64_usize;
    let t = 1.0_f64;
    let w = 2.0_f64;
    let h_mat = aubry_andre_hamiltonian(n, t, w, GOLDEN_RATIO, 0.0);
    let (_, ev) = jacobi_eigh(&h_mat, n);

    let cpu_iprs: Vec<f64> = (0..n)
        .map(|k| {
            let col: Vec<f64> = (0..n).map(|i| ev[i * n + k]).collect();
            ipr(&col)
        })
        .collect();
    let cpu_mean = cpu_iprs.iter().sum::<f64>() / cpu_iprs.len() as f64;

    let dev = Arc::clone(gpu.wgpu_device());
    match ReduceScalarPipeline::new(Arc::clone(&dev), n) {
        Ok(reducer) => {
            let ipr_bytes: Vec<u8> = cpu_iprs.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let ipr_buf = gpu.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ipr_f64"),
                contents: &ipr_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            match reducer.sum_f64(&ipr_buf) {
                Ok(gpu_sum) => {
                    let gpu_mean = gpu_sum / n as f64;
                    let diff = (gpu_mean - cpu_mean).abs();
                    h.check_upper(
                        &format!(
                            "ReduceScalarPipeline mean IPR: GPU {gpu_mean:.8} vs CPU {cpu_mean:.8}, diff {diff:.2e}"
                        ),
                        diff,
                        tolerances::GPU_REDUCE_F64,
                    );
                }
                Err(e) => h.check_bool(&format!("ReduceScalarPipeline sum failed: {e}"), false),
            }
        }
        Err(e) => h.check_bool(&format!("ReduceScalarPipeline::new failed: {e}"), false),
    }
}

fn validate_upstream_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 32_usize;
    let dim = n as u32;
    let n_vectors = n as u32;
    let t = 1.0_f64;
    let w = 1.0_f64;
    let h_matrix = aubry_andre_hamiltonian(n, t, w, GOLDEN_RATIO, 0.0);
    let (_, eigvecs) = jacobi_eigh(&h_matrix, n);
    let flat: Vec<f32> = eigvecs.iter().map(|&v| v as f32).collect();

    let local = gpu_batch_ipr(gpu, &flat, dim, n_vectors);

    let dev = Arc::clone(gpu.wgpu_device());
    let device = gpu.device();
    let op = BatchIprGpu::new(dev);
    let ev_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ev"), contents: bytemuck::cast_slice(&flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let ipr_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ipr"), size: u64::from(n_vectors) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    op.dispatch(&ev_buf, &ipr_buf, dim, n_vectors);
    let upstream = gpu.read_buffer_f32(&ipr_buf, n_vectors as usize);

    match (local, upstream) {
        (Ok(l), Ok(u)) => {
            let max_diff: f64 = l.iter().zip(u.iter())
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                .fold(0.0_f64, f64::max);
            h.check_upper(
                &format!("upstream parity: local vs BatchIprGpu diff {max_diff:.2e}"),
                max_diff, tolerances::GPU_BATCH_IPR_F32,
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
