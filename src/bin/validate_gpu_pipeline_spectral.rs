// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: pure GPU IPR→mean pipeline (Papers 022-023).
//!
//! Chains two GPU kernels in a single `queue.submit` to prove that
//! intermediate data stays GPU-resident and no CPU math happens between
//! stages.
//!
//! ## Pipeline
//!
//! ```text
//! eigenvectors (upload once)
//!   ↓
//! ┌─────────────────────────────────────────────────────┐
//! │  Stage 1: batch_ipr.wgsl                             │
//! │    IPR per eigenvector → ipr_out[n_vectors]         │
//! │                                                      │
//! │  Stage 2: mean_reduce.wgsl                           │
//! │    ipr_out[n_vectors] → mean IPR (scalar)             │
//! └─────────────────────────────────────────────────────┘
//!   ↓  (single queue.submit — NO CPU round-trip)
//! Readback: 4 bytes (one f32 scalar)
//! ```
//!
//! ## Provenance
//!
//! GPU pipeline: `batch_ipr` → `mean_reduce`.
//! Validates: end-to-end GPU-resident computation with scalar-only readback.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const IPR_WGSL: &str = include_str!("../../metalForge/shaders/batch_ipr.wgsl");
const REDUCE_WGSL: &str = include_str!("../../metalForge/shaders/mean_reduce.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    dim: u32,
    n_vectors: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ReduceParams {
    n: u32,
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

    let mut h = ValidationHarness::new("gpu_pipeline_spectral");

    validate_uniform_vectors(&mut h, &gpu);
    validate_localized_vectors(&mut h, &gpu);
    validate_random_vectors(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_ipr(eigenvectors: &[f32], dim: usize, n_vectors: usize) -> f32 {
    let mut total = 0.0f32;
    for v in 0..n_vectors {
        let mut sum_p4 = 0.0f32;
        for i in 0..dim {
            let val = eigenvectors[v * dim + i];
            let p2 = val * val;
            sum_p4 += p2 * p2;
        }
        total += sum_p4;
    }
    total / n_vectors as f32
}

// ── GPU chained pipeline ───────────────────────────────────────────

fn gpu_chained_mean_ipr(
    gpu: &Gpu,
    eigenvectors: &[f32],
    dim: u32,
    n_vectors: u32,
) -> Result<f32, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    // Stage 1: batch IPR
    let ipr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_ipr"),
        source: wgpu::ShaderSource::Wgsl(IPR_WGSL.into()),
    });

    let ipr_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_ipr_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });

    let ipr_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_ipr_pl"),
        bind_group_layouts: &[&ipr_bgl],
        push_constant_ranges: &[],
    });

    let ipr_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_ipr_pipeline"),
        layout: Some(&ipr_pl),
        module: &ipr_shader,
        entry_point: "batch_ipr",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Stage 2: mean reduction
    let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_reduce"),
        source: wgpu::ShaderSource::Wgsl(REDUCE_WGSL.into()),
    });

    let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_reduce_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });

    let reduce_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_reduce_pl"),
        bind_group_layouts: &[&reduce_bgl],
        push_constant_ranges: &[],
    });

    let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_reduce_pipeline"),
        layout: Some(&reduce_pl),
        module: &reduce_shader,
        entry_point: "mean_reduce",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Buffers
    let eigenvectors_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_eigenvectors"),
        contents: bytemuck::cast_slice(eigenvectors),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let ipr_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_ipr_out"),
        size: u64::from(n_vectors) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let ipr_params = Params { dim, n_vectors };
    let ipr_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_ipr_params"),
        contents: bytemuck::bytes_of(&ipr_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_mean_result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let reduce_params = ReduceParams { n: n_vectors };
    let reduce_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_reduce_params"),
        contents: bytemuck::bytes_of(&reduce_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Bind groups
    let ipr_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_ipr_bg"),
        layout: &ipr_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: eigenvectors_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ipr_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ipr_params_buf.as_entire_binding(),
            },
        ],
    });

    let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_reduce_bg"),
        layout: &reduce_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ipr_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: reduce_params_buf.as_entire_binding(),
            },
        ],
    });

    // Single CommandEncoder — both stages, one submit, zero CPU round-trips
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("chain_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_ipr_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ipr_pipeline);
        pass.set_bind_group(0, &ipr_bg, &[]);
        pass.dispatch_workgroups(n_vectors.div_ceil(256), 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_reduce_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&reduce_pipeline);
        pass.set_bind_group(0, &reduce_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));

    let result = gpu.read_buffer_f32(&result_buf, 1)?;
    Ok(result[0])
}

// ── Validation functions ───────────────────────────────────────────

fn validate_uniform_vectors(h: &mut ValidationHarness, gpu: &Gpu) {
    // IPR = 1/dim for each uniform vector, mean = 1/dim
    let dim = 8_usize;
    let n_vectors = 4_usize;
    let val = 1.0f32 / (dim as f32).sqrt();
    let eigenvectors: Vec<f32> = vec![val; dim * n_vectors];
    let expected = 1.0f32 / dim as f32;

    match gpu_chained_mean_ipr(gpu, &eigenvectors, dim as u32, n_vectors as u32) {
        Ok(gpu_mean) => {
            h.check_bool(
                &format!("uniform: GPU mean finite ({gpu_mean:.6})"),
                gpu_mean.is_finite(),
            );
            h.check_abs(
                &format!("uniform: GPU={gpu_mean:.6} vs expected={expected:.6} (1/dim)"),
                f64::from(gpu_mean),
                f64::from(expected),
                tolerances::GPU_BATCH_IPR_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("uniform: dispatch failed — {e}"), false);
        }
    }
}

fn validate_localized_vectors(h: &mut ValidationHarness, gpu: &Gpu) {
    // One-hot vectors: IPR = 1.0 for each, mean = 1.0
    let dim = 8_usize;
    let n_vectors = 4_usize;
    let mut eigenvectors = vec![0.0f32; dim * n_vectors];
    for v in 0..n_vectors {
        eigenvectors[v * dim + v % dim] = 1.0f32;
    }

    match gpu_chained_mean_ipr(gpu, &eigenvectors, dim as u32, n_vectors as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("localized: GPU={gpu_mean:.6} vs expected=1.0"),
                f64::from(gpu_mean),
                1.0,
                tolerances::GPU_BATCH_IPR_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("localized: dispatch failed — {e}"), false);
        }
    }
}

fn validate_random_vectors(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 16_usize;
    let n_vectors = 8_usize;
    let mut rng = Rng::new(42);
    let eigenvectors: Vec<f32> = (0..dim * n_vectors).map(|_| rng.uniform() as f32).collect();

    let cpu_mean = cpu_mean_ipr(&eigenvectors, dim, n_vectors);

    match gpu_chained_mean_ipr(gpu, &eigenvectors, dim as u32, n_vectors as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("random: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_BATCH_IPR_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("random: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 16_usize;
    let n_vectors = 8_usize;
    let mut rng = Rng::new(42);
    let eigenvectors: Vec<f32> = (0..dim * n_vectors).map(|_| rng.uniform() as f32).collect();

    let r1 = gpu_chained_mean_ipr(gpu, &eigenvectors, dim as u32, n_vectors as u32);
    let r2 = gpu_chained_mean_ipr(gpu, &eigenvectors, dim as u32, n_vectors as u32);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}

// ── wgpu layout helpers ────────────────────────────────────────────

const fn storage_ro_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

const fn storage_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
