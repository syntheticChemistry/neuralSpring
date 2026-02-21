// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline: `pairwise_l2` → `mean_reduce` → scalar readback (Paper 012).
//!
//! Chains two shader stages in a single `CommandEncoder` with zero CPU round-trips.
//! Stage 1: `pairwise_l2` — all pairwise L2 distances from feature vectors.
//! Stage 2: `mean_reduce` — distance array to scalar mean.
//!
//! ## Pipeline
//!
//! ```text
//! Upload features [N x D] (once)
//!   ↓
//! ┌──────────────────────────────────────────────────┐
//! │  Stage 1: pairwise_l2.wgsl                       │
//! │    features[N*D] → distances[N*(N-1)/2]          │
//! │                                                  │
//! │  Stage 2: mean_reduce.wgsl                       │
//! │    distances[N*(N-1)/2] → mean_distance (scalar) │
//! └──────────────────────────────────────────────────┘
//!   ↓  (single queue.submit — NO CPU round-trip)
//! Readback: 4 bytes (one f32 scalar)
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::modes::l2_distance;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const PAIRWISE_L2_WGSL: &str = include_str!("../../metalForge/shaders/pairwise_l2.wgsl");
const REDUCE_WGSL: &str = include_str!("../../metalForge/shaders/mean_reduce.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct L2Params {
    n: u32,
    dim: u32,
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

    let mut h = ValidationHarness::new("gpu_pipeline_modes");

    validate_small(&mut h, &gpu);
    validate_identical(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_pairwise_l2(features: &[Vec<f64>]) -> f64 {
    let n = features.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += l2_distance(&features[i], &features[j]);
            count += 1;
        }
    }
    sum / count as f64
}

// ── GPU chained pipeline ───────────────────────────────────────────

fn gpu_pairwise_l2_mean(gpu: &Gpu, features_flat: &[f32], n: u32, dim: u32) -> Result<f32, String> {
    let device = gpu.device();
    let queue = gpu.queue();
    let n_pairs = (n * (n - 1) / 2) as usize;

    let pairwise_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_pairwise_l2"),
        source: wgpu::ShaderSource::Wgsl(PAIRWISE_L2_WGSL.into()),
    });

    let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_reduce"),
        source: wgpu::ShaderSource::Wgsl(REDUCE_WGSL.into()),
    });

    let pairwise_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_pairwise_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });

    let pairwise_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_pairwise_pl"),
        bind_group_layouts: &[&pairwise_bgl],
        push_constant_ranges: &[],
    });

    let pairwise_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_pairwise_pipeline"),
        layout: Some(&pairwise_pl),
        module: &pairwise_shader,
        entry_point: "pairwise_l2",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
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

    let features_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_features"),
        contents: bytemuck::cast_slice(features_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let distances_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_distances"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let l2_params = L2Params { n, dim };
    let l2_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_l2_params"),
        contents: bytemuck::bytes_of(&l2_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_mean_result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let reduce_params = ReduceParams { n: n_pairs as u32 };
    let reduce_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_reduce_params"),
        contents: bytemuck::bytes_of(&reduce_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let pairwise_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_pairwise_bg"),
        layout: &pairwise_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: features_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: distances_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: l2_params_buf.as_entire_binding(),
            },
        ],
    });

    let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_reduce_bg"),
        layout: &reduce_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: distances_buf.as_entire_binding(),
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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("chain_modes_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_pairwise_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pairwise_pipeline);
        pass.set_bind_group(0, &pairwise_bg, &[]);
        pass.dispatch_workgroups((n_pairs as u32).div_ceil(256), 1, 1);
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

fn validate_small(h: &mut ValidationHarness, gpu: &Gpu) {
    let features: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ];
    let n = 5_u32;
    let dim = 3_u32;

    let cpu_mean = cpu_mean_pairwise_l2(&features);

    let flat: Vec<f32> = features
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f32))
        .collect();

    match gpu_pairwise_l2_mean(gpu, &flat, n, dim) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("modes small 5×3: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                cpu_mean,
                tolerances::GPU_MODES_L2_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("modes small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_identical(h: &mut ValidationHarness, gpu: &Gpu) {
    let features: Vec<Vec<f64>> = vec![
        vec![0.5, 0.5, 0.5],
        vec![0.5, 0.5, 0.5],
        vec![0.5, 0.5, 0.5],
    ];
    let n = 3_u32;
    let dim = 3_u32;

    let cpu_mean = cpu_mean_pairwise_l2(&features);

    let flat: Vec<f32> = features
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f32))
        .collect();

    match gpu_pairwise_l2_mean(gpu, &flat, n, dim) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("modes identical: GPU={gpu_mean:.6} vs CPU mean=0"),
                f64::from(gpu_mean),
                cpu_mean,
                tolerances::GPU_MODES_L2_F32,
            );
            h.check_abs(
                "modes identical: mean distance should be 0",
                f64::from(gpu_mean),
                0.0,
                tolerances::GPU_MODES_L2_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("modes identical: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let features: Vec<Vec<f64>> = vec![
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
        vec![0.7, 0.8, 0.9],
        vec![0.2, 0.3, 0.4],
        vec![0.5, 0.6, 0.7],
    ];
    let flat: Vec<f32> = features
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f32))
        .collect();

    let r1 = gpu_pairwise_l2_mean(gpu, &flat, 5, 3);
    let r2 = gpu_pairwise_l2_mean(gpu, &flat, 5, 3);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("modes determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("modes determinism: dispatch failed", false);
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
