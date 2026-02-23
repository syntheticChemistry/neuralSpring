// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: pure GPU Jaccard→mean pipeline (Paper 024).
//!
//! Chains two GPU kernels in a single `queue.submit` to prove that
//! intermediate data stays GPU-resident and no CPU math happens between
//! stages.
//!
//! ## Pipeline
//!
//! ```text
//! PA matrix (upload once)
//!   ↓
//! ┌─────────────────────────────────────────────────────┐
//! │  Stage 1: pairwise_jaccard.wgsl                     │
//! │    upper-triangle Jaccard distances → distances[]   │
//! │                                                      │
//! │  Stage 2: mean_reduce.wgsl                           │
//! │    distances[] → mean distance (scalar)             │
//! └─────────────────────────────────────────────────────┘
//!   ↓  (single queue.submit — NO CPU round-trip)
//! Readback: 4 bytes (one f32 scalar)
//! ```
//!
//! ## Provenance
//!
//! GPU pipeline: `pairwise_jaccard` → `mean_reduce`.
//! Validates: end-to-end GPU-resident computation with scalar-only readback.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const JACCARD_WGSL: &str = include_str!("../../metalForge/shaders/pairwise_jaccard.wgsl");
const REDUCE_WGSL: &str = include_str!("../../metalForge/shaders/mean_reduce.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct JaccardParams {
    n_genomes: u32,
    n_genes: u32,
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

    let mut h = ValidationHarness::new("gpu_pipeline_genomics");

    validate_identical_genomes(&mut h, &gpu);
    validate_disjoint_genomes(&mut h, &gpu);
    validate_random_pa_small(&mut h, &gpu);
    validate_random_pa_larger(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_jaccard(pa: &[f32], n_genes: usize, n_genomes: usize) -> f32 {
    let n_pairs = n_genomes * (n_genomes - 1) / 2;
    let mut total = 0.0f32;
    for i in 0..n_genomes {
        for j in (i + 1)..n_genomes {
            let mut inter = 0.0f32;
            let mut union_c = 0.0f32;
            for g in 0..n_genes {
                let a = pa[g * n_genomes + i];
                let b = pa[g * n_genomes + j];
                inter += a * b;
                union_c += a.max(b);
            }
            let dist = if union_c > 0.0 {
                1.0 - inter / union_c
            } else {
                1.0
            };
            total += dist;
        }
    }
    total / n_pairs as f32
}

// ── GPU chained pipeline ───────────────────────────────────────────

fn gpu_chained_mean_jaccard(
    gpu: &Gpu,
    pa: &[f32],
    n_genomes: u32,
    n_genes: u32,
) -> Result<f32, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let n_pairs = (n_genomes as usize) * (n_genomes as usize - 1) / 2;

    // Stage 1: pairwise Jaccard
    let jaccard_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_jaccard"),
        source: wgpu::ShaderSource::Wgsl(JACCARD_WGSL.into()),
    });

    let jaccard_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_jaccard_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });

    let jaccard_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_jaccard_pl"),
        bind_group_layouts: &[&jaccard_bgl],
        push_constant_ranges: &[],
    });

    let jaccard_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_jaccard_pipeline"),
        layout: Some(&jaccard_pl),
        module: &jaccard_shader,
        entry_point: "pairwise_jaccard",
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
    let pa_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_pa"),
        contents: bytemuck::cast_slice(pa),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let distances_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_distances_out"),
        size: (n_pairs as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let jaccard_params = JaccardParams { n_genomes, n_genes };
    let jaccard_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_jaccard_params"),
        contents: bytemuck::bytes_of(&jaccard_params),
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

    // Bind groups
    let jaccard_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_jaccard_bg"),
        layout: &jaccard_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pa_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: distances_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: jaccard_params_buf.as_entire_binding(),
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

    // Single CommandEncoder — both stages, one submit, zero CPU round-trips
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("chain_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_jaccard_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&jaccard_pipeline);
        pass.set_bind_group(0, &jaccard_bg, &[]);
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

fn validate_identical_genomes(h: &mut ValidationHarness, gpu: &Gpu) {
    // All genomes identical → all distances = 0.0, mean = 0.0
    let n_genomes = 4_usize;
    let n_genes = 8_usize;
    let mut pa = vec![0.0f32; n_genes * n_genomes];
    for g in 0..n_genes {
        for _genome in 0..n_genomes {
            pa[g * n_genomes + _genome] = (g % 2) as f32; // same column for all
        }
    }

    match gpu_chained_mean_jaccard(gpu, &pa, n_genomes as u32, n_genes as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("identical: GPU={gpu_mean:.6} vs expected=0.0"),
                f64::from(gpu_mean),
                0.0,
                tolerances::GPU_JACCARD_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("identical: dispatch failed — {e}"), false);
        }
    }
}

fn validate_disjoint_genomes(h: &mut ValidationHarness, gpu: &Gpu) {
    // Disjoint genomes: each gene present in exactly one genome
    // All pairs have intersection=0, union>0 → distance = 1.0, mean = 1.0
    let n_genomes = 4_usize;
    let n_genes = 8_usize; // 2 genes per genome, disjoint
    let mut pa = vec![0.0f32; n_genes * n_genomes];
    for g in 0..n_genes {
        let genome = g % n_genomes;
        pa[g * n_genomes + genome] = 1.0f32;
    }

    match gpu_chained_mean_jaccard(gpu, &pa, n_genomes as u32, n_genes as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("disjoint: GPU={gpu_mean:.6} vs expected=1.0"),
                f64::from(gpu_mean),
                1.0,
                tolerances::GPU_JACCARD_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("disjoint: dispatch failed — {e}"), false);
        }
    }
}

fn validate_random_pa_small(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_genomes = 8_usize;
    let n_genes = 50_usize;
    let mut rng = Rng::new(42);
    let pa: Vec<f32> = (0..n_genes * n_genomes)
        .map(|_| if rng.uniform() < 0.5 { 0.0 } else { 1.0 })
        .collect();

    let cpu_mean = cpu_mean_jaccard(&pa, n_genes, n_genomes);

    match gpu_chained_mean_jaccard(gpu, &pa, n_genomes as u32, n_genes as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("random 8×50: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_JACCARD_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("random 8×50: dispatch failed — {e}"), false);
        }
    }
}

fn validate_random_pa_larger(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_genomes = 16_usize;
    let n_genes = 100_usize;
    let mut rng = Rng::new(777);
    let pa: Vec<f32> = (0..n_genes * n_genomes)
        .map(|_| if rng.uniform() < 0.5 { 0.0 } else { 1.0 })
        .collect();

    let cpu_mean = cpu_mean_jaccard(&pa, n_genes, n_genomes);

    match gpu_chained_mean_jaccard(gpu, &pa, n_genomes as u32, n_genes as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("random 16×100: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_JACCARD_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("random 16×100: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_genomes = 8_usize;
    let n_genes = 50_usize;
    let mut rng = Rng::new(42);
    let pa: Vec<f32> = (0..n_genes * n_genomes)
        .map(|_| if rng.uniform() < 0.5 { 0.0 } else { 1.0 })
        .collect();

    let r1 = gpu_chained_mean_jaccard(gpu, &pa, n_genomes as u32, n_genes as u32);
    let r2 = gpu_chained_mean_jaccard(gpu, &pa, n_genomes as u32, n_genes as u32);

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
