// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline: `multi_obj_fitness` → `mean_reduce` → scalar readback (Paper 014).
//!
//! Chains two shader stages in a single `CommandEncoder` with zero CPU round-trips.
//! Stage 1: `multi_obj_fitness` — per-chunk mean+0.1*std for each (individual, objective).
//! Stage 2: `mean_reduce` — fitness array to scalar mean.
//!
//! ## Pipeline
//!
//! ```text
//! Upload genotypes [pop_size x genome_len] (once)
//!   ↓
//! ┌──────────────────────────────────────────────────┐
//! │  Stage 1: multi_obj_fitness.wgsl                 │
//! │    genotypes[] → fitness[pop_size * n_objectives] │
//! │                                                  │
//! │  Stage 2: mean_reduce.wgsl                       │
//! │    fitness[pop*n_obj] → mean_fitness (scalar)    │
//! └──────────────────────────────────────────────────┘
//!   ↓  (single queue.submit — NO CPU round-trip)
//! Readback: 4 bytes
//! ```
//!
//! ## Provenance
//!
//! GPU pipeline: multi_obj_fitness → mean_reduce.
//! Validates: end-to-end GPU-resident computation with scalar-only readback.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

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
use neural_spring::directed_evolution::multi_objective_fitness;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const MULTI_OBJ_WGSL: &str = include_str!("../../metalForge/shaders/multi_obj_fitness.wgsl");
const REDUCE_WGSL: &str = include_str!("../../metalForge/shaders/mean_reduce.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MultiObjParams {
    pop_size: u32,
    genome_len: u32,
    n_objectives: u32,
    _pad: u32,
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

    let mut h = ValidationHarness::new("gpu_pipeline_directed");

    validate_single(&mut h, &gpu);
    validate_batch(&mut h, &gpu);
    validate_uniform(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_fitness(genotypes: &[Vec<f64>], n_objectives: usize) -> f64 {
    let fitnesses: Vec<f64> = genotypes
        .iter()
        .flat_map(|g| multi_objective_fitness(g, n_objectives))
        .collect();
    if fitnesses.is_empty() {
        return 0.0;
    }
    fitnesses.iter().sum::<f64>() / fitnesses.len() as f64
}

// ── GPU chained pipeline ───────────────────────────────────────────

fn gpu_multi_obj_mean(
    gpu: &Gpu,
    genotypes: &[f32],
    pop_size: u32,
    genome_len: u32,
    n_objectives: u32,
) -> Result<f32, String> {
    let device = gpu.device();
    let queue = gpu.queue();
    let n_fitness = (pop_size * n_objectives) as usize;

    let multi_obj_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_multi_obj"),
        source: wgpu::ShaderSource::Wgsl(MULTI_OBJ_WGSL.into()),
    });

    let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_reduce"),
        source: wgpu::ShaderSource::Wgsl(REDUCE_WGSL.into()),
    });

    let multi_obj_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_multi_obj_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });

    let multi_obj_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_multi_obj_pl"),
        bind_group_layouts: &[&multi_obj_bgl],
        push_constant_ranges: &[],
    });

    let multi_obj_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_multi_obj_pipeline"),
        layout: Some(&multi_obj_pl),
        module: &multi_obj_shader,
        entry_point: "multi_obj_fitness",
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

    let genotypes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_genotypes"),
        contents: bytemuck::cast_slice(genotypes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_fitness_out"),
        size: (n_fitness * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let multi_obj_params = MultiObjParams {
        pop_size,
        genome_len,
        n_objectives,
        _pad: 0,
    };
    let multi_obj_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_multi_obj_params"),
        contents: bytemuck::bytes_of(&multi_obj_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_mean_result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let reduce_params = ReduceParams {
        n: n_fitness as u32,
    };
    let reduce_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_reduce_params"),
        contents: bytemuck::bytes_of(&reduce_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let multi_obj_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_multi_obj_bg"),
        layout: &multi_obj_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: genotypes_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: fitness_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: multi_obj_params_buf.as_entire_binding(),
            },
        ],
    });

    let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_reduce_bg"),
        layout: &reduce_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: fitness_buf.as_entire_binding(),
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
        label: Some("chain_directed_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_multi_obj_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&multi_obj_pipeline);
        pass.set_bind_group(0, &multi_obj_bg, &[]);
        pass.dispatch_workgroups(n_fitness.div_ceil(256) as u32, 1, 1);
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

fn validate_single(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 1_u32;
    let genome_len = 40_u32;
    let n_objectives = 4_u32;

    let mut rng = Rng::new(42);
    let genotype_f64: Vec<f64> = (0..genome_len as usize).map(|_| rng.uniform()).collect();
    let genotype_f32: Vec<f32> = genotype_f64.iter().map(|&x| x as f32).collect();

    let genotypes: Vec<Vec<f64>> = vec![genotype_f64];
    let cpu_mean = cpu_mean_fitness(&genotypes, n_objectives as usize);

    match gpu_multi_obj_mean(gpu, &genotype_f32, pop_size, genome_len, n_objectives) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("directed single 1×4: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                cpu_mean,
                tolerances::GPU_MULTI_OBJ_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("directed single: dispatch failed — {e}"), false);
        }
    }
}

fn validate_batch(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 10_u32;
    let genome_len = 40_u32;
    let n_objectives = 4_u32;

    let mut rng = Rng::new(77);
    let genotypes_f64: Vec<Vec<f64>> = (0..pop_size as usize)
        .map(|_| (0..genome_len as usize).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_mean = cpu_mean_fitness(&genotypes_f64, n_objectives as usize);

    let genotypes_f32: Vec<f32> = genotypes_f64
        .iter()
        .flat_map(|g| g.iter().map(|&x| x as f32))
        .collect();

    match gpu_multi_obj_mean(gpu, &genotypes_f32, pop_size, genome_len, n_objectives) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("directed batch 10×4: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                cpu_mean,
                tolerances::GPU_MULTI_OBJ_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("directed batch: dispatch failed — {e}"), false);
        }
    }
}

fn validate_uniform(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 1_u32;
    let genome_len = 40_u32;
    let n_objectives = 4_u32;

    let genotypes_f32: Vec<f32> = vec![0.5_f32; (pop_size * genome_len) as usize];

    match gpu_multi_obj_mean(gpu, &genotypes_f32, pop_size, genome_len, n_objectives) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("directed uniform 0.5: mean≈0.5, GPU={gpu_mean:.6}"),
                f64::from(gpu_mean),
                0.5,
                tolerances::GPU_MULTI_OBJ_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("directed uniform: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 5_u32;
    let genome_len = 40_u32;
    let n_objectives = 4_u32;

    let mut rng = Rng::new(123);
    let genotypes_f32: Vec<f32> = (0..(pop_size * genome_len) as usize)
        .map(|_| rng.uniform() as f32)
        .collect();

    let r1 = gpu_multi_obj_mean(gpu, &genotypes_f32, pop_size, genome_len, n_objectives);
    let r2 = gpu_multi_obj_mean(gpu, &genotypes_f32, pop_size, genome_len, n_objectives);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("directed determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("directed determinism: dispatch failed", false);
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
