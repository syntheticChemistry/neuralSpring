// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: multi-objective fitness via `metalForge` WGSL shader.
//!
//! Validates `metalForge/shaders/multi_obj_fitness.wgsl` against CPU
//! `directed_evolution::multi_objective_fitness`. The GPU shader computes
//! per-chunk mean + 0.1*std for each (individual, objective) pair.
//!
//! ## Papers validated
//!
//! - Paper 014: Directed Evolution (multi-objective fitness)
//!
//! ## Provenance
//!
//! CPU reference: `directed_evolution::multi_objective_fitness` (seed=42, `pop_size`=10 `n_objectives`=4).
//! WGSL shader: `metalForge/shaders/multi_obj_fitness.wgsl`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::similar_names
)]

use neural_spring::directed_evolution::multi_objective_fitness;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = include_str!("../../metalForge/shaders/multi_obj_fitness.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    pop_size: u32,
    genome_len: u32,
    n_objectives: u32,
    _pad: u32,
}

fn gpu_multi_obj_fitness(
    gpu: &Gpu,
    genotypes: &[f32],
    pop_size: u32,
    genome_len: u32,
    n_objectives: u32,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("multi_obj_fitness"),
        source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("multi_obj_fitness_bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("multi_obj_fitness_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("multi_obj_fitness_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "multi_obj_fitness",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let genotypes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("genotypes"),
        contents: bytemuck::cast_slice(genotypes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let n_fitness = (pop_size * n_objectives) as usize;
    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fitness"),
        size: (n_fitness * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = Params {
        pop_size,
        genome_len,
        n_objectives,
        _pad: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("multi_obj_fitness_bg"),
        layout: &bgl,
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
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let workgroups = (pop_size * n_objectives).div_ceil(256);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("multi_obj_fitness_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("multi_obj_fitness_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f32(&fitness_buf, n_fitness)
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

    let mut h = ValidationHarness::new("gpu_directed");

    validate_single_genotype(&mut h, &gpu);
    validate_batch(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);
    validate_uniform_genotype(&mut h, &gpu);

    h.finish();
}

fn validate_single_genotype(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 1_u32;
    let genome_len = 40_u32;
    let n_objectives = 4_u32;

    let mut rng = Rng::new(42);
    let genotype_f64: Vec<f64> = (0..genome_len as usize).map(|_| rng.uniform()).collect();
    let genotype_f32: Vec<f32> = genotype_f64.iter().map(|&x| x as f32).collect();

    let cpu_fitness = multi_objective_fitness(&genotype_f64, n_objectives as usize);

    match gpu_multi_obj_fitness(gpu, &genotype_f32, pop_size, genome_len, n_objectives) {
        Ok(gpu_fitness) => {
            h.check_bool(
                &format!("single genotype: correct count ({})", gpu_fitness.len()),
                gpu_fitness.len() == n_objectives as usize,
            );

            let max_diff: f64 = gpu_fitness
                .iter()
                .zip(cpu_fitness.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("single genotype: max GPU-CPU diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_MULTI_OBJ_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("single genotype: dispatch failed — {e}"), false);
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

    let cpu_fitness: Vec<f64> = genotypes_f64
        .iter()
        .flat_map(|g| multi_objective_fitness(g, n_objectives as usize))
        .collect();

    let genotypes_f32: Vec<f32> = genotypes_f64
        .iter()
        .flat_map(|g| g.iter().map(|&x| x as f32))
        .collect();

    match gpu_multi_obj_fitness(gpu, &genotypes_f32, pop_size, genome_len, n_objectives) {
        Ok(gpu_fitness) => {
            h.check_bool(
                &format!("batch: correct count ({})", gpu_fitness.len()),
                gpu_fitness.len() == (pop_size * n_objectives) as usize,
            );

            let max_diff: f64 = gpu_fitness
                .iter()
                .zip(cpu_fitness.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("batch: max GPU-CPU diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_MULTI_OBJ_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("batch: dispatch failed — {e}"), false);
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

    let run1 = gpu_multi_obj_fitness(gpu, &genotypes_f32, pop_size, genome_len, n_objectives);
    let run2 = gpu_multi_obj_fitness(gpu, &genotypes_f32, pop_size, genome_len, n_objectives);

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

fn validate_uniform_genotype(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 1_u32;
    let genome_len = 40_u32;
    let n_objectives = 4_u32;

    let genotypes_f32: Vec<f32> = vec![0.5_f32; (pop_size * genome_len) as usize];

    match gpu_multi_obj_fitness(gpu, &genotypes_f32, pop_size, genome_len, n_objectives) {
        Ok(gpu_fitness) => {
            let expected = 0.5_f32;
            let max_diff: f64 = gpu_fitness
                .iter()
                .map(|&g| (f64::from(g) - f64::from(expected)).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("uniform genotype 0.5: all fitness≈0.5 (max diff {max_diff:.2e})"),
                max_diff,
                tolerances::GPU_MULTI_OBJ_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("uniform genotype: dispatch failed — {e}"), false);
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
