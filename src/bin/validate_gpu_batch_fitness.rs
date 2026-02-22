// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: batch fitness evaluation via `metalForge` WGSL shader.
//!
//! Validates `metalForge/shaders/batch_fitness_eval.wgsl` against CPU
//! dot-product fitness computation.  The GPU shader evaluates fitness for
//! an entire population in a single dispatch.
//!
//! Evolution path:
//! ```text
//! Python (numpy.dot) → Rust CPU (loop) → BarraCUDA CPU (variance)
//!   → GPU WGSL shader (batch_fitness_eval.wgsl) → ToadStool absorption
//! ```
//!
//! ## Papers validated
//!
//! - Paper 011: Counterdiabatic Evolution (Iram/Dolson, 2020)
//! - Paper 012: MODES Toolbox (Dolson et al., 2019)
//! - Paper 013: Ecological Dynamics (Dolson & Ofria, 2018)
//! - Paper 014: Directed Evolution (Dolson et al., 2022)
//! - Paper 015: Swarm Robotics (Foreback/Dolson, 2025)
//!
//! ## Provenance
//!
//! CPU reference: `directed_evolution::multi_objective_fitness` (linear dot-product).
//! WGSL shader: `metalForge/shaders/batch_fitness_eval.wgsl`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::cast_possible_truncation
)]

use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = include_str!("../../metalForge/shaders/batch_fitness_eval.wgsl");

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

    let mut h = ValidationHarness::new("gpu_batch_fitness");

    validate_small_population(&mut h, &gpu);
    validate_uniform_weights(&mut h, &gpu);
    validate_zero_genotype(&mut h, &gpu);
    validate_larger_population(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

/// CPU reference: dot product fitness for a population.
fn cpu_batch_fitness(
    population: &[f32],
    weights: &[f32],
    pop_size: usize,
    genome_len: usize,
) -> Vec<f32> {
    (0..pop_size)
        .map(|i| {
            let base = i * genome_len;
            (0..genome_len)
                .map(|g| population[base + g] * weights[g])
                .sum()
        })
        .collect()
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FitnessParams {
    pop_size: u32,
    genome_len: u32,
}

/// Run GPU batch fitness evaluation.
fn gpu_batch_fitness(
    gpu: &Gpu,
    population: &[f32],
    weights: &[f32],
    pop_size: u32,
    genome_len: u32,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("batch_fitness_eval"),
        source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fitness_bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fitness_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fitness_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "batch_fitness_linear",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let pop_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("population"),
        contents: bytemuck::cast_slice(population),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let weight_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weights"),
        contents: bytemuck::cast_slice(weights),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fitness"),
        size: u64::from(pop_size) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = FitnessParams {
        pop_size,
        genome_len,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fitness_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pop_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weight_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: fitness_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fitness_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fitness_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(pop_size.div_ceil(256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f32(&fitness_buf, pop_size as usize)
}

fn validate_small_population(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 8_u32;
    let genome_len = 4_u32;
    let mut rng = Rng::new(42);

    let population: Vec<f32> = (0..pop_size * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights: Vec<f32> = (0..genome_len).map(|_| rng.uniform() as f32).collect();

    let cpu = cpu_batch_fitness(
        &population,
        &weights,
        pop_size as usize,
        genome_len as usize,
    );

    match gpu_batch_fitness(gpu, &population, &weights, pop_size, genome_len) {
        Ok(gpu_fitness) => {
            h.check_bool(
                &format!("small pop: correct count ({})", gpu_fitness.len()),
                gpu_fitness.len() == pop_size as usize,
            );

            for (i, (&g, &c)) in gpu_fitness.iter().zip(cpu.iter()).enumerate() {
                h.check_abs(
                    &format!("small pop[{i}]: GPU ≈ CPU ({g:.6} vs {c:.6})"),
                    f64::from(g),
                    f64::from(c),
                    tolerances::GPU_FITNESS_F32,
                );
            }
        }
        Err(e) => {
            h.check_bool(&format!("small pop: dispatch failed — {e}"), false);
        }
    }
}

fn validate_uniform_weights(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 4_u32;
    let genome_len = 8_u32;
    let weights: Vec<f32> = vec![1.0; genome_len as usize];
    let population: Vec<f32> = (0..pop_size * genome_len)
        .map(|i| (i % genome_len) as f32)
        .collect();

    let expected_sum: f32 = (0..genome_len).map(|i| i as f32).sum();

    match gpu_batch_fitness(gpu, &population, &weights, pop_size, genome_len) {
        Ok(gpu_fitness) => {
            for (i, &g) in gpu_fitness.iter().enumerate() {
                h.check_abs(
                    &format!("uniform weights[{i}]: sum={g:.2} vs {expected_sum:.2}"),
                    f64::from(g),
                    f64::from(expected_sum),
                    tolerances::GPU_FITNESS_F32,
                );
            }
        }
        Err(e) => {
            h.check_bool(&format!("uniform weights: failed — {e}"), false);
        }
    }
}

fn validate_zero_genotype(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 4_u32;
    let genome_len = 8_u32;
    let weights: Vec<f32> = vec![1.0; genome_len as usize];
    let population: Vec<f32> = vec![0.0; (pop_size * genome_len) as usize];

    match gpu_batch_fitness(gpu, &population, &weights, pop_size, genome_len) {
        Ok(gpu_fitness) => {
            for (i, &g) in gpu_fitness.iter().enumerate() {
                h.check_abs(
                    &format!("zero genotype[{i}]: fitness={g:.6} vs 0.0"),
                    f64::from(g),
                    0.0,
                    tolerances::GPU_FITNESS_F32,
                );
            }
        }
        Err(e) => {
            h.check_bool(&format!("zero genotype: failed — {e}"), false);
        }
    }
}

fn validate_larger_population(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 512_u32;
    let genome_len = 16_u32;
    let mut rng = Rng::new(777);

    let population: Vec<f32> = (0..pop_size * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights: Vec<f32> = (0..genome_len).map(|_| rng.uniform() as f32).collect();

    let cpu = cpu_batch_fitness(
        &population,
        &weights,
        pop_size as usize,
        genome_len as usize,
    );

    match gpu_batch_fitness(gpu, &population, &weights, pop_size, genome_len) {
        Ok(gpu_fitness) => {
            let max_diff: f64 = gpu_fitness
                .iter()
                .zip(cpu.iter())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
                .fold(0.0_f64, f64::max);

            h.check_bool(
                &format!("512 individuals: max diff {max_diff:.2e} < tol"),
                max_diff < tolerances::GPU_FITNESS_F32,
            );

            h.check_bool(
                &format!("512 individuals: correct count ({})", gpu_fitness.len()),
                gpu_fitness.len() == pop_size as usize,
            );
        }
        Err(e) => {
            h.check_bool(&format!("512 pop: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 32_u32;
    let genome_len = 8_u32;
    let mut rng = Rng::new(42);

    let population: Vec<f32> = (0..pop_size * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights: Vec<f32> = (0..genome_len).map(|_| rng.uniform() as f32).collect();

    let run1 = gpu_batch_fitness(gpu, &population, &weights, pop_size, genome_len);
    let run2 = gpu_batch_fitness(gpu, &population, &weights, pop_size, genome_len);

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
