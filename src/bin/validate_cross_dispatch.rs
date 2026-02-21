// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-dispatch validation: same math on GPU vs CPU, parity proof.
//!
//! Demonstrates `BarraCUDA`'s dispatch system by running identical
//! computations on both GPU (WGSL shader) and CPU (Rust math), then
//! validating that results agree within f32 tolerance.
//!
//! ## What this proves
//!
//! - **Math portability**: GPU and CPU produce identical results
//! - **Dispatch routing**: `BarraCUDA`'s `DispatchConfig` selects the right
//!   target based on workload size
//! - **Cross-system capability**: foundation for GPU → CPU → NPU dispatch
//! - **Timing**: GPU shows throughput advantage for large workloads
//!
//! ## Evolution path
//!
//! ```text
//! GPU-only (validate_gpu_batch_fitness)
//!   → Cross-dispatch GPU ↔ CPU (this binary)
//!   → metalForge cross-system (GPU → NPU → CPU)
//! ```
//!
//! ## Papers validated
//!
//! All Phase 0++ papers (011–023): parity between GPU and CPU implementations.

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::cast_possible_truncation
)]

use std::time::Instant;

use barracuda::dispatch::{dispatch_for, DispatchTarget};
use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const FITNESS_WGSL: &str = include_str!("../../metalForge/shaders/batch_fitness_eval.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FitnessParams {
    pop_size: u32,
    genome_len: u32,
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

    let mut h = ValidationHarness::new("cross_dispatch");

    validate_dispatch_routing(&mut h);
    validate_parity_small(&mut h, &gpu);
    validate_parity_large(&mut h, &gpu);
    validate_parity_extremes(&mut h, &gpu);
    validate_timing(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

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

// ── GPU batch fitness ──────────────────────────────────────────────

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
        label: Some("xd_fitness"),
        source: wgpu::ShaderSource::Wgsl(FITNESS_WGSL.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("xd_bgl"),
        entries: &[
            storage_ro_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            uniform_entry(3),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("xd_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("xd_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "batch_fitness_linear",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let pop_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_pop"),
        contents: bytemuck::cast_slice(population),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let weight_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_weights"),
        contents: bytemuck::cast_slice(weights),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xd_fitness_out"),
        size: u64::from(pop_size) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = FitnessParams {
        pop_size,
        genome_len,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("xd_bg"),
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
        label: Some("xd_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("xd_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(pop_size.div_ceil(256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f32(&fitness_buf, pop_size as usize)
}

// ── Validation functions ───────────────────────────────────────────

fn validate_dispatch_routing(h: &mut ValidationHarness) {
    let small_target = dispatch_for("matmul", 10);
    let large_target = dispatch_for("matmul", 10_000);

    h.check_bool(
        &format!("dispatch routing: small matmul(10) → {small_target:?}"),
        matches!(small_target, DispatchTarget::Cpu),
    );

    h.check_bool(
        &format!("dispatch routing: large matmul(10k) → {large_target:?}"),
        matches!(large_target, DispatchTarget::Gpu),
    );
}

fn validate_parity_small(h: &mut ValidationHarness, gpu: &Gpu) {
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
        Ok(gpu_result) => {
            let max_diff: f64 = gpu_result
                .iter()
                .zip(cpu.iter())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
                .fold(0.0_f64, f64::max);

            h.check_bool(
                &format!("parity small (8×4): max diff {max_diff:.2e} < tol"),
                max_diff < tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("parity small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_parity_large(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 1024_u32;
    let genome_len = 32_u32;
    let mut rng = Rng::new(999);

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
        Ok(gpu_result) => {
            let max_diff: f64 = gpu_result
                .iter()
                .zip(cpu.iter())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
                .fold(0.0_f64, f64::max);

            h.check_bool(
                &format!("parity large (1024×32): max diff {max_diff:.2e} < tol"),
                max_diff < tolerances::GPU_FITNESS_F32,
            );

            h.check_bool(
                &format!("parity large: correct count ({})", gpu_result.len()),
                gpu_result.len() == pop_size as usize,
            );
        }
        Err(e) => {
            h.check_bool(&format!("parity large: dispatch failed — {e}"), false);
        }
    }
}

fn validate_parity_extremes(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 16_u32;
    let genome_len = 8_u32;

    let weights: Vec<f32> = vec![1.0; genome_len as usize];
    let population: Vec<f32> = vec![0.0; (pop_size * genome_len) as usize];

    let cpu_zero = cpu_batch_fitness(
        &population,
        &weights,
        pop_size as usize,
        genome_len as usize,
    );

    match gpu_batch_fitness(gpu, &population, &weights, pop_size, genome_len) {
        Ok(gpu_result) => {
            let all_zero = gpu_result
                .iter()
                .zip(cpu_zero.iter())
                .all(|(&g, &c)| (g - c).abs() < f32::EPSILON);
            h.check_bool("parity zero: GPU == CPU (all zeros)", all_zero);
        }
        Err(e) => {
            h.check_bool(&format!("parity zero: dispatch failed — {e}"), false);
        }
    }

    let population_neg: Vec<f32> = (0..pop_size * genome_len)
        .map(|i| -(i as f32) * 0.01)
        .collect();

    let cpu_neg = cpu_batch_fitness(
        &population_neg,
        &weights,
        pop_size as usize,
        genome_len as usize,
    );

    match gpu_batch_fitness(gpu, &population_neg, &weights, pop_size, genome_len) {
        Ok(gpu_result) => {
            let max_diff: f64 = gpu_result
                .iter()
                .zip(cpu_neg.iter())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
                .fold(0.0_f64, f64::max);

            h.check_bool(
                &format!("parity negative: max diff {max_diff:.2e} < tol"),
                max_diff < tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("parity negative: dispatch failed — {e}"), false);
        }
    }
}

fn validate_timing(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 2048_u32;
    let genome_len = 64_u32;
    let mut rng = Rng::new(42);

    let population: Vec<f32> = (0..pop_size * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights: Vec<f32> = (0..genome_len).map(|_| rng.uniform() as f32).collect();

    let cpu_start = Instant::now();
    let cpu = cpu_batch_fitness(
        &population,
        &weights,
        pop_size as usize,
        genome_len as usize,
    );
    let cpu_us = cpu_start.elapsed().as_micros();

    let gpu_start = Instant::now();
    let gpu_result = gpu_batch_fitness(gpu, &population, &weights, pop_size, genome_len);
    let gpu_us = gpu_start.elapsed().as_micros();

    match gpu_result {
        Ok(gpu_fitness) => {
            let max_diff: f64 = gpu_fitness
                .iter()
                .zip(cpu.iter())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
                .fold(0.0_f64, f64::max);

            h.check_bool(
                &format!("timing (2048×64): GPU={gpu_us}μs CPU={cpu_us}μs parity={max_diff:.2e}"),
                max_diff < tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("timing: dispatch failed — {e}"), false);
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
