// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: pure GPU multi-kernel pipeline (zero CPU math).
//!
//! Chains two GPU kernels in a single `queue.submit` to prove that
//! intermediate data stays GPU-resident and no CPU math happens between
//! stages.  This is the "final workload validation" for `BarraCUDA` on
//! pure GPU.
//!
//! ## Pipeline
//!
//! ```text
//! Upload population + weights (once)
//!   ↓
//! ┌─────────────────────────────────────────────────────┐
//! │  Stage 1: batch_fitness_eval.wgsl                   │
//! │    population × weights → fitness[N]                │
//! │                                                     │
//! │  Stage 2: mean_reduce.wgsl                          │
//! │    fitness[N] → mean_fitness (scalar)               │
//! └─────────────────────────────────────────────────────┘
//!   ↓  (single queue.submit — NO CPU round-trip)
//! Readback: 4 bytes (one f32 scalar)
//! ```
//!
//! ## What this proves
//!
//! - Multi-kernel chaining: fitness[] stays GPU-resident between stages
//! - Zero CPU round-trips: single `CommandEncoder`, single submit
//! - Only 4 bytes cross back (scalar mean), not N×4 bytes
//! - Matches CPU reference (dot-product → mean)
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
//! GPU pipeline: `batch_fitness_eval` → `mean_reduce` (multi-kernel chain).
//! Validates: kernel composition with scalar-only readback.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "validation binary"
)]

use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const FITNESS_WGSL: &str = barracuda::ops::bio::batch_fitness::WGSL_BATCH_FITNESS_EVAL;
const REDUCE_WGSL: &str = neural_spring_forge::shaders::MEAN_REDUCE;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FitnessParams {
    pop_size: u32,
    genome_len: u32,
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
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let mut h = ValidationHarness::new("gpu_pure_workload");

    validate_chained_small(&mut h, &gpu);
    validate_chained_larger(&mut h, &gpu);
    validate_zero_population(&mut h, &gpu);
    validate_uniform_weights(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_fitness(
    population: &[f32],
    weights: &[f32],
    pop_size: usize,
    genome_len: usize,
) -> f32 {
    let total: f32 = (0..pop_size)
        .map(|i| {
            let base = i * genome_len;
            (0..genome_len)
                .map(|g| population[base + g] * weights[g])
                .sum::<f32>()
        })
        .sum();
    total / pop_size as f32
}

// ── GPU chained pipeline ───────────────────────────────────────────

fn gpu_chained_mean_fitness(
    gpu: &Gpu,
    population: &[f32],
    weights: &[f32],
    pop_size: u32,
    genome_len: u32,
) -> Result<f32, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    // Stage 1: batch fitness eval
    let fitness_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_fitness"),
        source: wgpu::ShaderSource::Wgsl(FITNESS_WGSL.into()),
    });

    let fitness_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_fitness_bgl"),
        entries: &[
            storage_ro_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            uniform_entry(3),
        ],
    });

    let fitness_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_fitness_pl"),
        bind_group_layouts: &[&fitness_bgl],
        immediate_size: 0,
    });

    let fitness_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_fitness_pipeline"),
        layout: Some(&fitness_pl),
        module: &fitness_shader,
        entry_point: Some("batch_fitness_linear"),
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
        immediate_size: 0,
    });

    let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_reduce_pipeline"),
        layout: Some(&reduce_pl),
        module: &reduce_shader,
        entry_point: Some("mean_reduce"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Buffers
    let pop_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_pop"),
        contents: bytemuck::cast_slice(population),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let weight_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_weights"),
        contents: bytemuck::cast_slice(weights),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_fitness_out"),
        size: u64::from(pop_size) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let fitness_params = FitnessParams {
        pop_size,
        genome_len,
    };
    let fitness_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_fitness_params"),
        contents: bytemuck::bytes_of(&fitness_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_mean_result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let reduce_params = ReduceParams { n: pop_size };
    let reduce_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_reduce_params"),
        contents: bytemuck::bytes_of(&reduce_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Bind groups
    let fitness_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_fitness_bg"),
        layout: &fitness_bgl,
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
                resource: fitness_params_buf.as_entire_binding(),
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

    // Single CommandEncoder — both stages, one submit, zero CPU round-trips
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("chain_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_fitness_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&fitness_pipeline);
        pass.set_bind_group(0, Some(&fitness_bg), &[]);
        pass.dispatch_workgroups(pop_size.div_ceil(256), 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_reduce_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&reduce_pipeline);
        pass.set_bind_group(0, Some(&reduce_bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));

    let result = gpu.read_buffer_f32(&result_buf, 1)?;
    Ok(result[0])
}

// ── Validation functions ───────────────────────────────────────────

fn validate_chained_small(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 8_u32;
    let genome_len = 4_u32;
    let mut rng = Rng::new(42);

    let population: Vec<f32> = (0..pop_size * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights: Vec<f32> = (0..genome_len).map(|_| rng.uniform() as f32).collect();

    let cpu_mean = cpu_mean_fitness(
        &population,
        &weights,
        pop_size as usize,
        genome_len as usize,
    );

    match gpu_chained_mean_fitness(gpu, &population, &weights, pop_size, genome_len) {
        Ok(gpu_mean) => {
            h.check_bool(
                &format!("chain small: GPU mean finite ({gpu_mean:.6})"),
                gpu_mean.is_finite(),
            );
            h.check_abs(
                &format!("chain small: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("chain small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_chained_larger(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 512_u32;
    let genome_len = 16_u32;
    let mut rng = Rng::new(777);

    let population: Vec<f32> = (0..pop_size * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights: Vec<f32> = (0..genome_len).map(|_| rng.uniform() as f32).collect();

    let cpu_mean = cpu_mean_fitness(
        &population,
        &weights,
        pop_size as usize,
        genome_len as usize,
    );

    match gpu_chained_mean_fitness(gpu, &population, &weights, pop_size, genome_len) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("chain 512: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_FITNESS_F32,
            );
            h.check_bool(
                &format!("chain 512: GPU mean finite ({gpu_mean:.6})"),
                gpu_mean.is_finite(),
            );
        }
        Err(e) => {
            h.check_bool(&format!("chain 512: dispatch failed — {e}"), false);
        }
    }
}

fn validate_zero_population(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 4_u32;
    let genome_len = 8_u32;
    let weights: Vec<f32> = vec![1.0; genome_len as usize];
    let population: Vec<f32> = vec![0.0; (pop_size * genome_len) as usize];

    match gpu_chained_mean_fitness(gpu, &population, &weights, pop_size, genome_len) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("chain zero: mean={gpu_mean:.6} vs 0.0"),
                f64::from(gpu_mean),
                0.0,
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("chain zero: dispatch failed — {e}"), false);
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

    let expected_mean: f32 = (0..genome_len).map(|i| i as f32).sum::<f32>();

    match gpu_chained_mean_fitness(gpu, &population, &weights, pop_size, genome_len) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("chain uniform: mean={gpu_mean:.2} vs expected={expected_mean:.2}"),
                f64::from(gpu_mean),
                f64::from(expected_mean),
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("chain uniform: dispatch failed — {e}"), false);
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

    let r1 = gpu_chained_mean_fitness(gpu, &population, &weights, pop_size, genome_len);
    let r2 = gpu_chained_mean_fitness(gpu, &population, &weights, pop_size, genome_len);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("chain determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("chain determinism: dispatch failed", false);
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
