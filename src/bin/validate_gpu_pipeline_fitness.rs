// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline: batch fitness → `mean_reduce` → scalar readback (Paper 011).
//!
//! Chains two shader stages in a single `CommandEncoder` with zero CPU round-trips.
//! Stage 1: `batch_fitness_linear` — genotype · weights for entire population.
//! Stage 2: `mean_reduce` — fitness array to scalar mean.
//!
//! ## Pipeline
//!
//! ```text
//! Upload population + weights (once)
//!   ↓
//! ┌─────────────────────────────────────────────────────┐
//! │  Stage 1: batch_fitness_eval.wgsl                    │
//! │    genotypes × weights → fitness[pop_size]           │
//! │                                                     │
//! │  Stage 2: mean_reduce.wgsl                           │
//! │    fitness[pop_size] → mean_fitness (scalar)          │
//! └─────────────────────────────────────────────────────┘
//!   ↓  (single queue.submit — NO CPU round-trip)
//! Readback: 4 bytes (one f32 scalar)
//! ```
//!
//! ## Provenance
//!
//! GPU pipeline: batch_fitness_eval → mean_reduce.
//! Validates: end-to-end GPU-resident computation for Paper 011 (Counterdiabatic).

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
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const FITNESS_WGSL: &str = include_str!("../../metalForge/shaders/batch_fitness_eval.wgsl");
const REDUCE_WGSL: &str = include_str!("../../metalForge/shaders/mean_reduce.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FitnessParams {
    pop_size: u32,
    genome_len: u32,
    _pad0: u32,
    _pad1: u32,
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

    let mut h = ValidationHarness::new("gpu_pipeline_fitness");

    validate_small_population(&mut h, &gpu);
    validate_larger_population(&mut h, &gpu);
    validate_uniform_weights(&mut h, &gpu);
    validate_zero_genotype(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_fitness(
    genotypes: &[f32],
    weights: &[f32],
    pop_size: usize,
    genome_len: usize,
) -> f32 {
    let total: f32 = (0..pop_size)
        .map(|i| {
            let base = i * genome_len;
            (0..genome_len)
                .map(|g| genotypes[base + g] * weights[g])
                .sum::<f32>()
        })
        .sum();
    total / pop_size as f32
}

// ── GPU chained pipeline ───────────────────────────────────────────

fn gpu_mean_fitness(
    gpu: &Gpu,
    genotypes: &[f32],
    weights: &[f32],
    pop_size: u32,
    genome_len: u32,
) -> Result<f32, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let fitness_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_fitness"),
        source: wgpu::ShaderSource::Wgsl(FITNESS_WGSL.into()),
    });

    let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_reduce"),
        source: wgpu::ShaderSource::Wgsl(REDUCE_WGSL.into()),
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
        push_constant_ranges: &[],
    });

    let fitness_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_fitness_pipeline"),
        layout: Some(&fitness_pl),
        module: &fitness_shader,
        entry_point: "batch_fitness_linear",
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

    let geno_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_genotypes"),
        contents: bytemuck::cast_slice(genotypes),
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
        _pad0: 0,
        _pad1: 0,
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

    let fitness_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_fitness_bg"),
        layout: &fitness_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: geno_buf.as_entire_binding(),
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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("chain_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_fitness_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&fitness_pipeline);
        pass.set_bind_group(0, &fitness_bg, &[]);
        pass.dispatch_workgroups(pop_size.div_ceil(256), 1, 1);
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

fn validate_small_population(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 8_usize;
    let genome_len = 16_usize;
    let mut rng = Rng::new(42);
    let genotypes: Vec<f32> = (0..pop_size * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights: Vec<f32> = (0..genome_len).map(|_| rng.uniform() as f32).collect();

    let cpu_mean = cpu_mean_fitness(&genotypes, &weights, pop_size, genome_len);

    match gpu_mean_fitness(gpu, &genotypes, &weights, pop_size as u32, genome_len as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("fitness small 8×16: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("fitness small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_larger_population(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 64_usize;
    let genome_len = 32_usize;
    let mut rng = Rng::new(777);
    let genotypes: Vec<f32> = (0..pop_size * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights: Vec<f32> = (0..genome_len).map(|_| rng.uniform() as f32).collect();

    let cpu_mean = cpu_mean_fitness(&genotypes, &weights, pop_size, genome_len);

    match gpu_mean_fitness(gpu, &genotypes, &weights, pop_size as u32, genome_len as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("fitness larger 64×32: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("fitness larger: dispatch failed — {e}"), false);
        }
    }
}

fn validate_uniform_weights(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 16_usize;
    let genome_len = 8_usize;
    let mut rng = Rng::new(123);
    let genotypes: Vec<f32> = (0..pop_size * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights: Vec<f32> = vec![1.0; genome_len];

    let cpu_mean = cpu_mean_fitness(&genotypes, &weights, pop_size, genome_len);

    match gpu_mean_fitness(gpu, &genotypes, &weights, pop_size as u32, genome_len as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("fitness uniform weights: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("fitness uniform weights: dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_zero_genotype(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 4_usize;
    let genome_len = 4_usize;
    let genotypes: Vec<f32> = vec![0.0; pop_size * genome_len];
    let weights: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    match gpu_mean_fitness(gpu, &genotypes, &weights, pop_size as u32, genome_len as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("fitness zero genotype: mean={gpu_mean:.6} vs 0"),
                f64::from(gpu_mean),
                0.0,
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("fitness zero genotype: dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 32_usize;
    let genome_len = 12_usize;
    let mut rng = Rng::new(99);
    let genotypes: Vec<f32> = (0..pop_size * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights: Vec<f32> = (0..genome_len).map(|_| rng.uniform() as f32).collect();

    let r1 = gpu_mean_fitness(gpu, &genotypes, &weights, pop_size as u32, genome_len as u32);
    let r2 = gpu_mean_fitness(gpu, &genotypes, &weights, pop_size as u32, genome_len as u32);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("fitness determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("fitness determinism: dispatch failed", false);
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
