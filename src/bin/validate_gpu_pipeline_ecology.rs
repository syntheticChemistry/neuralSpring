// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline: spatial payoff → `mean_reduce` → scalar readback (Paper 019).
//!
//! Chains two shader stages in a single `CommandEncoder` with zero CPU round-trips.
//! Stage 1: `spatial_payoff` — PD fitness for entire grid.
//! Stage 2: `mean_reduce` — fitness array to scalar mean.
//!
//! ## Pipeline
//!
//! ```text
//! Upload grid (once)
//!   ↓
//! ┌─────────────────────────────────────────────────────┐
//! │  Stage 1: spatial_payoff.wgsl                       │
//! │    grid[n²] → fitness[n²]                           │
//! │                                                     │
//! │  Stage 2: mean_reduce.wgsl                           │
//! │    fitness[n²] → mean_fitness (scalar)               │
//! └─────────────────────────────────────────────────────┘
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
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const PAYOFF_WGSL: &str = include_str!("../../metalForge/shaders/spatial_payoff.wgsl");
const REDUCE_WGSL: &str = include_str!("../../metalForge/shaders/mean_reduce.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PayoffParams {
    grid_size: u32,
    b_x1000: u32,
    c_x1000: u32,
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

    let mut h = ValidationHarness::new("gpu_pipeline_ecology");

    validate_small_grid(&mut h, &gpu);
    validate_larger_grid(&mut h, &gpu);
    validate_all_cooperators(&mut h, &gpu);
    validate_all_defectors(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_spatial_mean_fitness(grid: &[u32], n: usize, b: f32, c: f32) -> f32 {
    let n_i = n as i32;
    let mut total = 0.0_f32;
    for idx in 0..(n * n) {
        let i = idx / n;
        let j = idx % n;
        let me = grid[idx];
        let mut fit = 0.0_f32;
        for di in -1..=1 {
            for dj in -1..=1 {
                if di == 0 && dj == 0 {
                    continue;
                }
                let ni = (i as i32 + di + n_i).rem_euclid(n_i) as usize;
                let nj = (j as i32 + dj + n_i).rem_euclid(n_i) as usize;
                let other = grid[ni * n + nj];
                if me == 1 && other == 1 {
                    fit += b - c;
                } else if me == 1 && other == 0 {
                    fit -= c;
                } else if me == 0 && other == 1 {
                    fit += b;
                }
            }
        }
        total += fit;
    }
    total / (n * n) as f32
}

// ── GPU chained pipeline ───────────────────────────────────────────

fn gpu_spatial_mean_fitness(
    gpu: &Gpu,
    grid: &[u32],
    n: u32,
    b: f32,
    c: f32,
) -> Result<f32, String> {
    let device = gpu.device();
    let queue = gpu.queue();
    let nn = (n * n) as usize;

    // Shader modules
    let payoff_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_payoff"),
        source: wgpu::ShaderSource::Wgsl(PAYOFF_WGSL.into()),
    });

    let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_reduce"),
        source: wgpu::ShaderSource::Wgsl(REDUCE_WGSL.into()),
    });

    // Payoff bind group layout
    let payoff_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_payoff_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });

    let payoff_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_payoff_pl"),
        bind_group_layouts: &[&payoff_bgl],
        push_constant_ranges: &[],
    });

    let payoff_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_payoff_pipeline"),
        layout: Some(&payoff_pl),
        module: &payoff_shader,
        entry_point: "spatial_payoff",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Reduce bind group layout
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
    let grid_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_grid"),
        contents: bytemuck::cast_slice(grid),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_fitness_out"),
        size: u64::from(n) * u64::from(n) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let payoff_params = PayoffParams {
        grid_size: n,
        b_x1000: (b * 1000.0).round() as u32,
        c_x1000: (c * 1000.0).round() as u32,
        _pad: 0,
    };
    let payoff_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_payoff_params"),
        contents: bytemuck::bytes_of(&payoff_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_mean_result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let reduce_params = ReduceParams { n: n * n };
    let reduce_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_reduce_params"),
        contents: bytemuck::bytes_of(&reduce_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Bind groups
    let payoff_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_payoff_bg"),
        layout: &payoff_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: grid_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: fitness_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: payoff_params_buf.as_entire_binding(),
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

    // Single CommandEncoder — both stages
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("chain_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_payoff_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&payoff_pipeline);
        pass.set_bind_group(0, &payoff_bg, &[]);
        pass.dispatch_workgroups(nn.div_ceil(256) as u32, 1, 1);
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

fn validate_small_grid(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 4_usize;
    let b = 0.5_f32;
    let c = 0.3_f32;
    let mut rng = Rng::new(42);
    let grid: Vec<u32> = (0..n * n).map(|_| u32::from(rng.uniform() > 0.5)).collect();

    let cpu_mean = cpu_spatial_mean_fitness(&grid, n, b, c);

    match gpu_spatial_mean_fitness(gpu, &grid, n as u32, b, c) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("ecology small 4×4: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_SPATIAL_PAYOFF_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("ecology small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_larger_grid(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 16_usize;
    let b = 0.5_f32;
    let c = 0.3_f32;
    let mut rng = Rng::new(777);
    let grid: Vec<u32> = (0..n * n).map(|_| u32::from(rng.uniform() > 0.5)).collect();

    let cpu_mean = cpu_spatial_mean_fitness(&grid, n, b, c);

    match gpu_spatial_mean_fitness(gpu, &grid, n as u32, b, c) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("ecology larger 16×16: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_SPATIAL_PAYOFF_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("ecology larger: dispatch failed — {e}"), false);
        }
    }
}

fn validate_all_cooperators(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 4_usize;
    let b = 0.5_f32;
    let c = 0.3_f32;
    let grid: Vec<u32> = vec![1; n * n];

    let cpu_mean = cpu_spatial_mean_fitness(&grid, n, b, c);

    match gpu_spatial_mean_fitness(gpu, &grid, n as u32, b, c) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("ecology all cooperators: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_SPATIAL_PAYOFF_F32,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("ecology all cooperators: dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_all_defectors(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 4_usize;
    let b = 0.5_f32;
    let c = 0.3_f32;
    let grid: Vec<u32> = vec![0; n * n];

    match gpu_spatial_mean_fitness(gpu, &grid, n as u32, b, c) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("ecology all defectors: mean fitness={gpu_mean:.6} vs 0"),
                f64::from(gpu_mean),
                0.0,
                tolerances::GPU_SPATIAL_PAYOFF_F32,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("ecology all defectors: dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 8_usize;
    let b = 0.5_f32;
    let c = 0.3_f32;
    let mut rng = Rng::new(42);
    let grid: Vec<u32> = (0..n * n).map(|_| u32::from(rng.uniform() > 0.5)).collect();

    let r1 = gpu_spatial_mean_fitness(gpu, &grid, n as u32, b, c);
    let r2 = gpu_spatial_mean_fitness(gpu, &grid, n as u32, b, c);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("ecology determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("ecology determinism: dispatch failed", false);
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
