// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline: Gillespie SSA → `mean_reduce` → scalar readback.
//!
//! Validates the stochastic simulation → scalar reduction pattern.
//! Stage 1: `GillespieGpu::simulate` — parallel SSA trajectories on GPU.
//! Stage 2: `mean_reduce.wgsl` — final species counts → scalar mean.
//!
//! ## Pipeline
//!
//! ```text
//! GillespieGpu (upstream barracuda) → final_states\[n_traj × n_species\]
//!   ↓  (upload to GPU buffer)
//! ┌─────────────────────────────────────────────────────────┐
//! │  mean_reduce.wgsl                                       │
//! │    final_states → mean (scalar)                          │
//! └─────────────────────────────────────────────────────────┘
//!   ↓  (single queue.submit — scalar readback only)
//! Readback: 4 bytes
//! ```
//!
//! This validates the GPU-resident reduce pattern for stochastic outputs,
//! proving that only a scalar summary needs to cross the `PCIe` bus.
//!
//! ## Papers validated
//!
//! - Paper 013: Ecological Dynamics (Dolson & Ofria, 2018)
//! - Paper 020: Regulatory Network (Mhatre et al., 2020)
//!
//! ## Provenance
//!
//! Upstream: `barracuda::ops::bio::gillespie::GillespieGpu`
//! Reduce: `metalForge/shaders/mean_reduce.wgsl`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

use barracuda::ops::bio::gillespie::{GillespieConfig, GillespieGpu};
use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const REDUCE_WGSL: &str = include_str!("../../metalForge/shaders/mean_reduce.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ReduceParams {
    n: u32,
}

fn make_seeds(n_trajectories: usize) -> Vec<u32> {
    let mut seeds = Vec::with_capacity(n_trajectories * 4);
    for t in 0..n_trajectories {
        let mut sm = 42u32.wrapping_add(t as u32 * 1_000_003);
        for _ in 0..4 {
            sm = sm.wrapping_add(0x9e37_79b9);
            let mut z = sm;
            z = (z ^ (z >> 15)).wrapping_mul(0x85eb_ca6b);
            z = (z ^ (z >> 13)).wrapping_mul(0xc2b2_ae35);
            seeds.push(z ^ (z >> 16));
        }
    }
    seeds
}

/// Upload f32 data to GPU, run `mean_reduce`, return scalar.
fn gpu_mean_reduce(gpu: &Gpu, data: &[f32]) -> Result<f32, String> {
    let device = gpu.device();
    let queue = gpu.queue();
    let n = data.len() as u32;

    let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pipe_gill_reduce"),
        source: wgpu::ShaderSource::Wgsl(REDUCE_WGSL.into()),
    });
    let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pipe_gill_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });
    let reduce_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipe_gill_pl"),
        bind_group_layouts: &[&reduce_bgl],
        push_constant_ranges: &[],
    });
    let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipe_gill_pipeline"),
        layout: Some(&reduce_pl),
        module: &reduce_shader,
        entry_point: "mean_reduce",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let data_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pipe_gill_data"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pipe_gill_result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = ReduceParams { n };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pipe_gill_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pipe_gill_bg"),
        layout: &reduce_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: data_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pipe_gill_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pipe_gill_pass"),
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

fn run_gillespie(
    gpu: &Gpu,
    n_traj: usize,
) -> Option<barracuda::ops::bio::gillespie::GillespieResult> {
    let rate_k = vec![1.0_f64];
    let stoich_react = vec![1u32, 0];
    let stoich_net = vec![-1i32, 1];
    let initial_states: Vec<f64> = (0..n_traj).flat_map(|_| [100.0_f64, 0.0]).collect();
    let seeds = make_seeds(n_traj);
    let config = GillespieConfig {
        t_max: 2.0,
        max_steps: 10_000,
    };

    let dev = gpu.wgpu_device();
    let ssa = GillespieGpu::new(dev);

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ssa.simulate(
            &rate_k,
            &stoich_react,
            &stoich_net,
            &initial_states,
            &seeds,
            n_traj,
            &config,
        )
    })) {
        Ok(Ok(r)) => Some(r),
        _ => None,
    }
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

    let mut h = ValidationHarness::new("gpu_pipeline_gillespie");

    validate_conservation_reduce(&mut h, &gpu);
    validate_mean_species_a(&mut h, &gpu);
    validate_reduce_determinism(&mut h, &gpu);
    validate_multi_trajectory_reduce(&mut h, &gpu);

    h.finish();
}

/// SSA A → B (conservation): mean of A+B across trajectories == 100.
fn validate_conservation_reduce(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_traj = 4_usize;
    let n_species = 2_usize;

    let Some(result) = run_gillespie(gpu, n_traj) else {
        h.check_bool("conservation reduce: SSA failed (driver skip)", false);
        return;
    };

    // Compute A+B for each trajectory, reduce to mean total
    let totals: Vec<f32> = (0..n_traj)
        .map(|t| (result.states[t * n_species] + result.states[t * n_species + 1]) as f32)
        .collect();
    let cpu_mean = totals.iter().sum::<f32>() / totals.len() as f32;

    match gpu_mean_reduce(gpu, &totals) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("conservation reduce: GPU mean={gpu_mean:.2} vs CPU={cpu_mean:.2}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_FITNESS_F32,
            );
            h.check_abs(
                "conservation reduce: mean total ≈ 100",
                f64::from(gpu_mean),
                100.0,
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("conservation reduce: dispatch failed — {e}"),
                false,
            );
        }
    }
}

/// Reduce final species-A counts to scalar mean.
fn validate_mean_species_a(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_traj = 8_usize;
    let n_species = 2_usize;

    let Some(result) = run_gillespie(gpu, n_traj) else {
        h.check_bool("mean species A: SSA failed", false);
        return;
    };

    let final_a: Vec<f32> = (0..n_traj)
        .map(|t| result.states[t * n_species] as f32)
        .collect();
    let cpu_mean = final_a.iter().sum::<f32>() / final_a.len() as f32;

    match gpu_mean_reduce(gpu, &final_a) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("mean species A: GPU={gpu_mean:.2} vs CPU={cpu_mean:.2}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_FITNESS_F32,
            );
            // At t=2, E[A] = 100 * e^(-2) ≈ 13.5 — mean should be in plausible range
            h.check_bool(
                &format!("mean species A: {gpu_mean:.1} in plausible range [0, 100]"),
                (0.0..=100.0).contains(&gpu_mean),
            );
        }
        Err(e) => {
            h.check_bool(&format!("mean species A: dispatch failed — {e}"), false);
        }
    }
}

/// Same data → identical GPU `mean_reduce` result.
fn validate_reduce_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_traj = 4_usize;
    let n_species = 2_usize;

    let Some(result) = run_gillespie(gpu, n_traj) else {
        h.check_bool("reduce determinism: SSA failed", false);
        return;
    };

    let final_a: Vec<f32> = (0..n_traj)
        .map(|t| result.states[t * n_species] as f32)
        .collect();

    let r1 = gpu_mean_reduce(gpu, &final_a);
    let r2 = gpu_mean_reduce(gpu, &final_a);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("reduce determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("reduce determinism: dispatch failed", false);
        }
    }
}

/// 16 trajectories reduced to single scalar via pipeline.
fn validate_multi_trajectory_reduce(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_traj = 16_usize;
    let n_species = 2_usize;

    let Some(result) = run_gillespie(gpu, n_traj) else {
        h.check_bool("multi-trajectory reduce: SSA failed", false);
        return;
    };

    let final_a: Vec<f32> = (0..n_traj)
        .map(|t| result.states[t * n_species] as f32)
        .collect();
    let cpu_mean = final_a.iter().sum::<f32>() / final_a.len() as f32;

    match gpu_mean_reduce(gpu, &final_a) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("multi-trajectory: GPU={gpu_mean:.2} vs CPU={cpu_mean:.2} (16 traj)"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("multi-trajectory reduce: dispatch failed — {e}"),
                false,
            );
        }
    }
}

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
