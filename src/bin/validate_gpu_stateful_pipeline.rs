// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: `StatefulPipeline`-driven iterative ODE integration.
//!
//! Demonstrates `BarraCUDA`'s `StatefulPipeline` for GPU-resident iterative
//! computation.  The RK4 ODE integrator runs entirely on GPU — state never
//! leaves the device between iterations.  Only a tiny convergence scalar
//! (8 bytes) crosses back to the CPU per readback window.
//!
//! ## What this proves
//!
//! - GPU-resident state across 100+ iterations (zero full-state readback)
//! - Single `queue.submit` per iteration batch
//! - Final state matches CPU RK4 reference within tolerance
//! - `StatefulPipeline` API works with `neuralSpring` shaders
//!
//! ## Evolution path
//!
//! ```text
//! Manual encoder loop (validate_gpu_rk4)
//!   → StatefulPipeline (BarraCUDA staging)
//!   → `BarraCUDA` absorption
//! ```
//!
//! ## Papers validated
//!
//! - Paper 020: Regulatory Network (Mhatre et al., 2020)
//! - Paper 021: Signal Integration (Srivastava et al., 2011)
//!
//! ## Provenance
//!
//! GPU pipeline: `StatefulPipeline` API (`rk4_parallel` iteration).
//! Validates: end-to-end GPU-resident computation with scalar-only readback.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "validation binary"
)]

use std::sync::Arc;

use barracuda::staging::{KernelDispatch, StatefulConfig, StatefulPipeline};
use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = neural_spring_forge::shaders::RK4_PARALLEL;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct OdeParams {
    n_systems: u32,
    dim: u32,
    n_steps: u32,
    dt: f32,
    n_coeffs: u32,
}

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            println!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let mut h = ValidationHarness::new("gpu_stateful_pipeline");

    validate_single_system(&mut h, &gpu);
    validate_multi_system(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);
    validate_batched_convergence(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_rk4_hill(initial: &[f32], coeffs: &[f32], dim: usize, n_steps: usize, dt: f32) -> Vec<f32> {
    fn hill(x: f32, k: f32, n: f32) -> f32 {
        let xn = x.powf(n);
        xn / (k.powf(n) + xn)
    }

    fn deriv(y: &[f32], coeffs: &[f32], dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|d| {
                let c_base = d * 3;
                let prod = coeffs[c_base];
                let deg = coeffs[c_base + 1];
                #[expect(clippy::cast_sign_loss, reason = "validation binary")]
                let act_idx = coeffs[c_base + 2] as usize;
                prod.mul_add(hill(y[act_idx], 0.5, 2.0), -(deg * y[d]))
            })
            .collect()
    }

    let half_dt = 0.5 * dt;
    let sixth_dt = dt / 6.0;
    let mut y: Vec<f32> = initial.to_vec();
    for _ in 0..n_steps {
        let k1 = deriv(&y, coeffs, dim);
        let y2: Vec<f32> = y
            .iter()
            .zip(k1.iter())
            .map(|(&yi, &ki)| half_dt.mul_add(ki, yi))
            .collect();
        let k2 = deriv(&y2, coeffs, dim);
        let y3: Vec<f32> = y
            .iter()
            .zip(k2.iter())
            .map(|(&yi, &ki)| half_dt.mul_add(ki, yi))
            .collect();
        let k3 = deriv(&y3, coeffs, dim);
        let y4: Vec<f32> = y
            .iter()
            .zip(k3.iter())
            .map(|(&yi, &ki)| dt.mul_add(ki, yi))
            .collect();
        let k4 = deriv(&y4, coeffs, dim);
        for d in 0..dim {
            let weighted = 2.0f32.mul_add(k2[d], k1[d]) + 2.0f32.mul_add(k3[d], k4[d]);
            y[d] = sixth_dt.mul_add(weighted, y[d]);
        }
    }
    y
}

// ── GPU setup via StatefulPipeline ─────────────────────────────────

struct StatefulRk4 {
    pipeline: StatefulPipeline,
    chain: Vec<KernelDispatch>,
    state_buf: wgpu::Buffer,
    convergence_buf: wgpu::Buffer,
}

fn setup_stateful_rk4(
    gpu: &Gpu,
    states: &[f32],
    coeffs: &[f32],
    n_systems: u32,
    dim: u32,
    dt: f32,
    n_coeffs: u32,
) -> StatefulRk4 {
    let device = gpu.device();
    let wgpu_dev = gpu.wgpu_device();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rk4_stateful"),
        source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rk4_sp_bgl"),
        entries: &[
            storage_rw_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            uniform_entry(3),
            storage_rw_entry(4),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rk4_sp_pl"),
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });

    let pipeline = Arc::new(
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rk4_sp_pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("rk4_step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        }),
    );

    let state_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sp_state"),
        contents: bytemuck::cast_slice(states),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let coeffs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sp_coeffs"),
        contents: bytemuck::cast_slice(coeffs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_size = u64::from(n_systems * dim) * 4;
    let state_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sp_state_out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = OdeParams {
        n_systems,
        dim,
        n_steps: 1,
        dt,
        n_coeffs,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sp_ode_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let scratch_size = u64::from(n_systems * dim * 5) * 4;
    let scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sp_scratch"),
        size: scratch_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let convergence_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sp_convergence"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bg = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rk4_sp_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: coeffs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state_out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: scratch_buf.as_entire_binding(),
            },
        ],
    }));

    let kernel = KernelDispatch::new(pipeline, bg, (n_systems.div_ceil(64), 1, 1));

    let sp = StatefulPipeline::new(
        Arc::clone(wgpu_dev),
        StatefulConfig {
            convergence_scalars: 1,
            label: Some("RK4-stateful".into()),
        },
    );

    StatefulRk4 {
        pipeline: sp,
        chain: vec![kernel],
        state_buf,
        convergence_buf,
    }
}

// ── Validation functions ───────────────────────────────────────────

fn validate_single_system(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 2_u32;
    let initial: Vec<f32> = vec![1.0, 0.5];
    let coeffs: Vec<f32> = vec![0.5, 0.1, 1.0, 0.3, 0.2, 0.0];
    let n_iters = 100_usize;
    let dt = 0.01_f32;

    let cpu_result = cpu_rk4_hill(&initial, &coeffs, dim as usize, n_iters, dt);

    let sp_rk4 = setup_stateful_rk4(gpu, &initial, &coeffs, 1, dim, dt, coeffs.len() as u32);

    match sp_rk4
        .pipeline
        .run_iterations(&sp_rk4.chain, &sp_rk4.convergence_buf, n_iters)
    {
        Ok(scalars) => {
            h.check_bool(
                "SP single: pipeline returned convergence scalar",
                !scalars.is_empty(),
            );

            match gpu.read_buffer_f32(&sp_rk4.state_buf, dim as usize) {
                Ok(gpu_state) => {
                    for (d, (&g, &c)) in gpu_state.iter().zip(cpu_result.iter()).enumerate() {
                        h.check_abs(
                            &format!("SP single y[{d}]: GPU={g:.6} vs CPU={c:.6}"),
                            f64::from(g),
                            f64::from(c),
                            tolerances::GPU_RK4_F32,
                        );
                    }
                }
                Err(e) => {
                    h.check_bool(&format!("SP single: readback failed — {e}"), false);
                }
            }
        }
        Err(e) => {
            h.check_bool(&format!("SP single: pipeline failed — {e}"), false);
        }
    }
}

fn validate_multi_system(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 2_u32;
    let n_systems = 4_u32;
    let n_iters = 50_usize;
    let dt = 0.01_f32;
    let coeffs_template: Vec<f32> = vec![0.5, 0.1, 1.0, 0.3, 0.2, 0.0];

    let mut states: Vec<f32> = Vec::new();
    let mut all_coeffs: Vec<f32> = Vec::new();
    for i in 0..n_systems {
        let fi = i as f32;
        states.push(0.25f32.mul_add(fi, 0.5));
        states.push(0.1f32.mul_add(fi, 0.3));
        all_coeffs.extend_from_slice(&coeffs_template);
    }

    let sp_rk4 = setup_stateful_rk4(
        gpu,
        &states,
        &all_coeffs,
        n_systems,
        dim,
        dt,
        coeffs_template.len() as u32,
    );

    match sp_rk4
        .pipeline
        .run_iterations(&sp_rk4.chain, &sp_rk4.convergence_buf, n_iters)
    {
        Ok(_) => match gpu.read_buffer_f32(&sp_rk4.state_buf, (n_systems * dim) as usize) {
            Ok(gpu_state) => {
                let mut all_match = true;
                for sys in 0..n_systems as usize {
                    let fs = sys as f32;
                    let sys_init = &[0.25f32.mul_add(fs, 0.5), 0.1f32.mul_add(fs, 0.3)];
                    let cpu = cpu_rk4_hill(sys_init, &coeffs_template, dim as usize, n_iters, dt);
                    for d in 0..dim as usize {
                        let g = gpu_state[sys * dim as usize + d];
                        let c = cpu[d];
                        if (f64::from(g) - f64::from(c)).abs() > tolerances::GPU_RK4_F32 {
                            all_match = false;
                        }
                    }
                }
                h.check_bool(
                    &format!("SP multi: {n_systems} systems all match CPU"),
                    all_match,
                );
                h.check_bool(
                    &format!("SP multi: correct count ({})", gpu_state.len()),
                    gpu_state.len() == (n_systems * dim) as usize,
                );
                let all_finite = gpu_state.iter().all(|v| v.is_finite());
                h.check_bool("SP multi: all values finite", all_finite);
            }
            Err(e) => {
                h.check_bool(&format!("SP multi: readback failed — {e}"), false);
            }
        },
        Err(e) => {
            h.check_bool(&format!("SP multi: pipeline failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 2_u32;
    let initial: Vec<f32> = vec![1.0, 0.5];
    let coeffs: Vec<f32> = vec![0.5, 0.1, 1.0, 0.3, 0.2, 0.0];
    let n_iters = 50_usize;
    let dt = 0.01_f32;
    let n_coeffs = coeffs.len() as u32;

    let run = || -> Option<Vec<f32>> {
        let sp = setup_stateful_rk4(gpu, &initial, &coeffs, 1, dim, dt, n_coeffs);
        sp.pipeline
            .run_iterations(&sp.chain, &sp.convergence_buf, n_iters)
            .ok()?;
        gpu.read_buffer_f32(&sp.state_buf, dim as usize).ok()
    };

    match (run(), run()) {
        (Some(r1), Some(r2)) => {
            let identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(&a, &b)| (a - b).abs() < f32::EPSILON);
            h.check_bool("SP determinism: two runs identical", identical);
        }
        _ => {
            h.check_bool("SP determinism: dispatch failed", false);
        }
    }
}

fn validate_batched_convergence(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 2_u32;
    let initial: Vec<f32> = vec![1.0, 0.5];
    let coeffs: Vec<f32> = vec![0.5, 0.5, 1.0, 0.5, 0.5, 0.0];
    let dt = 0.01_f32;
    let n_coeffs = coeffs.len() as u32;

    let sp_rk4 = setup_stateful_rk4(gpu, &initial, &coeffs, 1, dim, dt, n_coeffs);

    match sp_rk4
        .pipeline
        .run_iterations(&sp_rk4.chain, &sp_rk4.convergence_buf, 200)
    {
        Ok(_) => match gpu.read_buffer_f32(&sp_rk4.state_buf, dim as usize) {
            Ok(gpu_state) => {
                let all_finite = gpu_state.iter().all(|v| v.is_finite());
                h.check_bool(
                    "SP convergence: all values finite after 200 iters",
                    all_finite,
                );
                for (d, &g) in gpu_state.iter().enumerate() {
                    h.check_bool(
                        &format!("SP convergence y[{d}]={g:.4} (near steady state)"),
                        (g - 1.0).abs() < tolerances::ODE_STEADY_STATE_SLACK as f32,
                    );
                }
            }
            Err(e) => {
                h.check_bool(&format!("SP convergence: readback failed — {e}"), false);
            }
        },
        Err(e) => {
            h.check_bool(&format!("SP convergence: pipeline failed — {e}"), false);
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
