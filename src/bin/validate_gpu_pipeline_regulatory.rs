// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline: rk4_parallel → `mean_reduce` → scalar readback (Paper 020).
//!
//! Chains two shader stages in a single `CommandEncoder` with zero CPU round-trips.
//! Stage 1: `rk4_step` — RK4 ODE integration for N systems.
//! Stage 2: `mean_reduce` — final state array to scalar mean.
//!
//! ## Pipeline
//!
//! ```text
//! Upload states + coeffs (once)
//!   ↓
//! ┌─────────────────────────────────────────────────────┐
//! │  Stage 1: rk4_parallel.wgsl                         │
//! │    state → state_out (n_steps of RK4)                │
//! │                                                     │
//! │  Stage 2: mean_reduce.wgsl                           │
//! │    state_out[] → mean (scalar)                        │
//! └─────────────────────────────────────────────────────┘
//!   ↓  (single queue.submit — NO CPU round-trip)
//! Readback: 4 bytes (one f32 scalar)
//! ```
//!
//! ## Provenance
//!
//! GPU pipeline: rk4_parallel → mean_reduce.
//! Validates: regulatory network mean final state (Mhatre et al., 2020).

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
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const RK4_WGSL: &str = include_str!("../../metalForge/shaders/rk4_parallel.wgsl");
const REDUCE_WGSL: &str = include_str!("../../metalForge/shaders/mean_reduce.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct OdeParams {
    n_systems: u32,
    dim: u32,
    n_steps: u32,
    dt: f32,
    n_coeffs: u32,
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

    let mut h = ValidationHarness::new("gpu_pipeline_regulatory");

    validate_regulatory_single(&mut h, &gpu);
    validate_regulatory_multi(&mut h, &gpu);
    validate_regulatory_decay(&mut h, &gpu);
    validate_regulatory_steady(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn hill(x: f32, k: f32, n: f32) -> f32 {
    let xn = x.powf(n);
    xn / (k.powf(n) + xn)
}

#[allow(clippy::cast_sign_loss)]
fn cpu_rk4_hill(initial: &[f32], coeffs: &[f32], dim: usize, n_steps: usize, dt: f32) -> Vec<f32> {
    fn deriv(y: &[f32], coeffs: &[f32], dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|d| {
                let c_base = d * 3;
                let prod = coeffs[c_base];
                let deg = coeffs[c_base + 1];
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

fn cpu_mean_final_state(
    initial: &[f32],
    coeffs: &[f32],
    n_systems: usize,
    dim: usize,
    n_steps: usize,
    dt: f32,
) -> f32 {
    let n_total = n_systems * dim;
    let mut total = 0.0_f32;
    for sys in 0..n_systems {
        let start = sys * dim;
        let init = &initial[start..start + dim];
        let sys_coeffs = &coeffs[sys * (dim * 3)..(sys + 1) * (dim * 3)];
        let final_state = cpu_rk4_hill(init, sys_coeffs, dim, n_steps, dt);
        for v in final_state {
            total += v;
        }
    }
    total / n_total as f32
}

// ── GPU chained pipeline ───────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn gpu_mean_rk4(
    gpu: &Gpu,
    states: &[f32],
    coeffs: &[f32],
    n_systems: u32,
    dim: u32,
    n_steps: u32,
    dt: f32,
    n_coeffs: u32,
) -> Result<f32, String> {
    let device = gpu.device();
    let queue = gpu.queue();
    let n_total = (n_systems * dim) as usize;

    let rk4_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_rk4"),
        source: wgpu::ShaderSource::Wgsl(RK4_WGSL.into()),
    });

    let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_regulatory_reduce"),
        source: wgpu::ShaderSource::Wgsl(REDUCE_WGSL.into()),
    });

    let rk4_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_rk4_bgl"),
        entries: &[
            storage_rw_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            uniform_entry(3),
            storage_rw_entry(4),
        ],
    });

    let rk4_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_rk4_pl"),
        bind_group_layouts: &[&rk4_bgl],
        push_constant_ranges: &[],
    });

    let rk4_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_rk4_pipeline"),
        layout: Some(&rk4_pl),
        module: &rk4_shader,
        entry_point: "rk4_step",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_regulatory_reduce_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });

    let reduce_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_regulatory_reduce_pl"),
        bind_group_layouts: &[&reduce_bgl],
        push_constant_ranges: &[],
    });

    let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_regulatory_reduce_pipeline"),
        layout: Some(&reduce_pl),
        module: &reduce_shader,
        entry_point: "mean_reduce",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let state_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_regulatory_state"),
        contents: bytemuck::cast_slice(states),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let coeffs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_regulatory_coeffs"),
        contents: bytemuck::cast_slice(coeffs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let state_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_regulatory_state_out"),
        size: (n_total * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = OdeParams {
        n_systems,
        dim,
        n_steps,
        dt,
        n_coeffs,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_regulatory_ode_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_regulatory_scratch"),
        size: (n_total * 5 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_regulatory_mean_result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let reduce_params = ReduceParams { n: n_total as u32 };
    let reduce_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_regulatory_reduce_params"),
        contents: bytemuck::bytes_of(&reduce_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let rk4_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_rk4_bg"),
        layout: &rk4_bgl,
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
    });

    let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_regulatory_reduce_bg"),
        layout: &reduce_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state_out_buf.as_entire_binding(),
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
        label: Some("chain_regulatory_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_rk4_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&rk4_pipeline);
        pass.set_bind_group(0, &rk4_bg, &[]);
        pass.dispatch_workgroups(n_systems.div_ceil(64), 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_regulatory_reduce_pass"),
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

fn validate_regulatory_single(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 2_u32;
    let initial: Vec<f32> = vec![1.0, 0.5];
    let coeffs: Vec<f32> = vec![
        0.5, 0.1, 1.0, // y0
        0.3, 0.2, 0.0, // y1
    ];
    let n_steps = 50_u32;
    let dt = 0.02_f32;

    let cpu_mean = cpu_mean_final_state(&initial, &coeffs, 1, dim as usize, n_steps as usize, dt);

    match gpu_mean_rk4(gpu, &initial, &coeffs, 1, dim, n_steps, dt, 6) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("regulatory single 2D: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_RK4_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("regulatory single: dispatch failed — {e}"), false);
        }
    }
}

fn validate_regulatory_multi(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_systems = 4_usize;
    let dim = 2_usize;
    let initial: Vec<f32> = vec![1.0, 0.3, 0.8, 0.2, 0.6, 0.5, 0.9, 0.1];
    let coeffs: Vec<f32> = vec![
        0.5, 0.1, 1.0, 0.3, 0.2, 0.0, // sys0
        0.4, 0.15, 1.0, 0.25, 0.25, 0.0, // sys1
        0.6, 0.08, 1.0, 0.35, 0.18, 0.0, // sys2
        0.45, 0.12, 1.0, 0.28, 0.22, 0.0, // sys3
    ];
    let n_steps = 30_u32;
    let dt = 0.01_f32;

    let cpu_mean = cpu_mean_final_state(&initial, &coeffs, n_systems, dim, n_steps as usize, dt);

    match gpu_mean_rk4(
        gpu,
        &initial,
        &coeffs,
        n_systems as u32,
        dim as u32,
        n_steps,
        dt,
        (dim * 3) as u32,
    ) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("regulatory multi 4×2D: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_RK4_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("regulatory multi: dispatch failed — {e}"), false);
        }
    }
}

fn validate_regulatory_decay(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 1_u32;
    let initial: Vec<f32> = vec![2.0];
    let coeffs: Vec<f32> = vec![0.0, 0.5, 0.0]; // prod=0, deg=0.5, no activator
    let n_steps = 100_u32;
    let dt = 0.02_f32;

    let cpu_mean = cpu_mean_final_state(&initial, &coeffs, 1, dim as usize, n_steps as usize, dt);

    match gpu_mean_rk4(gpu, &initial, &coeffs, 1, dim, n_steps, dt, 3) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("regulatory decay: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_RK4_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("regulatory decay: dispatch failed — {e}"), false);
        }
    }
}

fn validate_regulatory_steady(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 2_u32;
    let initial: Vec<f32> = vec![0.5, 0.5];
    let coeffs: Vec<f32> = vec![0.2, 0.2, 1.0, 0.2, 0.2, 0.0];
    let n_steps = 20_u32;
    let dt = 0.05_f32;

    match gpu_mean_rk4(gpu, &initial, &coeffs, 1, dim, n_steps, dt, 6) {
        Ok(gpu_mean) => {
            h.check_bool(
                &format!("regulatory steady: mean={gpu_mean:.6} finite"),
                gpu_mean.is_finite(),
            );
        }
        Err(e) => {
            h.check_bool(&format!("regulatory steady: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 2_u32;
    let initial: Vec<f32> = vec![1.0, 0.4];
    let coeffs: Vec<f32> = vec![0.5, 0.1, 1.0, 0.3, 0.2, 0.0];
    let n_steps = 40_u32;
    let dt = 0.01_f32;

    let r1 = gpu_mean_rk4(gpu, &initial, &coeffs, 1, dim, n_steps, dt, 6);
    let r2 = gpu_mean_rk4(gpu, &initial, &coeffs, 1, dim, n_steps, dt, 6);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("regulatory determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("regulatory determinism: dispatch failed", false);
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
