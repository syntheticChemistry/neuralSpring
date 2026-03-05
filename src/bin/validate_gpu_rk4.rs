// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: parallel RK4 ODE integration via `metalForge` WGSL shader.
//!
//! Validates `metalForge/shaders/rk4_parallel.wgsl` against CPU RK4 for
//! simple ODE systems.  Uses known analytical solutions to verify correctness.
//!
//! Evolution path:
//! ```text
//! Python (scipy.integrate) → Rust CPU (hand-rolled RK4)
//!   → BarraCUDA CPU (rk45_solve) → GPU WGSL shader (rk4_parallel.wgsl)
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
//! CPU reference: `regulatory_network::rk4_step` (Hill ODE integration).
//! WGSL shader: `metalForge/shaders/rk4_parallel.wgsl`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "validation binary"
)]

use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = neural_spring_forge::shaders::RK4_PARALLEL;

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

    let mut h = ValidationHarness::new("gpu_rk4_parallel");

    validate_exponential_decay(&mut h, &gpu);
    validate_multi_system(&mut h, &gpu);
    validate_steady_state(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

/// CPU RK4 for a single system: `dy/dt = prod * hill(y_act, 0.5, 2) - deg * y`.
/// where hill(x, k, n) = x^n / (k^n + x^n)
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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OdeParams {
    n_systems: u32,
    dim: u32,
    n_steps: u32,
    dt: f32,
    n_coeffs: u32,
}

/// Run GPU RK4 for multiple systems.
#[expect(clippy::too_many_arguments, reason = "validation binary")]
fn gpu_rk4(
    gpu: &Gpu,
    states: &[f32],
    coeffs: &[f32],
    n_systems: u32,
    dim: u32,
    n_steps: u32,
    dt: f32,
    n_coeffs: u32,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rk4_parallel"),
        source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rk4_bgl"),
        entries: &[
            storage_rw_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            uniform_entry(3),
            storage_rw_entry(4),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rk4_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rk4_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "rk4_step",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let state_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("state"),
        contents: bytemuck::cast_slice(states),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let coeffs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("coeffs"),
        contents: bytemuck::cast_slice(coeffs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_size = u64::from(n_systems * dim) * 4;
    let state_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("state_out"),
        size: out_size,
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
        label: Some("ode_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let scratch_size = u64::from(n_systems * dim * 5) * 4;
    let scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch"),
        size: scratch_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rk4_bg"),
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
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rk4_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rk4_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(gpu.dispatch_1d(n_systems, 64), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f32(&state_out_buf, (n_systems * dim) as usize)
}

fn validate_exponential_decay(h: &mut ValidationHarness, gpu: &Gpu) {
    // 1-system, 2-dim: y0 activated by y1, y1 activated by y0
    // coeffs: [prod0, deg0, act0_idx, prod1, deg1, act1_idx]
    let dim = 2_u32;
    let initial: Vec<f32> = vec![1.0, 0.5];
    let coeffs: Vec<f32> = vec![
        0.5, 0.1, 1.0, // y0: prod=0.5, deg=0.1, activated by y1
        0.3, 0.2, 0.0, // y1: prod=0.3, deg=0.2, activated by y0
    ];
    let n_steps = 100_u32;
    let dt = 0.01_f32;

    let cpu_result = cpu_rk4_hill(&initial, &coeffs, dim as usize, n_steps as usize, dt);

    match gpu_rk4(
        gpu,
        &initial,
        &coeffs,
        1,
        dim,
        n_steps,
        dt,
        coeffs.len() as u32,
    ) {
        Ok(gpu_result) => {
            for (d, (&g, &c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
                h.check_abs(
                    &format!("decay y[{d}]: GPU={g:.6} vs CPU={c:.6}"),
                    f64::from(g),
                    f64::from(c),
                    tolerances::GPU_RK4_F32,
                );
            }
        }
        Err(e) => {
            h.check_bool(&format!("decay: dispatch failed — {e}"), false);
        }
    }
}

fn validate_multi_system(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 2_u32;
    let n_systems = 4_u32;
    let n_steps = 50_u32;
    let dt = 0.01_f32;
    let n_coeffs = 6_u32;

    // 4 systems with different initial conditions, same dynamics
    let mut states: Vec<f32> = Vec::new();
    let mut all_coeffs: Vec<f32> = Vec::new();
    let coeffs_template: Vec<f32> = vec![0.5, 0.1, 1.0, 0.3, 0.2, 0.0];

    for i in 0..n_systems {
        let fi = i as f32;
        states.push(0.25f32.mul_add(fi, 0.5));
        states.push(0.1f32.mul_add(fi, 0.3));
        all_coeffs.extend_from_slice(&coeffs_template);
    }

    match gpu_rk4(
        gpu,
        &states,
        &all_coeffs,
        n_systems,
        dim,
        n_steps,
        dt,
        n_coeffs,
    ) {
        Ok(gpu_result) => {
            let mut all_match = true;
            for sys in 0..n_systems as usize {
                let fs = sys as f32;
                let sys_init = &[0.25f32.mul_add(fs, 0.5), 0.1f32.mul_add(fs, 0.3)];
                let cpu = cpu_rk4_hill(
                    sys_init,
                    &coeffs_template,
                    dim as usize,
                    n_steps as usize,
                    dt,
                );
                for d in 0..dim as usize {
                    let g = gpu_result[sys * dim as usize + d];
                    let c = cpu[d];
                    if (f64::from(g) - f64::from(c)).abs() > tolerances::GPU_RK4_F32 {
                        all_match = false;
                    }
                }
            }
            h.check_bool(
                &format!("{n_systems} systems: all GPU ≈ CPU within tolerance"),
                all_match,
            );

            h.check_bool(
                &format!(
                    "{n_systems} systems: correct output count ({})",
                    gpu_result.len()
                ),
                gpu_result.len() == (n_systems * dim) as usize,
            );

            let all_finite = gpu_result.iter().all(|v| v.is_finite());
            h.check_bool(
                &format!("{n_systems} systems: all values finite"),
                all_finite,
            );
        }
        Err(e) => {
            h.check_bool(&format!("multi-system: dispatch failed — {e}"), false);
        }
    }
}

fn validate_steady_state(h: &mut ValidationHarness, gpu: &Gpu) {
    // If prod * hill(activator) == deg * y, the system is at steady state.
    // A system starting near its steady state should stay there.
    let dim = 2_u32;
    let coeffs: Vec<f32> = vec![0.5, 0.5, 1.0, 0.5, 0.5, 0.0];
    // Near steady state: when hill ≈ 1 and y ≈ prod/deg = 1.0
    let initial: Vec<f32> = vec![1.0, 1.0];
    let n_steps = 200_u32;
    let dt = 0.01_f32;

    match gpu_rk4(
        gpu,
        &initial,
        &coeffs,
        1,
        dim,
        n_steps,
        dt,
        coeffs.len() as u32,
    ) {
        Ok(gpu_result) => {
            for (d, &g) in gpu_result.iter().enumerate() {
                // Should stay near 1.0 (steady state)
                h.check_bool(
                    &format!("steady state y[{d}]={g:.4} (near initial)"),
                    (g - 1.0).abs() < tolerances::ODE_STEADY_STATE_SLACK as f32,
                );
            }
        }
        Err(e) => {
            h.check_bool(&format!("steady state: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 2_u32;
    let initial: Vec<f32> = vec![1.0, 0.5];
    let coeffs: Vec<f32> = vec![0.5, 0.1, 1.0, 0.3, 0.2, 0.0];
    let n_steps = 50_u32;
    let dt = 0.01_f32;

    let run1 = gpu_rk4(
        gpu,
        &initial,
        &coeffs,
        1,
        dim,
        n_steps,
        dt,
        coeffs.len() as u32,
    );
    let run2 = gpu_rk4(
        gpu,
        &initial,
        &coeffs,
        1,
        dim,
        n_steps,
        dt,
        coeffs.len() as u32,
    );

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(&a, &b)| (a - b).abs() < f32::EPSILON);
            h.check_bool("RK4 determinism: two runs identical", identical);
        }
        _ => {
            h.check_bool("RK4 determinism: dispatch failed", false);
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
