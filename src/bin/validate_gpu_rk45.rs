// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: adaptive Dormand-Prince RK45 ODE integration via `metalForge` WGSL shader.
//!
//! Validates `metalForge/shaders/rk45_adaptive.wgsl` against CPU Dormand-Prince single step
//! for Hill-function regulatory network ODEs.
//!
//! ## Papers validated
//!
//! - Paper 020: Regulatory Network (Mhatre et al., 2020)
//! - Paper 021: Signal Integration (Srivastava et al., 2011)
//!
//! ## Provenance
//!
//! CPU reference: Dormand-Prince 5(4) embedded pair, Hill RHS.
//! WGSL shader: `metalForge/shaders/rk45_adaptive.wgsl`
//! Reference: Dormand & Prince (1980)

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::suboptimal_flops,
    clippy::too_many_lines
)]

use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = include_str!("../../metalForge/shaders/rk45_adaptive.wgsl");

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

    let mut h = ValidationHarness::new("gpu_rk45");

    validate_single_step(&mut h, &gpu);
    validate_multi_system(&mut h, &gpu);
    validate_error_estimate(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

fn hill(x: f32, k: f32, n: f32) -> f32 {
    let xn = x.powf(n);
    xn / (k.powf(n) + xn)
}

/// CPU Dormand-Prince single step with Hill function RHS.
/// RHS for variable d: prod * hill(y\[`act_idx`\], 0.5, 2.0) - deg * y\[d\]
/// where coeffs\[d*3\] = prod, coeffs\[d*3+1\] = deg, coeffs\[d*3+2\] = `act_idx`
fn cpu_rk45_step(
    state: &[f32],
    coeffs: &[f32],
    dim: usize,
    _n_coeffs: usize,
    dt: f32,
) -> (Vec<f32>, Vec<f32>) {
    let f = |y: &[f32]| -> Vec<f32> {
        (0..dim)
            .map(|d| {
                let prod = coeffs[d * 3];
                let deg = coeffs[d * 3 + 1];
                #[allow(clippy::cast_sign_loss)]
                let act_idx = coeffs[d * 3 + 2] as usize;
                prod * hill(y[act_idx], 0.5, 2.0) - deg * y[d]
            })
            .collect()
    };

    let y = state;
    let k1 = f(y);

    let y2: Vec<f32> = (0..dim).map(|d| y[d] + dt * (1.0 / 5.0) * k1[d]).collect();
    let k2 = f(&y2);

    let y3: Vec<f32> = (0..dim)
        .map(|d| y[d] + dt * ((3.0 / 40.0) * k1[d] + (9.0 / 40.0) * k2[d]))
        .collect();
    let k3 = f(&y3);

    let y4: Vec<f32> = (0..dim)
        .map(|d| y[d] + dt * ((44.0 / 45.0) * k1[d] - (56.0 / 15.0) * k2[d] + (32.0 / 9.0) * k3[d]))
        .collect();
    let k4 = f(&y4);

    let y5: Vec<f32> = (0..dim)
        .map(|d| {
            y[d] + dt
                * ((19372.0 / 6561.0) * k1[d] - (25360.0 / 2187.0) * k2[d]
                    + (64448.0 / 6561.0) * k3[d]
                    - (212.0 / 729.0) * k4[d])
        })
        .collect();
    let k5 = f(&y5);

    let y6: Vec<f32> = (0..dim)
        .map(|d| {
            y[d] + dt
                * ((9017.0 / 3168.0) * k1[d] - (355.0 / 33.0) * k2[d]
                    + (46732.0 / 5247.0) * k3[d]
                    + (49.0 / 176.0) * k4[d]
                    - (5103.0 / 18656.0) * k5[d])
        })
        .collect();
    let k6 = f(&y6);

    let new_state: Vec<f32> = (0..dim)
        .map(|d| {
            y[d] + dt
                * ((35.0 / 384.0) * k1[d] + (500.0 / 1113.0) * k3[d] + (125.0 / 192.0) * k4[d]
                    - (2187.0 / 6784.0) * k5[d]
                    + (11.0 / 84.0) * k6[d])
        })
        .collect();

    let error: Vec<f32> = (0..dim)
        .map(|d| {
            (dt * ((71.0 / 57600.0) * k1[d] - (71.0 / 16695.0) * k3[d] + (71.0 / 1920.0) * k4[d]
                - (17253.0 / 339_200.0) * k5[d]
                + (22.0 / 525.0) * k6[d]))
                .abs()
        })
        .collect();

    (new_state, error)
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Rk45Params {
    n_systems: u32,
    dim: u32,
    n_coeffs: u32,
    _pad: u32,
    dt: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
}

#[allow(clippy::too_many_arguments)]
fn gpu_rk45(
    gpu: &Gpu,
    states: &[f32],
    coeffs: &[f32],
    n_systems: u32,
    dim: u32,
    n_coeffs: u32,
    dt: f32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rk45_adaptive"),
        source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rk45_bgl"),
        entries: &[
            storage_ro_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            storage_rw_entry(3),
            uniform_entry(4),
            storage_rw_entry(5),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rk45_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rk45_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "rk45_step",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let state_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("state"),
        contents: bytemuck::cast_slice(states),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let coeffs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("coeffs"),
        contents: bytemuck::cast_slice(coeffs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_size = u64::from(n_systems * dim) * 4;
    let new_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("new_state"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let error_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("error"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let scratch_size = u64::from(n_systems * dim * 8) * 4;
    let scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch"),
        size: scratch_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let params = Rk45Params {
        n_systems,
        dim,
        n_coeffs,
        _pad: 0,
        dt,
        _pad2: 0.0,
        _pad3: 0.0,
        _pad4: 0.0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rk45_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rk45_bg"),
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
                resource: new_state_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: error_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: scratch_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rk45_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rk45_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(gpu.dispatch_1d(n_systems, 64), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    let new_state = gpu.read_buffer_f32(&new_state_buf, (n_systems * dim) as usize)?;
    let error = gpu.read_buffer_f32(&error_buf, (n_systems * dim) as usize)?;
    Ok((new_state, error))
}

fn validate_single_step(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 2_usize;
    let initial = [1.0_f32, 0.5_f32];
    let coeffs = [0.5, 0.1, 1.0, 0.3, 0.2, 0.0_f32];
    let dt = 0.01_f32;
    let n_coeffs = 6_u32;

    let (cpu_new_state, _cpu_error) = cpu_rk45_step(&initial, &coeffs, dim, n_coeffs as usize, dt);

    match gpu_rk45(gpu, &initial, &coeffs, 1, dim as u32, n_coeffs, dt) {
        Ok((gpu_new_state, _)) => {
            for (d, (&g, &c)) in gpu_new_state.iter().zip(cpu_new_state.iter()).enumerate() {
                h.check_abs(
                    &format!("single step y[{d}]: GPU={g:.6} vs CPU={c:.6}"),
                    f64::from(g),
                    f64::from(c),
                    tolerances::GPU_RK45_F32,
                );
            }
        }
        Err(e) => {
            h.check_bool(&format!("single step: dispatch failed — {e}"), false);
        }
    }
}

fn validate_multi_system(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 2_u32;
    let n_systems = 4_u32;
    let n_coeffs = 6_u32;
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

    match gpu_rk45(gpu, &states, &all_coeffs, n_systems, dim, n_coeffs, dt) {
        Ok((gpu_result, _)) => {
            let mut all_match = true;
            for sys in 0..n_systems as usize {
                let fs = sys as f32;
                let sys_init = &[0.25f32.mul_add(fs, 0.5), 0.1f32.mul_add(fs, 0.3)];
                let (cpu, _) = cpu_rk45_step(
                    sys_init,
                    &coeffs_template,
                    dim as usize,
                    n_coeffs as usize,
                    dt,
                );
                for d in 0..dim as usize {
                    let g = gpu_result[sys * dim as usize + d];
                    let c = cpu[d];
                    if (f64::from(g) - f64::from(c)).abs() > tolerances::GPU_RK45_F32 {
                        all_match = false;
                    }
                }
            }
            h.check_bool(
                &format!("{n_systems} systems: all GPU ≈ CPU within tolerance"),
                all_match,
            );
        }
        Err(e) => {
            h.check_bool(&format!("multi-system: dispatch failed — {e}"), false);
        }
    }
}

fn validate_error_estimate(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 2_u32;
    let initial = [1.0_f32, 0.5_f32];
    let coeffs = [0.5, 0.1, 1.0, 0.3, 0.2, 0.0_f32];
    let dt = 0.01_f32;
    let n_coeffs = 6_u32;

    match gpu_rk45(gpu, &initial, &coeffs, 1, dim, n_coeffs, dt) {
        Ok((new_state, error)) => {
            let non_negative = error.iter().all(|&e| e >= 0.0);
            h.check_bool("error estimate: all non-negative", non_negative);

            // Sanity: error estimate should be on same order as step (not absurdly larger)
            let state_change_max = initial
                .iter()
                .zip(new_state.iter())
                .map(|(&o, &n)| (n - o).abs())
                .fold(0.0f32, f32::max);
            let error_max = error.iter().copied().fold(0.0f32, f32::max);
            let negligible = tolerances::TENSOR_EXACT_F32 as f32;
            let sane = state_change_max > negligible && error_max < 100.0 * state_change_max
                || state_change_max <= negligible;
            h.check_bool("error estimate: same order as state change (sanity)", sane);
        }
        Err(e) => {
            h.check_bool(&format!("error estimate: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 2_u32;
    let initial = [1.0_f32, 0.5_f32];
    let coeffs = [0.5, 0.1, 1.0, 0.3, 0.2, 0.0_f32];
    let dt = 0.01_f32;
    let n_coeffs = 6_u32;

    let run1 = gpu_rk45(gpu, &initial, &coeffs, 1, dim, n_coeffs, dt);
    let run2 = gpu_rk45(gpu, &initial, &coeffs, 1, dim, n_coeffs, dt);

    match (run1, run2) {
        (Ok((r1_state, r1_err)), Ok((r2_state, r2_err))) => {
            let state_identical = r1_state
                .iter()
                .zip(r2_state.iter())
                .all(|(&a, &b)| (a - b).abs() < f32::EPSILON);
            let err_identical = r1_err
                .iter()
                .zip(r2_err.iter())
                .all(|(&a, &b)| (a - b).abs() < f32::EPSILON);
            h.check_bool(
                "RK45 determinism: two runs identical",
                state_identical && err_identical,
            );
        }
        _ => {
            h.check_bool("RK45 determinism: dispatch failed", false);
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
