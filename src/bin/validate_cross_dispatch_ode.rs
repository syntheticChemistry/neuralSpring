// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-dispatch validation: RK4 ODE integration (Paper 020).
//!
//! Validates GPU ↔ CPU parity for the RK4 parallel integrator.
//! Uses upstream `barracuda::ops::rk_stage::WGSL_RK4_PARALLEL` shader source
//! (generic Hill-function ODE) vs CPU RK4 for the Hill-ODE system.
//!
//! Note: `BatchedOdeRK4F64` targets the specialized QS/c-di-GMP ODE and
//! cannot map to this generic Hill-ODE. We use the upstream WGSL constant
//! instead of the local metalForge `include_str`.

#![expect(
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "validation binary"
)]

use barracuda::dispatch::{dispatch_for, DispatchTarget};
use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring_forge::shaders::RK4_PARALLEL as WGSL_RK4_PARALLEL;
use wgpu::util::DeviceExt;

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

    let mut h = ValidationHarness::new("cross_dispatch_ode");
    validate_dispatch_routing(&mut h);
    validate_rk4_parity(&mut h, &gpu);
    h.finish();
}

// ── Dispatch routing ─────────────────────────────────────────────

fn validate_dispatch_routing(h: &mut ValidationHarness) {
    let small = dispatch_for("rk4", 100);
    let large = dispatch_for("rk4", 10_000);

    h.check_bool(
        &format!("dispatch: rk4(100) → {small:?}"),
        matches!(small, DispatchTarget::Cpu),
    );
    h.check_bool(
        &format!("dispatch: rk4(10k) → {large:?}"),
        matches!(large, DispatchTarget::Gpu),
    );
}

// ── CPU RK4 reference (Hill ODE, matches rk4_parallel.wgsl) ───────

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

// ── GPU RK4 (upstream WGSL_RK4_PARALLEL) ─────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct OdeParams {
    n_systems: u32,
    dim: u32,
    n_steps: u32,
    dt: f32,
    n_coeffs: u32,
}

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
        label: Some("xd_rk4"),
        source: wgpu::ShaderSource::Wgsl(WGSL_RK4_PARALLEL.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("xd_rk4_bgl"),
        entries: &[
            storage_rw_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            uniform_entry(3),
            storage_rw_entry(4),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("xd_rk4_pl"),
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("xd_rk4_pipe"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("rk4_step"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let state_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_state"),
        contents: bytemuck::cast_slice(states),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let coeffs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_coeffs"),
        contents: bytemuck::cast_slice(coeffs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_size = u64::from(n_systems * dim) * 4;
    let state_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xd_state_out"),
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
        label: Some("xd_ode_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let scratch_size = u64::from(n_systems * dim * 5) * 4;
    let scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xd_scratch"),
        size: scratch_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("xd_rk4_bg"),
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
        label: Some("xd_rk4_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("xd_rk4_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(n_systems.div_ceil(64), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f32(&state_out_buf, (n_systems * dim) as usize)
}

// ── wgpu layout helpers ──────────────────────────────────────────

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

// ── RK4 parity validation ───────────────────────────────────────

fn validate_rk4_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let dim = 4_u32;
    let n_parallel = 8_u32;
    let n_steps = 100_u32;
    let dt = 0.01_f32;

    let initial: Vec<f32> = vec![1.0, 0.5, 0.0, 0.0];
    let coeffs: Vec<f32> = vec![
        1.0, 0.5, 0.0, // y0: prod=1, deg=0.5, activator=y0
        0.5, 0.3, 0.0, // y1: prod=0.5, deg=0.3, activator=y0
        0.3, 0.2, 1.0, // y2: prod=0.3, deg=0.2, activator=y1
        0.2, 0.1, 2.0, // y3: prod=0.2, deg=0.1, activator=y2
    ];
    let n_coeffs = coeffs.len() as u32;

    let cpu_result = cpu_rk4_hill(&initial, &coeffs, dim as usize, n_steps as usize, dt);

    let mut gpu_states: Vec<f32> = Vec::new();
    let mut gpu_coeffs: Vec<f32> = Vec::new();
    for _ in 0..n_parallel {
        gpu_states.extend_from_slice(&initial);
        gpu_coeffs.extend_from_slice(&coeffs);
    }

    match gpu_rk4(
        gpu,
        &gpu_states,
        &gpu_coeffs,
        n_parallel,
        dim,
        n_steps,
        dt,
        n_coeffs,
    ) {
        Ok(gpu_result) => {
            let mut max_diff = 0.0_f64;
            for sys in 0..n_parallel as usize {
                for d in 0..dim as usize {
                    let g = gpu_result[sys * dim as usize + d];
                    let c = cpu_result[d];
                    let diff = (f64::from(g) - f64::from(c)).abs();
                    max_diff = max_diff.max(diff);
                }
            }

            h.check_upper(
                &format!("RK4 parity: max diff {max_diff:.2e} across {n_parallel} systems"),
                max_diff,
                tolerances::GPU_RK4_F32,
            );

            let all_finite = gpu_result.iter().all(|v| v.is_finite());
            h.check_bool("RK4 outputs finite", all_finite);
        }
        Err(e) => {
            h.check_bool(&format!("RK4 parity: GPU failed — {e}"), false);
        }
    }
}
