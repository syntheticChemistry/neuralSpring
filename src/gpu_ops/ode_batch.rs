// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-batched ODE integration via encoder batching (Pattern 1).
//!
//! Runs N systems × T timesteps in a single GPU dispatch using `rk4_parallel.wgsl`.
//! Only reads back the final state (scalar-only readback pattern).
//!
//! Closes the Phase B gap: "Full ODE loops → encoder batching with GPU PRNG".
//! The existing `rk4_parallel` shader runs T steps internally; no CPU round-trips.
//!
//! ## ODE format
//!
//! Uses the generic Hill-function ODE from `rk4_parallel.wgsl`:
//! `dy_d/dt = prod_d * hill(y[act_d], 0.5, 2) - deg_d * y_d`
//! with coeffs per dimension: `[prod, deg, activator_idx]`.
//!
//! For deterministic validation, use zero noise. The `signal_integration` vpsT ODE
//! uses a different RHS; full vpsT GPU support would require a dedicated shader.

// ODE batch GPU integration — casts handled inline where needed.

use barracuda::device::WgpuDevice;
use bytemuck::{Pod, Zeroable};
use neural_spring_forge::shaders::RK4_PARALLEL;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Parameters for batch RK4 integration (matches `rk4_parallel.wgsl` layout).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct OdeParams {
    n_systems: u32,
    dim: u32,
    n_steps: u32,
    dt: f32,
    n_coeffs: u32,
}

/// Run batch ODE integration on GPU: N systems × T timesteps, final state only.
///
/// Uses `rk4_parallel.wgsl` which runs all T steps in a single dispatch.
/// Encoder batching: one encoder, one submit, one readback of final states.
///
/// # Arguments
///
/// * `states` - Initial conditions: `n_systems * dim` f32 values (row-major)
/// * `coeffs` - Per-system coefficients: `n_systems * n_coeffs` f32 values.
///   Each dimension uses 3 coeffs: `[prod, deg, activator_idx]`.
/// * `n_systems` - Number of independent ODE systems
/// * `dim` - State dimension (e.g. 4 for [cdg, ai, vpsT, biofilm])
/// * `n_steps` - Number of RK4 timesteps
/// * `dt` - Timestep size
/// * `n_coeffs` - Coefficients per system (typically `dim * 3`)
/// * `device` - `WgpuDevice` for GPU execution
///
/// # Errors
///
/// Returns an error if GPU allocation or dispatch fails.
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "GPU ODE batch requires all physics parameters; single dispatch pass is more efficient than splitting"
)]
pub fn integrate_ode_batch_gpu(
    states: &[f32],
    coeffs: &[f32],
    n_systems: u32,
    dim: u32,
    n_steps: u32,
    dt: f32,
    n_coeffs: u32,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f32>, String> {
    let wgpu_device = device.device();
    let queue = device.queue();

    let shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ode_batch_rk4"),
        source: wgpu::ShaderSource::Wgsl(RK4_PARALLEL.into()),
    });

    let bgl = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ode_batch_bgl"),
        entries: &[
            storage_rw_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            uniform_entry(3),
            storage_rw_entry(4),
        ],
    });

    let pl = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ode_batch_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ode_batch_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "rk4_step",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let state_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ode_batch_state"),
        contents: bytemuck::cast_slice(states),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let coeffs_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ode_batch_coeffs"),
        contents: bytemuck::cast_slice(coeffs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_size = u64::from(n_systems * dim) * 4;
    let state_out_buf = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ode_batch_state_out"),
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
    let params_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ode_batch_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let scratch_size = u64::from(n_systems * dim * 5) * 4;
    let scratch_buf = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ode_batch_scratch"),
        size: scratch_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bg = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ode_batch_bg"),
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

    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ode_batch_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ode_batch_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n_systems.div_ceil(64), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    device
        .read_buffer_f32(&state_out_buf, (n_systems * dim) as usize)
        .map_err(|e| format!("ode_batch readback: {e}"))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ode_params_pod_layout() {
        assert_eq!(
            std::mem::size_of::<OdeParams>(),
            5 * 4,
            "OdeParams must be 20 bytes (5 × u32/f32)"
        );
    }

    #[test]
    fn ode_params_roundtrip() {
        let p = OdeParams {
            n_systems: 10,
            dim: 4,
            n_steps: 100,
            dt: 0.01,
            n_coeffs: 12,
        };
        let bytes = bytemuck::bytes_of(&p);
        let recovered: &OdeParams = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.n_systems, 10);
        assert_eq!(recovered.dim, 4);
        assert_eq!(recovered.n_steps, 100);
        assert!((recovered.dt - 0.01).abs() < 1e-6);
        assert_eq!(recovered.n_coeffs, 12);
    }

    #[test]
    fn storage_ro_entry_is_readonly() {
        let entry = storage_ro_entry(0);
        assert_eq!(entry.binding, 0);
        assert!(
            matches!(
                entry.ty,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    ..
                }
            ),
            "storage_ro_entry must be read-only storage"
        );
    }

    #[test]
    fn storage_rw_entry_is_readwrite() {
        let entry = storage_rw_entry(1);
        assert_eq!(entry.binding, 1);
        assert!(
            matches!(
                entry.ty,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    ..
                }
            ),
            "storage_rw_entry must be read-write storage"
        );
    }

    #[test]
    fn uniform_entry_is_uniform() {
        let entry = uniform_entry(3);
        assert_eq!(entry.binding, 3);
        assert!(
            matches!(
                entry.ty,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    ..
                }
            ),
            "uniform_entry must be Uniform"
        );
    }
}
