// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-resident HMM forward pass — log-domain logsumexp via WGSL.
//!
//! Computes the forward algorithm for Hidden Markov Models entirely on GPU
//! using log-domain arithmetic.  Each timestep dispatches one compute pass
//! where each thread handles one destination state; the logsumexp over
//! source states uses the max-subtract trick for numerical stability.
//!
//! For T observations and N states, the CPU-side loop issues T dispatches
//! into a single `CommandEncoder` (one `queue.submit`), keeping the alpha
//! vector GPU-resident between timesteps.
//!
//! ## Papers validated
//!
//! - Paper 016: HMM Forward/Backward/Viterbi (Liu et al., 2014)
//! - Paper 017: `SATé` Alignment (Liu et al., 2009)
//! - Paper 018: Introgression Detection (Liu et al., 2015)
//!
//! ## Absorption target
//!
//! `barracuda::ops::hmm` or `staging::StatefulPipeline` extension.
//! See `metalForge/shaders/ABSORPTION_TRACKER.md`.

use crate::gpu::Gpu;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = include_str!("../../metalForge/shaders/hmm_forward_log.wgsl");

/// Uniform params matching the WGSL `HmmParams` struct.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HmmParams {
    n_states: u32,
}

/// GPU-resident HMM forward result.
///
/// The final log-alpha vector stays in a `wgpu::Buffer` until explicitly
/// read back.
pub struct HmmForwardOutput {
    pub buffer: wgpu::Buffer,
    pub n_states: usize,
    pub log_likelihood: Option<f32>,
}

impl HmmForwardOutput {
    /// Read the final alpha vector back to CPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU readback fails.
    pub fn readback(&self, gpu: &Gpu) -> Result<Vec<f32>, String> {
        gpu.read_buffer_f32(&self.buffer, self.n_states)
    }
}

/// Run the HMM forward algorithm on GPU.
///
/// # Arguments
///
/// * `gpu` — GPU device wrapper
/// * `log_initial` — log `P(state_i)` for initial distribution, length N
/// * `log_trans` — log transition matrix, N×N row-major
/// * `log_emissions` — log emission matrix, T×N row-major (T observations, N states)
///
/// # Panics
///
/// Panics if `log_trans.len() != N*N` or `log_emissions.len()` is not a
/// multiple of N.
///
/// # Errors
///
/// Returns an error if shader compilation or buffer operations fail.
#[allow(clippy::too_many_lines)]
pub fn hmm_forward_gpu(
    gpu: &Gpu,
    log_initial: &[f32],
    log_trans: &[f32],
    log_emissions: &[f32],
) -> Result<HmmForwardOutput, String> {
    let n_states = log_initial.len();
    let n_obs = log_emissions.len() / n_states;
    assert_eq!(
        log_trans.len(),
        n_states * n_states,
        "log_trans must be N×N"
    );
    assert_eq!(
        log_emissions.len(),
        n_obs * n_states,
        "log_emissions must be T×N"
    );

    let device = gpu.device();
    let queue = gpu.queue();

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hmm_forward_log"),
        source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hmm_forward_bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            uniform_entry(4),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hmm_forward_pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hmm_forward_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "hmm_forward_log",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let log_trans_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("log_trans"),
        contents: bytemuck::cast_slice(log_trans),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let params = HmmParams {
        #[allow(clippy::cast_possible_truncation)]
        n_states: n_states as u32,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hmm_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let buf_size = std::mem::size_of_val(log_initial) as u64;

    // Standard HMM forward: α_0(j) = log_init(j) + log_emit(j, o_0)
    let mut alpha_init: Vec<f32> = log_initial.to_vec();
    for (j, a) in alpha_init.iter_mut().enumerate() {
        *a += log_emissions[j];
    }

    let alpha_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("alpha_a"),
        contents: bytemuck::cast_slice(&alpha_init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let alpha_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("alpha_b"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    #[allow(clippy::cast_possible_truncation)]
    let workgroup_count = (n_states as u32).div_ceil(256);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hmm_forward_encoder"),
    });

    let mut src_is_a = true;
    for t in 1..n_obs {
        let emit_offset = t * n_states;
        let log_emit_t = &log_emissions[emit_offset..emit_offset + n_states];

        let log_emit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("log_emit"),
            contents: bytemuck::cast_slice(log_emit_t),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let (src, dst) = if src_is_a {
            (&alpha_a, &alpha_b)
        } else {
            (&alpha_b, &alpha_a)
        };

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hmm_forward_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: log_trans_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: log_emit_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hmm_forward_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        src_is_a = !src_is_a;
    }

    queue.submit(std::iter::once(encoder.finish()));

    // Return the compute buffer directly — `Gpu::read_buffer_f32` handles
    // the MAP_READ staging internally via barracuda's readback path.
    let final_buf = if src_is_a { alpha_a } else { alpha_b };

    Ok(HmmForwardOutput {
        buffer: final_buf,
        n_states,
        log_likelihood: None,
    })
}

const fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
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
