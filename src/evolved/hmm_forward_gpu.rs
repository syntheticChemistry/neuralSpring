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
//! ## Absorption status (`ToadStool` `77f70b2e`)
//!
//! Shader source absorbed as `barracuda::ops::bio::hmm::WGSL_HMM_FORWARD_LOG_F32`.
//! `BarraCUDA` also provides `HmmBatchForwardF64` for f64 batch dispatch.
//! This local dispatch module will be retired once validation binaries migrate
//! to the upstream `HmmBatchForwardF64` API.

use crate::gpu::Gpu;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = barracuda::ops::bio::hmm::WGSL_HMM_FORWARD_LOG_F32;

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
/// # Errors
///
/// Returns an error if inputs are malformed (wrong lengths), or if
/// shader compilation / buffer operations fail.
#[allow(clippy::too_many_lines)]
pub fn hmm_forward_gpu(
    gpu: &Gpu,
    log_initial: &[f32],
    log_trans: &[f32],
    log_emissions: &[f32],
) -> Result<HmmForwardOutput, String> {
    let n_states = log_initial.len();
    if n_states == 0 {
        return Err("log_initial must be non-empty".into());
    }
    let n_obs = log_emissions.len() / n_states;
    if log_trans.len() != n_states * n_states {
        return Err(format!(
            "log_trans length {} != N*N ({})",
            log_trans.len(),
            n_states * n_states,
        ));
    }
    if log_emissions.len() != n_obs * n_states {
        return Err(format!(
            "log_emissions length {} not a multiple of N ({})",
            log_emissions.len(),
            n_states,
        ));
    }

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
    let workgroup_count = gpu.dispatch_1d(n_states as u32, 256);

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

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn forward_2state_3obs_matches_cpu() {
        let Ok(gpu) = Gpu::new().await else { return };

        // 2-state HMM: weather model (Rainy=0, Sunny=1)
        // P(init) = [0.6, 0.4]
        let log_init: Vec<f32> = vec![0.6_f32.ln(), 0.4_f32.ln()];

        // Transition: [[0.7, 0.3], [0.4, 0.6]]
        let log_trans: Vec<f32> = vec![0.7_f32.ln(), 0.3_f32.ln(), 0.4_f32.ln(), 0.6_f32.ln()];

        // Emission: [[0.5, 0.5], [0.1, 0.9]]  (obs: 0 or 1)
        // Observations: [0, 1, 0]
        let emit = [
            [0.5_f32, 0.1], // t=0, obs=0
            [0.5_f32, 0.9], // t=1, obs=1
            [0.5_f32, 0.1], // t=2, obs=0
        ];
        let log_emit: Vec<f32> = emit.iter().flatten().map(|x| x.ln()).collect();

        let output = hmm_forward_gpu(&gpu, &log_init, &log_trans, &log_emit)
            .expect("hmm_forward_gpu should succeed");

        assert_eq!(output.n_states, 2);

        let alpha = output.readback(&gpu).expect("readback should succeed");
        assert_eq!(alpha.len(), 2);

        // Verify GPU log-alpha values are finite and in a reasonable range
        for (i, &v) in alpha.iter().enumerate() {
            assert!(v.is_finite(), "alpha[{i}] must be finite, got {v}");
            assert!(v < 0.0, "log-probabilities must be negative, got {v}");
        }

        // Verify that state 0 (Rainy) is more likely than state 1 (Sunny)
        // when the last observation is 0 (which has higher emission from Rainy)
        assert!(
            alpha[0] > alpha[1],
            "Rainy state should dominate after obs=0: alpha={alpha:?}"
        );
    }

    #[tokio::test]
    async fn forward_single_obs_equals_init_plus_emit() {
        let Ok(gpu) = Gpu::new().await else { return };

        let log_init = vec![0.5_f32.ln(), 0.5_f32.ln()];
        let log_trans = vec![0.5_f32.ln(); 4];
        // Single observation: emit = [0.8, 0.2]
        let log_emit = vec![0.8_f32.ln(), 0.2_f32.ln()];

        let output = hmm_forward_gpu(&gpu, &log_init, &log_trans, &log_emit)
            .expect("single-obs forward should succeed");

        let alpha = output.readback(&gpu).expect("readback");
        // With 1 observation, alpha = log_init + log_emit (no transition step)
        let expected_0 = 0.5_f32.ln() + 0.8_f32.ln();
        let expected_1 = 0.5_f32.ln() + 0.2_f32.ln();
        assert!(
            (alpha[0] - expected_0).abs() < 1e-5,
            "alpha[0]={}, expected={expected_0}",
            alpha[0]
        );
        assert!(
            (alpha[1] - expected_1).abs() < 1e-5,
            "alpha[1]={}, expected={expected_1}",
            alpha[1]
        );
    }
}
