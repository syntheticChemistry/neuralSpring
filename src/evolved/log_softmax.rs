// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-resident log-softmax — no round-trip.
//!
//! Barracuda's stock `log_softmax_wgsl` calls `read_buffer` after dispatch
//! to pull data back to CPU, then constructs a `Tensor::new()` from CPU
//! data.  This breaks streaming for any chained pipeline.
//!
//! This evolved version keeps the result in a GPU buffer.
//!
//! ## Shortcoming reference
//!
//! `barracuda/src/ops/log_softmax_wgsl.rs` lines 175-178:
//! ```text
//! let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;
//! Ok(Tensor::new(output_data, shape.to_vec(), device.clone()))
//! ```

use crate::gpu::Gpu;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = include_str!(
    "../../../phase1/toadstool/crates/barracuda/src/shaders/activation/log_softmax.wgsl"
);

/// Uniform params matching the WGSL `Params` struct.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    batch_size: u32,
    feature_size: u32,
}

/// GPU-resident log-softmax result.
pub struct LogSoftmaxOutput {
    pub buffer: wgpu::Buffer,
    pub shape: Vec<usize>,
    pub count: usize,
}

impl LogSoftmaxOutput {
    /// Read the output buffer back to CPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU readback fails.
    pub fn readback(&self, gpu: &Gpu) -> Result<Vec<f32>, String> {
        gpu.read_buffer_f32(&self.buffer, self.count)
    }
}

/// Dispatch GPU-resident log-softmax.
///
/// `input_buffer` must contain `count` f32 values.
/// `shape` is the tensor shape; last dimension is the feature dimension.
///
/// # Errors
///
/// Returns an error if buffer allocation or shader compilation fails.
#[allow(clippy::cast_possible_truncation)]
pub fn log_softmax(
    gpu: &Gpu,
    input_buffer: &wgpu::Buffer,
    shape: &[usize],
) -> Result<LogSoftmaxOutput, String> {
    let count: usize = shape.iter().product();
    let feature_size = *shape.last().ok_or("empty shape")?;
    let batch_size = count / feature_size;

    let output_buffer = gpu.create_buffer_f32(count)?;

    let params = Params {
        batch_size: batch_size as u32,
        feature_size: feature_size as u32,
    };

    let params_buffer = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("evolved::log_softmax params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let bind_group_layout =
        gpu.device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("evolved::log_softmax BGL"),
                entries: &[
                    bgl_entry(0, true),
                    bgl_entry(1, false),
                    bgl_uniform_entry(2),
                ],
            });

    let bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("evolved::log_softmax BG"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let shader = gpu.compile_shader(WGSL_SOURCE, "evolved::log_softmax");

    let pipeline_layout = gpu
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("evolved::log_softmax PL"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = gpu
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("evolved::log_softmax pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("evolved::log_softmax encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("evolved::log_softmax pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let workgroups = (batch_size as u32).div_ceil(256);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    gpu.queue().submit(Some(encoder.finish()));

    Ok(LogSoftmaxOutput {
        buffer: output_buffer,
        shape: shape.to_vec(),
        count,
    })
}

// ── Bind group layout helpers ──────────────────────────────────────────

const fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

const fn bgl_uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
