// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-resident layer normalization — no round-trip.
//!
//! Barracuda's stock `layer_norm_wgsl` calls `read_buffer` after dispatch
//! to pull data back to CPU, then constructs a new `Tensor::new()` from
//! the CPU-side `Vec<f32>`.  This breaks streaming for any pipeline that
//! chains ops on the GPU.
//!
//! This evolved version keeps everything in GPU buffers.  The same WGSL
//! shader is used; only the Rust host dispatch is changed to avoid the
//! round-trip.
//!
//! ## Shortcoming reference
//!
//! `barracuda/src/ops/layer_norm_wgsl.rs` lines 179-182:
//! ```text
//! let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;
//! Ok(Tensor::new(output_data, shape.to_vec(), device.clone()))
//! ```

use crate::gpu::Gpu;
use barracuda::device::capabilities::WORKGROUP_SIZE_1D;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str =
    include_str!("../../../phase1/toadstool/crates/barracuda/src/shaders/norm/layer_norm.wgsl");

/// Uniform params matching the WGSL `Params` struct.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    size: u32,
    feature_size: u32,
    epsilon: f32,
}

/// GPU-resident layer normalization result.
///
/// The output stays in a `wgpu::Buffer` until explicitly read back.
pub struct LayerNormOutput {
    pub buffer: wgpu::Buffer,
    pub shape: Vec<usize>,
    pub count: usize,
}

impl LayerNormOutput {
    /// Read the output buffer back to CPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU readback fails.
    pub fn readback(&self, gpu: &Gpu) -> Result<Vec<f32>, String> {
        gpu.read_buffer_f32(&self.buffer, self.count)
    }
}

/// Dispatch GPU-resident layer norm.
///
/// `input_buffer` must contain `count` f32 values.
/// `shape` is the tensor shape; last dimension is the feature dimension.
/// Returns a `LayerNormOutput` whose buffer lives on the GPU.
///
/// # Errors
///
/// Returns an error if buffer allocation or shader compilation fails.
#[expect(clippy::cast_possible_truncation, reason = "fossil record — dimension casts in evolved code")]
pub fn layer_norm(
    gpu: &Gpu,
    input_buffer: &wgpu::Buffer,
    shape: &[usize],
    epsilon: f32,
) -> Result<LayerNormOutput, String> {
    let count: usize = shape.iter().product();
    let feature_size = *shape.last().ok_or("empty shape")?;

    let output_buffer = gpu.create_buffer_f32(count)?;

    let params = Params {
        size: count as u32,
        feature_size: feature_size as u32,
        epsilon,
    };

    let params_buffer = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("evolved::layer_norm params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let bind_group_layout =
        gpu.device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("evolved::layer_norm BGL"),
                entries: &[
                    bgl_entry(0, true),
                    bgl_entry(1, false),
                    bgl_uniform_entry(2),
                ],
            });

    let bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("evolved::layer_norm BG"),
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

    let shader = gpu.compile_shader(WGSL_SOURCE, "evolved::layer_norm");

    let pipeline_layout = gpu
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("evolved::layer_norm PL"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = gpu
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("evolved::layer_norm pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("evolved::layer_norm encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("evolved::layer_norm pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let num_batches = (count / feature_size) as u32;
        let workgroups = num_batches.div_ceil(WORKGROUP_SIZE_1D);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    gpu.queue().submit(Some(encoder.finish()));

    Ok(LayerNormOutput {
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

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_precision_loss, clippy::expect_used)]

    use super::*;

    fn cpu_layer_norm(data: &[f32], feature_size: usize, epsilon: f32) -> Vec<f32> {
        let mut out = data.to_vec();
        for batch in out.chunks_exact_mut(feature_size) {
            let mean: f32 = batch.iter().sum::<f32>() / feature_size as f32;
            let var: f32 =
                batch.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / feature_size as f32;
            let inv_std = 1.0 / (var + epsilon).sqrt();
            for x in batch.iter_mut() {
                *x = (*x - mean) * inv_std;
            }
        }
        out
    }

    #[tokio::test]
    async fn layer_norm_matches_cpu_reference() {
        let Ok(gpu) = Gpu::new().await else { return };

        let input = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = vec![2, 4];
        let epsilon = 1e-5_f32;

        let input_buf = gpu
            .upload_f32(&input)
            .expect("upload_f32 should succeed for test data");
        let result = layer_norm(&gpu, &input_buf, &shape, epsilon)
            .expect("layer_norm dispatch should succeed");
        let output = result.readback(&gpu).expect("readback should succeed");

        let expected = cpu_layer_norm(&input, 4, epsilon);
        for (i, (got, want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "element {i}: got {got}, want {want}"
            );
        }
    }

    #[tokio::test]
    async fn layer_norm_output_has_correct_shape() {
        let Ok(gpu) = Gpu::new().await else { return };

        let input = vec![0.0_f32; 12];
        let shape = vec![3, 4];
        let input_buf = gpu.upload_f32(&input).expect("upload_f32 should succeed");
        let result = layer_norm(&gpu, &input_buf, &shape, 1e-5).expect("layer_norm should succeed");

        assert_eq!(result.shape, vec![3, 4]);
        assert_eq!(result.count, 12);
    }

    #[test]
    fn layer_norm_rejects_empty_shape() {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        rt.block_on(async {
            let Ok(gpu) = Gpu::new().await else { return };
            let buf = gpu.create_buffer_f32(4).expect("buf");
            let result = layer_norm(&gpu, &buf, &[], 1e-5);
            assert!(result.is_err());
        });
    }
}
