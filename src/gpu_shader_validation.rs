// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared infrastructure for GPU WGSL shader validation binaries.
//!
//! Eliminates the per-shader wgpu ceremony (bind group layout creation,
//! bind group creation, pipeline compilation, dispatch, readback) that
//! previously bloated validation binaries.
//!
//! ## Design
//!
//! [`ShaderBinding`] + [`dispatch_shader`] replace ~30 lines of boilerplate
//! per shader test with a single function call.  The builder discovers the
//! layout from the binding list — no manual `BindGroupLayoutEntry` arrays.
//!
//! ## Usage
//!
//! ```ignore
//! use neural_spring::gpu_shader_validation::*;
//!
//! let result = dispatch_and_read(gpu, &shader, "main", &[
//!     ShaderBinding::StorageRo(&input_buf),
//!     ShaderBinding::StorageRw(&output_buf),
//!     ShaderBinding::Uniform(&params_buf),
//! ], (workgroups_x, 1, 1), output_count);
//! ```

use crate::gpu::Gpu;

/// Buffer binding for GPU shader validation dispatch.
pub enum ShaderBinding<'a> {
    /// Read-only storage buffer (SSBO).
    StorageRo(&'a wgpu::Buffer),
    /// Read-write storage buffer (output).
    StorageRw(&'a wgpu::Buffer),
    /// Uniform buffer (parameters).
    Uniform(&'a wgpu::Buffer),
}

const fn layout_entry(binding: u32, kind: &ShaderBinding<'_>) -> wgpu::BindGroupLayoutEntry {
    let ty = match kind {
        ShaderBinding::StorageRo(_) => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        ShaderBinding::StorageRw(_) => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        ShaderBinding::Uniform(_) => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    };
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

const fn buffer_of<'a>(binding: &'a ShaderBinding<'a>) -> &'a wgpu::Buffer {
    match binding {
        ShaderBinding::StorageRo(b) | ShaderBinding::StorageRw(b) | ShaderBinding::Uniform(b) => b,
    }
}

/// Compile pipeline, create bind groups from bindings, dispatch, and submit.
///
/// This is the core boilerplate eliminator.  Call sites provide only the
/// semantically meaningful parts: shader, entry point, typed bindings, and
/// workgroup dimensions.
pub fn dispatch_shader(
    gpu: &Gpu,
    shader: &wgpu::ShaderModule,
    entry: &str,
    bindings: &[ShaderBinding<'_>],
    workgroups: (u32, u32, u32),
) {
    let device = gpu.device();
    let queue = gpu.queue();

    #[allow(clippy::cast_possible_truncation)]
    let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
        .iter()
        .enumerate()
        .map(|(i, b)| layout_entry(i as u32, b))
        .collect();

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &layout_entries,
    });

    #[allow(clippy::cast_possible_truncation)]
    let bg_entries: Vec<wgpu::BindGroupEntry<'_>> = bindings
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buffer_of(b).as_entire_binding(),
        })
        .collect();

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &bg_entries,
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pl),
        module: shader,
        entry_point: entry,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }
    queue.submit(std::iter::once(encoder.finish()));
}

/// Dispatch a shader and read back f64 results.
///
/// Convenience wrapper: dispatches, then reads `output_count` f64 values
/// from the output buffer (identified as the first [`ShaderBinding::StorageRw`]
/// in `bindings`).
///
/// # Panics
///
/// Panics if no `StorageRw` binding is present or GPU readback fails.
/// Acceptable in validation binaries (which report pass/fail via exit code).
#[must_use]
#[allow(clippy::expect_used)]
pub fn dispatch_and_read(
    gpu: &Gpu,
    shader: &wgpu::ShaderModule,
    entry: &str,
    bindings: &[ShaderBinding<'_>],
    workgroups: (u32, u32, u32),
    output_count: usize,
) -> Vec<f64> {
    let out_buf = bindings
        .iter()
        .find_map(|b| match b {
            ShaderBinding::StorageRw(buf) => Some(*buf),
            _ => None,
        })
        .expect("dispatch_and_read requires a StorageRw binding");

    dispatch_shader(gpu, shader, entry, bindings, workgroups);
    gpu.read_buffer_f64(out_buf, output_count)
        .expect("GPU f64 readback")
}

/// Upload f64 data to a read-only storage buffer.
#[must_use]
pub fn upload_f64(gpu: &Gpu, data: &[f64], label: &str) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    gpu.device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
}

/// Upload a `bytemuck::Pod` struct as a uniform buffer.
#[must_use]
pub fn upload_params<T: bytemuck::Pod>(gpu: &Gpu, params: &T, label: &str) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    gpu.device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

/// Maximum absolute element-wise difference between two f64 slices.
#[must_use]
pub fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Workgroup count for a 1D dispatch: `ceil(n / 256)`.
#[must_use]
pub const fn wg1d(n: u32) -> (u32, u32, u32) {
    (n.div_ceil(256), 1, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_diff_identical() {
        assert!((max_diff(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])).abs() < 1e-15);
    }

    #[test]
    fn max_diff_known() {
        let d = max_diff(&[1.0, 5.0], &[1.0, 2.0]);
        assert!((d - 3.0).abs() < 1e-15);
    }

    #[test]
    fn max_diff_empty() {
        assert!((max_diff(&[], &[])).abs() < 1e-15);
    }

    #[test]
    fn wg1d_exact() {
        assert_eq!(wg1d(256), (1, 1, 1));
        assert_eq!(wg1d(257), (2, 1, 1));
        assert_eq!(wg1d(512), (2, 1, 1));
        assert_eq!(wg1d(1), (1, 1, 1));
    }
}
