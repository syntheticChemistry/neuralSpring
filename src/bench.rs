// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU benchmark infrastructure for measuring dispatch overhead.
//!
//! Shared helpers for timing local `metalForge` dispatch vs upstream
//! `BarraCUDA` wrapper dispatch.  Each benchmark binary uses these
//! primitives instead of reimplementing timing/pipeline/buffer helpers.

#![allow(clippy::cast_precision_loss)]

use crate::gpu::Gpu;
use bytemuck::Pod;
use std::time::{Duration, Instant};

/// Result of a single local-vs-upstream benchmark comparison.
pub struct BenchResult {
    pub name: String,
    pub origin: &'static str,
    pub local_us: f64,
    pub upstream_us: f64,
}

/// Parameters for a local GPU compute dispatch.
pub struct DispatchParams<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub gpu: &'a Gpu,
    pub pipeline: &'a wgpu::ComputePipeline,
    pub bg: &'a wgpu::BindGroup,
    pub workgroups: u32,
    pub readback_buf: &'a wgpu::Buffer,
    pub readback_count: usize,
}

/// Time a local dispatch (warmup + iterations), returning median microseconds.
#[must_use]
pub fn time_dispatch(params: &DispatchParams<'_>, warmup: usize, iterations: usize) -> f64 {
    for _ in 0..warmup {
        dispatch_once(params);
    }
    let mut timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        dispatch_once(params);
        timings.push(start.elapsed());
    }
    median_us(&timings)
}

fn dispatch_once(params: &DispatchParams<'_>) {
    let mut enc = params
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(params.pipeline);
        pass.set_bind_group(0, params.bg, &[]);
        pass.dispatch_workgroups(params.workgroups, 1, 1);
    }
    params.queue.submit(std::iter::once(enc.finish()));
    let _ = params
        .gpu
        .read_buffer_f32(params.readback_buf, params.readback_count);
}

/// Time an upstream wrapper dispatch (warmup + iterations), returning median microseconds.
pub fn time_upstream(warmup: usize, iterations: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        timings.push(start.elapsed());
    }
    median_us(&timings)
}

fn median_us(timings: &[Duration]) -> f64 {
    let mut sorted: Vec<f64> = timings
        .iter()
        .map(|d| d.as_nanos() as f64 / 1000.0)
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[sorted.len() / 2]
}

/// Print a formatted summary table of benchmark results.
pub fn print_summary(results: &[BenchResult]) {
    eprintln!();
    eprintln!("╔════════════════════════════════════════════════════════════════════════════════════════╗");
    eprintln!(
        "║  LOCAL vs UPSTREAM — Same Shaders, Different Dispatch Paths                           ║"
    );
    eprintln!("╚════════════════════════════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!(
        "{:<35} {:>30} {:>10} {:>10} {:>10}",
        "Kernel", "Origin", "Local µs", "Upstr µs", "Ratio"
    );
    eprintln!("{}", "─".repeat(99));
    for r in results {
        let ratio = r.upstream_us / r.local_us;
        let marker = if ratio < 1.1 {
            "≈"
        } else if ratio > 1.5 {
            "⚠"
        } else {
            "~"
        };
        eprintln!(
            "{:<35} {:>30} {:>10.1} {:>10.1} {:>8.2}× {marker}",
            r.name, r.origin, r.local_us, r.upstream_us, ratio
        );
    }
    eprintln!("{}", "─".repeat(99));
    eprintln!("≈ = negligible overhead, ~ = minor overhead, ⚠ = investigate");
    eprintln!(
        "Upstream wrappers re-create params buffer per dispatch (expected ~0.5-1µs overhead)."
    );
}

/// Allocate a GPU storage buffer for `count` f32 values.
#[must_use]
pub fn alloc_f32(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (count * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Create a buffer-init descriptor from a `Pod` slice.
pub fn buf_desc<'a, T: Pod>(
    label: &'a str,
    data: &'a [T],
    usage: wgpu::BufferUsages,
) -> wgpu::util::BufferInitDescriptor<'a> {
    wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    }
}

/// Shorthand for a bind-group entry referencing an entire buffer.
pub fn bind_entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

/// Binding kind for pipeline layout construction.
#[derive(Copy, Clone)]
pub enum BindingKind {
    StorageRead,
    StorageWrite,
    Uniform,
}

/// Create a compute pipeline and its bind-group layout from binding descriptors.
#[must_use]
pub fn create_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    entry: &str,
    bindings: &[BindingKind],
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
        .iter()
        .enumerate()
        .map(|(i, k)| {
            let ty = match k {
                BindingKind::StorageRead => wgpu::BufferBindingType::Storage { read_only: true },
                BindingKind::StorageWrite => wgpu::BufferBindingType::Storage { read_only: false },
                BindingKind::Uniform => wgpu::BufferBindingType::Uniform,
            };
            #[allow(clippy::cast_possible_truncation)]
            wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        })
        .collect();
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
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
    (pipeline, bgl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_us_odd_count() {
        let timings = vec![
            Duration::from_micros(10),
            Duration::from_micros(30),
            Duration::from_micros(20),
        ];
        let med = median_us(&timings);
        assert!((med - 20.0).abs() < 0.5, "median of [10,30,20] = 20µs");
    }

    #[test]
    fn median_us_even_count() {
        let timings = vec![
            Duration::from_micros(10),
            Duration::from_micros(40),
            Duration::from_micros(20),
            Duration::from_micros(30),
        ];
        let med = median_us(&timings);
        assert!((med - 20.0).abs() < 1.0 || (med - 30.0).abs() < 1.0);
    }

    #[test]
    fn median_us_single() {
        let timings = vec![Duration::from_micros(42)];
        let med = median_us(&timings);
        assert!((med - 42.0).abs() < 0.5);
    }

    #[test]
    fn time_upstream_measures_closure() {
        let mut counter = 0_u32;
        let _t = time_upstream(2, 5, || {
            counter += 1;
        });
        assert_eq!(counter, 7, "2 warmup + 5 iterations");
    }

    #[test]
    fn bench_result_fields() {
        let r = BenchResult {
            name: "test_kernel".to_string(),
            origin: "unit test",
            local_us: 10.0,
            upstream_us: 15.0,
        };
        assert_eq!(r.name, "test_kernel");
        assert_eq!(r.origin, "unit test");
        assert!((r.upstream_us / r.local_us - 1.5).abs() < 1e-10);
    }

    #[test]
    fn print_summary_no_panic() {
        let results = vec![
            BenchResult {
                name: "matmul".to_string(),
                origin: "linalg",
                local_us: 10.0,
                upstream_us: 10.5,
            },
            BenchResult {
                name: "softmax".to_string(),
                origin: "activation",
                local_us: 5.0,
                upstream_us: 12.0,
            },
            BenchResult {
                name: "reduce".to_string(),
                origin: "reduction",
                local_us: 8.0,
                upstream_us: 10.0,
            },
        ];
        print_summary(&results);
    }

    #[test]
    fn binding_kind_copy() {
        let a = BindingKind::StorageRead;
        let b = a;
        assert!(matches!(b, BindingKind::StorageRead));
        let c = BindingKind::StorageWrite;
        assert!(matches!(c, BindingKind::StorageWrite));
        let d = BindingKind::Uniform;
        assert!(matches!(d, BindingKind::Uniform));
    }
}
