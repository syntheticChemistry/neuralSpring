// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU benchmark infrastructure for measuring dispatch overhead.
//!
//! Shared helpers for timing local `metalForge` dispatch vs upstream
//! `BarraCUDA` wrapper dispatch.  Each benchmark binary uses these
//! primitives instead of reimplementing timing/pipeline/buffer helpers.

#![expect(
    clippy::cast_precision_loss,
    reason = "iteration counts and buffer sizes → f64 for throughput metrics"
)]

use crate::gpu::Gpu;
use bytemuck::Pod;
use std::time::{Duration, Instant};

/// Overhead ratio below which wrapper cost is negligible ("≈" marker).
const RATIO_NEGLIGIBLE: f64 = 1.1;
/// Overhead ratio above which dispatch cost should be investigated ("⚠" marker).
const RATIO_INVESTIGATE: f64 = 1.5;
/// Nanoseconds per microsecond for timing conversion.
const NANOS_PER_MICROSECOND: f64 = 1000.0;

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
        .map(|d| d.as_nanos() as f64 / NANOS_PER_MICROSECOND)
        .collect();
    sorted.sort_by(f64::total_cmp);
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
        let marker = if ratio < RATIO_NEGLIGIBLE {
            "≈"
        } else if ratio > RATIO_INVESTIGATE {
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
            #[expect(clippy::cast_possible_truncation, reason = "binding index fits in u32")]
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

    #[test]
    fn buf_desc_creates_descriptor() {
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let desc = buf_desc(
            "test",
            &data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        assert_eq!(desc.label, Some("test"));
        assert_eq!(desc.contents.len(), 16);
    }

    #[test]
    fn alloc_f32_and_bind_entry_gpu() {
        let _lock = crate::test_gpu_lock::acquire();
        let Some(gpu) = crate::gpu::tests::shared_gpu() else {
            return;
        };
        let buf = alloc_f32(gpu.device(), 64);
        let entry = bind_entry(0, &buf);
        assert_eq!(entry.binding, 0);
    }

    #[test]
    fn create_pipeline_gpu() {
        let _lock = crate::test_gpu_lock::acquire();
        let Some(gpu) = crate::gpu::tests::shared_gpu() else {
            return;
        };
        let shader = gpu.compile_shader(
            "@group(0) @binding(0) var<storage, read> input: array<f32>;
             @group(0) @binding(1) var<storage, read_write> output: array<f32>;
             @compute @workgroup_size(64)
             fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                 output[gid.x] = input[gid.x];
             }",
            "test_copy",
        );
        let (pipeline, bgl) = create_pipeline(
            gpu.device(),
            &shader,
            "main",
            &[BindingKind::StorageRead, BindingKind::StorageWrite],
        );
        let in_buf = alloc_f32(gpu.device(), 64);
        let out_buf = alloc_f32(gpu.device(), 64);
        let _bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[bind_entry(0, &in_buf), bind_entry(1, &out_buf)],
        });
        drop(pipeline);
    }

    #[test]
    fn time_dispatch_runs_gpu() {
        let _lock = crate::test_gpu_lock::acquire();
        let Some(gpu) = crate::gpu::tests::shared_gpu() else {
            return;
        };
        let shader = gpu.compile_shader(
            "@group(0) @binding(0) var<storage, read_write> data: array<f32>;
             @compute @workgroup_size(1)
             fn main() { data[0] = 1.0; }",
            "bench_noop",
        );
        let (pipeline, bgl) =
            create_pipeline(gpu.device(), &shader, "main", &[BindingKind::StorageWrite]);
        let buf = alloc_f32(gpu.device(), 1);
        let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[bind_entry(0, &buf)],
        });
        let params = DispatchParams {
            device: gpu.device(),
            queue: gpu.queue(),
            gpu: &gpu,
            pipeline: &pipeline,
            bg: &bg,
            workgroups: 1,
            readback_buf: &buf,
            readback_count: 1,
        };
        let us = time_dispatch(&params, 1, 3);
        assert!(us > 0.0, "dispatch should take non-zero time");
    }
}
