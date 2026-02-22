// SPDX-License-Identifier: AGPL-3.0-or-later

//! Local metalForge dispatch vs upstream BarraCUDA wrapper benchmarks.
//!
//! Compares the performance of:
//! - **Local**: Manual wgpu dispatch using `include_str!` metalForge shaders
//! - **Upstream**: BarraCUDA Rust wrapper APIs (`BatchFitnessGpu`, etc.)
//!
//! Both paths use the same absorbed WGSL shaders (the wrappers encapsulate
//! the same kernels), but the wrapper adds its own buffer management and
//! dispatch geometry. This benchmark quantifies that overhead.
//!
//! ## Cross-Spring Evolution Context
//!
//! neuralSpring evolved these shaders → ToadStool absorbed them → BarraCUDA
//! wrapped them in ergonomic Rust APIs. This benchmark proves the wrappers
//! add no meaningful overhead vs raw manual dispatch.
//!
//! ```text
//! cargo run --release --bin bench_upstream_vs_local
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::needless_range_loop
)]

use barracuda::ops::bio::{
    BatchFitnessGpu, LocusVarianceGpu, PairwiseHammingGpu, PairwiseJaccardGpu, SpatialPayoffGpu,
};
use barracuda::spectral::BatchIprGpu;
use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "GPU: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(e) => {
            eprintln!("SKIP: {e} — no GPU adapter");
            std::process::exit(0);
        }
    };

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  Local metalForge vs Upstream BarraCUDA Wrapper Benchmarks  ║");
    eprintln!("║  Warmup: {WARMUP}, Iterations: {ITERATIONS}                              ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();

    let results = vec![
        bench_fitness(&gpu),
        bench_hamming(&gpu),
        bench_jaccard(&gpu),
        bench_locus_var(&gpu),
        bench_spatial(&gpu),
        bench_ipr(&gpu),
    ];

    print_summary(&results);
}

struct BenchResult {
    name: String,
    origin: &'static str,
    local_us: f64,
    upstream_us: f64,
}

// ─── Batch Fitness ───────────────────────────────────────────────────

const FITNESS_WGSL: &str = include_str!("../../metalForge/shaders/batch_fitness_eval.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FitnessParams { pop_size: u32, genome_len: u32 }

fn bench_fitness(gpu: &Gpu) -> BenchResult {
    let pop_size = 10_000_u32;
    let genome_len = 32_u32;
    let mut rng = Rng::new(42);
    let pop: Vec<f32> = (0..pop_size * genome_len).map(|_| rng.uniform() as f32).collect();
    let weights: Vec<f32> = (0..genome_len).map(|_| rng.uniform() as f32).collect();

    let device = gpu.device();
    let queue = gpu.queue();

    // Local dispatch
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_fitness"), source: wgpu::ShaderSource::Wgsl(FITNESS_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "batch_fitness_linear", &[SR, SR, SW, UNI]);
    let pop_buf = device.create_buffer_init(&buf_desc("pop", &pop, wgpu::BufferUsages::STORAGE));
    let wt_buf = device.create_buffer_init(&buf_desc("wt", &weights, wgpu::BufferUsages::STORAGE));
    let fit_buf = alloc_f32(device, pop_size as usize);
    let params_buf = device.create_buffer_init(&buf_desc("p", &[FitnessParams { pop_size, genome_len }], wgpu::BufferUsages::UNIFORM));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[be(0, &pop_buf), be(1, &wt_buf), be(2, &fit_buf), be(3, &params_buf)],
    });
    let wg = pop_size.div_ceil(256);
    let local_us = time_dispatch(device, queue, gpu, &pipeline, &bg, wg, &fit_buf, pop_size as usize);

    // Upstream dispatch
    let op = BatchFitnessGpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = time_upstream(gpu, WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, pop_size as usize);
        op.dispatch(&pop_buf, &wt_buf, &out, pop_size, genome_len);
        gpu.read_buffer_f32(&out, pop_size as usize).ok();
    });

    BenchResult { name: format!("BatchFitness {pop_size}×{genome_len}"), origin: "neuralSpring 011-015", local_us, upstream_us }
}

// ─── Pairwise Hamming ────────────────────────────────────────────────

const HAMMING_WGSL: &str = include_str!("../../metalForge/shaders/pairwise_hamming.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HammingParams { n_seqs: u32, seq_len: u32 }

fn bench_hamming(gpu: &Gpu) -> BenchResult {
    let n_seqs = 200_u32;
    let seq_len = 500_u32;
    let mut rng = Rng::new(42);
    let seqs: Vec<u32> = (0..n_seqs * seq_len).map(|_| rng.usize(4) as u32).collect();
    let n_pairs = (n_seqs * (n_seqs - 1) / 2) as usize;

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_hamming"), source: wgpu::ShaderSource::Wgsl(HAMMING_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "pairwise_hamming", &[SR, SW, UNI]);
    let seq_buf = device.create_buffer_init(&buf_desc("seqs", &seqs, wgpu::BufferUsages::STORAGE));
    let dist_buf = alloc_f32(device, n_pairs);
    let params_buf = device.create_buffer_init(&buf_desc("p", &[HammingParams { n_seqs, seq_len }], wgpu::BufferUsages::UNIFORM));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[be(0, &seq_buf), be(1, &dist_buf), be(2, &params_buf)],
    });
    let wg = (n_pairs as u32).div_ceil(256);
    let local_us = time_dispatch(device, queue, gpu, &pipeline, &bg, wg, &dist_buf, n_pairs);

    let op = PairwiseHammingGpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = time_upstream(gpu, WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, n_pairs);
        op.dispatch(&seq_buf, &out, n_seqs, seq_len);
        gpu.read_buffer_f32(&out, n_pairs).ok();
    });

    BenchResult { name: format!("Hamming {n_seqs}×{seq_len}"), origin: "neuralSpring 017 (SATé)", local_us, upstream_us }
}

// ─── Pairwise Jaccard ────────────────────────────────────────────────

const JACCARD_WGSL: &str = include_str!("../../metalForge/shaders/pairwise_jaccard.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct JaccardParams { n_genomes: u32, n_genes: u32 }

fn bench_jaccard(gpu: &Gpu) -> BenchResult {
    let n_genomes = 100_u32;
    let n_genes = 500_u32;
    let mut rng = Rng::new(42);
    let pa: Vec<f32> = (0..n_genes * n_genomes).map(|_| if rng.uniform() < 0.5 { 1.0 } else { 0.0 }).collect();
    let n_pairs = (n_genomes * (n_genomes - 1) / 2) as usize;

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_jaccard"), source: wgpu::ShaderSource::Wgsl(JACCARD_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "pairwise_jaccard", &[SR, SW, UNI]);
    let pa_buf = device.create_buffer_init(&buf_desc("pa", &pa, wgpu::BufferUsages::STORAGE));
    let dist_buf = alloc_f32(device, n_pairs);
    let params_buf = device.create_buffer_init(&buf_desc("p", &[JaccardParams { n_genomes, n_genes }], wgpu::BufferUsages::UNIFORM));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[be(0, &pa_buf), be(1, &dist_buf), be(2, &params_buf)],
    });
    let wg = (n_pairs as u32).div_ceil(256);
    let local_us = time_dispatch(device, queue, gpu, &pipeline, &bg, wg, &dist_buf, n_pairs);

    let op = PairwiseJaccardGpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = time_upstream(gpu, WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, n_pairs);
        op.dispatch(&pa_buf, &out, n_genomes, n_genes);
        gpu.read_buffer_f32(&out, n_pairs).ok();
    });

    BenchResult { name: format!("Jaccard {n_genomes}×{n_genes}"), origin: "neuralSpring 024 (Pangenome)", local_us, upstream_us }
}

// ─── Locus Variance ──────────────────────────────────────────────────

const LOCUS_VAR_WGSL: &str = include_str!("../../metalForge/shaders/locus_variance.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LocusParams { n_pops: u32, n_loci: u32 }

fn bench_locus_var(gpu: &Gpu) -> BenchResult {
    let n_pops = 50_u32;
    let n_loci = 500_u32;
    let mut rng = Rng::new(42);
    let freqs: Vec<f32> = (0..n_pops * n_loci).map(|_| rng.uniform() as f32).collect();

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_locus"), source: wgpu::ShaderSource::Wgsl(LOCUS_VAR_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "locus_variance", &[SR, SW, UNI]);
    let freq_buf = device.create_buffer_init(&buf_desc("freqs", &freqs, wgpu::BufferUsages::STORAGE));
    let var_buf = alloc_f32(device, n_loci as usize);
    let params_buf = device.create_buffer_init(&buf_desc("p", &[LocusParams { n_pops, n_loci }], wgpu::BufferUsages::UNIFORM));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[be(0, &freq_buf), be(1, &var_buf), be(2, &params_buf)],
    });
    let wg = n_loci.div_ceil(256);
    let local_us = time_dispatch(device, queue, gpu, &pipeline, &bg, wg, &var_buf, n_loci as usize);

    let op = LocusVarianceGpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = time_upstream(gpu, WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, n_loci as usize);
        op.dispatch(&freq_buf, &out, n_pops, n_loci);
        gpu.read_buffer_f32(&out, n_loci as usize).ok();
    });

    BenchResult { name: format!("LocusVariance {n_pops}×{n_loci}"), origin: "neuralSpring 025 (MetaPop)", local_us, upstream_us }
}

// ─── Spatial Payoff ──────────────────────────────────────────────────

const SPATIAL_WGSL: &str = include_str!("../../metalForge/shaders/spatial_payoff.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PayoffParams { grid_size: u32, b_x1000: u32, c_x1000: u32, _pad: u32 }

fn bench_spatial(gpu: &Gpu) -> BenchResult {
    let grid_size = 256_u32;
    let mut rng = Rng::new(42);
    let grid: Vec<u32> = (0..grid_size * grid_size).map(|_| u32::from(rng.uniform() > 0.5)).collect();
    let n_cells = (grid_size * grid_size) as usize;

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_spatial"), source: wgpu::ShaderSource::Wgsl(SPATIAL_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "spatial_payoff", &[SR, SW, UNI]);
    let grid_buf = device.create_buffer_init(&buf_desc("grid", &grid, wgpu::BufferUsages::STORAGE));
    let fit_buf = alloc_f32(device, n_cells);
    let params_buf = device.create_buffer_init(&buf_desc("p", &[PayoffParams { grid_size, b_x1000: 3000, c_x1000: 1000, _pad: 0 }], wgpu::BufferUsages::UNIFORM));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[be(0, &grid_buf), be(1, &fit_buf), be(2, &params_buf)],
    });
    let wg = (grid_size * grid_size).div_ceil(256);
    let local_us = time_dispatch(device, queue, gpu, &pipeline, &bg, wg, &fit_buf, n_cells);

    let op = SpatialPayoffGpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = time_upstream(gpu, WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, n_cells);
        op.dispatch(&grid_buf, &out, grid_size, 3.0, 1.0);
        gpu.read_buffer_f32(&out, n_cells).ok();
    });

    BenchResult { name: format!("SpatialPayoff {grid_size}×{grid_size}"), origin: "neuralSpring 019 (GameTheory)", local_us, upstream_us }
}

// ─── Batch IPR ───────────────────────────────────────────────────────

const IPR_WGSL: &str = include_str!("../../metalForge/shaders/batch_ipr.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct IprParams { dim: u32, n_vectors: u32 }

fn bench_ipr(gpu: &Gpu) -> BenchResult {
    let dim = 256_u32;
    let n_vectors = 1000_u32;
    let mut rng = Rng::new(42);
    let ev: Vec<f32> = (0..dim * n_vectors).map(|_| rng.uniform() as f32).collect();

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_ipr"), source: wgpu::ShaderSource::Wgsl(IPR_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "batch_ipr", &[SR, SW, UNI]);
    let ev_buf = device.create_buffer_init(&buf_desc("ev", &ev, wgpu::BufferUsages::STORAGE));
    let ipr_buf = alloc_f32(device, n_vectors as usize);
    let params_buf = device.create_buffer_init(&buf_desc("p", &[IprParams { dim, n_vectors }], wgpu::BufferUsages::UNIFORM));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[be(0, &ev_buf), be(1, &ipr_buf), be(2, &params_buf)],
    });
    let wg = n_vectors.div_ceil(256);
    let local_us = time_dispatch(device, queue, gpu, &pipeline, &bg, wg, &ipr_buf, n_vectors as usize);

    let op = BatchIprGpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = time_upstream(gpu, WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, n_vectors as usize);
        op.dispatch(&ev_buf, &out, dim, n_vectors);
        gpu.read_buffer_f32(&out, n_vectors as usize).ok();
    });

    BenchResult { name: format!("BatchIPR {n_vectors}×{dim}"), origin: "neuralSpring 022-023 (Anderson)", local_us, upstream_us }
}

// ─── Timing Helpers ──────────────────────────────────────────────────

fn time_dispatch(
    device: &wgpu::Device, queue: &wgpu::Queue, gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline, bg: &wgpu::BindGroup,
    workgroups: u32, readback_buf: &wgpu::Buffer, readback_count: usize,
) -> f64 {
    for _ in 0..WARMUP {
        dispatch_once(device, queue, gpu, pipeline, bg, workgroups, readback_buf, readback_count);
    }
    let mut timings = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        dispatch_once(device, queue, gpu, pipeline, bg, workgroups, readback_buf, readback_count);
        timings.push(start.elapsed());
    }
    median_us(&timings)
}

fn dispatch_once(
    device: &wgpu::Device, queue: &wgpu::Queue, gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline, bg: &wgpu::BindGroup,
    workgroups: u32, readback_buf: &wgpu::Buffer, readback_count: usize,
) {
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    queue.submit(std::iter::once(enc.finish()));
    let _ = gpu.read_buffer_f32(readback_buf, readback_count);
}

fn time_upstream<F: FnMut()>(gpu: &Gpu, warmup: usize, iters: usize, mut f: F) -> f64 {
    let _ = gpu;
    for _ in 0..warmup { f(); }
    let mut timings = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f();
        timings.push(start.elapsed());
    }
    median_us(&timings)
}

fn median_us(timings: &[Duration]) -> f64 {
    let mut sorted: Vec<f64> = timings.iter().map(|d| d.as_nanos() as f64 / 1000.0).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[sorted.len() / 2]
}

fn print_summary(results: &[BenchResult]) {
    eprintln!();
    eprintln!("╔════════════════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  LOCAL vs UPSTREAM — Same Shaders, Different Dispatch Paths                           ║");
    eprintln!("╚════════════════════════════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("{:<35} {:>30} {:>10} {:>10} {:>10}", "Kernel", "Origin", "Local µs", "Upstr µs", "Ratio");
    eprintln!("{}", "─".repeat(99));
    for r in results {
        let ratio = r.upstream_us / r.local_us;
        let marker = if ratio < 1.1 { "≈" } else if ratio > 1.5 { "⚠" } else { "~" };
        eprintln!(
            "{:<35} {:>30} {:>10.1} {:>10.1} {:>8.2}× {marker}",
            r.name, r.origin, r.local_us, r.upstream_us, ratio
        );
    }
    eprintln!("{}", "─".repeat(99));
    eprintln!("≈ = negligible overhead, ~ = minor overhead, ⚠ = investigate");
    eprintln!("Upstream wrappers re-create params buffer per dispatch (expected ~0.5-1µs overhead).");
}

// ─── Buffer / Pipeline Helpers ───────────────────────────────────────

fn alloc_f32(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (count * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn buf_desc<'a, T: Pod>(label: &'a str, data: &'a [T], usage: wgpu::BufferUsages) -> wgpu::util::BufferInitDescriptor<'a> {
    wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytemuck::cast_slice(data), usage }
}

fn be(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: buf.as_entire_binding() }
}

const SR: BK = BK::SR;
const SW: BK = BK::SW;
const UNI: BK = BK::UNI;

#[derive(Copy, Clone)]
enum BK { SR, SW, UNI }

fn create_pipeline(device: &wgpu::Device, shader: &wgpu::ShaderModule, entry: &str, bindings: &[BK]) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings.iter().enumerate().map(|(i, k)| {
        let (ty, ro) = match k { BK::SR => (wgpu::BufferBindingType::Storage { read_only: true }, true), BK::SW => (wgpu::BufferBindingType::Storage { read_only: false }, false), BK::UNI => (wgpu::BufferBindingType::Uniform, true) };
        let _ = ro;
        wgpu::BindGroupLayoutEntry { binding: i as u32, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None }
    }).collect();
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pl), module: shader, entry_point: entry, compilation_options: Default::default(), cache: None });
    (pipeline, bgl)
}
