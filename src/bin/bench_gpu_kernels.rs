// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU kernel benchmarks: WGSL shader timing vs Rust CPU.
//!
//! Completes the full Python → Rust CPU → GPU WGSL performance chain.
//! Each shader is timed at the same problem size as its `bench_phase0pp_kernels`
//! Rust CPU counterpart, so speedups are directly comparable.
//!
//! ```text
//! cargo run --release --bin bench_gpu_kernels
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::doc_markdown
)]

use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::pangenome_selection;
use neural_spring::rng::Rng;
use neural_spring::sate_alignment;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

// Phase 4a Rust CPU reference timings (median µs from bench_phase0pp_kernels)
const RUST_HAMMING_US: f64 = 34.3;
const RUST_JACCARD_US: f64 = 142.3;
const RUST_NK_FITNESS_US: f64 = 17.9;
const PYTHON_HAMMING_US: f64 = 408.3;
const PYTHON_JACCARD_US: f64 = 2045.4;
const PYTHON_NK_FITNESS_US: f64 = 14087.2;

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
    eprintln!("║  neuralSpring — GPU WGSL Kernel Benchmarks                  ║");
    eprintln!("║  Warmup: {WARMUP}, Iterations: {ITERATIONS}                              ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();

    eprintln!("── Small scale (matches Phase 4a sizes) ──");
    eprintln!();
    let mut results: Vec<GpuBenchResult> = vec![
        bench_pairwise_hamming(&gpu, 20, 500, "small"),
        bench_pairwise_jaccard(&gpu, 30, 500, "small"),
        bench_batch_fitness(&gpu, 1000, 10, "small"),
        bench_spatial_payoff(&gpu, 64, "small"),
        bench_batch_ipr(&gpu, 64, 100, "small"),
    ];

    eprintln!();
    eprintln!("── Large scale (GPU parallelism wins) ──");
    eprintln!();
    results.push(bench_pairwise_hamming_with_cpu(&gpu, 200, 1000));
    results.push(bench_pairwise_jaccard_with_cpu(&gpu, 100, 2000));
    results.push(bench_batch_fitness(&gpu, 50_000, 64, "large"));
    results.push(bench_spatial_payoff(&gpu, 512, "large"));
    results.push(bench_batch_ipr(&gpu, 256, 2000, "large"));

    print_summary(&results);
}

#[allow(dead_code)]
struct GpuBenchResult {
    name: String,
    shader: String,
    papers: String,
    gpu_us: f64,
    rust_cpu_us: Option<f64>,
    python_us: Option<f64>,
}

// ── Pairwise Hamming (Paper 017) ──────────────────────────────────────

const HAMMING_WGSL: &str = include_str!("../../metalForge/shaders/pairwise_hamming.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HammingParams {
    n_seqs: u32,
    seq_len: u32,
}

fn bench_pairwise_hamming(gpu: &Gpu, n_seqs_p: u32, seq_len_p: u32, scale: &str) -> GpuBenchResult {
    let n_seqs = n_seqs_p;
    let seq_len = seq_len_p;
    let mut rng = Rng::new(42);
    let seqs: Vec<u32> = (0..n_seqs * seq_len).map(|_| rng.usize(4) as u32).collect();
    let n_pairs = n_seqs * (n_seqs - 1) / 2;

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bench_hamming"),
        source: wgpu::ShaderSource::Wgsl(HAMMING_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "pairwise_hamming", &[SR, SW, UNI]);

    let seq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&seqs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(n_pairs) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = HammingParams { n_seqs, seq_len };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            bind_entry(0, &seq_buf),
            bind_entry(1, &dist_buf),
            bind_entry(2, &params_buf),
        ],
    });

    let wg = n_pairs.div_ceil(256);
    let timings = bench_gpu_dispatch(
        device,
        queue,
        gpu,
        &pipeline,
        &bg,
        wg,
        &dist_buf,
        n_pairs as usize,
    );
    let us = median_us(&timings);

    println!("BENCH_HAMMING_{n_seqs}x{seq_len}_{scale}_GPU_US={us:.1}");

    let (rust_ref, py_ref) = if scale == "small" {
        (Some(RUST_HAMMING_US), Some(PYTHON_HAMMING_US))
    } else {
        (None, None)
    };

    GpuBenchResult {
        name: format!("Hamming {n_seqs}×{seq_len} ({scale})"),
        shader: "pairwise_hamming.wgsl".into(),
        papers: "017".into(),
        gpu_us: us,
        rust_cpu_us: rust_ref,
        python_us: py_ref,
    }
}

// ── Pairwise Jaccard (Paper 024) ──────────────────────────────────────

const JACCARD_WGSL: &str = include_str!("../../metalForge/shaders/pairwise_jaccard.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct JaccardParams {
    n_genomes: u32,
    n_genes: u32,
}

fn bench_pairwise_jaccard(
    gpu: &Gpu,
    n_genomes_p: u32,
    n_genes_p: u32,
    scale: &str,
) -> GpuBenchResult {
    let n_genomes = n_genomes_p;
    let n_genes = n_genes_p;
    let mut rng = Rng::new(42);
    let pa: Vec<f32> = (0..n_genes * n_genomes)
        .map(|_| if rng.uniform() < 0.5 { 1.0 } else { 0.0 })
        .collect();
    let n_pairs = n_genomes * (n_genomes - 1) / 2;

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bench_jaccard"),
        source: wgpu::ShaderSource::Wgsl(JACCARD_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "pairwise_jaccard", &[SR, SW, UNI]);

    let pa_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&pa),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(n_pairs) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = JaccardParams { n_genomes, n_genes };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            bind_entry(0, &pa_buf),
            bind_entry(1, &dist_buf),
            bind_entry(2, &params_buf),
        ],
    });

    let wg = n_pairs.div_ceil(256);
    let timings = bench_gpu_dispatch(
        device,
        queue,
        gpu,
        &pipeline,
        &bg,
        wg,
        &dist_buf,
        n_pairs as usize,
    );
    let us = median_us(&timings);

    println!("BENCH_JACCARD_{n_genomes}x{n_genes}_{scale}_GPU_US={us:.1}");

    let (rust_ref, py_ref) = if scale == "small" {
        (Some(RUST_JACCARD_US), Some(PYTHON_JACCARD_US))
    } else {
        (None, None)
    };

    GpuBenchResult {
        name: format!("Jaccard {n_genomes}×{n_genes} ({scale})"),
        shader: "pairwise_jaccard.wgsl".into(),
        papers: "024".into(),
        gpu_us: us,
        rust_cpu_us: rust_ref,
        python_us: py_ref,
    }
}

// ── Large-scale with inline CPU timing ────────────────────────────────

fn bench_pairwise_hamming_with_cpu(gpu: &Gpu, n_seqs: u32, seq_len: u32) -> GpuBenchResult {
    let mut gpu_result = bench_pairwise_hamming(gpu, n_seqs, seq_len, "large");

    let mut rng = Rng::new(42);
    let seqs: Vec<u8> = (0..(n_seqs * seq_len) as usize)
        .map(|_| rng.usize(4) as u8)
        .collect();

    let cpu_timings: Vec<Duration> = (0..50)
        .map(|_| {
            let start = Instant::now();
            let _ = sate_alignment::pairwise_distance_matrix(
                &seqs,
                n_seqs as usize,
                seq_len as usize,
                false,
            );
            start.elapsed()
        })
        .collect();
    let cpu_us = median_us(&cpu_timings);
    println!("BENCH_HAMMING_{n_seqs}x{seq_len}_large_CPU_US={cpu_us:.1}");
    gpu_result.rust_cpu_us = Some(cpu_us);
    gpu_result
}

fn bench_pairwise_jaccard_with_cpu(gpu: &Gpu, n_genomes: u32, n_genes: u32) -> GpuBenchResult {
    let mut gpu_result = bench_pairwise_jaccard(gpu, n_genomes, n_genes, "large");

    let mut rng = Rng::new(42);
    let pa: Vec<f64> = (0..n_genes * n_genomes)
        .map(|_| if rng.uniform() < 0.5 { 1.0 } else { 0.0 })
        .collect();

    let cpu_timings: Vec<Duration> = (0..50)
        .map(|_| {
            let start = Instant::now();
            let _ = pangenome_selection::jaccard_distance_matrix(
                &pa,
                n_genes as usize,
                n_genomes as usize,
            );
            start.elapsed()
        })
        .collect();
    let cpu_us = median_us(&cpu_timings);
    println!("BENCH_JACCARD_{n_genomes}x{n_genes}_large_CPU_US={cpu_us:.1}");
    gpu_result.rust_cpu_us = Some(cpu_us);
    gpu_result
}

// ── Batch Fitness (Papers 011-015) ────────────────────────────────────

const FITNESS_WGSL: &str = include_str!("../../metalForge/shaders/batch_fitness_eval.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FitnessParams {
    pop_size: u32,
    genome_len: u32,
}

fn bench_batch_fitness(
    gpu: &Gpu,
    pop_size_p: u32,
    genome_len_p: u32,
    scale: &str,
) -> GpuBenchResult {
    let pop_size = pop_size_p;
    let genome_len = genome_len_p;
    let mut rng = Rng::new(42);
    let population: Vec<f32> = (0..pop_size * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights: Vec<f32> = (0..genome_len).map(|_| rng.uniform() as f32).collect();

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bench_fitness"),
        source: wgpu::ShaderSource::Wgsl(FITNESS_WGSL.into()),
    });
    let (pipeline, bgl) =
        create_pipeline(device, &shader, "batch_fitness_linear", &[SR, SR, SW, UNI]);

    let pop_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&population),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let wt_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&weights),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let fit_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(pop_size) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = FitnessParams {
        pop_size,
        genome_len,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            bind_entry(0, &pop_buf),
            bind_entry(1, &wt_buf),
            bind_entry(2, &fit_buf),
            bind_entry(3, &params_buf),
        ],
    });

    let wg = pop_size.div_ceil(256);
    let timings = bench_gpu_dispatch(
        device,
        queue,
        gpu,
        &pipeline,
        &bg,
        wg,
        &fit_buf,
        pop_size as usize,
    );
    let us = median_us(&timings);

    println!("BENCH_FITNESS_{pop_size}x{genome_len}_{scale}_GPU_US={us:.1}");

    let (rust_ref, py_ref) = if scale == "small" {
        (Some(RUST_NK_FITNESS_US), Some(PYTHON_NK_FITNESS_US))
    } else {
        (None, None)
    };

    GpuBenchResult {
        name: format!("Fitness {pop_size}×{genome_len} ({scale})"),
        shader: "batch_fitness_eval.wgsl".into(),
        papers: "011-015".into(),
        gpu_us: us,
        rust_cpu_us: rust_ref,
        python_us: py_ref,
    }
}

// ── Spatial Payoff (Paper 019) ────────────────────────────────────────

const PAYOFF_WGSL: &str = include_str!("../../metalForge/shaders/spatial_payoff.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PayoffParams {
    grid_size: u32,
    b_x1000: u32,
    c_x1000: u32,
    _pad: u32,
}

fn bench_spatial_payoff(gpu: &Gpu, grid_size_p: u32, scale: &str) -> GpuBenchResult {
    let grid_size = grid_size_p;
    let mut rng = Rng::new(42);
    let grid: Vec<u32> = (0..grid_size * grid_size)
        .map(|_| u32::from(rng.uniform() > 0.5))
        .collect();
    let n_cells = (grid_size * grid_size) as usize;

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bench_payoff"),
        source: wgpu::ShaderSource::Wgsl(PAYOFF_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "spatial_payoff", &[SR, SW, UNI]);

    let grid_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&grid),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let fit_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n_cells * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = PayoffParams {
        grid_size,
        b_x1000: 3000,
        c_x1000: 1000,
        _pad: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            bind_entry(0, &grid_buf),
            bind_entry(1, &fit_buf),
            bind_entry(2, &params_buf),
        ],
    });

    let wg = (grid_size * grid_size).div_ceil(256);
    let timings = bench_gpu_dispatch(device, queue, gpu, &pipeline, &bg, wg, &fit_buf, n_cells);
    let us = median_us(&timings);

    println!("BENCH_SPATIAL_{grid_size}x{grid_size}_{scale}_GPU_US={us:.1}");

    GpuBenchResult {
        name: format!("Spatial {grid_size}×{grid_size} ({scale})"),
        shader: "spatial_payoff.wgsl".into(),
        papers: "019".into(),
        gpu_us: us,
        rust_cpu_us: None,
        python_us: None,
    }
}

// ── Batch IPR (Papers 022-023) ────────────────────────────────────────

const IPR_WGSL: &str = include_str!("../../metalForge/shaders/batch_ipr.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct IprParams {
    dim: u32,
    n_vectors: u32,
}

fn bench_batch_ipr(gpu: &Gpu, dim_p: u32, n_vectors_p: u32, scale: &str) -> GpuBenchResult {
    let dim = dim_p;
    let n_vectors = n_vectors_p;
    let mut rng = Rng::new(42);
    let eigenvectors: Vec<f32> = (0..dim * n_vectors).map(|_| rng.uniform() as f32).collect();

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bench_ipr"),
        source: wgpu::ShaderSource::Wgsl(IPR_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "batch_ipr", &[SR, SW, UNI]);

    let ev_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&eigenvectors),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let ipr_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(n_vectors) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = IprParams { dim, n_vectors };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            bind_entry(0, &ev_buf),
            bind_entry(1, &ipr_buf),
            bind_entry(2, &params_buf),
        ],
    });

    let wg = n_vectors.div_ceil(256);
    let timings = bench_gpu_dispatch(
        device,
        queue,
        gpu,
        &pipeline,
        &bg,
        wg,
        &ipr_buf,
        n_vectors as usize,
    );
    let us = median_us(&timings);

    println!("BENCH_IPR_{n_vectors}x{dim}_{scale}_GPU_US={us:.1}");

    GpuBenchResult {
        name: format!("IPR {n_vectors}×{dim} ({scale})"),
        shader: "batch_ipr.wgsl".into(),
        papers: "022-023".into(),
        gpu_us: us,
        rust_cpu_us: None,
        python_us: None,
    }
}

// ── GPU Dispatch Timing ───────────────────────────────────────────────

fn bench_gpu_dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    workgroups: u32,
    readback_buf: &wgpu::Buffer,
    readback_count: usize,
) -> Vec<Duration> {
    for _ in 0..WARMUP {
        dispatch_and_sync(
            device,
            queue,
            gpu,
            pipeline,
            bg,
            workgroups,
            readback_buf,
            readback_count,
        );
    }

    let mut timings = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        dispatch_and_sync(
            device,
            queue,
            gpu,
            pipeline,
            bg,
            workgroups,
            readback_buf,
            readback_count,
        );
        timings.push(start.elapsed());
    }
    timings
}

fn dispatch_and_sync(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    workgroups: u32,
    readback_buf: &wgpu::Buffer,
    readback_count: usize,
) {
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    let _ = gpu.read_buffer_f32(readback_buf, readback_count);
}

// ── Summary ───────────────────────────────────────────────────────────

fn print_summary(results: &[GpuBenchResult]) {
    eprintln!();
    eprintln!("╔════════════════════════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  BENCHMARK RESULTS — Full Python → Rust CPU → GPU WGSL Performance Chain                     ║");
    eprintln!("╚════════════════════════════════════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!(
        "{:<40} {:>7} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "Kernel", "Paper", "GPU µs", "Rust µs", "Py µs", "GPU/Rust", "GPU/Python"
    );
    eprintln!("{}", "─".repeat(104));

    for r in results {
        let rust_str = r
            .rust_cpu_us
            .map_or_else(|| "—".to_string(), |v| format!("{v:.1}"));
        let py_str = r
            .python_us
            .map_or_else(|| "—".to_string(), |v| format!("{v:.0}"));
        let gpu_vs_rust = r
            .rust_cpu_us
            .map_or_else(|| "—".to_string(), |cpu| format!("{:.1}×", cpu / r.gpu_us));
        let gpu_vs_python = r
            .python_us
            .map_or_else(|| "—".to_string(), |py| format!("{:.0}×", py / r.gpu_us));

        eprintln!(
            "{:<40} {:>7} {:>10.1} {:>10} {:>10} {:>12} {:>12}",
            r.name, r.papers, r.gpu_us, rust_str, py_str, gpu_vs_rust, gpu_vs_python
        );
    }

    eprintln!("{}", "─".repeat(104));
    eprintln!();
    eprintln!("GPU/Rust > 1.0× means GPU is faster. GPU/Python is total speedup vs interpreted.");
    eprintln!(
        "Small sizes show dispatch overhead (~1.5ms/op). Large sizes show GPU parallelism winning."
    );
    eprintln!("Cross-dispatch routes small→CPU, large→GPU based on these crossover points.");
}

// ── Helpers ───────────────────────────────────────────────────────────

fn median_us(timings: &[Duration]) -> f64 {
    let mut sorted: Vec<f64> = timings
        .iter()
        .map(|d| d.as_nanos() as f64 / 1000.0)
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[sorted.len() / 2]
}

const SR: BindingKind = BindingKind::StorageRead;
const SW: BindingKind = BindingKind::StorageWrite;
const UNI: BindingKind = BindingKind::Uniform;

#[derive(Copy, Clone)]
enum BindingKind {
    StorageRead,
    StorageWrite,
    Uniform,
}

fn create_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
    bindings: &[BindingKind],
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
        .iter()
        .enumerate()
        .map(|(i, kind)| match kind {
            BindingKind::StorageRead => storage_entry(i as u32, true),
            BindingKind::StorageWrite => storage_entry(i as u32, false),
            BindingKind::Uniform => uniform_entry(i as u32),
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
        entry_point,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    (pipeline, bgl)
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

fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
