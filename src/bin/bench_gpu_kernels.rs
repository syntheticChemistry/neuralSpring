// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU kernel benchmarks: `BarraCUDA` typed op timing vs Rust CPU.
//!
//! Completes the full Python → Rust CPU → GPU performance chain.
//! Each benchmark uses `BarraCUDA` typed op APIs (`PairwiseHammingGpu`,
//! `PairwiseJaccardGpu`, `BatchFitnessGpu`, `SpatialPayoffGpu`, `BatchIprGpu`)
//! at the same problem size as its `bench_phase0pp_kernels` Rust CPU counterpart,
//! so speedups are directly comparable.
//!
//! ```text
//! cargo run --release --bin bench_gpu_kernels
//! ```

#![expect(clippy::cast_possible_truncation, reason = "validation binary")]

use barracuda::ops::bio::{
    BatchFitnessGpu, PairwiseHammingGpu, PairwiseJaccardGpu, SpatialPayoffGpu,
};
use barracuda::spectral::BatchIprGpu;
use neural_spring::gpu::Gpu;
use neural_spring::pangenome_selection;
use neural_spring::rng::Rng;
use neural_spring::sate_alignment;
use neural_spring::validation::median_duration_us;
use std::sync::Arc;
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
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  neuralSpring — GPU BarraCUDA Typed Op Benchmarks          ║");
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

struct GpuBenchResult {
    name: String,
    papers: String,
    gpu_us: f64,
    rust_cpu_us: Option<f64>,
    python_us: Option<f64>,
}

// ── Pairwise Hamming (Paper 017) ──────────────────────────────────────

fn bench_pairwise_hamming(gpu: &Gpu, n_seqs_p: u32, seq_len_p: u32, scale: &str) -> GpuBenchResult {
    let n_seqs = n_seqs_p;
    let seq_len = seq_len_p;
    let mut rng = Rng::new(42);
    let seqs: Vec<u32> = (0..n_seqs * seq_len).map(|_| rng.usize(4) as u32).collect();
    let n_pairs = n_seqs * (n_seqs - 1) / 2;

    let device = gpu.device();
    let op = PairwiseHammingGpu::new(Arc::clone(gpu.wgpu_device()));

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

    let mut timings = bench_typed_op(|| {
        op.dispatch(&seq_buf, &dist_buf, n_seqs, seq_len);
        let _ = gpu.read_buffer_f32(&dist_buf, n_pairs as usize);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_HAMMING_{n_seqs}x{seq_len}_{scale}_GPU_US={us:.1}");

    let (rust_ref, py_ref) = if scale == "small" {
        (Some(RUST_HAMMING_US), Some(PYTHON_HAMMING_US))
    } else {
        (None, None)
    };

    GpuBenchResult {
        name: format!("Hamming {n_seqs}×{seq_len} ({scale})"),

        papers: "017".into(),
        gpu_us: us,
        rust_cpu_us: rust_ref,
        python_us: py_ref,
    }
}

// ── Pairwise Jaccard (Paper 024) ──────────────────────────────────────

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
    let op = PairwiseJaccardGpu::new(Arc::clone(gpu.wgpu_device()));

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

    let mut timings = bench_typed_op(|| {
        op.dispatch(&pa_buf, &dist_buf, n_genomes, n_genes);
        let _ = gpu.read_buffer_f32(&dist_buf, n_pairs as usize);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_JACCARD_{n_genomes}x{n_genes}_{scale}_GPU_US={us:.1}");

    let (rust_ref, py_ref) = if scale == "small" {
        (Some(RUST_JACCARD_US), Some(PYTHON_JACCARD_US))
    } else {
        (None, None)
    };

    GpuBenchResult {
        name: format!("Jaccard {n_genomes}×{n_genes} ({scale})"),

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

    let mut cpu_timings: Vec<Duration> = (0..50)
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
    let cpu_us = median_duration_us(&mut cpu_timings);
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

    let mut cpu_timings: Vec<Duration> = (0..50)
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
    let cpu_us = median_duration_us(&mut cpu_timings);
    println!("BENCH_JACCARD_{n_genomes}x{n_genes}_large_CPU_US={cpu_us:.1}");
    gpu_result.rust_cpu_us = Some(cpu_us);
    gpu_result
}

// ── Batch Fitness (Papers 011-015) ────────────────────────────────────

fn bench_batch_fitness(
    gpu: &Gpu,
    pop_size_p: u32,
    genome_len_p: u32,
    scale: &str,
) -> GpuBenchResult {
    let pop_size = pop_size_p;
    let genome_len = genome_len_p;
    let mut rng = Rng::new(42);
    let population: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..genome_len).map(|_| rng.uniform()).collect();

    let device = gpu.device();
    let op = BatchFitnessGpu::new(Arc::clone(gpu.wgpu_device()));

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
        size: u64::from(pop_size) * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut timings = bench_typed_op(|| {
        op.dispatch(&pop_buf, &wt_buf, &fit_buf, pop_size, genome_len);
        let _ = gpu.read_buffer_f64(&fit_buf, pop_size as usize);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_FITNESS_{pop_size}x{genome_len}_{scale}_GPU_US={us:.1}");

    let (rust_ref, py_ref) = if scale == "small" {
        (Some(RUST_NK_FITNESS_US), Some(PYTHON_NK_FITNESS_US))
    } else {
        (None, None)
    };

    GpuBenchResult {
        name: format!("Fitness {pop_size}×{genome_len} ({scale})"),

        papers: "011-015".into(),
        gpu_us: us,
        rust_cpu_us: rust_ref,
        python_us: py_ref,
    }
}

// ── Spatial Payoff (Paper 019) ────────────────────────────────────────

fn bench_spatial_payoff(gpu: &Gpu, grid_size_p: u32, scale: &str) -> GpuBenchResult {
    let grid_size = grid_size_p;
    let b = 3.0_f32;
    let c = 1.0_f32;
    let mut rng = Rng::new(42);
    let grid: Vec<u32> = (0..grid_size * grid_size)
        .map(|_| u32::from(rng.uniform() > 0.5))
        .collect();
    let n_cells = (grid_size * grid_size) as usize;

    let device = gpu.device();
    let op = SpatialPayoffGpu::new(Arc::clone(gpu.wgpu_device()));

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

    let mut timings = bench_typed_op(|| {
        op.dispatch(&grid_buf, &fit_buf, grid_size, b, c);
        let _ = gpu.read_buffer_f32(&fit_buf, n_cells);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_SPATIAL_{grid_size}x{grid_size}_{scale}_GPU_US={us:.1}");

    GpuBenchResult {
        name: format!("Spatial {grid_size}×{grid_size} ({scale})"),

        papers: "019".into(),
        gpu_us: us,
        rust_cpu_us: None,
        python_us: None,
    }
}

// ── Batch IPR (Papers 022-023) ────────────────────────────────────────

fn bench_batch_ipr(gpu: &Gpu, dim_p: u32, n_vectors_p: u32, scale: &str) -> GpuBenchResult {
    let dim = dim_p;
    let n_vectors = n_vectors_p;
    let mut rng = Rng::new(42);
    let eigenvectors: Vec<f32> = (0..dim * n_vectors).map(|_| rng.uniform() as f32).collect();

    let device = gpu.device();
    let op = BatchIprGpu::new(Arc::clone(gpu.wgpu_device()));

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

    let mut timings = bench_typed_op(|| {
        op.dispatch(&ev_buf, &ipr_buf, dim, n_vectors);
        let _ = gpu.read_buffer_f32(&ipr_buf, n_vectors as usize);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_IPR_{n_vectors}x{dim}_{scale}_GPU_US={us:.1}");

    GpuBenchResult {
        name: format!("IPR {n_vectors}×{dim} ({scale})"),

        papers: "022-023".into(),
        gpu_us: us,
        rust_cpu_us: None,
        python_us: None,
    }
}

// ── GPU Dispatch Timing ───────────────────────────────────────────────

fn bench_typed_op<F>(f: F) -> Vec<Duration>
where
    F: Fn(),
{
    for _ in 0..WARMUP {
        f();
    }
    let mut timings = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        f();
        timings.push(start.elapsed());
    }
    timings
}

// ── Summary ───────────────────────────────────────────────────────────

fn print_summary(results: &[GpuBenchResult]) {
    eprintln!();
    eprintln!("╔════════════════════════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  BENCHMARK RESULTS — Full Python → Rust CPU → GPU BarraCUDA Typed Op Performance Chain       ║");
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
