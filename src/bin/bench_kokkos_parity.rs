// SPDX-License-Identifier: AGPL-3.0-or-later

//! Kokkos GPU parity benchmark harness.
//!
//! Runs representative barraCuda GPU ops at production scale and produces
//! timing output formatted for direct comparison against groundSpring's
//! Kokkos-CUDA baseline numbers (published in `wateringHole/
//! BARRACUDA_KOKKOS_GPU_BENCHMARK_RESULTS_MAR04_2026.md`).
//!
//! ## Known Gaps (groundSpring Kokkos baseline, RTX 4070)
//!
//! | Kernel | Kokkos CUDA | barraCuda WGSL | Gap |
//! |--------|-------------|----------------|-----|
//! | Anderson Lyapunov | 36 ms | 126 ms | 3.5× |
//! | mean (1M f64) | 58 µs | 8,454 µs | 146× |
//! | variance (1M f64) | 24 µs | 8,515 µs | 355× |
//! | Pearson r (1M f64) | 47 µs | 125 ms | 2,669× |
//! | Bootstrap mean | 2.2 ms | 123 ms | 57× |
//!
//! This harness produces timing for neuralSpring's domain-specific ops
//! to establish the full picture: which ops are compute-bound (small gap)
//! vs dispatch-overhead-bound (large gap).
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release --bin bench_kokkos_parity
//! ```

#![expect(
    clippy::cast_precision_loss,
    reason = "benchmark binary with index → f32/f64 conversions"
)]

use barracuda::ops::bio::{
    BatchFitnessGpu, HillGateGpu, HillGateParams, LocusVarianceGpu, MultiObjFitnessGpu,
    PairwiseHammingGpu, PairwiseJaccardGpu, PairwiseL2Gpu, SmithWatermanGpu, SpatialPayoffGpu,
    SwConfig,
};
use neural_spring::bench::{alloc_f32, buf_desc};
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const WARMUP: usize = 5;
const ITERATIONS: usize = 30;

struct KokkosEntry {
    name: &'static str,
    scale: String,
    warmup_us: f64,
    median_us: f64,
    min_us: f64,
    category: &'static str,
}

fn median_us(timings: &mut [Duration]) -> f64 {
    timings.sort();
    timings[timings.len() / 2].as_nanos() as f64 / 1000.0
}

fn min_us(timings: &[Duration]) -> f64 {
    timings
        .iter()
        .map(|d| d.as_nanos() as f64 / 1000.0)
        .fold(f64::INFINITY, f64::min)
}

fn bench_op(
    name: &'static str,
    scale: String,
    category: &'static str,
    mut op: impl FnMut(),
) -> KokkosEntry {
    for _ in 0..WARMUP {
        op();
    }
    let warmup_single = {
        let t = Instant::now();
        op();
        t.elapsed().as_nanos() as f64 / 1000.0
    };

    let mut timings = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        op();
        timings.push(start.elapsed());
    }

    KokkosEntry {
        name,
        scale,
        warmup_us: warmup_single,
        median_us: median_us(&mut timings),
        min_us: min_us(&timings),
        category,
    }
}

fn print_results(entries: &[KokkosEntry], adapter: &str) {
    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  KOKKOS PARITY BENCHMARK — barraCuda GPU ops at production scale                                   ║");
    eprintln!("║  Adapter: {adapter:<84}║");
    eprintln!("║  Warmup: {WARMUP}, Iterations: {ITERATIONS}                                                                          ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!(
        "{:<25} {:>15} {:>12} {:>12} {:>12} {:>15}",
        "Kernel", "Scale", "Warmup µs", "Median µs", "Min µs", "Category"
    );
    eprintln!("{}", "─".repeat(95));

    for e in entries {
        eprintln!(
            "{:<25} {:>15} {:>12.1} {:>12.1} {:>12.1} {:>15}",
            e.name, e.scale, e.warmup_us, e.median_us, e.min_us, e.category
        );
    }
    eprintln!("{}", "─".repeat(95));
    eprintln!();
    eprintln!("Categories:");
    eprintln!("  parallel_for  = Kokkos::parallel_for equivalent (map over elements)");
    eprintln!("  parallel_reduce = Kokkos::parallel_reduce equivalent (reduction)");
    eprintln!("  domain        = Domain-specific (no direct Kokkos equivalent)");
    eprintln!();
    eprintln!("Compare median against groundSpring Kokkos-CUDA baseline.");
    eprintln!("  <2×  = at parity    2-10× = dispatch overhead    >10× = structural gap");
    eprintln!();

    println!("kernel\tscale\twarmup_us\tmedian_us\tmin_us\tcategory");
    for e in entries {
        println!(
            "{}\t{}\t{:.1}\t{:.1}\t{:.1}\t{}",
            e.name, e.scale, e.warmup_us, e.median_us, e.min_us, e.category
        );
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "benchmark runner — sequential GPU kernel dispatch requires inline setup per kernel"
)]
#[tokio::main]
async fn main() {
    let gpu = neural_spring::validation::gpu_or_exit().await;
    let adapter = format!(
        "{} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend
    );
    eprintln!("  adapter: {adapter}");

    let dev_arc = Arc::clone(gpu.wgpu_device());
    let device = gpu.device();

    let mut results: Vec<KokkosEntry> = Vec::new();

    // ── NK Fitness (parallel_for: pop_size threads) ──────────────────────
    {
        let pop_size = 10_000_u32;
        let genome_len = 32_u32;
        let pop: Vec<f32> = (0..pop_size * genome_len)
            .map(|i| (i % 100) as f32 / 100.0)
            .collect();
        let weights: Vec<f32> = (0..genome_len).map(|i| (i % 10) as f32 / 10.0).collect();

        let pop_buf =
            device.create_buffer_init(&buf_desc("pop", &pop, wgpu::BufferUsages::STORAGE));
        let wt_buf =
            device.create_buffer_init(&buf_desc("wt", &weights, wgpu::BufferUsages::STORAGE));

        let op = BatchFitnessGpu::new(Arc::clone(&dev_arc));
        results.push(bench_op(
            "NK Fitness",
            format!("{pop_size}x{genome_len}"),
            "parallel_for",
            || {
                let out = alloc_f32(device, pop_size as usize);
                op.dispatch(&pop_buf, &wt_buf, &out, pop_size, genome_len);
            },
        ));
    }

    // ── Pairwise Hamming (parallel_reduce: n*(n-1)/2 pairs) ─────────────
    {
        let n_seqs = 200_u32;
        let seq_len = 1000_u32;
        let data: Vec<u32> = (0..(n_seqs * seq_len)).map(|i| i % 4).collect();

        let seq_buf =
            device.create_buffer_init(&buf_desc("seq", &data, wgpu::BufferUsages::STORAGE));
        let n_pairs = (n_seqs * (n_seqs - 1) / 2) as usize;

        let op = PairwiseHammingGpu::new(Arc::clone(&dev_arc));
        results.push(bench_op(
            "Pairwise Hamming",
            format!("{n_seqs}x{seq_len}"),
            "parallel_reduce",
            || {
                let out = alloc_f32(device, n_pairs);
                op.dispatch(&seq_buf, &out, n_seqs, seq_len);
            },
        ));
    }

    // ── Pairwise Jaccard (parallel_reduce) ───────────────────────────────
    {
        let n_seqs = 200_u32;
        let n_features = 500_u32;
        let data: Vec<f32> = (0..(n_seqs * n_features))
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();

        let data_buf =
            device.create_buffer_init(&buf_desc("jac", &data, wgpu::BufferUsages::STORAGE));
        let n_pairs = (n_seqs * (n_seqs - 1) / 2) as usize;

        let op = PairwiseJaccardGpu::new(Arc::clone(&dev_arc));
        results.push(bench_op(
            "Pairwise Jaccard",
            format!("{n_seqs}x{n_features}"),
            "parallel_reduce",
            || {
                let out = alloc_f32(device, n_pairs);
                op.dispatch(&data_buf, &out, n_seqs, n_features);
            },
        ));
    }

    // ── Pairwise L2 (parallel_reduce) ────────────────────────────────────
    {
        let n_seqs = 200_u32;
        let dim = 64_u32;
        let data: Vec<f32> = (0..(n_seqs * dim))
            .map(|i| (i % 100) as f32 / 50.0)
            .collect();

        let data_buf =
            device.create_buffer_init(&buf_desc("l2", &data, wgpu::BufferUsages::STORAGE));
        let n_pairs = (n_seqs * (n_seqs - 1) / 2) as usize;

        let op = PairwiseL2Gpu::new(Arc::clone(&dev_arc));
        results.push(bench_op(
            "Pairwise L2",
            format!("{n_seqs}x{dim}"),
            "parallel_reduce",
            || {
                let out = alloc_f32(device, n_pairs);
                let _ = op.dispatch(&data_buf, &out, n_seqs, dim);
            },
        ));
    }

    // ── Locus Variance (parallel_reduce over pops per locus) ─────────────
    {
        let n_pops = 10_u32;
        let n_loci = 500_u32;
        let freqs: Vec<f64> = (0..(n_pops * n_loci))
            .map(|i| f64::from(i % 100) / 100.0)
            .collect();

        let freq_buf =
            device.create_buffer_init(&buf_desc("freq", &freqs, wgpu::BufferUsages::STORAGE));

        let op = LocusVarianceGpu::new(Arc::clone(&dev_arc));
        results.push(bench_op(
            "Locus Variance",
            format!("{n_pops}x{n_loci}"),
            "parallel_reduce",
            || {
                let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: u64::from(n_loci) * 8,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                op.dispatch(&freq_buf, &out_buf, n_pops, n_loci);
            },
        ));
    }

    // ── Spatial Payoff (parallel_for: grid_size^2 threads) ───────────────
    {
        let grid = 128_u32;
        let strategies: Vec<u32> = (0..(grid * grid)).map(|i| i % 2).collect();

        let grid_buf =
            device.create_buffer_init(&buf_desc("grid", &strategies, wgpu::BufferUsages::STORAGE));

        let op = SpatialPayoffGpu::new(Arc::clone(&dev_arc));
        results.push(bench_op(
            "Spatial Payoff",
            format!("{grid}x{grid}"),
            "parallel_for",
            || {
                let out = alloc_f32(device, (grid * grid) as usize);
                op.dispatch(&grid_buf, &out, grid, 3.0, 1.0);
            },
        ));
    }

    // ── Hill Gate (parallel_for: n elements) ─────────────────────────────
    {
        let n = 40_000_u32;
        let a_vals: Vec<f32> = (0..n).map(|i| (i % 100) as f32 / 50.0).collect();
        let b_vals: Vec<f32> = (0..n).map(|i| (i % 100) as f32 / 50.0).collect();

        let a_buf =
            device.create_buffer_init(&buf_desc("ha", &a_vals, wgpu::BufferUsages::STORAGE));
        let b_buf =
            device.create_buffer_init(&buf_desc("hb", &b_vals, wgpu::BufferUsages::STORAGE));

        let params = HillGateParams {
            n_a: n,
            n_b: 1,
            mode: 0,
            _pad: 0,
            k_a: 1.0,
            k_b: 1.0,
            n_a_exp: 2.0,
            n_b_exp: 2.0,
            vmax: 1.0,
            _pad2: 0.0,
        };

        let op = HillGateGpu::new(Arc::clone(&dev_arc));
        results.push(bench_op(
            "Hill Gate",
            format!("{n}"),
            "parallel_for",
            || {
                let out = alloc_f32(device, n as usize);
                op.dispatch(&a_buf, &b_buf, &out, &params);
            },
        ));
    }

    // ── Multi-Obj Fitness (parallel_for: pop threads) ────────────────────
    {
        let pop = 10_000_u32;
        let genome_len = 30_u32;
        let n_obj = 3_u32;
        let solutions: Vec<f32> = (0..(pop * genome_len))
            .map(|i| (i % 100) as f32 / 100.0)
            .collect();

        let sol_buf =
            device.create_buffer_init(&buf_desc("sol", &solutions, wgpu::BufferUsages::STORAGE));

        let op = MultiObjFitnessGpu::new(Arc::clone(&dev_arc));
        results.push(bench_op(
            "Multi-Obj Fitness",
            format!("{pop}x{genome_len}x{n_obj}"),
            "parallel_for",
            || {
                let out = alloc_f32(device, (pop * n_obj) as usize);
                op.dispatch(&sol_buf, &out, pop, genome_len, n_obj);
            },
        ));
    }

    // ── Smith-Waterman (domain: anti-diagonal wavefront) ─────────────────
    {
        let n = 256_u32;
        let query: Vec<u32> = (0..n).map(|i| i % 4).collect();
        let target: Vec<u32> = (0..n).map(|i| (i + 1) % 4).collect();
        let subst = {
            let mut m = vec![-1.0_f64; 16];
            m[0] = 2.0;
            m[5] = 2.0;
            m[10] = 2.0;
            m[15] = 2.0;
            m
        };
        let config = SwConfig {
            gap_open: 11.0,
            gap_extend: 1.0,
            band_width: 64,
        };

        let sw = SmithWatermanGpu::new(&dev_arc);
        results.push(bench_op(
            "Smith-Waterman",
            format!("{n}x{n}, bw=64"),
            "domain",
            || {
                let _ = sw.align(&query, &target, &subst, &config);
            },
        ));
    }

    print_results(&results, &adapter);
}
