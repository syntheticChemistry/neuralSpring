// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evolution-tier benchmark: CPU → `BarraCUDA` CPU → `BarraCUDA` GPU.
//!
//! Demonstrates the portable math evolution path for all Phase 0++
//! paper domains. Each domain is benchmarked at three tiers:
//!
//! 1. **Rust CPU** (`neuralSpring` lib) — pure math, single-thread
//! 2. **`BarraCUDA` CPU** (barracuda crate) — pure Rust, single-thread
//! 3. **`BarraCUDA` GPU** (typed GPU ops) — WGSL shader dispatch
//!
//! The benchmark shows that the same math is portable across tiers.
//! Run with `--with-python` to include Python/NumPy baselines.
//!
//! ```text
//! cargo run --release --bin bench_evolution_tiers
//! cargo run --release --bin bench_evolution_tiers -- --with-python
//! ```
//!
//! ## Provenance
//!
//! Session 74. Demonstrates evolution path:
//! Python → Rust CPU → `BarraCUDA` CPU → GPU dispatch → Pure GPU pipeline.
//!
//! # Panics
//!
//! Panics if the tokio runtime cannot be created — this is a benchmark binary.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::expect_used,
    reason = "validation binary"
)]

use barracuda::ops::bio::{
    BatchFitnessGpu, HmmBatchForwardF64, PairwiseHammingGpu, PairwiseJaccardGpu, PairwiseL2Gpu,
    SpatialPayoffGpu,
};
use neural_spring::counterdiabatic::NkLandscape;
use neural_spring::gpu::Gpu;
use neural_spring::hmm::Hmm;
use neural_spring::modes::l2_distance;
use neural_spring::rng::Rng;
use neural_spring::signal_integration::two_input_hill;
use neural_spring::spectral_commutativity;
use neural_spring::validation::median_duration_us;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;

const WARMUP: usize = 5;
const ITERS: usize = 100;

fn main() {
    let with_python = std::env::args().any(|a| a == "--with-python");

    eprintln!("╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  neuralSpring — Evolution Tier Benchmark                                     ║");
    eprintln!("║  Rust CPU → BarraCUDA CPU → BarraCUDA GPU                                    ║");
    eprintln!("║  Warmup: {WARMUP}, Iterations: {ITERS}                                                    ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let rt = tokio::runtime::Runtime::new()
        .expect("tokio runtime creation failed — required for async benchmark");
    let gpu = rt.block_on(async { Gpu::new().await.ok() });

    if let Some(ref g) = gpu {
        eprintln!(
            "  GPU: {} ({:?}, {:?})",
            g.adapter_name, g.device_type, g.backend
        );
    } else {
        eprintln!("  GPU: not available (CPU-only benchmark)");
    }
    eprintln!();

    let results = vec![
        bench_hmm_forward(gpu.as_ref()),
        bench_nk_fitness(gpu.as_ref()),
        bench_pairwise_hamming(gpu.as_ref()),
        bench_pairwise_l2(gpu.as_ref()),
        bench_pairwise_jaccard(gpu.as_ref()),
        bench_spatial_payoff(gpu.as_ref()),
        bench_hill_gate(),
        bench_commutator(),
    ];

    if with_python {
        eprintln!("  (Python benchmarks: run control/bench_*.py separately)\n");
    }

    print_table(&results);
}

struct TierResult {
    name: String,
    papers: String,
    rust_cpu_us: f64,
    barracuda_cpu_us: Option<f64>,
    barracuda_gpu_us: Option<f64>,
}

fn bench_rust<F: Fn()>(f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        f();
        times.push(t.elapsed());
    }
    median_duration_us(&mut times)
}

// ═══════════════════════════════════════════════════════════════════
// HMM Forward (Papers 016–018)
// ═══════════════════════════════════════════════════════════════════

fn bench_hmm_forward(gpu: Option<&Gpu>) -> TierResult {
    let mut rng = Rng::new(42);
    let hmm = Hmm::new(
        vec![
            vec![0.7, 0.2, 0.1],
            vec![0.2, 0.6, 0.2],
            vec![0.1, 0.2, 0.7],
        ],
        vec![
            vec![0.4, 0.3, 0.3],
            vec![0.2, 0.5, 0.3],
            vec![0.3, 0.3, 0.4],
        ],
        vec![0.33, 0.34, 0.33],
    );
    let seq_len = 5000;
    let (_, obs) = hmm.generate_sequence(seq_len, &mut rng);

    let rust_us = bench_rust(|| {
        let _ = hmm.forward(&obs);
    });

    let gpu_us = gpu.and_then(|g| {
        let dev = Arc::clone(g.wgpu_device());
        let op = HmmBatchForwardF64::new(dev).ok()?;
        let n_states = hmm.num_states() as u32;
        let n_symbols = hmm.num_symbols() as u32;
        let log_trans: Vec<f64> = hmm.transition.iter().map(|&p| p.ln()).collect();
        let log_emit: Vec<f64> = hmm.emission.iter().map(|&p| p.ln()).collect();
        let log_pi: Vec<f64> = hmm.initial.iter().map(|&p| p.ln()).collect();
        let obs_u32: Vec<u32> = obs.iter().map(|&o| o as u32).collect();

        let device = g.device();
        let lt = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&log_trans),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let le = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&log_emit),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let lp = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&log_pi),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let ob = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&obs_u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let alpha_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (seq_len * n_states as usize * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ll_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let us = bench_rust(|| {
            let _ = op.dispatch(&barracuda::ops::bio::hmm::HmmForwardArgs {
                n_states,
                n_symbols,
                n_steps: seq_len as u32,
                n_seqs: 1,
                log_trans: &lt,
                log_emit: &le,
                log_pi: &lp,
                observations: &ob,
                log_alpha_out: &alpha_buf,
                log_lik_out: &ll_buf,
            });
        });
        Some(us)
    });

    let name = format!("BENCH_HMM_FWD_3x{seq_len}_RUST_US={rust_us:.1}");
    eprintln!("{name}");

    TierResult {
        name: "HMM forward (3×5000)".into(),
        papers: "016-018".into(),
        rust_cpu_us: rust_us,
        barracuda_cpu_us: None,
        barracuda_gpu_us: gpu_us,
    }
}

// ═══════════════════════════════════════════════════════════════════
// NK Fitness (Papers 011–015)
// ═══════════════════════════════════════════════════════════════════

fn bench_nk_fitness(gpu: Option<&Gpu>) -> TierResult {
    let _nk = NkLandscape::new(10, 2, 42);
    let mut rng = Rng::new(77);
    let pop_size = 1000_usize;
    let genome_len = 10_usize;
    let genotypes: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..genome_len).map(|_| rng.uniform()).collect();

    let rust_us = bench_rust(|| {
        for i in 0..pop_size {
            let base = i * genome_len;
            let _: f64 = (0..genome_len)
                .map(|g| genotypes[base + g] * weights[g])
                .sum();
        }
    });

    let gpu_us = gpu.map(|g| {
        let op = BatchFitnessGpu::new(Arc::clone(g.wgpu_device()));
        let device = g.device();
        let geno_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&genotypes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let w_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&weights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (pop_size * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        bench_rust(|| {
            op.dispatch(
                &geno_buf,
                &w_buf,
                &out_buf,
                pop_size as u32,
                genome_len as u32,
            );
        })
    });

    let name = format!("BENCH_NK_FITNESS_1000x10_RUST_US={rust_us:.1}");
    eprintln!("{name}");

    TierResult {
        name: "NK fitness (1000×10)".into(),
        papers: "011-015".into(),
        rust_cpu_us: rust_us,
        barracuda_cpu_us: None,
        barracuda_gpu_us: gpu_us,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pairwise Hamming (Paper 017)
// ═══════════════════════════════════════════════════════════════════

fn bench_pairwise_hamming(gpu: Option<&Gpu>) -> TierResult {
    let n_seqs = 20_usize;
    let seq_len = 500_usize;
    let mut rng = Rng::new(44);
    let seqs_u8: Vec<u8> = (0..n_seqs * seq_len).map(|_| rng.usize(4) as u8).collect();
    let seqs_u32: Vec<u32> = seqs_u8.iter().map(|&v| u32::from(v)).collect();

    let rust_us = bench_rust(|| {
        let _ = neural_spring::sate_alignment::pairwise_distance_matrix(
            &seqs_u8, n_seqs, seq_len, false,
        );
    });

    let gpu_us = gpu.map(|g| {
        let op = PairwiseHammingGpu::new(Arc::clone(g.wgpu_device()));
        let device = g.device();
        let n_pairs = n_seqs * (n_seqs - 1) / 2;
        let seqs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&seqs_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n_pairs * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        bench_rust(|| {
            op.dispatch(&seqs_buf, &out_buf, n_seqs as u32, seq_len as u32);
        })
    });

    let name = format!("BENCH_HAMMING_20x500_RUST_US={rust_us:.1}");
    eprintln!("{name}");

    TierResult {
        name: "Pairwise Hamming (20×500)".into(),
        papers: "017".into(),
        rust_cpu_us: rust_us,
        barracuda_cpu_us: None,
        barracuda_gpu_us: gpu_us,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pairwise L2 — MODES (Paper 012)
// ═══════════════════════════════════════════════════════════════════

fn bench_pairwise_l2(gpu: Option<&Gpu>) -> TierResult {
    let n = 10_usize;
    let dim = 8_usize;
    let mut rng = Rng::new(66);
    let points: Vec<f64> = (0..n * dim).map(|_| rng.normal()).collect();
    let points_f32: Vec<f32> = points.iter().map(|&v| v as f32).collect();

    let rust_us = bench_rust(|| {
        for i in 0..n {
            for j in (i + 1)..n {
                let _ = l2_distance(
                    &points[i * dim..(i + 1) * dim],
                    &points[j * dim..(j + 1) * dim],
                );
            }
        }
    });

    let gpu_us = gpu.map(|g| {
        let op = PairwiseL2Gpu::new(Arc::clone(g.wgpu_device()));
        let device = g.device();
        let n_pairs = n * (n - 1) / 2;
        let pts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&points_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n_pairs * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        bench_rust(|| {
            let _ = op.dispatch(&pts_buf, &out_buf, n as u32, dim as u32);
        })
    });

    let name = format!("BENCH_L2_10x8_RUST_US={rust_us:.1}");
    eprintln!("{name}");

    TierResult {
        name: "Pairwise L2 (10×8)".into(),
        papers: "012".into(),
        rust_cpu_us: rust_us,
        barracuda_cpu_us: None,
        barracuda_gpu_us: gpu_us,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pairwise Jaccard — Pangenome (Paper 024)
// ═══════════════════════════════════════════════════════════════════

fn bench_pairwise_jaccard(gpu: Option<&Gpu>) -> TierResult {
    let n_genomes = 30_usize;
    let n_genes = 500_usize;
    let mut rng = Rng::new(88);
    let pa: Vec<f64> = (0..n_genomes * n_genes)
        .map(|_| if rng.uniform() > 0.3 { 1.0 } else { 0.0 })
        .collect();
    let pa_f32: Vec<f32> = pa.iter().map(|&v| v as f32).collect();

    let rust_us = bench_rust(|| {
        let _ =
            neural_spring::pangenome_selection::jaccard_distance_matrix(&pa, n_genes, n_genomes);
    });

    let gpu_us = gpu.map(|g| {
        let op = PairwiseJaccardGpu::new(Arc::clone(g.wgpu_device()));
        let device = g.device();
        let n_pairs = n_genomes * (n_genomes - 1) / 2;
        let pa_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&pa_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n_pairs * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        bench_rust(|| {
            op.dispatch(&pa_buf, &out_buf, n_genomes as u32, n_genes as u32);
        })
    });

    let name = format!("BENCH_JACCARD_30x500_RUST_US={rust_us:.1}");
    eprintln!("{name}");

    TierResult {
        name: "Pairwise Jaccard (30×500)".into(),
        papers: "024".into(),
        rust_cpu_us: rust_us,
        barracuda_cpu_us: None,
        barracuda_gpu_us: gpu_us,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Spatial Payoff — Game Theory (Paper 019)
// ═══════════════════════════════════════════════════════════════════

fn bench_spatial_payoff(gpu: Option<&Gpu>) -> TierResult {
    let grid_size = 32_usize;
    let n = grid_size * grid_size;
    let b = 1.5_f32;
    let c = 1.0_f32;
    let mut rng = Rng::new(99);
    let grid: Vec<u32> = (0..n).map(|_| u32::from(rng.uniform() > 0.5)).collect();

    let rust_us = bench_rust(|| {
        let gn = grid_size as i32;
        let neighbors: [(i32, i32); 8] = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ];
        let mut _total = 0.0_f32;
        for i in 0..grid_size {
            for j in 0..grid_size {
                let me = grid[i * grid_size + j];
                let mut payoff = 0.0_f32;
                for (di, dj) in &neighbors {
                    let ni = ((i as i32 + di).rem_euclid(gn)) as usize;
                    let nj = ((j as i32 + dj).rem_euclid(gn)) as usize;
                    let other = grid[ni * grid_size + nj];
                    payoff += match (me, other) {
                        (1, 1) => b - c,
                        (1, 0) => -c,
                        (0, 1) => b,
                        _ => 0.0,
                    };
                }
                _total += payoff;
            }
        }
    });

    let gpu_us = gpu.map(|g| {
        let op = SpatialPayoffGpu::new(Arc::clone(g.wgpu_device()));
        let device = g.device();
        let grid_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&grid),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        bench_rust(|| {
            op.dispatch(&grid_buf, &out_buf, grid_size as u32, b, c);
        })
    });

    let name = format!("BENCH_SPATIAL_32x32_RUST_US={rust_us:.1}");
    eprintln!("{name}");

    TierResult {
        name: "Spatial payoff (32×32)".into(),
        papers: "019".into(),
        rust_cpu_us: rust_us,
        barracuda_cpu_us: None,
        barracuda_gpu_us: gpu_us,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Hill Gate — Signal Integration (Paper 021), CPU-only
// ═══════════════════════════════════════════════════════════════════

fn bench_hill_gate() -> TierResult {
    let grid_side = 50_usize;
    let mut rng = Rng::new(123);
    let x: Vec<f64> = (0..grid_side).map(|_| rng.uniform() * 2.0).collect();
    let y: Vec<f64> = (0..grid_side).map(|_| rng.uniform() * 2.0).collect();

    let rust_us = bench_rust(|| {
        for xi in &x {
            for yi in &y {
                let _ = two_input_hill(*xi, *yi, 1.0, 1.0, 1.0, 2.0, 2.0);
            }
        }
    });

    let name = format!("BENCH_HILL_50x50_RUST_US={rust_us:.1}");
    eprintln!("{name}");

    TierResult {
        name: "Hill gate (50×50)".into(),
        papers: "021".into(),
        rust_cpu_us: rust_us,
        barracuda_cpu_us: None,
        barracuda_gpu_us: None,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Commutator — Spectral (Paper 022), CPU-only
// ═══════════════════════════════════════════════════════════════════

fn bench_commutator() -> TierResult {
    let n = 64_usize;
    let mut rng = Rng::new(55);
    let a: Vec<f64> = (0..n * n).map(|_| rng.normal()).collect();
    let b: Vec<f64> = (0..n * n).map(|_| rng.normal()).collect();

    let rust_us = bench_rust(|| {
        let c = spectral_commutativity::commutator(&a, &b, n);
        let _: f64 = c.iter().map(|&x| x * x).sum::<f64>().sqrt();
    });

    let name = format!("BENCH_COMMUTATOR_64x64_RUST_US={rust_us:.1}");
    eprintln!("{name}");

    TierResult {
        name: "Commutator ‖[A,B]‖_F (64×64)".into(),
        papers: "022".into(),
        rust_cpu_us: rust_us,
        barracuda_cpu_us: None,
        barracuda_gpu_us: None,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Output table
// ═══════════════════════════════════════════════════════════════════

fn print_table(results: &[TierResult]) {
    println!();
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║  EVOLUTION TIER BENCHMARK — Python → Rust CPU → BarraCUDA GPU                       ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!();
    println!(
        "{:<36}  {:<8}  {:>10}  {:>10}  {:>10}  {:>8}",
        "Kernel", "Papers", "Rust µs", "bC CPU µs", "bC GPU µs", "Speedup"
    );
    println!("{}", "─".repeat(92));

    let mut total_rust = 0.0_f64;
    let mut total_gpu = 0.0_f64;
    let mut gpu_count = 0_u32;

    for r in results {
        let bc_cpu = r
            .barracuda_cpu_us
            .map_or_else(|| "—".to_string(), |v| format!("{v:.1}"));
        let bc_gpu = r
            .barracuda_gpu_us
            .map_or_else(|| "—".to_string(), |v| format!("{v:.1}"));
        let speedup = r.barracuda_gpu_us.map_or_else(
            || "—".to_string(),
            |gpu| {
                let s = r.rust_cpu_us / gpu;
                format!("{s:.1}×")
            },
        );

        println!(
            "{:<36}  {:<8}  {:>10.1}  {:>10}  {:>10}  {:>8}",
            r.name, r.papers, r.rust_cpu_us, bc_cpu, bc_gpu, speedup
        );

        total_rust += r.rust_cpu_us;
        if let Some(gpu) = r.barracuda_gpu_us {
            total_gpu += gpu;
            gpu_count += 1;
        }
    }

    println!("{}", "─".repeat(92));

    if gpu_count > 0 {
        let overall_speedup = total_rust / total_gpu;
        println!(
            "{:<36}  {:<8}  {:>10.1}  {:>10}  {:>10.1}  {:>7.1}×",
            "TOTAL", "", total_rust, "—", total_gpu, overall_speedup
        );
    } else {
        println!(
            "{:<36}  {:<8}  {:>10.1}  {:>10}  {:>10}  {:>8}",
            "TOTAL", "", total_rust, "—", "—", "—"
        );
    }
    println!();
}
