// SPDX-License-Identifier: AGPL-3.0-or-later

//! Local metalForge dispatch vs upstream `BarraCUDA` wrapper benchmarks.
//!
//! Compares the performance of:
//! - **Local**: Manual wgpu dispatch using `include_str!` metalForge shaders
//! - **Upstream**: `BarraCUDA` Rust wrapper APIs (`BatchFitnessGpu`, etc.)
//!
//! Both paths use the same absorbed WGSL shaders (the wrappers encapsulate
//! the same kernels), but the wrapper adds its own buffer management and
//! dispatch geometry. This benchmark quantifies that overhead.
//!
//! ## Cross-Spring Evolution Context
//!
//! neuralSpring evolved these shaders → `ToadStool` absorbed them → `BarraCUDA`
//! wrapped them in ergonomic Rust APIs. This benchmark proves the wrappers
//! add no meaningful overhead vs raw manual dispatch.
//!
//! ```text
//! cargo run --release --bin bench_upstream_vs_local
//! ```

#![expect(clippy::cast_possible_truncation, reason = "validation binary")]

mod extended;

use barracuda::ops::bio::{
    BatchFitnessGpu, HillGateGpu, HillGateParams, LocusVarianceGpu, PairwiseHammingGpu,
    PairwiseJaccardGpu, SpatialPayoffGpu,
};
use barracuda::spectral::BatchIprGpu;
use bytemuck::{Pod, Zeroable};
use neural_spring::bench::BenchResult;
use neural_spring::bench::{
    self, BindingKind, DispatchParams, alloc_f32, bind_entry as be, buf_desc, create_pipeline,
};
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const SR: BindingKind = BindingKind::StorageRead;
const SW: BindingKind = BindingKind::StorageWrite;
const UNI: BindingKind = BindingKind::Uniform;

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            println!(
                "GPU: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Local metalForge vs Upstream BarraCUDA Wrapper Benchmarks  ║");
    println!("║  Warmup: {WARMUP}, Iterations: {ITERATIONS}                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let results = vec![
        bench_fitness(&gpu),
        bench_hamming(&gpu),
        bench_jaccard(&gpu),
        bench_locus_var(&gpu),
        bench_spatial(&gpu),
        bench_ipr(&gpu),
        bench_hill_gate(&gpu),
        extended::bench_multi_obj(&gpu),
        extended::bench_pairwise_l2(&gpu),
        extended::bench_swarm_nn(&gpu),
    ];

    bench::print_summary(&results);
}

// ─── Batch Fitness ───────────────────────────────────────────────────

const FITNESS_WGSL: &str = neural_spring_forge::shaders::BATCH_FITNESS_EVAL;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FitnessParams {
    pop_size: u32,
    genome_len: u32,
}

fn bench_fitness(gpu: &Gpu) -> BenchResult {
    let pop_size = 10_000_u32;
    let genome_len = 32_u32;
    let mut rng = Rng::new(42);
    let pop: Vec<f32> = (0..pop_size * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights: Vec<f32> = (0..genome_len).map(|_| rng.uniform() as f32).collect();

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_fitness"),
        source: wgpu::ShaderSource::Wgsl(FITNESS_WGSL.into()),
    });
    let (pipeline, bgl) =
        create_pipeline(device, &shader, "batch_fitness_linear", &[SR, SR, SW, UNI]);
    let pop_buf = device.create_buffer_init(&buf_desc("pop", &pop, wgpu::BufferUsages::STORAGE));
    let wt_buf = device.create_buffer_init(&buf_desc("wt", &weights, wgpu::BufferUsages::STORAGE));
    let fit_buf = alloc_f32(device, pop_size as usize);
    let params_buf = device.create_buffer_init(&buf_desc(
        "p",
        &[FitnessParams {
            pop_size,
            genome_len,
        }],
        wgpu::BufferUsages::UNIFORM,
    ));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            be(0, &pop_buf),
            be(1, &wt_buf),
            be(2, &fit_buf),
            be(3, &params_buf),
        ],
    });
    let wg = pop_size.div_ceil(256);
    let local_us = bench::time_dispatch(
        &DispatchParams {
            device,
            queue,
            gpu,
            pipeline: &pipeline,
            bg: &bg,
            workgroups: wg,
            readback_buf: &fit_buf,
            readback_count: pop_size as usize,
        },
        WARMUP,
        ITERATIONS,
    );

    let op = BatchFitnessGpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = bench::time_upstream(WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, pop_size as usize);
        op.dispatch(&pop_buf, &wt_buf, &out, pop_size, genome_len);
        gpu.read_buffer_f32(&out, pop_size as usize).ok();
    });

    BenchResult {
        name: format!("BatchFitness {pop_size}×{genome_len}"),
        origin: "neuralSpring 011-015",
        local_us,
        upstream_us,
    }
}

// ─── Pairwise Hamming ────────────────────────────────────────────────

const HAMMING_WGSL: &str = neural_spring_forge::shaders::PAIRWISE_HAMMING;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HammingParams {
    n_seqs: u32,
    seq_len: u32,
}

fn bench_hamming(gpu: &Gpu) -> BenchResult {
    let n_seqs = 200_u32;
    let seq_len = 500_u32;
    let mut rng = Rng::new(42);
    let seqs: Vec<u32> = (0..n_seqs * seq_len).map(|_| rng.usize(4) as u32).collect();
    let n_pairs = (n_seqs * (n_seqs - 1) / 2) as usize;

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_hamming"),
        source: wgpu::ShaderSource::Wgsl(HAMMING_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "pairwise_hamming", &[SR, SW, UNI]);
    let seq_buf = device.create_buffer_init(&buf_desc("seqs", &seqs, wgpu::BufferUsages::STORAGE));
    let dist_buf = alloc_f32(device, n_pairs);
    let params_buf = device.create_buffer_init(&buf_desc(
        "p",
        &[HammingParams { n_seqs, seq_len }],
        wgpu::BufferUsages::UNIFORM,
    ));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[be(0, &seq_buf), be(1, &dist_buf), be(2, &params_buf)],
    });
    let wg = (n_pairs as u32).div_ceil(256);
    let local_us = bench::time_dispatch(
        &DispatchParams {
            device,
            queue,
            gpu,
            pipeline: &pipeline,
            bg: &bg,
            workgroups: wg,
            readback_buf: &dist_buf,
            readback_count: n_pairs,
        },
        WARMUP,
        ITERATIONS,
    );

    let op = PairwiseHammingGpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = bench::time_upstream(WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, n_pairs);
        op.dispatch(&seq_buf, &out, n_seqs, seq_len);
        gpu.read_buffer_f32(&out, n_pairs).ok();
    });

    BenchResult {
        name: format!("Hamming {n_seqs}×{seq_len}"),
        origin: "neuralSpring 017 (SATé)",
        local_us,
        upstream_us,
    }
}

// ─── Pairwise Jaccard ────────────────────────────────────────────────

const JACCARD_WGSL: &str = neural_spring_forge::shaders::PAIRWISE_JACCARD;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct JaccardParams {
    n_genomes: u32,
    n_genes: u32,
}

fn bench_jaccard(gpu: &Gpu) -> BenchResult {
    let n_genomes = 100_u32;
    let n_genes = 500_u32;
    let mut rng = Rng::new(42);
    let pa: Vec<f32> = (0..n_genes * n_genomes)
        .map(|_| if rng.uniform() < 0.5 { 1.0 } else { 0.0 })
        .collect();
    let n_pairs = (n_genomes * (n_genomes - 1) / 2) as usize;

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_jaccard"),
        source: wgpu::ShaderSource::Wgsl(JACCARD_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "pairwise_jaccard", &[SR, SW, UNI]);
    let pa_buf = device.create_buffer_init(&buf_desc("pa", &pa, wgpu::BufferUsages::STORAGE));
    let dist_buf = alloc_f32(device, n_pairs);
    let params_buf = device.create_buffer_init(&buf_desc(
        "p",
        &[JaccardParams { n_genomes, n_genes }],
        wgpu::BufferUsages::UNIFORM,
    ));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[be(0, &pa_buf), be(1, &dist_buf), be(2, &params_buf)],
    });
    let wg = (n_pairs as u32).div_ceil(256);
    let local_us = bench::time_dispatch(
        &DispatchParams {
            device,
            queue,
            gpu,
            pipeline: &pipeline,
            bg: &bg,
            workgroups: wg,
            readback_buf: &dist_buf,
            readback_count: n_pairs,
        },
        WARMUP,
        ITERATIONS,
    );

    let op = PairwiseJaccardGpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = bench::time_upstream(WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, n_pairs);
        op.dispatch(&pa_buf, &out, n_genomes, n_genes);
        gpu.read_buffer_f32(&out, n_pairs).ok();
    });

    BenchResult {
        name: format!("Jaccard {n_genomes}×{n_genes}"),
        origin: "neuralSpring 024 (Pangenome)",
        local_us,
        upstream_us,
    }
}

// ─── Locus Variance ──────────────────────────────────────────────────

const LOCUS_VAR_WGSL: &str = neural_spring_forge::shaders::LOCUS_VARIANCE;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LocusParams {
    n_pops: u32,
    n_loci: u32,
}

fn bench_locus_var(gpu: &Gpu) -> BenchResult {
    let n_pops = 50_u32;
    let n_loci = 500_u32;
    let mut rng = Rng::new(42);
    let freqs: Vec<f32> = (0..n_pops * n_loci).map(|_| rng.uniform() as f32).collect();

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_locus"),
        source: wgpu::ShaderSource::Wgsl(LOCUS_VAR_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "locus_variance", &[SR, SW, UNI]);
    let freq_buf =
        device.create_buffer_init(&buf_desc("freqs", &freqs, wgpu::BufferUsages::STORAGE));
    let var_buf = alloc_f32(device, n_loci as usize);
    let params_buf = device.create_buffer_init(&buf_desc(
        "p",
        &[LocusParams { n_pops, n_loci }],
        wgpu::BufferUsages::UNIFORM,
    ));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[be(0, &freq_buf), be(1, &var_buf), be(2, &params_buf)],
    });
    let wg = n_loci.div_ceil(256);
    let local_us = bench::time_dispatch(
        &DispatchParams {
            device,
            queue,
            gpu,
            pipeline: &pipeline,
            bg: &bg,
            workgroups: wg,
            readback_buf: &var_buf,
            readback_count: n_loci as usize,
        },
        WARMUP,
        ITERATIONS,
    );

    let op = LocusVarianceGpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = bench::time_upstream(WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, n_loci as usize);
        op.dispatch(&freq_buf, &out, n_pops, n_loci);
        gpu.read_buffer_f32(&out, n_loci as usize).ok();
    });

    BenchResult {
        name: format!("LocusVariance {n_pops}×{n_loci}"),
        origin: "neuralSpring 025 (MetaPop)",
        local_us,
        upstream_us,
    }
}

// ─── Spatial Payoff ──────────────────────────────────────────────────

const SPATIAL_WGSL: &str = neural_spring_forge::shaders::SPATIAL_PAYOFF;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PayoffParams {
    grid_size: u32,
    b_x1000: u32,
    c_x1000: u32,
    _pad: u32,
}

fn bench_spatial(gpu: &Gpu) -> BenchResult {
    let grid_size = 256_u32;
    let mut rng = Rng::new(42);
    let grid: Vec<u32> = (0..grid_size * grid_size)
        .map(|_| u32::from(rng.uniform() > 0.5))
        .collect();
    let n_cells = (grid_size * grid_size) as usize;

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_spatial"),
        source: wgpu::ShaderSource::Wgsl(SPATIAL_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "spatial_payoff", &[SR, SW, UNI]);
    let grid_buf = device.create_buffer_init(&buf_desc("grid", &grid, wgpu::BufferUsages::STORAGE));
    let fit_buf = alloc_f32(device, n_cells);
    let params_buf = device.create_buffer_init(&buf_desc(
        "p",
        &[PayoffParams {
            grid_size,
            b_x1000: 3000,
            c_x1000: 1000,
            _pad: 0,
        }],
        wgpu::BufferUsages::UNIFORM,
    ));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[be(0, &grid_buf), be(1, &fit_buf), be(2, &params_buf)],
    });
    let wg = (grid_size * grid_size).div_ceil(256);
    let local_us = bench::time_dispatch(
        &DispatchParams {
            device,
            queue,
            gpu,
            pipeline: &pipeline,
            bg: &bg,
            workgroups: wg,
            readback_buf: &fit_buf,
            readback_count: n_cells,
        },
        WARMUP,
        ITERATIONS,
    );

    let op = SpatialPayoffGpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = bench::time_upstream(WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, n_cells);
        op.dispatch(&grid_buf, &out, grid_size, 3.0, 1.0);
        gpu.read_buffer_f32(&out, n_cells).ok();
    });

    BenchResult {
        name: format!("SpatialPayoff {grid_size}×{grid_size}"),
        origin: "neuralSpring 019 (GameTheory)",
        local_us,
        upstream_us,
    }
}

// ─── Batch IPR ───────────────────────────────────────────────────────

const IPR_WGSL: &str = neural_spring_forge::shaders::BATCH_IPR;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct IprParams {
    dim: u32,
    n_vectors: u32,
}

fn bench_ipr(gpu: &Gpu) -> BenchResult {
    let dim = 256_u32;
    let n_vectors = 1000_u32;
    let mut rng = Rng::new(42);
    let ev: Vec<f32> = (0..dim * n_vectors).map(|_| rng.uniform() as f32).collect();

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_ipr"),
        source: wgpu::ShaderSource::Wgsl(IPR_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "batch_ipr", &[SR, SW, UNI]);
    let ev_buf = device.create_buffer_init(&buf_desc("ev", &ev, wgpu::BufferUsages::STORAGE));
    let ipr_buf = alloc_f32(device, n_vectors as usize);
    let params_buf = device.create_buffer_init(&buf_desc(
        "p",
        &[IprParams { dim, n_vectors }],
        wgpu::BufferUsages::UNIFORM,
    ));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[be(0, &ev_buf), be(1, &ipr_buf), be(2, &params_buf)],
    });
    let wg = n_vectors.div_ceil(256);
    let local_us = bench::time_dispatch(
        &DispatchParams {
            device,
            queue,
            gpu,
            pipeline: &pipeline,
            bg: &bg,
            workgroups: wg,
            readback_buf: &ipr_buf,
            readback_count: n_vectors as usize,
        },
        WARMUP,
        ITERATIONS,
    );

    let op = BatchIprGpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = bench::time_upstream(WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, n_vectors as usize);
        op.dispatch(&ev_buf, &out, dim, n_vectors);
        gpu.read_buffer_f32(&out, n_vectors as usize).ok();
    });

    BenchResult {
        name: format!("BatchIPR {n_vectors}×{dim}"),
        origin: "neuralSpring 022-023 (Anderson)",
        local_us,
        upstream_us,
    }
}

// ─── Hill Gate (Signal Integration 021 — wetSpring→BarraCuda lineage) ─

const HILL_GATE_WGSL: &str = neural_spring_forge::shaders::HILL_GATE;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HillParams {
    nx: u32,
    ny: u32,
    vmax: f32,
    k1: f32,
    k2: f32,
    n1: f32,
    n2: f32,
    _pad: u32,
}

fn bench_hill_gate(gpu: &Gpu) -> BenchResult {
    let nx = 100_u32;
    let ny = 100_u32;
    let n_total = (nx * ny) as usize;
    let mut rng = Rng::new(42);
    let cdg: Vec<f32> = (0..nx)
        .map(|_| (rng.uniform() as f32).mul_add(4.5, 0.5))
        .collect();
    let ai: Vec<f32> = (0..ny)
        .map(|_| (rng.uniform() as f32).mul_add(4.5, 0.5))
        .collect();

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_hill_gate"),
        source: wgpu::ShaderSource::Wgsl(HILL_GATE_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "hill_gate", &[SR, SR, SW, UNI]);
    let cdg_buf = device.create_buffer_init(&buf_desc("cdg", &cdg, wgpu::BufferUsages::STORAGE));
    let ai_buf = device.create_buffer_init(&buf_desc("ai", &ai, wgpu::BufferUsages::STORAGE));
    let out_buf = alloc_f32(device, n_total);
    let params_buf = device.create_buffer_init(&buf_desc(
        "p",
        &[HillParams {
            nx,
            ny,
            vmax: 1.0,
            k1: 1.0,
            k2: 1.0,
            n1: 2.0,
            n2: 2.0,
            _pad: 0,
        }],
        wgpu::BufferUsages::UNIFORM,
    ));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            be(0, &cdg_buf),
            be(1, &ai_buf),
            be(2, &out_buf),
            be(3, &params_buf),
        ],
    });
    let wg = (nx * ny).div_ceil(256);
    let local_us = bench::time_dispatch(
        &DispatchParams {
            device,
            queue,
            gpu,
            pipeline: &pipeline,
            bg: &bg,
            workgroups: wg,
            readback_buf: &out_buf,
            readback_count: n_total,
        },
        WARMUP,
        ITERATIONS,
    );

    let op = HillGateGpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = bench::time_upstream(WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, n_total);
        op.dispatch(
            &cdg_buf,
            &ai_buf,
            &out,
            &HillGateParams {
                n_a: nx,
                n_b: ny,
                mode: 1,
                _pad: 0,
                k_a: 1.0,
                k_b: 1.0,
                n_a_exp: 2.0,
                n_b_exp: 2.0,
                vmax: 1.0,
                _pad2: 0.0,
            },
        );
        gpu.read_buffer_f32(&out, n_total).ok();
    });

    BenchResult {
        name: format!("HillGate {nx}×{ny}"),
        origin: "neuralSpring 021 (Signal)",
        local_us,
        upstream_us,
    }
}
