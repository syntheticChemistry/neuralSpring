// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-spring shader evolution benchmark.
//!
//! Benchmarks GPU operations that originated from different springs in the
//! ecoPrimals ecosystem, demonstrating how `ToadStool`'s shader absorption
//! enables cross-project performance benefits.
//!
//! ## Spring Origins
//!
//! | Op | Origin | Absorption |
//! |----|--------|------------|
//! | `BatchFitnessGpu` | neuralSpring (ML/evolution) | `77f70b2e` (S-25) |
//! | `PairwiseL2Gpu` | neuralSpring (MODES) | `5437c170` (S-42) |
//! | `BatchIprGpu` | neuralSpring (Anderson) | `77f70b2e` (S-25) |
//! | `HmmBatchForwardF64` | wetSpring (bio/dN-dS) | `a115da8f` (S-39) |
//! | `BatchedEighGpu` | hotSpring (nuclear physics) | `a115da8f` (S-39) |
//! | `SpatialPayoffGpu` | neuralSpring (game theory) | `77f70b2e` (S-25) |
//! | `PairwiseHammingGpu` | neuralSpring (`SATé`) | `77f70b2e` (S-25) |
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --bin bench_cross_spring_evolution
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]

use barracuda::ops::bio::{
    BatchFitnessGpu, HmmBatchForwardF64, PairwiseHammingGpu, PairwiseL2Gpu, SpatialPayoffGpu,
};
use barracuda::ops::linalg::BatchedEighGpu;
use barracuda::spectral::BatchIprGpu;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;

const WARMUP: u32 = 3;
const ITERATIONS: u32 = 20;

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(e) => {
            eprintln!("SKIP: {e}");
            return;
        }
    };

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Cross-Spring Shader Evolution Benchmark");
    println!("  Adapter: {} ({:?})", gpu.adapter_name, gpu.backend);
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    bench_neuralspring_ops(&gpu);
    bench_wetspring_ops(&gpu);
    bench_hotspring_ops(&gpu);

    println!("═══════════════════════════════════════════════════════════════");
    println!("  All benchmarks complete.");
    println!("═══════════════════════════════════════════════════════════════");
}

fn bench_neuralspring_ops(gpu: &Gpu) {
    println!("--- neuralSpring origins (ML / evolution) ---");
    println!();

    let dev = Arc::clone(gpu.wgpu_device());

    // BatchFitnessGpu (Paper 011-015, absorbed S-25)
    {
        let op = BatchFitnessGpu::new(dev.clone());
        let device = gpu.device();
        let pop_size = 1024_u32;
        let genome_len = 64_u32;
        let mut rng = Rng::new(42);
        let pop: Vec<f32> = (0..pop_size * genome_len)
            .map(|_| rng.uniform() as f32)
            .collect();
        let wt: Vec<f32> = (0..genome_len).map(|_| rng.uniform() as f32).collect();
        let pop_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&pop),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let wt_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&wt),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let us = time_op(WARMUP, ITERATIONS, || {
            let fit = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: u64::from(pop_size) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            op.dispatch(&pop_buf, &wt_buf, &fit, pop_size, genome_len);
            gpu.read_buffer_f32(&fit, pop_size as usize).ok();
        });
        println!("  BatchFitnessGpu  1024×64   origin=neuralSpring(S-25)  {us:>8.1} µs");
    }

    // PairwiseL2Gpu (Paper 012, absorbed S-42)
    {
        let op = PairwiseL2Gpu::new(dev.clone());
        let device = gpu.device();
        let n = 128_u32;
        let dim = 16_u32;
        let mut rng = Rng::new(77);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.uniform() as f32).collect();
        let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let us = time_op(WARMUP, ITERATIONS, || {
            let out = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: u64::from(n) * u64::from(n) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            op.dispatch(&in_buf, &out, n, dim);
            gpu.read_buffer_f32(&out, (n * n) as usize).ok();
        });
        println!("  PairwiseL2Gpu    128×16    origin=neuralSpring(S-42)  {us:>8.1} µs");
    }

    // BatchIprGpu (Paper 023, absorbed S-25)
    {
        let op = BatchIprGpu::new(dev.clone());
        let device = gpu.device();
        let dim = 32_u32;
        let n_vecs = 64_u32;
        let mut rng = Rng::new(123);
        let mut evecs: Vec<f32> = (0..n_vecs * dim).map(|_| rng.uniform() as f32).collect();
        for row in evecs.chunks_mut(dim as usize) {
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                row.iter_mut().for_each(|v| *v /= norm);
            }
        }
        let ev_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&evecs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let us = time_op(WARMUP, ITERATIONS, || {
            let ipr_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: u64::from(n_vecs) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            op.dispatch(&ev_buf, &ipr_buf, dim, n_vecs);
            gpu.read_buffer_f32(&ipr_buf, n_vecs as usize).ok();
        });
        println!("  BatchIprGpu      32×64     origin=neuralSpring(S-25)  {us:>8.1} µs");
    }

    // SpatialPayoffGpu (Paper 019, absorbed S-25)
    {
        let op = SpatialPayoffGpu::new(dev.clone());
        let device = gpu.device();
        let grid_size = 32_u32;
        let n = grid_size * grid_size;
        let mut rng = Rng::new(55);
        let grid: Vec<f32> = (0..n)
            .map(|_| if rng.uniform() > 0.5 { 1.0 } else { 0.0 })
            .collect();
        let grid_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&grid),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let us = time_op(WARMUP, ITERATIONS, || {
            let fit = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: u64::from(n) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            op.dispatch(&grid_buf, &fit, grid_size, 3.0, 1.0);
            gpu.read_buffer_f32(&fit, n as usize).ok();
        });
        println!("  SpatialPayoffGpu 32×32     origin=neuralSpring(S-25)  {us:>8.1} µs");
    }

    // PairwiseHammingGpu (Paper 017, absorbed S-25)
    {
        let op = PairwiseHammingGpu::new(dev);
        let device = gpu.device();
        let n_seqs = 64_u32;
        let seq_len = 100_u32;
        let mut rng = Rng::new(88);
        let seqs: Vec<f32> = (0..n_seqs * seq_len)
            .map(|_| (rng.uniform() * 4.0).floor() as f32)
            .collect();
        let seq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&seqs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let us = time_op(WARMUP, ITERATIONS, || {
            let dist = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: u64::from(n_seqs) * u64::from(n_seqs) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            op.dispatch(&seq_buf, &dist, n_seqs, seq_len);
            gpu.read_buffer_f32(&dist, (n_seqs * n_seqs) as usize).ok();
        });
        println!("  PairwiseHammingGpu 64×100  origin=neuralSpring(S-25)  {us:>8.1} µs");
    }

    println!();
}

fn bench_wetspring_ops(gpu: &Gpu) {
    println!("--- wetSpring origins (bio / genomics) ---");
    println!();

    let dev = Arc::clone(gpu.wgpu_device());

    // HmmBatchForwardF64 (wetSpring dN/dS, absorbed S-39)
    {
        match HmmBatchForwardF64::new(dev) {
            Ok(op) => {
                let device = gpu.device();
                let n_states = 4_u32;
                let n_symbols = 3_u32;
                let n_steps = 50_u32;
                let n_seqs = 32_u32;

                let mut rng = Rng::new(42);
                let mut make_log = |len: u32| -> Vec<f64> {
                    (0..len)
                        .map(|_| rng.uniform().mul_add(0.5, 0.01).ln())
                        .collect()
                };

                let log_trans = make_log(n_states * n_states);
                let log_emit = make_log(n_states * n_symbols);
                let log_pi = make_log(n_states);
                let obs: Vec<u32> = (0..n_seqs * n_steps)
                    .map(|_| (rng.uniform() * f64::from(n_symbols)).floor() as u32)
                    .collect();

                let trans_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&log_trans),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let emit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&log_emit),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let pi_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&log_pi),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let obs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&obs),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let us = time_op(WARMUP, ITERATIONS, || {
                    let alpha_size =
                        u64::from(n_seqs) * u64::from(n_steps) * u64::from(n_states) * 8;
                    let alpha = device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: alpha_size,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });
                    let ll = device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: u64::from(n_seqs) * 8,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });
                    let _ = op.dispatch(
                        n_states, n_symbols, n_steps, n_seqs, &trans_buf, &emit_buf, &pi_buf,
                        &obs_buf, &alpha, &ll,
                    );
                    gpu.read_buffer_f64(&ll, n_seqs as usize).ok();
                });
                println!("  HmmBatchForwardF64 4s×50t×32b  origin=wetSpring(S-39)   {us:>8.1} µs");
            }
            Err(e) => {
                println!("  HmmBatchForwardF64 SKIP: {e}");
            }
        }
    }

    println!();
}

fn bench_hotspring_ops(gpu: &Gpu) {
    println!("--- hotSpring origins (physics / precision) ---");
    println!();

    let dev = Arc::clone(gpu.wgpu_device());

    // BatchedEighGpu (nuclear physics / spectral theory, absorbed S-39)
    {
        let n = 12_usize;
        let batch = 40_usize;
        let mut rng = Rng::new(42);

        let mut data = vec![0.0_f64; batch * n * n];
        for b in 0..batch {
            for i in 0..n {
                for j in i..n {
                    let v = rng.uniform().mul_add(2.0, -1.0);
                    data[b * n * n + i * n + j] = v;
                    data[b * n * n + j * n + i] = v;
                }
            }
        }

        let us = time_op(WARMUP, ITERATIONS, || {
            let _ =
                BatchedEighGpu::execute_single_dispatch(dev.clone(), &data, n, batch, 30, 1e-12);
        });
        println!("  BatchedEighGpu 12×12×40     origin=hotSpring(S-39)    {us:>8.1} µs");
    }

    println!();
}

fn time_op(warmup: u32, iterations: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    elapsed.as_micros() as f64 / f64::from(iterations)
}
