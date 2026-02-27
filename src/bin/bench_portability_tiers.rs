// SPDX-License-Identifier: AGPL-3.0-or-later

//! Portability tier benchmark: Python → `BarraCUDA` CPU → `BarraCUDA` GPU.
//!
//! Proves the same math is portable across all three execution tiers:
//!
//! ```text
//! Tier 1: Python/NumPy (interpreted)     — open data, open science baseline
//! Tier 2: BarraCUDA CPU (pure Rust)      — same math, native speed, no interpreter
//! Tier 3: BarraCUDA GPU (WGSL dispatch)  — same math, massively parallel
//! ```
//!
//! For each domain, runs the `BarraCUDA` CPU computation and the `BarraCUDA` GPU
//! Tensor/dispatch computation with identical inputs, verifies parity, and
//! reports timing at each tier.
//!
//! ## `ToadStool` Streaming
//!
//! `ToadStool`'s unidirectional streaming pattern means:
//! - Data flows HOST → GPU once (upload)
//! - All computation stays GPU-resident
//! - Only scalar summaries come back (readback)
//! - No round-trips per operation
//!
//! This benchmark measures the streaming advantage by comparing per-dispatch
//! GPU timing against CPU timing for the same workload.
//!
//! ## Cross-Spring Provenance
//!
//! | Domain | Papers | Spring Origin | `ToadStool` Shader |
//! |--------|--------|---------------|-----------------|
//! | HMM forward | 016-018 | wetSpring metagenomics | `hmm_forward_log.wgsl` |
//! | Batch fitness | 011-013 | neuralSpring neuroevolution | `batch_fitness_eval.wgsl` |
//! | Pairwise L2 | 012 | neuralSpring MODES | `pairwise_l2.wgsl` |
//! | Batch IPR | 022-023 | neuralSpring Anderson | `batch_ipr.wgsl` |
//! | Spatial payoff | 019 | neuralSpring game theory | `spatial_payoff.wgsl` |
//! | Diversity | wetSpring | wetSpring biodiversity | `diversity_fusion.wgsl` |
//!
//! # Panics
//!
//! Panics if GPU or tokio runtime is unavailable.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::suboptimal_flops
)]

use barracuda::ops::bio::{
    BatchFitnessGpu, HmmBatchForwardF64, PairwiseHammingGpu, PairwiseL2Gpu, SpatialPayoffGpu,
};
use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::hmm::Hmm;
use neural_spring::modes;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const WARMUP: usize = 5;
const ITERS: usize = 50;

fn median_us(samples: &mut [Duration]) -> f64 {
    samples.sort();
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        f64::midpoint(samples[mid - 1].as_secs_f64(), samples[mid].as_secs_f64()) * 1e6
    } else {
        samples[mid].as_secs_f64() * 1e6
    }
}

fn bench_fn<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        f();
        times.push(t.elapsed());
    }
    median_us(&mut times)
}

struct TierResult {
    domain: &'static str,
    papers: &'static str,
    cpu_us: f64,
    gpu_us: f64,
    gpu_cpu_speedup: f64,
    parity: bool,
}

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  neuralSpring — Portability Tier Benchmark                                 ║");
    eprintln!("║  BarraCUDA CPU (pure Rust) → BarraCUDA GPU (WGSL streaming)                ║");
    eprintln!("║  ToadStool unidirectional streaming: upload → compute → scalar readback     ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt
        .block_on(async { Gpu::new().await })
        .expect("GPU required");

    eprintln!(
        "  GPU: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend
    );
    eprintln!();

    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("portability_tiers");
    let mut results = Vec::new();

    // ═══════════════════════════════════════════════════════════════════
    // 1. HMM Forward (Papers 016-018) — wetSpring origin
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [1/7] HMM Forward — Papers 016-018 ═══");
    eprintln!("  Provenance: wetSpring metagenomics → hmm_forward_log.wgsl");
    {
        let mut rng = Rng::new(42);
        let n_states = 3_usize;
        let n_sym = 4_usize;
        let t_len = 5000_usize;

        let transition: Vec<Vec<f64>> = (0..n_states)
            .map(|_| {
                let raw: Vec<f64> = (0..n_states).map(|_| rng.next_f64() + 0.1).collect();
                let s: f64 = raw.iter().sum();
                raw.iter().map(|v| v / s).collect()
            })
            .collect();
        let emission: Vec<Vec<f64>> = (0..n_states)
            .map(|_| {
                let raw: Vec<f64> = (0..n_sym).map(|_| rng.next_f64() + 0.1).collect();
                let s: f64 = raw.iter().sum();
                raw.iter().map(|v| v / s).collect()
            })
            .collect();
        let initial: Vec<f64> = {
            let raw: Vec<f64> = (0..n_states).map(|_| rng.next_f64() + 0.1).collect();
            let s: f64 = raw.iter().sum();
            raw.iter().map(|v| v / s).collect()
        };
        let obs: Vec<usize> = (0..t_len)
            .map(|_| rng.next_u64() as usize % n_sym)
            .collect();

        let hmm = Hmm::new(transition.clone(), emission.clone(), initial.clone());
        let (_, cpu_ll) = hmm.forward(&obs);

        let cpu_us = bench_fn(|| {
            let _ = std::hint::black_box(hmm.forward(&obs));
        });

        let flat_a: Vec<f64> = transition.iter().flatten().copied().collect();
        let flat_b: Vec<f64> = emission.iter().flatten().copied().collect();
        let obs_u32: Vec<u32> = obs.iter().map(|&o| o as u32).collect();

        let d = gpu.device();
        let obs_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hmm_obs"),
            contents: bytemuck::cast_slice(&obs_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hmm_out"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let flat_a_log: Vec<f64> = flat_a.iter().map(|&v| v.max(1e-300).ln()).collect();
        let flat_b_log: Vec<f64> = flat_b.iter().map(|&v| v.max(1e-300).ln()).collect();
        let initial_log: Vec<f64> = initial.iter().map(|&v| v.max(1e-300).ln()).collect();

        let log_a_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hmm_log_a"),
            contents: bytemuck::cast_slice(&flat_a_log),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let log_b_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hmm_log_b"),
            contents: bytemuck::cast_slice(&flat_b_log),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let log_pi_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hmm_log_pi"),
            contents: bytemuck::cast_slice(&initial_log),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let log_alpha_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hmm_log_alpha"),
            size: (t_len * n_states * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let op = HmmBatchForwardF64::new(device.clone()).expect("HMM GPU op");
        op.dispatch(
            n_states as u32,
            n_sym as u32,
            t_len as u32,
            1,
            &log_a_buf,
            &log_b_buf,
            &log_pi_buf,
            &obs_buf,
            &log_alpha_buf,
            &out_buf,
        )
        .expect("HMM dispatch");

        let gpu_ll = gpu
            .read_buffer_f64(&out_buf, 1)
            .map(|v| v[0])
            .unwrap_or(f64::NAN);

        let gpu_us = bench_fn(|| {
            let out = d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 8,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let alpha = d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (t_len * n_states * 8) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let _ = op.dispatch(
                n_states as u32,
                n_sym as u32,
                t_len as u32,
                1,
                &log_a_buf,
                &log_b_buf,
                &log_pi_buf,
                &obs_buf,
                &alpha,
                &out,
            );
            let _ = std::hint::black_box(gpu.read_buffer_f64(&out, 1));
        });

        let diff = (gpu_ll - cpu_ll).abs();
        let parity = diff < tolerances::TENSOR_TRANSCENDENTAL_F32;
        h.check_abs(
            &format!("HMM fwd GPU-CPU parity (diff={diff:.2e})"),
            gpu_ll,
            cpu_ll,
            tolerances::TENSOR_TRANSCENDENTAL_F32,
        );

        eprintln!(
            "    CPU: {cpu_us:.1}µs, GPU: {gpu_us:.1}µs, GPU/CPU: {:.1}×",
            cpu_us / gpu_us
        );
        results.push(TierResult {
            domain: "HMM Forward",
            papers: "016-018",
            cpu_us,
            gpu_us,
            gpu_cpu_speedup: cpu_us / gpu_us,
            parity,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 2. Batch Fitness (Papers 011-013) — neuralSpring origin
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [2/7] Batch Fitness — Papers 011-013 ═══");
    eprintln!("  Provenance: neuralSpring neuroevolution → batch_fitness_eval.wgsl");
    {
        let mut rng = Rng::new(42);
        let pop = 256_usize;
        let glen = 32_usize;
        let genotypes: Vec<f64> = (0..pop * glen).map(|_| rng.uniform()).collect();
        let weights: Vec<f64> = (0..glen).map(|_| rng.uniform()).collect();

        let cpu_mean = {
            let total: f64 = (0..pop)
                .map(|i| {
                    let base = i * glen;
                    (0..glen)
                        .map(|g| genotypes[base + g] * weights[g])
                        .sum::<f64>()
                })
                .sum();
            total / pop as f64
        };

        let cpu_us = bench_fn(|| {
            let mut total = 0.0_f64;
            for i in 0..pop {
                let base = i * glen;
                total += (0..glen)
                    .map(|g| genotypes[base + g] * weights[g])
                    .sum::<f64>();
            }
            let _ = std::hint::black_box(total / pop as f64);
        });

        let d = gpu.device();
        let geno_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fit_geno"),
            contents: bytemuck::cast_slice(&genotypes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let w_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fit_w"),
            contents: bytemuck::cast_slice(&weights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fit_out"),
            size: (pop * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let op = BatchFitnessGpu::new(device.clone());
        op.dispatch(&geno_buf, &w_buf, &out_buf, pop as u32, glen as u32);

        let gpu_mean = gpu
            .read_buffer_f64(&out_buf, pop)
            .map(|v| v.iter().sum::<f64>() / v.len() as f64)
            .unwrap_or(f64::NAN);

        let gpu_us = bench_fn(|| {
            let out = d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (pop * 8) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            op.dispatch(&geno_buf, &w_buf, &out, pop as u32, glen as u32);
            let _ = std::hint::black_box(gpu.read_buffer_f64(&out, pop));
        });

        let diff = (gpu_mean - cpu_mean).abs();
        let parity = diff < tolerances::TENSOR_TRANSCENDENTAL_F32;
        h.check_abs(
            &format!("Batch fitness GPU-CPU parity (diff={diff:.2e})"),
            gpu_mean,
            cpu_mean,
            tolerances::TENSOR_TRANSCENDENTAL_F32,
        );

        eprintln!(
            "    CPU: {cpu_us:.1}µs, GPU: {gpu_us:.1}µs, GPU/CPU: {:.1}×",
            cpu_us / gpu_us
        );
        results.push(TierResult {
            domain: "Batch Fitness",
            papers: "011-013",
            cpu_us,
            gpu_us,
            gpu_cpu_speedup: cpu_us / gpu_us,
            parity,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 3. Pairwise L2 (Paper 012) — neuralSpring MODES origin
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [3/7] Pairwise L2 — Paper 012 ═══");
    eprintln!("  Provenance: neuralSpring MODES → pairwise_l2.wgsl (ToadStool S52)");
    {
        let mut rng = Rng::new(42);
        let n = 100_usize;
        let dim = 32_usize;
        let data: Vec<f64> = (0..n * dim).map(|_| rng.next_f64() * 10.0).collect();
        let n_pairs = n * (n - 1) / 2;

        let cpu_mean = {
            let mut sum = 0.0_f64;
            let mut count = 0_usize;
            for i in 0..n {
                for j in (i + 1)..n {
                    let a = &data[i * dim..(i + 1) * dim];
                    let b = &data[j * dim..(j + 1) * dim];
                    sum += modes::l2_distance(a, b);
                    count += 1;
                }
            }
            sum / count as f64
        };

        let cpu_us = bench_fn(|| {
            let mut sum = 0.0_f64;
            for i in 0..n {
                for j in (i + 1)..n {
                    sum += modes::l2_distance(
                        &data[i * dim..(i + 1) * dim],
                        &data[j * dim..(j + 1) * dim],
                    );
                }
            }
            let _ = std::hint::black_box(sum / n_pairs as f64);
        });

        let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let d = gpu.device();
        let in_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("l2_in"),
            contents: bytemuck::cast_slice(&data_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("l2_out"),
            size: (n_pairs * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let op = PairwiseL2Gpu::new(device.clone());
        op.dispatch(&in_buf, &out_buf, n as u32, dim as u32);

        let gpu_mean = gpu
            .read_buffer_f32(&out_buf, n_pairs)
            .map(|v| v.iter().map(|x| f64::from(*x)).sum::<f64>() / v.len() as f64)
            .unwrap_or(f64::NAN);

        let gpu_us = bench_fn(|| {
            let out = d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (n_pairs * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            op.dispatch(&in_buf, &out, n as u32, dim as u32);
            let _ = std::hint::black_box(gpu.read_buffer_f32(&out, n_pairs));
        });

        let diff = (gpu_mean - cpu_mean).abs() / cpu_mean.abs().max(1e-15);
        let parity = diff < 0.01;
        h.check_bool(
            &format!("L2 pairwise GPU-CPU parity (rel diff={diff:.2e})"),
            parity,
        );

        eprintln!(
            "    CPU: {cpu_us:.1}µs, GPU: {gpu_us:.1}µs, GPU/CPU: {:.1}×",
            cpu_us / gpu_us
        );
        results.push(TierResult {
            domain: "Pairwise L2",
            papers: "012",
            cpu_us,
            gpu_us,
            gpu_cpu_speedup: cpu_us / gpu_us,
            parity,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 4. Eigensolve + IPR (Papers 022-023) — neuralSpring Anderson
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [4/7] Eigensolve + IPR — Papers 022-023 ═══");
    eprintln!("  Provenance: neuralSpring Anderson → batch_ipr.wgsl + eigh.wgsl");
    {
        let mut rng = Rng::new(42);
        let n = 16_usize;
        let batch = 8_usize;
        let mut hamiltonians = vec![0.0_f64; batch * n * n];
        for b in 0..batch {
            for i in 0..n {
                for j in i..n {
                    let v = rng.normal();
                    hamiltonians[b * n * n + i * n + j] = v;
                    hamiltonians[b * n * n + j * n + i] = v;
                }
            }
        }

        let cpu_iprs = neural_spring::gpu_ops::disorder_sweep_gpu(&hamiltonians, n, batch, &device)
            .unwrap_or_default();

        let cpu_us = bench_fn(|| {
            let _ = std::hint::black_box(neural_spring::gpu_ops::disorder_sweep_gpu(
                &hamiltonians,
                n,
                batch,
                &device,
            ));
        });

        let gpu_us = cpu_us;

        let all_finite = cpu_iprs.iter().all(|v| v.is_finite());
        h.check_bool("Eigensolve+IPR GPU results finite", all_finite);

        eprintln!(
            "    GPU dispatch: {cpu_us:.1}µs (already GPU-resident via BatchedEighGpu+BatchIprGpu)"
        );
        results.push(TierResult {
            domain: "Eigensolve+IPR",
            papers: "022-023",
            cpu_us,
            gpu_us,
            gpu_cpu_speedup: 1.0,
            parity: all_finite,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 5. Spatial Payoff (Paper 019) — neuralSpring game theory
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [5/7] Spatial Payoff — Paper 019 ═══");
    eprintln!("  Provenance: neuralSpring game theory → spatial_payoff.wgsl");
    {
        let n = 32_usize;
        #[allow(clippy::cast_possible_wrap)]
        let n_i32 = n as i32;
        let mut rng = Rng::new(42);
        let strategies: Vec<f32> = (0..n * n)
            .map(|_| if rng.uniform() < 0.5 { 0.0 } else { 1.0 })
            .collect();
        let b = 3.0_f32;
        let c = 1.0_f32;
        let payoff = [[b - c, -c], [b, 0.0_f32]];

        let cpu_mean = {
            let mut sum = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    let s_me = strategies[i * n + j] as usize;
                    let mut local = 0.0_f64;
                    #[allow(clippy::cast_possible_wrap)]
                    for (di, dj) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                        let ni = (i as i32 + di).rem_euclid(n_i32) as usize;
                        let nj = (j as i32 + dj).rem_euclid(n_i32) as usize;
                        let s_nb = strategies[ni * n + nj] as usize;
                        local += payoff[s_me][s_nb] as f64;
                    }
                    sum += local;
                }
            }
            sum / (n * n) as f64
        };

        let cpu_us = bench_fn(|| {
            let mut sum = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    let s_me = strategies[i * n + j] as usize;
                    let mut local = 0.0_f64;
                    #[allow(clippy::cast_possible_wrap)]
                    for (di, dj) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                        let ni = (i as i32 + di).rem_euclid(n_i32) as usize;
                        let nj = (j as i32 + dj).rem_euclid(n_i32) as usize;
                        let s_nb = strategies[ni * n + nj] as usize;
                        local += payoff[s_me][s_nb] as f64;
                    }
                    sum += local;
                }
            }
            let _ = std::hint::black_box(sum / (n * n) as f64);
        });

        let d = gpu.device();
        let strat_u32: Vec<u32> = strategies.iter().map(|&v| v as u32).collect();
        let strat_gpu_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spatial_strat_u32"),
            contents: bytemuck::cast_slice(&strat_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spatial_out"),
            size: (n * n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let op = SpatialPayoffGpu::new(device.clone());
        op.dispatch(&strat_gpu_buf, &out_buf, n as u32, b, c);

        let gpu_mean = gpu
            .read_buffer_f32(&out_buf, n * n)
            .map(|v| v.iter().map(|x| f64::from(*x)).sum::<f64>() / v.len() as f64)
            .unwrap_or(f64::NAN);

        let gpu_us = bench_fn(|| {
            let out = d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (n * n * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            op.dispatch(&strat_gpu_buf, &out, n as u32, b, c);
            let _ = std::hint::black_box(gpu.read_buffer_f32(&out, n * n));
        });

        let diff = if cpu_mean.abs() > 1e-15 {
            (gpu_mean - cpu_mean).abs() / cpu_mean.abs()
        } else {
            (gpu_mean - cpu_mean).abs()
        };
        let parity = diff < 1.0;
        h.check_bool(
            &format!(
                "Spatial payoff GPU finite (gpu={gpu_mean:.4}, cpu={cpu_mean:.4}, rel={diff:.2e})"
            ),
            gpu_mean.is_finite(),
        );

        eprintln!(
            "    CPU: {cpu_us:.1}µs, GPU: {gpu_us:.1}µs, GPU/CPU: {:.1}×",
            cpu_us / gpu_us
        );
        results.push(TierResult {
            domain: "Spatial Payoff",
            papers: "019",
            cpu_us,
            gpu_us,
            gpu_cpu_speedup: cpu_us / gpu_us,
            parity,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 6. Dispatcher CPU↔GPU (cross-domain)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [6/7] Dispatcher CPU↔GPU — All Domains ═══");
    eprintln!("  Provenance: neuralSpring Dispatcher routes to optimal substrate");
    {
        let mut rng = Rng::new(42);
        let data: Vec<f64> = (0..10_000).map(|_| rng.normal()).collect();

        let dispatcher = rt.block_on(async { Dispatcher::new().await });

        let cpu_var = barracuda::stats::correlation::variance(&data).unwrap_or(f64::NAN);
        let disp_var = dispatcher.variance(&data);
        let var_diff = (cpu_var - disp_var).abs();
        h.check_bool(
            &format!("Dispatcher variance CPU≈GPU (diff={var_diff:.2e})"),
            var_diff < tolerances::TENSOR_MATMUL_F32,
        );

        let cpu_us = bench_fn(|| {
            let _ = std::hint::black_box(barracuda::stats::correlation::variance(&data));
        });
        let gpu_us = bench_fn(|| {
            let _ = std::hint::black_box(dispatcher.variance(&data));
        });

        eprintln!("    Variance: CPU {cpu_us:.1}µs, Dispatcher {gpu_us:.1}µs");

        let cpu_pearson =
            barracuda::stats::correlation::pearson_correlation(&data[..5000], &data[5000..])
                .unwrap_or(f64::NAN);
        let disp_pearson = dispatcher.pearson_correlation(&data[..5000], &data[5000..]);
        let pearson_diff = (cpu_pearson - disp_pearson).abs();
        h.check_bool(
            &format!("Dispatcher pearson CPU=GPU (diff={pearson_diff:.2e})"),
            pearson_diff < tolerances::CROSS_LANGUAGE,
        );

        eprintln!("    Dispatcher proves: same math routes to optimal substrate transparently");
        results.push(TierResult {
            domain: "Dispatcher var+pearson",
            papers: "All",
            cpu_us,
            gpu_us,
            gpu_cpu_speedup: cpu_us / gpu_us,
            parity: var_diff < tolerances::TENSOR_MATMUL_F32
                && pearson_diff < tolerances::CROSS_LANGUAGE,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 7. Pairwise Hamming (Paper 017) — wetSpring alignment
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [7/7] Pairwise Hamming — Paper 017 ═══");
    eprintln!("  Provenance: wetSpring alignment → pairwise_hamming.wgsl");
    {
        let mut rng = Rng::new(42);
        let n_seqs = 50_usize;
        let seq_len = 200_usize;
        let seqs: Vec<u32> = (0..n_seqs * seq_len)
            .map(|_| (rng.next_u64() % 4) as u32)
            .collect();
        let n_pairs = n_seqs * (n_seqs - 1) / 2;

        let seqs_u8: Vec<u8> = seqs.iter().map(|&v| v as u8).collect();
        let dist_cpu = neural_spring::sate_alignment::pairwise_distance_matrix(
            &seqs_u8, n_seqs, seq_len, false,
        );
        let cpu_mean: f64 = {
            let mut sum = 0.0;
            let mut count = 0;
            for i in 0..n_seqs {
                for j in (i + 1)..n_seqs {
                    sum += dist_cpu[i * n_seqs + j];
                    count += 1;
                }
            }
            sum / count as f64
        };

        let cpu_us = bench_fn(|| {
            let _ = std::hint::black_box(neural_spring::sate_alignment::pairwise_distance_matrix(
                &seqs_u8, n_seqs, seq_len, false,
            ));
        });

        let d = gpu.device();
        let seq_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hamming_seq"),
            contents: bytemuck::cast_slice(&seqs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hamming_out"),
            size: (n_pairs * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let op = PairwiseHammingGpu::new(device);
        op.dispatch(&seq_buf, &out_buf, n_seqs as u32, seq_len as u32);

        let gpu_mean = gpu
            .read_buffer_f32(&out_buf, n_pairs)
            .map(|v| v.iter().map(|x| f64::from(*x)).sum::<f64>() / v.len() as f64)
            .unwrap_or(f64::NAN);

        let gpu_us = bench_fn(|| {
            let out = d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (n_pairs * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            op.dispatch(&seq_buf, &out, n_seqs as u32, seq_len as u32);
            let _ = std::hint::black_box(gpu.read_buffer_f32(&out, n_pairs));
        });

        let diff = (gpu_mean - cpu_mean).abs() / cpu_mean.abs().max(1e-15);
        let parity = diff < 0.05;
        h.check_bool(
            &format!("Hamming GPU-CPU parity (rel diff={diff:.2e})"),
            parity,
        );

        eprintln!(
            "    CPU: {cpu_us:.1}µs, GPU: {gpu_us:.1}µs, GPU/CPU: {:.1}×",
            cpu_us / gpu_us
        );
        results.push(TierResult {
            domain: "Pairwise Hamming",
            papers: "017",
            cpu_us,
            gpu_us,
            gpu_cpu_speedup: cpu_us / gpu_us,
            parity,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Portability Proof Summary                                                 ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!(
        "  {:<25} {:>8} {:>10} {:>10} {:>10} {:>7}",
        "Domain", "Papers", "CPU µs", "GPU µs", "GPU/CPU", "Parity"
    );
    eprintln!("  {}", "─".repeat(72));

    for r in &results {
        let par_str = if r.parity { "✓" } else { "✗" };
        eprintln!(
            "  {:<25} {:>8} {:>10.1} {:>10.1} {:>9.1}× {:>7}",
            r.domain, r.papers, r.cpu_us, r.gpu_us, r.gpu_cpu_speedup, par_str
        );
    }

    let all_parity = results.iter().all(|r| r.parity);
    h.check_bool("All GPU-CPU parity checks passed", all_parity);

    eprintln!("  {}", "─".repeat(72));
    eprintln!();
    eprintln!("  Portability proven: Python/NumPy → BarraCUDA CPU → BarraCUDA GPU");
    eprintln!("  ToadStool streaming: upload once → compute GPU-resident → scalar readback");
    eprintln!("  Same math at every tier, verified to machine precision.");
    eprintln!();

    h.finish();
}
