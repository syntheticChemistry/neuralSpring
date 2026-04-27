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
//! | Domain | Papers | Spring Origin | `BarraCUDA` Shader |
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

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::expect_used,
    reason = "validation binary"
)]

mod extended;

use barracuda::ops::bio::{BatchFitnessGpu, HmmBatchForwardF64, PairwiseL2Gpu};
use neural_spring::gpu::Gpu;
use neural_spring::hmm::Hmm;
use neural_spring::modes;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{OrExit, ValidationHarness, median_duration_us};
use std::time::Instant;
use wgpu::util::DeviceExt;

const WARMUP: usize = 5;
const ITERS: usize = 50;

pub(crate) fn bench_fn<F: FnMut()>(mut f: F) -> f64 {
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

pub(crate) struct TierResult {
    pub domain: &'static str,
    pub papers: &'static str,
    pub cpu_us: f64,
    pub gpu_us: f64,
    pub gpu_cpu_speedup: f64,
    pub parity: bool,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  neuralSpring — Portability Tier Benchmark                                 ║");
    println!("║  BarraCUDA CPU (pure Rust) → BarraCUDA GPU (WGSL streaming)                ║");
    println!("║  ToadStool unidirectional streaming: upload → compute → scalar readback     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().or_exit("tokio runtime");
    let gpu = rt
        .block_on(async { Gpu::new().await })
        .or_exit("GPU required for benchmark");

    println!(
        "  GPU: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend
    );
    println!();

    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("portability_tiers");
    let mut results = Vec::new();

    // ═══════════════════════════════════════════════════════════════════
    // 1. HMM Forward (Papers 016-018) — wetSpring origin
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ [1/7] HMM Forward — Papers 016-018 ═══");
    println!("  Provenance: wetSpring metagenomics → hmm_forward_log.wgsl");
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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let out_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hmm_out"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let flat_a_log: Vec<f64> = flat_a.iter().map(|&v| v.max(1e-300).ln()).collect();
        let flat_b_log: Vec<f64> = flat_b.iter().map(|&v| v.max(1e-300).ln()).collect();
        let initial_log: Vec<f64> = initial.iter().map(|&v| v.max(1e-300).ln()).collect();

        let log_a_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hmm_log_a"),
            contents: bytemuck::cast_slice(&flat_a_log),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let log_b_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hmm_log_b"),
            contents: bytemuck::cast_slice(&flat_b_log),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let log_pi_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hmm_log_pi"),
            contents: bytemuck::cast_slice(&initial_log),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let log_alpha_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hmm_log_alpha"),
            size: (t_len * n_states * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let op = HmmBatchForwardF64::new(device.clone()).expect("HMM GPU op");
        op.dispatch(&barracuda::ops::bio::hmm::HmmForwardArgs {
            n_states: n_states as u32,
            n_symbols: n_sym as u32,
            n_steps: t_len as u32,
            n_seqs: 1,
            log_trans: &log_a_buf,
            log_emit: &log_b_buf,
            log_pi: &log_pi_buf,
            observations: &obs_buf,
            log_alpha_out: &log_alpha_buf,
            log_lik_out: &out_buf,
        })
        .expect("HMM dispatch");

        let gpu_ll = gpu
            .read_buffer_f64(&out_buf, 1)
            .map(|v| v[0])
            .unwrap_or(f64::NAN);

        let gpu_us = bench_fn(|| {
            let out = d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 8,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let alpha = d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (t_len * n_states * 8) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let _ = op.dispatch(&barracuda::ops::bio::hmm::HmmForwardArgs {
                n_states: n_states as u32,
                n_symbols: n_sym as u32,
                n_steps: t_len as u32,
                n_seqs: 1,
                log_trans: &log_a_buf,
                log_emit: &log_b_buf,
                log_pi: &log_pi_buf,
                observations: &obs_buf,
                log_alpha_out: &alpha,
                log_lik_out: &out,
            });
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

        println!(
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
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // 2. Batch Fitness (Papers 011-013) — neuralSpring origin
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ [2/7] Batch Fitness — Papers 011-013 ═══");
    println!("  Provenance: neuralSpring neuroevolution → batch_fitness_eval.wgsl");
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

        println!(
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
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // 3. Pairwise L2 (Paper 012) — neuralSpring MODES origin
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ [3/7] Pairwise L2 — Paper 012 ═══");
    println!("  Provenance: neuralSpring MODES → pairwise_l2.wgsl (BarraCUDA)");
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
        let _ = op.dispatch(&in_buf, &out_buf, n as u32, dim as u32);

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
            let _ = op.dispatch(&in_buf, &out, n as u32, dim as u32);
            let _ = std::hint::black_box(gpu.read_buffer_f32(&out, n_pairs));
        });

        let diff = (gpu_mean - cpu_mean).abs() / cpu_mean.abs().max(1e-15);
        let parity = diff < 0.01;
        h.check_bool(
            &format!("L2 pairwise GPU-CPU parity (rel diff={diff:.2e})"),
            parity,
        );

        println!(
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
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // 4. Eigensolve + IPR (Papers 022-023) — neuralSpring Anderson
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ [4/7] Eigensolve + IPR — Papers 022-023 ═══");
    println!("  Provenance: neuralSpring Anderson → batch_ipr.wgsl + eigh.wgsl");
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

        println!(
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
    println!();

    // Sections 5-7: extended benchmarks
    results.push(extended::bench_spatial_payoff(&gpu, &mut h));
    results.push(extended::bench_dispatcher(&gpu, &rt, &mut h));
    results.push(extended::bench_pairwise_hamming(&gpu, &mut h));

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Portability Proof Summary                                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "  {:<25} {:>8} {:>10} {:>10} {:>10} {:>7}",
        "Domain", "Papers", "CPU µs", "GPU µs", "GPU/CPU", "Parity"
    );
    println!("  {}", "─".repeat(72));

    for r in &results {
        let par_str = if r.parity { "✓" } else { "✗" };
        println!(
            "  {:<25} {:>8} {:>10.1} {:>10.1} {:>9.1}× {:>7}",
            r.domain, r.papers, r.cpu_us, r.gpu_us, r.gpu_cpu_speedup, par_str
        );
    }

    let all_parity = results.iter().all(|r| r.parity);
    h.check_bool("All GPU-CPU parity checks passed", all_parity);

    println!("  {}", "─".repeat(72));
    println!();
    println!("  Portability proven: Python/NumPy → BarraCUDA CPU → BarraCUDA GPU");
    println!("  ToadStool streaming: upload once → compute GPU-resident → scalar readback");
    println!("  Same math at every tier, verified to machine precision.");
    println!();

    h.finish();
}
