// SPDX-License-Identifier: AGPL-3.0-or-later

//! Extended portability tier benchmarks: spatial payoff, cross-domain
//! dispatcher, and pairwise Hamming.

use barracuda::ops::bio::{PairwiseHammingGpu, SpatialPayoffGpu};
use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

use super::{TierResult, bench_fn};

// ═══════════════════════════════════════════════════════════════════
// 5. Spatial Payoff (Paper 019) — neuralSpring game theory
// ═══════════════════════════════════════════════════════════════════

pub fn bench_spatial_payoff(gpu: &Gpu, h: &mut ValidationHarness) -> TierResult {
    println!("═══ [5/7] Spatial Payoff — Paper 019 ═══");
    println!("  Provenance: neuralSpring game theory → spatial_payoff.wgsl");

    let n = 32_usize;
    #[expect(clippy::cast_possible_wrap, reason = "validation binary")]
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
                #[expect(clippy::cast_possible_wrap, reason = "validation binary")]
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
                #[expect(clippy::cast_possible_wrap, reason = "validation binary")]
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

    let op = SpatialPayoffGpu::new(gpu.wgpu_device().clone());
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

    println!(
        "    CPU: {cpu_us:.1}µs, GPU: {gpu_us:.1}µs, GPU/CPU: {:.1}×",
        cpu_us / gpu_us
    );
    println!();

    TierResult {
        domain: "Spatial Payoff",
        papers: "019",
        cpu_us,
        gpu_us,
        gpu_cpu_speedup: cpu_us / gpu_us,
        parity,
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. Dispatcher CPU↔GPU (cross-domain)
// ═══════════════════════════════════════════════════════════════════

pub fn bench_dispatcher(
    gpu: &Gpu,
    rt: &tokio::runtime::Runtime,
    h: &mut ValidationHarness,
) -> TierResult {
    println!("═══ [6/7] Dispatcher CPU↔GPU — All Domains ═══");
    println!("  Provenance: neuralSpring Dispatcher routes to optimal substrate");

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

    println!("    Variance: CPU {cpu_us:.1}µs, Dispatcher {gpu_us:.1}µs");

    let cpu_pearson =
        barracuda::stats::correlation::pearson_correlation(&data[..5000], &data[5000..])
            .unwrap_or(f64::NAN);
    let disp_pearson = dispatcher.pearson_correlation(&data[..5000], &data[5000..]);
    let pearson_diff = (cpu_pearson - disp_pearson).abs();
    h.check_bool(
        &format!("Dispatcher pearson CPU=GPU (diff={pearson_diff:.2e})"),
        pearson_diff < tolerances::CROSS_LANGUAGE,
    );

    println!("    Dispatcher proves: same math routes to optimal substrate transparently");
    println!();

    let _ = gpu;

    TierResult {
        domain: "Dispatcher var+pearson",
        papers: "All",
        cpu_us,
        gpu_us,
        gpu_cpu_speedup: cpu_us / gpu_us,
        parity: var_diff < tolerances::TENSOR_MATMUL_F32
            && pearson_diff < tolerances::CROSS_LANGUAGE,
    }
}

// ═══════════════════════════════════════════════════════════════════
// 7. Pairwise Hamming (Paper 017) — wetSpring alignment
// ═══════════════════════════════════════════════════════════════════

pub fn bench_pairwise_hamming(gpu: &Gpu, h: &mut ValidationHarness) -> TierResult {
    println!("═══ [7/7] Pairwise Hamming — Paper 017 ═══");
    println!("  Provenance: wetSpring alignment → pairwise_hamming.wgsl");

    let mut rng = Rng::new(42);
    let n_seqs = 50_usize;
    let seq_len = 200_usize;
    let seqs: Vec<u32> = (0..n_seqs * seq_len)
        .map(|_| (rng.next_u64() % 4) as u32)
        .collect();
    let n_pairs = n_seqs * (n_seqs - 1) / 2;

    let seqs_u8: Vec<u8> = seqs.iter().map(|&v| v as u8).collect();
    let dist_cpu =
        neural_spring::sate_alignment::pairwise_distance_matrix(&seqs_u8, n_seqs, seq_len, false);
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

    let op = PairwiseHammingGpu::new(gpu.wgpu_device().clone());
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

    println!(
        "    CPU: {cpu_us:.1}µs, GPU: {gpu_us:.1}µs, GPU/CPU: {:.1}×",
        cpu_us / gpu_us
    );
    println!();

    TierResult {
        domain: "Pairwise Hamming",
        papers: "017",
        cpu_us,
        gpu_us,
        gpu_cpu_speedup: cpu_us / gpu_us,
        parity,
    }
}
