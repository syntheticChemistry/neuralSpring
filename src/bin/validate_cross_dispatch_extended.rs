// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-dispatch validation: extended workloads on GPU vs CPU.
//!
//! Uses `BarraCUDA` typed op APIs to validate GPU ↔ CPU parity for
//! `SpatialPayoffGpu` (Paper 019), `BatchIprGpu` (Papers 022-023),
//! and `PairwiseHammingGpu` (Paper 017).
//!
//! ## Evolution path
//!
//! ```text
//! GPU-only (validate_gpu_game_theory, validate_gpu_anderson, validate_gpu_sate)
//!   → Cross-dispatch GPU ↔ CPU (this binary)
//!   → metalForge cross-system (GPU → NPU → CPU)
//! ```
//!
//! ## Provenance
//!
//! CPU/GPU dispatch: extended domain via `barracuda::ops::bio` and
//! `barracuda::spectral`. Validates: stencil, `batch_reduce`, hamming GPU↔CPU parity.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    reason = "validation binary"
)]

use std::sync::Arc;

use barracuda::dispatch::{DispatchTarget, dispatch_for};
use barracuda::ops::bio::{PairwiseHammingGpu, SpatialPayoffGpu};
use barracuda::spectral::BatchIprGpu;
use neural_spring::anderson_localization::{
    GOLDEN_RATIO, aubry_andre_hamiltonian, ipr, jacobi_eigh,
};
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            println!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let mut h = ValidationHarness::new("cross_dispatch_extended");

    validate_dispatch_routing(&mut h);
    validate_payoff_parity(&mut h, &gpu);
    validate_ipr_parity(&mut h, &gpu);
    validate_hamming_parity(&mut h, &gpu);

    h.finish();
}

// ── Dispatch routing ─────────────────────────────────────────────

fn validate_dispatch_routing(h: &mut ValidationHarness) {
    let small_stencil = dispatch_for("stencil", 100);
    let large_stencil = dispatch_for("stencil", 10_000);
    let small_batch = dispatch_for("batch_reduce", 16);
    let large_batch = dispatch_for("batch_reduce", 10_000);
    let small_hamming = dispatch_for("hamming", 20);
    let large_hamming = dispatch_for("hamming", 10_000);

    h.check_bool(
        &format!("dispatch: stencil(100) → {small_stencil:?}"),
        matches!(small_stencil, DispatchTarget::Cpu),
    );
    h.check_bool(
        &format!("dispatch: stencil(10k) → {large_stencil:?}"),
        matches!(large_stencil, DispatchTarget::Gpu),
    );
    h.check_bool(
        &format!("dispatch: batch_reduce(16) → {small_batch:?}"),
        matches!(small_batch, DispatchTarget::Cpu),
    );
    h.check_bool(
        &format!("dispatch: batch_reduce(10k) → {large_batch:?}"),
        matches!(large_batch, DispatchTarget::Gpu),
    );
    h.check_bool(
        &format!("dispatch: hamming(20) → {small_hamming:?}"),
        matches!(small_hamming, DispatchTarget::Cpu),
    );
    h.check_bool(
        &format!("dispatch: hamming(10k) → {large_hamming:?}"),
        matches!(large_hamming, DispatchTarget::Gpu),
    );
}

// ── Spatial payoff parity (Paper 019) ────────────────────────────

fn cpu_payoff_fitness(grid: &[u32], grid_size: usize, b: f32, c: f32) -> Vec<f32> {
    let n = grid_size as i32;
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

    let mut fitness = Vec::with_capacity(grid_size * grid_size);
    for i in 0..grid_size {
        for j in 0..grid_size {
            let me = grid[i * grid_size + j];
            let mut total = 0.0_f32;
            for (di, dj) in &neighbors {
                let ni = ((i as i32 + di).rem_euclid(n)) as usize;
                let nj = ((j as i32 + dj).rem_euclid(n)) as usize;
                let other = grid[ni * grid_size + nj];
                total += match (me, other) {
                    (1, 1) => b - c,
                    (1, 0) => -c,
                    (0, 1) => b,
                    _ => 0.0,
                };
            }
            fitness.push(total);
        }
    }
    fitness
}

fn gpu_payoff(gpu: &Gpu, grid: &[u32], grid_size: u32, b: f32, c: f32) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let op = SpatialPayoffGpu::new(Arc::clone(gpu.wgpu_device()));

    let n_cells = (grid_size * grid_size) as usize;
    let grid_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_grid"),
        contents: bytemuck::cast_slice(grid),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xd_payoff_out"),
        size: (n_cells * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&grid_buf, &fitness_buf, grid_size, b, c);

    gpu.read_buffer_f32(&fitness_buf, n_cells)
}

fn validate_payoff_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let mut rng = Rng::new(42);
    let grid_size = 15_usize;
    let grid: Vec<u32> = (0..grid_size * grid_size)
        .map(|_| u32::from(rng.uniform() >= 0.5))
        .collect();

    let b = 3.0_f32;
    let c = 1.0_f32;

    let cpu_fitness = cpu_payoff_fitness(&grid, grid_size, b, c);

    match gpu_payoff(gpu, &grid, grid_size as u32, b, c) {
        Ok(gpu_fitness) => {
            let max_diff: f64 = gpu_fitness
                .iter()
                .zip(cpu_fitness.iter())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "payoff parity (15×15): max diff {max_diff:.2e}, {} cells",
                    gpu_fitness.len()
                ),
                max_diff,
                tolerances::GPU_SPATIAL_PAYOFF_F32,
            );

            let mean_fitness: f64 =
                gpu_fitness.iter().map(|&v| f64::from(v)).sum::<f64>() / gpu_fitness.len() as f64;
            h.check_lower(
                &format!("payoff: mean fitness > 0 ({mean_fitness:.4})"),
                mean_fitness,
                0.0,
            );
        }
        Err(e) => {
            h.check_bool(&format!("payoff parity: failed — {e}"), false);
        }
    }
}

// ── Batch IPR parity (Papers 022-023) ─────────────────────────────

fn gpu_batch_ipr(
    gpu: &Gpu,
    eigenvectors: &[f32],
    dim: u32,
    n_vectors: u32,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let op = BatchIprGpu::new(Arc::clone(gpu.wgpu_device()));

    let n_vectors_usize = n_vectors as usize;
    let ev_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_eigenvectors"),
        contents: bytemuck::cast_slice(eigenvectors),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let ipr_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xd_ipr_out"),
        size: (n_vectors_usize * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&ev_buf, &ipr_buf, dim, n_vectors);

    gpu.read_buffer_f32(&ipr_buf, n_vectors_usize)
}

fn validate_ipr_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 16_usize;
    let t = 1.0_f64;
    let w = 3.0_f64;
    let alpha = 1.0 / GOLDEN_RATIO;
    let phi = 0.0_f64;

    let h_mat = aubry_andre_hamiltonian(n, t, w, alpha, phi);
    let (_eigvals, ev) = jacobi_eigh(&h_mat, n);

    let mut flat: Vec<f32> = Vec::with_capacity(n * n);
    for k in 0..n {
        for i in 0..n {
            flat.push(ev[i * n + k] as f32);
        }
    }

    let cpu_ipr: Vec<f64> = (0..n)
        .map(|k| {
            let col: Vec<f64> = (0..n).map(|i| ev[i * n + k]).collect();
            ipr(&col)
        })
        .collect();

    match gpu_batch_ipr(gpu, &flat, n as u32, n as u32) {
        Ok(gpu_ipr) => {
            let max_diff: f64 = gpu_ipr
                .iter()
                .zip(cpu_ipr.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "IPR parity (n=16, w=3): max diff {max_diff:.2e}, {} vectors",
                    gpu_ipr.len()
                ),
                max_diff,
                tolerances::GPU_BATCH_IPR_F32,
            );

            let mean_ipr: f64 =
                gpu_ipr.iter().map(|&v| f64::from(v)).sum::<f64>() / gpu_ipr.len() as f64;
            h.check_lower(&format!("IPR: mean IPR > 0 ({mean_ipr:.6})"), mean_ipr, 0.0);
        }
        Err(e) => {
            h.check_bool(&format!("IPR parity: failed — {e}"), false);
        }
    }
}

// ── Pairwise Hamming parity (Paper 017) ───────────────────────────

fn cpu_hamming_upper(seqs: &[Vec<u8>]) -> Vec<f64> {
    let n = seqs.len();
    let mut upper = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = seqs[i]
                .iter()
                .zip(seqs[j].iter())
                .filter(|(a, b)| a != b)
                .count();
            upper.push(diff as f64 / seqs[i].len() as f64);
        }
    }
    upper
}

fn gpu_pairwise_hamming(
    gpu: &Gpu,
    sequences: &[u32],
    n_seqs: u32,
    seq_len: u32,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let op = PairwiseHammingGpu::new(Arc::clone(gpu.wgpu_device()));

    let n_pairs = (n_seqs * (n_seqs - 1) / 2) as usize;
    let seq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_sequences"),
        contents: bytemuck::cast_slice(sequences),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xd_distances"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&seq_buf, &dist_buf, n_seqs, seq_len);

    gpu.read_buffer_f32(&dist_buf, n_pairs)
}

fn validate_hamming_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let mut rng = Rng::new(42);
    let n_seqs = 12_usize;
    let seq_len = 80_usize;

    let flat: Vec<u32> = (0..n_seqs * seq_len).map(|_| rng.usize(4) as u32).collect();

    let seqs: Vec<Vec<u8>> = flat
        .chunks(seq_len)
        .map(|chunk| chunk.iter().map(|&v| v as u8).collect())
        .collect();

    let cpu_upper = cpu_hamming_upper(&seqs);

    match gpu_pairwise_hamming(gpu, &flat, n_seqs as u32, seq_len as u32) {
        Ok(gpu_dist) => {
            let max_diff: f64 = gpu_dist
                .iter()
                .zip(cpu_upper.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "Hamming parity (12×80): max diff {max_diff:.2e}, {} pairs",
                    gpu_dist.len()
                ),
                max_diff,
                tolerances::GPU_HAMMING_F32,
            );

            let mean_dist: f64 =
                gpu_dist.iter().map(|&v| f64::from(v)).sum::<f64>() / gpu_dist.len() as f64;
            h.check_lower(
                &format!("Hamming: mean distance > 0 ({mean_dist:.6})"),
                mean_dist,
                0.0,
            );
        }
        Err(e) => {
            h.check_bool(&format!("Hamming parity: failed — {e}"), false);
        }
    }
}
