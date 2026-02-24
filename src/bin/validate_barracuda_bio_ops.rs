// SPDX-License-Identifier: AGPL-3.0-or-later

//! Upstream `BarraCUDA` bio-op wrapper validation.
//!
//! Validates the **barracuda Rust wrapper APIs** (not local metalForge shaders)
//! for 6 bio ops that `ToadStool` absorbed from `neuralSpring`. Each wrapper
//! encapsulates shader dispatch — this proves the upstream API produces
//! correct results against CPU references.
//!
//! ## Cross-Spring Evolution
//!
//! These shaders originated in neuralSpring's metalForge, were absorbed by
//! ToadStool/BarraCUDA (Sessions 25–39), and now have first-class Rust
//! wrapper APIs. This validator closes the loop: neuralSpring validates
//! the upstream wrappers that grew from its own shaders.
//!
//! | Op | Origin | Upstream Wrapper | Absorbed |
//! |----|--------|------------------|----------|
//! | Batch Fitness | neuralSpring Paper 011–015 | `BatchFitnessGpu` | `77f70b2e` |
//! | Pairwise Hamming | neuralSpring Paper 017 | `PairwiseHammingGpu` | `77f70b2e` |
//! | Pairwise Jaccard | neuralSpring Paper 024 | `PairwiseJaccardGpu` | `77f70b2e` |
//! | Locus Variance | neuralSpring Paper 025 | `LocusVarianceGpu` | `77f70b2e` |
//! | Spatial Payoff | neuralSpring Paper 019 | `SpatialPayoffGpu` | `77f70b2e` |
//! | Batch IPR | neuralSpring Paper 022–023 | `BatchIprGpu` | `77f70b2e` |

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use barracuda::ops::bio::{
    BatchFitnessGpu, LocusVarianceGpu, PairwiseHammingGpu, PairwiseJaccardGpu, SpatialPayoffGpu,
};
use barracuda::spectral::BatchIprGpu;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let mut h = ValidationHarness::new("barracuda_bio_ops");

    validate_batch_fitness(&mut h, &gpu);
    validate_pairwise_hamming(&mut h, &gpu);
    validate_pairwise_jaccard(&mut h, &gpu);
    validate_locus_variance(&mut h, &gpu);
    validate_spatial_payoff(&mut h, &gpu);
    validate_batch_ipr(&mut h, &gpu);

    h.finish();
}

// ─── Batch Fitness (Papers 011–015) ──────────────────────────────────

fn cpu_batch_fitness(pop: &[f64], weights: &[f64], pop_size: usize, genome_len: usize) -> Vec<f64> {
    (0..pop_size)
        .map(|i| {
            let base = i * genome_len;
            (0..genome_len).map(|g| pop[base + g] * weights[g]).sum()
        })
        .collect()
}

fn validate_batch_fitness(h: &mut ValidationHarness, gpu: &Gpu) {
    let dev = Arc::clone(gpu.wgpu_device());
    let device = gpu.device();

    let op = BatchFitnessGpu::new(dev);

    let pop_size = 64_u32;
    let genome_len = 16_u32;
    let mut rng = Rng::new(42);

    let population: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..genome_len).map(|_| rng.uniform()).collect();

    let cpu = cpu_batch_fitness(
        &population,
        &weights,
        pop_size as usize,
        genome_len as usize,
    );

    let pop_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pop"),
        contents: bytemuck::cast_slice(&population),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let weight_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weights"),
        contents: bytemuck::cast_slice(&weights),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fitness"),
        size: u64::from(pop_size) * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&pop_buf, &weight_buf, &fitness_buf, pop_size, genome_len);

    match gpu.read_buffer_f64(&fitness_buf, pop_size as usize) {
        Ok(gpu_result) => {
            let max_diff: f64 = gpu_result
                .iter()
                .zip(cpu.iter())
                .map(|(&g, &c)| (g - c).abs())
                .fold(0.0_f64, f64::max);
            h.check_upper(
                &format!("BatchFitnessGpu: max diff {max_diff:.2e} ({pop_size}×{genome_len})"),
                max_diff,
                tolerances::GPU_FITNESS_F32,
            );
            h.check_bool(
                &format!("BatchFitnessGpu: correct count ({})", gpu_result.len()),
                gpu_result.len() == pop_size as usize,
            );
        }
        Err(e) => h.check_bool(&format!("BatchFitnessGpu: readback failed — {e}"), false),
    }
}

// ─── Pairwise Hamming (Paper 017 — SATé alignment) ──────────────────

fn cpu_pairwise_hamming(seqs: &[u32], n_seqs: usize, seq_len: usize) -> Vec<f32> {
    let n_pairs = n_seqs * (n_seqs - 1) / 2;
    let mut out = Vec::with_capacity(n_pairs);
    for i in 0..n_seqs {
        for j in (i + 1)..n_seqs {
            let mismatches: u32 = (0..seq_len)
                .map(|k| u32::from(seqs[i * seq_len + k] != seqs[j * seq_len + k]))
                .sum();
            out.push(mismatches as f32 / seq_len as f32);
        }
    }
    out
}

fn validate_pairwise_hamming(h: &mut ValidationHarness, gpu: &Gpu) {
    let dev = Arc::clone(gpu.wgpu_device());
    let device = gpu.device();

    let op = PairwiseHammingGpu::new(dev);

    let n_seqs = 20_u32;
    let seq_len = 100_u32;
    let mut rng = Rng::new(17);

    let sequences: Vec<u32> = (0..n_seqs * seq_len)
        .map(|_| (rng.uniform() * 4.0) as u32)
        .collect();

    let cpu = cpu_pairwise_hamming(&sequences, n_seqs as usize, seq_len as usize);
    let n_pairs = (n_seqs * (n_seqs - 1) / 2) as usize;

    let seq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("seqs"),
        contents: bytemuck::cast_slice(&sequences),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("distances"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&seq_buf, &dist_buf, n_seqs, seq_len);

    match gpu.read_buffer_f32(&dist_buf, n_pairs) {
        Ok(gpu_result) => {
            let max_diff: f64 = gpu_result
                .iter()
                .zip(cpu.iter())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
                .fold(0.0_f64, f64::max);
            h.check_upper(
                &format!("PairwiseHammingGpu: max diff {max_diff:.2e} ({n_seqs} seqs)"),
                max_diff,
                tolerances::GPU_HAMMING_F32,
            );
            h.check_bool(
                &format!("PairwiseHammingGpu: correct pair count ({n_pairs})"),
                gpu_result.len() == n_pairs,
            );
        }
        Err(e) => h.check_bool(&format!("PairwiseHammingGpu: readback failed — {e}"), false),
    }
}

// ─── Pairwise Jaccard (Paper 024 — Pangenome) ───────────────────────

/// CPU Jaccard distance over **column-major** PA matrix: `pa[gene * n_genomes + genome]`.
fn cpu_pairwise_jaccard(pa: &[f32], n_genomes: usize, n_genes: usize) -> Vec<f32> {
    let n_pairs = n_genomes * (n_genomes - 1) / 2;
    let mut out = Vec::with_capacity(n_pairs);
    for i in 0..n_genomes {
        for j in (i + 1)..n_genomes {
            let mut intersection = 0.0_f32;
            let mut union = 0.0_f32;
            for g in 0..n_genes {
                let a = pa[g * n_genomes + i];
                let b = pa[g * n_genomes + j];
                if a > 0.5 || b > 0.5 {
                    union += 1.0;
                    if a > 0.5 && b > 0.5 {
                        intersection += 1.0;
                    }
                }
            }
            let jaccard = if union > 0.0 {
                intersection / union
            } else {
                1.0
            };
            out.push(1.0 - jaccard);
        }
    }
    out
}

fn validate_pairwise_jaccard(h: &mut ValidationHarness, gpu: &Gpu) {
    let dev = Arc::clone(gpu.wgpu_device());
    let device = gpu.device();

    let op = PairwiseJaccardGpu::new(dev);

    let n_genomes = 15_u32;
    let n_genes = 80_u32;
    let mut rng = Rng::new(24);

    // Column-major: pa[gene * n_genomes + genome]
    let pa: Vec<f32> = (0..n_genes * n_genomes)
        .map(|_| if rng.uniform() > 0.3 { 1.0 } else { 0.0 })
        .collect();

    let cpu = cpu_pairwise_jaccard(&pa, n_genomes as usize, n_genes as usize);
    let n_pairs = (n_genomes * (n_genomes - 1) / 2) as usize;

    let pa_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pa"),
        contents: bytemuck::cast_slice(&pa),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("distances"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&pa_buf, &dist_buf, n_genomes, n_genes);

    match gpu.read_buffer_f32(&dist_buf, n_pairs) {
        Ok(gpu_result) => {
            let max_diff: f64 = gpu_result
                .iter()
                .zip(cpu.iter())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
                .fold(0.0_f64, f64::max);
            h.check_upper(
                &format!("PairwiseJaccardGpu: max diff {max_diff:.2e} ({n_genomes} genomes)"),
                max_diff,
                tolerances::GPU_JACCARD_F32,
            );
            h.check_bool(
                &format!("PairwiseJaccardGpu: correct pair count ({n_pairs})"),
                gpu_result.len() == n_pairs,
            );
        }
        Err(e) => h.check_bool(&format!("PairwiseJaccardGpu: readback failed — {e}"), false),
    }
}

// ─── Locus Variance (Paper 025 — Meta-population) ───────────────────

fn cpu_locus_variance(freqs: &[f64], n_pops: usize, n_loci: usize) -> Vec<f64> {
    (0..n_loci)
        .map(|l| {
            let mean: f64 = (0..n_pops).map(|p| freqs[p * n_loci + l]).sum::<f64>() / n_pops as f64;
            let var: f64 = (0..n_pops)
                .map(|p| {
                    let d = freqs[p * n_loci + l] - mean;
                    d * d
                })
                .sum::<f64>()
                / n_pops as f64;
            var
        })
        .collect()
}

fn validate_locus_variance(h: &mut ValidationHarness, gpu: &Gpu) {
    let dev = Arc::clone(gpu.wgpu_device());
    let device = gpu.device();

    let op = LocusVarianceGpu::new(dev);

    let n_pops = 8_u32;
    let n_loci = 50_u32;
    let mut rng = Rng::new(25);

    let freqs: Vec<f64> = (0..n_pops * n_loci).map(|_| rng.uniform()).collect();

    let cpu = cpu_locus_variance(&freqs, n_pops as usize, n_loci as usize);

    let freq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("freqs"),
        contents: bytemuck::cast_slice(&freqs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let var_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("variance"),
        size: u64::from(n_loci) * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&freq_buf, &var_buf, n_pops, n_loci);

    match gpu.read_buffer_f64(&var_buf, n_loci as usize) {
        Ok(gpu_result) => {
            let max_diff: f64 = gpu_result
                .iter()
                .zip(cpu.iter())
                .map(|(&g, &c)| (g - c).abs())
                .fold(0.0_f64, f64::max);
            h.check_upper(
                &format!("LocusVarianceGpu: max diff {max_diff:.2e} ({n_pops}×{n_loci})"),
                max_diff,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );
            h.check_bool(
                &format!(
                    "LocusVarianceGpu: correct locus count ({})",
                    gpu_result.len()
                ),
                gpu_result.len() == n_loci as usize,
            );
        }
        Err(e) => h.check_bool(&format!("LocusVarianceGpu: readback failed — {e}"), false),
    }
}

// ─── Spatial Payoff (Paper 019 — Game Theory) ────────────────────────

fn cpu_spatial_fitness(grid: &[u32], grid_size: usize, b: f32, c: f32) -> Vec<f32> {
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

fn validate_spatial_payoff(h: &mut ValidationHarness, gpu: &Gpu) {
    let dev = Arc::clone(gpu.wgpu_device());
    let device = gpu.device();

    let op = SpatialPayoffGpu::new(dev);

    let grid_size = 20_u32;
    let b = 3.0_f32;
    let c = 1.0_f32;
    let mut rng = Rng::new(19);

    let grid: Vec<u32> = (0..grid_size * grid_size)
        .map(|_| u32::from(rng.uniform() >= 0.5))
        .collect();

    let cpu = cpu_spatial_fitness(&grid, grid_size as usize, b, c);
    let n_cells = (grid_size * grid_size) as usize;

    let grid_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grid"),
        contents: bytemuck::cast_slice(&grid),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fitness"),
        size: (n_cells * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&grid_buf, &fitness_buf, grid_size, b, c);

    match gpu.read_buffer_f32(&fitness_buf, n_cells) {
        Ok(gpu_result) => {
            let max_diff: f64 = gpu_result
                .iter()
                .zip(cpu.iter())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
                .fold(0.0_f64, f64::max);
            h.check_upper(
                &format!("SpatialPayoffGpu: max diff {max_diff:.2e} ({grid_size}×{grid_size})"),
                max_diff,
                tolerances::GPU_SPATIAL_PAYOFF_F32,
            );
            h.check_bool(
                &format!("SpatialPayoffGpu: correct cell count ({n_cells})"),
                gpu_result.len() == n_cells,
            );
        }
        Err(e) => h.check_bool(&format!("SpatialPayoffGpu: readback failed — {e}"), false),
    }
}

// ─── Batch IPR (Papers 022–023 — Anderson localization) ──────────────

/// Upstream `BarraCUDA` definition: `IPR` = `Σ|ψ_i|⁴` (raw, not reciprocal).
fn cpu_batch_ipr(eigenvectors: &[f32], dim: usize, n_vectors: usize) -> Vec<f32> {
    (0..n_vectors)
        .map(|v| {
            let base = v * dim;
            (0..dim)
                .map(|i| {
                    let x = eigenvectors[base + i];
                    x * x * x * x
                })
                .sum()
        })
        .collect()
}

fn validate_batch_ipr(h: &mut ValidationHarness, gpu: &Gpu) {
    let dev = Arc::clone(gpu.wgpu_device());
    let device = gpu.device();

    let op = BatchIprGpu::new(dev);

    let dim = 64_u32;
    let n_vectors = 32_u32;
    let mut rng = Rng::new(23);

    let mut eigenvectors: Vec<f32> = Vec::with_capacity((dim * n_vectors) as usize);
    for _ in 0..n_vectors {
        let raw: Vec<f32> = (0..dim).map(|_| rng.uniform() as f32).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        eigenvectors.extend(raw.iter().map(|x| x / norm));
    }

    let cpu = cpu_batch_ipr(&eigenvectors, dim as usize, n_vectors as usize);

    let ev_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("eigenvectors"),
        contents: bytemuck::cast_slice(&eigenvectors),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let ipr_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ipr"),
        size: u64::from(n_vectors) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(&ev_buf, &ipr_buf, dim, n_vectors);

    match gpu.read_buffer_f32(&ipr_buf, n_vectors as usize) {
        Ok(gpu_result) => {
            let max_diff: f64 = gpu_result
                .iter()
                .zip(cpu.iter())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
                .fold(0.0_f64, f64::max);
            h.check_upper(
                &format!("BatchIprGpu: max diff {max_diff:.2e} ({n_vectors}×{dim})"),
                max_diff,
                tolerances::GPU_BATCH_IPR_F32,
            );
            h.check_bool(
                &format!("BatchIprGpu: correct vector count ({})", gpu_result.len()),
                gpu_result.len() == n_vectors as usize,
            );
        }
        Err(e) => h.check_bool(&format!("BatchIprGpu: readback failed — {e}"), false),
    }
}
