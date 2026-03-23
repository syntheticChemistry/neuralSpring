// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spectral batch IPR (022–023) and pairwise Hamming / L2 / Jaccard distance ops.

use barracuda::ops::bio::{PairwiseHammingGpu, PairwiseJaccardGpu, PairwiseL2Gpu};
use barracuda::spectral::BatchIprGpu;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, output_buf, storage_buf};
use std::sync::Arc;

pub fn validate_batch_ipr(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_vectors = 8_usize;
    let dim = 16_usize;
    let mut rng = Rng::new(55);
    let mut vecs_f64: Vec<f64> = (0..n_vectors * dim).map(|_| rng.normal()).collect();

    for i in 0..n_vectors {
        let base = i * dim;
        let norm: f64 = (0..dim)
            .map(|d| vecs_f64[base + d] * vecs_f64[base + d])
            .sum::<f64>()
            .sqrt();
        for d in 0..dim {
            vecs_f64[base + d] /= norm;
        }
    }

    let vecs_f32: Vec<f32> = vecs_f64.iter().map(|&v| v as f32).collect();

    let cpu_iprs: Vec<f64> = (0..n_vectors)
        .map(|i| {
            let base = i * dim;
            (0..dim)
                .map(|d| {
                    let a = vecs_f64[base + d];
                    a * a * a * a
                })
                .sum::<f64>()
        })
        .collect();
    let cpu_mean = cpu_iprs.iter().sum::<f64>() / cpu_iprs.len() as f64;

    let op = BatchIprGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let vecs_buf = storage_buf(device, "ipr_v", bytemuck::cast_slice(&vecs_f32));
    let out_buf = output_buf(device, "ipr_out", (n_vectors * 4) as u64);

    op.dispatch(&vecs_buf, &out_buf, dim as u32, n_vectors as u32);

    match gpu.read_buffer_f32(&out_buf, n_vectors) {
        Ok(gpu_ipr) => {
            let gpu_mean: f64 =
                gpu_ipr.iter().map(|&v| f64::from(v)).sum::<f64>() / gpu_ipr.len() as f64;
            h.check_abs(
                &format!("IPR 8×16: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_BATCH_IPR_F32,
            );
        }
        Err(e) => h.check_bool(&format!("IPR: {e}"), false),
    }
}

fn check_gpu_f32_mean(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    label: &str,
    out_buf: &wgpu::Buffer,
    n: usize,
    cpu_mean: f64,
    tol: f64,
) {
    match gpu.read_buffer_f32(out_buf, n) {
        Ok(gpu_d) => {
            let gpu_mean: f64 =
                gpu_d.iter().map(|&v| f64::from(v)).sum::<f64>() / gpu_d.len() as f64;
            h.check_abs(
                &format!("{label}: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tol,
            );
        }
        Err(e) => h.check_bool(&format!("{label}: {e}"), false),
    }
}

pub fn validate_hamming(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 6_usize;
    let seq_len = 20_usize;
    let mut rng = Rng::new(44);
    let seqs: Vec<u32> = (0..n_seqs * seq_len).map(|_| rng.usize(4) as u32).collect();

    let n_pairs = n_seqs * (n_seqs - 1) / 2;
    let cpu_mean = {
        let mut total = 0.0_f32;
        let mut count = 0_usize;
        for i in 0..n_seqs {
            for j in (i + 1)..n_seqs {
                let mut diff = 0_u32;
                for k in 0..seq_len {
                    if seqs[i * seq_len + k] != seqs[j * seq_len + k] {
                        diff += 1;
                    }
                }
                total += diff as f32 / seq_len as f32;
                count += 1;
            }
        }
        total / count as f32
    };

    let op = PairwiseHammingGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let seqs_buf = storage_buf(device, "ham_s", bytemuck::cast_slice(&seqs));
    let out_buf = output_buf(device, "ham_out", (n_pairs * 4) as u64);
    op.dispatch(&seqs_buf, &out_buf, n_seqs as u32, seq_len as u32);
    check_gpu_f32_mean(
        h,
        gpu,
        "Hamming 6×20",
        &out_buf,
        n_pairs,
        f64::from(cpu_mean),
        tolerances::GPU_HAMMING_F32,
    );
}

pub fn validate_l2(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 8_usize;
    let dim = 6_usize;
    let mut rng = Rng::new(66);
    let points_f64: Vec<f64> = (0..n * dim).map(|_| rng.normal()).collect();
    let points_f32: Vec<f32> = points_f64.iter().map(|&v| v as f32).collect();

    let n_pairs = n * (n - 1) / 2;
    let mut cpu_dist = Vec::with_capacity(n_pairs);
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = (0..dim)
                .map(|k| {
                    let diff = points_f64[i * dim + k] - points_f64[j * dim + k];
                    diff * diff
                })
                .sum::<f64>()
                .sqrt();
            cpu_dist.push(d);
        }
    }
    let cpu_mean = cpu_dist.iter().sum::<f64>() / cpu_dist.len() as f64;

    let op = PairwiseL2Gpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let pts_buf = storage_buf(device, "l2_pts", bytemuck::cast_slice(&points_f32));
    let out_buf = output_buf(device, "l2_out", (n_pairs * 4) as u64);

    if let Err(e) = op.dispatch(&pts_buf, &out_buf, n as u32, dim as u32) {
        h.check_bool(&format!("PairwiseL2 dispatch: {e}"), false);
        return;
    }
    check_gpu_f32_mean(
        h,
        gpu,
        "L2 8×6",
        &out_buf,
        n_pairs,
        cpu_mean,
        tolerances::GPU_MODES_L2_F32,
    );
}

pub fn validate_jaccard(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_genomes = 8_usize;
    let n_genes = 32_usize;
    let mut rng = Rng::new(88);
    let pa_f64: Vec<f64> = (0..n_genomes * n_genes)
        .map(|_| if rng.uniform() > 0.3 { 1.0 } else { 0.0 })
        .collect();

    let cpu_jd =
        neural_spring::pangenome_selection::jaccard_distance_matrix(&pa_f64, n_genes, n_genomes);
    let mut cpu_upper = Vec::new();
    for i in 0..n_genomes {
        for j in (i + 1)..n_genomes {
            cpu_upper.push(cpu_jd[i * n_genomes + j]);
        }
    }
    let cpu_mean = cpu_upper.iter().sum::<f64>() / cpu_upper.len() as f64;

    let pa_f32: Vec<f32> = pa_f64.iter().map(|&v| v as f32).collect();
    let op = PairwiseJaccardGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let n_pairs = n_genomes * (n_genomes - 1) / 2;
    let pa_buf = storage_buf(device, "jac_pa", bytemuck::cast_slice(&pa_f32));
    let out_buf = output_buf(device, "jac_out", (n_pairs * 4) as u64);
    op.dispatch(&pa_buf, &out_buf, n_genomes as u32, n_genes as u32);
    check_gpu_f32_mean(
        h,
        gpu,
        "Jaccard 8×32",
        &out_buf,
        n_pairs,
        cpu_mean,
        tolerances::GPU_JACCARD_F32,
    );
}
