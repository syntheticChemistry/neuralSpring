// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-accelerated population genetics and game theory operations.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use std::sync::Arc;

use super::reduction::{l2_distance_gpu, mean_gpu, pearson_correlation_gpu, variance_gpu};

/// GPU allele frequencies: column-sum of genotype matrix / (2 × n\_individuals).
///
/// Replaces `meta_population::allele_frequencies`.
/// Uses `Tensor::sum_dim(0)` for parallel column reduction.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn allele_frequencies_gpu(
    pop: &[f64],
    n_individuals: usize,
    n_loci: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    let pop_f32: Vec<f32> = pop.iter().map(|&x| x as f32).collect();
    let mat = Tensor::from_data(&pop_f32, vec![n_individuals, n_loci], device.clone())
        .map_err(|e| format!("allele_freq upload: {e}"))?;
    let col_sums = mat
        .sum_dim(0, false)
        .map_err(|e| format!("allele_freq sum: {e}"))?;
    let sums = col_sums
        .to_vec()
        .map_err(|e| format!("allele_freq read: {e}"))?;

    let denom = 2.0 * n_individuals as f64;
    Ok(sums.iter().map(|&s| f64::from(s) / denom).collect())
}

/// GPU nucleotide diversity: `mean(2 * p * (1-p) * n/(n-1))`.
///
/// Replaces `meta_population::nucleotide_diversity`.
/// Composes allele frequency GPU reduction with elementwise Tensor ops.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn nucleotide_diversity_gpu(
    pop: &[f64],
    n_individuals: usize,
    n_loci: usize,
    device: &Arc<WgpuDevice>,
) -> Result<f64, String> {
    if n_individuals < 2 {
        return Ok(0.0);
    }
    let freqs = allele_frequencies_gpu(pop, n_individuals, n_loci, device)?;
    let correction = (n_individuals as f64 / (n_individuals as f64 - 1.0)) as f32;

    let p_f32: Vec<f32> = freqs.iter().map(|&p| p as f32).collect();
    let p_t = Tensor::from_data(&p_f32, vec![n_loci], device.clone())
        .map_err(|e| format!("nuc_div p: {e}"))?;
    let ones = Tensor::from_data(&vec![1.0_f32; n_loci], vec![n_loci], device.clone())
        .map_err(|e| format!("nuc_div ones: {e}"))?;
    let one_minus_p = ones.sub(&p_t).map_err(|e| format!("nuc_div sub: {e}"))?;
    let het = p_t
        .mul(&one_minus_p)
        .map_err(|e| format!("nuc_div mul: {e}"))?;
    let scaled = het
        .mul_scalar(2.0 * correction)
        .map_err(|e| format!("nuc_div scale: {e}"))?;
    let mean = scaled.mean().map_err(|e| format!("nuc_div mean: {e}"))?;

    let result = mean.to_vec().map_err(|e| format!("nuc_div read: {e}"))?;
    Ok(f64::from(result[0]))
}

/// GPU matrix correlation: Pearson of upper-triangle elements.
///
/// Replaces `meta_population::matrix_correlation`.
/// Extracts upper triangle on CPU, then routes through
/// [`pearson_correlation_gpu`] for the Pearson computation.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn matrix_correlation_gpu(
    a: &[f64],
    b: &[f64],
    n: usize,
    device: &Arc<WgpuDevice>,
) -> Result<f64, String> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            xs.push(a[i * n + j]);
            ys.push(b[i * n + j]);
        }
    }
    if xs.len() < 2 {
        return Ok(0.0);
    }
    pearson_correlation_gpu(&xs, &ys, device)
}

/// GPU geographic distance matrix: pairwise Euclidean from 2D coordinates.
///
/// Replaces `meta_population::geographic_distance_matrix`.
/// Each pairwise L2 distance computed via GPU subtraction + norm.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn geographic_distance_matrix_gpu(
    coords: &[(f64, f64)],
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    let n = coords.len();
    let mut dist = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let a = [coords[i].0, coords[i].1];
            let b = [coords[j].0, coords[j].1];
            let d = l2_distance_gpu(&a, &b, device)?;
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    Ok(dist)
}

/// GPU thermal diversity correlation: Pearson correlation via GPU.
///
/// Replaces `meta_population::thermal_diversity_correlation`.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn thermal_diversity_correlation_gpu(
    pi_values: &[f64],
    temperatures: &[f64],
    device: &Arc<WgpuDevice>,
) -> Result<f64, String> {
    pearson_correlation_gpu(pi_values, temperatures, device)
}

/// GPU inter-population allele frequency variance.
///
/// Replaces `meta_population::inter_population_af_variance`.
/// GPU pipeline: `allele_frequencies` per population → per-locus variance → mean.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn inter_population_af_variance_gpu(
    populations: &[&[f64]],
    n_individuals: &[usize],
    n_loci: usize,
    device: &Arc<WgpuDevice>,
) -> Result<f64, String> {
    let n_pops = populations.len();
    if n_pops == 0 || n_loci == 0 {
        return Ok(0.0);
    }

    let all_freqs: Vec<Vec<f64>> = populations
        .iter()
        .zip(n_individuals.iter())
        .map(|(pop, &n)| allele_frequencies_gpu(pop, n, n_loci, device))
        .collect::<Result<Vec<_>, _>>()?;

    let mut locus_variances = Vec::with_capacity(n_loci);
    for j in 0..n_loci {
        let vals: Vec<f64> = all_freqs.iter().map(|f| f[j]).collect();
        locus_variances.push(variance_gpu(&vals, device)?);
    }

    mean_gpu(&locus_variances, device)
}

/// GPU replicator dynamics step: fitness via GPU matmul, update on CPU.
///
/// Demonstrates 2×2 payoff GEMV on GPU for math portability.
/// `f = P @ x`, then replicator update with normalization.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn replicator_step_gpu(
    freq: &[f64; 2],
    payoff: &[[f64; 2]; 2],
    dt: f64,
    device: &Arc<WgpuDevice>,
) -> Result<[f64; 2], String> {
    let payoff_flat: [f32; 4] = [
        payoff[0][0] as f32,
        payoff[0][1] as f32,
        payoff[1][0] as f32,
        payoff[1][1] as f32,
    ];
    let x_f32 = [freq[0] as f32, freq[1] as f32];

    let p_t = Tensor::from_data(&payoff_flat, vec![2, 2], device.clone())
        .map_err(|e| format!("repl payoff: {e}"))?;
    let x_col = Tensor::from_data(&x_f32, vec![2, 1], device.clone())
        .map_err(|e| format!("repl x: {e}"))?;
    let f_t = p_t
        .matmul(&x_col)
        .map_err(|e| format!("repl matmul: {e}"))?;

    let f_vec = f_t.to_vec().map_err(|e| format!("repl read: {e}"))?;
    let f0 = f64::from(f_vec[0]);
    let f1 = f64::from(f_vec[1]);

    let (x0, x1) = (freq[0], freq[1]);
    let f_bar = x0.mul_add(f0, x1 * f1);

    let mut new_x0 = (dt * x0).mul_add(f0 - f_bar, x0).max(0.0);
    let mut new_x1 = (dt * x1).mul_add(f1 - f_bar, x1).max(0.0);
    let sum = new_x0 + new_x1;
    if sum > 0.0 {
        new_x0 /= sum;
        new_x1 /= sum;
    }

    Ok([new_x0, new_x1])
}
