// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-accelerated science operations for pure GPU execution.
//!
//! Each function provides a GPU path for operations that lib modules
//! implement on CPU. The CPU implementations remain as validation
//! references; these GPU variants are the production execution path.
//!
//! ## Design
//!
//! - All functions take an `Arc<WgpuDevice>` — no global state
//! - f32 GPU execution (matches Tensor API), f64 CPU references
//! - Errors propagated via `Result`, never panics in production
//! - Capability-based: callers check `GpuCapabilities` before dispatch
//!
//! ## Naming
//!
//! Each function mirrors its CPU counterpart with a `_gpu` suffix:
//! `mat_mul` → `mat_mul_gpu`, `frobenius_norm` → `frobenius_norm_gpu`, etc.

#![allow(
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════
// Linear algebra (spectral_commutativity GPU promotion)
// ═══════════════════════════════════════════════════════════════════

/// GPU matrix multiplication C = A × B for n×n matrices.
///
/// Replaces `spectral_commutativity::mat_mul` (triple-nested CPU loop).
/// Uses `Tensor::matmul` which dispatches through `BarraCUDA`'s 4-tier
/// kernel router (`Naive`/`Tiled16`/`CpuTiled32`/`GpuEvolved32`).
///
/// # Errors
///
/// Returns an error if GPU tensor creation or matmul fails.
pub fn mat_mul_gpu(
    a: &[f64],
    b: &[f64],
    n: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();

    let a_t = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("mat_mul_gpu A upload: {e}"))?;
    let b_t = Tensor::from_data(&b_f32, vec![n, n], device.clone())
        .map_err(|e| format!("mat_mul_gpu B upload: {e}"))?;

    let c_t = a_t
        .matmul(&b_t)
        .map_err(|e| format!("mat_mul_gpu matmul: {e}"))?;

    let c_f32 = c_t
        .to_vec()
        .map_err(|e| format!("mat_mul_gpu readback: {e}"))?;

    Ok(c_f32.into_iter().map(f64::from).collect())
}

/// GPU Frobenius norm: sqrt(sum of squares).
///
/// Replaces `spectral_commutativity::frobenius_norm` (CPU `.iter().sum()`).
/// Uses `Tensor::norm` (L2 norm reduction).
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn frobenius_norm_gpu(a: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, String> {
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let n = a_f32.len();

    let a_t = Tensor::from_data(&a_f32, vec![n], device.clone())
        .map_err(|e| format!("frobenius_norm_gpu upload: {e}"))?;

    let norm_t = a_t
        .norm()
        .map_err(|e| format!("frobenius_norm_gpu norm: {e}"))?;

    let result = norm_t
        .to_vec()
        .map_err(|e| format!("frobenius_norm_gpu readback: {e}"))?;

    Ok(f64::from(result[0]))
}

/// GPU transpose for n×n matrix.
///
/// Replaces `spectral_commutativity::transpose` (CPU double loop).
/// Uses `Tensor::transpose`.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn transpose_gpu(a: &[f64], n: usize, device: &Arc<WgpuDevice>) -> Result<Vec<f64>, String> {
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();

    let a_t = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("transpose_gpu upload: {e}"))?;

    let t_t = a_t
        .transpose()
        .map_err(|e| format!("transpose_gpu transpose: {e}"))?;

    let t_f32 = t_t
        .to_vec()
        .map_err(|e| format!("transpose_gpu readback: {e}"))?;

    Ok(t_f32.into_iter().map(f64::from).collect())
}

/// GPU commutator [A,B] = AB - BA.
///
/// Replaces `spectral_commutativity::commutator`.
/// Two GPU matmuls + elementwise subtract, all on GPU.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn commutator_gpu(
    a: &[f64],
    b: &[f64],
    n: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();

    let a_t = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("commutator_gpu A: {e}"))?;
    let b_t = Tensor::from_data(&b_f32, vec![n, n], device.clone())
        .map_err(|e| format!("commutator_gpu B: {e}"))?;

    // matmul(self, &other) consumes self; recreate for second product
    let b_t2 = Tensor::from_data(&b_f32, vec![n, n], device.clone())
        .map_err(|e| format!("commutator_gpu B2: {e}"))?;
    let a_t2 = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("commutator_gpu A2: {e}"))?;

    let ab = a_t
        .matmul(&b_t)
        .map_err(|e| format!("commutator_gpu AB: {e}"))?;
    let ba = b_t2
        .matmul(&a_t2)
        .map_err(|e| format!("commutator_gpu BA: {e}"))?;

    let diff = ab
        .sub(&ba)
        .map_err(|e| format!("commutator_gpu sub: {e}"))?;

    let out = diff
        .to_vec()
        .map_err(|e| format!("commutator_gpu readback: {e}"))?;

    Ok(out.into_iter().map(f64::from).collect())
}

/// GPU distance to normal: ||A*A - AA*||_F / (2||A||_F).
///
/// Replaces `spectral_commutativity::distance_to_normal`.
/// Full computation on GPU: transpose, two matmuls, subtract, norms.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn distance_to_normal_gpu(
    a: &[f64],
    n: usize,
    device: &Arc<WgpuDevice>,
) -> Result<f64, String> {
    let norm = frobenius_norm_gpu(a, device)?;
    if norm < crate::primitives::LOG_GUARD {
        return Ok(0.0);
    }

    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();

    // transpose(&self) borrows; matmul(self, &other) consumes self — recreate per product
    let a_for_at = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("distance_to_normal_gpu A_at: {e}"))?;
    let at = a_for_at
        .transpose()
        .map_err(|e| format!("distance_to_normal_gpu transpose: {e}"))?;

    let a_for_ata = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("distance_to_normal_gpu A_ata: {e}"))?;
    let ata = at
        .matmul(&a_for_ata)
        .map_err(|e| format!("distance_to_normal_gpu AtA: {e}"))?;

    let a_for_aat = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("distance_to_normal_gpu A_aat: {e}"))?;
    let a_for_aat_t = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("distance_to_normal_gpu A_aat2: {e}"))?;
    let at_for_aat = a_for_aat_t
        .transpose()
        .map_err(|e| format!("distance_to_normal_gpu At_aat: {e}"))?;
    let aat = a_for_aat
        .matmul(&at_for_aat)
        .map_err(|e| format!("distance_to_normal_gpu AAt: {e}"))?;

    let diff = ata
        .sub(&aat)
        .map_err(|e| format!("distance_to_normal_gpu sub: {e}"))?;

    let diff_norm = diff
        .norm()
        .map_err(|e| format!("distance_to_normal_gpu norm: {e}"))?;

    let result = diff_norm
        .to_vec()
        .map_err(|e| format!("distance_to_normal_gpu readback: {e}"))?;

    Ok(f64::from(result[0]) / (2.0 * norm))
}

// ═══════════════════════════════════════════════════════════════════
// Activations and reductions (transformer, counterdiabatic GPU promotion)
// ═══════════════════════════════════════════════════════════════════

/// GPU softmax over a 1D vector.
///
/// Replaces `transformer::softmax` and `counterdiabatic::boltzmann_distribution`.
/// Uses `Tensor::softmax` (global softmax — normalizes over all elements).
///
/// For Boltzmann: pre-multiply by beta before calling.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn softmax_gpu(x: &[f64], device: &Arc<WgpuDevice>) -> Result<Vec<f64>, String> {
    let x_f32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let n = x_f32.len();

    let x_t = Tensor::from_data(&x_f32, vec![n], device.clone())
        .map_err(|e| format!("softmax_gpu upload: {e}"))?;

    let sm = x_t.softmax().map_err(|e| format!("softmax_gpu: {e}"))?;

    let out = sm
        .to_vec()
        .map_err(|e| format!("softmax_gpu readback: {e}"))?;

    Ok(out.into_iter().map(f64::from).collect())
}

/// GPU Boltzmann distribution: softmax(beta * fitnesses).
///
/// Replaces `counterdiabatic::boltzmann_distribution`.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn boltzmann_gpu(
    fitnesses: &[f64],
    beta: f64,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    let scaled: Vec<f64> = fitnesses.iter().map(|&f| f * beta).collect();
    softmax_gpu(&scaled, device)
}

/// GPU GELU activation.
///
/// Replaces `transformer::gelu`.
/// Uses `Tensor::gelu_wgsl`.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn gelu_gpu(x: &[f64], device: &Arc<WgpuDevice>) -> Result<Vec<f64>, String> {
    let x_f32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let n = x_f32.len();

    let x_t = Tensor::from_data(&x_f32, vec![n], device.clone())
        .map_err(|e| format!("gelu_gpu upload: {e}"))?;

    let out_t = x_t.gelu_wgsl().map_err(|e| format!("gelu_gpu gelu: {e}"))?;

    let out = out_t
        .to_vec()
        .map_err(|e| format!("gelu_gpu readback: {e}"))?;

    Ok(out.into_iter().map(f64::from).collect())
}

// ═══════════════════════════════════════════════════════════════════
// Reductions (modes, eco_dynamics, meta_population GPU promotion)
// ═══════════════════════════════════════════════════════════════════

/// GPU L2 distance between two vectors.
///
/// Replaces `modes::l2_distance`.
/// Computes sqrt(sum((a-b)^2)) on GPU via subtract + norm.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn l2_distance_gpu(a: &[f64], b: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, String> {
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
    let n = a_f32.len();

    let a_t = Tensor::from_data(&a_f32, vec![n], device.clone())
        .map_err(|e| format!("l2_distance_gpu A: {e}"))?;
    let b_t = Tensor::from_data(&b_f32, vec![n], device.clone())
        .map_err(|e| format!("l2_distance_gpu B: {e}"))?;

    let diff = a_t
        .sub(&b_t)
        .map_err(|e| format!("l2_distance_gpu sub: {e}"))?;

    let norm = diff
        .norm()
        .map_err(|e| format!("l2_distance_gpu norm: {e}"))?;

    let result = norm
        .to_vec()
        .map_err(|e| format!("l2_distance_gpu readback: {e}"))?;

    Ok(f64::from(result[0]))
}

/// GPU mean reduction over a vector.
///
/// Replaces various `.iter().sum::<f64>() / n as f64` patterns.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn mean_gpu(data: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, String> {
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let n = data_f32.len();

    let t = Tensor::from_data(&data_f32, vec![n], device.clone())
        .map_err(|e| format!("mean_gpu upload: {e}"))?;

    let m = t.mean().map_err(|e| format!("mean_gpu mean: {e}"))?;

    let result = m.to_vec().map_err(|e| format!("mean_gpu readback: {e}"))?;

    Ok(f64::from(result[0]))
}

/// GPU sum reduction over a vector.
///
/// Replaces `.iter().sum()` patterns across modules.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn sum_gpu(data: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, String> {
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let n = data_f32.len();

    let t = Tensor::from_data(&data_f32, vec![n], device.clone())
        .map_err(|e| format!("sum_gpu upload: {e}"))?;

    let s = t.sum().map_err(|e| format!("sum_gpu sum: {e}"))?;

    let result = s.to_vec().map_err(|e| format!("sum_gpu readback: {e}"))?;

    Ok(f64::from(result[0]))
}

/// GPU max reduction over a vector.
///
/// Replaces `.fold(f64::NEG_INFINITY, f64::max)` patterns.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn max_gpu(data: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, String> {
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let n = data_f32.len();

    let t = Tensor::from_data(&data_f32, vec![n], device.clone())
        .map_err(|e| format!("max_gpu upload: {e}"))?;

    let m = t.max().map_err(|e| format!("max_gpu max: {e}"))?;

    let result = m.to_vec().map_err(|e| format!("max_gpu readback: {e}"))?;

    Ok(f64::from(result[0]))
}

// ═══════════════════════════════════════════════════════════════════
// Neural network forward (swarm_robotics GPU promotion)
// ═══════════════════════════════════════════════════════════════════

/// GPU neural network forward pass: input → hidden (sigmoid) → output (sigmoid).
///
/// Replaces `swarm_robotics::neural_forward`.
/// Uses Tensor matmul + sigmoid for each layer.
///
/// # Layout
///
/// `params` is flat: `[w_hidden (h×in), b_hidden (h), w_out (out×h), b_out (out)]`
///
/// # Errors
///
/// Returns an error if GPU operations fail.
#[allow(clippy::too_many_arguments)]
pub fn neural_forward_gpu(
    weights_hidden: &[f64],
    bias_hidden: &[f64],
    weights_output: &[f64],
    bias_output: &[f64],
    input: &[f64],
    hidden_size: usize,
    output_size: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    let input_size = input.len();

    let w_h: Vec<f32> = weights_hidden.iter().map(|&x| x as f32).collect();
    let b_h: Vec<f32> = bias_hidden.iter().map(|&x| x as f32).collect();
    let w_o: Vec<f32> = weights_output.iter().map(|&x| x as f32).collect();
    let b_o: Vec<f32> = bias_output.iter().map(|&x| x as f32).collect();
    let inp: Vec<f32> = input.iter().map(|&x| x as f32).collect();

    let input_t = Tensor::from_data(&inp, vec![1, input_size], device.clone())
        .map_err(|e| format!("nn_forward input: {e}"))?;

    let wh_t = Tensor::from_data(&w_h, vec![hidden_size, input_size], device.clone())
        .map_err(|e| format!("nn_forward W_h: {e}"))?;
    let bh_t = Tensor::from_data(&b_h, vec![1, hidden_size], device.clone())
        .map_err(|e| format!("nn_forward b_h: {e}"))?;

    let wo_t = Tensor::from_data(&w_o, vec![output_size, hidden_size], device.clone())
        .map_err(|e| format!("nn_forward W_o: {e}"))?;
    let bo_t = Tensor::from_data(&b_o, vec![1, output_size], device.clone())
        .map_err(|e| format!("nn_forward b_o: {e}"))?;

    // Hidden layer: sigmoid(input × W_h^T + b_h)
    // transpose(&self) borrows; matmul(self, &other) and sigmoid(self) consume self
    let wh_transposed = wh_t
        .transpose()
        .map_err(|e| format!("nn_forward W_h^T: {e}"))?;
    let hidden_pre = input_t
        .matmul(&wh_transposed)
        .map_err(|e| format!("nn_forward hidden matmul: {e}"))?;
    let hidden_biased = hidden_pre
        .add(&bh_t)
        .map_err(|e| format!("nn_forward hidden+bias: {e}"))?;
    let hidden = hidden_biased
        .sigmoid()
        .map_err(|e| format!("nn_forward hidden sigmoid: {e}"))?;

    // Output layer: sigmoid(hidden × W_o^T + b_o)
    let wo_transposed = wo_t
        .transpose()
        .map_err(|e| format!("nn_forward W_o^T: {e}"))?;
    let output_pre = hidden
        .matmul(&wo_transposed)
        .map_err(|e| format!("nn_forward output matmul: {e}"))?;
    let output_biased = output_pre
        .add(&bo_t)
        .map_err(|e| format!("nn_forward output+bias: {e}"))?;
    let output = output_biased
        .sigmoid()
        .map_err(|e| format!("nn_forward output sigmoid: {e}"))?;

    let result = output
        .to_vec()
        .map_err(|e| format!("nn_forward readback: {e}"))?;

    Ok(result.into_iter().map(f64::from).collect())
}

// ═══════════════════════════════════════════════════════════════════
// Shannon entropy (primitives GPU promotion)
// ═══════════════════════════════════════════════════════════════════

/// GPU Shannon entropy: -sum(p * ln(p)).
///
/// Replaces `primitives::shannon_entropy`.
/// Uses Tensor log + mul + sum.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn shannon_entropy_gpu(probabilities: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, String> {
    let p_f32: Vec<f32> = probabilities
        .iter()
        .map(|&p| (p.max(crate::primitives::LOG_GUARD)) as f32)
        .collect();
    let n = p_f32.len();

    let p_for_log = Tensor::from_data(&p_f32, vec![n], device.clone())
        .map_err(|e| format!("entropy_gpu upload_log: {e}"))?;
    let p_for_mul = Tensor::from_data(&p_f32, vec![n], device.clone())
        .map_err(|e| format!("entropy_gpu upload_mul: {e}"))?;

    let log_p = p_for_log
        .log_wgsl()
        .map_err(|e| format!("entropy_gpu log: {e}"))?;

    let p_log_p = p_for_mul
        .mul(&log_p)
        .map_err(|e| format!("entropy_gpu mul: {e}"))?;

    let total = p_log_p.sum().map_err(|e| format!("entropy_gpu sum: {e}"))?;

    let result = total
        .to_vec()
        .map_err(|e| format!("entropy_gpu readback: {e}"))?;

    Ok(-f64::from(result[0]))
}

// ═══════════════════════════════════════════════════════════════════
// Variance (meta_population, counterdiabatic GPU promotion)
// ═══════════════════════════════════════════════════════════════════

/// GPU population variance: E[x^2] - E[x]^2.
///
/// Replaces manual variance loops across modules.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn variance_gpu(data: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, String> {
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let n = data_f32.len();

    let t = Tensor::from_data(&data_f32, vec![n], device.clone())
        .map_err(|e| format!("variance_gpu upload: {e}"))?;

    let mean_t = t.mean().map_err(|e| format!("variance_gpu mean: {e}"))?;
    let mean_val = mean_t
        .to_vec()
        .map_err(|e| format!("variance_gpu mean readback: {e}"))?[0];

    // Compute (x - mean)^2 via scalar ops: x^2 - 2*mean*x + mean^2
    // Simpler: re-upload data, subtract mean_broadcast, square, mean
    let mean_vec = vec![mean_val; n];
    let mean_broadcast = Tensor::from_data(&mean_vec, vec![n], device.clone())
        .map_err(|e| format!("variance_gpu mean_vec: {e}"))?;

    let diff = t
        .sub(&mean_broadcast)
        .map_err(|e| format!("variance_gpu sub: {e}"))?;
    let sq = diff
        .mul(&diff)
        .map_err(|e| format!("variance_gpu sq: {e}"))?;
    let var = sq.mean().map_err(|e| format!("variance_gpu var: {e}"))?;

    let result = var
        .to_vec()
        .map_err(|e| format!("variance_gpu readback: {e}"))?;

    Ok(f64::from(result[0]))
}

// ═══════════════════════════════════════════════════════════════════
// Statistics (meta_population, pangenome_selection GPU promotion)
// ═══════════════════════════════════════════════════════════════════

/// GPU Pearson correlation between two vectors.
///
/// `r = cov(x,y) / (std_x * std_y)`.
///
/// Replaces `meta_population` Pearson correlation.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn pearson_correlation_gpu(
    x: &[f64],
    y: &[f64],
    device: &Arc<WgpuDevice>,
) -> Result<f64, String> {
    let mean_x = mean_gpu(x, device)?;
    let mean_y = mean_gpu(y, device)?;

    let n = x.len();
    let dx: Vec<f64> = x.iter().map(|&v| v - mean_x).collect();
    let dy: Vec<f64> = y.iter().map(|&v| v - mean_y).collect();

    let dx_f32: Vec<f32> = dx.iter().map(|&v| v as f32).collect();
    let dy_f32: Vec<f32> = dy.iter().map(|&v| v as f32).collect();

    let dx_t = Tensor::from_data(&dx_f32, vec![n], device.clone())
        .map_err(|e| format!("pearson dx: {e}"))?;
    let dy_t = Tensor::from_data(&dy_f32, vec![n], device.clone())
        .map_err(|e| format!("pearson dy: {e}"))?;

    let cov_t = dx_t.mul(&dy_t).map_err(|e| format!("pearson mul: {e}"))?;
    let cov = cov_t
        .sum()
        .map_err(|e| format!("pearson cov sum: {e}"))?
        .to_vec()
        .map_err(|e| format!("pearson cov read: {e}"))?[0];

    let var_x = variance_gpu(x, device)?;
    let var_y = variance_gpu(y, device)?;

    let denom = (var_x * var_y).sqrt() * n as f64;
    if denom < crate::primitives::LOG_GUARD {
        return Ok(0.0);
    }

    Ok(f64::from(cov) / denom)
}

/// GPU chi-squared statistic: sum((observed - expected)^2 / expected).
///
/// Replaces `pangenome_selection::spectrum_chi_squared`.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn chi_squared_gpu(
    observed: &[f64],
    expected: &[f64],
    device: &Arc<WgpuDevice>,
) -> Result<f64, String> {
    let n = observed.len();
    let obs_f32: Vec<f32> = observed.iter().map(|&x| x as f32).collect();
    let exp_f32: Vec<f32> = expected.iter().map(|&x| (x as f32).max(1e-30)).collect();

    let obs_t = Tensor::from_data(&obs_f32, vec![n], device.clone())
        .map_err(|e| format!("chi2 obs: {e}"))?;
    let exp_t = Tensor::from_data(&exp_f32, vec![n], device.clone())
        .map_err(|e| format!("chi2 exp: {e}"))?;

    let diff = obs_t.sub(&exp_t).map_err(|e| format!("chi2 sub: {e}"))?;
    let sq = diff.mul(&diff).map_err(|e| format!("chi2 sq: {e}"))?;

    let ratio_vals: Vec<f32> = sq
        .to_vec()
        .map_err(|e| format!("chi2 sq_read: {e}"))?
        .iter()
        .zip(exp_f32.iter())
        .map(|(&s, &e)| s / e)
        .collect();

    let ratio_t = Tensor::from_data(&ratio_vals, vec![n], device.clone())
        .map_err(|e| format!("chi2 ratio: {e}"))?;

    let result = ratio_t
        .sum()
        .map_err(|e| format!("chi2 sum: {e}"))?
        .to_vec()
        .map_err(|e| format!("chi2 read: {e}"))?;

    Ok(f64::from(result[0]))
}

/// GPU KL divergence: sum(p * ln(p/q)).
///
/// Replaces `counterdiabatic::kl_divergence`.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn kl_divergence_gpu(p: &[f64], q: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, String> {
    let n = p.len();
    let guard = crate::primitives::LOG_GUARD;
    let p_sum: f64 = p.iter().sum();
    let q_sum: f64 = q.iter().sum();

    let log_ratios: Vec<f32> = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            let pi_n = (pi / p_sum).max(guard);
            let qi_n = (qi / q_sum).max(guard);
            (pi_n * (pi_n / qi_n).ln()) as f32
        })
        .collect();

    let t = Tensor::from_data(&log_ratios, vec![n], device.clone())
        .map_err(|e| format!("kl upload: {e}"))?;

    let result = t
        .sum()
        .map_err(|e| format!("kl sum: {e}"))?
        .to_vec()
        .map_err(|e| format!("kl read: {e}"))?;

    Ok(f64::from(result[0]))
}

// ═══════════════════════════════════════════════════════════════════
// HMM operations (hmm, introgression GPU promotion)
// ═══════════════════════════════════════════════════════════════════

/// GPU HMM forward step: `alpha[t] = normalize(B[:,o_t] * (A^T @ alpha[t-1]))`.
///
/// Single timestep of the forward algorithm via GPU GEMV + elementwise.
/// The full forward pass calls this in a loop; each step's GPU matmul
/// replaces the CPU double loop.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn hmm_forward_step_gpu(
    alpha_prev: &[f64],
    transition: &[f64],
    emission_col: &[f64],
    n_states: usize,
    device: &Arc<WgpuDevice>,
) -> Result<(Vec<f64>, f64), String> {
    let a_f32: Vec<f32> = alpha_prev.iter().map(|&x| x as f32).collect();
    let t_f32: Vec<f32> = transition.iter().map(|&x| x as f32).collect();
    let e_f32: Vec<f32> = emission_col.iter().map(|&x| x as f32).collect();

    let alpha_t = Tensor::from_data(&a_f32, vec![1, n_states], device.clone())
        .map_err(|e| format!("hmm_fwd alpha: {e}"))?;
    let trans_t = Tensor::from_data(&t_f32, vec![n_states, n_states], device.clone())
        .map_err(|e| format!("hmm_fwd trans: {e}"))?;

    // alpha @ A (row-vector × matrix = row-vector of n_states)
    let propagated = alpha_t
        .matmul(&trans_t)
        .map_err(|e| format!("hmm_fwd matmul: {e}"))?;

    let emit_t = Tensor::from_data(&e_f32, vec![1, n_states], device.clone())
        .map_err(|e| format!("hmm_fwd emit: {e}"))?;

    let raw = propagated
        .mul(&emit_t)
        .map_err(|e| format!("hmm_fwd mul: {e}"))?;

    let scale_t = raw.sum().map_err(|e| format!("hmm_fwd sum: {e}"))?;
    let scale_val = scale_t
        .to_vec()
        .map_err(|e| format!("hmm_fwd scale_read: {e}"))?[0];

    let raw_vec = raw.to_vec().map_err(|e| format!("hmm_fwd raw_read: {e}"))?;

    let scale = f64::from(scale_val).max(crate::primitives::LOG_GUARD);
    let alpha_new: Vec<f64> = raw_vec.iter().map(|&x| f64::from(x) / scale).collect();

    Ok((alpha_new, scale))
}

/// GPU pairwise distance matrix for n vectors of dimension d.
///
/// Returns flat upper-triangle distances (n*(n-1)/2 elements).
/// Replaces `meta_population::geographic_distance_matrix` and
/// `pangenome_selection::jaccard_distance_matrix` distance loops.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn pairwise_l2_matrix_gpu(
    data: &[f64],
    n: usize,
    dim: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    let n_pairs = n * (n - 1) / 2;
    let mut distances = Vec::with_capacity(n_pairs);

    for i in 0..n {
        for j in (i + 1)..n {
            let a = &data[i * dim..(i + 1) * dim];
            let b = &data[j * dim..(j + 1) * dim];
            distances.push(l2_distance_gpu(a, b, device)?);
        }
    }

    Ok(distances)
}

// ═══════════════════════════════════════════════════════════════════
// Hill function (signal_integration, regulatory_network GPU promotion)
// ═══════════════════════════════════════════════════════════════════

/// GPU batch Hill activation: `V_max * x^n / (K^n + x^n)`.
///
/// Genuinely GPU-computed via Tensor log → scale → exp → div pipeline.
/// Replaces `primitives::hill_activation` for batch processing.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn hill_activation_batch_gpu(
    x: &[f64],
    vmax: f64,
    k: f64,
    n_hill: f64,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    let len = x.len();
    if len == 0 {
        return Ok(Vec::new());
    }

    let kn = (k.powf(n_hill)) as f32;
    let n_f32 = n_hill as f32;
    let vmax_f32 = vmax as f32;
    let guard = crate::primitives::HILL_EPS as f32;

    let x_f32: Vec<f32> = x.iter().map(|&v| (v.max(1e-30)) as f32).collect();

    // x^n = exp(n * ln(x))
    let x_t =
        Tensor::from_data(&x_f32, vec![len], device.clone()).map_err(|e| format!("hill x: {e}"))?;
    let log_x = x_t.log_wgsl().map_err(|e| format!("hill log: {e}"))?;
    let scaled_log = log_x
        .mul_scalar(n_f32)
        .map_err(|e| format!("hill scale: {e}"))?;
    let x_pow_n = scaled_log
        .exp_wgsl()
        .map_err(|e| format!("hill exp: {e}"))?;

    // denominator: K^n + x^n + eps
    let kn_t = Tensor::from_data(&vec![kn; len], vec![len], device.clone())
        .map_err(|e| format!("hill kn: {e}"))?;
    let eps_t = Tensor::from_data(&vec![guard; len], vec![len], device.clone())
        .map_err(|e| format!("hill eps: {e}"))?;
    let sum1 = x_pow_n.add(&kn_t).map_err(|e| format!("hill add1: {e}"))?;
    let denom = sum1.add(&eps_t).map_err(|e| format!("hill add2: {e}"))?;

    // V_max * x^n / (K^n + x^n + eps)
    let ratio = x_pow_n.div(&denom).map_err(|e| format!("hill div: {e}"))?;
    let result = ratio
        .mul_scalar(vmax_f32)
        .map_err(|e| format!("hill vmax: {e}"))?;

    let out = result.to_vec().map_err(|e| format!("hill read: {e}"))?;
    Ok(out.into_iter().map(f64::from).collect())
}

// ═══════════════════════════════════════════════════════════════════
// Phase B: HMM backward + Viterbi (hmm, introgression GPU promotion)
// ═══════════════════════════════════════════════════════════════════

/// GPU HMM backward step: `β_t[i] = sum_j(A[i,j] * B[j,o] * β_{t+1}[j]) / scale`.
///
/// Single reverse-timestep via GPU GEMV. The full backward pass calls
/// this in a loop from T-2 down to 0.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn hmm_backward_step_gpu(
    beta_next: &[f64],
    transition: &[f64],
    emission_col: &[f64],
    scale: f64,
    n_states: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    let b_f32: Vec<f32> = beta_next.iter().map(|&x| x as f32).collect();
    let t_f32: Vec<f32> = transition.iter().map(|&x| x as f32).collect();
    let e_f32: Vec<f32> = emission_col.iter().map(|&x| x as f32).collect();

    // weighted = emit ⊙ β_{t+1}
    let beta_t = Tensor::from_data(&b_f32, vec![1, n_states], device.clone())
        .map_err(|e| format!("hmm_bwd beta: {e}"))?;
    let emit_t = Tensor::from_data(&e_f32, vec![1, n_states], device.clone())
        .map_err(|e| format!("hmm_bwd emit: {e}"))?;
    let weighted = beta_t
        .mul(&emit_t)
        .map_err(|e| format!("hmm_bwd mul: {e}"))?;

    // β_t = weighted @ A^T / scale
    // β_t[i] = sum_j(A[i,j] * weighted[j]) = (weighted @ A^T)[i]
    let trans_t = Tensor::from_data(&t_f32, vec![n_states, n_states], device.clone())
        .map_err(|e| format!("hmm_bwd trans: {e}"))?;
    let at = trans_t
        .transpose()
        .map_err(|e| format!("hmm_bwd transpose: {e}"))?;
    let result = weighted
        .matmul(&at)
        .map_err(|e| format!("hmm_bwd matmul: {e}"))?;

    let result_vec = result.to_vec().map_err(|e| format!("hmm_bwd read: {e}"))?;

    let guard = crate::primitives::LOG_GUARD;
    let safe_scale = if scale.abs() < guard { guard } else { scale };
    let beta_new: Vec<f64> = result_vec
        .iter()
        .map(|&x| f64::from(x) / safe_scale)
        .collect();

    Ok(beta_new)
}

/// GPU HMM Viterbi step: `δ_t[j] = max_i(δ_{t-1}[i] + logA[i,j]) + logB[j,o_t]`.
///
/// Returns `(delta_t, psi_t)` where `psi_t[j] = argmax_i(...)`.
/// Score matrix construction and max-reduction run on GPU; argmax
/// (N comparisons per state) runs on CPU since `Tensor` lacks argmax.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn hmm_viterbi_step_gpu(
    delta_prev: &[f64],
    log_transition: &[f64],
    log_emission_col: &[f64],
    n_states: usize,
    device: &Arc<WgpuDevice>,
) -> Result<(Vec<f64>, Vec<usize>), String> {
    let n = n_states;
    let d_f32: Vec<f32> = delta_prev.iter().map(|&x| x as f32).collect();
    let la_f32: Vec<f32> = log_transition.iter().map(|&x| x as f32).collect();

    // Score matrix S[i,j] = δ_{t-1}[i] + logA[i,j]
    // Broadcast δ from [N,1] to [N,N], then add logA
    let delta_col = Tensor::from_data(&d_f32, vec![n, 1], device.clone())
        .map_err(|e| format!("viterbi delta: {e}"))?;
    let delta_broad = delta_col
        .broadcast(vec![n, n])
        .map_err(|e| format!("viterbi broadcast: {e}"))?;
    let log_a = Tensor::from_data(&la_f32, vec![n, n], device.clone())
        .map_err(|e| format!("viterbi log_a: {e}"))?;
    let scores = delta_broad
        .add(&log_a)
        .map_err(|e| format!("viterbi add: {e}"))?;

    // Max along dim 0 (across source states i, for each target state j)
    let max_vals = scores
        .max_dim(0, false)
        .map_err(|e| format!("viterbi max: {e}"))?;

    // Read back for argmax (CPU) and max values
    let scores_flat = scores
        .to_vec()
        .map_err(|e| format!("viterbi scores: {e}"))?;
    let max_f32 = max_vals
        .to_vec()
        .map_err(|e| format!("viterbi max_read: {e}"))?;

    let mut delta_new = Vec::with_capacity(n);
    let mut psi = Vec::with_capacity(n);
    for j in 0..n {
        delta_new.push(f64::from(max_f32[j]) + log_emission_col[j]);

        let mut best_i = 0;
        let mut best_val = f32::NEG_INFINITY;
        for i in 0..n {
            let val = scores_flat[i * n + j];
            if val > best_val {
                best_val = val;
                best_i = i;
            }
        }
        psi.push(best_i);
    }

    Ok((delta_new, psi))
}

// ═══════════════════════════════════════════════════════════════════
// Phase B: Meta-population GPU promotion
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
// Phase B: Game theory GPU promotion
// ═══════════════════════════════════════════════════════════════════

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

    // fitness = P @ x  ([2,2] @ [2,1] → [2,1])
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

#[cfg(test)]
mod tests {
    #[test]
    fn f32_f64_roundtrip_precision() {
        let x = [1.0_f64, 2.0, 3.0, 0.5, -1.0];
        let f32s: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let back: Vec<f64> = f32s.into_iter().map(f64::from).collect();
        for (orig, rt) in x.iter().zip(back.iter()) {
            assert!((orig - rt).abs() < 1e-6, "roundtrip: {orig} -> {rt}");
        }
    }
}
