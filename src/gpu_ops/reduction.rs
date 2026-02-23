// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-accelerated reductions and statistics: L2, mean, sum, max, variance,
//! entropy, Pearson correlation, chi-squared, KL divergence, neural forward.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use std::sync::Arc;

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

/// GPU population variance: `E[x²] - E[x]²`.
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

    let sq_vals = sq.to_vec().map_err(|e| format!("chi2 sq_read: {e}"))?;
    let ratio_vals: Vec<f32> = sq_vals
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
