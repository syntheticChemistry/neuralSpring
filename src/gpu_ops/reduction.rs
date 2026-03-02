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

/// Two-layer neural network parameters for GPU forward pass.
pub struct NeuralForwardParams<'a> {
    pub weights_hidden: &'a [f64],
    pub bias_hidden: &'a [f64],
    pub weights_output: &'a [f64],
    pub bias_output: &'a [f64],
    pub input: &'a [f64],
    pub hidden_size: usize,
    pub output_size: usize,
}

/// GPU neural network forward pass: input → hidden (sigmoid) → output (sigmoid).
///
/// Replaces `swarm_robotics::neural_forward`.
/// Uses Tensor matmul + sigmoid for each layer.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn neural_forward_gpu(
    params: &NeuralForwardParams<'_>,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    let NeuralForwardParams {
        weights_hidden,
        bias_hidden,
        weights_output,
        bias_output,
        input,
        hidden_size,
        output_size,
    } = params;
    let input_size = input.len();

    let w_h: Vec<f32> = weights_hidden.iter().map(|&x| x as f32).collect();
    let b_h: Vec<f32> = bias_hidden.iter().map(|&x| x as f32).collect();
    let w_o: Vec<f32> = weights_output.iter().map(|&x| x as f32).collect();
    let b_o: Vec<f32> = bias_output.iter().map(|&x| x as f32).collect();
    let inp: Vec<f32> = input.iter().map(|&x| x as f32).collect();

    let input_t = Tensor::from_data(&inp, vec![1, input_size], device.clone())
        .map_err(|e| format!("nn_forward input: {e}"))?;

    let wh_t = Tensor::from_data(&w_h, vec![*hidden_size, input_size], device.clone())
        .map_err(|e| format!("nn_forward W_h: {e}"))?;
    let bh_t = Tensor::from_data(&b_h, vec![1, *hidden_size], device.clone())
        .map_err(|e| format!("nn_forward b_h: {e}"))?;

    let wo_t = Tensor::from_data(&w_o, vec![*output_size, *hidden_size], device.clone())
        .map_err(|e| format!("nn_forward W_o: {e}"))?;
    let bo_t = Tensor::from_data(&b_o, vec![1, *output_size], device.clone())
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

/// GPU Shannon entropy: `-sum(p * ln(p))`.
///
/// Delegates to `barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64`
/// (f64 precision, fused map-reduce WGSL shader). Origin: wetSpring
/// bio shaders → hotSpring precision infrastructure → `ToadStool`.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn shannon_entropy_gpu(probabilities: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, String> {
    use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

    let op =
        FusedMapReduceF64::new(device.clone()).map_err(|e| format!("entropy_gpu init: {e}"))?;
    op.shannon_entropy(probabilities)
        .map_err(|e| format!("entropy_gpu: {e}"))
}

/// GPU population variance (divides by N) via Welford's algorithm.
///
/// Delegates to `barracuda::ops::variance_reduce_f64::VarianceReduceF64`
/// (f64 precision, Welford online WGSL shader). Origin: hotSpring
/// precision infrastructure → `ToadStool`.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn variance_gpu(data: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, String> {
    use barracuda::ops::variance_reduce_f64::VarianceReduceF64;

    VarianceReduceF64::population_variance(device.clone(), data)
        .map_err(|e| format!("variance_gpu: {e}"))
}

/// GPU Pearson correlation between two vectors.
///
/// Delegates to `barracuda::ops::correlation_f64_wgsl::CorrelationF64`
/// (f64 precision, dedicated WGSL shader). Origin: wetSpring
/// bio shaders → hotSpring precision infrastructure → `ToadStool`.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn pearson_correlation_gpu(
    x: &[f64],
    y: &[f64],
    device: &Arc<WgpuDevice>,
) -> Result<f64, String> {
    use barracuda::ops::correlation_f64_wgsl::CorrelationF64;

    let op = CorrelationF64::new(device.clone()).map_err(|e| format!("pearson_gpu init: {e}"))?;
    op.correlation(x, y)
        .map_err(|e| format!("pearson_gpu: {e}"))
}

/// GPU chi-squared statistic: sum((observed - expected)^2 / expected).
///
/// Delegates to `barracuda::ops::fused_chi_squared_f64::FusedChiSquaredGpu`
/// (f64 precision, single-dispatch WGSL shader). Origin: neuralSpring
/// `chi_squared_f64.wgsl` → `ToadStool` S76 absorption → `FusedChiSquaredGpu`.
///
/// Cross-spring: hotSpring precision infrastructure (f64 shader compilation
/// pipeline) + neuralSpring domain shader → fused upstream op.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn chi_squared_gpu(
    observed: &[f64],
    expected: &[f64],
    device: &Arc<WgpuDevice>,
) -> Result<f64, String> {
    use barracuda::ops::fused_chi_squared_f64::FusedChiSquaredGpu;

    FusedChiSquaredGpu::execute(device.clone(), observed, expected)
        .map(|r| r.statistic)
        .map_err(|e| format!("chi_squared_gpu: {e}"))
}

/// GPU KL divergence: sum(p * ln(p/q)).
///
/// Delegates to `barracuda::ops::fused_kl_divergence_f64::FusedKlDivergenceGpu`
/// (f64 precision, single-dispatch WGSL shader). Origin: neuralSpring
/// `kl_divergence_f64.wgsl` → `ToadStool` S76 absorption → `FusedKlDivergenceGpu`.
///
/// Cross-spring: neuralSpring domain shader → hotSpring f64 compilation
/// infrastructure → fused upstream op consumed by all Springs.
///
/// Normalizes inputs to probability distributions before dispatch to
/// maintain backward compatibility with the original CPU path.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn kl_divergence_gpu(p: &[f64], q: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, String> {
    use barracuda::ops::fused_kl_divergence_f64::FusedKlDivergenceGpu;

    let p_sum: f64 = p.iter().sum();
    let q_sum: f64 = q.iter().sum();
    let p_norm: Vec<f64> = p.iter().map(|&x| x / p_sum).collect();
    let q_norm: Vec<f64> = q.iter().map(|&x| x / q_sum).collect();

    FusedKlDivergenceGpu::execute(device.clone(), &p_norm, &q_norm)
        .map_err(|e| format!("kl_divergence_gpu: {e}"))
}
