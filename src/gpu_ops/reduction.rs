// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-accelerated reductions and statistics: L2, mean, sum, max, variance,
//! entropy, Pearson correlation, chi-squared, KL divergence, neural forward.

#![expect(
    clippy::cast_possible_truncation,
    clippy::similar_names,
    reason = "GPU reductions convert f64→f32 for hardware; weight/bias tensor pairs share name prefixes"
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use std::sync::Arc;

use crate::error::TensorError;

/// GPU L2 distance between two vectors.
///
/// Replaces `modes::l2_distance`.
/// Computes sqrt(sum((a-b)^2)) on GPU via subtract + norm.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn l2_distance_gpu(a: &[f64], b: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, TensorError> {
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
    let n = a_f32.len();

    let a_t =
        Tensor::from_data(&a_f32, vec![n], device.clone()).map_err(|e| TensorError::Create {
            context: "l2_distance_gpu A".into(),
            reason: e.to_string(),
        })?;
    let b_t =
        Tensor::from_data(&b_f32, vec![n], device.clone()).map_err(|e| TensorError::Create {
            context: "l2_distance_gpu B".into(),
            reason: e.to_string(),
        })?;

    let diff = a_t.sub(&b_t).map_err(|e| TensorError::Operation {
        op: "sub",
        reason: e.to_string(),
    })?;

    let norm = diff.norm().map_err(|e| TensorError::Operation {
        op: "norm",
        reason: e.to_string(),
    })?;

    let result = norm.to_vec().map_err(|e| TensorError::Readback {
        context: "l2_distance_gpu".into(),
        reason: e.to_string(),
    })?;

    Ok(f64::from(result[0]))
}

/// GPU mean reduction over a vector.
///
/// Replaces various `.iter().sum::<f64>() / n as f64` patterns.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn mean_gpu(data: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, TensorError> {
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let n = data_f32.len();

    let t =
        Tensor::from_data(&data_f32, vec![n], device.clone()).map_err(|e| TensorError::Create {
            context: "mean_gpu upload".into(),
            reason: e.to_string(),
        })?;

    let m = t.mean().map_err(|e| TensorError::Operation {
        op: "mean",
        reason: e.to_string(),
    })?;

    let result = m.to_vec().map_err(|e| TensorError::Readback {
        context: "mean_gpu".into(),
        reason: e.to_string(),
    })?;

    Ok(f64::from(result[0]))
}

/// GPU sum reduction over a vector.
///
/// Replaces `.iter().sum()` patterns across modules.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn sum_gpu(data: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, TensorError> {
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let n = data_f32.len();

    let t =
        Tensor::from_data(&data_f32, vec![n], device.clone()).map_err(|e| TensorError::Create {
            context: "sum_gpu upload".into(),
            reason: e.to_string(),
        })?;

    let s = t.sum().map_err(|e| TensorError::Operation {
        op: "sum",
        reason: e.to_string(),
    })?;

    let result = s.to_vec().map_err(|e| TensorError::Readback {
        context: "sum_gpu".into(),
        reason: e.to_string(),
    })?;

    Ok(f64::from(result[0]))
}

/// GPU max reduction over a vector.
///
/// Replaces `.fold(f64::NEG_INFINITY, f64::max)` patterns.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn max_gpu(data: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, TensorError> {
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let n = data_f32.len();

    let t =
        Tensor::from_data(&data_f32, vec![n], device.clone()).map_err(|e| TensorError::Create {
            context: "max_gpu upload".into(),
            reason: e.to_string(),
        })?;

    let m = t.max().map_err(|e| TensorError::Operation {
        op: "max",
        reason: e.to_string(),
    })?;

    let result = m.to_vec().map_err(|e| TensorError::Readback {
        context: "max_gpu".into(),
        reason: e.to_string(),
    })?;

    Ok(f64::from(result[0]))
}

/// Two-layer neural network parameters for GPU forward pass.
pub struct NeuralForwardParams<'a> {
    /// Hidden-layer weight matrix in row-major layout.
    pub weights_hidden: &'a [f64],
    /// Hidden-layer bias vector.
    pub bias_hidden: &'a [f64],
    /// Output-layer weight matrix in row-major layout.
    pub weights_output: &'a [f64],
    /// Output-layer bias vector.
    pub bias_output: &'a [f64],
    /// Input activation vector passed into the network.
    pub input: &'a [f64],
    /// Number of hidden units.
    pub hidden_size: usize,
    /// Number of output units.
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
) -> Result<Vec<f64>, TensorError> {
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

    let input_t = Tensor::from_data(&inp, vec![1, input_size], device.clone()).map_err(|e| {
        TensorError::Create {
            context: "nn_forward input".into(),
            reason: e.to_string(),
        }
    })?;

    let wh_t =
        Tensor::from_data(&w_h, vec![*hidden_size, input_size], device.clone()).map_err(|e| {
            TensorError::Create {
                context: "nn_forward W_h".into(),
                reason: e.to_string(),
            }
        })?;
    let bh_t = Tensor::from_data(&b_h, vec![1, *hidden_size], device.clone()).map_err(|e| {
        TensorError::Create {
            context: "nn_forward b_h".into(),
            reason: e.to_string(),
        }
    })?;

    let wo_t =
        Tensor::from_data(&w_o, vec![*output_size, *hidden_size], device.clone()).map_err(|e| {
            TensorError::Create {
                context: "nn_forward W_o".into(),
                reason: e.to_string(),
            }
        })?;
    let bo_t = Tensor::from_data(&b_o, vec![1, *output_size], device.clone()).map_err(|e| {
        TensorError::Create {
            context: "nn_forward b_o".into(),
            reason: e.to_string(),
        }
    })?;

    let wh_transposed = wh_t.transpose().map_err(|e| TensorError::Operation {
        op: "transpose",
        reason: e.to_string(),
    })?;
    let hidden_pre = input_t
        .matmul(&wh_transposed)
        .map_err(|e| TensorError::Operation {
            op: "matmul",
            reason: e.to_string(),
        })?;
    let hidden_biased = hidden_pre.add(&bh_t).map_err(|e| TensorError::Operation {
        op: "add",
        reason: e.to_string(),
    })?;
    let hidden = hidden_biased
        .sigmoid()
        .map_err(|e| TensorError::Operation {
            op: "sigmoid",
            reason: e.to_string(),
        })?;

    let wo_transposed = wo_t.transpose().map_err(|e| TensorError::Operation {
        op: "transpose",
        reason: e.to_string(),
    })?;
    let output_pre = hidden
        .matmul(&wo_transposed)
        .map_err(|e| TensorError::Operation {
            op: "matmul",
            reason: e.to_string(),
        })?;
    let output_biased = output_pre.add(&bo_t).map_err(|e| TensorError::Operation {
        op: "add",
        reason: e.to_string(),
    })?;
    let output = output_biased
        .sigmoid()
        .map_err(|e| TensorError::Operation {
            op: "sigmoid",
            reason: e.to_string(),
        })?;

    let result = output.to_vec().map_err(|e| TensorError::Readback {
        context: "neural_forward_gpu".into(),
        reason: e.to_string(),
    })?;

    Ok(result.into_iter().map(f64::from).collect())
}

/// GPU Shannon entropy: `-sum(p * ln(p))`.
///
/// Delegates to `barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64`
/// (f64 precision, fused map-reduce WGSL shader). Origin: wetSpring
/// bio shaders → hotSpring precision infrastructure → `BarraCUDA`.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn shannon_entropy_gpu(
    probabilities: &[f64],
    device: &Arc<WgpuDevice>,
) -> Result<f64, TensorError> {
    use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

    let op = FusedMapReduceF64::new(device.clone()).map_err(|e| TensorError::Create {
        context: "entropy_gpu init".into(),
        reason: e.to_string(),
    })?;
    op.shannon_entropy(probabilities)
        .map_err(|e| TensorError::Operation {
            op: "shannon_entropy",
            reason: e.to_string(),
        })
}

/// GPU population variance (divides by N) via fused Welford's algorithm.
///
/// Delegates to `barracuda::ops::variance_f64_wgsl::VarianceF64`
/// (f64 precision, fused mean+variance Welford WGSL shader — single
/// GPU dispatch, no intermediate readback). Origin: hotSpring
/// precision infrastructure → `BarraCUDA` S93 fused shader evolution.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn variance_gpu(data: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, TensorError> {
    use barracuda::ops::variance_f64_wgsl::VarianceF64;

    let op = VarianceF64::new(device.clone()).map_err(|e| TensorError::Create {
        context: "variance_gpu init".into(),
        reason: e.to_string(),
    })?;
    op.variance(data).map_err(|e| TensorError::Operation {
        op: "variance",
        reason: e.to_string(),
    })
}

/// GPU fused mean+variance in a single dispatch (Welford's algorithm).
///
/// Returns `[mean, variance]` from one kernel launch — no intermediate
/// readback between mean and deviation passes. Uses `ddof=0` (population).
///
/// Cross-spring: hotSpring Welford fused shader → `BarraCUDA` v0.3.5
/// `VarianceF64::mean_variance()`.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn mean_variance_gpu(data: &[f64], device: &Arc<WgpuDevice>) -> Result<[f64; 2], TensorError> {
    use barracuda::ops::variance_f64_wgsl::VarianceF64;

    let op = VarianceF64::new(device.clone()).map_err(|e| TensorError::Create {
        context: "mean_variance_gpu init".into(),
        reason: e.to_string(),
    })?;
    op.mean_variance(data, 0)
        .map_err(|e| TensorError::Operation {
            op: "mean_variance",
            reason: e.to_string(),
        })
}

/// GPU Pearson correlation between two vectors.
///
/// Delegates to `barracuda::ops::correlation_f64_wgsl::CorrelationF64`
/// (f64 precision, fused single-pass WGSL shader). Origin: wetSpring
/// bio shaders → hotSpring precision infrastructure → `BarraCUDA`.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn pearson_correlation_gpu(
    x: &[f64],
    y: &[f64],
    device: &Arc<WgpuDevice>,
) -> Result<f64, TensorError> {
    use barracuda::ops::correlation_f64_wgsl::CorrelationF64;

    let op = CorrelationF64::new(device.clone()).map_err(|e| TensorError::Create {
        context: "pearson_gpu init".into(),
        reason: e.to_string(),
    })?;
    op.correlation(x, y).map_err(|e| TensorError::Operation {
        op: "correlation",
        reason: e.to_string(),
    })
}

/// GPU full correlation statistics in a single fused dispatch.
///
/// Returns means, variances, and Pearson r — all from one kernel launch.
/// Cross-spring: wetSpring bio shaders (diversity correlation) → hotSpring
/// precision infrastructure (f64 compilation) → `BarraCUDA` v0.3.5
/// `CorrelationResult` fused shader.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn correlation_full_gpu(
    x: &[f64],
    y: &[f64],
    device: &Arc<WgpuDevice>,
) -> Result<barracuda::ops::correlation_f64_wgsl::CorrelationResult, TensorError> {
    use barracuda::ops::correlation_f64_wgsl::CorrelationF64;

    let op = CorrelationF64::new(device.clone()).map_err(|e| TensorError::Create {
        context: "correlation_full init".into(),
        reason: e.to_string(),
    })?;
    op.correlation_full(x, y)
        .map_err(|e| TensorError::Operation {
            op: "correlation_full",
            reason: e.to_string(),
        })
}

/// GPU correlation matrix (n×p data → p×p Pearson correlation matrix).
///
/// Single-dispatch WGSL shader. Data is row-major `[n_samples, n_features]`.
/// Cross-spring: airSpring sensor correlation + groundSpring stats →
/// `BarraCUDA` `matrix_correlation_f64.wgsl`.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn correlation_matrix_gpu(
    data: &[f64],
    n_samples: u32,
    n_features: u32,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, TensorError> {
    use barracuda::ops::stats_f64::matrix_correlation;

    matrix_correlation(device, data, n_samples, n_features).map_err(|e| TensorError::Operation {
        op: "matrix_correlation",
        reason: e.to_string(),
    })
}

/// GPU chi-squared statistic: sum((observed - expected)^2 / expected).
///
/// Delegates to `barracuda::ops::fused_chi_squared_f64::FusedChiSquaredGpu`
/// (f64 precision, single-dispatch WGSL shader). Origin: neuralSpring
/// `chi_squared_f64.wgsl` → `BarraCUDA` (via `ToadStool` S76) → `FusedChiSquaredGpu`.
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
) -> Result<f64, TensorError> {
    use barracuda::ops::fused_chi_squared_f64::FusedChiSquaredGpu;

    FusedChiSquaredGpu::execute(device.clone(), observed, expected)
        .map(|r| r.statistic)
        .map_err(|e| TensorError::Operation {
            op: "chi_squared",
            reason: e.to_string(),
        })
}

/// GPU KL divergence: sum(p * ln(p/q)).
///
/// Delegates to `barracuda::ops::fused_kl_divergence_f64::FusedKlDivergenceGpu`
/// (f64 precision, single-dispatch WGSL shader). Origin: neuralSpring
/// `kl_divergence_f64.wgsl` → `BarraCUDA` (via `ToadStool` S76) → `FusedKlDivergenceGpu`.
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
pub fn kl_divergence_gpu(
    p: &[f64],
    q: &[f64],
    device: &Arc<WgpuDevice>,
) -> Result<f64, TensorError> {
    use barracuda::ops::fused_kl_divergence_f64::FusedKlDivergenceGpu;

    let p_sum: f64 = p.iter().sum();
    let q_sum: f64 = q.iter().sum();
    let p_norm: Vec<f64> = p.iter().map(|&x| x / p_sum).collect();
    let q_norm: Vec<f64> = q.iter().map(|&x| x / q_sum).collect();

    FusedKlDivergenceGpu::execute(device.clone(), &p_norm, &q_norm).map_err(|e| {
        TensorError::Operation {
            op: "kl_divergence",
            reason: e.to_string(),
        }
    })
}
