// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-accelerated activations: softmax, Boltzmann, GELU.

#![expect(
    clippy::cast_possible_truncation,
    reason = "GPU activations convert f64→f32 for hardware tensor API"
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use std::sync::Arc;

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
