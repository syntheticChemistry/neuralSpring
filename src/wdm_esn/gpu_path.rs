// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU classification path via `barracuda::tensor::Tensor` ops.

#![expect(
    clippy::cast_possible_truncation,
    reason = "f64→f32 narrowing is intentional for GPU tensor ops"
)]

use super::argmax_f32;
use super::classifier::EsnClassifier;

/// Classify using barracuda Tensor ops on GPU.
///
/// Implements the full ESN 2-step recurrence + readout using barracuda
/// `Tensor` operations (matmul, add, tanh). This routes through `BarraCUDA`
/// WGSL shaders when a GPU is available, falling back to CPU otherwise.
///
/// Returns `(label, raw_scores_f32)` matching [`EsnClassifier::classify`].
///
/// # Errors
///
/// Returns `Err` on GPU/Tensor operation failure.
pub fn classify_via_barracuda(
    classifier: &EsnClassifier,
    log_rho: f64,
    log_t: f64,
    device: &std::sync::Arc<barracuda::device::WgpuDevice>,
) -> Result<(usize, Vec<f32>), String> {
    let rs = classifier.reservoir_size;
    let nc = classifier.n_classes;

    let x0 = ((log_rho - classifier.norm.x_mean[0]) / classifier.norm.x_std[0]) as f32;
    let x1 = ((log_t - classifier.norm.x_mean[1]) / classifier.norm.x_std[1]) as f32;

    let x = barracuda::tensor::Tensor::from_data(&[x0, x1], vec![1, 2], device.clone())
        .map_err(|e| format!("x tensor: {e}"))?;

    let w_in_f32: Vec<f32> = classifier.w_in.iter().map(|&v| v as f32).collect();
    let b_f32: Vec<f32> = classifier.b_res.iter().map(|&v| v as f32).collect();

    let w_in = barracuda::tensor::Tensor::from_data(&w_in_f32, vec![rs, 2], device.clone())
        .map_err(|e| format!("W_in: {e}"))?;
    let w_in_t = w_in.transpose().map_err(|e| format!("W_in^T: {e}"))?;
    let b = barracuda::tensor::Tensor::from_data(&b_f32, vec![1, rs], device.clone())
        .map_err(|e| format!("b_res: {e}"))?;

    let z1 = x
        .matmul_ref(&w_in_t)
        .map_err(|e| format!("step1 matmul: {e}"))?;
    let z1b = z1.add(&b).map_err(|e| format!("step1 add: {e}"))?;
    let h1 = z1b.tanh().map_err(|e| format!("step1 tanh: {e}"))?;

    let w_res_f32: Vec<f32> = classifier.w_res.iter().map(|&v| v as f32).collect();
    let w_res = barracuda::tensor::Tensor::from_data(&w_res_f32, vec![rs, rs], device.clone())
        .map_err(|e| format!("W_res: {e}"))?;
    let w_res_t = w_res.transpose().map_err(|e| format!("W_res^T: {e}"))?;

    let input_proj = x.matmul(&w_in_t).map_err(|e| format!("step2 input: {e}"))?;
    let res_proj = h1.matmul(&w_res_t).map_err(|e| format!("step2 res: {e}"))?;
    let z2 = input_proj
        .add(&res_proj)
        .map_err(|e| format!("step2 add: {e}"))?;
    let z2b = z2.add(&b).map_err(|e| format!("step2 bias: {e}"))?;
    let h2 = z2b.tanh().map_err(|e| format!("step2 tanh: {e}"))?;

    let w_out_f32: Vec<f32> = classifier.w_out.iter().map(|&v| v as f32).collect();
    let b_out_f32: Vec<f32> = classifier.b_out.iter().map(|&v| v as f32).collect();

    let w_out = barracuda::tensor::Tensor::from_data(&w_out_f32, vec![rs, nc], device.clone())
        .map_err(|e| format!("W_out: {e}"))?;
    let b_out = barracuda::tensor::Tensor::from_data(&b_out_f32, vec![1, nc], device.clone())
        .map_err(|e| format!("b_out: {e}"))?;

    let scores_raw = h2
        .matmul(&w_out)
        .map_err(|e| format!("readout matmul: {e}"))?;
    let scores = scores_raw
        .add(&b_out)
        .map_err(|e| format!("readout add: {e}"))?;

    let scores_vec = scores.to_vec().map_err(|e| format!("readback: {e}"))?;
    let label = argmax_f32(&scores_vec);

    Ok((label, scores_vec))
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test assertions use expect for clear messages"
)]
mod tests {
    use super::*;
    use crate::wdm_esn::classifier::{EsnClassifier, EsnNormalization};
    use approx::assert_relative_eq;
    use serial_test::serial;
    use std::sync::Arc;

    fn tiny_esn() -> EsnClassifier {
        let rs = 4;
        let nc = 3;
        EsnClassifier {
            w_in: vec![0.1; rs * 2],
            w_res: vec![0.01; rs * rs],
            b_res: vec![0.0; rs],
            w_out: vec![0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            b_out: vec![0.0; nc],
            reservoir_size: rs,
            n_classes: nc,
            norm: EsnNormalization {
                x_mean: [0.5, 6.0],
                x_std: [1.0, 1.5],
            },
        }
    }

    /// CPU-relaxed `WgpuDevice` (Tensor fallback). Keep in a separate `#[serial]` test from
    /// `multi_head` GPU work — mixing both device lifetimes in one test can trip wgpu bind groups.
    #[serial]
    #[tokio::test]
    async fn classify_via_barracuda_matches_cpu_reference() {
        let tensor_dev = Arc::new(
            barracuda::device::WgpuDevice::new_cpu_relaxed()
                .await
                .expect("CPU-relaxed WgpuDevice"),
        );
        let esn = tiny_esn();
        for &(lr, lt) in &[(0.5_f64, 5.5_f64), (-0.25, 7.0)] {
            let (label_cpu, scores_cpu) = esn.classify(lr, lt);
            let (label_gpu, scores_gpu) =
                classify_via_barracuda(&esn, lr, lt, &tensor_dev).expect("Tensor classify");
            assert_eq!(label_cpu, label_gpu);
            assert_eq!(scores_cpu.len(), scores_gpu.len());
            for (a, b) in scores_cpu.iter().zip(scores_gpu.iter()) {
                assert_relative_eq!(*a as f32, *b, epsilon = 1e-3, max_relative = 2e-3);
            }
        }
    }
}
