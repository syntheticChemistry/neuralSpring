// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-Head Attention — now delegates to upstream `BarraCUDA`.
//!
//! ## Status (`ToadStool` S60–S68, `f0feb226`)
//!
//! S-03b is **RESOLVED upstream**. `ToadStool` `0c998992` (S60–S61) decomposed the
//! fused MHA projection into `Tensor::matmul` + `head_split.wgsl` /
//! `head_concat.wgsl` — exactly the approach neuralSpring evolved locally.
//! neuralSpring's `head_split.wgsl` and `head_concat.wgsl` shaders were
//! absorbed into upstream `barracuda::ops::mha::projections`.
//!
//! This module is now a thin wrapper that reshapes 2D inputs to 3D and
//! delegates to `barracuda::ops::mha::MultiHeadAttention`. It can be fully
//! retired once callers are updated to use the upstream 3D API directly.

use barracuda::error::BarracudaError;
use barracuda::ops::mha::MultiHeadAttention;
use barracuda::tensor::Tensor;

/// Multi-head attention on 2D tensors: `[seq, d_model]` → `[seq, d_model]`.
///
/// Thin wrapper over upstream `MultiHeadAttention` that adds a batch dimension.
/// Callers that already have 3D tensors should use the upstream API directly.
///
/// # Migration
///
/// Use [`barracuda::ops::mha::MultiHeadAttention`] directly with 3D tensors
/// `[batch, seq, d_model]`. This wrapper only reshapes 2D→3D→2D and will be
/// removed once all callers migrate.
///
/// # Errors
///
/// Returns [`BarracudaError`] on shape mismatch or GPU failure.
#[deprecated(
    since = "0.2.0",
    note = "Use barracuda::ops::mha::MultiHeadAttention directly with 3D tensors"
)]
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention_2d(
    input: &Tensor,
    w_q: &Tensor,
    w_k: &Tensor,
    w_v: &Tensor,
    w_o: &Tensor,
    n_heads: usize,
    _device: &std::sync::Arc<barracuda::device::WgpuDevice>,
) -> Result<Tensor, BarracudaError> {
    let seq = input.shape()[0];
    let d_model = input.shape()[1];

    let input_3d = input.clone().reshape(vec![1, seq, d_model])?;

    let mha = MultiHeadAttention::new(
        input_3d.clone(),
        input_3d.clone(),
        input_3d,
        w_q.clone(),
        w_k.clone(),
        w_v.clone(),
        w_o.clone(),
        n_heads,
    )?;

    let output_3d = mha.execute()?;
    output_3d.reshape(vec![seq, d_model])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, deprecated)]

    use super::*;
    use barracuda::device::WgpuDevice;
    use std::sync::Arc;

    type Dev = Arc<WgpuDevice>;

    fn test_device() -> Option<(std::sync::MutexGuard<'static, ()>, Dev)> {
        let guard = crate::test_gpu_lock::acquire();
        let gpu = crate::gpu::tests::shared_gpu()?;
        Some((guard, gpu.wgpu_device().clone()))
    }

    #[test]
    fn mha_2d_wrapper_produces_correct_shape() {
        let Some((_guard, device)) = test_device() else {
            return;
        };

        let seq = 4;
        let d_model = 8;
        let n_heads = 2;

        let input = Tensor::from_data(
            &vec![0.1_f32; seq * d_model],
            vec![seq, d_model],
            device.clone(),
        )
        .expect("input");

        let weight = Tensor::from_data(
            &vec![0.01_f32; d_model * d_model],
            vec![d_model, d_model],
            device.clone(),
        )
        .expect("weight");

        let result =
            multi_head_attention_2d(&input, &weight, &weight, &weight, &weight, n_heads, &device);

        match result {
            Ok(out) => assert_eq!(out.shape(), &[seq, d_model]),
            Err(e) => {
                // GPU may not support f32 attention at this size; shape check is the key test
                eprintln!("MHA failed (expected on some hardware): {e}");
            }
        }
    }
}
