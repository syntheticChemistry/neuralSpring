// SPDX-License-Identifier: AGPL-3.0-or-later

//! Locally evolved Multi-Head Attention.
//!
//! ## Status (`ToadStool` `9abd6857`)
//!
//! `ToadStool` S46 (`fe573095`) fixed the z-dispatch bug. Native MHA works
//! for non-projection paths. However, full projection shaders still hang
//! on RTX 4070 / Vulkan. Retirement blocked on upstream projection shader
//! stability.
//!
//! ## What this module does
//!
//! Composes correct barracuda primitives as a workaround:
//! - `matmul` for Q/K/V and output projections (verified correct)
//! - `attention` for scaled dot-product attention (dispatch is correct)
//! - CPU-side head-split and concat (unavoidable until barracuda adds
//!   a correct `transpose` / `permute` op)
//!
//! ## Retirement criteria
//!
//! This module can be retired when upstream `barracuda::ops::mha`
//! projection shaders work on RTX 4070 + Vulkan at production sizes
//! (B=4, S=128, H=8, d=512). See `validate_mha_gpu` for the test suite.

use barracuda::device::WgpuDevice;
use barracuda::error::BarracudaError;
use barracuda::tensor::Tensor;
use std::sync::Arc;

type Dev = Arc<WgpuDevice>;

/// Manual multi-head attention: Q/K/V projections + SDPA + output projection.
///
/// All inputs are 2D: `input` is `[seq, d_model]`, weights are `[d_model, d_model]`.
/// Returns `[seq, d_model]`.
///
/// # Errors
///
/// Returns [`BarracudaError`] on shape mismatch or GPU failure.
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention_2d(
    input: &Tensor,
    w_q: &Tensor,
    w_k: &Tensor,
    w_v: &Tensor,
    w_o: &Tensor,
    n_heads: usize,
    device: &Dev,
) -> Result<Tensor, BarracudaError> {
    let seq = input.shape()[0];
    let d_model = input.shape()[1];
    let d_head = d_model / n_heads;

    let q_flat = input.clone().matmul(w_q)?; // [seq, d_model]
    let k_flat = input.clone().matmul(w_k)?;
    let v_flat = input.clone().matmul(w_v)?;

    let q_4d = head_split(&q_flat, seq, n_heads, d_head, device)?;
    let k_4d = head_split(&k_flat, seq, n_heads, d_head, device)?;
    let v_4d = head_split(&v_flat, seq, n_heads, d_head, device)?;

    // barracuda's attention: [B, H, S, D/H] → [B, H, S, D/H]
    let attn_4d = q_4d.attention(&k_4d, &v_4d)?;

    let concat = head_concat(&attn_4d, seq, n_heads, d_head, device)?; // [seq, d_model]

    concat.matmul(w_o)
}

/// Reorder `[seq, d_model]` → `[1, heads, seq, d_head]` via CPU.
fn head_split(
    flat: &Tensor,
    seq: usize,
    heads: usize,
    d_head: usize,
    device: &Dev,
) -> Result<Tensor, BarracudaError> {
    let data = flat.to_vec()?;
    let d_model = heads * d_head;
    let mut out = vec![0.0_f32; seq * d_model];

    // src layout: [seq, d_model] where d_model = heads * d_head
    // dst layout: [1, heads, seq, d_head]
    for s in 0..seq {
        for h in 0..heads {
            for d in 0..d_head {
                let src_idx = s * d_model + h * d_head + d;
                let dst_idx = h * seq * d_head + s * d_head + d;
                out[dst_idx] = data[src_idx];
            }
        }
    }

    Tensor::from_data(&out, vec![1, heads, seq, d_head], device.clone())
}

/// Reorder `[1, heads, seq, d_head]` → `[seq, d_model]` via CPU.
fn head_concat(
    attn: &Tensor,
    seq: usize,
    heads: usize,
    d_head: usize,
    device: &Dev,
) -> Result<Tensor, BarracudaError> {
    let data = attn.to_vec()?;
    let d_model = heads * d_head;
    let mut out = vec![0.0_f32; seq * d_model];

    for s in 0..seq {
        for h in 0..heads {
            for d in 0..d_head {
                let src_idx = h * seq * d_head + s * d_head + d;
                let dst_idx = s * d_model + h * d_head + d;
                out[dst_idx] = data[src_idx];
            }
        }
    }

    Tensor::from_data(&out, vec![seq, d_model], device.clone())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    fn test_device() -> Option<Dev> {
        let rt = tokio::runtime::Runtime::new().ok()?;
        rt.block_on(async { WgpuDevice::new().await.ok().map(|d| Arc::new(d) as Dev) })
    }

    #[test]
    fn head_split_concat_roundtrip() {
        let Some(device) = test_device() else { return };

        let seq = 4;
        let heads = 2;
        let d_head = 3;
        let d_model = heads * d_head;
        #[allow(clippy::cast_precision_loss)]
        let data: Vec<f32> = (0..seq * d_model).map(|i| i as f32).collect();

        let flat = Tensor::from_data(&data, vec![seq, d_model], device.clone()).expect("from_data");
        let split = head_split(&flat, seq, heads, d_head, &device).expect("head_split");
        assert_eq!(split.shape(), &[1, heads, seq, d_head]);

        let reconstructed = head_concat(&split, seq, heads, d_head, &device).expect("head_concat");
        assert_eq!(reconstructed.shape(), &[seq, d_model]);

        let out = reconstructed.to_vec().expect("to_vec");
        for (i, (&got, &want)) in out.iter().zip(data.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "roundtrip mismatch at {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn head_split_layout_is_correct() {
        let Some(device) = test_device() else { return };

        let seq = 2;
        let heads = 2;
        let d_head = 2;
        let d_model = heads * d_head;
        // [seq=2, d_model=4]: row0=[0,1,2,3], row1=[4,5,6,7]
        #[allow(clippy::cast_precision_loss)]
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();

        let flat = Tensor::from_data(&data, vec![seq, d_model], device.clone()).expect("from_data");
        let split = head_split(&flat, seq, heads, d_head, &device).expect("head_split");
        let out = split.to_vec().expect("to_vec");

        // head 0: seq0=[0,1], seq1=[4,5] → [0,1,4,5]
        // head 1: seq0=[2,3], seq1=[6,7] → [2,3,6,7]
        let expected = [0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0];
        for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "split layout at {i}: got {got}, want {want}"
            );
        }
    }
}
