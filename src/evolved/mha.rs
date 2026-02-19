// SPDX-License-Identifier: AGPL-3.0-only

//! Locally evolved Multi-Head Attention.
//!
//! Barracuda's `multi_head_attention` has a dispatch bug in its WGSL
//! projection shaders: the workgroup size is (16, 16, 1) but the
//! z-dimension dispatch divides by 16 instead of 1, causing only a
//! fraction of sequence positions / output dimensions to be computed.
//!
//! This workaround composes correct barracuda primitives:
//! - `matmul` for Q/K/V and output projections (verified correct)
//! - `attention` for scaled dot-product attention (dispatch is correct)
//! - CPU-side head-split and concat (unavoidable until barracuda adds
//!   a correct `transpose` / `permute` op)
//!
//! **`ToadStool` handoff**: Fix MHA projection dispatch in
//! `barracuda/src/ops/mha/projections.rs` — change z-dimension
//! `div_ceil(16)` to `div_ceil(1)` for both `project_with_head_split`
//! and `concat_and_project`.  Then this module can be retired.

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
