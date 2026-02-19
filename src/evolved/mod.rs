// SPDX-License-Identifier: AGPL-3.0-only

//! Locally evolved GPU-resident ops.
//!
//! These replace barracuda ops that break streaming with GPU→CPU→GPU
//! round-trips (via `read_buffer`).  Each evolved op:
//!
//! - Takes raw `wgpu::Buffer` inputs (not `Tensor`)
//! - Dispatches a WGSL compute pass
//! - Returns `wgpu::Buffer` output (stays on GPU)
//! - Only reads back on explicit request
//!
//! Once `ToadStool` upstreams fixes (e.g. making `Tensor::from_buffer` public),
//! these evolutions can be retired.

pub mod fused_mlp;
pub mod fused_pipeline;
pub mod fused_transformer;
pub mod layer_norm;
pub mod log_softmax;
pub mod mha;
