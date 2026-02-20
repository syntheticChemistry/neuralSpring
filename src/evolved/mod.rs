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
//! ## `ToadStool` alignment (reviewed `82f953c8`, Feb 19, 2026)
//!
//! `ToadStool` has evolved significantly (80+ commits since Feb 15): wgpu v22
//! migration, `WORKGROUP_SIZE_1D`/`WORKGROUP_SIZE_2D` constants,
//! `GpuDriverProfile` for data-driven shader specialization, concurrency
//! hardening, and deep-debt compliance.  We now import these primitives.
//!
//! **All 11 neuralSpring handoff items remain pending** in `ToadStool`.  Until
//! they are absorbed, these evolutions cannot be retired:
//!
//! - `Tensor::from_buffer` still `pub(crate)` — blocks #1, #2, #3
//! - Per-op command submission still present — blocks #4, #10
//! - `science_limits()` still 512 MB — blocks #5
//! - `leaky_relu` / `elu` params still mismatched — blocks #6, #7
//! - MHA z-dispatch still buggy — blocks #8
//! - Softmax `arrayLength` on pooled buffers — blocks #9
//! - 4-tier shader router not absorbed — blocks #11

pub mod fused_mlp;
pub mod fused_pipeline;
pub mod fused_transformer;
pub mod layer_norm;
pub mod log_softmax;
pub mod mha;
