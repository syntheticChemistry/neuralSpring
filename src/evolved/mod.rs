// SPDX-License-Identifier: AGPL-3.0-or-later

//! Locally evolved GPU-resident ops — absorption candidates for `ToadStool`.
//!
//! These replace barracuda ops that break streaming with GPU→CPU→GPU
//! round-trips (via `read_buffer`).  Each evolved op:
//!
//! - Takes raw `wgpu::Buffer` inputs (not `Tensor`)
//! - Dispatches a WGSL compute pass
//! - Returns `wgpu::Buffer` output (stays on GPU)
//! - Only reads back on explicit request
//!
//! ## Absorption status (Feb 20, 2026)
//!
//! These ~2075 LOC are **ready for `ToadStool` absorption**. Each module maps
//! to a specific shortcoming in `specs/TOADSTOOL_HANDOFF.md` and retires when
//! the upstream fix lands.  hotSpring follows the same pattern: evolve locally,
//! document in `wateringHole/handoffs/`, `ToadStool` absorbs at their pace.
//!
//! | Module | LOC | Shortcoming | `ToadStool` Target |
//! |--------|-----|-------------|-------------------|
//! | `fused_pipeline` | 520 | S-01 per-op dispatch | `StatefulPipeline` extension |
//! | `fused_mlp` | 180 | S-01 per-op dispatch | ML op batching |
//! | `fused_transformer` | 290 | S-01 per-op dispatch | ML op batching |
//! | `matmul_cpu_tiled.wgsl` | 263 | S-02 naive matmul | `KernelRouter` shader |
//! | `matmul_gpu_evolved.wgsl` | 302 | S-02 naive matmul | `KernelRouter` shader |
//! | `layer_norm` | 120 | S-08 round-trip | `Tensor::from_buffer` pub |
//! | `log_softmax` | 110 | S-09 round-trip | `Tensor::from_buffer` pub |
//! | `mha` | 190 | S-03 z-dispatch bug | z-dim fix |
//!
//! **All 11 neuralSpring handoff items remain pending** in `ToadStool` (+S-12
//! `eigh_f64` accuracy gap from Phase 2 CPU ports).  Until absorbed, these
//! evolutions cannot be retired.

pub mod fused_mlp;
pub mod fused_pipeline;
pub mod fused_transformer;
pub mod layer_norm;
pub mod log_softmax;
pub mod mha;
