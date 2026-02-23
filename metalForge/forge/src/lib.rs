// SPDX-License-Identifier: AGPL-3.0-or-later

//! neuralSpring forge — ML dispatch, shader catalog, and `BarraCUDA` bridge.
//!
//! This crate packages neuralSpring's locally evolved WGSL shaders and dispatch
//! logic in an absorption-friendly layout for `ToadStool`/`BarraCUDA`. Following
//! the hotSpring `metalForge/forge` pattern:
//!
//! - **`shaders`**: All 16 WGSL shader sources as `pub const` (single source of truth)
//! - **`bindings`**: Binding layout structs and dispatch geometry for each shader
//! - **`dispatch`**: ML workload routing (GPU vs CPU crossover logic)
//! - **`bridge`**: neuralSpring `Gpu` ↔ `barracuda::device::WgpuDevice` bridge
//!
//! ## Absorption pattern
//!
//! `ToadStool` can absorb shaders by:
//! 1. Copying the WGSL source from [`shaders`]
//! 2. Copying the binding layout from [`bindings`]
//! 3. Creating a `barracuda::ops::*` wrapper using the dispatch geometry
//! 4. neuralSpring switches to upstream, removes local shader
//!
//! ## Lifecycle
//!
//! ```text
//! evolve → validate → export (this crate) → handoff → ToadStool absorbs → retire
//! ```

pub mod bindings;
pub mod bridge;
pub mod dispatch;
pub mod mixed;
pub mod pcie_bridge;
pub mod shaders;
