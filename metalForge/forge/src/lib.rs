// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! neuralSpring forge — ML dispatch, substrate discovery, shader catalog,
//! and `BarraCUDA` bridge.
//!
//! This crate packages neuralSpring's locally evolved implementations and
//! dispatch logic in an absorption-friendly layout for `ToadStool`/`BarraCUDA`.
//! Following the hotSpring/wetSpring `metalForge/forge` pattern:
//!
//! - **`substrate`**: Runtime compute device abstraction (GPU, CPU)
//! - **`probe`**: Hardware discovery via wgpu (GPU) and procfs (CPU)
//! - **`inventory`**: Assemble all probed substrates
//! - **`workloads`**: ML workloads with `ShaderOrigin` tracking (Absorbed/Local/`CpuOnly`)
//! - **`dispatch`**: ML workload routing (GPU vs CPU crossover logic)
//! - **`bridge`**: neuralSpring `Gpu` ↔ `barracuda::device::WgpuDevice` bridge
//! - **`shaders`**: WGSL shader sources as `pub const`
//! - **`bindings`**: Binding layout structs for each shader
//! - **`mixed`**: Mixed-substrate transfer cost model
//! - **`pcie_bridge`**: `PCIe` P2P detection and transfer strategy
//! - **`graph`**: biomeOS pipeline DAG — topological execution of capability-addressed stages
//!
//! ## Write → Absorb → Lean
//!
//! 1. **Write**: Evolve local implementations in this crate
//! 2. **Absorb**: `ToadStool` absorbs mature, validated code
//! 3. **Lean**: neuralSpring switches to upstream, retires local copy
//!
//! ## Absorption tracking
//!
//! [`workloads::origin_summary`] counts absorbed (20), local (6), and
//! CPU-only (2) workloads. As `ToadStool` absorbs more, local count drops.

pub mod bindings;
pub mod bridge;
pub mod coralreef_bridge;
pub mod dispatch;
pub mod graph;
pub mod inventory;
pub mod mixed;
pub mod pcie_bridge;
pub mod pipeline;
pub mod probe;
pub mod shaders;
pub mod substrate;
pub mod workloads;
