// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! # neural-spring
//!
//! Validation harness proving `BarraCUDA` Rust/WGSL primitives reproduce
//! Python ML baselines.  This crate provides the library code that
//! validation binaries (`validate_*`) link against.
//!
//! ## Architecture
//!
//! Imitates the hotSpring pattern so each Spring evolves independently
//! and the `ToadStool`/`BarraCUDA` team can absorb changes asynchronously:
//!
//! - **`validation`** — [`ValidationHarness`](validation::ValidationHarness) with exit 0/1
//! - **`tolerances`** — centralized constants, no ad-hoc magic numbers
//! - **`provenance`** — Python baseline metadata with full trace
//!
//! ## Evolution Path
//!
//! ```text
//! Python baseline (control/) → Rust validation (src/) → GPU (WGSL) → sovereign pipeline
//! ```
//!
//! ## Modules
//!
//! | Module | Python Source | `BarraCUDA` Target |
//! |--------|-------------|-----------------|
//! | `surrogate` | `control/surrogate/` | `gemm_f64` + `nn::ReLU` |
//! | `transformer` | `control/transformer/` | `attention.wgsl` + `layer_norm.wgsl` |
//! | `sequence` | `control/sequence/` | `lstm_cell.wgsl` + `gru_cell.wgsl` |
//! | `metrics` | shared across all | `FusedMapReduceF64` |
//! | `counterdiabatic` | `control/counterdiabatic/` | `gemm_f64` + `softmax.wgsl` |
//! | `modes` | `control/modes/` | reduce ops + `elementwise` |
//! | `eco_dynamics` | `control/eco_dynamics/` | batch GEMM + `reduce_sum` |
//! | `directed_evolution` | `control/directed_evolution/` | batch GEMM + `reduce_max` |
//! | `hmm` | `control/hmm_phylo/` | `gemm_f64` chain (forward/Viterbi) |
//! | `game_theory` | `control/game_theory/` | `gemm_f64` + `softmax.wgsl` |
//! | `swarm_robotics` | `control/swarm_robotics/` | batch GEMM + `elementwise` |
//! | `sate_alignment` | `control/sate_alignment/` | `gemm_f64` (distance matrix) |
//! | `introgression` | `control/introgression/` | `gemm_f64` chain (HMM) |
//! | `regulatory_network` | `control/regulatory_network/` | ODE `elementwise` + Hill |
//! | `signal_integration` | `control/signal_integration/` | ODE `elementwise` + Hill |
//! | `spectral_commutativity` | `control/spectral_commutativity/` | `gemm_f64` + eigendecomp |
//! | `anderson_localization` | `control/anderson_localization/` | tridiag solve + eigendecomp |
//! | `pangenome_selection` | `control/pangenome_selection/` | sparse GEMM + chi-squared reduce |
//! | `meta_population` | `control/meta_population/` | variance decomp + `pearson` |
//! | `coral_forge` | `control/coral_forge/` | triangle ops + SDPA + layernorm + IPA + diffusion |
//!
//! ## baseCamp Modules (Biophysical AI Interpretability)
//!
//! | Module | Sub-thesis | Experiments |
//! |--------|-----------|-------------|
//! | `weight_spectral` | nS-01: Weight Matrices as Disordered Hamiltonians | nS-101..106 |
//! | `information_flow` | nS-02: Information Flow as Wave Propagation | nS-201..206 |
//! | `loss_landscape` | nS-03: Loss Landscapes as Energy Landscapes | nS-301..305 |
//! | `neural_pgm` | nS-04: Neural Networks as PGMs | nS-401..406 |
//! | `agent_coordination` | nS-05: Multi-Agent AI as Quorum Sensing | nS-501..505 |
//! | `immunological_anderson` | nS-06: Anderson Localization in Immunological Signaling | nS-601..605 |

#[cfg(test)]
mod determinism_tests;
#[cfg(test)]
mod property_tests;

/// Crate-level GPU test serialization.
///
/// wgpu's Vulkan backend races when multiple tests submit to the same
/// driver concurrently. All GPU-touching tests must hold this lock.
/// Recovers from poisoning so one GPU test panic doesn't cascade.
#[cfg(test)]
pub(crate) mod test_gpu_lock {
    use std::sync::{Mutex, MutexGuard, OnceLock};
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    pub fn acquire() -> MutexGuard<'static, ()> {
        let mtx = LOCK.get_or_init(|| Mutex::new(()));
        mtx.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

pub mod agent_coordination;
pub mod anderson_localization;
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "attention matrix dimension casts and f64 statistics"
)]
pub mod attention_anderson;
pub mod bench;
pub mod config;
pub mod coral_forge;
pub mod counterdiabatic;
pub mod deeponet;
pub mod digester_anderson;
pub mod digestion_prediction;
pub mod directed_evolution;
pub mod eco_dynamics;
pub mod eigh;
pub mod evolved;
pub mod fft;
pub mod game_theory;
pub mod glucose_prediction;
pub mod gpu;
pub mod gpu_dispatch;
pub mod gpu_ops;
pub mod gpu_shader_validation;
pub mod hmm;
pub mod immunological_anderson;
pub mod information_flow;
pub mod introgression;
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "layer/path casts and f64 statistics"
)]
pub mod introgression_nn;
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "matrix dimension casts (n ≤ 512) and f64 statistics"
)]
pub mod isomorphic_reservoir;
pub mod lenet;
pub mod loss_landscape;
pub mod meta_population;
pub mod metrics;
pub mod modes;
pub mod nautilus_bridge;
pub mod neural_pgm;
pub mod nucleus_pipeline;
pub mod pangenome_selection;
pub mod pinn;
pub mod primitives;
pub mod provenance;
pub mod quantized;
pub mod regulatory_network;
pub mod rng;
pub mod sate_alignment;
pub mod search;
pub mod sequence;
pub mod signal_integration;
pub mod spectral_commutativity;
pub mod streaming;
pub mod surrogate;
pub mod swarm_robotics;
pub mod tolerances;
pub mod training_monitor;
pub mod transformer;
pub mod validation;
pub mod visualization;
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "grid dimension casts (n ≤ 1024) and f64 statistics"
)]
pub mod wdm_ensemble_qs;
pub mod wdm_esn;
pub mod wdm_sqw;
pub mod wdm_surrogate;
pub mod wdm_transport;
pub mod weight_loader;
pub mod weight_spectral;

#[cfg(feature = "primal")]
pub mod rpc_service;
