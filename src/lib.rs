// SPDX-License-Identifier: AGPL-3.0-or-later

//! # neural-spring
//!
//! Validation harness proving `BarraCUDA` Rust/WGSL primitives reproduce
//! Python ML baselines.  This crate provides the library code that
//! validation binaries (`validate_*`) link against.
//!
//! ## Architecture
//!
//! Imitates the hotSpring pattern so each Spring evolves independently
//! and the ToadStool/BarraCUDA team can absorb changes asynchronously:
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

#[cfg(test)]
mod determinism_tests;

pub mod anderson_localization;
pub mod counterdiabatic;
pub mod deeponet;
pub mod directed_evolution;
pub mod eco_dynamics;
pub mod eigh;
pub mod evolved;
pub mod fft;
pub mod game_theory;
pub mod gpu;
pub mod hmm;
pub mod introgression;
pub mod meta_population;
pub mod metrics;
pub mod modes;
pub mod pangenome_selection;
pub mod pinn;
pub mod primitives;
pub mod provenance;
pub mod regulatory_network;
pub mod rng;
pub mod sate_alignment;
pub mod sequence;
pub mod signal_integration;
pub mod spectral_commutativity;
pub mod surrogate;
pub mod swarm_robotics;
pub mod tolerances;
pub mod transformer;
pub mod validation;
