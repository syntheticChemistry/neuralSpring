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

pub mod metrics;
pub mod provenance;
pub mod sequence;
pub mod surrogate;
pub mod tolerances;
pub mod transformer;
pub mod validation;
