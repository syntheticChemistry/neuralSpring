// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-accelerated science operations for pure GPU execution.
//!
//! Each function provides a GPU path for operations that lib modules
//! implement on CPU. The CPU implementations remain as validation
//! references; these GPU variants are the production execution path.
//!
//! ## Design
//!
//! - All functions take an `Arc<WgpuDevice>` — no global state
//! - f32 GPU execution (matches Tensor API), f64 CPU references
//! - Errors propagated via `Result`, never panics in production
//! - Capability-based: callers check `GpuCapabilities` before dispatch
//!
//! ## Naming
//!
//! Each function mirrors its CPU counterpart with a `_gpu` suffix:
//! `mat_mul` → `mat_mul_gpu`, `frobenius_norm` → `frobenius_norm_gpu`, etc.

mod activation;
mod bio;
mod eigensolver;
mod linalg;
mod ode_batch;
mod population;
mod reduction;

pub use activation::*;
pub use bio::*;
pub use eigensolver::*;
pub use linalg::*;
pub use ode_batch::*;
pub use population::*;
pub use reduction::*;

#[cfg(test)]
mod tests_ops;
