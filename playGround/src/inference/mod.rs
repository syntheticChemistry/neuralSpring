// SPDX-License-Identifier: AGPL-3.0-or-later

//! Model inference via barraCuda's WGSL shader pipeline.
//!
//! Loads `HuggingFace` safetensors weights into GPU tensors and runs
//! transformer forward passes through barraCuda's `TensorSession`.

pub mod transformer;
pub mod weights;
