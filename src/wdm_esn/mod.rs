// SPDX-License-Identifier: AGPL-3.0-or-later

//! WDM regime classifier: Echo State Network for plasma phase detection.
//!
//! nW-05: Classifies (ρ, T) conditions into three WDM regimes:
//!   0 = Classical (Γ < 1)
//!   1 = Warm Dense Matter (1 ≤ Γ ≤ 10)
//!   2 = Degenerate (Γ > 10)
//!
//! ## Python Baseline Provenance
//!
//! | Metric | Value | Source |
//! |--------|-------|--------|
//! | Test accuracy | 0.965 | `control/wdm/esn_regime_classifier.py`, seed=42 |
//! | Classical acc | 0.964 | same |
//! | WDM acc | 1.000 | same |
//! | Degenerate acc | 0.939 | same |
//!
//! ## Architecture
//!
//! ESN(input=2, reservoir=64, 2-step recurrence) → linear readout → 3 classes.
//!
//! ## Evolution (Session 105)
//!
//! - `EsnClassifier` / `classify()`: CPU reference path (backward compat)
//! - `classify_via_barracuda()`: GPU path via `barracuda::tensor::Tensor`
//! - `MultiHeadWdmClassifier`: `barracuda::esn_v2::MultiHeadEsn` with
//!   head disagreement for uncertainty quantification
//! - NPU export via `barracuda::esn_v2::quantize_affine_i8_f64`
//!
//! ## Cross-Spring Provenance
//!
//! ```text
//! Jaeger ESN → Python (scikit-learn) → Rust CPU
//!   → barracuda::esn_v2 (hotSpring absorption)
//!   → MultiHeadEsn (hotSpring 36-head concept, adapted to 3 WDM heads)
//!   → head_disagreement → phase boundary signal
//! ```
//!
//! ## Reference
//!
//! Jaeger, "The echo state approach" (2001)
//! Ichimaru, "Statistical Plasma Physics" (1994)

mod classifier;
mod gpu_path;
mod multi_head;

pub use classifier::{load_esn_from_json, EsnClassifier, EsnNormalization};
pub use gpu_path::classify_via_barracuda;
pub use multi_head::{wdm_head_configs, wdm_heads, MultiHeadResult, MultiHeadWdmClassifier};

pub use barracuda::esn_v2::{
    quantize_affine_i8_f64, ESNConfig, ExportedWeights, HeadConfig, HeadGroup, MultiHeadEsn,
    NpuReadoutWeights,
};

pub(crate) fn argmax_f64(scores: &[f64]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or(0, |(i, _)| i)
}

pub(crate) fn argmax_f32(scores: &[f32]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or(0, |(i, _)| i)
}

#[cfg(test)]
mod tests;
