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

#![allow(clippy::cast_possible_truncation, clippy::doc_markdown)]

use serde::Deserialize;

/// Input normalization for (log_rho, log_T).
#[derive(Debug, Clone)]
pub struct EsnNormalization {
    pub x_mean: [f64; 2],
    pub x_std: [f64; 2],
}

/// Trained ESN regime classifier (CPU reference path).
#[derive(Debug, Clone)]
pub struct EsnClassifier {
    /// Input weights `[reservoir_size, 2]` flattened row-major.
    pub w_in: Vec<f64>,
    /// Reservoir weights `[reservoir_size, reservoir_size]` flattened row-major.
    pub w_res: Vec<f64>,
    /// Reservoir bias `[reservoir_size]`.
    pub b_res: Vec<f64>,
    /// Output weights `[reservoir_size, n_classes]` flattened row-major.
    pub w_out: Vec<f64>,
    /// Output bias `[n_classes]`.
    pub b_out: Vec<f64>,
    /// Reservoir size.
    pub reservoir_size: usize,
    /// Number of classes.
    pub n_classes: usize,
    /// Input normalization.
    pub norm: EsnNormalization,
}

impl EsnClassifier {
    /// Classify (log_rho, log_T) → regime label (0=Classical, 1=WDM, 2=Degenerate).
    ///
    /// Returns (predicted_label, raw_scores).
    #[must_use]
    pub fn classify(&self, log_rho: f64, log_t: f64) -> (usize, Vec<f64>) {
        let rs = self.reservoir_size;

        let x0 = (log_rho - self.norm.x_mean[0]) / self.norm.x_std[0];
        let x1 = (log_t - self.norm.x_mean[1]) / self.norm.x_std[1];

        let mut h: Vec<f64> = (0..rs)
            .map(|i| {
                (self.w_in[i * 2].mul_add(x0, self.w_in[i * 2 + 1] * x1) + self.b_res[i]).tanh()
            })
            .collect();

        let h_prev = h.clone();
        for (i, h_val) in h.iter_mut().enumerate() {
            let input_proj = self.w_in[i * 2].mul_add(x0, self.w_in[i * 2 + 1] * x1);
            let res_proj: f64 = h_prev
                .iter()
                .enumerate()
                .map(|(j, hj)| self.w_res[i * rs + j] * hj)
                .sum();
            *h_val = (input_proj + res_proj + self.b_res[i]).tanh();
        }

        let nc = self.n_classes;
        let mut scores = self.b_out.clone();
        for (j, h_val) in h.iter().enumerate() {
            for (s, score) in scores.iter_mut().enumerate() {
                *score = self.w_out[j * nc + s].mul_add(*h_val, *score);
            }
        }

        let label = argmax_f64(&scores);
        (label, scores)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Typed JSON deserialization (replaces manual serde_json::Value parsing)
// ═══════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct EsnWeightsJson {
    normalization: NormJson,
    weights: WeightsJson,
}

#[derive(Deserialize)]
struct NormJson {
    x_mean: [f64; 2],
    x_std: [f64; 2],
}

#[derive(Deserialize)]
struct WeightsJson {
    reservoir_size: usize,
    n_classes: usize,
    #[serde(rename = "W_in")]
    w_in: Vec<f64>,
    #[serde(rename = "W_res")]
    w_res: Vec<f64>,
    b_res: Vec<f64>,
    #[serde(rename = "W_out")]
    w_out: Vec<f64>,
    b_out: Vec<f64>,
}

/// Load an [`EsnClassifier`] from the Python baseline JSON.
///
/// Uses typed deserialization instead of manual `serde_json::Value` traversal.
///
/// # Errors
///
/// Returns `Err` if the JSON structure is unexpected.
pub fn load_esn_from_json(json_str: &str) -> Result<EsnClassifier, String> {
    let parsed: EsnWeightsJson =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

    Ok(EsnClassifier {
        w_in: parsed.weights.w_in,
        w_res: parsed.weights.w_res,
        b_res: parsed.weights.b_res,
        w_out: parsed.weights.w_out,
        b_out: parsed.weights.b_out,
        reservoir_size: parsed.weights.reservoir_size,
        n_classes: parsed.weights.n_classes,
        norm: EsnNormalization {
            x_mean: parsed.normalization.x_mean,
            x_std: parsed.normalization.x_std,
        },
    })
}

// ═══════════════════════════════════════════════════════════════════════
// GPU path via barracuda Tensor ops (retained for backward compat)
// ═══════════════════════════════════════════════════════════════════════

/// Classify using barracuda Tensor ops on GPU.
///
/// Implements the full ESN 2-step recurrence + readout using barracuda
/// `Tensor` operations (matmul, add, tanh). This routes through `ToadStool`
/// WGSL shaders when a GPU is available, falling back to CPU otherwise.
///
/// Returns `(label, raw_scores_f32)` matching [`EsnClassifier::classify`].
///
/// # Errors
///
/// Returns `Err` on GPU/Tensor operation failure.
#[allow(clippy::cast_precision_loss)]
pub fn classify_via_barracuda(
    classifier: &EsnClassifier,
    log_rho: f64,
    log_t: f64,
    device: &std::sync::Arc<barracuda::device::WgpuDevice>,
) -> Result<(usize, Vec<f32>), String> {
    let rs = classifier.reservoir_size;
    let nc = classifier.n_classes;

    let x0 = ((log_rho - classifier.norm.x_mean[0]) / classifier.norm.x_std[0]) as f32;
    let x1 = ((log_t - classifier.norm.x_mean[1]) / classifier.norm.x_std[1]) as f32;

    let x = barracuda::tensor::Tensor::from_data(&[x0, x1], vec![1, 2], device.clone())
        .map_err(|e| format!("x tensor: {e}"))?;

    let w_in_f32: Vec<f32> = classifier.w_in.iter().map(|&v| v as f32).collect();
    let b_f32: Vec<f32> = classifier.b_res.iter().map(|&v| v as f32).collect();

    let w_in = barracuda::tensor::Tensor::from_data(&w_in_f32, vec![rs, 2], device.clone())
        .map_err(|e| format!("W_in: {e}"))?;
    let w_in_t = w_in.transpose().map_err(|e| format!("W_in^T: {e}"))?;
    let b = barracuda::tensor::Tensor::from_data(&b_f32, vec![1, rs], device.clone())
        .map_err(|e| format!("b_res: {e}"))?;

    let z1 = x
        .matmul_ref(&w_in_t)
        .map_err(|e| format!("step1 matmul: {e}"))?;
    let z1b = z1.add(&b).map_err(|e| format!("step1 add: {e}"))?;
    let h1 = z1b.tanh().map_err(|e| format!("step1 tanh: {e}"))?;

    let w_res_f32: Vec<f32> = classifier.w_res.iter().map(|&v| v as f32).collect();
    let w_res = barracuda::tensor::Tensor::from_data(&w_res_f32, vec![rs, rs], device.clone())
        .map_err(|e| format!("W_res: {e}"))?;
    let w_res_t = w_res.transpose().map_err(|e| format!("W_res^T: {e}"))?;

    let input_proj = x.matmul(&w_in_t).map_err(|e| format!("step2 input: {e}"))?;
    let res_proj = h1.matmul(&w_res_t).map_err(|e| format!("step2 res: {e}"))?;
    let z2 = input_proj
        .add(&res_proj)
        .map_err(|e| format!("step2 add: {e}"))?;
    let z2b = z2.add(&b).map_err(|e| format!("step2 bias: {e}"))?;
    let h2 = z2b.tanh().map_err(|e| format!("step2 tanh: {e}"))?;

    let w_out_f32: Vec<f32> = classifier.w_out.iter().map(|&v| v as f32).collect();
    let b_out_f32: Vec<f32> = classifier.b_out.iter().map(|&v| v as f32).collect();

    let w_out = barracuda::tensor::Tensor::from_data(&w_out_f32, vec![rs, nc], device.clone())
        .map_err(|e| format!("W_out: {e}"))?;
    let b_out = barracuda::tensor::Tensor::from_data(&b_out_f32, vec![1, nc], device.clone())
        .map_err(|e| format!("b_out: {e}"))?;

    let scores_raw = h2
        .matmul(&w_out)
        .map_err(|e| format!("readout matmul: {e}"))?;
    let scores = scores_raw
        .add(&b_out)
        .map_err(|e| format!("readout add: {e}"))?;

    let scores_vec = scores.to_vec().map_err(|e| format!("readback: {e}"))?;
    let label = argmax_f32(&scores_vec);

    Ok((label, scores_vec))
}

// ═══════════════════════════════════════════════════════════════════════
// Multi-head ESN via barracuda::esn_v2 (hotSpring cross-spring evolution)
//
// Wraps barracuda's MultiHeadEsn with WDM-specific head configuration
// and head disagreement for uncertainty quantification.
//
// Evolution chain:
//   hotSpring 36-head concept → barracuda::esn_v2::MultiHeadEsn
//   → neuralSpring MultiHeadWdmClassifier (3 WDM-domain heads)
//   → head_disagreement → phase boundary signal
// ═══════════════════════════════════════════════════════════════════════

pub use barracuda::esn_v2::{
    quantize_affine_i8_f64, ESNConfig, ExportedWeights, HeadConfig, HeadGroup, MultiHeadEsn,
    NpuReadoutWeights,
};

/// WDM head indices for the 3-head multi-head ESN.
pub mod wdm_heads {
    /// Regime label head (Anderson group): predicts class probabilities.
    pub const REGIME_LABEL: usize = 0;
    /// Spectral bandwidth head (Steering group): predicts spectral spread.
    pub const SPECTRAL_BANDWIDTH: usize = 1;
    /// Confidence head (Meta group): classification confidence signal.
    pub const CONFIDENCE: usize = 2;
    /// Total number of WDM heads.
    pub const COUNT: usize = 3;
}

/// Multi-head WDM classifier wrapping `barracuda::esn_v2::MultiHeadEsn`.
///
/// Three heads predict different aspects of WDM regime classification:
/// - **Regime label** (Anderson group): class probabilities for 3 WDM regimes
/// - **Spectral bandwidth** (Steering group): predicted spectral spread
/// - **Confidence** (Meta group): classification certainty
///
/// `head_disagreement()` provides uncertainty via mean pairwise L2 distance
/// between head predictions (phase boundary signal from hotSpring).
pub struct MultiHeadWdmClassifier {
    esn: MultiHeadEsn,
    last_state: Option<barracuda::tensor::Tensor>,
    norm: EsnNormalization,
    n_classes: usize,
}

/// Result of a multi-head WDM classification.
#[derive(Debug, Clone)]
pub struct MultiHeadResult {
    /// Predicted regime label (0=Classical, 1=WDM, 2=Degenerate).
    pub label: usize,
    /// Raw class scores from the regime head.
    pub scores: Vec<f32>,
    /// Head disagreement (mean pairwise L2 distance). Higher values
    /// indicate the input is near a phase boundary.
    pub disagreement: f64,
}

/// WDM-specific head configuration for `MultiHeadEsn`.
#[must_use]
pub fn wdm_head_configs(n_classes: usize) -> Vec<HeadConfig> {
    vec![
        HeadConfig {
            group: HeadGroup::Anderson,
            label: "regime_label".to_string(),
            output_size: n_classes,
        },
        HeadConfig {
            group: HeadGroup::Steering,
            label: "spectral_bandwidth".to_string(),
            output_size: 1,
        },
        HeadConfig {
            group: HeadGroup::Meta,
            label: "confidence".to_string(),
            output_size: 1,
        },
    ]
}

impl MultiHeadWdmClassifier {
    /// Create a new multi-head WDM classifier.
    ///
    /// Initializes a `MultiHeadEsn` with 3 WDM-specific heads on a shared
    /// reservoir of `reservoir_size` neurons.
    ///
    /// # Errors
    ///
    /// Returns `Err` if ESN initialization fails (e.g., no GPU device).
    pub async fn new(reservoir_size: usize, n_classes: usize) -> Result<Self, String> {
        let config = ESNConfig {
            input_size: 2,
            reservoir_size,
            output_size: 1,
            spectral_radius: 0.95,
            connectivity: 0.1,
            leak_rate: 0.3,
            regularization: 1e-6,
            seed: 42,
        };

        let esn = MultiHeadEsn::new(config, wdm_head_configs(n_classes))
            .await
            .map_err(|e| format!("MultiHeadEsn init: {e}"))?;

        Ok(Self {
            esn,
            last_state: None,
            norm: EsnNormalization {
                x_mean: [0.0, 0.0],
                x_std: [1.0, 1.0],
            },
            n_classes,
        })
    }

    /// Set input normalization parameters (typically from Python baseline).
    pub const fn set_normalization(&mut self, norm: EsnNormalization) {
        self.norm = norm;
    }

    /// Feed an input through the shared reservoir.
    ///
    /// Returns the reservoir state tensor (also cached internally for
    /// subsequent `classify_from_state()` or `head_disagreement()` calls).
    ///
    /// # Errors
    ///
    /// Returns `Err` on GPU/Tensor failure.
    #[allow(clippy::cast_precision_loss)]
    pub async fn update(
        &mut self,
        log_rho: f64,
        log_t: f64,
        device: &std::sync::Arc<barracuda::device::WgpuDevice>,
    ) -> Result<barracuda::tensor::Tensor, String> {
        let x0 = ((log_rho - self.norm.x_mean[0]) / self.norm.x_std[0]) as f32;
        let x1 = ((log_t - self.norm.x_mean[1]) / self.norm.x_std[1]) as f32;

        let input = barracuda::tensor::Tensor::from_data(&[x0, x1], vec![2, 1], device.clone())
            .map_err(|e| format!("input tensor: {e}"))?;

        let state = self
            .esn
            .update(&input)
            .await
            .map_err(|e| format!("reservoir update: {e}"))?;

        self.last_state = Some(state.clone());
        Ok(state)
    }

    /// Train a specific head via ridge regression on collected states.
    ///
    /// # Errors
    ///
    /// Returns `Err` if training fails.
    pub fn train_head(
        &mut self,
        head_idx: usize,
        states: &[f64],
        targets: &[f64],
        lambda: f64,
    ) -> Result<(), String> {
        self.esn
            .train_head(head_idx, states, targets, lambda)
            .map_err(|e| format!("train head {head_idx}: {e}"))
    }

    /// Classify using the regime label head from the last reservoir state.
    ///
    /// # Errors
    ///
    /// Returns `Err` if no state is cached or prediction fails.
    pub fn classify_from_state(&self) -> Result<MultiHeadResult, String> {
        let state = self
            .last_state
            .as_ref()
            .ok_or("no cached state — call update() first")?;

        let scores = self
            .esn
            .predict_head(wdm_heads::REGIME_LABEL, state)
            .map_err(|e| format!("regime head predict: {e}"))?
            .to_vec()
            .map_err(|e| format!("scores readback: {e}"))?;

        let label = argmax_f32(&scores);

        let disagreement = self
            .esn
            .head_disagreement(state)
            .map_err(|e| format!("head disagreement: {e}"))?;

        Ok(MultiHeadResult {
            label,
            scores,
            disagreement,
        })
    }

    /// Feed input + classify in one call.
    ///
    /// # Errors
    ///
    /// Returns `Err` on GPU/Tensor failure.
    pub async fn classify_multi_head(
        &mut self,
        log_rho: f64,
        log_t: f64,
        device: &std::sync::Arc<barracuda::device::WgpuDevice>,
    ) -> Result<MultiHeadResult, String> {
        self.update(log_rho, log_t, device).await?;
        self.classify_from_state()
    }

    /// Head disagreement for the current reservoir state.
    ///
    /// Higher disagreement indicates the input is near a phase boundary.
    ///
    /// # Errors
    ///
    /// Returns `Err` if no state is cached or tensor op fails.
    pub fn head_disagreement(&self) -> Result<f64, String> {
        let state = self
            .last_state
            .as_ref()
            .ok_or("no cached state — call update() first")?;
        self.esn
            .head_disagreement(state)
            .map_err(|e| format!("head_disagreement: {e}"))
    }

    /// Export int8-quantized readout weights for NPU deployment (AKD1000).
    ///
    /// Exports the combined multi-head weights and quantizes them.
    ///
    /// # Errors
    ///
    /// Returns `Err` on export or quantization failure.
    pub fn export_npu_weights(&self) -> Result<NpuReadoutWeights, String> {
        let exported = self.export_weights()?;
        let w_out = exported.w_out.ok_or("no trained heads to export")?;

        let w_out_f64: Vec<f64> = w_out.iter().map(|&x| f64::from(x)).collect();
        let (weights_i8, scale, zero_point) = quantize_affine_i8_f64(&w_out_f64);

        Ok(NpuReadoutWeights {
            weights_i8,
            scale,
            zero_point,
            input_dim: exported.reservoir_size,
            output_dim: exported.output_size,
        })
    }

    /// Export all weights for cross-device deployment.
    ///
    /// # Errors
    ///
    /// Returns `Err` on tensor readback failure.
    pub fn export_weights(&self) -> Result<ExportedWeights, String> {
        self.esn
            .export_weights()
            .map_err(|e| format!("export weights: {e}"))
    }

    /// Access normalization parameters.
    #[must_use]
    pub const fn norm(&self) -> &EsnNormalization {
        &self.norm
    }

    /// Number of output classes.
    #[must_use]
    pub const fn n_classes(&self) -> usize {
        self.n_classes
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

fn argmax_f64(scores: &[f64]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

fn argmax_f32(scores: &[f32]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn tiny_esn() -> EsnClassifier {
        let rs = 4;
        let nc = 3;
        EsnClassifier {
            w_in: vec![0.1; rs * 2],
            w_res: vec![0.01; rs * rs],
            b_res: vec![0.0; rs],
            w_out: vec![0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            b_out: vec![0.0; nc],
            reservoir_size: rs,
            n_classes: nc,
            norm: EsnNormalization {
                x_mean: [0.5, 6.0],
                x_std: [1.0, 1.5],
            },
        }
    }

    #[test]
    fn classify_deterministic() {
        let esn = tiny_esn();
        let (l1, s1) = esn.classify(0.5, 5.5);
        let (l2, s2) = esn.classify(0.5, 5.5);
        assert_eq!(l1, l2);
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert!((a - b).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn classify_finite_scores() {
        let esn = tiny_esn();
        let (_, scores) = esn.classify(1.0, 6.0);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn classify_label_in_range() {
        let esn = tiny_esn();
        for &(lr, lt) in &[(-1.0, 8.0), (2.0, 4.0), (0.5, 5.5)] {
            let (label, _) = esn.classify(lr, lt);
            assert!(label < 3, "label {label} out of range");
        }
    }

    #[test]
    fn load_roundtrip() {
        let json = r#"{
            "normalization": {"x_mean": [0.5, 6.0], "x_std": [1.0, 1.5]},
            "weights": {
                "reservoir_size": 2, "input_dim": 2, "n_classes": 3,
                "W_in": [0.1, 0.2, 0.3, 0.4],
                "W_res": [0.01, 0.02, 0.03, 0.04],
                "b_res": [0.0, 0.0],
                "W_out": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "b_out": [0.0, 0.0, 0.0]
            }
        }"#;
        let esn = load_esn_from_json(json).expect("valid JSON should parse");
        let (label, scores) = esn.classify(0.5, 5.5);
        assert!(label < 3);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn load_invalid_json() {
        assert!(load_esn_from_json("nope").is_err());
    }

    #[test]
    fn load_missing_weights() {
        let json = r#"{"normalization": {"x_mean": [0, 0], "x_std": [1, 1]}}"#;
        assert!(load_esn_from_json(json).is_err());
    }

    #[test]
    fn argmax_helpers() {
        assert_eq!(argmax_f64(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax_f32(&[0.1, 0.9, 0.5]), 1);
        assert_eq!(argmax_f64(&[]), 0);
    }

    #[test]
    fn wdm_head_configs_correct() {
        let heads = wdm_head_configs(3);
        assert_eq!(heads.len(), wdm_heads::COUNT);
        assert_eq!(heads[wdm_heads::REGIME_LABEL].output_size, 3);
        assert_eq!(heads[wdm_heads::SPECTRAL_BANDWIDTH].output_size, 1);
        assert_eq!(heads[wdm_heads::CONFIDENCE].output_size, 1);
    }
}
