// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-head ESN via `barracuda::esn_v2` (hotSpring cross-spring evolution).
//!
//! Wraps barracuda's `MultiHeadEsn` with WDM-specific head configuration
//! and head disagreement for uncertainty quantification.
//!
//! Evolution chain:
//!   hotSpring 36-head concept → `barracuda::esn_v2::MultiHeadEsn`
//!   → neuralSpring `MultiHeadWdmClassifier` (3 WDM-domain heads)
//!   → `head_disagreement` → phase boundary signal

#![expect(
    clippy::cast_possible_truncation,
    reason = "f64→f32 narrowing is intentional for GPU tensor ops"
)]

use super::argmax_f32;
use super::classifier::EsnNormalization;

/// Tikhonov regularization parameter for ESN readout layer.
///
/// Prevents ill-conditioning in the reservoir-to-output weight solve.
/// Standard value for ESN applications (Lukoševičius & Jaeger, 2009).
///
/// This is a physics/ML constant (ridge penalty), not a validation tolerance —
/// it controls the ESN training algorithm, not a pass/fail comparison.
/// Changing it affects trained model quality, not validation accuracy.
const ESN_TIKHONOV_REGULARIZATION: f32 = 1e-6;

pub use barracuda::esn_v2::{
    ExportedWeights, HeadConfig, HeadGroup, MultiHeadEsn, NpuReadoutWeights, quantize_affine_i8_f64,
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
    device: std::sync::Arc<barracuda::device::WgpuDevice>,
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
        let config = barracuda::esn_v2::ESNConfig {
            input_size: 2,
            reservoir_size,
            output_size: 1,
            spectral_radius: 0.95,
            connectivity: 0.1,
            leak_rate: 0.3,
            regularization: ESN_TIKHONOV_REGULARIZATION,
            seed: 42,
            sgd_learning_rate: 0.01,
            sgd_min_iterations: 50,
            sgd_max_iterations: 1000,
        };

        let esn = MultiHeadEsn::new(config, wdm_head_configs(n_classes))
            .await
            .map_err(|e| format!("MultiHeadEsn init: {e}"))?;

        let device = std::sync::Arc::new(
            barracuda::device::WgpuDevice::new()
                .await
                .map_err(|e| format!("WgpuDevice init: {e}"))?,
        );

        Ok(Self {
            esn,
            device,
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

    /// `WgpuDevice` used for tensor creation.
    ///
    /// `Tensor` inputs in [`Self::update`](Self::update) must be created on this
    /// device (typically `self.wgpu_device().clone()`).
    #[must_use]
    pub const fn wgpu_device(&self) -> &std::sync::Arc<barracuda::device::WgpuDevice> {
        &self.device
    }

    /// Feed an input through the shared reservoir.
    ///
    /// Returns the reservoir state tensor (also cached internally for
    /// subsequent `classify_from_state()` or `head_disagreement()` calls).
    ///
    /// # Errors
    ///
    /// Returns `Err` on GPU/Tensor failure.
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

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test assertions use expect for clear messages"
)]
mod tests {
    use super::*;
    use crate::wdm_esn::classifier::EsnNormalization;
    use approx::assert_relative_eq;
    use serial_test::serial;

    #[test]
    fn wdm_head_configs_structure() {
        let heads = wdm_head_configs(3);
        assert_eq!(heads.len(), wdm_heads::COUNT);
        assert_eq!(heads[wdm_heads::REGIME_LABEL].label, "regime_label");
        assert_eq!(
            heads[wdm_heads::SPECTRAL_BANDWIDTH].group,
            HeadGroup::Steering
        );
        assert_eq!(heads[wdm_heads::CONFIDENCE].output_size, 1);
    }

    #[tokio::test]
    async fn new_export_and_train_errors_without_reservoir_step() {
        let mut c = MultiHeadWdmClassifier::new(8, 3)
            .await
            .expect("MultiHeadEsn init");
        assert_eq!(c.n_classes(), 3);

        c.set_normalization(EsnNormalization {
            x_mean: [0.25, -1.0],
            x_std: [2.0, 0.5],
        });
        assert_relative_eq!(c.norm().x_mean[0], 0.25);

        let err = c
            .classify_from_state()
            .expect_err("should fail without cached state");
        assert!(err.contains("cached state"));
        let err = c
            .head_disagreement()
            .expect_err("should fail without cached state");
        assert!(err.contains("cached state"));
        let err = c
            .export_npu_weights()
            .expect_err("should fail without trained heads");
        assert!(err.contains("trained"));

        assert!(c.train_head(99, &[0.0; 8], &[0.0; 3], 1e-3).is_err());
        assert!(c.train_head(0, &[], &[0.0; 3], 1e-3).is_err());
        assert!(c.train_head(0, &[0.0; 8], &[0.0; 3], -1.0).is_err());

        let _ = c.wgpu_device();
    }

    /// Full reservoir `update` / `train_head` / `classify`. Ignored in default `cargo test`
    /// because parallel GPU tests elsewhere in the crate race global wgpu state; run with
    /// `cargo test -p neural-spring multi_head_wdm_reservoir_smoke --lib -- --ignored --test-threads=1`.
    #[ignore = "parallel GPU tests race wgpu global state; run with --test-threads=1"]
    #[serial]
    #[tokio::test]
    async fn multi_head_wdm_reservoir_smoke() {
        let mut c = MultiHeadWdmClassifier::new(8, 3)
            .await
            .expect("MultiHeadEsn init");

        let device = c.wgpu_device().clone();

        let state_tensor = c
            .update(0.15, -0.4, &device)
            .await
            .expect("reservoir update");
        let state_f64: Vec<f64> = state_tensor
            .to_vec()
            .expect("readback")
            .into_iter()
            .map(f64::from)
            .collect();

        c.train_head(0, &state_f64, &[1.0, 0.0, 0.0], 1e-3)
            .expect("train regime head");

        let result = c.classify_from_state().expect("classify");
        assert!(result.label < 3);
        assert_eq!(result.scores.len(), 3);

        let exported = c.export_weights().expect("export weights");
        assert!(exported.w_out.is_some());
        let npu = c.export_npu_weights().expect("NPU export");
        assert_eq!(npu.input_dim, 8);
        assert!(!npu.weights_i8.is_empty());

        let t = c.update(0.0, 0.0, &device).await.expect("update");
        let s: Vec<f64> = t
            .to_vec()
            .expect("to_vec")
            .into_iter()
            .map(f64::from)
            .collect();
        c.train_head(0, &s, &[0.0, 1.0, 0.0], 1e-2)
            .expect("train again");

        let out = c
            .classify_multi_head(0.3, 0.1, &device)
            .await
            .expect("multi-head classify");
        assert_eq!(out.scores.len(), 3);
    }
}
