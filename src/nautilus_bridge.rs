// SPDX-License-Identifier: AGPL-3.0-or-later

//! Nautilus Shell bridge — evolutionary reservoir computing from hotSpring.
//!
//! Bridges `bingocube_nautilus::NautilusBrain` with neuralSpring's spectral
//! analysis and ESN infrastructure. The Nautilus Shell is a feed-forward
//! alternative to recurrent ESN: board populations replace temporal feedback.
//!
//! ## Cross-Spring Provenance
//!
//! ```text
//! hotSpring (brain arch + proxy.rs) → BingoCube Nautilus Shell → neuralSpring
//! ```
//!
//! ## Architecture Comparison
//!
//! | Feature | Traditional ESN | Nautilus Shell |
//! |---------|----------------|----------------|
//! | Feedback | Temporal recurrence | None (feed-forward) |
//! | State | Hidden reservoir state | Board population |
//! | Learning | Ridge regression on states | Evolution + ridge on responses |
//! | Hardware | CPU/GPU (matmul) | CPU (board eval), NPU (export) |
//! | Cross-run | Weight serialization | Shell serialization + merge |
//!
//! ## Integration Points
//!
//! - **`SpectralNautilusBridge`**: Feeds `WeightSpectralResult` features into
//!   `NautilusBrain` as `BetaObservation` (repurposed: beta → disorder strength,
//!   `cg_iters` → iteration count, plaquette → level spacing ratio).
//! - **`DriftMonitor`**: Training stability detection for evolutionary populations.
//!   When `N_e * s < 1` for 3+ generations, selection is losing to drift.

use bingocube_nautilus::{BetaObservation, DriftMonitor, NautilusBrain, NautilusBrainConfig};

/// Bridge between neuralSpring spectral analysis and Nautilus evolutionary reservoir.
///
/// Maps weight spectral features (disorder, level spacing, bandwidth) to
/// `BetaObservation` for the Nautilus brain.
pub struct SpectralNautilusBridge {
    brain: NautilusBrain,
}

impl SpectralNautilusBridge {
    /// Create a new bridge with default Nautilus configuration.
    #[must_use]
    pub fn new(instance_name: &str) -> Self {
        let config = NautilusBrainConfig::default();
        let brain = NautilusBrain::new(config, instance_name);
        Self { brain }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: NautilusBrainConfig, instance_name: &str) -> Self {
        let brain = NautilusBrain::new(config, instance_name);
        Self { brain }
    }

    /// Feed a spectral observation into the Nautilus brain.
    ///
    /// Maps neuralSpring spectral features to hotSpring's `BetaObservation`:
    /// - `disorder` → `beta` (coupling strength analog)
    /// - `level_spacing_ratio` → `anderson_r`
    /// - `lambda_min` → `anderson_lambda_min`
    /// - `bandwidth` → `plaquette` (spread analog)
    /// - `ipr` → `cg_iters` (localization cost analog)
    pub fn observe_spectral(
        &mut self,
        disorder: f64,
        level_spacing_ratio: f64,
        lambda_min: f64,
        bandwidth: f64,
        ipr: f64,
    ) {
        let obs = BetaObservation {
            beta: disorder,
            quenched_plaq: None,
            quenched_plaq_var: None,
            plaquette: bandwidth,
            cg_iters: ipr * 1000.0,
            acceptance: level_spacing_ratio,
            delta_h_abs: 0.0,
            anderson_r: Some(level_spacing_ratio),
            anderson_lambda_min: Some(lambda_min),
        };
        self.brain.observe(obs);
    }

    /// Train the Nautilus shell on accumulated observations.
    /// Returns MSE if enough data is available.
    pub fn train(&mut self) -> Option<f64> {
        self.brain.train()
    }

    /// Predict spectral properties for a given disorder strength.
    /// Returns `(predicted_ipr_scaled, predicted_bandwidth, predicted_lsr)`.
    #[must_use] 
    pub fn predict(&self, disorder: f64) -> Option<(f64, f64, f64)> {
        self.brain.predict_dynamical(disorder, None)
    }

    /// Screen candidate disorder values by information content.
    #[must_use] 
    pub fn screen_candidates(&self, disorders: &[f64]) -> Vec<(f64, f64)> {
        self.brain.screen_candidates(disorders)
    }

    /// Detect concept edges (phase transitions) via LOO analysis.
    pub fn detect_concept_edges(&mut self) -> Vec<(f64, f64)> {
        self.brain.detect_concept_edges()
    }

    /// Whether the evolutionary population is drifting (`N_e`*s too low).
    #[must_use]
    pub fn is_drifting(&self) -> bool {
        self.brain.is_drifting()
    }

    /// Access the underlying drift monitor.
    #[must_use]
    pub const fn drift_monitor(&self) -> &DriftMonitor {
        &self.brain.drift
    }

    /// Serialize to JSON for cross-run transfer.
    ///
    /// # Errors
    /// Returns error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        self.brain.to_json()
    }

    /// Restore from JSON.
    ///
    /// # Errors
    /// Returns error if deserialization fails.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let brain = NautilusBrain::from_json(json)?;
        Ok(Self { brain })
    }

    /// Number of observations accumulated.
    #[must_use]
    pub const fn observation_count(&self) -> usize {
        self.brain.observations.len()
    }

    /// Whether the brain has been trained.
    #[must_use]
    pub const fn is_trained(&self) -> bool {
        self.brain.trained
    }
}

/// Re-export key Nautilus types for downstream use.
pub use bingocube_nautilus::{
    EvolutionConfig, NautilusShell, SelectionMethod, ShellConfig as NautilusShellConfig,
};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn bridge_creation() {
        let bridge = SpectralNautilusBridge::new("test-bridge");
        assert_eq!(bridge.observation_count(), 0);
        assert!(!bridge.is_trained());
    }

    #[test]
    fn bridge_observe_and_train() {
        let mut bridge = SpectralNautilusBridge::new("test-train");

        for i in 0..8 {
            let w = f64::from(i).mul_add(0.5, 2.0);
            let lsr = if w < 4.0 { 0.53 } else { 0.39 };
            let lam_min = 0.1 / w;
            let bw = w * 0.3;
            let ipr = if w < 4.0 { 0.01 } else { 0.5 };
            bridge.observe_spectral(w, lsr, lam_min, bw, ipr);
        }

        assert_eq!(bridge.observation_count(), 8);
        let mse = bridge.train();
        assert!(mse.is_some(), "should train with 8 points");
        assert!(bridge.is_trained());
    }

    #[test]
    fn bridge_predict_after_training() {
        let mut bridge = SpectralNautilusBridge::new("test-predict");

        for i in 0..10 {
            let w = f64::from(i).mul_add(0.4, 2.0);
            bridge.observe_spectral(w, 0.45 + w * 0.01, 0.1 / w, w * 0.3, 0.02 * w);
        }
        bridge.train();

        let pred = bridge.predict(3.0);
        assert!(pred.is_some(), "should predict after training");
        let (ipr_s, bw, lsr) = pred.unwrap();
        assert!(ipr_s.is_finite());
        assert!(bw.is_finite());
        assert!(lsr.is_finite());
    }

    #[test]
    fn bridge_serialization_roundtrip() {
        let mut bridge = SpectralNautilusBridge::new("test-ser");

        for i in 0..6 {
            let w = f64::from(i).mul_add(0.5, 3.0);
            bridge.observe_spectral(w, 0.45, 0.05, w * 0.3, 0.03 * w);
        }
        bridge.train();

        let json = bridge.to_json().expect("serialize");
        let restored = SpectralNautilusBridge::from_json(&json).expect("deserialize");

        assert_eq!(restored.observation_count(), bridge.observation_count());
        assert_eq!(restored.is_trained(), bridge.is_trained());
    }

    #[test]
    fn bridge_drift_detection() {
        let bridge = SpectralNautilusBridge::new("test-drift");
        assert!(!bridge.is_drifting());
    }

    #[test]
    fn bridge_screen_candidates() {
        let mut bridge = SpectralNautilusBridge::new("test-screen");

        for i in 0..8 {
            let w = f64::from(i).mul_add(0.5, 2.0);
            bridge.observe_spectral(w, 0.45, 0.05, w * 0.3, 0.02 * w);
        }
        bridge.train();

        let scored = bridge.screen_candidates(&[2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(scored.len(), 5);
    }

    #[test]
    fn bridge_concept_edges() {
        let mut bridge = SpectralNautilusBridge::new("test-edges");

        // Extended regime (low disorder)
        for i in 0..4 {
            let w = f64::from(i).mul_add(0.3, 1.0);
            bridge.observe_spectral(w, 0.53, 0.2, w * 0.2, 0.01);
        }
        // Critical regime (transition)
        bridge.observe_spectral(3.5, 0.46, 0.05, 1.0, 0.1);
        // Localized regime (high disorder)
        for i in 0..4 {
            let w = f64::from(i).mul_add(0.5, 5.0);
            bridge.observe_spectral(w, 0.39, 0.001, w * 0.4, 0.8);
        }

        bridge.train();
        let edges = bridge.detect_concept_edges();
        // edges may or may not be detected depending on shell evolution;
        // we just verify no panic and the API works
        assert!(edges.len() <= bridge.observation_count());
    }
}
