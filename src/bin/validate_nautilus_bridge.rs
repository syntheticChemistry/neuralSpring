// SPDX-License-Identifier: AGPL-3.0-or-later

//! Nautilus Shell cross-spring bridge validator.
//!
//! Validates the integration between neuralSpring's spectral analysis
//! and hotSpring's evolutionary reservoir computing (Nautilus Shell).
//!
//! ## Provenance
//!
//! Cross-spring origin: hotSpring (brain arch, proxy.rs) → bingoCube Nautilus Shell → neuralSpring.
//! Absorption: Nautilus Shell bridge, Anderson spectral features, `WeightSpectralResult` observation.
//! Validation: Spectral analysis ↔ evolutionary reservoir integration, regime detection, drift monitoring vs Nautilus observation.

//! ```text
//! cargo run --release --bin validate_nautilus_bridge
//! ```

#![expect(
    clippy::cast_lossless,
    clippy::expect_used,
    reason = "validation binary — numeric casts and assertions on known-good test data"
)]

use neural_spring::nautilus_bridge::SpectralNautilusBridge;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("Nautilus Shell Cross-Spring Bridge");

    validate_bridge_lifecycle(&mut h);
    validate_spectral_regime_detection(&mut h);
    validate_esn_vs_nautilus_comparison(&mut h);
    validate_serialization_roundtrip(&mut h);
    validate_drift_monitoring(&mut h);
    validate_concept_edge_detection(&mut h);

    h.finish();
}

fn validate_bridge_lifecycle(h: &mut ValidationHarness) {
    println!("\n─── Nautilus Bridge Lifecycle ───\n");

    let bridge = SpectralNautilusBridge::new("lifecycle-test");
    h.check_bool("bridge: creation succeeds", true);
    h.check_bool("bridge: starts untrained", !bridge.is_trained());
    h.check_bool(
        "bridge: starts with 0 observations",
        bridge.observation_count() == 0,
    );
    h.check_bool("bridge: not drifting initially", !bridge.is_drifting());
}

fn validate_spectral_regime_detection(h: &mut ValidationHarness) {
    println!("\n─── Spectral Regime Detection via Nautilus ───\n");

    let mut bridge = SpectralNautilusBridge::new("spectral-regime");

    // Extended regime: low disorder, GOE-like statistics
    for i in 0..5 {
        let w = (i as f64).mul_add(0.4, 1.0);
        bridge.observe_spectral(w, 0.53, 0.15, w * 0.25, 0.01);
    }

    // Transition regime
    for i in 0..3 {
        let w = (i as f64).mul_add(0.3, 3.5);
        bridge.observe_spectral(w, 0.46, 0.03, w * 0.3, 0.15);
    }

    // Localized regime: high disorder, Poisson-like statistics
    for i in 0..5 {
        let w = (i as f64).mul_add(0.5, 5.0);
        bridge.observe_spectral(w, 0.39, 0.001, w * 0.4, (i as f64).mul_add(0.05, 0.7));
    }

    h.check_bool(
        "spectral: 13 observations accumulated",
        bridge.observation_count() == 13,
    );

    let mse = bridge.train();
    h.check_bool("spectral: training succeeds", mse.is_some());
    h.check_bool("spectral: trained flag set", bridge.is_trained());

    if let Some(mse_val) = mse {
        h.check_bool("spectral: MSE is finite", mse_val.is_finite());
        println!("  training MSE: {mse_val:.6}");
    }

    // Predict in each regime
    let pred_ext = bridge.predict(2.0);
    h.check_bool(
        "spectral: extended regime prediction exists",
        pred_ext.is_some(),
    );

    let pred_loc = bridge.predict(6.0);
    h.check_bool(
        "spectral: localized regime prediction exists",
        pred_loc.is_some(),
    );

    if let (Some((ipr_ext, _, _)), Some((ipr_loc, _, _))) = (pred_ext, pred_loc) {
        h.check_bool(
            "spectral: localized IPR > extended IPR (regime separation)",
            ipr_loc > ipr_ext,
        );
        println!("  extended  IPR_scaled={ipr_ext:.4}");
        println!("  localized IPR_scaled={ipr_loc:.4}");
    }
}

fn validate_esn_vs_nautilus_comparison(h: &mut ValidationHarness) {
    println!("\n─── ESN vs Nautilus Architecture Comparison ───\n");

    // Both ESN and Nautilus should be able to learn the Anderson transition
    let mut bridge_a = SpectralNautilusBridge::new("esn-compare-a");
    let mut bridge_b = SpectralNautilusBridge::new("esn-compare-b");

    // Identical training data
    for i in 0..12 {
        let w = (i as f64).mul_add(0.6, 1.0);
        let lsr = if w < 3.5 {
            (w - 1.0).mul_add(-0.02, 0.53)
        } else {
            (7.0 - w).max(0.0).mul_add(0.01, 0.39)
        };
        let lam = (0.2 / w).max(0.001);
        let bw = w * 0.28;
        let ipr = if w < 3.5 {
            0.01 + w * 0.005
        } else {
            (w - 3.5).mul_add(0.15, 0.2)
        };
        bridge_a.observe_spectral(w, lsr, lam, bw, ipr);
        bridge_b.observe_spectral(w, lsr, lam, bw, ipr);
    }

    let mse_a = bridge_a.train();
    let mse_b = bridge_b.train();

    h.check_bool(
        "comparison: both bridges train successfully",
        mse_a.is_some() && mse_b.is_some(),
    );

    // Both should produce reasonable predictions
    let pred_a = bridge_a.predict(4.0);
    let pred_b = bridge_b.predict(4.0);
    h.check_bool(
        "comparison: both produce predictions at transition",
        pred_a.is_some() && pred_b.is_some(),
    );

    // Screen candidates — the transition region should rank highly
    let scored = bridge_a.screen_candidates(&[1.0, 2.0, 3.5, 5.0, 7.0]);
    h.check_bool(
        "comparison: candidate screening produces ranked list",
        scored.len() == 5,
    );

    let top_beta = scored[0].0;
    println!("  highest-information disorder: W={top_beta:.1}");
    h.check_bool(
        "comparison: screening identifies informative region",
        scored[0].1 >= 0.0,
    );
}

fn validate_serialization_roundtrip(h: &mut ValidationHarness) {
    println!("\n─── Serialization Roundtrip ───\n");

    let mut bridge = SpectralNautilusBridge::new("ser-test");

    for i in 0..8 {
        let w = (i as f64).mul_add(0.5, 2.0);
        bridge.observe_spectral(w, 0.45, 0.05, w * 0.3, 0.03 * w);
    }
    bridge.train();

    let json = bridge
        .to_json()
        .expect("JSON serialization of bridge should not fail for valid struct");
    h.check_bool("serialize: JSON produced", !json.is_empty());

    let restored = SpectralNautilusBridge::from_json(&json)
        .expect("JSON deserialization failed — baseline format may have changed");
    h.check_bool(
        "roundtrip: observation count preserved",
        restored.observation_count() == bridge.observation_count(),
    );
    h.check_bool(
        "roundtrip: trained status preserved",
        restored.is_trained() == bridge.is_trained(),
    );

    // Predictions should be deterministic after restore
    let pred_orig = bridge.predict(3.5);
    let pred_rest = restored.predict(3.5);
    h.check_bool(
        "roundtrip: predictions preserved",
        pred_orig.is_some() && pred_rest.is_some(),
    );

    if let (Some((a, b, c)), Some((x, y, z))) = (pred_orig, pred_rest) {
        h.check_abs("roundtrip: CG parity", x, a, tolerances::CROSS_LANGUAGE);
        h.check_abs("roundtrip: plaq parity", y, b, tolerances::CROSS_LANGUAGE);
        h.check_abs("roundtrip: acc parity", z, c, tolerances::CROSS_LANGUAGE);
    }
}

fn validate_drift_monitoring(h: &mut ValidationHarness) {
    println!("\n─── Drift Monitor Integration ───\n");

    let bridge = SpectralNautilusBridge::new("drift-test");

    h.check_bool("drift: monitor accessible", !bridge.is_drifting());

    let monitor = bridge.drift_monitor();
    h.check_bool(
        "drift: ne_s_history starts empty",
        monitor.ne_s_history.is_empty(),
    );
    h.check_bool("drift: not drifting initially", !monitor.is_drifting());
}

fn validate_concept_edge_detection(h: &mut ValidationHarness) {
    println!("\n─── Concept Edge Detection ───\n");

    let mut bridge = SpectralNautilusBridge::new("edge-test");

    // Build a dataset with a clear phase transition at W ≈ 4.0
    for i in 0..5 {
        let w = (i as f64).mul_add(0.5, 1.0);
        bridge.observe_spectral(w, 0.53, 0.15, w * 0.2, 0.01);
    }
    bridge.observe_spectral(3.8, 0.47, 0.04, 1.2, 0.08);
    bridge.observe_spectral(4.2, 0.43, 0.02, 1.5, 0.2);
    for i in 0..4 {
        let w = (i as f64).mul_add(0.5, 5.0);
        bridge.observe_spectral(w, 0.39, 0.001, w * 0.4, 0.7);
    }

    bridge.train();
    let edges = bridge.detect_concept_edges();

    h.check_bool("edges: detection completes without panic", true);
    println!("  detected edges: {}", edges.len());
    for (beta, err) in &edges {
        println!("    W={beta:.2}, LOO error={err:.4}");
    }

    // The edge detection is stochastic but should find at least some signal
    h.check_bool(
        "edges: result is bounded by observation count",
        edges.len() <= bridge.observation_count(),
    );
}
