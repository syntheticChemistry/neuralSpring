// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: sequence forecasting primitives.
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/sequence/sequence_forecasting.py`
//! Run: 2026-02-16, Eastgate, Python 3.10, `PyTorch` 2.9.0+cu128, seed=42
//! Command: `python3 control/sequence/sequence_forecasting.py`
//! Reference: [`SEQUENCE_PROVENANCE`](neural_spring::provenance::SEQUENCE_PROVENANCE)
//!
//! Validates Rust-portable sequence primitives: windowed sequence creation,
//! persistence forecast, seasonal climatology, and gate activations
//! (sigmoid/tanh).  LSTM/GRU training remains Python-only (Phase 0).

use neural_spring::sequence::{
    create_sequences, persistence_forecast, seasonal_tmax, sigmoid, tanh_activation,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("sequence");

    validate_create_sequences(&mut h);
    validate_persistence(&mut h);
    validate_seasonal_model(&mut h);
    validate_sigmoid(&mut h);
    validate_tanh(&mut h);
    validate_determinism(&mut h);

    h.finish();
}

fn validate_create_sequences(h: &mut ValidationHarness) {
    // Analytical: data=[0..19], len=5, h=1 ⇒ seq[0][0]=0, target[0]=data[5]=5
    let data: Vec<f64> = (0..20).map(f64::from).collect();

    let (inputs, targets) = create_sequences(&data, 5, 1);
    h.check_bool(
        "seq count: 20 samples, len=5, h=1 → 15 windows",
        inputs.len() == 15,
    );
    h.check_abs(
        "seq[0] first element == 0",
        inputs[0][0],
        0.0,
        tolerances::EXACT_F64,
    );
    h.check_abs("seq[0] target == 5", targets[0], 5.0, tolerances::EXACT_F64);

    // Analytical: horizon=3 ⇒ target[0]=data[5+2]=7
    let (inputs_h3, targets_h3) = create_sequences(&data, 5, 3);
    h.check_abs(
        "horizon=3: target[0] == 7",
        targets_h3[0],
        7.0,
        tolerances::EXACT_F64,
    );
    h.check_bool(
        "horizon=3 produces fewer windows",
        inputs_h3.len() < inputs.len(),
    );
}

fn validate_persistence(h: &mut ValidationHarness) {
    // Analytical: persistence = last value of each window → 3, 30, 5
    let windows = vec![
        vec![1.0, 2.0, 3.0],
        vec![10.0, 20.0, 30.0],
        vec![5.0, 5.0, 5.0],
    ];
    let pred = persistence_forecast(&windows);
    h.check_abs("persist[0] == 3", pred[0], 3.0, tolerances::EXACT_F64);
    h.check_abs("persist[1] == 30", pred[1], 30.0, tolerances::EXACT_F64);
    h.check_abs("persist[2] == 5", pred[2], 5.0, tolerances::EXACT_F64);
}

fn validate_seasonal_model(h: &mut ValidationHarness) {
    // Analytical: seasonal_tmax(doy) sinusoid; doy 190≈summer, 10≈winter; annual mean≈8.5°C
    let summer = seasonal_tmax(190);
    let winter = seasonal_tmax(10);
    let annual_mean: f64 = (0..365).map(seasonal_tmax).sum::<f64>() / 365.0;

    h.check_bool("summer > 20°C", summer > 20.0);
    h.check_bool("winter < 0°C", winter < 0.0);
    h.check_abs(
        "annual mean ≈ 8.5°C (DC offset)",
        annual_mean,
        tolerances::SEASONAL_ANNUAL_MEAN,
        tolerances::SEASONAL_ANNUAL_MEAN_TOL,
    );
    h.check_bool("summer > winter", summer > winter);
}

fn validate_sigmoid(h: &mut ValidationHarness) {
    // Analytical: σ(0)=1/(1+e⁰)=0.5
    h.check_abs("σ(0) == 0.5", sigmoid(0.0), 0.5, tolerances::EXACT_F64);
    // Analytical: σ(-x)+σ(x)=1 (sigmoid symmetry)
    h.check_abs(
        "σ(-x) + σ(x) == 1",
        sigmoid(2.5) + sigmoid(-2.5),
        1.0,
        tolerances::EXACT_F64,
    );
    h.check_upper(
        &format!("σ(-100) < {}", tolerances::REGULATORY_RESPONSE_MIN),
        sigmoid(-100.0),
        tolerances::REGULATORY_RESPONSE_MIN,
    );
    h.check_lower("σ(100) > 0.99", sigmoid(100.0), 0.99);
}

fn validate_tanh(h: &mut ValidationHarness) {
    // Analytical: tanh(0)=0
    h.check_abs(
        "tanh(0) == 0",
        tanh_activation(0.0),
        0.0,
        tolerances::EXACT_F64,
    );
    // Analytical: tanh(x)+tanh(-x)=0 (antisymmetry)
    h.check_abs(
        "tanh antisymmetry",
        tanh_activation(2.0) + tanh_activation(-2.0),
        0.0,
        tolerances::EXACT_F64,
    );
    h.check_lower("tanh(100) > 0.99", tanh_activation(100.0), 0.99);
}

fn validate_determinism(h: &mut ValidationHarness) {
    let data: Vec<f64> = (0..100).map(f64::from).collect();
    let (inp1, tgt1) = create_sequences(&data, 14, 1);
    let (inp2, tgt2) = create_sequences(&data, 14, 1);
    h.check_bool(
        "create_sequences deterministic",
        inp1 == inp2 && tgt1 == tgt2,
    );

    let pred1 = persistence_forecast(&inp1);
    let pred2 = persistence_forecast(&inp2);
    h.check_bool("persistence deterministic", pred1 == pred2);

    let seasonal: Vec<f64> = (0..365).map(seasonal_tmax).collect();
    let seasonal2: Vec<f64> = (0..365).map(seasonal_tmax).collect();
    h.check_bool("seasonal_tmax deterministic", seasonal == seasonal2);
}
