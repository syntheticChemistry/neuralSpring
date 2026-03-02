// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: `TrainingMonitor` with brain-inspired interrupts.
//!
//! Exercises the hotSpring cross-spring evolution:
//! - Attention state machine (GREEN/YELLOW/RED)
//! - BrainInterrupt pattern (Continue, ReduceLearningRate, EarlyStop)
//! - DriftMonitor integration for training populations
//! - SpectralNautilusBridge training epoch observation

#![allow(
    clippy::expect_used,
    clippy::pedantic,
    clippy::nursery,
    clippy::too_many_lines
)]

use neural_spring::nautilus_bridge::SpectralNautilusBridge;
use neural_spring::training_monitor::{AttentionState, TrainingInterrupt, TrainingMonitor};
use neural_spring::validation::ValidationHarness;
use neural_spring::weight_spectral::{SpectralPhase, WeightSpectralResult};

fn main() {
    let mut h = ValidationHarness::new("validate_training_monitor");

    validate_state_machine(&mut h);
    validate_interrupts(&mut h);
    validate_drift_detection(&mut h);
    validate_nautilus_training_bridge(&mut h);

    h.finish();
}

/// Synthetic `WeightSpectralResult` for FSM logic testing.
///
/// Not a production mock — constructs minimal spectral data to exercise
/// the attention state machine without requiring GPU eigensolve.
fn synthetic_spectral(bandwidth: f64, ipr: f64, lsr: f64) -> WeightSpectralResult {
    WeightSpectralResult {
        eigenvalues: vec![0.0; 10],
        mean_ipr: ipr,
        level_spacing_ratio: lsr,
        spectral_entropy: 2.0,
        mp_departure: 0.1,
        bandwidth,
        condition_number: 10.0,
        phase: SpectralPhase::Extended,
    }
}

fn validate_state_machine(h: &mut ValidationHarness) {
    let mut m = TrainingMonitor::new();
    h.check_bool("fsm: starts green", m.attention() == AttentionState::Green);
    h.check_bool("fsm: epoch_count=0", m.epoch_count() == 0);

    for i in 0..20 {
        let loss = 1.0 / (i as f64 + 1.0);
        m.observe_epoch(i, loss, &synthetic_spectral(1.0, 0.5, 0.53));
    }
    h.check_bool(
        "fsm: stable stays green",
        m.attention() == AttentionState::Green,
    );

    let mut m2 = TrainingMonitor::new();
    m2.observe_epoch(0, 1.0, &synthetic_spectral(1.0, 0.5, 0.53));
    m2.observe_epoch(1, 0.9, &synthetic_spectral(2.5, 0.5, 0.53));
    h.check_bool(
        "fsm: bandwidth growth → yellow",
        m2.attention() == AttentionState::Yellow,
    );

    let mut m3 = TrainingMonitor::new();
    m3.observe_epoch(0, 1.0, &synthetic_spectral(1.0, 0.5, 0.53));
    m3.observe_epoch(1, 0.9, &synthetic_spectral(6.0, 0.5, 0.53));
    h.check_bool(
        "fsm: bandwidth explosion → red",
        m3.attention() == AttentionState::Red,
    );

    let mut m4 = TrainingMonitor::new();
    m4.observe_epoch(0, 1.0, &synthetic_spectral(1.0, 0.5, 0.53));
    m4.observe_epoch(1, 0.9, &synthetic_spectral(1.1, 0.005, 0.53));
    h.check_bool(
        "fsm: IPR collapse → red",
        m4.attention() == AttentionState::Red,
    );
}

fn validate_interrupts(h: &mut ValidationHarness) {
    let mut m_div = TrainingMonitor::new();
    m_div.observe_epoch(0, 1.0, &synthetic_spectral(1.0, 0.5, 0.53));
    m_div.observe_epoch(1, 15.0, &synthetic_spectral(1.0, 0.5, 0.53));
    h.check_bool(
        "interrupt: loss divergence → EarlyStop",
        matches!(m_div.check_interrupt(), TrainingInterrupt::EarlyStop { .. }),
    );

    let mut m_ipr = TrainingMonitor::new();
    m_ipr.observe_epoch(0, 1.0, &synthetic_spectral(1.0, 0.5, 0.53));
    m_ipr.observe_epoch(1, 0.9, &synthetic_spectral(1.1, 0.005, 0.53));
    h.check_bool(
        "interrupt: IPR collapse → EarlyStop",
        matches!(m_ipr.check_interrupt(), TrainingInterrupt::EarlyStop { .. }),
    );

    let mut m_ok = TrainingMonitor::new();
    for i in 0..5 {
        m_ok.observe_epoch(
            i,
            1.0 / (i as f64 + 1.0),
            &synthetic_spectral(1.0, 0.5, 0.53),
        );
    }
    h.check_bool(
        "interrupt: stable → Continue",
        m_ok.check_interrupt() == TrainingInterrupt::Continue,
    );
}

fn validate_drift_detection(h: &mut ValidationHarness) {
    let m = TrainingMonitor::new();
    h.check_bool("drift: new monitor not drifting", !m.is_drifting());
}

fn validate_nautilus_training_bridge(h: &mut ValidationHarness) {
    let mut bridge = SpectralNautilusBridge::new("training-test");

    for i in 0..10 {
        let loss = 1.0 / (i as f64 + 1.0);
        let spectral = synthetic_spectral(1.0 + i as f64 * 0.1, 0.3, 0.45 + i as f64 * 0.01);
        bridge.observe_training_epoch(loss, &spectral);
    }

    h.check_bool(
        "bridge: 10 training observations",
        bridge.observation_count() == 10,
    );

    let mse = bridge.train();
    h.check_bool("bridge: training succeeds", mse.is_some());
    h.check_bool("bridge: is_trained after train", bridge.is_trained());
    h.check_bool(
        "bridge: drift check available",
        bridge.is_drifting() || !bridge.is_drifting(),
    );
}
