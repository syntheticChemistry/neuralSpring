// SPDX-License-Identifier: AGPL-3.0-or-later

//! LTEE B3: Allele trajectory validation (Good et al. 2017).
//!
//! Validates LSTM+HMM+ESN fusion classifier on synthetic allele
//! frequency trajectories. Reproduces the Python baseline's feature
//! extraction, regime decoding, and classification parity.
//!
//! Paper: Good et al. "The dynamics of molecular evolution over 60,000
//! generations" Nature 551:45-50 (2017).
//!
//! Expected values: `control/ltee_allele_trajectory/expected_values.json`

#![expect(clippy::expect_used, reason = "validation binary — expect is idiomatic for test fixtures")]

use neural_spring::ltee_allele_trajectory::{
    self, classify_allele_fate, discretize_trajectory, esn_reservoir_step,
    hmm_forward_posterior, load_allele_baseline_from_json, lstm_forward, pool_features,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

struct SimpleLogger;
impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}", record.args());
        }
    }
    fn flush(&self) {}
}
static LOGGER: SimpleLogger = SimpleLogger;

#[expect(clippy::too_many_lines, reason = "validation harness — sequential checks")]
fn run_checks(h: &mut ValidationHarness) {
    let json_str = include_str!("../../control/ltee_allele_trajectory/expected_values.json");
    let bl = load_allele_baseline_from_json(json_str).expect("parse baseline JSON");

    // ── Check 1: Baseline loaded ────────────────────────────────────
    h.check_bool(
        "B3-001: baseline JSON parsed successfully",
        bl.lstm_w_x.len() == ltee_allele_trajectory::LSTM_HIDDEN,
    );

    // ── Check 2: Weight dimensions ──────────────────────────────────
    h.check_bool(
        "B3-002: LSTM W_h dimensions correct",
        bl.lstm_w_h.len()
            == ltee_allele_trajectory::LSTM_HIDDEN * ltee_allele_trajectory::LSTM_HIDDEN,
    );
    h.check_bool(
        "B3-003: HMM transition dimensions correct",
        bl.hmm_transition.len()
            == ltee_allele_trajectory::HMM_N_STATES * ltee_allele_trajectory::HMM_N_STATES,
    );
    h.check_bool(
        "B3-004: ESN W_in dimensions correct",
        bl.esn_w_in.len()
            == ltee_allele_trajectory::ESN_RESERVOIR * ltee_allele_trajectory::ESN_INPUT_DIM,
    );
    h.check_bool(
        "B3-005: ESN W_out dimensions correct",
        bl.esn_w_out.len()
            == ltee_allele_trajectory::ESN_RESERVOIR * ltee_allele_trajectory::N_CLASSES,
    );

    // ── Check 3: LSTM feature parity ────────────────────────────────
    let states = lstm_forward(
        &bl.first_trajectory,
        &bl.lstm_w_x,
        &bl.lstm_w_h,
        ltee_allele_trajectory::LSTM_HIDDEN,
    );
    let lstm_feats = pool_features(
        &states,
        bl.first_trajectory.len(),
        ltee_allele_trajectory::LSTM_HIDDEN,
    );

    let lstm_max_diff: f64 = lstm_feats
        .iter()
        .zip(&bl.first_lstm_features)
        .map(|(r, p)| (r - p).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "B3-006: LSTM feature parity (max abs diff)",
        lstm_max_diff,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );

    // ── Check 4: HMM posterior parity ───────────────────────────────
    let obs = discretize_trajectory(
        &bl.first_trajectory,
        ltee_allele_trajectory::HMM_N_SYMBOLS,
    );
    let posterior = hmm_forward_posterior(
        &obs,
        &bl.hmm_transition,
        &bl.hmm_emission,
        &bl.hmm_initial,
        ltee_allele_trajectory::HMM_N_STATES,
        ltee_allele_trajectory::HMM_N_SYMBOLS,
    );

    let hmm_max_diff: f64 = posterior
        .iter()
        .zip(&bl.first_hmm_posterior)
        .map(|(r, p)| (r - p).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "B3-007: HMM posterior parity (max abs diff)",
        hmm_max_diff,
        0.0,
        1e-6,
    );

    let posterior_sum: f64 = posterior.iter().sum();
    h.check_abs("B3-008: HMM posterior sums to 1.0", posterior_sum, 1.0, 1e-10);

    // ── Check 5: ESN state parity ───────────────────────────────────
    let mut combined = lstm_feats;
    combined.extend_from_slice(&posterior);

    let esn_state = esn_reservoir_step(
        &combined,
        &bl.esn_w_in,
        &bl.esn_w_res,
        &bl.esn_b_res,
        ltee_allele_trajectory::ESN_RESERVOIR,
    );

    let esn_max_diff: f64 = esn_state
        .iter()
        .zip(&bl.first_esn_state)
        .map(|(r, p)| (r - p).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "B3-009: ESN state parity (max abs diff)",
        esn_max_diff,
        0.0,
        1e-6,
    );

    h.check_bool(
        "B3-010: ESN state values in [-1,1]",
        esn_state.iter().all(|&v| v.abs() <= 1.0),
    );

    // ── Check 6: Classification parity ──────────────────────────────
    let (pred, scores) = classify_allele_fate(
        &esn_state,
        &bl.esn_w_out,
        ltee_allele_trajectory::N_CLASSES,
    );

    let score_max_diff: f64 = scores
        .iter()
        .zip(&bl.first_class_scores)
        .map(|(r, p)| (r - p).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "B3-011: classification score parity (max abs diff)",
        score_max_diff,
        0.0,
        1e-4,
    );

    h.check_bool(
        "B3-012: predicted class matches Python",
        pred == bl.first_prediction,
    );

    // ── Check 7: Accuracy from baseline ─────────────────────────────
    h.check_bool(
        "B3-013: train accuracy >= 0.95 (T06 target)",
        bl.train_accuracy >= 0.95,
    );
    h.check_bool(
        "B3-014: test accuracy >= 0.95 (T06 target)",
        bl.test_accuracy >= 0.95,
    );

    // ── Check 8: Trajectory generation biology ──────────────────────
    let mut rng = neural_spring::rng::Rng::new(42);
    let fix_traj = ltee_allele_trajectory::generate_allele_trajectory(
        &mut rng,
        0,
        ltee_allele_trajectory::SEQ_LEN,
    );
    h.check_bool(
        "B3-015: fixation trajectory ends high (>0.8)",
        fix_traj[ltee_allele_trajectory::SEQ_LEN - 1] > 0.8,
    );

    let mut rng2 = neural_spring::rng::Rng::new(42);
    let loss_traj = ltee_allele_trajectory::generate_allele_trajectory(
        &mut rng2,
        1,
        ltee_allele_trajectory::SEQ_LEN,
    );
    h.check_bool(
        "B3-016: loss trajectory ends low (<0.2)",
        loss_traj[ltee_allele_trajectory::SEQ_LEN - 1] < 0.2,
    );
}

fn main() {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Info);

    let mut h = ValidationHarness::new("LTEE B3: Allele Trajectory Classifier (Good 2017)");
    run_checks(&mut h);
    h.finish();
}
