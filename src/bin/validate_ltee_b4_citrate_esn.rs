// SPDX-License-Identifier: AGPL-3.0-or-later

//! LTEE B4: ESN citrate early-warning validation (Blount et al. 2008).
//!
//! Validates ESN early-warning classification on synthetic LTEE
//! population trajectories. Reproduces the Python baseline's reservoir
//! dynamics, classification metrics, and score parity.
//!
//! Paper: Blount et al. "Historical contingency and the evolution of a
//! key innovation in an experimental population of *Escherichia coli*"
//! PNAS 105:7899-7906 (2008).
//!
//! Expected values: `control/ltee_citrate_esn/expected_values.json`

#![expect(clippy::cast_precision_loss, reason = "LTEE data indexing")]

use neural_spring::ltee_citrate_esn::{
    self, early_warning_metrics, load_citrate_esn_from_json,
};
use neural_spring::rng::Rng;
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

fn run_checks(h: &mut ValidationHarness) {
    let json_str = include_str!("../../control/ltee_citrate_esn/expected_values.json");
    let baseline = load_citrate_esn_from_json(json_str).expect("parse baseline JSON");

    // ── Check 1: Baseline loaded correctly ──────────────────────────
    h.check_bool(
        "B4-001: baseline JSON parsed successfully",
        baseline.predictor.reservoir_size == ltee_citrate_esn::RESERVOIR_SIZE,
    );

    // ── Check 2: Weight dimensions ──────────────────────────────────
    let rs = baseline.predictor.reservoir_size;
    let id = ltee_citrate_esn::INPUT_DIM;
    h.check_bool(
        "B4-002: W_in dimensions correct",
        baseline.predictor.w_in.len() == rs * id,
    );
    h.check_bool(
        "B4-003: W_res dimensions correct",
        baseline.predictor.w_res.len() == rs * rs,
    );
    h.check_bool(
        "B4-004: W_out dimensions correct",
        baseline.predictor.w_out.len() == rs,
    );

    // ── Check 3: Reservoir dynamics ─────────────────────────────────
    let x_test = [1.01, 0.002, 0.8, 0.005];
    let h_state = baseline.predictor.reservoir_step(&x_test);
    h.check_bool(
        "B4-005: reservoir state dimension correct",
        h_state.len() == rs,
    );
    h.check_bool(
        "B4-006: reservoir state values in [-1,1] (tanh)",
        h_state.iter().all(|&v| v.abs() <= 1.0),
    );

    let state_norm: f64 = h_state.iter().map(|v| v * v).sum::<f64>().sqrt();
    h.check_bool("B4-007: reservoir state norm > 0", state_norm > 0.0);

    // ── Check 4: First trajectory Rust-Python parity ────────────────
    let raw: serde_json::Value = serde_json::from_str(json_str).expect("re-parse");
    let features: Vec<f64> = raw["first_trajectory"]["features"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
        })
        .collect();

    let n_gens = baseline.n_generations;
    let states = baseline.predictor.reservoir_drive(&features, n_gens);
    let (preds, scores) = baseline.predictor.classify(&states, n_gens, 0.5);

    let max_score_diff: f64 = scores
        .iter()
        .zip(&baseline.first_trajectory_scores)
        .map(|(r, p)| (r - p).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "B4-008: score parity (max abs diff)",
        max_score_diff,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );

    h.check_bool(
        "B4-009: prediction labels match Python",
        preds == baseline.first_trajectory_predictions,
    );

    // ── Check 5: Train/test accuracy from baseline ──────────────────
    h.check_bool(
        "B4-010: train accuracy > 0.85",
        baseline.train_accuracy > 0.85,
    );
    h.check_bool(
        "B4-011: test accuracy > 0.85",
        baseline.test_accuracy > 0.85,
    );

    // ── Check 6: Trajectory generation parity ───────────────────────
    let mut rng = Rng::new(baseline.seed);
    let (traj_features, traj_labels) =
        ltee_citrate_esn::generate_trajectory(&mut rng, true);

    h.check_bool(
        "B4-012: generated trajectory has correct feature count",
        traj_features.len() == n_gens * ltee_citrate_esn::INPUT_DIM,
    );
    let pos_count: usize = traj_labels.iter().map(|&l| l as usize).sum();
    h.check_bool(
        "B4-013: potentiation window labels correct",
        pos_count == ltee_citrate_esn::WINDOW_GENS,
    );

    // ── Check 7: Potentiation biology ───────────────────────────────
    h.check_bool(
        "B4-014: potentiation before Cit+ (biology constraint)",
        baseline.potentiation_gen < baseline.cit_plus_gen,
    );

    let pot_window = baseline.cit_plus_gen - baseline.potentiation_gen;
    h.check_bool(
        "B4-015: potentiation window ~2000 gens (scaled)",
        pot_window == ltee_citrate_esn::CIT_PLUS_GEN - ltee_citrate_esn::POTENTIATION_GEN,
    );

    // ── Check 8: Classification on fresh trajectory ─────────────────
    let fresh_states = baseline.predictor.reservoir_drive(&traj_features, n_gens);
    let (fresh_preds, _) = baseline.predictor.classify(&fresh_states, n_gens, 0.5);
    let metrics = early_warning_metrics(&fresh_preds, &traj_labels);
    h.check_bool(
        "B4-016: fresh trajectory accuracy > 0.5",
        metrics.accuracy > 0.5,
    );
}

fn main() {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Info);

    let mut h = ValidationHarness::new("LTEE B4: ESN Citrate Early-Warning (Blount 2008)");
    run_checks(&mut h);
    h.finish();
}
