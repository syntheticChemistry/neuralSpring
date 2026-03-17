// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 099: `BarraCUDA` CPU + GPU validator for HMM introgression on NN layers.

use neural_spring::introgression_nn::load_introgression_nn_from_json;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

const BASELINE_JSON: &str =
    include_str!("../../control/introgression_nn/introgression_nn_baseline.json");

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("barracuda_introgression_nn");

    println!("\n── Exp 099: BarraCUDA Introgression NN ──");

    let baseline = match load_introgression_nn_from_json(BASELINE_JSON) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool("JSON load", false);
            println!("FATAL: {e}");
            h.finish();
        }
    };

    h.check_bool("baseline loaded", baseline.n_layers == 100);

    // Tier 1: BarraCUDA CPU stats on detection metrics
    println!("\n── Tier 1: BarraCUDA CPU ──");

    #[expect(clippy::cast_precision_loss, reason = "HMM states 0/1 fit in f64")]
    let path_f64: Vec<f64> = baseline.viterbi_path.iter().map(|&s| s as f64).collect();
    #[expect(clippy::cast_precision_loss, reason = "HMM states 0/1 fit in f64")]
    let truth_f64: Vec<f64> = baseline.true_states.iter().map(|&s| s as f64).collect();

    let r2_self = barracuda::stats::r_squared(&path_f64, &path_f64);
    h.check_abs("bC CPU self-R²", r2_self, 1.0, tolerances::EXACT_F64);

    match barracuda::stats::correlation::pearson_correlation(&path_f64, &truth_f64) {
        Ok(r) => {
            h.check_bool("bC CPU Pearson(path, truth) > 0", r > 0.0);
            h.check_bool("bC CPU Pearson finite", r.is_finite());
        }
        Err(e) => {
            println!("  Pearson error: {e}");
            h.check_bool("bC CPU Pearson", false);
        }
    }

    match barracuda::stats::correlation::variance(&path_f64) {
        Ok(v) => h.check_bool("bC CPU path variance > 0", v > 0.0),
        Err(e) => {
            println!("  variance error: {e}");
            h.check_bool("bC CPU variance", false);
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "observation indices 0..2 fit in f64"
    )]
    let obs_f64: Vec<f64> = baseline.observations.iter().map(|&o| o as f64).collect();
    let bc_rmse = barracuda::stats::rmse(&obs_f64, &path_f64);
    h.check_bool("bC CPU RMSE finite", bc_rmse.is_finite());

    h.finish();
}
