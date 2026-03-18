// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 099: CPU validation of HMM introgression on NN layers.

use neural_spring::introgression_nn::{
    build_nn_hmm, build_null_hmm, detection_metrics, introgression_fraction,
    load_introgression_nn_from_json,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

const BASELINE_JSON: &str =
    include_str!("../../control/introgression_nn/introgression_nn_baseline.json");

fn main() {
    let mut h = ValidationHarness::new("introgression_nn");

    println!("\n── Exp 099: HMM Introgression on NN Layers ──");

    let baseline = match load_introgression_nn_from_json(BASELINE_JSON) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool("JSON load", false);
            println!("FATAL: {e}");
            h.finish();
        }
    };

    h.check_bool("baseline loaded", baseline.n_layers == 100);
    h.check_bool("observations len", baseline.observations.len() == 100);
    h.check_bool("viterbi path len", baseline.viterbi_path.len() == 100);

    validate_viterbi_parity(&mut h, &baseline);
    validate_metrics(&mut h, &baseline);
    validate_likelihood(&mut h, &baseline);

    h.finish();
}

fn validate_viterbi_parity(
    h: &mut ValidationHarness,
    baseline: &neural_spring::introgression_nn::IntrogressionNnBaseline,
) {
    println!("\n── Viterbi parity ──");

    let hmm = build_nn_hmm();
    let (rust_path, _) = hmm.viterbi(&baseline.observations);

    h.check_bool("Rust path len == 100", rust_path.len() == 100);

    #[expect(clippy::cast_precision_loss, reason = "count ≤ 100 fits in f64")]
    let match_rate = {
        let matching = rust_path
            .iter()
            .zip(baseline.viterbi_path.iter())
            .filter(|(a, b)| a == b)
            .count();
        matching as f64 / 100.0
    };
    println!("  Viterbi match rate: {match_rate:.2}");
    h.check_bool("Viterbi match > 0.9", match_rate > 0.9);

    let rust_frac = introgression_fraction(&rust_path);
    h.check_abs(
        "introgression fraction",
        rust_frac,
        baseline.introgression_fraction,
        tolerances::INTROGRESSION_FRACTION_CROSS,
    );
}

fn validate_metrics(
    h: &mut ValidationHarness,
    baseline: &neural_spring::introgression_nn::IntrogressionNnBaseline,
) {
    println!("\n── Detection metrics ──");

    let (tpr, fpr, acc) = detection_metrics(&baseline.viterbi_path, &baseline.true_states);

    h.check_abs(
        "TPR",
        tpr,
        baseline.tpr,
        tolerances::CLASSIFIER_METRIC_CROSS,
    );
    h.check_abs(
        "FPR",
        fpr,
        baseline.fpr,
        tolerances::CLASSIFIER_METRIC_CROSS,
    );
    h.check_abs(
        "Accuracy",
        acc,
        baseline.accuracy,
        tolerances::CLASSIFIER_METRIC_CROSS,
    );

    h.check_bool("TPR > 0.5", baseline.tpr > 0.5);
    h.check_bool("FPR < 0.3", baseline.fpr < 0.3);
    h.check_bool("Accuracy > 0.7", baseline.accuracy > 0.7);
}

fn validate_likelihood(
    h: &mut ValidationHarness,
    baseline: &neural_spring::introgression_nn::IntrogressionNnBaseline,
) {
    println!("\n── Likelihood ratio ──");

    let hmm_introg = build_nn_hmm();
    let hmm_baseline = build_null_hmm();

    let (_, log_lik_introg) = hmm_introg.forward(&baseline.observations);
    let (_, log_lik_baseline) = hmm_baseline.forward(&baseline.observations);

    let rust_llr = 2.0 * (log_lik_introg - log_lik_baseline);
    println!("  Rust LLR: {rust_llr:.2}, Python LLR: {:.2}", baseline.llr);

    h.check_abs("LLR parity", rust_llr, baseline.llr, 2.0);
    h.check_bool("LLR > 0", baseline.llr > 0.0);
    h.check_bool("Rust LLR > 0", rust_llr > 0.0);
}
