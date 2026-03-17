// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 098: CPU validation of WDM surrogate ensemble QS.
//!
//! Validates disagreement→disorder mapping, Anderson localization
//! physics, and QS cooperation dynamics.

use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::wdm_ensemble_qs::{
    anderson_from_disorder, disagreement_to_disorder, load_ensemble_from_json, pearson_r,
    replicator_final_coop, snowdrift_payoff,
};

const BASELINE_JSON: &str =
    include_str!("../../control/wdm_ensemble_qs/wdm_ensemble_qs_baseline.json");

const W_SCALE: f64 = 20.0;

fn main() {
    let mut h = ValidationHarness::new("wdm_ensemble_qs");

    println!("\n── Exp 098: WDM Surrogate Ensemble QS ──");

    let baseline = match load_ensemble_from_json(BASELINE_JSON) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool("JSON load", false);
            println!("FATAL: {e}");
            h.finish();
        }
    };

    h.check_bool("baseline loaded", !baseline.slices.is_empty());
    h.check_bool("32 temp slices", baseline.slices.len() == 32);
    h.check_bool("5 surrogates", baseline.n_surrogates == 5);

    validate_disorder_mapping(&mut h, &baseline);
    validate_anderson_physics(&mut h, &baseline);
    validate_qs_dynamics(&mut h, &baseline);
    validate_coupling(&mut h, &baseline);
    validate_reference_disorder(&mut h, &baseline);

    h.finish();
}

fn validate_disorder_mapping(
    h: &mut ValidationHarness,
    baseline: &neural_spring::wdm_ensemble_qs::EnsembleBaseline,
) {
    println!("\n── Disorder mapping ──");

    let w0 = disagreement_to_disorder(0.0, 0.0, 1.0, W_SCALE);
    h.check_abs("W(d=0) = 0", w0, 0.0, tolerances::CROSS_LANGUAGE);

    let w1 = disagreement_to_disorder(1.0, 0.0, 1.0, W_SCALE);
    h.check_abs("W(d=1) = 20", w1, W_SCALE, tolerances::CROSS_LANGUAGE);

    let w05 = disagreement_to_disorder(0.5, 0.0, 1.0, W_SCALE);
    h.check_abs("W(d=0.5) = 10", w05, 10.0, tolerances::CROSS_LANGUAGE);

    h.check_bool("W field mean > 0", baseline.w_field_mean > 0.0);
    h.check_bool("W field std > 0", baseline.w_field_std > 0.0);
}

fn validate_anderson_physics(
    h: &mut ValidationHarness,
    baseline: &neural_spring::wdm_ensemble_qs::EnsembleBaseline,
) {
    println!("\n── Anderson physics ──");

    for s in &baseline.slices {
        h.check_bool(&format!("slice {} IPR > 0", s.temp_idx), s.mean_ipr > 0.0);
        h.check_bool(&format!("slice {} ξ > 0", s.temp_idx), s.xi > 0.0);
    }

    let low_disorder: Vec<f64> = (0..16)
        .map(|i| 0.01_f64.mul_add(f64::from(i), 0.5))
        .collect();
    let high_disorder: Vec<f64> = (0..16)
        .map(|i| 0.5_f64.mul_add(f64::from(i), 10.0))
        .collect();
    let (ipr_low, _) = anderson_from_disorder(&low_disorder);
    let (ipr_high, _) = anderson_from_disorder(&high_disorder);
    h.check_bool("high W → higher IPR", ipr_high > ipr_low);
}

fn validate_qs_dynamics(
    h: &mut ValidationHarness,
    baseline: &neural_spring::wdm_ensemble_qs::EnsembleBaseline,
) {
    println!("\n── QS dynamics ──");

    let p_low = snowdrift_payoff(0.1);
    let fc_low = replicator_final_coop(&p_low, 500);

    let p_high = snowdrift_payoff(0.9);
    let fc_high = replicator_final_coop(&p_high, 500);

    h.check_bool("low W → higher cooperation", fc_low > fc_high);
    h.check_bool(
        "baseline low-W coop > high-W coop",
        baseline.mean_coop_low_w > baseline.mean_coop_high_w,
    );
}

fn validate_coupling(
    h: &mut ValidationHarness,
    baseline: &neural_spring::wdm_ensemble_qs::EnsembleBaseline,
) {
    println!("\n── Coupling ──");

    h.check_bool("r(W, ξ) < 0", baseline.r_w_xi < 0.0);
    h.check_bool("r(W, ξ) > -1", baseline.r_w_xi > -1.0);

    let ws: Vec<f64> = baseline.slices.iter().map(|s| s.mean_w).collect();
    let xis: Vec<f64> = baseline.slices.iter().map(|s| s.xi).collect();
    let r_rust = pearson_r(&ws, &xis);

    h.check_abs("Rust r(W,ξ) vs Python", r_rust, baseline.r_w_xi, 0.05);
}

fn validate_reference_disorder(
    h: &mut ValidationHarness,
    baseline: &neural_spring::wdm_ensemble_qs::EnsembleBaseline,
) {
    println!("\n── Reference disorder ──");

    h.check_bool(
        "10 reference disorder values",
        baseline.reference_disorder.len() == 10,
    );
    h.check_bool(
        "all finite",
        baseline.reference_disorder.iter().all(|v| v.is_finite()),
    );
    h.check_bool(
        "all non-negative",
        baseline.reference_disorder.iter().all(|&v| v >= 0.0),
    );
}
