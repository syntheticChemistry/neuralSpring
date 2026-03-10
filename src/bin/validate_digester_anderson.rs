// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 096: CPU validation of digester-Anderson coupling.
//!
//! Validates that Anderson disorder W (from microbial community diversity)
//! predicts ESN yield prediction quality — composing Paper 027 (ESN
//! digestion) and Paper 023 (Anderson localization).

use neural_spring::digester_anderson::{
    evenness_to_disorder, load_coupling_from_json, noise_from_xi, pearson_r, CouplingBaseline,
};
use neural_spring::digestion_prediction::biogas_yield;
use neural_spring::validation::ValidationHarness;

const BASELINE_JSON: &str =
    include_str!("../../control/digester_anderson/digester_anderson_baseline.json");

fn main() {
    let mut h = ValidationHarness::new("digester_anderson");

    eprintln!("\n── Exp 096: Digester-Anderson Coupling ──");

    let baseline = match load_coupling_from_json(BASELINE_JSON) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool("JSON load", false);
            eprintln!("FATAL: failed to load baseline: {e}");
            h.finish();
        }
    };

    let pred = &baseline.predictor;
    h.check_bool("baseline loaded", pred.reservoir_size > 0);
    h.check_bool("reservoir_size = 512", pred.reservoir_size == 512);
    h.check_bool("12 communities", baseline.communities.len() == 12);
    h.check_bool("has references", !baseline.reference_predictions.is_empty());

    validate_mapping(&mut h, &baseline);
    validate_anderson_physics(&mut h, &baseline);
    validate_noise_model(&mut h, &baseline);
    validate_esn_parity(&mut h, &baseline);
    validate_analytical_parity(&mut h, &baseline);
    validate_coupling(&mut h, &baseline);
    validate_physics_expectations(&mut h, &baseline);

    h.finish();
}

fn validate_mapping(h: &mut ValidationHarness, baseline: &CouplingBaseline) {
    eprintln!("\n── Diversity → Disorder mapping ──");
    h.check_abs("evenness=0 → W=20", evenness_to_disorder(0.0), 20.0, 1e-10);
    h.check_abs("evenness=1 → W=0", evenness_to_disorder(1.0), 0.0, 1e-10);
    h.check_abs(
        "evenness=0.5 → W=10",
        evenness_to_disorder(0.5),
        10.0,
        1e-10,
    );

    for comm in &baseline.communities {
        let w_rust = evenness_to_disorder(comm.evenness);
        h.check_abs(
            &format!("comm {} disorder parity", comm.id),
            w_rust,
            comm.disorder_w,
            1e-10,
        );
    }
}

fn validate_anderson_physics(h: &mut ValidationHarness, baseline: &CouplingBaseline) {
    eprintln!("\n── Anderson physics ──");

    let w_vals: Vec<f64> = baseline.communities.iter().map(|c| c.disorder_w).collect();
    let ipr_vals: Vec<f64> = baseline.communities.iter().map(|c| c.mean_ipr).collect();
    let xi_vals: Vec<f64> = baseline.communities.iter().map(|c| c.loc_length_xi).collect();

    let w_med = median_f64(&w_vals);

    let avg_ipr_hi = avg_where(&ipr_vals, &w_vals, |w| w > w_med);
    let avg_ipr_lo = avg_where(&ipr_vals, &w_vals, |w| w <= w_med);
    h.check_bool("high W → high IPR", avg_ipr_hi > avg_ipr_lo);

    let avg_xi_hi = avg_where(&xi_vals, &w_vals, |w| w > w_med);
    let avg_xi_lo = avg_where(&xi_vals, &w_vals, |w| w <= w_med);
    h.check_bool("high W → low ξ", avg_xi_hi < avg_xi_lo);
}

fn validate_noise_model(h: &mut ValidationHarness, baseline: &CouplingBaseline) {
    eprintln!("\n── Noise model ──");
    for comm in &baseline.communities {
        let noise_rust = noise_from_xi(comm.loc_length_xi);
        h.check_abs(
            &format!("comm {} noise parity", comm.id),
            noise_rust,
            comm.noise_std,
            1e-10,
        );
    }
}

fn validate_esn_parity(h: &mut ValidationHarness, baseline: &CouplingBaseline) {
    eprintln!("\n── ESN inference parity (Rust vs Python) ──");
    let pred = &baseline.predictor;

    for (i, rp) in baseline.reference_predictions.iter().enumerate() {
        let y_rust = pred.predict(rp.input[0], rp.input[1], rp.input[2], rp.input[3], rp.input[4]);
        h.check_abs(
            &format!("ref {i} ESN parity"),
            y_rust,
            rp.esn_yield,
            1e-6,
        );
    }
}

fn validate_analytical_parity(h: &mut ValidationHarness, baseline: &CouplingBaseline) {
    eprintln!("\n── Analytical yield parity ──");
    for (i, rp) in baseline.reference_predictions.iter().enumerate() {
        let y_analytical =
            biogas_yield(rp.input[0], rp.input[1], rp.input[2], rp.input[3], rp.input[4]);
        h.check_abs(
            &format!("ref {i} analytical parity"),
            y_analytical,
            rp.analytical_yield,
            1e-10,
        );
    }
}

fn validate_coupling(h: &mut ValidationHarness, baseline: &CouplingBaseline) {
    eprintln!("\n── Coupling signal ──");

    let w_vals: Vec<f64> = baseline.communities.iter().map(|c| c.disorder_w).collect();
    let r2_vals: Vec<f64> = baseline.communities.iter().map(|c| c.r2_test).collect();
    let xi_vals: Vec<f64> = baseline.communities.iter().map(|c| c.loc_length_xi).collect();

    let r_w = pearson_r(&w_vals, &r2_vals);
    h.check_bool("Pearson r(W, R²) < 0", r_w < 0.0);
    h.check_abs("r(W, R²) parity", r_w, baseline.metrics.pearson_w_r2, 1e-6);

    let r_xi = pearson_r(&xi_vals, &r2_vals);
    h.check_bool("Pearson r(ξ, R²) > 0", r_xi > 0.0);
    h.check_abs("r(ξ, R²) parity", r_xi, baseline.metrics.pearson_xi_r2, 1e-6);

    h.check_bool("|r(W, R²)| > 0.3", r_w.abs() > 0.3);
    h.check_bool("r(ξ, R²) > 0.5", r_xi > 0.5);
}

fn validate_physics_expectations(h: &mut ValidationHarness, baseline: &CouplingBaseline) {
    eprintln!("\n── Physical expectations ──");

    let r2_vals: Vec<f64> = baseline.communities.iter().map(|c| c.r2_test).collect();
    let w_vals: Vec<f64> = baseline.communities.iter().map(|c| c.disorder_w).collect();

    let lo_idx = min_idx(&w_vals);
    let hi_idx = max_idx(&w_vals);

    h.check_bool("low-W R² > high-W R²", r2_vals[lo_idx] > r2_vals[hi_idx]);
    h.check_bool("best community R² > 0.5", r2_vals[lo_idx] > 0.5);
    h.check_bool("pooled R² > 0.5", baseline.metrics.pooled_r2_test > 0.5);

    let pred = &baseline.predictor;
    let optimal = pred.predict(35.0, 7.2, 3.0, 20.0, 70.0);
    let stressed = pred.predict(25.0, 5.8, 7.0, 8.0, 55.0);
    h.check_bool("ESN: optimal > stressed", optimal > stressed);
}

fn median_f64(vals: &[f64]) -> f64 {
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[sorted.len() / 2]
}

fn avg_where(vals: &[f64], keys: &[f64], pred: impl Fn(f64) -> bool) -> f64 {
    let (sum, cnt) = vals
        .iter()
        .zip(keys)
        .filter(|(_, &k)| pred(k))
        .fold((0.0_f64, 0_u32), |(s, c), (&v, _)| (s + v, c + 1));
    sum / f64::from(cnt)
}

fn min_idx(vals: &[f64]) -> usize {
    vals.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

fn max_idx(vals: &[f64]) -> usize {
    vals.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}
