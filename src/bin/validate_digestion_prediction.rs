// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paper 027: Rust-side validation of ESN anaerobic digestion prediction.
//!
//! Validates Wang et al. (2020) key findings:
//! - ESN predicts biogas yield from operational parameters (R² > 0.80)
//! - Temperature shows dual-optimum behavior
//! - pH sensitivity follows bell curve around 7.2
//! - OLR exhibits Monod saturation + inhibition
//! - HRT shows exponential approach to conversion
//! - Low pH / high OLR / short HRT reduce yield vs. optimum
//!
//! Loads ESN weights from Python baseline JSON and runs independent
//! Rust validation. Validates process model and ESN inference parity.
//!
//! ## Provenance
//!
//! Python baseline: `control/digestion_prediction/digestion_prediction.py`
//! Reference: Wang et al. (2020), Bioresour Technol 298:122495

use neural_spring::digestion_prediction::{
    biogas_yield, hrt_response, load_digestion_from_json, olr_response, ph_response,
    temperature_response,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

const BASELINE_JSON: &str =
    include_str!("../../control/digestion_prediction/digestion_prediction_baseline.json");

fn main() {
    let mut h = ValidationHarness::new("digestion_prediction");

    println!("\n── Paper 027: ML Digestion Prediction (Wang et al. 2020) ──");

    // ── Load baseline ──
    let baseline = match load_digestion_from_json(BASELINE_JSON) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool("JSON load", false);
            println!("FATAL: failed to load baseline: {e}");
            h.finish();
        }
    };

    let pred = &baseline.predictor;
    h.check_bool("baseline loaded", pred.reservoir_size > 0);
    h.check_bool("reservoir_size = 512", pred.reservoir_size == 512);
    h.check_bool("has references", !baseline.reference_predictions.is_empty());

    // ── Process model validation ──
    println!("\n── Process model response functions ──");

    let f_t_meso = temperature_response(35.0);
    h.check_abs("T response mesophilic peak", f_t_meso, 0.7, 0.05);

    let f_t_thermo = temperature_response(55.0);
    h.check_abs("T response thermophilic peak", f_t_thermo, 0.3, 0.05);

    let f_t_cold = temperature_response(20.0);
    h.check_bool("cold T < mesophilic", f_t_cold < f_t_meso);

    let f_ph_opt = ph_response(7.2);
    h.check_abs("pH response at 7.2", f_ph_opt, 1.0, tolerances::EXACT_F64);

    let f_ph_acid = ph_response(5.5);
    h.check_bool("pH 5.5 < pH 7.2", f_ph_acid < f_ph_opt);

    let f_olr_low = olr_response(0.5);
    let f_olr_mid = olr_response(3.0);
    let f_olr_high = olr_response(8.0);
    h.check_bool("OLR 3 > OLR 0.5 (saturation)", f_olr_mid > f_olr_low);
    h.check_bool("OLR 8 < OLR 3 (inhibition)", f_olr_high < f_olr_mid);

    let f_hrt_short = hrt_response(5.0);
    let f_hrt_long = hrt_response(40.0);
    h.check_bool("HRT 40d > HRT 5d", f_hrt_long > f_hrt_short);
    h.check_bool("HRT 40d near complete", f_hrt_long > 0.95);

    let y_opt = biogas_yield(35.0, 7.2, 3.0, 20.0, 75.0);
    h.check_bool("optimum yield > 250", y_opt > 250.0);
    h.check_bool("optimum yield < 400", y_opt < 400.0);

    // ── ESN inference parity ──
    println!("\n── ESN inference vs Python baseline ──");

    for rp in &baseline.reference_predictions {
        let [t, ph, olr, hrt, vs_ts] = rp.inputs;
        let rs_pred = pred.predict(t, ph, olr, hrt, vs_ts);
        let py_pred = rp.predicted;
        let diff = (rs_pred - py_pred).abs();

        println!(
            "  {}: Rust={rs_pred:.2}, Python={py_pred:.2}, diff={diff:.2e}",
            rp.desc
        );

        h.check_abs(
            &format!("{} yield parity", rp.desc),
            rs_pred,
            py_pred,
            tolerances::CROSS_LANGUAGE,
        );

        let rs_h = pred.reservoir_state(t, ph, olr, hrt, vs_ts);
        let max_h_diff: f64 = rs_h
            .iter()
            .zip(&rp.reservoir_state)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        h.check_abs(
            &format!("{} reservoir state parity", rp.desc),
            max_h_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );
    }

    // ── Analytical yield vs ESN prediction comparison ──
    println!("\n── Analytical model vs ESN prediction ──");

    for rp in &baseline.reference_predictions {
        let [t, ph, olr, hrt, vs_ts] = rp.inputs;
        let analytical = biogas_yield(t, ph, olr, hrt, vs_ts);
        h.check_abs(
            &format!("{} analytical parity", rp.desc),
            analytical,
            rp.analytical,
            tolerances::CROSS_LANGUAGE,
        );
    }

    // ── Physical expectation checks ──
    println!("\n── Physical expectations ──");

    let y_meso = pred.predict(35.0, 7.2, 3.0, 20.0, 75.0);
    let y_thermo = pred.predict(55.0, 7.2, 3.0, 20.0, 75.0);
    let y_low_ph = pred.predict(35.0, 5.5, 3.0, 20.0, 75.0);
    let y_high_olr = pred.predict(35.0, 7.2, 7.0, 20.0, 75.0);
    let y_short_hrt = pred.predict(35.0, 7.2, 3.0, 5.0, 75.0);

    h.check_bool("mesophilic yield > 100", y_meso > 100.0);
    h.check_bool("thermophilic yield > 50", y_thermo > 50.0);
    h.check_bool("low pH reduces yield", y_low_ph < y_meso);
    h.check_bool("high OLR inhibition", y_high_olr < y_meso);
    h.check_bool("short HRT reduces yield", y_short_hrt < y_meso);

    // ── Metrics sanity ──
    h.check_bool("R²(test) > 0.80", baseline.r2_test > 0.80);
    h.check_bool("RMSE(test) < 40", baseline.rmse_test < 40.0);

    h.finish();
}
