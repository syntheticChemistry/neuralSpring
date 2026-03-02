// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: immunological Anderson localization (baseCamp nS-06).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## baseCamp Sub-thesis 06
//!
//! Anderson Localization in Immunological Signaling.
//! Experiments nS-601 through nS-605.
//!
//! ## Provenance
//!
//! Cross-validates Rust `immunological_anderson` module against Python
//! baseline (`control/immunological_anderson/immunological_anderson.py`).
//! 20 Python checks → 20+ Rust checks with cross-language parity.

#![allow(clippy::too_many_lines, clippy::expect_used)]

use neural_spring::immunological_anderson::{
    classify_ad_state, dimensional_promotion, evenness_to_disorder, gonzales_ic50,
    ic50_to_w_reduction, lokivetmab_pk, pielou_evenness, tissue_geometry_factor, AdSkinState,
    AndersonDrugScore, DrugMechanism, PharmacoMonitor, SKIN_LAYERS,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("immunological_anderson");

    let baseline: serde_json::Value = serde_json::from_str(include_str!(
        "../../control/immunological_anderson/immunological_anderson_baseline.json"
    ))
    .expect("baseline JSON");

    // ── nS-601: Pielou evenness — cross-language parity ─────────────

    let j_even = pielou_evenness(&[0.25, 0.25, 0.25, 0.25]);
    let py_even = baseline["pielou_even"].as_f64().expect("pielou_even");
    h.check_abs(
        "Pielou evenness (even) Rust vs Python",
        j_even,
        py_even,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "Pielou evenness (even) = 1.0",
        j_even,
        1.0,
        tolerances::EXACT_F64,
    );

    let j_dom = pielou_evenness(&[0.97, 0.01, 0.01, 0.01]);
    let py_dom = baseline["pielou_dominated"]
        .as_f64()
        .expect("pielou_dominated");
    h.check_abs(
        "Pielou evenness (dominated) Rust vs Python",
        j_dom,
        py_dom,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool("Pielou dominated < 0.3", j_dom < 0.3);

    // ── nS-601: Evenness to disorder mapping ────────────────────────

    let w = evenness_to_disorder(0.8, 10.0);
    h.check_abs("Evenness → disorder W", w, 8.0, tolerances::EXACT_F64);

    // ── nS-601: Realistic skin cell populations ─────────────────────

    let healthy_dermis = [0.60, 0.15, 0.10, 0.08, 0.05, 0.02];
    let inflamed_dermis = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10];
    let j_healthy = pielou_evenness(&healthy_dermis);
    let j_inflamed = pielou_evenness(&inflamed_dermis);

    let py_j_healthy = baseline["pielou_healthy_dermis"]
        .as_f64()
        .expect("pielou_healthy_dermis");
    let py_j_inflamed = baseline["pielou_inflamed_dermis"]
        .as_f64()
        .expect("pielou_inflamed_dermis");
    h.check_abs(
        "Pielou healthy dermis Rust vs Python",
        j_healthy,
        py_j_healthy,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "Pielou inflamed dermis Rust vs Python",
        j_inflamed,
        py_j_inflamed,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "Inflamed evenness > healthy evenness",
        j_inflamed > j_healthy,
    );

    // ── nS-601: IC50 → W reduction — cross-language parity ─────────

    let at_ic50 = ic50_to_w_reduction(gonzales_ic50::JAK1, gonzales_ic50::JAK1, 1.0);
    let py_ic50 = baseline["ic50_half"].as_f64().expect("ic50_half");
    h.check_abs(
        "IC50 half-maximal Rust vs Python",
        at_ic50,
        py_ic50,
        tolerances::CROSS_LANGUAGE,
    );

    let at_10x = ic50_to_w_reduction(100.0, gonzales_ic50::JAK1, 1.0);
    let py_10x = baseline["ic50_10x"].as_f64().expect("ic50_10x");
    h.check_abs(
        "IC50 at 10x Rust vs Python",
        at_10x,
        py_10x,
        tolerances::CROSS_LANGUAGE,
    );

    // ── nS-601: IC50 sweep monotonicity ─────────────────────────────

    let concs = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0];
    let w_reductions: Vec<f64> = concs
        .iter()
        .map(|&c| ic50_to_w_reduction(c, gonzales_ic50::JAK1, 1.0))
        .collect();
    let mono = w_reductions.windows(2).all(|w| w[0] <= w[1]);
    h.check_bool("IC50 sweep monotonic", mono);

    // Cross-validate sweep values against Python
    let py_sweep = baseline["ic50_sweep"]["w_reductions"]
        .as_array()
        .expect("ic50_sweep w_reductions as array");
    for (i, (rs, py)) in w_reductions
        .iter()
        .zip(
            py_sweep
                .iter()
                .map(|v| v.as_f64().expect("ic50_sweep w_reductions element")),
        )
        .enumerate()
    {
        h.check_abs(
            &format!("IC50 sweep[{i}] Rust vs Python"),
            *rs,
            py,
            tolerances::CROSS_LANGUAGE,
        );
    }

    // ── nS-602: Dimensional promotion — cross-language parity ───────

    let d_intact = dimensional_promotion(1.0, 2.0, 3.0);
    let py_intact = baseline["dim_intact"].as_f64().expect("dim_intact");
    h.check_abs(
        "Dim promotion (intact) Rust vs Python",
        d_intact,
        py_intact,
        tolerances::CROSS_LANGUAGE,
    );

    let d_breach = dimensional_promotion(0.0, 2.0, 3.0);
    let py_breach = baseline["dim_breached"].as_f64().expect("dim_breached");
    h.check_abs(
        "Dim promotion (breached) Rust vs Python",
        d_breach,
        py_breach,
        tolerances::CROSS_LANGUAGE,
    );

    // Breach sweep
    let fracs = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0];
    let d_effs: Vec<f64> = fracs
        .iter()
        .map(|&f| dimensional_promotion(f, 2.0, 3.0))
        .collect();
    let mono_breach = d_effs.windows(2).all(|w| w[0] <= w[1]);
    h.check_bool("Breach sweep d_eff monotonic", mono_breach);

    let py_d_effs = baseline["breach_sweep"]["d_effs"]
        .as_array()
        .expect("breach_sweep d_effs as array");
    for (i, (rs, py)) in d_effs
        .iter()
        .zip(
            py_d_effs
                .iter()
                .map(|v| v.as_f64().expect("breach_sweep d_effs element")),
        )
        .enumerate()
    {
        h.check_abs(
            &format!("Breach sweep[{i}] Rust vs Python"),
            *rs,
            py,
            tolerances::CROSS_LANGUAGE,
        );
    }

    // ── nS-602: Cross-species barrier comparison ────────────────────

    let d_canine = dimensional_promotion(0.7, 2.0, 3.0);
    let d_human = dimensional_promotion(0.9, 2.0, 3.0);
    let py_canine = baseline["cross_species"]["canine_d_eff"]
        .as_f64()
        .expect("cross_species canine_d_eff");
    let py_human = baseline["cross_species"]["human_d_eff"]
        .as_f64()
        .expect("cross_species human_d_eff");
    h.check_abs(
        "Cross-species canine Rust vs Python",
        d_canine,
        py_canine,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "Cross-species human Rust vs Python",
        d_human,
        py_human,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool("Canine d_eff > human d_eff", d_canine > d_human);

    // ── nS-603: AD classification — cross-language parity ───────────

    h.check_bool(
        "AD classify healthy",
        classify_ad_state(0.40, 2.0, false) == AdSkinState::Healthy,
    );
    h.check_bool(
        "AD classify chronic",
        classify_ad_state(0.60, 2.8, false) == AdSkinState::Chronic,
    );
    h.check_bool(
        "AD classify flare",
        classify_ad_state(0.60, 2.6, false) == AdSkinState::Flare,
    );
    h.check_bool(
        "AD classify treated",
        classify_ad_state(0.40, 2.6, true) == AdSkinState::Treated,
    );

    // ── nS-604: Tissue geometry — cross-language parity ─────────────

    let g_sys = tissue_geometry_factor(0.3, true, 0.0);
    let py_g_sys = baseline["geom_systemic_small"]
        .as_f64()
        .expect("geom_systemic_small");
    h.check_abs(
        "Geometry systemic small Rust vs Python",
        g_sys,
        py_g_sys,
        tolerances::CROSS_LANGUAGE,
    );

    let g_top = tissue_geometry_factor(150.0, false, 0.0);
    let py_g_top = baseline["geom_topical_large"]
        .as_f64()
        .expect("geom_topical_large");
    h.check_abs(
        "Geometry topical large Rust vs Python",
        g_top,
        py_g_top,
        tolerances::CROSS_LANGUAGE,
    );

    let g_intact = tissue_geometry_factor(0.3, false, 0.0);
    let g_breached = tissue_geometry_factor(0.3, false, 0.5);
    let py_g_intact = baseline["topical_breach_effect"]["intact"]
        .as_f64()
        .expect("topical_breach_effect intact");
    let py_g_breached = baseline["topical_breach_effect"]["breached"]
        .as_f64()
        .expect("topical_breach_effect breached");
    h.check_abs(
        "Topical intact Rust vs Python",
        g_intact,
        py_g_intact,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "Topical breached Rust vs Python",
        g_breached,
        py_g_breached,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool("Breach improves topical access", g_breached > g_intact);

    // ── nS-605: Anderson drug score ─────────────────────────────────

    let score = AndersonDrugScore::compute(
        "Oclacitinib",
        0.95,
        0.90,
        DrugMechanism::TransductionBlock,
        true,
    );
    let py_combined = baseline["drug_score"]["combined"]
        .as_f64()
        .expect("drug_score combined");
    h.check_abs(
        "Drug score combined Rust vs Python",
        score.combined_score,
        py_combined,
        tolerances::CROSS_LANGUAGE,
    );

    // ── nS-605: Lokivetmab PK data consistency ──────────────────────

    let pk = lokivetmab_pk::DOSE_DURATION;
    h.check_bool(
        "Lokivetmab duration monotonic",
        pk.windows(2).all(|w| w[0].2 < w[1].2),
    );
    let py_durations = baseline["lokivetmab_durations"]
        .as_array()
        .expect("lokivetmab_durations as array");
    for (i, (rs, py)) in pk
        .iter()
        .map(|d| d.2)
        .zip(
            py_durations
                .iter()
                .map(|v| v.as_f64().expect("lokivetmab_durations element")),
        )
        .enumerate()
    {
        h.check_abs(
            &format!("Lokivetmab duration[{i}] Rust vs Python"),
            rs,
            py,
            tolerances::EXACT_F64,
        );
    }

    // ── nS-605: Gonzales IC50 data consistency ──────────────────────

    let py_ic50 = &baseline["gonzales_ic50"];
    h.check_abs(
        "Gonzales JAK1 IC50 parity",
        gonzales_ic50::JAK1,
        py_ic50["JAK1"].as_f64().expect("gonzales_ic50 JAK1"),
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "Gonzales IL31 IC50 parity",
        gonzales_ic50::IL31,
        py_ic50["IL31"].as_f64().expect("gonzales_ic50 IL31"),
        tolerances::EXACT_F64,
    );
    h.check_bool(
        "JAK1 most potent (lowest IC50)",
        gonzales_ic50::JAK1 <= gonzales_ic50::IL2
            && gonzales_ic50::JAK1 <= gonzales_ic50::IL4
            && gonzales_ic50::JAK1 <= gonzales_ic50::IL6
            && gonzales_ic50::JAK1 <= gonzales_ic50::IL13
            && gonzales_ic50::JAK1 <= gonzales_ic50::IL31,
    );

    // ── nS-605: Skin layer stack ────────────────────────────────────

    h.check_bool("5 skin layers", SKIN_LAYERS.len() == 5);
    h.check_bool("Stratum corneum acellular", SKIN_LAYERS[0].acellular);
    h.check_bool("Viable epidermis cellular", !SKIN_LAYERS[1].acellular);
    h.check_abs(
        "Papillary dermis d=3",
        SKIN_LAYERS[3].effective_dimension,
        3.0,
        tolerances::EXACT_F64,
    );

    // ── nS-603: PharmacoMonitor basic construction ──────────────────

    let pm = PharmacoMonitor::new(2.0);
    h.check_abs(
        "PharmacoMonitor dose",
        pm.dose(),
        2.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "PharmacoMonitor initial hours",
        pm.hours_elapsed(),
        0.0,
        tolerances::EXACT_F64,
    );
    h.check_bool("PharmacoMonitor not drifting initially", !pm.is_drifting());

    h.finish();
}
