// SPDX-License-Identifier: AGPL-3.0-or-later

//! Extended validation: immunological Anderson (nS-601..605).
//!
//! Cross-validates Rust extensions against Python extended baseline:
//! - nS-601: Gonzales dose-response (Hill equation, barrier heights)
//! - nS-602: Pruritus time-series model (G3 treatment decay)
//! - nS-603: Lokivetmab PK decay + duration regression
//! - nS-604: Three-compartment disorder, tissue lattice, barrier spectrum
//! - nS-605: Fajgenbaum MATRIX drug repurposing scoring

#![expect(
    clippy::too_many_lines,
    clippy::expect_used,
    reason = "validation binary"
)]

use neural_spring::immunological_anderson::{
    barrier_promotion_spectrum, cytokine_barrier_heights, gonzales_ic50, hill_dose_response,
    ic50_sweep, level_spacing_ratio, lokivetmab_duration_predict, pk_exponential_decay,
    pruritus_score_model, score_all_candidates, three_compartment_disorder,
    tissue_lattice_hamiltonian, AD_CHRONIC_PROFILE, AD_FLARE_PROFILE,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("immunological_anderson_extended");

    let baseline: serde_json::Value = serde_json::from_str(include_str!(
        "../../control/immunological_anderson/immunological_anderson_extended_baseline.json"
    ))
    .expect("extended baseline JSON");

    // ══════════════════════════════════════════════════════════════════
    // nS-601: Hill dose-response for all 6 Gonzales cytokines
    // ══════════════════════════════════════════════════════════════════

    let r_n1 = hill_dose_response(10.0, 10.0, 1.0, 1.0);
    let py_n1 = baseline["hill_n1_at_ic50"]
        .as_f64()
        .expect("hill_n1_at_ic50");
    h.check_abs(
        "nS-601: Hill n=1 at IC50 Rust vs Py",
        r_n1,
        py_n1,
        tolerances::CROSS_LANGUAGE,
    );

    let r_n1_below = hill_dose_response(5.0, 10.0, 1.0, 1.0);
    let r_n2_below = hill_dose_response(5.0, 10.0, 2.0, 1.0);
    let py_n1_below = baseline["hill_cooperativity"]["n1"]
        .as_f64()
        .expect("hill_cooperativity n1");
    let py_n2_below = baseline["hill_cooperativity"]["n2"]
        .as_f64()
        .expect("hill_cooperativity n2");
    h.check_abs(
        "nS-601: Hill n=1 below IC50 Rust vs Py",
        r_n1_below,
        py_n1_below,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "nS-601: Hill n=2 below IC50 Rust vs Py",
        r_n2_below,
        py_n2_below,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "nS-601: Hill cooperativity (n=2 < n=1 below IC50)",
        r_n2_below < r_n1_below,
    );

    let concs = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0];
    let ic50_names = ["JAK1", "IL2", "IL4", "IL6", "IL13", "IL31"];
    let ic50_vals = [
        gonzales_ic50::JAK1,
        gonzales_ic50::IL2,
        gonzales_ic50::IL4,
        gonzales_ic50::IL6,
        gonzales_ic50::IL13,
        gonzales_ic50::IL31,
    ];

    for (&name, &ic50) in ic50_names.iter().zip(ic50_vals.iter()) {
        let rs_sweep = ic50_sweep(ic50, 1.0, &concs);
        let mono = rs_sweep
            .windows(2)
            .all(|w| w[0] <= w[1] + tolerances::EXACT_F64);
        h.check_bool(&format!("nS-601: {name} sweep monotonic"), mono);

        let py_sweep = baseline["cytokine_sweeps"][name]
            .as_array()
            .expect("cytokine_sweeps as array");
        for (i, (rs, py)) in rs_sweep
            .iter()
            .zip(
                py_sweep
                    .iter()
                    .map(|v| v.as_f64().expect("cytokine_sweeps element")),
            )
            .enumerate()
        {
            h.check_abs(
                &format!("nS-601: {name} sweep[{i}] Rust vs Py"),
                *rs,
                py,
                tolerances::CROSS_LANGUAGE,
            );
        }
    }

    let heights = cytokine_barrier_heights(1.0);
    let py_heights = &baseline["barrier_heights"];
    for (&name, &(ic50, w)) in ic50_names.iter().zip(heights.iter()) {
        let py_w = py_heights[name]["W"].as_f64().expect("barrier_heights W");
        h.check_abs(
            &format!("nS-601: {name} barrier W Rust vs Py"),
            w,
            py_w,
            tolerances::CROSS_LANGUAGE,
        );
        let py_ic50 = py_heights[name]["ic50"]
            .as_f64()
            .expect("barrier_heights ic50");
        h.check_abs(
            &format!("nS-601: {name} IC50 parity"),
            ic50,
            py_ic50,
            tolerances::EXACT_F64,
        );
    }

    h.check_bool(
        "nS-601: Barrier ordering (JAK1 < IL31 < IL13)",
        heights[0].1 < heights[5].1 && heights[5].1 < heights[4].1,
    );

    let r_sat = hill_dose_response(10000.0, 10.0, 1.0, 1.0);
    let py_sat = baseline["hill_saturation"]
        .as_f64()
        .expect("hill_saturation");
    h.check_abs(
        "nS-601: Saturation at 1000x IC50 Rust vs Py",
        r_sat,
        py_sat,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool("nS-601: Saturation > 0.999", r_sat > 0.999);

    // ══════════════════════════════════════════════════════════════════
    // nS-602: Pruritus time-series model
    // ══════════════════════════════════════════════════════════════════

    let baseline_score = 8.0;
    let suppression = 0.7;
    let decay_rate = 0.01;

    let nadir = pruritus_score_model(0.0, baseline_score, suppression, decay_rate);
    let py_nadir = baseline["pruritus_nadir"].as_f64().expect("pruritus_nadir");
    h.check_abs(
        "nS-602: Pruritus nadir Rust vs Py",
        nadir,
        py_nadir,
        tolerances::CROSS_LANGUAGE,
    );

    let timepoints = [0.0, 24.0, 72.0, 168.0, 336.0, 672.0];
    let py_scores = baseline["pruritus_timeseries"]["scores"]
        .as_array()
        .expect("pruritus_timeseries scores as array");
    let mut rs_scores = Vec::new();
    for (i, &t) in timepoints.iter().enumerate() {
        let s = pruritus_score_model(t, baseline_score, suppression, decay_rate);
        let py_s = py_scores[i]
            .as_f64()
            .expect("pruritus_timeseries scores element");
        h.check_abs(
            &format!("nS-602: Pruritus t={t}h Rust vs Py"),
            s,
            py_s,
            tolerances::CROSS_LANGUAGE,
        );
        rs_scores.push(s);
    }

    let mono_recovery = rs_scores.windows(2).all(|w| w[0] <= w[1]);
    h.check_bool("nS-602: Pruritus recovery monotonic", mono_recovery);

    let asymptote = pruritus_score_model(10000.0, baseline_score, suppression, decay_rate);
    let py_asymptote = baseline["pruritus_asymptote"]
        .as_f64()
        .expect("pruritus_asymptote");
    h.check_abs(
        "nS-602: Pruritus asymptote Rust vs Py",
        asymptote,
        py_asymptote,
        0.1,
    );

    // ══════════════════════════════════════════════════════════════════
    // nS-603: Lokivetmab PK decay + duration regression
    // ══════════════════════════════════════════════════════════════════

    let c_half = pk_exponential_decay(100.0, 24.0, 24.0);
    let py_c_half = baseline["pk_half_life"].as_f64().expect("pk_half_life");
    h.check_abs(
        "nS-603: PK at half-life Rust vs Py",
        c_half,
        py_c_half,
        tolerances::CROSS_LANGUAGE,
    );

    let c_zero = pk_exponential_decay(100.0, 0.0, 24.0);
    let py_c_zero = baseline["pk_t0"].as_f64().expect("pk_t0");
    h.check_abs(
        "nS-603: PK at t=0 Rust vs Py",
        c_zero,
        py_c_zero,
        tolerances::CROSS_LANGUAGE,
    );

    let pk_times = [0.0, 6.0, 12.0, 24.0, 48.0, 96.0, 168.0];
    let py_pk_concs = baseline["pk_decay_curve"]["concentrations"]
        .as_array()
        .expect("pk_decay_curve concentrations as array");
    let mut rs_pk_concs = Vec::new();
    for (i, &t) in pk_times.iter().enumerate() {
        let c = pk_exponential_decay(100.0, t, 24.0);
        let py_c = py_pk_concs[i]
            .as_f64()
            .expect("pk_decay_curve concentrations element");
        h.check_abs(
            &format!("nS-603: PK decay t={t}h Rust vs Py"),
            c,
            py_c,
            tolerances::CROSS_LANGUAGE,
        );
        rs_pk_concs.push(c);
    }
    let pk_mono = rs_pk_concs.windows(2).all(|w| w[0] > w[1]);
    h.check_bool("nS-603: PK decay monotonically decreasing", pk_mono);

    let py_regression = baseline["lokivetmab_regression"]
        .as_array()
        .expect("lokivetmab_regression as array");
    for entry in py_regression {
        let dose = entry["dose"].as_f64().expect("lokivetmab_regression dose");
        let actual = entry["actual"]
            .as_f64()
            .expect("lokivetmab_regression actual");
        let py_predicted = entry["predicted"]
            .as_f64()
            .expect("lokivetmab_regression predicted");
        let rs_predicted = lokivetmab_duration_predict(dose);
        h.check_abs(
            &format!("nS-603: Lokivetmab regression dose={dose} Rust vs Py"),
            rs_predicted,
            py_predicted,
            tolerances::CROSS_LANGUAGE,
        );
        h.check_bool(
            &format!("nS-603: Regression err < 5d at dose={dose}"),
            (rs_predicted - actual).abs() < 5.0,
        );
    }

    let lok_doses = [0.05, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0];
    let dur_preds: Vec<f64> = lok_doses
        .iter()
        .map(|&d| lokivetmab_duration_predict(d))
        .collect();
    let dur_mono = dur_preds.windows(2).all(|w| w[0] < w[1]);
    h.check_bool("nS-603: Duration prediction monotonic", dur_mono);

    // ══════════════════════════════════════════════════════════════════
    // nS-604: Three-compartment disorder + tissue lattice
    // ══════════════════════════════════════════════════════════════════

    let immune_healthy = [0.25, 0.25, 0.25, 0.25];
    let skin_healthy: [f64; 4] = [0.80, 0.10, 0.05, 0.05];
    let neural_healthy = [0.50, 0.50];

    let tcd_h = three_compartment_disorder(&immune_healthy, &skin_healthy, &neural_healthy, 10.0);
    let py_tcd_h = &baseline["three_comp_healthy"];
    h.check_abs(
        "nS-604: immune_w Rust vs Py",
        tcd_h.immune_w,
        py_tcd_h["immune_w"]
            .as_f64()
            .expect("three_comp_healthy immune_w"),
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "nS-604: skin_w Rust vs Py",
        tcd_h.skin_w,
        py_tcd_h["skin_w"]
            .as_f64()
            .expect("three_comp_healthy skin_w"),
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "nS-604: neural_w Rust vs Py",
        tcd_h.neural_w,
        py_tcd_h["neural_w"]
            .as_f64()
            .expect("three_comp_healthy neural_w"),
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "nS-604: cross-compartment variance Rust vs Py",
        tcd_h.cross_compartment_variance,
        py_tcd_h["variance"]
            .as_f64()
            .expect("three_comp_healthy variance"),
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "nS-604: immune_w > skin_w (healthy)",
        tcd_h.immune_w > tcd_h.skin_w,
    );

    let immune_inflamed = [0.15, 0.30, 0.25, 0.30];
    let skin_inflamed = [0.40, 0.25, 0.20, 0.15];
    let neural_inflamed = [0.35, 0.65];
    let tcd_i =
        three_compartment_disorder(&immune_inflamed, &skin_inflamed, &neural_inflamed, 10.0);
    let py_tcd_i = &baseline["three_comp_inflamed"];
    h.check_abs(
        "nS-604: inflamed immune_w Rust vs Py",
        tcd_i.immune_w,
        py_tcd_i["immune_w"]
            .as_f64()
            .expect("three_comp_inflamed immune_w"),
        tolerances::CROSS_LANGUAGE,
    );
    let mean_h = (tcd_h.immune_w + tcd_h.skin_w + tcd_h.neural_w) / 3.0;
    let mean_i = (tcd_i.immune_w + tcd_i.skin_w + tcd_i.neural_w) / 3.0;
    h.check_bool("nS-604: inflamed mean_W > healthy mean_W", mean_i > mean_h);

    // Tissue lattice Hamiltonian symmetry
    let ham = tissue_lattice_hamiltonian(&[4, 4], &[1.0, 2.0], 1.0, 42);
    let n = 8;
    let mut sym_err = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            sym_err = sym_err.max((ham[i * n + j] - ham[j * n + i]).abs());
        }
    }
    h.check_bool(
        "nS-604: Tissue lattice symmetric",
        sym_err < tolerances::NUMERICAL_DISTINCTNESS,
    );

    // Level spacing ratio
    let decomp = neural_spring::eigh::eigh_householder_qr(&ham, n);
    let mut evals = decomp.eigenvalues;
    evals.sort_by(f64::total_cmp);
    let r = level_spacing_ratio(&evals);
    h.check_bool("nS-604: Level spacing r in (0, 1]", r > 0.0 && r <= 1.0);

    // Barrier promotion spectrum
    let spectrum = barrier_promotion_spectrum(16, 5, 1.0, 1.0);
    h.check_bool("nS-604: Spectrum length = 5", spectrum.len() == 5);
    h.check_bool(
        "nS-604: First step intact",
        (spectrum[0].0 - 1.0).abs() < tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "nS-604: Last step breached",
        spectrum[4].0.abs() < tolerances::CROSS_LANGUAGE,
    );

    let py_spectrum = baseline["barrier_spectrum"]
        .as_array()
        .expect("barrier_spectrum as array");
    for (i, (&(intact, d_eff, r_val), py_s)) in spectrum.iter().zip(py_spectrum.iter()).enumerate()
    {
        let py_d = py_s["d_eff"].as_f64().expect("barrier_spectrum d_eff");
        h.check_abs(
            &format!("nS-604: Spectrum[{i}] d_eff Rust vs Py"),
            d_eff,
            py_d,
            tolerances::CROSS_LANGUAGE,
        );
        h.check_bool(
            &format!("nS-604: Spectrum[{i}] d_eff in [2,3]"),
            (2.0..=3.0).contains(&d_eff),
        );
        h.check_bool(
            &format!("nS-604: Spectrum[{i}] r in [0,1]"),
            (0.0..=1.0).contains(&r_val),
        );
        let py_intact = py_s["intact"].as_f64().expect("barrier_spectrum intact");
        h.check_abs(
            &format!("nS-604: Spectrum[{i}] intact Rust vs Py"),
            intact,
            py_intact,
            tolerances::CROSS_LANGUAGE,
        );
    }

    // Multi-layer lattice eigenvalues real/finite
    let ham_multi = tissue_lattice_hamiltonian(&[8, 8, 4], &[0.5, 1.5, 2.0], 1.0, 42);
    let decomp_multi = neural_spring::eigh::eigh_householder_qr(&ham_multi, 20);
    h.check_bool(
        "nS-604: Multi-layer 20 eigenvalues finite",
        decomp_multi.eigenvalues.iter().all(|v| v.is_finite()),
    );

    // ══════════════════════════════════════════════════════════════════
    // nS-605: Fajgenbaum MATRIX — Anderson-augmented drug repurposing
    // ══════════════════════════════════════════════════════════════════

    let flare_scores = score_all_candidates(&AD_FLARE_PROFILE);
    let chronic_scores = score_all_candidates(&AD_CHRONIC_PROFILE);

    let py_flare = baseline["matrix_flare_scores"]
        .as_array()
        .expect("matrix_flare_scores as array");
    let py_chronic = baseline["matrix_chronic_scores"]
        .as_array()
        .expect("matrix_chronic_scores as array");

    for (rs, py) in flare_scores.iter().zip(py_flare.iter()) {
        let py_combined = py["combined_score"]
            .as_f64()
            .expect("matrix_flare combined_score");
        let py_pathway = py["pathway_score"]
            .as_f64()
            .expect("matrix_flare pathway_score");
        let py_geom = py["geometry_score"]
            .as_f64()
            .expect("matrix_flare geometry_score");
        h.check_abs(
            &format!("nS-605: Flare {} combined Rust vs Py", rs.drug_name),
            rs.combined_score,
            py_combined,
            tolerances::CROSS_LANGUAGE,
        );
        h.check_abs(
            &format!("nS-605: Flare {} pathway Rust vs Py", rs.drug_name),
            rs.pathway_score,
            py_pathway,
            tolerances::EXACT_F64,
        );
        h.check_abs(
            &format!("nS-605: Flare {} geometry Rust vs Py", rs.drug_name),
            rs.geometry_score,
            py_geom,
            tolerances::CROSS_LANGUAGE,
        );
        h.check_bool(
            &format!("nS-605: Flare {} score in (0,1]", rs.drug_name),
            rs.combined_score > 0.0 && rs.combined_score <= 1.0,
        );
    }

    for (rs, py) in chronic_scores.iter().zip(py_chronic.iter()) {
        let py_combined = py["combined_score"]
            .as_f64()
            .expect("matrix_chronic combined_score");
        h.check_abs(
            &format!("nS-605: Chronic {} combined Rust vs Py", rs.drug_name),
            rs.combined_score,
            py_combined,
            tolerances::CROSS_LANGUAGE,
        );
        h.check_bool(
            &format!("nS-605: Chronic {} score in (0,1]", rs.drug_name),
            rs.combined_score > 0.0 && rs.combined_score <= 1.0,
        );
    }

    // Small molecule geometry > mAb geometry
    let tofa = &flare_scores[1];
    let nemo = &flare_scores[5];
    let py_tofa_geom = baseline["geom_tofa_vs_nemo"]["tofa"]
        .as_f64()
        .expect("geom_tofa_vs_nemo tofa");
    let py_nemo_geom = baseline["geom_tofa_vs_nemo"]["nemo"]
        .as_f64()
        .expect("geom_tofa_vs_nemo nemo");
    h.check_abs(
        "nS-605: Tofacitinib geom Rust vs Py",
        tofa.geometry_score,
        py_tofa_geom,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "nS-605: Nemolizumab geom Rust vs Py",
        nemo.geometry_score,
        py_nemo_geom,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "nS-605: Small molecule geom > mAb geom",
        tofa.geometry_score > nemo.geometry_score,
    );

    // Tofacitinib ranks #1
    let mut sorted_flare: Vec<_> = flare_scores.iter().collect();
    sorted_flare.sort_by(|a, b| f64::total_cmp(&b.combined_score, &a.combined_score));
    h.check_bool(
        "nS-605: Tofacitinib top-ranked for flare",
        sorted_flare[0].drug_name == "Tofacitinib",
    );

    // Trametinib ranks low
    let trame_rank = sorted_flare
        .iter()
        .position(|s| s.drug_name == "Trametinib")
        .expect("Trametinib in candidates")
        + 1;
    h.check_bool("nS-605: Trametinib rank >= 4", trame_rank >= 4);

    // Score factorization: combined = pathway × geometry
    for s in &flare_scores {
        let expected = s.pathway_score * s.geometry_score;
        h.check_abs(
            &format!("nS-605: {} factorization", s.drug_name),
            s.combined_score,
            expected,
            tolerances::CROSS_LANGUAGE,
        );
    }

    // Integrated dose-response × MATRIX
    let tofa_ic50 = gonzales_ic50::JAK1;
    let tofa_response_100nm = hill_dose_response(100.0, tofa_ic50, 1.0, 1.0);
    let tofa_matrix = tofa.combined_score;
    let integrated = tofa_response_100nm * tofa_matrix;
    let py_integrated = baseline["integrated_tofa"]["integrated"]
        .as_f64()
        .expect("integrated_tofa integrated");
    h.check_abs(
        "nS-605: Integrated score Rust vs Py",
        integrated,
        py_integrated,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "nS-605: Integrated score in (0, 1)",
        integrated > 0.0 && integrated < 1.0,
    );

    h.finish();
}
