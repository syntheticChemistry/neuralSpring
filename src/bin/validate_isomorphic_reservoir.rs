// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 097: CPU validation of isomorphic reservoir ensemble.
//!
//! Validates spectral universality across three domains (digester ESN,
//! glucose LSTM, weather LSTM). Loads symmetrized weight matrices from
//! Python baseline and recomputes spectral properties in Rust.

use neural_spring::isomorphic_reservoir::{
    cross_domain_metrics, load_isomorphic_from_json, spectral_properties,
};
use neural_spring::validation::ValidationHarness;

const BASELINE_JSON: &str =
    include_str!("../../control/isomorphic_reservoir/isomorphic_reservoir_baseline.json");

fn main() {
    let mut h = ValidationHarness::new("isomorphic_reservoir");

    eprintln!("\n── Exp 097: Isomorphic Reservoir Ensemble ──");

    let baseline = match load_isomorphic_from_json(BASELINE_JSON) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool("JSON load", false);
            eprintln!("FATAL: {e}");
            h.finish();
        }
    };

    h.check_bool("baseline loaded", !baseline.spectra.is_empty());
    h.check_bool("3 domains", baseline.domain_matrices.len() == 3);
    h.check_bool("3 spectra", baseline.spectra.len() == 3);

    validate_spectral_parity(&mut h, &baseline);
    validate_cross_domain(&mut h, &baseline);
    validate_universality(&mut h, &baseline);
    validate_reference_sums(&mut h, &baseline);

    h.finish();
}

fn validate_spectral_parity(
    h: &mut ValidationHarness,
    baseline: &neural_spring::isomorphic_reservoir::IsomorphicBaseline,
) {
    eprintln!("\n── Spectral parity (Rust vs Python) ──");

    for (name, matrix, n) in &baseline.domain_matrices {
        let sp_rust = spectral_properties(matrix, *n, name);

        let sp_py = baseline
            .spectra
            .iter()
            .find(|s| s.name.contains(name.split('_').next().unwrap_or(name)))
            .or_else(|| {
                baseline.spectra.iter().find(|s| {
                    s.name.contains(name.as_str())
                        || name.contains(s.name.split('_').next().unwrap_or(&s.name))
                })
            });

        if let Some(py) = sp_py {
            eprintln!(
                "  {name}: SR_rust={:.4}, SR_py={:.4}",
                sp_rust.spectral_radius, py.spectral_radius
            );

            h.check_abs(
                &format!("{name} spectral_radius"),
                sp_rust.spectral_radius,
                py.spectral_radius,
                0.01,
            );
            h.check_abs(
                &format!("{name} mean_ipr"),
                sp_rust.mean_ipr,
                py.mean_ipr,
                0.005,
            );
            h.check_abs(
                &format!("{name} eff_ratio"),
                sp_rust.effective_ratio,
                py.effective_ratio,
                0.01,
            );
            h.check_abs(
                &format!("{name} ev_mean"),
                sp_rust.eigenvalue_mean,
                py.eigenvalue_mean,
                0.01,
            );
            h.check_abs(
                &format!("{name} spacing_ratio"),
                sp_rust.mean_spacing_ratio,
                py.mean_spacing_ratio,
                0.05,
            );
        } else {
            eprintln!("  {name}: no matching Python spectrum found");
            h.check_bool(&format!("{name} spectrum found"), false);
        }
    }
}

fn validate_cross_domain(
    h: &mut ValidationHarness,
    baseline: &neural_spring::isomorphic_reservoir::IsomorphicBaseline,
) {
    eprintln!("\n── Cross-domain metrics (Rust-computed) ──");

    let mut rust_spectra = Vec::new();
    for (name, matrix, n) in &baseline.domain_matrices {
        rust_spectra.push(spectral_properties(matrix, *n, name));
    }

    let cd = cross_domain_metrics(&rust_spectra);
    eprintln!(
        "  Rust eff_ratio: {:.4} ± {:.4} (CV={:.4})",
        cd.eff_ratio_mean, cd.eff_ratio_std, cd.eff_ratio_cv
    );
    eprintln!(
        "  Rust IPR: {:.4} ± {:.4} (CV={:.4})",
        cd.ipr_mean, cd.ipr_std, cd.ipr_cv
    );

    h.check_abs(
        "cross-domain eff_ratio_mean",
        cd.eff_ratio_mean,
        baseline.cross_domain.eff_ratio_mean,
        0.02,
    );
    h.check_abs(
        "cross-domain ipr_mean",
        cd.ipr_mean,
        baseline.cross_domain.ipr_mean,
        0.005,
    );
    h.check_abs(
        "cross-domain eff_ratio_cv",
        cd.eff_ratio_cv,
        baseline.cross_domain.eff_ratio_cv,
        0.05,
    );
}

fn validate_universality(
    h: &mut ValidationHarness,
    baseline: &neural_spring::isomorphic_reservoir::IsomorphicBaseline,
) {
    eprintln!("\n── Universality checks ──");

    h.check_bool(
        "eff_ratio CV < 0.5",
        baseline.cross_domain.eff_ratio_cv < 0.5,
    );
    h.check_bool("IPR CV < 0.5", baseline.cross_domain.ipr_cv < 0.5);

    for sp in &baseline.spectra {
        h.check_bool(&format!("{} IPR < 1", sp.name), sp.mean_ipr < 1.0);
        h.check_bool(
            &format!("{} eff_dim > 1", sp.name),
            sp.effective_dimension > 1.0,
        );
        h.check_bool(
            &format!("{} spacing in [0,1]", sp.name),
            (0.0..=1.0).contains(&sp.mean_spacing_ratio),
        );
    }
}

fn validate_reference_sums(
    h: &mut ValidationHarness,
    baseline: &neural_spring::isomorphic_reservoir::IsomorphicBaseline,
) {
    eprintln!("\n── Reference sums ──");
    for (name, val) in &baseline.reference_sums {
        h.check_bool(&format!("{name} w_out sum finite"), val.is_finite());
    }
}
