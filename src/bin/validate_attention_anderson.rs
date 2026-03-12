// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 100: CPU validation of attention Anderson spectral analysis.

use neural_spring::attention_anderson::{
    attention_spectral, load_attention_anderson_from_json, pearson_r,
};
use neural_spring::validation::ValidationHarness;

const BASELINE_JSON: &str =
    include_str!("../../control/attention_anderson/attention_anderson_baseline.json");

fn main() {
    let mut h = ValidationHarness::new("attention_anderson");

    eprintln!("\n── Exp 100: Attention Anderson Spectral ──");

    let baseline = match load_attention_anderson_from_json(BASELINE_JSON) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool("JSON load", false);
            eprintln!("FATAL: {e}");
            h.finish();
        }
    };

    h.check_bool("baseline loaded", !baseline.results.is_empty());
    h.check_bool("20 configs", baseline.results.len() == 20);

    validate_spectral_parity(&mut h, &baseline);
    validate_correlations(&mut h, &baseline);
    validate_physics(&mut h, &baseline);

    h.finish();
}

fn validate_spectral_parity(
    h: &mut ValidationHarness,
    baseline: &neural_spring::attention_anderson::AttentionAndersonBaseline,
) {
    eprintln!("\n── Spectral parity (reference matrix) ──");

    let sp = attention_spectral(&baseline.reference_matrix, baseline.reference_n);

    let ref_result = &baseline.results[baseline.n_configs / 2];
    eprintln!(
        "  Rust: SR={:.4}, IPR={:.4}, ξ={:.4}",
        sp.spectral_radius, sp.mean_ipr, sp.xi
    );

    h.check_bool("SR > 0", sp.spectral_radius > 0.0);
    h.check_bool("IPR > 0", sp.mean_ipr > 0.0);
    h.check_bool("IPR < 1", sp.mean_ipr < 1.0);
    h.check_bool("participation > 1", sp.participation > 1.0);
    h.check_bool("xi in (0, 1]", sp.xi > 0.0 && sp.xi <= 1.0);

    h.check_abs(
        "spectral radius vs Python mid",
        sp.spectral_radius,
        ref_result.spectral_radius,
        0.1,
    );
}

fn validate_correlations(
    h: &mut ValidationHarness,
    baseline: &neural_spring::attention_anderson::AttentionAndersonBaseline,
) {
    eprintln!("\n── Correlations ──");

    let qs: Vec<f64> = baseline.results.iter().map(|r| r.quality).collect();
    let entropies: Vec<f64> = baseline.results.iter().map(|r| r.entropy).collect();
    let iprs: Vec<f64> = baseline.results.iter().map(|r| r.mean_ipr).collect();

    let corr_quality_entropy = pearson_r(&qs, &entropies);
    let corr_quality_ipr = pearson_r(&qs, &iprs);
    let corr_entropy_ipr = pearson_r(&entropies, &iprs);

    eprintln!(
        "  r(q,entropy)={corr_quality_entropy:.4} r(q,ipr)={corr_quality_ipr:.4} r(ent,ipr)={corr_entropy_ipr:.4}"
    );

    h.check_abs(
        "r(quality, entropy)",
        corr_quality_entropy,
        baseline.r_quality_entropy,
        0.1,
    );
    h.check_abs(
        "r(quality, ipr)",
        corr_quality_ipr,
        baseline.r_quality_ipr,
        0.1,
    );
    h.check_abs(
        "r(entropy, ipr)",
        corr_entropy_ipr,
        baseline.r_entropy_ipr,
        0.1,
    );

    h.check_bool("r(quality, entropy) < 0", baseline.r_quality_entropy < 0.0);
}

fn validate_physics(
    h: &mut ValidationHarness,
    baseline: &neural_spring::attention_anderson::AttentionAndersonBaseline,
) {
    eprintln!("\n── Physics checks ──");

    for r in &baseline.results {
        h.check_bool(
            &format!("q={:.2} IPR finite", r.quality),
            r.mean_ipr.is_finite(),
        );
    }

    h.check_bool(
        "all ξ in (0,1]",
        baseline.results.iter().all(|r| r.xi > 0.0 && r.xi <= 1.0),
    );
    h.check_bool(
        "all participation > 1",
        baseline.results.iter().all(|r| r.participation > 1.0),
    );
}
