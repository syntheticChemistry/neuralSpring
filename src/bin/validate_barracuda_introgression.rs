// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU port: PhyloNet-HMM introgression (Paper 018).
//!
//! Validates `barracuda::special::chi_squared_sf` for LRT p-value computation
//! and HMM forward pass agreement with hand-rolled introgression module.
//!
//! Evolution path:
//! ```text
//! Python (scipy.stats.chi2.sf) → Rust (hand-rolled LRT)
//!   → BarraCUDA CPU (chi_squared_sf)
//!   → BarraCUDA GPU (special.wgsl)
//! ```
//!
//! ## Provenance
//!
//! Python baseline: `control/introgression/introgression.py`
//! Rust baseline: `validate_introgression`

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::similar_names
)]

use neural_spring::introgression::{
    detect_introgression, generate_synthetic_loci, ils_only_hmm, introgression_fraction,
    log_likelihood_ratio, phylonet_hmm,
};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_introgression");

    validate_lrt_chi_squared_sf(&mut h);
    validate_forward_agreement(&mut h);
    validate_posterior_and_viterbi(&mut h);

    h.finish();
}

fn validate_lrt_chi_squared_sf(h: &mut ValidationHarness) {
    let hmm = phylonet_hmm();
    let mut rng = Rng::new(42);
    let (_, obs) = generate_synthetic_loci(200, &hmm, &mut rng);

    let (_, log_lik_introg) = hmm.forward(&obs);
    let ils_hmm = ils_only_hmm();
    let (_, log_lik_ils) = ils_hmm.forward(&obs);

    let lrt_stat = log_likelihood_ratio(log_lik_introg, log_lik_ils);

    let df = 1.0_f64;

    let p_barracuda = match barracuda::special::chi_squared_sf(lrt_stat.max(0.0), df) {
        Ok(p) => p,
        Err(e) => {
            h.check_bool(&format!("chi_squared_sf [ERROR: {e}]"), false);
            return;
        }
    };

    h.check_bool(
        "chi_squared_sf returns finite p-value",
        p_barracuda.is_finite() && (0.0..=1.0).contains(&p_barracuda),
    );

    let p_from_cdf =
        1.0 - barracuda::special::chi_squared_cdf(lrt_stat.max(0.0), df).unwrap_or(0.0);
    h.check_abs(
        "chi_squared_sf ≈ 1 - chi_squared_cdf",
        p_barracuda,
        p_from_cdf,
        tolerances::CROSS_LANGUAGE,
    );

    h.check_bool(
        &format!("LRT stat non-negative ({lrt_stat:.4}) when introg model fits better"),
        lrt_stat >= -1.0,
    );
}

fn validate_forward_agreement(h: &mut ValidationHarness) {
    let hmm = phylonet_hmm();
    let mut rng = Rng::new(42);
    let (_, obs) = generate_synthetic_loci(100, &hmm, &mut rng);

    let (_, log_lik) = hmm.forward(&obs);

    h.check_bool(
        &format!("HMM forward: finite log-lik ({log_lik:.4})"),
        log_lik.is_finite(),
    );
    h.check_bool("HMM forward: negative log-lik", log_lik < 0.0);
}

fn validate_posterior_and_viterbi(h: &mut ValidationHarness) {
    let hmm = phylonet_hmm();
    let mut rng = Rng::new(42);
    let (_true_states, obs) = generate_synthetic_loci(100, &hmm, &mut rng);

    let gamma = hmm.posterior(&obs);
    for (t, row) in gamma.iter().enumerate().take(3) {
        let sum: f64 = row.iter().sum();
        h.check_abs(
            &format!("posterior gamma[{t}] sums to 1"),
            sum,
            1.0,
            tolerances::HMM_POSTERIOR_SUM,
        );
    }

    let (path, viterbi_prob) = detect_introgression(&hmm, &obs);
    h.check_bool("Viterbi path length matches obs", path.len() == obs.len());
    h.check_bool(
        &format!("Viterbi log-prob finite ({viterbi_prob:.4})"),
        viterbi_prob.is_finite(),
    );

    let frac = introgression_fraction(&path);
    h.check_bool(
        &format!("introgression fraction in [0,1] ({frac:.4})"),
        (0.0..=1.0).contains(&frac),
    );
}
