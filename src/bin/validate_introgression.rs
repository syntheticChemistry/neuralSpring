// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: PhyloNet-HMM introgression detection (Paper 018).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/introgression/introgression.py`
//! Paper: Liu et al. (2015) PNAS 112:196-201.
//! Command: `python3 control/introgression/introgression.py`
//! Result: 8/8 PASS (seed=42, 500 loci)

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use neural_spring::introgression::{
    detect_introgression, generate_ils_only_loci, generate_synthetic_loci, ils_only_hmm,
    introgression_fraction, log_likelihood_ratio, phylonet_hmm,
};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("introgression");
    let n_loci = 500;

    let hmm = phylonet_hmm();
    let mut rng = Rng::new(42);

    let (true_states, obs) = generate_synthetic_loci(n_loci, &hmm, &mut rng);
    let true_frac = introgression_fraction(&true_states);

    // 1. Forward produces finite log-likelihood
    let (_, log_lik) = hmm.forward(&obs);
    h.check_bool(
        &format!("forward: finite negative log-lik ({log_lik:.4})"),
        log_lik.is_finite() && log_lik < 0.0,
    );

    // 2. Viterbi accuracy > random
    let (path, viterbi_prob) = detect_introgression(&hmm, &obs);
    let accuracy = path
        .iter()
        .zip(true_states.iter())
        .filter(|(a, b)| a == b)
        .count() as f64
        / path.len() as f64;
    h.check_lower(
        &format!(
            "Viterbi accuracy ({accuracy:.4}) > 0.5+{}",
            tolerances::HMM_DECODE_ACCURACY_MIN
        ),
        accuracy,
        0.5 + tolerances::HMM_DECODE_ACCURACY_MIN,
    );

    // 3. Introgression model preferred over ILS-only (LRT)
    let ils_hmm = ils_only_hmm();
    let (_, log_lik_ils) = ils_hmm.forward(&obs);
    let lr = log_likelihood_ratio(log_lik, log_lik_ils);
    h.check_lower(
        &format!("LRT: introg model preferred (LR={lr:.2})"),
        lr,
        0.0,
    );

    // 4. Posterior sums to 1 per locus
    let gamma = hmm.posterior(&obs);
    let n = hmm.num_states();
    for (t, row) in gamma.chunks(n).enumerate().take(5) {
        let sum: f64 = row.iter().sum();
        h.check_abs(
            &format!("posterior gamma[{t}] sums to 1"),
            sum,
            1.0,
            tolerances::HMM_POSTERIOR_SUM,
        );
    }

    // 5. Detected fraction near true
    let detected_frac = introgression_fraction(&path);
    h.check_abs(
        &format!("detected frac ({detected_frac:.3}) near true ({true_frac:.3})"),
        detected_frac,
        true_frac,
        tolerances::INTROGRESSION_FRACTION_ABS,
    );

    // 6. FPR low when no introgression
    let (_, obs_ils) = generate_ils_only_loci(n_loci, &mut rng);
    let (path_ils, _) = detect_introgression(&hmm, &obs_ils);
    let fp_rate = introgression_fraction(&path_ils);
    h.check_upper(
        &format!(
            "FPR when no introgression ({fp_rate:.3}) < {}",
            tolerances::INTROGRESSION_FPR_MAX
        ),
        fp_rate,
        tolerances::INTROGRESSION_FPR_MAX,
    );

    // 7. Gene tree topology frequencies sensible
    let concordant = obs.iter().filter(|&&o| o == 0).count() as f64 / obs.len() as f64;
    let introg_like = obs.iter().filter(|&&o| o == 1).count() as f64 / obs.len() as f64;
    h.check_lower(
        &format!(
            "concordant frac ({concordant:.3}) > {}",
            tolerances::GENE_TREE_CONCORDANT_MIN
        ),
        concordant,
        tolerances::GENE_TREE_CONCORDANT_MIN,
    );
    h.check_lower(
        &format!("introg-like frac ({introg_like:.3}) > 0.05"),
        introg_like,
        tolerances::INTROGRESSION_FRACTION_MIN,
    );

    // 8. Viterbi log-prob finite
    h.check_bool(
        &format!("Viterbi finite log-prob ({viterbi_prob:.4})"),
        viterbi_prob.is_finite(),
    );

    h.finish();
}
