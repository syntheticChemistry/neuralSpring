// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU port: HMM phylogenetics (Paper 016).
//!
//! Validates that barracuda primitives reproduce HMM matrix chain operations.
//! Uses `barracuda::stats::variance` on posterior probabilities and
//! `barracuda::linalg::solve_f64` for stationary distribution.
//!
//! Evolution path:
//! ```text
//! Python (hmmlearn) → Rust (hand-rolled forward/backward/Viterbi)
//!   → BarraCUDA CPU (stats::variance, linalg::solve_f64)
//!   → BarraCUDA GPU (batched matrix ops)
//! ```
//!
//! ## Provenance
//!
//! Python baseline: `control/hmm_phylo/hmm_phylo.py`
//! Rust baseline: `validate_hmm`

#![allow(clippy::cast_precision_loss, clippy::similar_names)]

use barracuda::device::WgpuDevice;
use neural_spring::hmm::Hmm;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let device = rt
        .block_on(async { WgpuDevice::new().await })
        .map(Arc::new)
        .expect("GPU device");

    let mut h = ValidationHarness::new("barracuda_hmm");

    validate_forward_log_likelihood(&mut h);
    validate_posterior_variance(&mut h);
    validate_stationary_distribution(&mut h, &device);
    validate_posterior_sums(&mut h);

    h.finish();
}

/// 2-state weather HMM: transition [[0.7,0.3],[0.4,0.6]], emission [[0.1,0.4,0.5],[0.6,0.3,0.1]],
/// initial [0.6, 0.4].
fn weather_hmm() -> Hmm {
    Hmm::new(
        vec![vec![0.7, 0.3], vec![0.4, 0.6]],
        vec![vec![0.1, 0.4, 0.5], vec![0.6, 0.3, 0.1]],
        vec![0.6, 0.4],
    )
}

fn validate_forward_log_likelihood(h: &mut ValidationHarness) {
    let hmm = weather_hmm();
    let mut rng = Rng::new(42);
    let (_, obs) = hmm.generate_sequence(50, &mut rng);

    let (_, log_lik) = hmm.forward(&obs);

    h.check_bool(
        &format!("forward: finite log-likelihood ({log_lik:.4})"),
        log_lik.is_finite(),
    );
    h.check_bool("forward: negative log-likelihood (prob < 1)", log_lik < 0.0);
}

fn validate_posterior_variance(h: &mut ValidationHarness) {
    let hmm = weather_hmm();
    let mut rng = Rng::new(42);
    let (_, obs) = hmm.generate_sequence(50, &mut rng);

    let gamma = hmm.posterior(&obs);
    let n = hmm.num_states();

    let state0_probs: Vec<f64> = gamma.chunks(n).map(|row| row[0]).collect();
    let state1_probs: Vec<f64> = gamma.chunks(n).map(|row| row[1]).collect();

    let var0 = barracuda::stats::correlation::variance(&state0_probs).unwrap_or(f64::NAN);
    let var1 = barracuda::stats::correlation::variance(&state1_probs).unwrap_or(f64::NAN);

    h.check_bool(
        &format!("posterior state0 variance finite ({var0:.6})"),
        var0.is_finite() && var0 >= 0.0,
    );
    h.check_bool(
        &format!("posterior state1 variance finite ({var1:.6})"),
        var1.is_finite() && var1 >= 0.0,
    );
}

/// Solve for stationary distribution: π A = π with sum(π) = 1.
/// Use (A^T - I) with last row replaced by \[1,1\], b = \[0, 1\].
fn validate_stationary_distribution(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let trans = [vec![0.7, 0.3], vec![0.4, 0.6]];
    let at: Vec<Vec<f64>> = (0..2)
        .map(|j| (0..2).map(|i| trans[i][j]).collect())
        .collect();

    let mut m = vec![0.0; 4];
    m[0] = at[0][0] - 1.0;
    m[1] = at[0][1];
    m[2] = 1.0;
    m[3] = 1.0;
    let b = vec![0.0, 1.0];

    match barracuda::linalg::solve_f64(device.clone(), &m, &b, 2) {
        Ok(pi) => {
            let sum: f64 = pi.iter().sum();
            h.check_abs(
                "stationary distribution sums to 1",
                sum,
                1.0,
                tolerances::CROSS_LANGUAGE,
            );
            h.check_bool("stationary π[0] in (0,1)", pi[0] > 0.0 && pi[0] < 1.0);
            h.check_bool("stationary π[1] in (0,1)", pi[1] > 0.0 && pi[1] < 1.0);

            let pi_a_0 = pi[0].mul_add(trans[0][0], pi[1] * trans[1][0]);
            let pi_a_1 = pi[0].mul_add(trans[0][1], pi[1] * trans[1][1]);
            h.check_abs(
                "πA ≈ π (element 0)",
                pi_a_0,
                pi[0],
                tolerances::CROSS_LANGUAGE,
            );
            h.check_abs(
                "πA ≈ π (element 1)",
                pi_a_1,
                pi[1],
                tolerances::CROSS_LANGUAGE,
            );
        }
        Err(e) => {
            h.check_bool(&format!("solve_f64 stationary [ERROR: {e}]"), false);
        }
    }
}

fn validate_posterior_sums(h: &mut ValidationHarness) {
    let hmm = weather_hmm();
    let mut rng = Rng::new(42);
    let (_, obs) = hmm.generate_sequence(50, &mut rng);

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
}
