// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: HMM forward/backward/Viterbi (Paper 016).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/hmm_phylo/hmm_phylo.py`
//! Paper: Liu et al. (2014) `PLoS` Comp Bio 10:e1003649.
//! Command: `python3 control/hmm_phylo/hmm_phylo.py`
//! Result: 10/10 PASS (weather HMM + phylo HMM, seed=42)

#![allow(clippy::cast_precision_loss)]

use neural_spring::hmm::Hmm;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

#[allow(clippy::too_many_lines)]
fn main() {
    let mut h = ValidationHarness::new("hmm");

    // Weather HMM (classic 2-state, 3-observation)
    let hmm = Hmm::new(
        vec![vec![0.7, 0.3], vec![0.4, 0.6]],
        vec![vec![0.1, 0.4, 0.5], vec![0.6, 0.3, 0.1]],
        vec![0.6, 0.4],
    );

    // Part 1: Forward algorithm
    let obs = vec![0, 1, 2, 0, 2];
    let (alpha, log_lik) = hmm.forward(&obs);

    h.check_bool(
        &format!("forward: finite negative log-lik ({log_lik:.6})"),
        log_lik.is_finite() && log_lik < 0.0,
    );

    for (t, row) in alpha.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        h.check_abs(
            &format!("forward alpha[{t}] sums to 1"),
            sum,
            1.0,
            tolerances::CROSS_LANGUAGE,
        );
    }

    // Part 2: Viterbi
    let mut rng = Rng::new(42);
    let (true_states, gen_obs) = hmm.generate_sequence(100, &mut rng);
    let (viterbi_path, viterbi_prob) = hmm.viterbi(&gen_obs);

    let accuracy = viterbi_path
        .iter()
        .zip(true_states.iter())
        .filter(|(a, b)| a == b)
        .count() as f64
        / gen_obs.len() as f64;
    let chance = 1.0 / hmm.num_states() as f64;

    h.check_lower(
        &format!(
            "Viterbi accuracy ({accuracy:.4}) > chance+0.05 ({:.4})",
            chance + 0.05
        ),
        accuracy,
        chance + 0.05,
    );

    h.check_bool(
        &format!("Viterbi finite log-prob ({viterbi_prob:.4})"),
        viterbi_prob.is_finite(),
    );

    // Part 3: Posterior
    let gamma = hmm.posterior(&gen_obs);
    for (t, row) in gamma.iter().enumerate().take(5) {
        let sum: f64 = row.iter().sum();
        h.check_abs(
            &format!("posterior gamma[{t}] sums to 1"),
            sum,
            1.0,
            tolerances::HMM_POSTERIOR_SUM,
        );
    }

    let posterior_acc = gamma
        .iter()
        .zip(true_states.iter())
        .filter(|(row, &s)| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i)
                == s
        })
        .count() as f64
        / gen_obs.len() as f64;

    h.check_bool(
        &format!(
            "posterior accuracy ({posterior_acc:.4}) >= Viterbi-0.05 ({:.4})",
            accuracy - 0.05
        ),
        posterior_acc >= accuracy - 0.05,
    );

    // Part 4: Phylo HMM (genomic scale)
    let phylo = create_phylo_hmm(4, 4, 42);
    let (true_phylo, phylo_obs) = phylo.generate_sequence(5000, &mut rng);
    let (_, phylo_loglik) = phylo.forward(&phylo_obs);
    let (phylo_path, _) = phylo.viterbi(&phylo_obs);

    h.check_bool(
        &format!("phylo forward: no underflow (log-lik={phylo_loglik:.2})"),
        phylo_loglik.is_finite(),
    );

    let phylo_acc = phylo_path
        .iter()
        .zip(true_phylo.iter())
        .filter(|(a, b)| a == b)
        .count() as f64
        / phylo_obs.len() as f64;
    let phylo_chance = 1.0 / phylo.num_states() as f64;

    h.check_lower(
        &format!(
            "phylo Viterbi ({phylo_acc:.4}) > chance+0.02 ({:.4})",
            phylo_chance + 0.02
        ),
        phylo_acc,
        phylo_chance + 0.02,
    );

    // Part 5: GEMM equivalence (manual forward matches library)
    let obs_short = &gen_obs[..10];
    let (alpha_lib, _) = hmm.forward(obs_short);
    let alpha_manual = manual_forward(&hmm, obs_short);
    let max_diff = alpha_lib
        .iter()
        .zip(alpha_manual.iter())
        .flat_map(|(a, b)| a.iter().zip(b.iter()))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        &format!("GEMM chain matches forward (diff={max_diff:.2e})"),
        max_diff,
        0.0,
        tolerances::EXACT_F64,
    );

    h.finish();
}

fn create_phylo_hmm(n_states: usize, n_symbols: usize, seed: u64) -> Hmm {
    let mut rng = Rng::new(seed);
    let transition = dirichlet_matrix(&mut rng, n_states, n_states, 10.0);
    let emission = dirichlet_matrix(&mut rng, n_states, n_symbols, 2.0);
    let initial = dirichlet_vec(&mut rng, n_states, 5.0);
    Hmm::new(transition, emission, initial)
}

fn dirichlet_matrix(rng: &mut Rng, rows: usize, cols: usize, alpha: f64) -> Vec<Vec<f64>> {
    (0..rows).map(|_| dirichlet_vec(rng, cols, alpha)).collect()
}

fn dirichlet_vec(rng: &mut Rng, n: usize, alpha: f64) -> Vec<f64> {
    let raw: Vec<f64> = (0..n)
        .map(|_| {
            let g = gamma_sample(rng, alpha);
            g.max(1e-10)
        })
        .collect();
    let sum: f64 = raw.iter().sum();
    raw.iter().map(|x| x / sum).collect()
}

#[allow(clippy::many_single_char_names)]
fn gamma_sample(rng: &mut Rng, alpha: f64) -> f64 {
    if alpha < 1.0 {
        return gamma_sample(rng, alpha + 1.0) * rng.uniform().powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = rng.normal();
        let v = (1.0 + c * x).powi(3);
        if v > 0.0 {
            let u = rng.uniform().max(1e-300);
            if u.ln() < 0.5f64.mul_add(x * x, d - d * v + d * v.ln()) {
                return d * v;
            }
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn manual_forward(hmm: &Hmm, obs: &[usize]) -> Vec<Vec<f64>> {
    let n = hmm.num_states();
    let mut alpha = vec![vec![0.0; n]; obs.len()];

    for j in 0..n {
        alpha[0][j] = hmm.initial[j] * hmm.emission[j][obs[0]];
    }
    let s0: f64 = alpha[0].iter().sum();
    if s0 > 0.0 {
        for v in &mut alpha[0] {
            *v /= s0;
        }
    }

    for t in 1..obs.len() {
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..n {
                sum += alpha[t - 1][i] * hmm.transition[i][j];
            }
            alpha[t][j] = sum * hmm.emission[j][obs[t]];
        }
        let s: f64 = alpha[t].iter().sum();
        if s > 0.0 {
            for v in &mut alpha[t] {
                *v /= s;
            }
        }
    }
    alpha
}
