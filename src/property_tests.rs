// SPDX-License-Identifier: AGPL-3.0-or-later

//! Property-based tests: verify invariants hold over many random inputs.
//!
//! Uses the project's own `Rng` for deterministic reproducibility.
//! Covers mathematical invariants that should hold for ALL valid inputs,
//! not just specific test cases.

use crate::eigh;
use crate::hmm::Hmm;
use crate::primitives;
use crate::rng::Rng;
use crate::spectral_commutativity;
use crate::transformer;

const N_TRIALS: usize = 50;

// ── Softmax invariants ──────────────────────────────────────────────

#[test]
fn softmax_always_sums_to_one() {
    let mut rng = Rng::new(1001);
    for _ in 0..N_TRIALS {
        let len = 2 + rng.usize(20);
        let input: Vec<f64> = (0..len).map(|_| rng.normal_params(0.0, 5.0)).collect();
        let sm = transformer::softmax(&input);
        let sum: f64 = sm.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "softmax sum = {sum}, input len = {len}"
        );
    }
}

#[test]
fn softmax_always_nonnegative() {
    let mut rng = Rng::new(1002);
    for _ in 0..N_TRIALS {
        let len = 2 + rng.usize(20);
        let input: Vec<f64> = (0..len).map(|_| rng.normal_params(0.0, 10.0)).collect();
        let sm = transformer::softmax(&input);
        for (i, &v) in sm.iter().enumerate() {
            assert!(v >= 0.0, "softmax[{i}] = {v} < 0");
        }
    }
}

// ── Sigmoid invariants ──────────────────────────────────────────────

#[test]
fn sigmoid_always_in_unit_interval() {
    let mut rng = Rng::new(1003);
    for _ in 0..(N_TRIALS * 10) {
        let x = rng.normal_params(0.0, 100.0);
        let s = primitives::sigmoid(x);
        assert!((0.0..=1.0).contains(&s), "sigmoid({x}) = {s} outside [0,1]");
    }
}

#[test]
fn sigmoid_symmetry_property() {
    let mut rng = Rng::new(1004);
    for _ in 0..N_TRIALS {
        let x = rng.normal_params(0.0, 10.0);
        let s_pos = primitives::sigmoid(x);
        let s_neg = primitives::sigmoid(-x);
        assert!(
            (s_pos + s_neg - 1.0).abs() < 1e-12,
            "σ({x}) + σ(-{x}) = {} ≠ 1",
            s_pos + s_neg
        );
    }
}

// ── Commutator antisymmetry ─────────────────────────────────────────

#[test]
fn commutator_antisymmetric_sweep() {
    let mut rng = Rng::new(1005);
    for _ in 0..N_TRIALS {
        let n = 4 + rng.usize(8);
        let a = spectral_commutativity::random_matrix(n, &mut rng);
        let b = spectral_commutativity::random_matrix(n, &mut rng);
        let ab = spectral_commutativity::commutator(&a, &b, n);
        let ba = spectral_commutativity::commutator(&b, &a, n);
        let sum_norm = spectral_commutativity::frobenius_norm(
            &ab.iter()
                .zip(ba.iter())
                .map(|(x, y)| x + y)
                .collect::<Vec<_>>(),
        );
        assert!(
            sum_norm < 1e-8,
            "[A,B] + [B,A] norm = {sum_norm} (should be ~0), n={n}"
        );
    }
}

// ── Symmetric matrices are normal ───────────────────────────────────

#[test]
fn symmetric_matrices_have_zero_distance_to_normal() {
    let mut rng = Rng::new(1006);
    for _ in 0..N_TRIALS {
        let n = 4 + rng.usize(12);
        let sym = spectral_commutativity::random_symmetric(n, &mut rng);
        let d = spectral_commutativity::distance_to_normal(&sym, n);
        assert!(
            d < 1e-8,
            "symmetric matrix d(normal) = {d} (should be ~0), n={n}"
        );
    }
}

// ── Eigensolver: eigenvalues of symmetric matrices are real ─────────

#[test]
fn eigh_eigenvalues_real_and_sorted() {
    let mut rng = Rng::new(1007);
    for _ in 0..N_TRIALS {
        let n = 3 + rng.usize(6);
        let sym = spectral_commutativity::random_symmetric(n, &mut rng);
        let result = eigh::eigh_householder_qr(&sym, n);
        assert_eq!(result.eigenvalues.len(), n);
        for (i, ev) in result.eigenvalues.iter().enumerate() {
            assert!(ev.is_finite(), "eigenvalue[{i}] not finite, n={n}");
        }
        for w in result.eigenvalues.windows(2) {
            assert!(
                w[0] <= w[1] + 1e-10,
                "eigenvalues not sorted: {} > {}, n={n}",
                w[0],
                w[1]
            );
        }
    }
}

// ── Eigensolver: trace equals sum of eigenvalues ────────────────────

#[test]
fn eigh_trace_equals_eigenvalue_sum() {
    let mut rng = Rng::new(1008);
    for _ in 0..N_TRIALS {
        let n = 3 + rng.usize(6);
        let sym = spectral_commutativity::random_symmetric(n, &mut rng);
        let trace: f64 = (0..n).map(|i| sym[i * n + i]).sum();
        let result = eigh::eigh_householder_qr(&sym, n);
        let ev_sum: f64 = result.eigenvalues.iter().sum();
        let scale = trace.abs().max(1.0);
        assert!(
            (trace - ev_sum).abs() / scale < 0.01,
            "trace={trace} vs eigenvalue_sum={ev_sum}, n={n}"
        );
    }
}

// ── HMM forward probabilities sum to 1 ─────────────────────────────

#[test]
fn hmm_forward_alpha_always_sums_to_one() {
    let mut rng = Rng::new(1010);
    for _ in 0..N_TRIALS {
        let n = 2 + rng.usize(4);
        let m = 2 + rng.usize(4);
        let hmm = random_valid_hmm(n, m, &mut rng);
        let obs: Vec<usize> = (0..20).map(|_| rng.usize(m)).collect();
        let fwd = hmm.forward_full(&obs);
        for t in 0..obs.len() {
            let sum: f64 = fwd.alpha_at(t).iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-8,
                "alpha sum = {sum} at t={t}, n={n}, m={m}"
            );
        }
    }
}

// ── HMM posterior probabilities sum to 1 ─────────────────────────────

#[test]
fn hmm_posterior_always_sums_to_one() {
    let mut rng = Rng::new(1011);
    for _ in 0..N_TRIALS {
        let n = 2 + rng.usize(3);
        let m = 2 + rng.usize(3);
        let hmm = random_valid_hmm(n, m, &mut rng);
        let obs: Vec<usize> = (0..15).map(|_| rng.usize(m)).collect();
        let gamma = hmm.posterior(&obs);
        for t in 0..obs.len() {
            let sum: f64 = gamma[t * n..(t + 1) * n].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "posterior sum = {sum} at t={t}, n={n}, m={m}"
            );
        }
    }
}

// ── HMM Viterbi path is valid ───────────────────────────────────────

#[test]
fn hmm_viterbi_path_always_valid() {
    let mut rng = Rng::new(1012);
    for _ in 0..N_TRIALS {
        let n = 2 + rng.usize(4);
        let m = 2 + rng.usize(4);
        let hmm = random_valid_hmm(n, m, &mut rng);
        let obs: Vec<usize> = (0..30).map(|_| rng.usize(m)).collect();
        let (path, log_prob) = hmm.viterbi(&obs);
        assert_eq!(path.len(), obs.len());
        assert!(log_prob.is_finite(), "Viterbi log_prob not finite");
        for (t, &s) in path.iter().enumerate() {
            assert!(s < n, "Viterbi state {s} >= n={n} at t={t}");
        }
    }
}

// ── RK4 energy conservation (harmonic oscillator) ───────────────────

#[test]
fn rk4_harmonic_oscillator_energy_bounded() {
    let mut rng = Rng::new(1013);
    for _ in 0..N_TRIALS {
        let x0 = rng.normal_params(0.0, 2.0);
        let v0 = rng.normal_params(0.0, 2.0);
        let dt = 0.01;
        let mut state = [x0, v0, 0.0, 0.0];
        let initial_energy = 0.5 * (x0 * x0 + v0 * v0);

        for _ in 0..1000 {
            state = primitives::rk4_step(&state, dt, |y| [y[1], -y[0], 0.0, 0.0]);
        }

        let final_energy = 0.5 * state[0].mul_add(state[0], state[1] * state[1]);
        let drift = ((final_energy - initial_energy) / initial_energy).abs();
        assert!(
            drift < 1e-6,
            "RK4 energy drift = {drift} (x0={x0}, v0={v0})"
        );
    }
}

// ── Matrix multiplication associativity ─────────────────────────────

#[test]
fn mat_mul_associative() {
    let mut rng = Rng::new(1014);
    for _ in 0..N_TRIALS {
        let n = 3 + rng.usize(5);
        let a = spectral_commutativity::random_matrix(n, &mut rng);
        let b = spectral_commutativity::random_matrix(n, &mut rng);
        let c = spectral_commutativity::random_matrix(n, &mut rng);

        let ab = spectral_commutativity::mat_mul(&a, &b, n);
        let bc = spectral_commutativity::mat_mul(&b, &c, n);
        let ab_c = spectral_commutativity::mat_mul(&ab, &c, n);
        let a_bc = spectral_commutativity::mat_mul(&a, &bc, n);

        let diff = spectral_commutativity::frobenius_norm(
            &ab_c
                .iter()
                .zip(a_bc.iter())
                .map(|(x, y)| x - y)
                .collect::<Vec<_>>(),
        );
        let scale = spectral_commutativity::frobenius_norm(&ab_c).max(1e-15);
        assert!(
            diff / scale < 1e-8,
            "(AB)C vs A(BC) relative diff = {}, n={n}",
            diff / scale
        );
    }
}

// ── Helper: generate a valid random HMM ────────────────────────────

fn random_valid_hmm(n: usize, m: usize, rng: &mut Rng) -> Hmm {
    let transition = random_stochastic_matrix(n, n, rng);
    let emission = random_stochastic_matrix(n, m, rng);
    let initial = random_distribution(n, rng);
    Hmm::from_flat(
        transition.into_iter().flatten().collect(),
        emission.into_iter().flatten().collect(),
        initial,
        n,
        m,
    )
}

fn random_stochastic_matrix(rows: usize, cols: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|_| {
            let raw: Vec<f64> = (0..cols).map(|_| rng.uniform().max(1e-6)).collect();
            let sum: f64 = raw.iter().sum();
            raw.iter().map(|x| x / sum).collect()
        })
        .collect()
}

fn random_distribution(n: usize, rng: &mut Rng) -> Vec<f64> {
    let raw: Vec<f64> = (0..n).map(|_| rng.uniform().max(1e-6)).collect();
    let sum: f64 = raw.iter().sum();
    raw.iter().map(|x| x / sum).collect()
}
