// SPDX-License-Identifier: AGPL-3.0-or-later

//! Property-based tests: verify invariants hold over many random inputs.
//!
//! Uses the project's own `Rng` for deterministic reproducibility.
//! Covers mathematical invariants that should hold for ALL valid inputs,
//! not just specific test cases.

use std::time::Duration;

use crate::eigh;
use crate::hmm::Hmm;
use crate::primitives;
use crate::rng::Rng;
use crate::spectral_commutativity;
use crate::tolerances;
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
            (sum - 1.0).abs() < tolerances::CROSS_LANGUAGE,
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
            (s_pos + s_neg - 1.0).abs() < tolerances::EXACT_F64,
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
            sum_norm < tolerances::HMM_POSTERIOR_SUM,
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
            d < tolerances::HMM_POSTERIOR_SUM,
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
                w[0] <= w[1] + tolerances::CROSS_LANGUAGE,
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
                (sum - 1.0).abs() < tolerances::HMM_POSTERIOR_SUM,
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
                (sum - 1.0).abs() < tolerances::SPECIAL_FUNCTION_F64,
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
            drift < tolerances::SPECIAL_FUNCTION_F64,
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
            diff / scale < tolerances::HMM_POSTERIOR_SUM,
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

// ── GELU monotonicity (for x > 0) ──────────────────────────────────

#[test]
fn gelu_monotone_for_positive_inputs() {
    let mut rng = Rng::new(1015);
    for _ in 0..N_TRIALS {
        let x1 = rng.uniform() * 10.0;
        let x2 = rng.uniform().mul_add(5.0, x1);
        let g1 = transformer::gelu(x1);
        let g2 = transformer::gelu(x2);
        assert!(
            g2 >= g1 - tolerances::EXACT_F64,
            "GELU not monotone: gelu({x1})={g1} > gelu({x2})={g2}"
        );
    }
}

// ── Layer norm invariant: zero mean, unit variance ──────────────────

#[test]
fn layer_norm_zero_mean_unit_var() {
    use crate::coral_forge::layer_norm;
    let mut rng = Rng::new(1016);
    for _ in 0..N_TRIALS {
        let dim = 4 + rng.usize(50);
        let rows = 1;
        let input: Vec<f64> = (0..dim).map(|_| rng.normal_params(5.0, 3.0)).collect();
        let gamma = vec![1.0; dim];
        let beta = vec![0.0; dim];
        let normed = layer_norm(&input, rows, dim, &gamma, &beta, 1e-5);
        #[expect(clippy::cast_precision_loss, reason = "dim ≤ 64 fits in f64 mantissa")]
        let d = dim as f64;
        let mean: f64 = normed.iter().sum::<f64>() / d;
        let var: f64 = normed.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / d;
        assert!(mean.abs() < 0.01, "layer_norm mean = {mean} (should be ~0)");
        assert!(
            (var - 1.0).abs() < 0.05,
            "layer_norm var = {var} (should be ~1)"
        );
    }
}

// ── Shannon entropy non-negativity ──────────────────────────────────

#[test]
fn shannon_entropy_always_nonnegative() {
    let mut rng = Rng::new(1017);
    for _ in 0..N_TRIALS {
        let len = 2 + rng.usize(10);
        let data: Vec<f64> = (0..len).map(|_| rng.uniform().max(1e-12)).collect();
        let h = primitives::shannon_entropy(&data);
        assert!(h >= -tolerances::EXACT_F64, "entropy = {h} < 0");
    }
}

// ── Hill activation bounded in [0, vmax] ────────────────────────────

#[test]
fn hill_activation_bounded() {
    let mut rng = Rng::new(1018);
    for _ in 0..N_TRIALS {
        let x = rng.uniform() * 10.0;
        let vmax = rng.uniform().mul_add(5.0, 1.0);
        let k = rng.uniform().mul_add(2.0, 0.1);
        let n = rng.uniform().mul_add(4.0, 1.0);
        let h = primitives::hill_activation(x, vmax, k, n);
        assert!(
            h >= -tolerances::EXACT_F64 && h <= vmax + tolerances::EXACT_F64,
            "hill({x}, vmax={vmax}, k={k}, n={n}) = {h} outside [0, {vmax}]"
        );
    }
}

// ── FFT energy invariants ────────────────────────────────────────────

#[test]
fn fft_energy_nonnegative_sweep() {
    use crate::fft;
    let mut rng = Rng::new(1019);
    for _ in 0..N_TRIALS {
        let n = 4 + rng.usize(60);
        let signal: Vec<f64> = (0..n * 2).map(|_| rng.normal()).collect();
        let energy = fft::complex_energy_f64(&signal);
        assert!(energy >= 0.0, "complex energy must be nonneg, got {energy}");
    }
}

#[test]
fn fft_cosine_signal_has_bounded_energy() {
    use crate::fft;
    for n in [8, 16, 32, 64] {
        for freq in 1..n / 2 {
            let signal = fft::cosine_signal_f64(n, freq);
            let energy = fft::complex_energy_f64(&signal);
            assert!(
                energy.is_finite(),
                "cosine(n={n}, freq={freq}) energy not finite"
            );
        }
    }
}

// ── Chaos / fault injection: numerical stability under extreme inputs ─

#[test]
fn softmax_stable_with_large_inputs() {
    let large: Vec<f64> = vec![1e10, 1e10 + 1.0, 1e10 - 1.0];
    let sm = transformer::softmax(&large);
    let sum: f64 = sm.iter().sum();
    assert!(
        (sum - 1.0).abs() < tolerances::CROSS_LANGUAGE,
        "softmax unstable with large inputs: sum = {sum}"
    );
    assert!(
        sm.iter().all(|v| v.is_finite()),
        "softmax produced non-finite values"
    );
}

#[test]
fn softmax_stable_with_extreme_negative() {
    let extreme: Vec<f64> = vec![-1e15, -1e15, -1e15];
    let sm = transformer::softmax(&extreme);
    let sum: f64 = sm.iter().sum();
    assert!(
        (sum - 1.0).abs() < tolerances::CROSS_LANGUAGE,
        "softmax unstable with extreme negatives: sum = {sum}"
    );
}

#[test]
fn sigmoid_stable_at_extremes() {
    assert!((primitives::sigmoid(1e10) - 1.0).abs() < tolerances::EXACT_F64);
    assert!(primitives::sigmoid(-1e10).abs() < tolerances::EXACT_F64);
    assert!((primitives::sigmoid(0.0) - 0.5).abs() < tolerances::EXACT_F64);
}

#[test]
fn gelu_finite_for_extreme_inputs() {
    for &x in &[-1e10, -1e5, 0.0, 1e5, 1e10] {
        let g = transformer::gelu(x);
        assert!(g.is_finite(), "gelu({x}) = {g} is not finite");
    }
}

#[test]
fn eigh_stable_with_near_singular_matrix() {
    let n = 5;
    let mut mat = vec![0.0; n * n];
    for i in 0..n {
        #[expect(clippy::cast_precision_loss, reason = "i ≤ 4 fits in f64 mantissa")]
        let scale = (i + 1) as f64;
        mat[i * n + i] = 1e-15 * scale;
    }
    let result = eigh::eigh_householder_qr(&mat, n);
    assert_eq!(result.eigenvalues.len(), n);
    for ev in &result.eigenvalues {
        assert!(
            ev.is_finite(),
            "eigenvalue not finite for near-singular matrix"
        );
    }
}

#[test]
fn hmm_forward_stable_with_near_zero_emissions() {
    let hmm = Hmm::new(
        vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        vec![vec![1e-300, 1.0 - 1e-300], vec![1.0 - 1e-300, 1e-300]],
        vec![0.5, 0.5],
    );
    let obs = &[0, 1, 0, 1];
    let (_, log_lik) = hmm.forward(obs);
    assert!(
        log_lik.is_finite(),
        "HMM forward unstable: log_lik = {log_lik}"
    );
}

// ── IPC resilience: RetryPolicy invariants ──────────────────────────

#[test]
fn retry_policy_delay_never_exceeds_max() {
    use crate::ipc_resilience::RetryPolicy;

    let configs: &[(u64, u64, f64)] = &[
        (50, 2000, 2.0),
        (1, 100, 10.0),
        (100, 500, 1.5),
        (10, 10, 3.0),
    ];
    for &(init_ms, max_ms, mult) in configs {
        let p = RetryPolicy {
            max_retries: 20,
            initial_delay: Duration::from_millis(init_ms),
            max_delay: Duration::from_millis(max_ms),
            multiplier: mult,
        };
        for attempt in 0..100 {
            let delay = p.delay_for_attempt(attempt);
            assert!(
                delay <= p.max_delay,
                "delay {delay:?} > max {:?} at attempt {attempt} (init={init_ms}ms, mult={mult})",
                p.max_delay
            );
        }
    }
}

// ── IPC resilience: CircuitBreaker state machine ────────────────────

#[test]
fn circuit_breaker_state_machine_sweep() {
    use crate::ipc_resilience::{CircuitBreaker, CircuitState};

    let mut rng = Rng::new(2001);
    for _ in 0..N_TRIALS {
        let threshold = 1 + (rng.usize(5) as u32);
        let cb = CircuitBreaker::new(threshold, Duration::from_millis(0));

        assert_eq!(cb.state(), CircuitState::Closed, "initial state");
        assert!(cb.is_allowed(), "initial is_allowed");

        for i in 0..threshold.saturating_sub(1) {
            cb.record_failure();
            assert_eq!(
                cb.state(),
                CircuitState::Closed,
                "still Closed after {i} failures (threshold={threshold})"
            );
        }

        cb.record_failure();
        let after_threshold = cb.state();
        assert!(
            after_threshold == CircuitState::Open || after_threshold == CircuitState::HalfOpen,
            "should be Open or HalfOpen after {threshold} failures, got {after_threshold:?}"
        );

        cb.record_success();
        assert_eq!(
            cb.state(),
            CircuitState::Closed,
            "should reset to Closed after success"
        );
        assert!(cb.is_allowed(), "should be allowed after reset");
    }
}

// ── IPC resilience: CircuitBreaker never panics under rapid cycling ─

#[test]
fn circuit_breaker_rapid_cycle_no_panic() {
    use crate::ipc_resilience::CircuitBreaker;

    let cb = CircuitBreaker::new(2, Duration::from_millis(0));
    let mut rng = Rng::new(2002);
    for _ in 0..1000 {
        if rng.uniform() < 0.3 {
            cb.record_success();
        } else {
            cb.record_failure();
        }
        let _ = cb.state();
        let _ = cb.is_allowed();
    }
}
