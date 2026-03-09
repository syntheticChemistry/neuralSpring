// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU-path HMM tests for [`Dispatcher`].

use super::*;
use crate::tolerances;

fn cpu() -> Dispatcher {
    Dispatcher::cpu_only()
}

#[test]
fn cpu_hmm_backward_step_basic() {
    let d = cpu();
    let beta_next = vec![1.0, 1.0];
    #[rustfmt::skip]
    let trans = vec![0.7, 0.3, 0.4, 0.6];
    let emit = vec![0.5, 0.5];
    let result = d.hmm_backward_step(&beta_next, &trans, &emit, 1.0, 2);
    assert_eq!(result.len(), 2);
    assert!((result[0] - 0.5).abs() < tolerances::EXACT_F64);
    assert!((result[1] - 0.5).abs() < tolerances::EXACT_F64);
}

#[test]
fn cpu_hmm_backward_step_zero_scale() {
    let d = cpu();
    let result = d.hmm_backward_step(&[1.0], &[1.0], &[1.0], 0.0, 1);
    assert!(result[0].is_finite(), "zero scale should use guard");
}

#[test]
fn cpu_hmm_viterbi_step() {
    let d = cpu();
    let delta_prev = vec![0.0_f64.ln(), (-1.0_f64).exp().ln()];
    #[rustfmt::skip]
    let log_trans = vec![
        0.7_f64.ln(), 0.3_f64.ln(),
        0.4_f64.ln(), 0.6_f64.ln(),
    ];
    let log_emit = vec![0.6_f64.ln(), 0.4_f64.ln()];
    let (delta, psi) = d.hmm_viterbi_step(&delta_prev, &log_trans, &log_emit, 2);
    assert_eq!(delta.len(), 2);
    assert_eq!(psi.len(), 2);
}

#[test]
fn cpu_hmm_forward_step_basic() {
    let d = cpu();
    let alpha = vec![0.6, 0.4];
    #[rustfmt::skip]
    let trans = vec![0.7, 0.3, 0.4, 0.6];
    let emit = vec![0.5, 0.5];
    let (new_alpha, scale) = d.hmm_forward_step(&alpha, &trans, &emit, 2);
    assert_eq!(new_alpha.len(), 2);
    assert!(scale > 0.0, "scale must be positive");
    let sum: f64 = new_alpha.iter().sum();
    assert!(
        (sum - 1.0).abs() < tolerances::EXACT_F64,
        "forward step normalizes"
    );
}

#[test]
fn cpu_hmm_forward_chain_basic() {
    let d = cpu();
    let initial = vec![0.6, 0.4];
    #[rustfmt::skip]
    let transition = vec![0.7, 0.3, 0.4, 0.6];
    #[rustfmt::skip]
    let emission = vec![0.5, 0.4, 0.1, 0.1, 0.3, 0.6];
    let obs = vec![0, 1, 2, 0];
    let ll = d.hmm_forward_chain(&initial, &transition, &emission, &obs, 2, 3);
    assert!(ll.is_finite(), "log-likelihood must be finite");
    assert!(ll < 0.0, "log-likelihood should be negative");
}

#[test]
fn cpu_hmm_viterbi_chain_basic() {
    let d = cpu();
    let initial = vec![0.6, 0.4];
    #[rustfmt::skip]
    let transition = vec![0.7, 0.3, 0.4, 0.6];
    #[rustfmt::skip]
    let emission = vec![0.5, 0.4, 0.1, 0.1, 0.3, 0.6];
    let obs = vec![0, 1, 2, 0];
    let (path, log_prob) = d.hmm_viterbi_chain(&initial, &transition, &emission, &obs, 2, 3);
    assert_eq!(path.len(), 4);
    assert!(log_prob.is_finite());
    for &s in &path {
        assert!(s < 2, "state must be < n_states");
    }
}
