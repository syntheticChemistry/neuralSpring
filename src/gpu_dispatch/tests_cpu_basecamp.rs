// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU-path tests for baseCamp dispatcher operations.
//!
//! Covers `gpu_dispatch/basecamp.rs`: weight spectral analysis, numerical
//! Hessian, belief propagation, agent interaction graph, landscape analysis,
//! attention spectral analysis, and MLP signal propagation.

#![allow(clippy::expect_used)]

use super::*;
use crate::tolerances;

fn cpu() -> Dispatcher {
    Dispatcher::cpu_only()
}

// ── baseCamp (gpu_dispatch/basecamp.rs coverage) ─────────────

#[test]
fn basecamp_weight_spectral_analysis() {
    let d = cpu();
    let weights = vec![1.0, 0.0, 0.0, 1.0];
    let result = d.weight_spectral_analysis(&weights, 2, 2);
    assert_eq!(result.eigenvalues.len(), 4);
    assert!(result.mean_ipr.is_finite());
    assert!(result.level_spacing_ratio.is_finite());
    assert!(result.spectral_entropy.is_finite());
    assert!(result.mp_departure.is_finite());
}

#[test]
fn basecamp_numerical_hessian_quadratic() {
    let d = cpu();
    let quadratic = |x: &[f64]| -> f64 { x.iter().map(|&v| v * v).sum() };
    let point = vec![1.0, 2.0];
    let hess = d.numerical_hessian(quadratic, &point, tolerances::HESSIAN_FD_STEP);
    assert_eq!(hess.len(), 4);
    assert!(
        (hess[0] - 2.0).abs() < tolerances::OPTIMIZER_VALUE_AT_MIN,
        "d²/dx² of x² = 2"
    );
    assert!(
        (hess[3] - 2.0).abs() < tolerances::OPTIMIZER_VALUE_AT_MIN,
        "d²/dy² of y² = 2"
    );
    assert!(
        hess[1].abs() < tolerances::OPTIMIZER_VALUE_AT_MIN,
        "cross-term ≈ 0"
    );
}

#[test]
fn basecamp_belief_propagation_preserves_probability() {
    let d = cpu();
    let input = vec![0.25, 0.25, 0.25, 0.25];
    #[rustfmt::skip]
    let transition = vec![
        0.7, 0.3,
        0.6, 0.4,
        0.5, 0.5,
        0.4, 0.6,
    ];
    let dists = d.belief_propagation(&input, &[transition.as_slice()], &[2]);
    assert_eq!(dists.len(), 2);
    let final_sum: f64 = dists.last().expect("non-empty").iter().sum();
    assert!(
        (final_sum - 1.0).abs() < tolerances::CROSS_LANGUAGE,
        "output should be normalized, got sum={final_sum}"
    );
}

#[test]
fn basecamp_agent_interaction_graph() {
    let d = cpu();
    let positions = vec![0.0, 0.0, 1.0, 0.0, 5.0, 5.0];
    let adj = d.agent_interaction_graph(&positions, 3, 2, 2.0);
    assert_eq!(adj.len(), 9);
    assert!(adj[1] > 0.0, "agents 0-1 within range");
    assert!(adj[3] > 0.0, "symmetric: adj[1][0]");
    assert!(
        adj[2].abs() < tolerances::ZERO_DETECTION,
        "agents 0-2 outside range"
    );
}

#[test]
fn basecamp_landscape_analysis_quadratic() {
    let d = cpu();
    let quadratic = |x: &[f64]| -> f64 { x[0].mul_add(x[0], x[1] * x[1]) };
    let result = d.landscape_analysis(&quadratic, &[1.0, 1.0], tolerances::HESSIAN_FD_STEP, 0.1);
    assert!(result.loss.is_finite(), "loss must be finite");
    assert!(
        (result.loss - 2.0).abs() < tolerances::EXACT_F64,
        "f(1,1)=2, got {}",
        result.loss
    );
    assert!(result.flatness.is_finite(), "flatness must be finite");
    assert!(result.sharpness.is_finite(), "sharpness must be finite");
    assert_eq!(
        result.saddle_index, 0,
        "quadratic has no negative curvature"
    );
    assert!(result.spectral_gap.is_finite());
    assert_eq!(result.hessian_eigenvalues.len(), 2);
    for ev in &result.hessian_eigenvalues {
        assert!(
            (*ev - 2.0).abs() < tolerances::HESSIAN_FD_ABS,
            "eigenvalue should be ~2, got {ev}"
        );
    }
}

#[test]
fn basecamp_landscape_analysis_saddle() {
    let d = cpu();
    let saddle = |x: &[f64]| -> f64 { x[0].mul_add(x[0], -(x[1] * x[1])) };
    let result = d.landscape_analysis(&saddle, &[0.0, 0.0], tolerances::HESSIAN_FD_STEP, 0.1);
    assert_eq!(
        result.saddle_index, 1,
        "monkey saddle has 1 negative eigenvalue"
    );
}

#[test]
fn basecamp_attention_spectral_analysis() {
    let d = cpu();
    #[rustfmt::skip]
    let attention = vec![
        0.5, 0.5,
        0.5, 0.5,
    ];
    let result = d.attention_spectral_analysis(&attention, 2);
    assert_eq!(result.eigenvalues.len(), 2);
    assert!(result.mean_ipr.is_finite());
    assert!(result.level_spacing_ratio.is_finite());
    for ev in &result.eigenvalues {
        assert!(ev.is_finite(), "eigenvalue must be finite");
    }
}

#[test]
fn basecamp_attention_spectral_identity() {
    let d = cpu();
    #[rustfmt::skip]
    let identity = vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    let result = d.attention_spectral_analysis(&identity, 3);
    assert_eq!(result.eigenvalues.len(), 3);
    for ev in &result.eigenvalues {
        assert!(
            (*ev - 1.0).abs() < tolerances::EIGH_JACOBI_EIGENVALUE,
            "identity eigenvalue should be ~1, got {ev}"
        );
    }
}

#[test]
fn basecamp_mlp_signal_propagation() {
    let d = cpu();
    let input = vec![1.0, 0.5, -0.3];
    #[rustfmt::skip]
    let w0 = vec![
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
    ];
    let variances = d.mlp_signal_propagation(&input, &[w0.as_slice()], &[2]);
    assert_eq!(variances.len(), 2, "input variance + 1 layer variance");
    for v in &variances {
        assert!(v.is_finite(), "variance must be finite");
        assert!(*v >= 0.0, "variance must be non-negative");
    }
}

#[test]
fn basecamp_mlp_signal_propagation_deep() {
    let d = cpu();
    let input = vec![1.0, 0.5];
    #[rustfmt::skip]
    let w0 = vec![0.5, 0.5, 0.3, 0.7, 0.1, 0.9];
    #[rustfmt::skip]
    let w1 = vec![0.4, 0.4, 0.4, 0.6, 0.6, 0.6];
    let variances = d.mlp_signal_propagation(&input, &[w0.as_slice(), w1.as_slice()], &[3, 2]);
    assert_eq!(variances.len(), 3, "input + 2 layers");
    assert!(variances[0] > 0.0, "non-zero input has positive variance");
}

#[test]
fn basecamp_belief_propagation_chain() {
    let d = cpu();
    let input = vec![0.5, 0.5];
    #[rustfmt::skip]
    let t1 = vec![0.9, 0.1, 0.2, 0.8];
    #[rustfmt::skip]
    let t2 = vec![0.7, 0.3, 0.4, 0.6];
    let dists = d.belief_propagation(&input, &[t1.as_slice(), t2.as_slice()], &[2, 2]);
    assert_eq!(dists.len(), 3, "input + 2 transitions");
    for (i, dist) in dists.iter().enumerate() {
        let sum: f64 = dist.iter().sum();
        assert!(
            (sum - 1.0).abs() < tolerances::CROSS_LANGUAGE,
            "distribution {i} not normalized: sum={sum}"
        );
    }
}

#[test]
fn basecamp_belief_propagation_identity_transition() {
    let d = cpu();
    let input = vec![0.3, 0.7];
    #[rustfmt::skip]
    let identity = vec![1.0, 0.0, 0.0, 1.0];
    let dists = d.belief_propagation(&input, &[identity.as_slice()], &[2]);
    let output = &dists[1];
    assert!(
        (output[0] - 0.3).abs() < tolerances::CROSS_LANGUAGE,
        "identity transition preserves distribution"
    );
    assert!(
        (output[1] - 0.7).abs() < tolerances::CROSS_LANGUAGE,
        "identity transition preserves distribution"
    );
}

#[test]
fn basecamp_agent_interaction_graph_no_connections() {
    let d = cpu();
    let positions = vec![0.0, 0.0, 100.0, 100.0];
    let adj = d.agent_interaction_graph(&positions, 2, 2, 1.0);
    assert_eq!(adj.len(), 4);
    assert!(
        adj.iter().all(|&v| v.abs() < tolerances::ZERO_DETECTION),
        "agents far apart should have no connections"
    );
}

#[test]
fn basecamp_agent_interaction_graph_symmetric() {
    let d = cpu();
    let positions = vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5];
    let adj = d.agent_interaction_graph(&positions, 3, 2, 2.0);
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (adj[i * 3 + j] - adj[j * 3 + i]).abs() < tolerances::ZERO_DETECTION,
                "adjacency matrix must be symmetric"
            );
        }
    }
}
