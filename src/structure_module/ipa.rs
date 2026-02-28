// SPDX-License-Identifier: AGPL-3.0-or-later

//! Invariant Point Attention (Algorithm 22).

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use super::frame::{apply_frame, get_frame};
use crate::sovereign_folding::softmax_rows;

/// Configuration for Invariant Point Attention (Algorithm 22).
///
/// Groups the dimensional and weighting parameters that are constant
/// across a single IPA computation, keeping data slices as function args.
#[derive(Debug, Clone)]
pub struct IpaConfig {
    /// Number of residues in the protein.
    pub n_res: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Scalar channels per head (D).
    pub head_dim: usize,
    /// Number of 3D query/key points per head.
    pub n_points: usize,
    /// Scalar attention weight multiplier.
    pub w_l: f64,
    /// Pair bias weight multiplier.
    pub w_c: f64,
    /// Point distance weight multiplier.
    pub w_p: f64,
    /// Per-head point weight (controls SE(3) distance sensitivity).
    pub gamma: f64,
}

/// IPA attention scores (Algorithm 22, score computation).
///
/// Computes the three-term IPA logit:
///
/// ```text
/// a[h,i,j] = w_L * Σ_d Q[i,h,d]*K[j,h,d] / √c
///          + w_C * pair_bias[h,i,j]
///          + w_P * (-γ/2) * Σ_p ||T_i(q_p[i,h,p]) - T_j(k_p[j,h,p])||²
/// ```
///
/// - `q_scalar`/`k_scalar`: `[N, H, D]` — standard attention projections
/// - `pair_bias`: `[H, N, N]` — bias from pair representation
/// - `q_points`/`k_points`: `[N, H, P, 3]` — 3D query/key points (local frame)
/// - `frames`: `[N, 12]` — backbone frames (rot9 + trans3)
/// - `cfg`: [`IpaConfig`] — dimensional and weighting parameters
///
/// Returns `[H, N, N]` (pre-softmax logits).
///
/// # Panics
///
/// Panics if slice lengths don't match declared dimensions.
#[must_use]
pub fn ipa_scores(
    q_scalar: &[f64],
    k_scalar: &[f64],
    pair_bias: &[f64],
    q_points: &[f64],
    k_points: &[f64],
    frames: &[f64],
    cfg: &IpaConfig,
) -> Vec<f64> {
    let IpaConfig {
        n_res,
        n_heads,
        head_dim,
        n_points,
        w_l,
        w_c,
        w_p,
        gamma,
    } = *cfg;

    assert_eq!(q_scalar.len(), n_res * n_heads * head_dim);
    assert_eq!(k_scalar.len(), n_res * n_heads * head_dim);
    assert_eq!(pair_bias.len(), n_heads * n_res * n_res);
    assert_eq!(q_points.len(), n_res * n_heads * n_points * 3);
    assert_eq!(k_points.len(), n_res * n_heads * n_points * 3);
    assert_eq!(frames.len(), n_res * 12);

    let scale = (head_dim as f64).sqrt();
    let point_coeff = -gamma / 2.0;
    let mut scores = vec![0.0; n_heads * n_res * n_res];

    for h in 0..n_heads {
        for i in 0..n_res {
            let (ri, ti) = get_frame(frames, i);

            for j in 0..n_res {
                let (rj, tj) = get_frame(frames, j);

                let mut dot = 0.0_f64;
                for d in 0..head_dim {
                    let qi = q_scalar[(i * n_heads + h) * head_dim + d];
                    let ki = k_scalar[(j * n_heads + h) * head_dim + d];
                    dot = qi.mul_add(ki, dot);
                }
                let scalar_term = w_l * dot / scale;

                let pair_term = w_c * pair_bias[(h * n_res + i) * n_res + j];

                let mut point_dist_sq = 0.0_f64;
                for p in 0..n_points {
                    let qp_base = ((i * n_heads + h) * n_points + p) * 3;
                    let kp_base = ((j * n_heads + h) * n_points + p) * 3;
                    let qp_local = [
                        q_points[qp_base],
                        q_points[qp_base + 1],
                        q_points[qp_base + 2],
                    ];
                    let kp_local = [
                        k_points[kp_base],
                        k_points[kp_base + 1],
                        k_points[kp_base + 2],
                    ];

                    let qp_global = apply_frame(&ri, &ti, &qp_local);
                    let kp_global = apply_frame(&rj, &tj, &kp_local);

                    for xyz in 0..3 {
                        let diff = qp_global[xyz] - kp_global[xyz];
                        point_dist_sq += diff * diff;
                    }
                }
                let point_term = w_p * point_coeff * point_dist_sq;

                scores[(h * n_res + i) * n_res + j] = scalar_term + pair_term + point_term;
            }
        }
    }
    scores
}

/// Full IPA: scores → softmax → weighted sum of scalar values.
///
/// Returns `[N, H, D]` (scalar output, before linear projection).
///
/// # Panics
///
/// Panics if slice lengths don't match declared dimensions.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn ipa_apply(
    q_scalar: &[f64],
    k_scalar: &[f64],
    v_scalar: &[f64],
    pair_bias: &[f64],
    q_points: &[f64],
    k_points: &[f64],
    frames: &[f64],
    cfg: &IpaConfig,
) -> Vec<f64> {
    let IpaConfig {
        n_res,
        n_heads,
        head_dim,
        ..
    } = *cfg;

    assert_eq!(v_scalar.len(), n_res * n_heads * head_dim);

    let logits = ipa_scores(
        q_scalar, k_scalar, pair_bias, q_points, k_points, frames, cfg,
    );

    let weights = softmax_rows(&logits, n_heads * n_res, n_res);

    let mut out = vec![0.0; n_res * n_heads * head_dim];
    for h in 0..n_heads {
        for i in 0..n_res {
            for d in 0..head_dim {
                let mut acc = 0.0_f64;
                for j in 0..n_res {
                    let w = weights[(h * n_res + i) * n_res + j];
                    let v = v_scalar[(j * n_heads + h) * head_dim + d];
                    acc = w.mul_add(v, acc);
                }
                out[(i * n_heads + h) * head_dim + d] = acc;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn make_ipa_cfg(n: usize, h: usize, d: usize, p: usize) -> IpaConfig {
        IpaConfig {
            n_res: n,
            n_heads: h,
            head_dim: d,
            n_points: p,
            w_l: 1.0,
            w_c: 1.0,
            w_p: 1.0,
            gamma: 1.0,
        }
    }

    fn identity_frames(n: usize) -> Vec<f64> {
        let mut frames = vec![0.0; n * 12];
        for i in 0..n {
            frames[i * 12] = 1.0; // R[0,0]
            frames[i * 12 + 4] = 1.0; // R[1,1]
            frames[i * 12 + 8] = 1.0; // R[2,2]
        }
        frames
    }

    #[test]
    fn ipa_scores_shape() {
        let (n, h, d, p) = (4, 2, 4, 3);
        let q_s = vec![0.1; n * h * d];
        let k_s = vec![0.1; n * h * d];
        let bias = vec![0.0; h * n * n];
        let q_p = vec![0.0; n * h * p * 3];
        let k_p = vec![0.0; n * h * p * 3];
        let frames = identity_frames(n);
        let cfg = make_ipa_cfg(n, h, d, p);

        let scores = ipa_scores(&q_s, &k_s, &bias, &q_p, &k_p, &frames, &cfg);
        assert_eq!(scores.len(), h * n * n);
    }

    #[test]
    fn ipa_scores_zero_points_reduces_to_scalar_plus_bias() {
        let (n, h, d, p) = (3, 1, 2, 0);
        let q_s = vec![1.0; n * h * d];
        let k_s = vec![1.0; n * h * d];
        let bias = vec![0.5; h * n * n];
        let q_p: Vec<f64> = vec![];
        let k_p: Vec<f64> = vec![];
        let frames = identity_frames(n);
        let cfg = IpaConfig {
            w_p: 0.0,
            ..make_ipa_cfg(n, h, d, p)
        };

        let scores = ipa_scores(&q_s, &k_s, &bias, &q_p, &k_p, &frames, &cfg);
        let expected_scalar = (d as f64) / (d as f64).sqrt();
        for &s in &scores {
            assert!(
                (s - expected_scalar - 0.5).abs() < EPS,
                "w_p=0 → pure scalar+bias, got {s}"
            );
        }
    }

    #[test]
    fn ipa_scores_point_distance_self_is_zero() {
        let (n, h, d, p) = (2, 1, 2, 2);
        let q_s = vec![0.0; n * h * d];
        let k_s = vec![0.0; n * h * d];
        let bias = vec![0.0; h * n * n];
        let points = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let frames = identity_frames(n);
        let cfg = IpaConfig {
            w_l: 0.0,
            w_c: 0.0,
            ..make_ipa_cfg(n, h, d, p)
        };

        let scores = ipa_scores(&q_s, &k_s, &bias, &points, &points, &frames, &cfg);
        for i in 0..n {
            let diag = scores[i * n + i];
            assert!(
                diag.abs() < EPS,
                "self-distance = 0 → score = 0, got {diag}"
            );
        }
    }

    #[test]
    fn ipa_apply_output_finite() {
        let (n, h, d, p) = (4, 2, 4, 2);
        let q_s: Vec<f64> = (0..n * h * d).map(|i| (i as f64) * 0.01).collect();
        let k_s: Vec<f64> = (0..n * h * d)
            .map(|i| (i as f64).mul_add(0.02, -0.3))
            .collect();
        let v_s: Vec<f64> = (0..n * h * d)
            .map(|i| (i as f64).mul_add(0.03, -0.1))
            .collect();
        let bias = vec![0.0; h * n * n];
        let q_p: Vec<f64> = (0..n * h * p * 3).map(|i| (i as f64) * 0.01).collect();
        let k_p: Vec<f64> = (0..n * h * p * 3).map(|i| (i as f64) * 0.01).collect();
        let frames = identity_frames(n);
        let cfg = IpaConfig {
            gamma: 0.5,
            ..make_ipa_cfg(n, h, d, p)
        };

        let out = ipa_apply(&q_s, &k_s, &v_s, &bias, &q_p, &k_p, &frames, &cfg);
        assert_eq!(out.len(), n * h * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
