// SPDX-License-Identifier: AGPL-3.0-or-later

//! Structure Module: rigid-body frame operations and Invariant Point Attention.
//!
//! Phase B.3 of the sovereign folding track. Implements the Structure Module
//! from `AlphaFold2` (Jumper et al. 2021, Algorithm 22):
//!
//! - Frame operations: quaternion→rotation, frame application/inversion
//! - [`ipa_scores`] — Invariant Point Attention score computation
//! - [`ipa_apply`] — IPA weighted value summation (scalar + point outputs)
//! - [`backbone_update`] — Frame composition from predicted updates
//!
//! ## Frame representation
//!
//! Each residue frame is stored as 12 f64 values: rotation matrix (9, row-major)
//! followed by translation vector (3). This avoids quaternion singularities
//! during GPU computation.
//!
//! ## IPA attention score (Algorithm 22)
//!
//! ```text
//! a[h,i,j] = w_L * Q·K/√c
//!          + w_C * pair_bias[h,i,j]
//!          + w_P * (-γ/2) * Σ_p ||T_i(q_p) - T_j(k_p)||²
//! ```
//!
//! The point distance term makes attention SE(3)-equivariant: scores depend
//! on 3D proximity of query/key points projected through backbone frames.
//!
//! ## References
//!
//! - Jumper et al. "Highly accurate protein structure prediction with
//!   `AlphaFold`" Nature 596:583-589 (2021), Algorithm 22
//! - Ahdritz et al. "`OpenFold`" Nature Methods (2024)

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    clippy::suboptimal_flops,
    clippy::similar_names
)]

// ═══════════════════════════════════════════════════════════════════
// Frame operations
// ═══════════════════════════════════════════════════════════════════

/// Convert unit quaternion `[w, x, y, z]` to 3×3 rotation matrix (row-major).
///
/// Input need not be normalized — the function normalizes internally.
#[must_use]
pub fn quat_to_rotation(q: &[f64; 4]) -> [f64; 9] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (w, x, y, z) = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm);

    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

/// Apply rigid-body frame: `R @ point + translation`.
///
/// `rot`: row-major 3×3, `trans`: translation 3-vector.
#[must_use]
pub fn apply_frame(rot: &[f64; 9], trans: &[f64; 3], point: &[f64; 3]) -> [f64; 3] {
    [
        rot[0].mul_add(point[0], rot[1].mul_add(point[1], rot[2] * point[2])) + trans[0],
        rot[3].mul_add(point[0], rot[4].mul_add(point[1], rot[5] * point[2])) + trans[1],
        rot[6].mul_add(point[0], rot[7].mul_add(point[1], rot[8] * point[2])) + trans[2],
    ]
}

/// Invert rigid-body frame: `R^T @ (point - translation)`.
///
/// Returns `(R^T, -R^T @ t)`.
#[must_use]
pub fn invert_frame(rot: &[f64; 9], trans: &[f64; 3]) -> ([f64; 9], [f64; 3]) {
    let rt = [
        rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8],
    ];
    let neg_rt_t = [
        -(rt[0] * trans[0] + rt[1] * trans[1] + rt[2] * trans[2]),
        -(rt[3] * trans[0] + rt[4] * trans[1] + rt[5] * trans[2]),
        -(rt[6] * trans[0] + rt[7] * trans[1] + rt[8] * trans[2]),
    ];
    (rt, neg_rt_t)
}

/// Compose two frames: `T_out = T_1 ∘ T_2`, i.e. `R_out = R1 @ R2`, `t_out = R1 @ t2 + t1`.
#[must_use]
pub fn compose_frames(
    r1: &[f64; 9],
    t1: &[f64; 3],
    r2: &[f64; 9],
    t2: &[f64; 3],
) -> ([f64; 9], [f64; 3]) {
    let mut r_out = [0.0; 9];
    for row in 0..3 {
        for col in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc = r1[row * 3 + k].mul_add(r2[k * 3 + col], acc);
            }
            r_out[row * 3 + col] = acc;
        }
    }
    let t_out = apply_frame(r1, t1, t2);
    (r_out, t_out)
}

/// Extract frame (rotation 9 + translation 3) from flat frame array.
fn get_frame(frames: &[f64], idx: usize) -> ([f64; 9], [f64; 3]) {
    let base = idx * 12;
    let mut rot = [0.0; 9];
    let mut trans = [0.0; 3];
    rot.copy_from_slice(&frames[base..base + 9]);
    trans.copy_from_slice(&frames[base + 9..base + 12]);
    (rot, trans)
}

// ═══════════════════════════════════════════════════════════════════
// Invariant Point Attention
// ═══════════════════════════════════════════════════════════════════

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
/// - `w_l`, `w_c`, `w_p`: scalar/pair/point weight multipliers
/// - `gamma`: per-head point weight
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
    n_res: usize,
    n_heads: usize,
    head_dim: usize,
    n_points: usize,
    w_l: f64,
    w_c: f64,
    w_p: f64,
    gamma: f64,
) -> Vec<f64> {
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

                // Term 1: scalar attention
                let mut dot = 0.0_f64;
                for d in 0..head_dim {
                    let qi = q_scalar[(i * n_heads + h) * head_dim + d];
                    let ki = k_scalar[(j * n_heads + h) * head_dim + d];
                    dot = qi.mul_add(ki, dot);
                }
                let scalar_term = w_l * dot / scale;

                // Term 2: pair bias
                let pair_term = w_c * pair_bias[(h * n_res + i) * n_res + j];

                // Term 3: point distance
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
pub fn ipa_apply(
    q_scalar: &[f64],
    k_scalar: &[f64],
    v_scalar: &[f64],
    pair_bias: &[f64],
    q_points: &[f64],
    k_points: &[f64],
    frames: &[f64],
    n_res: usize,
    n_heads: usize,
    head_dim: usize,
    n_points: usize,
    w_l: f64,
    w_c: f64,
    w_p: f64,
    gamma: f64,
) -> Vec<f64> {
    assert_eq!(v_scalar.len(), n_res * n_heads * head_dim);

    let logits = ipa_scores(
        q_scalar, k_scalar, pair_bias, q_points, k_points, frames, n_res, n_heads, head_dim,
        n_points, w_l, w_c, w_p, gamma,
    );

    let weights = crate::sovereign_folding::softmax_rows(&logits, n_heads * n_res, n_res);

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

// ═══════════════════════════════════════════════════════════════════
// Backbone update
// ═══════════════════════════════════════════════════════════════════

/// Update backbone frames by composing with predicted delta transforms.
///
/// `delta_quats`: `[N, 4]` — quaternion updates (will be normalized).
/// `delta_trans`: `[N, 3]` — translation updates.
/// `current_frames`: `[N, 12]` — current backbone frames (rot9 + trans3).
///
/// Returns `[N, 12]` — updated frames: `T_new = T_current ∘ T_delta`.
///
/// # Panics
///
/// Panics if slice lengths don't match `n_res`.
#[must_use]
pub fn backbone_update(
    delta_quats: &[f64],
    delta_trans: &[f64],
    current_frames: &[f64],
    n_res: usize,
) -> Vec<f64> {
    assert_eq!(delta_quats.len(), n_res * 4);
    assert_eq!(delta_trans.len(), n_res * 3);
    assert_eq!(current_frames.len(), n_res * 12);

    let mut out = vec![0.0; n_res * 12];

    for i in 0..n_res {
        let q = [
            delta_quats[i * 4],
            delta_quats[i * 4 + 1],
            delta_quats[i * 4 + 2],
            delta_quats[i * 4 + 3],
        ];
        let delta_rot = quat_to_rotation(&q);
        let delta_t = [
            delta_trans[i * 3],
            delta_trans[i * 3 + 1],
            delta_trans[i * 3 + 2],
        ];

        let (cur_rot, cur_trans) = get_frame(current_frames, i);
        let (new_rot, new_trans) = compose_frames(&cur_rot, &cur_trans, &delta_rot, &delta_t);

        out[i * 12..i * 12 + 9].copy_from_slice(&new_rot);
        out[i * 12 + 9..i * 12 + 12].copy_from_slice(&new_trans);
    }
    out
}

// ═══════════════════════════════════════════════════════════════════
// Torsion angle prediction
// ═══════════════════════════════════════════════════════════════════

/// Dense linear transform: `y = x @ W + bias`, or `y = x @ W` when bias is empty.
///
/// `x`: `[rows, in_dim]`, `w`: `[in_dim, out_dim]`, `bias`: `[out_dim]` or empty.
fn linear(
    x: &[f64],
    w: &[f64],
    bias: &[f64],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f64> {
    let mut y = vec![0.0; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = if bias.is_empty() { 0.0 } else { bias[o] };
            for i in 0..in_dim {
                acc = x[r * in_dim + i].mul_add(w[i * out_dim + o], acc);
            }
            y[r * out_dim + o] = acc;
        }
    }
    y
}

/// `ReLU` activation: `max(0, x)`.
fn relu_inplace(x: &mut [f64]) {
    for v in x.iter_mut() {
        *v = v.max(0.0);
    }
}

/// One `ResNet` block: `x + Linear(ReLU(Linear(x)))`.
///
/// `w1`, `w2`: `[dim, dim]`; `b1`, `b2`: `[dim]`.
fn resnet_block(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    rows: usize,
    dim: usize,
) -> Vec<f64> {
    let mut h = linear(x, w1, b1, rows, dim, dim);
    relu_inplace(&mut h);
    let h2 = linear(&h, w2, b2, rows, dim, dim);
    let mut out = vec![0.0; rows * dim];
    for i in 0..out.len() {
        out[i] = x[i] + h2[i];
    }
    out
}

/// Predict side-chain torsion angles from single representation.
///
/// Architecture: `Linear → ResNet → ResNet → Linear → normalize`.
///
/// Returns `[N_res, 7, 2]` (sin, cos for 7 torsion angles), each pair
/// normalized to the unit circle.
///
/// ## Weight layout
///
/// All weights concatenated in order:
/// 1. `proj_in`:  `[c_single, c_hidden]`
/// 2. `proj_in_bias`: `[c_hidden]`
/// 3. `res1_w1`:  `[c_hidden, c_hidden]`
/// 4. `res1_b1`:  `[c_hidden]`
/// 5. `res1_w2`:  `[c_hidden, c_hidden]`
/// 6. `res1_b2`:  `[c_hidden]`
/// 7. `res2_w1`:  `[c_hidden, c_hidden]`
/// 8. `res2_b1`:  `[c_hidden]`
/// 9. `res2_w2`:  `[c_hidden, c_hidden]`
/// 10. `res2_b2`: `[c_hidden]`
/// 11. `proj_out`: `[c_hidden, 14]`
/// 12. `proj_out_bias`: `[14]`
///
/// # Panics
///
/// Panics if `weights` length doesn't match the expected total.
#[must_use]
pub fn torsion_angles(
    single: &[f64],
    weights: &[f64],
    n_res: usize,
    c_single: usize,
    c_hidden: usize,
) -> Vec<f64> {
    assert_eq!(single.len(), n_res * c_single);

    let hh = c_hidden * c_hidden;
    let expected_weights = c_single * c_hidden + c_hidden      // proj_in + bias
        + hh + c_hidden + hh + c_hidden                        // resblock 1
        + hh + c_hidden + hh + c_hidden                        // resblock 2
        + c_hidden * 14 + 14; // proj_out + bias
    assert_eq!(weights.len(), expected_weights);

    let mut off = 0;
    let proj_in_w = &weights[off..off + c_single * c_hidden];
    off += c_single * c_hidden;
    let proj_in_b = &weights[off..off + c_hidden];
    off += c_hidden;
    let r1_w1 = &weights[off..off + hh];
    off += hh;
    let r1_b1 = &weights[off..off + c_hidden];
    off += c_hidden;
    let r1_w2 = &weights[off..off + hh];
    off += hh;
    let r1_b2 = &weights[off..off + c_hidden];
    off += c_hidden;
    let r2_w1 = &weights[off..off + hh];
    off += hh;
    let r2_b1 = &weights[off..off + c_hidden];
    off += c_hidden;
    let r2_w2 = &weights[off..off + hh];
    off += hh;
    let r2_b2 = &weights[off..off + c_hidden];
    off += c_hidden;
    let proj_out_w = &weights[off..off + c_hidden * 14];
    off += c_hidden * 14;
    let proj_out_b = &weights[off..off + 14];

    // Forward: proj_in → resblock1 → resblock2 → proj_out → normalize
    let h = linear(single, proj_in_w, proj_in_b, n_res, c_single, c_hidden);
    let h = resnet_block(&h, r1_w1, r1_b1, r1_w2, r1_b2, n_res, c_hidden);
    let h = resnet_block(&h, r2_w1, r2_b1, r2_w2, r2_b2, n_res, c_hidden);
    let raw = linear(&h, proj_out_w, proj_out_b, n_res, c_hidden, 14);

    // Normalize each (sin, cos) pair to the unit circle
    let mut out = vec![0.0; n_res * 14];
    for i in 0..n_res {
        for a in 0..7 {
            let s = raw[i * 14 + a * 2];
            let c = raw[i * 14 + a * 2 + 1];
            let norm = s.hypot(c).max(1e-12);
            out[i * 14 + a * 2] = s / norm;
            out[i * 14 + a * 2 + 1] = c / norm;
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    // ── Frame operations ────────────────────────────────────────

    #[test]
    fn identity_quaternion_gives_identity_rotation() {
        let q = [1.0, 0.0, 0.0, 0.0];
        let r = quat_to_rotation(&q);
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for (a, b) in r.iter().zip(identity.iter()) {
            assert!((a - b).abs() < EPS, "identity quat → identity rot");
        }
    }

    #[test]
    fn rotation_180_about_z() {
        let q = [0.0, 0.0, 0.0, 1.0]; // 180° about z
        let r = quat_to_rotation(&q);
        assert!((r[0] - (-1.0)).abs() < EPS, "R[0,0] = -1");
        assert!((r[4] - (-1.0)).abs() < EPS, "R[1,1] = -1");
        assert!((r[8] - 1.0).abs() < EPS, "R[2,2] = 1");
    }

    #[test]
    fn rotation_preserves_norm() {
        let q = [0.5, 0.5, 0.5, 0.5]; // 120° about (1,1,1)
        let r = quat_to_rotation(&q);
        let p = [1.0, 2.0, 3.0];
        let rp = apply_frame(&r, &[0.0, 0.0, 0.0], &p);
        let norm_before = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let norm_after = (rp[0] * rp[0] + rp[1] * rp[1] + rp[2] * rp[2]).sqrt();
        assert!(
            (norm_before - norm_after).abs() < EPS,
            "rotation preserves ||p||"
        );
    }

    #[test]
    fn apply_then_invert_is_identity() {
        let q = [
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
            0.0,
        ];
        let r = quat_to_rotation(&q);
        let t = [1.0, 2.0, 3.0];
        let p = [4.0, 5.0, 6.0];

        let transformed = apply_frame(&r, &t, &p);
        let (ri, ti) = invert_frame(&r, &t);
        let recovered = apply_frame(&ri, &ti, &transformed);

        for (a, b) in recovered.iter().zip(p.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "T^{{-1}}(T(p)) = p: got {a}, want {b}"
            );
        }
    }

    #[test]
    fn compose_with_identity_is_same() {
        let q = [0.5, 0.5, 0.5, 0.5];
        let r = quat_to_rotation(&q);
        let t = [10.0, 20.0, 30.0];
        let id_r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let id_t = [0.0, 0.0, 0.0];

        let (cr, ct) = compose_frames(&r, &t, &id_r, &id_t);
        for (a, b) in cr.iter().zip(r.iter()) {
            assert!((a - b).abs() < EPS);
        }
        for (a, b) in ct.iter().zip(t.iter()) {
            assert!((a - b).abs() < EPS);
        }
    }

    // ── IPA ─────────────────────────────────────────────────────

    #[test]
    fn ipa_scores_shape() {
        let (n, h, d, p) = (4, 2, 4, 3);
        let q_s = vec![0.1; n * h * d];
        let k_s = vec![0.1; n * h * d];
        let bias = vec![0.0; h * n * n];
        let q_p = vec![0.0; n * h * p * 3];
        let k_p = vec![0.0; n * h * p * 3];
        let frames = identity_frames(n);

        let scores = ipa_scores(
            &q_s, &k_s, &bias, &q_p, &k_p, &frames, n, h, d, p, 1.0, 1.0, 1.0, 1.0,
        );
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

        let scores = ipa_scores(
            &q_s, &k_s, &bias, &q_p, &k_p, &frames, n, h, d, p, 1.0, 1.0, 0.0, 1.0,
        );
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

        let scores = ipa_scores(
            &q_s, &k_s, &bias, &points, &points, &frames, n, h, d, p, 0.0, 0.0, 1.0, 1.0,
        );
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
        let q_s: Vec<f64> = (0..n * h * d).map(|i| i as f64 * 0.01).collect();
        let k_s: Vec<f64> = (0..n * h * d).map(|i| i as f64 * 0.02 - 0.3).collect();
        let v_s: Vec<f64> = (0..n * h * d).map(|i| i as f64 * 0.03 - 0.1).collect();
        let bias = vec![0.0; h * n * n];
        let q_p: Vec<f64> = (0..n * h * p * 3).map(|i| i as f64 * 0.01).collect();
        let k_p: Vec<f64> = (0..n * h * p * 3).map(|i| i as f64 * 0.01).collect();
        let frames = identity_frames(n);

        let out = ipa_apply(
            &q_s, &k_s, &v_s, &bias, &q_p, &k_p, &frames, n, h, d, p, 1.0, 1.0, 1.0, 0.5,
        );
        assert_eq!(out.len(), n * h * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Backbone update ─────────────────────────────────────────

    #[test]
    fn backbone_identity_update_preserves_frame() {
        let n = 3;
        let frames = identity_frames(n);
        let quats = [1.0, 0.0, 0.0, 0.0].repeat(n);
        let trans = vec![0.0; n * 3];

        let updated = backbone_update(&quats, &trans, &frames, n);
        for (a, b) in updated.iter().zip(frames.iter()) {
            assert!((a - b).abs() < EPS, "identity update preserves frame");
        }
    }

    #[test]
    fn backbone_translation_update() {
        let n = 2;
        let frames = identity_frames(n);
        let quats = [1.0, 0.0, 0.0, 0.0].repeat(n);
        let trans = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let updated = backbone_update(&quats, &trans, &frames, n);
        assert!((updated[9] - 1.0).abs() < EPS, "t_x updated for residue 0");
        assert!((updated[10] - 2.0).abs() < EPS, "t_y updated for residue 0");
        assert!((updated[11] - 3.0).abs() < EPS, "t_z updated for residue 0");
    }

    // ── Torsion angles ────────────────────────────────────────

    #[test]
    fn torsion_output_on_unit_circle() {
        let (n, c_s, c_h) = (3, 4, 8);
        let single: Vec<f64> = (0..n * c_s).map(|i| (i as f64) * 0.1 - 0.5).collect();
        let n_weights = c_s * c_h + c_h + 4 * (c_h * c_h + c_h) + c_h * 14 + 14;
        let weights: Vec<f64> = (0..n_weights).map(|i| (i as f64) * 0.001 - 0.1).collect();

        let out = torsion_angles(&single, &weights, n, c_s, c_h);
        assert_eq!(out.len(), n * 14);

        for i in 0..n {
            for a in 0..7 {
                let s = out[i * 14 + a * 2];
                let c = out[i * 14 + a * 2 + 1];
                let r = s.hypot(c);
                assert!(
                    (r - 1.0).abs() < 1e-10,
                    "angle {a} at residue {i}: ||(sin,cos)|| = {r}, want 1.0"
                );
            }
        }
    }

    #[test]
    fn torsion_output_finite() {
        let (n, c_s, c_h) = (2, 3, 4);
        let single = vec![0.5; n * c_s];
        let n_weights = c_s * c_h + c_h + 4 * (c_h * c_h + c_h) + c_h * 14 + 14;
        let weights = vec![0.01; n_weights];

        let out = torsion_angles(&single, &weights, n, c_s, c_h);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn linear_identity() {
        let x = [1.0, 2.0, 3.0];
        let w = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // 3x3 identity
        let b = [0.0, 0.0, 0.0];
        let y = linear(&x, &w, &b, 1, 3, 3);
        for (a, e) in y.iter().zip(x.iter()) {
            assert!((a - e).abs() < EPS);
        }
    }

    #[test]
    fn resnet_block_skip_connection() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let zero_w = [0.0; 4];
        let zero_b = [0.0; 2];
        let out = resnet_block(&x, &zero_w, &zero_b, &zero_w, &zero_b, 2, 2);
        for (a, e) in out.iter().zip(x.iter()) {
            assert!(
                (a - e).abs() < EPS,
                "zero weights → skip-only: got {a}, want {e}"
            );
        }
    }

    // ── Helpers ─────────────────────────────────────────────────

    fn identity_frames(n: usize) -> Vec<f64> {
        let mut frames = vec![0.0; n * 12];
        for i in 0..n {
            frames[i * 12] = 1.0; // R[0,0]
            frames[i * 12 + 4] = 1.0; // R[1,1]
            frames[i * 12 + 8] = 1.0; // R[2,2]
        }
        frames
    }
}
