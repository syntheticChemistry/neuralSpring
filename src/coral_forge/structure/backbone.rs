// SPDX-License-Identifier: AGPL-3.0-or-later

//! Backbone update and torsion angle prediction.

#![expect(clippy::similar_names, reason = "domain-specific numeric patterns")]

use super::frame::{compose_frames, get_frame, quat_to_rotation};

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
///
/// Delegates to [`primitives::relu_inplace`](crate::primitives::relu_inplace).
fn relu_inplace(x: &mut [f64]) {
    crate::primitives::relu_inplace(x);
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
            let norm = s.hypot(c).max(crate::tolerances::FOLDING_EPS);
            out[i * 14 + a * 2] = s / norm;
            out[i * 14 + a * 2 + 1] = c / norm;
        }
    }
    out
}

#[cfg(test)]
#[expect(
    clippy::cast_precision_loss,
    reason = "test indices cast to f64 — safe for small N"
)]
mod tests {
    use super::*;
    use crate::tolerances;

    const EPS: f64 = tolerances::FOLDING_EPS;

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

    #[test]
    fn torsion_output_on_unit_circle() {
        let (n, c_s, c_h) = (3, 4, 8);
        let single: Vec<f64> = (0..n * c_s)
            .map(|i| (i as f64).mul_add(0.1, -0.5))
            .collect();
        let n_weights = c_s * c_h + c_h + 4 * (c_h * c_h + c_h) + c_h * 14 + 14;
        let weights: Vec<f64> = (0..n_weights)
            .map(|i| (i as f64).mul_add(0.001, -0.1))
            .collect();

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
}
