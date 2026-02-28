// SPDX-License-Identifier: AGPL-3.0-or-later

//! Frame operations: quaternion→rotation, frame application/inversion.

#![allow(clippy::suboptimal_flops, clippy::many_single_char_names)]

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
pub(super) fn get_frame(frames: &[f64], idx: usize) -> ([f64; 9], [f64; 3]) {
    let base = idx * 12;
    let mut rot = [0.0; 9];
    let mut trans = [0.0; 3];
    rot.copy_from_slice(&frames[base..base + 9]);
    trans.copy_from_slice(&frames[base + 9..base + 12]);
    (rot, trans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;

    const EPS: f64 = tolerances::FOLDING_EPS;

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
}
