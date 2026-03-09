// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU-path linear algebra tests for [`Dispatcher`].

use super::*;
use crate::tolerances;

fn cpu() -> Dispatcher {
    Dispatcher::cpu_only()
}

#[test]
fn cpu_mat_mul_identity() {
    let d = cpu();
    #[rustfmt::skip]
    let eye = vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    let result = d.mat_mul(&eye, &eye, 3);
    for (i, &v) in result.iter().enumerate() {
        let expected = if i / 3 == i % 3 { 1.0 } else { 0.0 };
        assert!(
            (v - expected).abs() < tolerances::ZERO_DETECTION,
            "mat_mul identity [{i}]"
        );
    }
}

#[test]
fn cpu_frobenius_norm() {
    let d = cpu();
    let a = vec![3.0, 4.0];
    assert!((d.frobenius_norm(&a) - 5.0).abs() < tolerances::EXACT_F64);
}

#[test]
fn cpu_transpose() {
    let d = cpu();
    #[rustfmt::skip]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let t = d.transpose(&a, 2);
    assert!((t[0] - 1.0).abs() < tolerances::ZERO_DETECTION);
    assert!((t[1] - 3.0).abs() < tolerances::ZERO_DETECTION);
    assert!((t[2] - 2.0).abs() < tolerances::ZERO_DETECTION);
    assert!((t[3] - 4.0).abs() < tolerances::ZERO_DETECTION);
}

#[test]
fn cpu_distance_to_normal() {
    let d = cpu();
    #[rustfmt::skip]
    let sym = vec![
        2.0, 1.0,
        1.0, 2.0,
    ];
    let dist = d.distance_to_normal(&sym, 2);
    assert!(
        dist < tolerances::EXACT_F64,
        "symmetric matrix should commute with transpose"
    );
}

#[test]
fn cpu_commutator_symmetric_zero() {
    let d = cpu();
    let a = vec![1.0, 0.0, 0.0, 1.0];
    let comm = d.commutator(&a, &a, 2);
    for &v in &comm {
        assert!(
            v.abs() < tolerances::ZERO_DETECTION,
            "A commutes with itself"
        );
    }
}

#[test]
fn cpu_eigh_diagonal() {
    let d = cpu();
    let a = vec![2.0, 0.0, 0.0, 3.0];
    let (vals, _vecs) = d.eigh(&a, 2);
    let mut sorted = vals;
    sorted.sort_by(f64::total_cmp);
    assert!((sorted[0] - 2.0).abs() < tolerances::CROSS_LANGUAGE);
    assert!((sorted[1] - 3.0).abs() < tolerances::CROSS_LANGUAGE);
}

#[test]
fn cpu_disorder_sweep_no_gpu() {
    let d = cpu();
    assert!(d.disorder_sweep(&[1.0, 0.0, 0.0, 1.0], 2, 1).is_none());
}
