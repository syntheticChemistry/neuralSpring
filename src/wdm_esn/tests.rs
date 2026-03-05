// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(clippy::expect_used, reason = "test assertions use expect for clarity")]

use super::*;

fn tiny_esn() -> EsnClassifier {
    let rs = 4;
    let nc = 3;
    EsnClassifier {
        w_in: vec![0.1; rs * 2],
        w_res: vec![0.01; rs * rs],
        b_res: vec![0.0; rs],
        w_out: vec![0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        b_out: vec![0.0; nc],
        reservoir_size: rs,
        n_classes: nc,
        norm: EsnNormalization {
            x_mean: [0.5, 6.0],
            x_std: [1.0, 1.5],
        },
    }
}

#[test]
fn classify_deterministic() {
    let esn = tiny_esn();
    let (l1, s1) = esn.classify(0.5, 5.5);
    let (l2, s2) = esn.classify(0.5, 5.5);
    assert_eq!(l1, l2);
    for (a, b) in s1.iter().zip(s2.iter()) {
        assert!((a - b).abs() < f64::EPSILON);
    }
}

#[test]
fn classify_finite_scores() {
    let esn = tiny_esn();
    let (_, scores) = esn.classify(1.0, 6.0);
    assert!(scores.iter().all(|s| s.is_finite()));
}

#[test]
fn classify_label_in_range() {
    let esn = tiny_esn();
    for &(lr, lt) in &[(-1.0, 8.0), (2.0, 4.0), (0.5, 5.5)] {
        let (label, _) = esn.classify(lr, lt);
        assert!(label < 3, "label {label} out of range");
    }
}

#[test]
fn load_roundtrip() {
    let json = r#"{
        "normalization": {"x_mean": [0.5, 6.0], "x_std": [1.0, 1.5]},
        "weights": {
            "reservoir_size": 2, "input_dim": 2, "n_classes": 3,
            "W_in": [0.1, 0.2, 0.3, 0.4],
            "W_res": [0.01, 0.02, 0.03, 0.04],
            "b_res": [0.0, 0.0],
            "W_out": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "b_out": [0.0, 0.0, 0.0]
        }
    }"#;
    let esn = load_esn_from_json(json).expect("valid JSON should parse");
    let (label, scores) = esn.classify(0.5, 5.5);
    assert!(label < 3);
    assert!(scores.iter().all(|s| s.is_finite()));
}

#[test]
fn load_invalid_json() {
    assert!(load_esn_from_json("nope").is_err());
}

#[test]
fn load_missing_weights() {
    let json = r#"{"normalization": {"x_mean": [0, 0], "x_std": [1, 1]}}"#;
    assert!(load_esn_from_json(json).is_err());
}

#[test]
fn argmax_helpers() {
    assert_eq!(argmax_f64(&[1.0, 3.0, 2.0]), 1);
    assert_eq!(argmax_f32(&[0.1, 0.9, 0.5]), 1);
    assert_eq!(argmax_f64(&[]), 0);
}

#[test]
fn wdm_head_configs_correct() {
    let heads = wdm_head_configs(3);
    assert_eq!(heads.len(), wdm_heads::COUNT);
    assert_eq!(heads[wdm_heads::REGIME_LABEL].output_size, 3);
    assert_eq!(heads[wdm_heads::SPECTRAL_BANDWIDTH].output_size, 1);
    assert_eq!(heads[wdm_heads::CONFIDENCE].output_size, 1);
}

#[test]
fn classify_extreme_inputs() {
    let esn = tiny_esn();
    let (label_cold, _) = esn.classify(-3.0, 3.0);
    let (label_hot, _) = esn.classify(3.0, 9.0);
    assert!(label_cold < 3);
    assert!(label_hot < 3);
}

#[test]
fn classify_reservoir_nonlinearity() {
    let esn = tiny_esn();
    let (_, scores_a) = esn.classify(0.0, 5.0);
    let (_, scores_b) = esn.classify(1.0, 7.0);
    let different = scores_a
        .iter()
        .zip(scores_b.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(
        different,
        "different inputs should produce different scores"
    );
}

#[test]
fn load_esn_full_roundtrip_classify() {
    let json = r#"{
        "normalization": {"x_mean": [0.5, 6.0], "x_std": [1.0, 1.5]},
        "weights": {
            "reservoir_size": 4, "input_dim": 2, "n_classes": 3,
            "W_in": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
            "W_res": [0.01, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0,
                       0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.01],
            "b_res": [0.0, 0.0, 0.0, 0.0],
            "W_out": [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            "b_out": [0.0, 0.0, 0.0]
        }
    }"#;
    let esn = load_esn_from_json(json).expect("valid JSON");
    assert_eq!(esn.reservoir_size, 4);
    assert_eq!(esn.n_classes, 3);
    for &(lr, lt) in &[(-2.0, 4.0), (0.5, 6.0), (3.0, 8.0)] {
        let (label, scores) = esn.classify(lr, lt);
        assert!(label < 3);
        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|s| s.is_finite()));
    }
}

#[test]
fn wdm_head_configs_two_classes() {
    let heads = wdm_head_configs(2);
    assert_eq!(heads[wdm_heads::REGIME_LABEL].output_size, 2);
}

#[test]
fn argmax_single_element() {
    assert_eq!(argmax_f64(&[42.0]), 0);
    assert_eq!(argmax_f32(&[42.0]), 0);
}

#[test]
fn argmax_negative_values() {
    assert_eq!(argmax_f64(&[-3.0, -1.0, -2.0]), 1);
    assert_eq!(argmax_f32(&[-3.0, -1.0, -2.0]), 1);
}
