// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: quantized inference primitives (Study 005).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/quantized/quantized_inference.py`
//! Run: 2026-02-16, southGate (Ryzen 7 5800X3D), Python 3.10, `PyTorch` 2.9.0+cu128, seed=42
//! Command: `python3 control/quantized/quantized_inference.py`
//!
//! Validates INT8 and INT4 quantization, dequantization, quantized GEMV,
//! and accuracy degradation bounds. These are the deployment-path primitives
//! that `BarraCUDA`'s `gemv_q8.wgsl` and `gemv_q4.wgsl` will implement.

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use neural_spring::quantized::{
    dequantize_q4, dequantize_q8, gemv_f64, gemv_q4, gemv_q8, q4_params, q8_params, quantize_q4,
    quantize_q8, relative_l2_error,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("quantized");

    validate_q8_round_trip(&mut h);
    validate_q4_round_trip(&mut h);
    validate_q8_gemv(&mut h);
    validate_q4_gemv(&mut h);
    validate_accuracy_degradation(&mut h);
    validate_edge_cases(&mut h);
    validate_determinism(&mut h);

    h.finish();
}

fn validate_q8_round_trip(h: &mut ValidationHarness) {
    let data = vec![0.0, 1.0, -1.0, 0.5, -0.5, 0.127, -0.127];
    let params = q8_params(&data);

    h.check_bool("Q8 scale > 0", params.scale > 0.0);
    h.check_abs(
        "Q8 scale = max/127",
        params.scale,
        1.0 / 127.0,
        tolerances::EXACT_F64,
    );

    let quantized = quantize_q8(&data, &params);
    let dequantized = dequantize_q8(&quantized, &params);

    let max_err: f64 = data
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        "Q8 round-trip max error < scale",
        max_err,
        tolerances::QUANT_Q8_ELEMENT_ERROR * params.scale,
    );

    h.check_abs(
        "Q8(0) = 0 exactly",
        dequantized[0],
        0.0,
        tolerances::EXACT_F64,
    );
}

fn validate_q4_round_trip(h: &mut ValidationHarness) {
    let data = vec![0.0, 1.0, -1.0, 0.5, -0.5];
    let params = q4_params(&data);

    h.check_bool("Q4 scale > 0", params.scale > 0.0);
    h.check_abs(
        "Q4 scale = max/7",
        params.scale,
        1.0 / 7.0,
        tolerances::EXACT_F64,
    );

    let quantized = quantize_q4(&data, &params);
    let dequantized = dequantize_q4(&quantized, &params);

    let max_err: f64 = data
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        "Q4 round-trip max error < scale",
        max_err,
        tolerances::QUANT_Q4_ELEMENT_ERROR * params.scale,
    );

    h.check_abs(
        "Q4(0) = 0 exactly",
        dequantized[0],
        0.0,
        tolerances::EXACT_F64,
    );
}

fn validate_q8_gemv(h: &mut ValidationHarness) {
    let rows = 4;
    let cols = 8;

    let matrix: Vec<f64> = (0..rows * cols)
        .map(|i| ((i as f64) - 16.0) * 0.1)
        .collect();
    let vector: Vec<f64> = (0..cols).map(|i| (i as f64) * 0.25).collect();

    let fp_result = gemv_f64(&matrix, &vector, rows, cols);

    let mat_params = q8_params(&matrix);
    let vec_params = q8_params(&vector);
    let q_matrix = quantize_q8(&matrix, &mat_params);
    let q_vector = quantize_q8(&vector, &vec_params);
    let q_result = gemv_q8(&q_matrix, &q_vector, rows, cols, &mat_params, &vec_params);

    let err = relative_l2_error(&q_result, &fp_result);

    h.check_upper(
        &format!("Q8 GEMV relative L2 error {err:.4} < 1%"),
        err,
        tolerances::QUANT_INT8_DEGRADATION,
    );

    for (idx, (fp, q)) in fp_result.iter().zip(q_result.iter()).enumerate() {
        h.check_bool(
            &format!("Q8 GEMV[{idx}]: fp={fp:.4}, q8={q:.4} (same sign)"),
            fp.signum() == q.signum() || fp.abs() < tolerances::QUANT_SIGN_AGREEMENT,
        );
    }
}

fn validate_q4_gemv(h: &mut ValidationHarness) {
    let rows = 4;
    let cols = 8;

    let matrix: Vec<f64> = (0..rows * cols)
        .map(|i| ((i as f64) - 16.0) * 0.1)
        .collect();
    let vector: Vec<f64> = (0..cols).map(|i| (i as f64) * 0.25).collect();

    let fp_result = gemv_f64(&matrix, &vector, rows, cols);

    let mat_params = q4_params(&matrix);
    let vec_params = q4_params(&vector);
    let q_matrix = quantize_q4(&matrix, &mat_params);
    let q_vector = quantize_q4(&vector, &vec_params);
    let q_result = gemv_q4(&q_matrix, &q_vector, rows, cols, &mat_params, &vec_params);

    let err = relative_l2_error(&q_result, &fp_result);

    h.check_upper(
        &format!("Q4 GEMV relative L2 error {err:.4} < 5%"),
        err,
        tolerances::QUANT_INT4_DEGRADATION,
    );
}

fn validate_accuracy_degradation(h: &mut ValidationHarness) {
    let rows = 16;
    let cols = 16;

    let mut rng_state = 42_u64;
    let next_f64 = |state: &mut u64| -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((*state >> 33) as f64) / (f64::from(u32::MAX)) - 0.5
    };

    let matrix: Vec<f64> = (0..rows * cols).map(|_| next_f64(&mut rng_state)).collect();
    let vector: Vec<f64> = (0..cols).map(|_| next_f64(&mut rng_state)).collect();

    let fp_result = gemv_f64(&matrix, &vector, rows, cols);

    let mat_p8 = q8_params(&matrix);
    let vec_p8 = q8_params(&vector);
    let q8_result = gemv_q8(
        &quantize_q8(&matrix, &mat_p8),
        &quantize_q8(&vector, &vec_p8),
        rows,
        cols,
        &mat_p8,
        &vec_p8,
    );
    let err_q8 = relative_l2_error(&q8_result, &fp_result);

    let mat_p4 = q4_params(&matrix);
    let vec_p4 = q4_params(&vector);
    let q4_result = gemv_q4(
        &quantize_q4(&matrix, &mat_p4),
        &quantize_q4(&vector, &vec_p4),
        rows,
        cols,
        &mat_p4,
        &vec_p4,
    );
    let err_q4 = relative_l2_error(&q4_result, &fp_result);

    h.check_bool(
        &format!("Q4 error ({err_q4:.4}) > Q8 error ({err_q8:.4})"),
        err_q4 > err_q8,
    );

    h.check_upper(
        &format!("Q8 random GEMV L2 error {err_q8:.4} < 5%"),
        err_q8,
        tolerances::QUANT_Q8_GEMV_ERROR,
    );
    h.check_upper(
        &format!("Q4 random GEMV L2 error {err_q4:.4} < 25%"),
        err_q4,
        tolerances::QUANT_Q4_GEMV_ERROR,
    );
}

fn validate_edge_cases(h: &mut ValidationHarness) {
    let data = vec![1000.0, -1000.0, 0.0];

    let p8 = q8_params(&data);
    let q8 = quantize_q8(&data, &p8);
    h.check_bool("Q8 clamp: max → 127", q8[0] == 127);
    h.check_bool("Q8 clamp: min → -127", q8[1] == -127);
    h.check_bool("Q8 zero → 0", q8[2] == 0);

    let p4 = q4_params(&data);
    let q4 = quantize_q4(&data, &p4);
    h.check_bool("Q4 clamp: max → 7", q4[0] == 7);
    h.check_bool("Q4 clamp: min → -7", q4[1] == -7);
    h.check_bool("Q4 zero → 0", q4[2] == 0);
}

fn validate_determinism(h: &mut ValidationHarness) {
    let data = vec![0.3, -0.7, 1.2, -0.1, 0.8];

    let p1 = q8_params(&data);
    let p2 = q8_params(&data);
    h.check_abs(
        "Q8 params deterministic",
        p1.scale,
        p2.scale,
        tolerances::EXACT_F64,
    );

    let q1 = quantize_q8(&data, &p1);
    let q2 = quantize_q8(&data, &p2);
    h.check_bool("Q8 quantize deterministic", q1 == q2);

    let d1 = dequantize_q8(&q1, &p1);
    let d2 = dequantize_q8(&q2, &p2);
    h.check_bool("Q8 dequantize deterministic", d1 == d2);
}
