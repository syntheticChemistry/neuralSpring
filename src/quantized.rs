// SPDX-License-Identifier: AGPL-3.0-or-later

//! Quantized inference primitives (Study 005).
//!
//! Pure-math implementations of INT8 and INT4 quantization, dequantization,
//! and quantized matrix-vector multiply. Validates the deployment path for
//! models trained in FP32 and compressed for consumer GPU inference.
//!
//! ## `BarraCUDA` Target
//!
//! - `dequant_q8.wgsl` — 8-bit dequantization
//! - `dequant_q4.wgsl` — 4-bit dequantization
//! - `gemv_q8.wgsl` — quantized matrix-vector multiply (INT8)
//! - `gemv_q4.wgsl` — quantized matrix-vector multiply (INT4)
//!
//! ## References
//!
//! - Dettmers et al. (2022) `LLM.int8()`
//! - Frantar et al. (2023) `GPTQ`

/// Symmetric INT8 quantization parameters.
#[derive(Debug, Clone)]
pub struct Q8Params {
    /// Scale factor: `max(|tensor|) / 127`
    pub scale: f64,
}

/// Symmetric INT4 quantization parameters.
#[derive(Debug, Clone)]
pub struct Q4Params {
    /// Scale factor: `max(|tensor|) / 7`
    pub scale: f64,
}

/// Compute symmetric INT8 quantization parameters for a tensor.
#[must_use]
pub fn q8_params(data: &[f64]) -> Q8Params {
    let abs_max = data
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max)
        .max(crate::primitives::QUANTIZATION_FLOOR);
    Q8Params {
        scale: abs_max / 127.0,
    }
}

/// Compute symmetric INT4 quantization parameters for a tensor.
#[must_use]
pub fn q4_params(data: &[f64]) -> Q4Params {
    let abs_max = data
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max)
        .max(crate::primitives::QUANTIZATION_FLOOR);
    Q4Params {
        scale: abs_max / 7.0,
    }
}

/// Quantize FP64 values to INT8 (symmetric, clamp to `[-128, 127]`).
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "f64 → i8 after clamp to [-128, 127]"
)]
pub fn quantize_q8(data: &[f64], params: &Q8Params) -> Vec<i8> {
    data.iter()
        .map(|&x| (x / params.scale).round().clamp(-128.0, 127.0) as i8)
        .collect()
}

/// Dequantize INT8 back to FP64.
#[must_use]
pub fn dequantize_q8(data: &[i8], params: &Q8Params) -> Vec<f64> {
    data.iter().map(|&q| f64::from(q) * params.scale).collect()
}

/// Quantize FP64 values to INT4 (symmetric, clamp to `[-8, 7]`).
///
/// Stored as `i8` since Rust has no native 4-bit type.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "f64 → i8 after clamp to [-8, 7]"
)]
pub fn quantize_q4(data: &[f64], params: &Q4Params) -> Vec<i8> {
    data.iter()
        .map(|&x| (x / params.scale).round().clamp(-8.0, 7.0) as i8)
        .collect()
}

/// Dequantize INT4 back to FP64.
#[must_use]
pub fn dequantize_q4(data: &[i8], params: &Q4Params) -> Vec<f64> {
    data.iter().map(|&q| f64::from(q) * params.scale).collect()
}

/// Quantized matrix-vector multiply (INT8).
///
/// Computes `y = dequant(Q_matrix · Q_vector)` where both matrix and vector
/// are quantized to INT8. The accumulation is done in `i32` to avoid overflow,
/// then scaled back to FP64.
///
/// - `q_matrix`: `[rows, cols]` row-major INT8
/// - `q_vector`: `[cols]` INT8
/// - `mat_params`: quantization scale for the matrix
/// - `vec_params`: quantization scale for the vector
#[must_use]
pub fn gemv_q8(
    q_matrix: &[i8],
    q_vector: &[i8],
    rows: usize,
    cols: usize,
    mat_params: &Q8Params,
    vec_params: &Q8Params,
) -> Vec<f64> {
    let combined_scale = mat_params.scale * vec_params.scale;
    let mut result = vec![0.0_f64; rows];
    for r in 0..rows {
        let mut acc: i32 = 0;
        for c in 0..cols {
            acc += i32::from(q_matrix[r * cols + c]) * i32::from(q_vector[c]);
        }
        result[r] = f64::from(acc) * combined_scale;
    }
    result
}

/// Quantized matrix-vector multiply (INT4).
///
/// Same as [`gemv_q8`] but for 4-bit quantized values (stored as `i8`).
#[must_use]
pub fn gemv_q4(
    q_matrix: &[i8],
    q_vector: &[i8],
    rows: usize,
    cols: usize,
    mat_params: &Q4Params,
    vec_params: &Q4Params,
) -> Vec<f64> {
    let combined_scale = mat_params.scale * vec_params.scale;
    let mut result = vec![0.0_f64; rows];
    for r in 0..rows {
        let mut acc: i32 = 0;
        for c in 0..cols {
            acc += i32::from(q_matrix[r * cols + c]) * i32::from(q_vector[c]);
        }
        result[r] = f64::from(acc) * combined_scale;
    }
    result
}

/// Full-precision matrix-vector multiply (reference implementation).
#[must_use]
pub fn gemv_f64(matrix: &[f64], vector: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; rows];
    for r in 0..rows {
        let mut acc = 0.0_f64;
        for c in 0..cols {
            acc += matrix[r * cols + c] * vector[c];
        }
        result[r] = acc;
    }
    result
}

/// Compute relative L2 error between two vectors.
#[must_use]
pub fn relative_l2_error(approx: &[f64], reference: &[f64]) -> f64 {
    let num: f64 = approx
        .iter()
        .zip(reference.iter())
        .map(|(a, r)| (a - r).powi(2))
        .sum();
    let den: f64 = reference.iter().map(|r| r.powi(2)).sum();
    if den < 1e-30 {
        return num.sqrt();
    }
    (num / den).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn q8_round_trip_identity() {
        let data = vec![0.0, 1.0, -1.0, 0.5, -0.5];
        let params = q8_params(&data);
        let q = quantize_q8(&data, &params);
        let dq = dequantize_q8(&q, &params);
        for (orig, recovered) in data.iter().zip(dq.iter()) {
            assert!((orig - recovered).abs() < params.scale, "Q8 round-trip");
        }
    }

    #[test]
    fn q4_round_trip_identity() {
        let data = vec![0.0, 1.0, -1.0, 0.5, -0.5];
        let params = q4_params(&data);
        let q = quantize_q4(&data, &params);
        let dq = dequantize_q4(&q, &params);
        for (orig, recovered) in data.iter().zip(dq.iter()) {
            assert!((orig - recovered).abs() < params.scale, "Q4 round-trip");
        }
    }

    #[test]
    fn gemv_q8_small() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0];
        let vector = vec![1.0, 1.0];
        let fp_result = gemv_f64(&matrix, &vector, 2, 2);
        assert!((fp_result[0] - 3.0).abs() < tolerances::EXACT_F64);
        assert!((fp_result[1] - 7.0).abs() < tolerances::EXACT_F64);

        let mp = q8_params(&matrix);
        let vp = q8_params(&vector);
        let qm = quantize_q8(&matrix, &mp);
        let qv = quantize_q8(&vector, &vp);
        let q_result = gemv_q8(&qm, &qv, 2, 2, &mp, &vp);
        let err = relative_l2_error(&q_result, &fp_result);
        assert!(err < 0.05, "Q8 GEMV L2 error {err} < 5%");
    }

    #[test]
    fn q8_clamp_range() {
        let data = vec![1000.0, -1000.0];
        let params = q8_params(&data);
        let q = quantize_q8(&data, &params);
        assert_eq!(q[0], 127);
        assert_eq!(q[1], -127);
    }

    #[test]
    fn q4_clamp_range() {
        let data = vec![1000.0, -1000.0];
        let params = q4_params(&data);
        let q = quantize_q4(&data, &params);
        assert_eq!(q[0], 7);
        assert_eq!(q[1], -7);
    }

    #[test]
    fn relative_l2_zero_reference() {
        let a = vec![0.1, 0.2];
        let b = vec![0.0, 0.0];
        let err = relative_l2_error(&a, &b);
        assert!(err > 0.0);
    }

    #[test]
    fn gemv_q4_small() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0];
        let vector = vec![1.0, 1.0];
        let fp_result = gemv_f64(&matrix, &vector, 2, 2);

        let mp = q4_params(&matrix);
        let vp = q4_params(&vector);
        let qm = quantize_q4(&matrix, &mp);
        let qv = quantize_q4(&vector, &vp);
        let q_result = gemv_q4(&qm, &qv, 2, 2, &mp, &vp);
        let err = relative_l2_error(&q_result, &fp_result);
        assert!(err < 0.15, "Q4 GEMV L2 error {err} < 15%");
    }

    #[test]
    fn relative_l2_exact_match() {
        let a = vec![1.0, 2.0, 3.0];
        let err = relative_l2_error(&a, &a);
        assert!(
            err < tolerances::ZERO_DETECTION,
            "identical vectors → 0 error"
        );
    }

    #[test]
    fn gemv_f64_3x2() {
        let m = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v = vec![3.0, 5.0];
        let r = gemv_f64(&m, &v, 3, 2);
        assert!((r[0] - 3.0).abs() < tolerances::EXACT_F64);
        assert!((r[1] - 5.0).abs() < tolerances::EXACT_F64);
        assert!((r[2] - 8.0).abs() < tolerances::EXACT_F64);
    }
}
