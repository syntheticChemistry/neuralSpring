// SPDX-License-Identifier: AGPL-3.0-or-later

pub(super) fn upcast_to_f64(
    view: &safetensors::tensor::TensorView<'_>,
) -> crate::error::Result<Vec<f64>> {
    use safetensors::Dtype;

    let bytes = view.data();

    match view.dtype() {
        Dtype::F32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err("F32 data length not aligned".into());
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect())
        }
        Dtype::F64 => {
            if !bytes.len().is_multiple_of(8) {
                return Err("F64 data length not aligned".into());
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect())
        }
        Dtype::F16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err("F16 data length not aligned".into());
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f64::from(f16_to_f32(bits))
                })
                .collect())
        }
        Dtype::BF16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err("BF16 data length not aligned".into());
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f64::from(bf16_to_f32(bits))
                })
                .collect())
        }
        other => Err(format!("unsupported dtype: {other:?}").into()),
    }
}

/// IEEE 754 half-precision (binary16) to single-precision conversion.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1);
    let exp = u32::from((bits >> 10) & 0x1F);
    let frac = u32::from(bits & 0x3FF);

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal: normalize by shifting mantissa left until leading 1
        let mut shift = 0_u32;
        let mut f = frac;
        while (f & 0x400) == 0 {
            f <<= 1;
            shift += 1;
        }
        f &= 0x3FF;
        let exp32 = 127 - 15 + 1 - shift;
        f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13))
    } else if exp == 31 {
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
        }
    } else {
        let exp32 = exp + (127 - 15);
        f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
    }
}

/// bfloat16 to single-precision: upper 16 bits of f32 encoding.
const fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
#[expect(
    clippy::float_cmp,
    clippy::expect_used,
    reason = "tests verify exact round-trip fidelity"
)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip() {
        let val: f32 = 1.5;
        let bits = (val.to_bits() >> 16) as u16;
        let recovered = bf16_to_f32(bits);
        assert!((recovered - val).abs() < 1e-6);
    }

    #[test]
    fn f16_special_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C01).is_nan());
    }

    #[test]
    fn f16_normal_values() {
        let one = f16_to_f32(0x3C00);
        assert!((one - 1.0).abs() < 1e-6, "f16 1.0 got {one}");

        let neg_two = f16_to_f32(0xC000);
        assert!((neg_two - (-2.0)).abs() < 1e-6, "f16 -2.0 got {neg_two}");

        let half = f16_to_f32(0x3800);
        assert!((half - 0.5).abs() < 1e-6, "f16 0.5 got {half}");
    }

    #[test]
    fn f16_subnormal() {
        let tiny = f16_to_f32(0x0001);
        assert!(tiny > 0.0, "smallest f16 subnormal must be positive");
        assert!(
            tiny < 1e-6,
            "smallest f16 subnormal must be tiny, got {tiny}"
        );
    }

    #[test]
    fn f16_negative_infinity() {
        let neg_inf = f16_to_f32(0xFC00);
        assert!(neg_inf.is_infinite() && neg_inf.is_sign_negative());
    }

    #[test]
    fn bf16_special_values() {
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        let bf16_inf = bf16_to_f32(0x7F80);
        assert!(bf16_inf.is_infinite());
        let bf16_nan = bf16_to_f32(0x7FC0);
        assert!(bf16_nan.is_nan());
    }

    #[test]
    fn bf16_normal_values() {
        let bf16_one = bf16_to_f32(0x3F80);
        assert!((bf16_one - 1.0).abs() < 1e-6, "bf16 1.0 got {bf16_one}");

        let bf16_neg = bf16_to_f32(0xC000);
        assert!((bf16_neg - (-2.0)).abs() < 1e-6, "bf16 -2.0 got {bf16_neg}");
    }

    #[test]
    fn upcast_f32_alignment_check() {
        use safetensors::tensor::TensorView;
        let bad_bytes: &[u8] = &[0, 0, 0];
        let view = TensorView::new(safetensors::Dtype::F32, vec![1], bad_bytes);
        if let Ok(v) = view {
            let result = upcast_to_f64(&v);
            assert!(result.is_err(), "misaligned F32 should fail");
        }
    }

    #[test]
    fn upcast_f64_alignment_check() {
        use safetensors::tensor::TensorView;
        let bad_bytes: &[u8] = &[0; 7];
        let view = TensorView::new(safetensors::Dtype::F64, vec![1], bad_bytes);
        if let Ok(v) = view {
            let result = upcast_to_f64(&v);
            assert!(result.is_err(), "misaligned F64 should fail");
        }
    }

    #[test]
    fn upcast_f32_valid_roundtrip() {
        use safetensors::tensor::TensorView;
        let val: f32 = std::f32::consts::PI;
        let bytes = val.to_le_bytes();
        let view = TensorView::new(safetensors::Dtype::F32, vec![1], &bytes).expect("valid f32");
        let result = upcast_to_f64(&view).expect("upcast f32");
        assert!((result[0] - f64::from(val)).abs() < 1e-6);
    }

    #[test]
    fn upcast_f64_valid_roundtrip() {
        use safetensors::tensor::TensorView;
        let val: f64 = std::f64::consts::E;
        let bytes = val.to_le_bytes();
        let view = TensorView::new(safetensors::Dtype::F64, vec![1], &bytes).expect("valid f64");
        let result = upcast_to_f64(&view).expect("upcast f64");
        assert!((result[0] - val).abs() < 1e-15);
    }

    #[test]
    fn upcast_f16_valid_roundtrip() {
        use safetensors::tensor::TensorView;
        let bits: u16 = 0x3C00; // 1.0 in f16
        let bytes = bits.to_le_bytes();
        let view = TensorView::new(safetensors::Dtype::F16, vec![1], &bytes).expect("valid f16");
        let result = upcast_to_f64(&view).expect("upcast f16");
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn upcast_bf16_valid_roundtrip() {
        use safetensors::tensor::TensorView;
        let bits: u16 = 0x3F80; // 1.0 in bf16
        let bytes = bits.to_le_bytes();
        let view = TensorView::new(safetensors::Dtype::BF16, vec![1], &bytes).expect("valid bf16");
        let result = upcast_to_f64(&view).expect("upcast bf16");
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn upcast_f16_alignment_check() {
        use safetensors::tensor::TensorView;
        let bad_bytes: &[u8] = &[0, 0, 0];
        let view = TensorView::new(safetensors::Dtype::F16, vec![1], bad_bytes);
        if let Ok(v) = view {
            let result = upcast_to_f64(&v);
            assert!(result.is_err(), "misaligned F16 should fail");
        }
    }

    #[test]
    fn upcast_bf16_alignment_check() {
        use safetensors::tensor::TensorView;
        let bad_bytes: &[u8] = &[0, 0, 0];
        let view = TensorView::new(safetensors::Dtype::BF16, vec![1], bad_bytes);
        if let Ok(v) = view {
            let result = upcast_to_f64(&v);
            assert!(result.is_err(), "misaligned BF16 should fail");
        }
    }

    #[test]
    fn upcast_unsupported_dtype_returns_err() {
        use safetensors::tensor::TensorView;
        let bytes = [1_i32, 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let view = TensorView::new(safetensors::Dtype::I32, vec![2], &bytes).expect("valid i32");
        let result = upcast_to_f64(&view);
        let err = result.expect_err("I32 should be unsupported").to_string();
        assert!(err.contains("unsupported dtype"), "got: {err}");
    }

    #[test]
    fn f16_negative_zero() {
        let neg_zero = f16_to_f32(0x8000);
        assert_eq!(neg_zero, -0.0);
        assert!(neg_zero.is_sign_negative());
    }

    #[test]
    fn f16_positive_infinity() {
        let pos_inf = f16_to_f32(0x7C00);
        assert!(pos_inf.is_infinite() && pos_inf.is_sign_positive());
    }
}
