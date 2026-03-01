// SPDX-License-Identifier: AGPL-3.0-or-later

//! Load neural network weight matrices from safetensors and JSON formats.
//!
//! Supports two formats:
//! - **safetensors** (`HuggingFace` standard): `.safetensors` files from
//!   `transformers`, `diffusers`, or manual conversion
//! - **JSON baselines**: `control/weight_spectral/` files following the
//!   neuralSpring baseline pattern
//!
//! All loaders upcast to `f64` for `eigh_f64` precision, regardless of
//! the stored dtype (f16, bf16, f32).

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Metadata for a loaded weight tensor.
#[derive(Debug, Clone)]
pub struct WeightTensor {
    pub name: String,
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
    pub dtype: String,
}

/// Summary of all weight tensors in a model file.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub source: String,
    pub tensors: Vec<WeightTensor>,
    pub metadata: HashMap<String, String>,
}

/// JSON baseline format for weight matrices.
#[derive(Debug, Deserialize)]
pub struct WeightBaseline {
    pub model_name: String,
    pub layer_name: String,
    pub weights: Vec<f64>,
    pub m: usize,
    pub n: usize,
}

/// List all tensor names and shapes in a safetensors file.
///
/// # Errors
///
/// Returns `Err` if the file cannot be read or is not valid safetensors.
pub fn list_safetensors(path: &Path) -> Result<Vec<(String, Vec<usize>, String)>, String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let tensors =
        safetensors::SafeTensors::deserialize(&data).map_err(|e| format!("parse: {e}"))?;

    let mut result = Vec::new();
    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let dtype = format!("{:?}", view.dtype());
        result.push((name.clone(), shape, dtype));
    }
    result.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(result)
}

/// Load a single tensor from a safetensors file, upcast to f64.
///
/// For 2D tensors, shape is `[rows, cols]`.
/// For 1D tensors (biases), returns `(data, 1, len)`.
/// For higher-dimensional tensors, flattens all but the last dim into rows.
///
/// # Errors
///
/// Returns `Err` if the file cannot be read, the tensor name is not found,
/// or the dtype is unsupported.
pub fn load_safetensors_layer(path: &Path, tensor_name: &str) -> Result<WeightTensor, String> {
    let raw = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let tensors = safetensors::SafeTensors::deserialize(&raw).map_err(|e| format!("parse: {e}"))?;

    let view = tensors
        .tensor(tensor_name)
        .map_err(|e| format!("tensor '{tensor_name}': {e}"))?;

    let shape: Vec<usize> = view.shape().to_vec();
    let dtype = format!("{:?}", view.dtype());
    let data = upcast_to_f64(&view)?;

    let (rows, cols) = match shape.len() {
        0 => (1, 1),
        1 => (1, shape[0]),
        2 => (shape[0], shape[1]),
        _ => {
            let cols = shape.last().copied().ok_or("empty shape in >2D tensor")?;
            let rows = data.len() / cols;
            (rows, cols)
        }
    };

    Ok(WeightTensor {
        name: tensor_name.to_string(),
        data,
        shape,
        rows,
        cols,
        dtype,
    })
}

/// Load all 2D weight tensors from a safetensors file.
///
/// Filters to tensors with >= 2 dimensions (skips biases, norms, embeddings
/// unless they happen to be 2D). Returns sorted by name.
///
/// # Errors
///
/// Returns `Err` if the file cannot be read, is not valid safetensors,
/// or any tensor has an unsupported dtype.
pub fn load_all_weight_matrices(path: &Path) -> Result<ModelWeights, String> {
    let raw = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let tensors = safetensors::SafeTensors::deserialize(&raw).map_err(|e| format!("parse: {e}"))?;

    let mut weight_tensors = Vec::new();

    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        if shape.len() < 2 {
            continue;
        }

        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        if rows < 2 || cols < 2 {
            continue;
        }

        let dtype = format!("{:?}", view.dtype());
        let data = upcast_to_f64(&view)?;

        weight_tensors.push(WeightTensor {
            name: name.clone(),
            data,
            shape,
            rows,
            cols,
            dtype,
        });
    }

    weight_tensors.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(ModelWeights {
        source: path.display().to_string(),
        tensors: weight_tensors,
        metadata: HashMap::new(),
    })
}

/// Load a weight baseline from JSON (neuralSpring `control/` pattern).
///
/// # Errors
///
/// Returns `Err` if the file cannot be opened or the JSON is malformed.
pub fn load_json_weights(path: &Path) -> Result<WeightBaseline, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    serde_json::from_reader(reader).map_err(|e| format!("parse JSON: {e}"))
}

fn upcast_to_f64(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f64>, String> {
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
        other => Err(format!("unsupported dtype: {other:?}")),
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
#[allow(clippy::float_cmp, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip() {
        let val: f32 = 1.5;
        #[allow(clippy::cast_possible_truncation)]
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
    fn json_baseline_round_trip() {
        let json =
            r#"{"model_name":"test","layer_name":"fc1","weights":[1.0,2.0,3.0,4.0],"m":2,"n":2}"#;
        let tmp = std::env::temp_dir().join("test_weight_baseline.json");
        std::fs::write(&tmp, json).expect("write test json");
        let baseline = load_json_weights(&tmp).expect("parse test json");
        assert_eq!(baseline.model_name, "test");
        assert_eq!(baseline.m, 2);
        assert_eq!(baseline.n, 2);
        assert_eq!(baseline.weights.len(), 4);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn json_missing_field_returns_err() {
        let json = r#"{"model_name":"test","weights":[1.0],"m":1,"n":1}"#;
        let tmp = std::env::temp_dir().join("test_weight_missing_field.json");
        std::fs::write(&tmp, json).expect("write test json");
        let result = load_json_weights(&tmp);
        assert!(result.is_err(), "missing layer_name should fail");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn json_nonexistent_file_returns_err() {
        let result = load_json_weights(Path::new("/nonexistent/path.json"));
        assert!(result.is_err());
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
}
