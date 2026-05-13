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
//!
//! # I/O strategy
//!
//! - **JSON baselines**: Streamed via `serde_json::from_reader` with
//!   `BufReader` — O(record) memory, compliant with streaming I/O spec.
//! - **Safetensors**: Loaded via `std::fs::read` (full buffer). The
//!   `safetensors::SafeTensors::deserialize` API requires `&[u8]` — there
//!   is no streaming/incremental parse mode. The crate documents
//!   `memmap2::MmapOptions` as the zero-copy path, but that requires
//!   `unsafe` at the call site, which conflicts with `forbid(unsafe_code)`.
//!
//! Evolution path: when barracuda provides a safe mmap abstraction (behind
//! a capability trait), or the `safetensors` crate adds a safe streaming
//! API, the loader should evolve to use it. Current files are <100 MB so
//! full buffering is acceptable.
//!
//! # `NestGate` content-addressed storage
//!
//! When a `NestGate` primal is available, weights can be stored and
//! retrieved by BLAKE3 hash via `content.put` / `content.get`. This
//! enables cross-session and cross-spring model artifact sharing with
//! automatic deduplication. See [`store_to_nestgate`] and
//! [`load_safetensors_from_nestgate`].

use crate::error::IpcError;
use crate::ipc;

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Metadata for a loaded weight tensor.
#[derive(Debug, Clone)]
pub struct WeightTensor {
    /// Tensor name as stored in the source file.
    pub name: String,
    /// Flattened weight values upcast to `f64`.
    pub data: Vec<f64>,
    /// Original tensor shape (dimensions).
    pub shape: Vec<usize>,
    /// Row count for the interpreted matrix view.
    pub rows: usize,
    /// Column count for the interpreted matrix view.
    pub cols: usize,
    /// Source dtype label (e.g. `F32`) before upcast.
    pub dtype: String,
}

/// Summary of all weight tensors in a model file.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Source path or identifier for this load (often the file path).
    pub source: String,
    /// All weight tensors extracted from the file.
    pub tensors: Vec<WeightTensor>,
    /// Arbitrary string metadata from the format (e.g. safetensors header).
    pub metadata: HashMap<String, String>,
}

/// JSON baseline format for weight matrices.
#[derive(Debug, Deserialize)]
pub struct WeightBaseline {
    /// Model name in the baseline JSON.
    pub model_name: String,
    /// Layer name for this weight matrix.
    pub layer_name: String,
    /// Flattened matrix entries in row-major order.
    pub weights: Vec<f64>,
    /// Row count `m` of the weight matrix.
    pub m: usize,
    /// Column count `n` of the weight matrix.
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

    let mut result: Vec<(String, Vec<usize>, String)> = tensors
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            let shape = view.shape().to_vec();
            let dtype = format!("{:?}", view.dtype());
            (name, shape, dtype)
        })
        .collect();
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

/// Store a local file's contents to `NestGate` via `content.put`.
///
/// Reads the file, base64-encodes it, and stores it as content-addressed
/// data. Returns the BLAKE3 hash that can be used to retrieve the weights
/// in future sessions or from other springs.
///
/// # Errors
///
/// Returns an error if the file cannot be read, the [`ipc::IpcMathClient`]
/// has no `NestGate` discovered, or the IPC call fails.
pub fn store_to_nestgate(
    path: &Path,
    client: &ipc::IpcMathClient,
    content_type: Option<&str>,
) -> Result<String, IpcError> {
    use base64::Engine;

    let raw = std::fs::read(path)
        .map_err(|e| IpcError::Other(format!("read {}: {e}", path.display())))?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&raw);

    let ct = content_type.unwrap_or("application/x-safetensors");
    let result = client.content_put(&encoded, Some(ct))?;
    Ok(result.hash)
}

/// Load a safetensors model from `NestGate` by BLAKE3 hash.
///
/// Retrieves the base64-encoded payload via `content.get`, decodes it,
/// and deserializes as safetensors. Returns the same [`ModelWeights`]
/// structure as [`load_all_weight_matrices`].
///
/// # Errors
///
/// Returns an error if the hash is not found, decoding fails, or the
/// payload is not valid safetensors.
pub fn load_safetensors_from_nestgate(
    hash: &str,
    client: &ipc::IpcMathClient,
) -> Result<ModelWeights, IpcError> {
    use base64::Engine;

    let get_result = client.content_get(hash)?;
    let raw = base64::engine::general_purpose::STANDARD
        .decode(&get_result.data)
        .map_err(|e| IpcError::Other(format!("base64 decode: {e}")))?;

    let tensors = safetensors::SafeTensors::deserialize(&raw)
        .map_err(|e| IpcError::Other(format!("safetensors parse: {e}")))?;

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
        let data = upcast_to_f64(&view)
            .map_err(|e| IpcError::Other(format!("upcast {name}: {e}")))?;

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
        source: format!("nestgate:{hash}"),
        tensors: weight_tensors,
        metadata: HashMap::new(),
    })
}

/// Load a single tensor from `NestGate` by BLAKE3 hash and tensor name.
///
/// Combines content retrieval with safetensors deserialization and single
/// tensor extraction.
///
/// # Errors
///
/// Returns an error if the hash is not found, the tensor name is missing,
/// or the payload is not valid safetensors.
pub fn load_safetensors_layer_from_nestgate(
    hash: &str,
    tensor_name: &str,
    client: &ipc::IpcMathClient,
) -> Result<WeightTensor, IpcError> {
    use base64::Engine;

    let get_result = client.content_get(hash)?;
    let raw = base64::engine::general_purpose::STANDARD
        .decode(&get_result.data)
        .map_err(|e| IpcError::Other(format!("base64 decode: {e}")))?;

    let tensors = safetensors::SafeTensors::deserialize(&raw)
        .map_err(|e| IpcError::Other(format!("safetensors parse: {e}")))?;

    let view = tensors
        .tensor(tensor_name)
        .map_err(|e| IpcError::Other(format!("tensor '{tensor_name}': {e}")))?;

    let shape: Vec<usize> = view.shape().to_vec();
    let dtype = format!("{:?}", view.dtype());
    let data = upcast_to_f64(&view)
        .map_err(|e| IpcError::Other(format!("upcast: {e}")))?;

    let (rows, cols) = match shape.len() {
        0 => (1, 1),
        1 => (1, shape[0]),
        2 => (shape[0], shape[1]),
        _ => {
            let cols = *shape.last().ok_or_else(|| IpcError::Other("empty shape".into()))?;
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
    fn json_baseline_round_trip() {
        let json =
            r#"{"model_name":"test","layer_name":"fc1","weights":[1.0,2.0,3.0,4.0],"m":2,"n":2}"#;
        let tmp = std::env::temp_dir().join("test_weight_baseline.json");
        std::fs::write(&tmp, json)
            .expect("failed to write temporary test JSON — check disk space and permissions");
        let baseline = load_json_weights(&tmp).expect(
            "failed to parse test JSON that was just written — indicates serialization bug",
        );
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
        std::fs::write(&tmp, json)
            .expect("failed to write temporary test JSON — check disk space and permissions");
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
    fn list_safetensors_nonexistent_returns_err() {
        let result = list_safetensors(Path::new("/nonexistent/model.safetensors"));
        assert!(result.is_err());
    }

    #[test]
    fn load_safetensors_layer_nonexistent_returns_err() {
        let result = load_safetensors_layer(Path::new("/nonexistent/model.safetensors"), "layer0");
        assert!(result.is_err());
    }

    #[test]
    fn load_all_weight_matrices_nonexistent_returns_err() {
        let result = load_all_weight_matrices(Path::new("/nonexistent/model.safetensors"));
        assert!(result.is_err());
    }

    #[test]
    fn weight_tensor_debug_and_clone() {
        let wt = WeightTensor {
            name: "test.weight".into(),
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            rows: 2,
            cols: 2,
            dtype: "F32".into(),
        };
        let cloned = wt.clone();
        assert_eq!(format!("{wt:?}"), format!("{cloned:?}"));
    }

    #[test]
    fn model_weights_debug_and_clone() {
        let mw = ModelWeights {
            source: "test".into(),
            tensors: vec![],
            metadata: HashMap::new(),
        };
        let cloned = mw.clone();
        assert_eq!(cloned.source, mw.source);
        assert!(cloned.tensors.is_empty());
    }

    #[test]
    fn json_malformed_returns_err() {
        let tmp = std::env::temp_dir().join("test_weight_malformed.json");
        std::fs::write(&tmp, "not json")
            .expect("failed to write temporary test file — check disk space and permissions");
        let result = load_json_weights(&tmp);
        assert!(result.is_err(), "malformed JSON should fail");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn store_to_nestgate_nonexistent_file_returns_err() {
        let client = crate::ipc::IpcMathClient::discover();
        let result = store_to_nestgate(Path::new("/nonexistent/model.safetensors"), &client, None);
        assert!(result.is_err());
    }

    #[test]
    fn load_safetensors_from_nestgate_returns_err_without_primal() {
        let client = crate::ipc::IpcMathClient::discover();
        let result = load_safetensors_from_nestgate(&"deadbeef".repeat(8), &client);
        assert!(result.is_err());
    }

    #[test]
    fn load_safetensors_layer_from_nestgate_returns_err_without_primal() {
        let client = crate::ipc::IpcMathClient::discover();
        let result = load_safetensors_layer_from_nestgate(
            &"deadbeef".repeat(8),
            "layer.weight",
            &client,
        );
        assert!(result.is_err());
    }
}
