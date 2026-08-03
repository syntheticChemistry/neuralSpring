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

mod dtype;
mod nestgate;
mod safetensors;

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

pub use nestgate::{
    load_safetensors_from_nestgate, load_safetensors_layer_from_nestgate, store_to_nestgate,
};
pub use safetensors::{list_safetensors, load_all_weight_matrices, load_safetensors_layer};

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

/// Load a weight baseline from JSON (neuralSpring `control/` pattern).
///
/// # Errors
///
/// Returns `Err` if the file cannot be opened or the JSON is malformed.
pub fn load_json_weights(path: &Path) -> crate::error::Result<WeightBaseline> {
    let file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    Ok(serde_json::from_reader(reader).map_err(|e| format!("parse JSON: {e}"))?)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;
    use std::path::Path;

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
    fn json_baseline_all_fields() {
        let json = r#"{
            "model_name":"mlp",
            "layer_name":"hidden",
            "weights":[10.0,20.0,30.0,40.0],
            "m":2,
            "n":2
        }"#;
        let tmp = std::env::temp_dir().join("test_weight_all_fields.json");
        std::fs::write(&tmp, json).expect("write temp json");
        let baseline = load_json_weights(&tmp).expect("parse json");
        assert_eq!(baseline.model_name, "mlp");
        assert_eq!(baseline.layer_name, "hidden");
        assert_eq!(baseline.weights, vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!(baseline.m, 2);
        assert_eq!(baseline.n, 2);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn weight_baseline_debug() {
        let baseline = WeightBaseline {
            model_name: "m".into(),
            layer_name: "l".into(),
            weights: vec![1.0],
            m: 1,
            n: 1,
        };
        let debug = format!("{baseline:?}");
        assert!(debug.contains("model_name"));
        assert!(debug.contains("layer_name"));
    }
}
