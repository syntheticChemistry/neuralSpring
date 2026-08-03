// SPDX-License-Identifier: AGPL-3.0-or-later

use super::dtype::upcast_to_f64;
use super::{ModelWeights, WeightTensor};
use std::collections::HashMap;
use std::path::Path;

/// List all tensor names and shapes in a safetensors file.
///
/// # Errors
///
/// Returns `Err` if the file cannot be read or is not valid safetensors.
pub fn list_safetensors(path: &Path) -> crate::error::Result<Vec<(String, Vec<usize>, String)>> {
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
pub fn load_safetensors_layer(
    path: &Path,
    tensor_name: &str,
) -> crate::error::Result<WeightTensor> {
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

/// Load all 2D weight matrices from a safetensors file.
///
/// Filters to tensors with >= 2 dimensions (skips biases, norms, embeddings
/// unless they happen to be 2D). Returns sorted by name.
///
/// # Errors
///
/// Returns `Err` if the file cannot be read, is not valid safetensors,
/// or any tensor has an unsupported dtype.
pub fn load_all_weight_matrices(path: &Path) -> crate::error::Result<ModelWeights> {
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

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;
    use std::path::Path;

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

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn f16_bytes(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|bits| bits.to_le_bytes()).collect()
    }

    fn write_safetensors(path: &Path, tensors: &[(&str, safetensors::Dtype, Vec<usize>, Vec<u8>)]) {
        use safetensors::tensor::{TensorView, serialize};
        use std::collections::HashMap;

        let map: HashMap<String, TensorView<'_>> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                let view = TensorView::new(*dtype, shape.clone(), data)
                    .unwrap_or_else(|e| panic!("tensor {name}: {e}"));
                (name.to_string(), view)
            })
            .collect();
        let bytes = serialize(&map, None).unwrap_or_else(|e| panic!("serialize: {e}"));
        std::fs::write(path, bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }

    fn temp_safetensors(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("neuralspring_weight_loader_{name}.safetensors"))
    }

    #[test]
    fn list_safetensors_lists_tensors_sorted() {
        let path = temp_safetensors("list_sorted");
        write_safetensors(
            &path,
            &[
                (
                    "z.weight",
                    safetensors::Dtype::F32,
                    vec![2, 2],
                    f32_bytes(&[0.0, 1.0, 2.0, 3.0]),
                ),
                (
                    "a.bias",
                    safetensors::Dtype::F32,
                    vec![2],
                    f32_bytes(&[0.5, 1.5]),
                ),
            ],
        );

        let listed = list_safetensors(&path).expect("list valid safetensors");
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].0, "a.bias");
        assert_eq!(listed[0].1, vec![2]);
        assert!(listed[0].2.contains("F32"));
        assert_eq!(listed[1].0, "z.weight");
        assert_eq!(listed[1].1, vec![2, 2]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn list_safetensors_invalid_bytes_returns_err() {
        let path = temp_safetensors("list_invalid");
        std::fs::write(&path, b"not-safetensors").expect("write temp file");
        let result = list_safetensors(&path);
        assert!(result.is_err(), "garbage bytes should fail parse");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_safetensors_layer_2d_f32() {
        let path = temp_safetensors("layer_2d");
        write_safetensors(
            &path,
            &[(
                "fc.weight",
                safetensors::Dtype::F32,
                vec![2, 2],
                f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
            )],
        );

        let layer = load_safetensors_layer(&path, "fc.weight").expect("load 2d layer");
        assert_eq!(layer.name, "fc.weight");
        assert_eq!(layer.rows, 2);
        assert_eq!(layer.cols, 2);
        assert_eq!(layer.shape, vec![2, 2]);
        assert_eq!(layer.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(layer.dtype.contains("F32"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_safetensors_layer_1d_bias() {
        let path = temp_safetensors("layer_1d");
        write_safetensors(
            &path,
            &[(
                "bias",
                safetensors::Dtype::F32,
                vec![3],
                f32_bytes(&[0.1, 0.2, 0.3]),
            )],
        );

        let layer = load_safetensors_layer(&path, "bias").expect("load 1d bias");
        assert_eq!(layer.rows, 1);
        assert_eq!(layer.cols, 3);
        assert_eq!(layer.data.len(), 3);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_safetensors_layer_3d_flattens_rows() {
        let path = temp_safetensors("layer_3d");
        let values: Vec<f32> = (0..24_i16).map(f32::from).collect();
        write_safetensors(
            &path,
            &[(
                "conv.weight",
                safetensors::Dtype::F32,
                vec![2, 3, 4],
                f32_bytes(&values),
            )],
        );

        let layer = load_safetensors_layer(&path, "conv.weight").expect("load 3d layer");
        assert_eq!(layer.cols, 4);
        assert_eq!(layer.rows, 6);
        assert_eq!(layer.data.len(), 24);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_safetensors_layer_scalar_shape() {
        let path = temp_safetensors("layer_scalar");
        write_safetensors(
            &path,
            &[(
                "scalar",
                safetensors::Dtype::F32,
                vec![],
                f32_bytes(&[42.0]),
            )],
        );

        let layer = load_safetensors_layer(&path, "scalar").expect("load scalar");
        assert_eq!(layer.rows, 1);
        assert_eq!(layer.cols, 1);
        assert!((layer.data[0] - 42.0).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_safetensors_layer_missing_tensor_returns_err() {
        let path = temp_safetensors("layer_missing");
        write_safetensors(
            &path,
            &[(
                "present",
                safetensors::Dtype::F32,
                vec![2, 2],
                f32_bytes(&[1.0, 0.0, 0.0, 1.0]),
            )],
        );

        let result = load_safetensors_layer(&path, "absent");
        assert!(result.is_err(), "missing tensor name should fail");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_safetensors_layer_invalid_bytes_returns_err() {
        let path = temp_safetensors("layer_invalid");
        std::fs::write(&path, b"bad-header").expect("write temp file");
        let result = load_safetensors_layer(&path, "any");
        assert!(result.is_err(), "invalid safetensors should fail parse");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_safetensors_layer_f16_and_bf16() {
        let path = temp_safetensors("layer_low_precision");
        write_safetensors(
            &path,
            &[
                (
                    "f16.weight",
                    safetensors::Dtype::F16,
                    vec![1, 2],
                    f16_bytes(&[0x3C00, 0xBC00]), // 1.0, -1.0
                ),
                (
                    "bf16.weight",
                    safetensors::Dtype::BF16,
                    vec![1, 2],
                    f16_bytes(&[0x3F80, 0xBF80]), // 1.0, -1.0
                ),
            ],
        );

        let f16_layer = load_safetensors_layer(&path, "f16.weight").expect("load f16");
        assert!((f16_layer.data[0] - 1.0).abs() < 1e-6);
        assert!((f16_layer.data[1] - (-1.0)).abs() < 1e-6);

        let bf16_layer = load_safetensors_layer(&path, "bf16.weight").expect("load bf16");
        assert!((bf16_layer.data[0] - 1.0).abs() < 1e-6);
        assert!((bf16_layer.data[1] - (-1.0)).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_safetensors_layer_unsupported_dtype_returns_err() {
        let path = temp_safetensors("layer_i32");
        write_safetensors(
            &path,
            &[(
                "int_tensor",
                safetensors::Dtype::I32,
                vec![2],
                vec![1, 0, 0, 0, 2, 0, 0, 0],
            )],
        );

        let result = load_safetensors_layer(&path, "int_tensor");
        let err = result
            .expect_err("I32 dtype should be rejected")
            .to_string();
        assert!(
            err.contains("unsupported dtype"),
            "expected unsupported dtype error, got: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_all_weight_matrices_filters_and_sorts() {
        let path = temp_safetensors("all_matrices");
        write_safetensors(
            &path,
            &[
                (
                    "z.weight",
                    safetensors::Dtype::F32,
                    vec![2, 2],
                    f32_bytes(&[4.0, 3.0, 2.0, 1.0]),
                ),
                (
                    "a.weight",
                    safetensors::Dtype::F32,
                    vec![2, 2],
                    f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
                ),
                (
                    "bias",
                    safetensors::Dtype::F32,
                    vec![2],
                    f32_bytes(&[0.0, 0.0]),
                ),
                (
                    "tiny",
                    safetensors::Dtype::F32,
                    vec![1, 2],
                    f32_bytes(&[1.0, 2.0]),
                ),
                (
                    "embedding",
                    safetensors::Dtype::F32,
                    vec![2, 1],
                    f32_bytes(&[1.0, 2.0]),
                ),
            ],
        );

        let model = load_all_weight_matrices(&path).expect("load matrices");
        assert_eq!(model.source, path.display().to_string());
        assert_eq!(model.tensors.len(), 2);
        assert_eq!(model.tensors[0].name, "a.weight");
        assert_eq!(model.tensors[1].name, "z.weight");
        assert!(model.metadata.is_empty());
        assert_eq!(model.tensors[0].data, vec![1.0, 2.0, 3.0, 4.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_all_weight_matrices_invalid_bytes_returns_err() {
        let path = temp_safetensors("all_invalid");
        std::fs::write(&path, b"not-valid").expect("write temp file");
        let result = load_all_weight_matrices(&path);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_all_weight_matrices_f64_dtype() {
        let path = temp_safetensors("all_f64");
        let data: Vec<u8> = [1.0_f64, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        write_safetensors(
            &path,
            &[("dense.weight", safetensors::Dtype::F64, vec![2, 2], data)],
        );

        let model = load_all_weight_matrices(&path).expect("load f64 matrices");
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.tensors[0].data, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(model.tensors[0].dtype.contains("F64"));

        let _ = std::fs::remove_file(&path);
    }
}
