// SPDX-License-Identifier: AGPL-3.0-or-later

use super::dtype::upcast_to_f64;
use super::{ModelWeights, WeightTensor};
use crate::error::IpcError;
use crate::ipc;
use std::collections::HashMap;
use std::path::Path;

/// Store a local file's contents to `NestGate` via `content.put`.
///
/// Reads the file, base64-encodes it, and stores it as content-addressed
/// data. Returns the BLAKE3 hash that can be used to retrieve the weights
/// in future sessions or from other springs.
///
/// Uses `nest.store` signal dispatch when a [`CompositionContext`] is
/// available (Wave 17 — biomeOS manages DAG + spine + braid provenance).
/// Falls back to direct `content.put` via [`ipc::IpcMathClient`] when
/// running standalone or when signal dispatch is unavailable.
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
/// structure as [`load_all_weight_matrices`](super::load_all_weight_matrices).
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
        let data =
            upcast_to_f64(&view).map_err(|e| IpcError::Other(format!("upcast {name}: {e}")))?;

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
    let data = upcast_to_f64(&view).map_err(|e| IpcError::Other(format!("upcast: {e}")))?;

    let (rows, cols) = match shape.len() {
        0 => (1, 1),
        1 => (1, shape[0]),
        2 => (shape[0], shape[1]),
        _ => {
            let cols = *shape
                .last()
                .ok_or_else(|| IpcError::Other("empty shape".into()))?;
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
mod tests {
    use super::*;
    use std::path::Path;

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
        let result =
            load_safetensors_layer_from_nestgate(&"deadbeef".repeat(8), "layer.weight", &client);
        assert!(result.is_err());
    }

    #[test]
    fn store_to_nestgate_custom_content_type_still_err_on_missing_file() {
        let client = crate::ipc::IpcMathClient::discover();
        let result = store_to_nestgate(
            Path::new("/nonexistent/custom.safetensors"),
            &client,
            Some("application/custom"),
        );
        assert!(result.is_err());
    }
}
