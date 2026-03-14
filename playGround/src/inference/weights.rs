// SPDX-License-Identifier: AGPL-3.0-or-later

//! Load safetensors weights into barraCuda GPU tensors.
//!
//! Handles f16/bf16/f32/f64 → f32 conversion and organizes tensors
//! into per-layer weight sets for transformer inference.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use barracuda::prelude::{Tensor, WgpuDevice};

use crate::model_config::TransformerConfig;

/// A single named weight tensor on the GPU.
#[derive(Debug)]
pub struct GpuWeight {
    pub name: String,
    pub tensor: Tensor,
    pub original_shape: Vec<usize>,
}

/// All weights for a single transformer layer.
#[derive(Debug, Default)]
pub struct LayerWeights {
    pub attn_q_weight: Option<Tensor>,
    pub attn_k_weight: Option<Tensor>,
    pub attn_v_weight: Option<Tensor>,
    pub attn_out_weight: Option<Tensor>,
    pub attn_q_bias: Option<Tensor>,
    pub attn_k_bias: Option<Tensor>,
    pub attn_v_bias: Option<Tensor>,
    pub attn_out_bias: Option<Tensor>,
    pub ln1_weight: Option<Tensor>,
    pub ln1_bias: Option<Tensor>,
    pub ffn_up_weight: Option<Tensor>,
    pub ffn_down_weight: Option<Tensor>,
    pub ffn_up_bias: Option<Tensor>,
    pub ffn_down_bias: Option<Tensor>,
    pub ln2_weight: Option<Tensor>,
    pub ln2_bias: Option<Tensor>,
    // GPT-2 uses combined QKV projection
    pub attn_qkv_weight: Option<Tensor>,
    pub attn_qkv_bias: Option<Tensor>,
}

/// Complete model weights organized for inference.
#[derive(Debug)]
pub struct ModelWeights {
    pub token_embedding: Option<Tensor>,
    pub position_embedding: Option<Tensor>,
    pub ln_final_weight: Option<Tensor>,
    pub ln_final_bias: Option<Tensor>,
    pub lm_head_weight: Option<Tensor>,
    pub layers: Vec<LayerWeights>,
    pub unmatched: Vec<GpuWeight>,
}

/// IEEE 754 half-precision (binary16) to f32 conversion.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1F);
    let mantissa = u32::from(bits & 0x3FF);

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: convert to normalized f32
        let mut m = mantissa;
        let mut e: i32 = -14;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        #[expect(
            clippy::cast_sign_loss,
            reason = "exponent is always positive after bias"
        )]
        let f32_bits = sign | (((e + 127) as u32) << 23) | (m << 13);
        return f32::from_bits(f32_bits);
    }
    if exponent == 31 {
        let f32_bits = sign | 0x7F80_0000 | (mantissa << 13);
        return f32::from_bits(f32_bits);
    }

    let f32_bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    f32::from_bits(f32_bits)
}

/// `BFloat16` to f32 conversion (simple left-shift of the upper 16 bits).
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Reinterpret raw bytes as a typed slice, handling unaligned data.
fn safe_cast_slice<T: bytemuck::Pod>(data: &[u8]) -> Vec<T> {
    let size = std::mem::size_of::<T>();
    let count = data.len() / size;
    let mut out = vec![T::zeroed(); count];
    let dst = bytemuck::cast_slice_mut::<T, u8>(&mut out);
    dst.copy_from_slice(&data[..count * size]);
    out
}

/// Load f32 data from a safetensors tensor, converting from f16/bf16/f64 as needed.
fn tensor_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    use safetensors::Dtype;
    match view.dtype() {
        Dtype::F32 => safe_cast_slice::<f32>(view.data()),
        Dtype::F16 => {
            let raw: Vec<u16> = safe_cast_slice(view.data());
            raw.iter().map(|&bits| f16_to_f32(bits)).collect()
        }
        Dtype::BF16 => {
            let raw: Vec<u16> = safe_cast_slice(view.data());
            raw.iter().map(|&bits| bf16_to_f32(bits)).collect()
        }
        Dtype::F64 => {
            let raw: Vec<f64> = safe_cast_slice(view.data());
            #[expect(
                clippy::cast_possible_truncation,
                reason = "intentional f64→f32 downcast for GPU"
            )]
            raw.iter().map(|&v| v as f32).collect()
        }
        _ => {
            log::warn!(
                "Unsupported dtype {:?}, treating as f32 (may be wrong)",
                view.dtype()
            );
            safe_cast_slice::<f32>(view.data())
        }
    }
}

/// Load all tensors from safetensors files into GPU memory.
pub fn load_safetensors(
    paths: &[impl AsRef<Path>],
    device: &Arc<WgpuDevice>,
) -> Result<Vec<GpuWeight>> {
    let mut weights = Vec::new();

    for path in paths {
        let path = path.as_ref();
        let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let tensors = safetensors::SafeTensors::deserialize(&data)
            .with_context(|| format!("parsing safetensors {}", path.display()))?;

        for (name, view) in tensors.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            let f32_data = tensor_to_f32(&view);

            let tensor = Tensor::from_data(&f32_data, shape.clone(), device.clone())
                .with_context(|| format!("uploading tensor {name} to GPU"))?;

            weights.push(GpuWeight {
                name: name.clone(),
                tensor,
                original_shape: shape,
            });
        }
    }

    Ok(weights)
}

/// Organize raw GPU weights into per-layer structure based on config.
#[must_use]
pub fn organize_weights(raw: Vec<GpuWeight>, config: &TransformerConfig) -> ModelWeights {
    let mut model = ModelWeights {
        token_embedding: None,
        position_embedding: None,
        ln_final_weight: None,
        ln_final_bias: None,
        lm_head_weight: None,
        layers: (0..config.num_layers)
            .map(|_| LayerWeights::default())
            .collect(),
        unmatched: Vec::new(),
    };

    for w in raw {
        let name = &w.name;

        // Token and position embeddings
        if name.contains("wte")
            || name.contains("embed_tokens")
            || name == "model.embed_tokens.weight"
        {
            model.token_embedding = Some(w.tensor);
            continue;
        }
        if name.contains("wpe") || name.contains("embed_positions") {
            model.position_embedding = Some(w.tensor);
            continue;
        }

        // Final layer norm
        if name.contains("ln_f") || name.contains("model.norm") || name.contains("final_layer_norm")
        {
            if name.contains("weight") || !name.contains("bias") {
                model.ln_final_weight = Some(w.tensor);
            } else {
                model.ln_final_bias = Some(w.tensor);
            }
            continue;
        }

        // LM head
        if name.contains("lm_head") {
            model.lm_head_weight = Some(w.tensor);
            continue;
        }

        // Per-layer weights — extract layer index
        if let Some(layer_idx) = extract_layer_index(name) {
            if layer_idx >= config.num_layers {
                model.unmatched.push(w);
                continue;
            }
            let layer = &mut model.layers[layer_idx];
            assign_layer_weight(layer, name, w.tensor);
            continue;
        }

        model.unmatched.push(w);
    }

    model
}

fn extract_layer_index(name: &str) -> Option<usize> {
    // Patterns: "h.0.", "layers.0.", "model.layers.0.", "transformer.h.0."
    for part in name.split('.') {
        if let Ok(idx) = part.parse::<usize>() {
            return Some(idx);
        }
    }
    None
}

fn assign_layer_weight(layer: &mut LayerWeights, name: &str, tensor: Tensor) {
    // GPT-2 combined QKV: "attn.c_attn"
    if name.contains("c_attn") && !name.contains("c_attn_proj") && !name.contains("c_attn.") {
        if name.contains("bias") {
            layer.attn_qkv_bias = Some(tensor);
        } else {
            layer.attn_qkv_weight = Some(tensor);
        }
        return;
    }

    // Attention Q/K/V projections
    if name.contains("q_proj") || name.contains("q_attn") {
        if name.contains("bias") {
            layer.attn_q_bias = Some(tensor);
        } else {
            layer.attn_q_weight = Some(tensor);
        }
        return;
    }
    if name.contains("k_proj") || name.contains("k_attn") {
        if name.contains("bias") {
            layer.attn_k_bias = Some(tensor);
        } else {
            layer.attn_k_weight = Some(tensor);
        }
        return;
    }
    if name.contains("v_proj") || name.contains("v_attn") {
        if name.contains("bias") {
            layer.attn_v_bias = Some(tensor);
        } else {
            layer.attn_v_weight = Some(tensor);
        }
        return;
    }

    // Attention output projection
    if name.contains("c_proj") || name.contains("o_proj") || name.contains("out_proj") {
        if name.contains("bias") {
            layer.attn_out_bias = Some(tensor);
        } else {
            layer.attn_out_weight = Some(tensor);
        }
        return;
    }

    // Layer norms
    if name.contains("ln_1") || name.contains("input_layernorm") {
        if name.contains("bias") {
            layer.ln1_bias = Some(tensor);
        } else {
            layer.ln1_weight = Some(tensor);
        }
        return;
    }
    if name.contains("ln_2") || name.contains("post_attention_layernorm") {
        if name.contains("bias") {
            layer.ln2_bias = Some(tensor);
        } else {
            layer.ln2_weight = Some(tensor);
        }
        return;
    }

    // FFN
    if name.contains("c_fc")
        || name.contains("up_proj")
        || name.contains("gate_proj")
        || name.contains("mlp.fc1")
    {
        if name.contains("bias") {
            layer.ffn_up_bias = Some(tensor);
        } else {
            layer.ffn_up_weight = Some(tensor);
        }
        return;
    }
    if name.contains("c_proj") && name.contains("mlp") {
        if name.contains("bias") {
            layer.ffn_down_bias = Some(tensor);
        } else {
            layer.ffn_down_weight = Some(tensor);
        }
        return;
    }
    if name.contains("down_proj") || name.contains("mlp.fc2") {
        if name.contains("bias") {
            layer.ffn_down_bias = Some(tensor);
        } else {
            layer.ffn_down_weight = Some(tensor);
        }
    }
}

/// Print a summary of loaded weights.
pub fn print_weight_summary(weights: &ModelWeights, config: &TransformerConfig) {
    println!("Model weight summary ({config}):");
    println!(
        "  Token embedding: {}",
        if weights.token_embedding.is_some() {
            "loaded"
        } else {
            "missing"
        }
    );
    println!(
        "  Position embedding: {}",
        if weights.position_embedding.is_some() {
            "loaded"
        } else {
            "missing/rotary"
        }
    );
    println!(
        "  Final layer norm: {}",
        if weights.ln_final_weight.is_some() {
            "loaded"
        } else {
            "missing"
        }
    );
    println!(
        "  LM head: {}",
        if weights.lm_head_weight.is_some() {
            "loaded"
        } else {
            "tied"
        }
    );

    let mut loaded_layers = 0;
    for (i, layer) in weights.layers.iter().enumerate() {
        let has_attn = layer.attn_q_weight.is_some() || layer.attn_qkv_weight.is_some();
        let has_ffn = layer.ffn_up_weight.is_some();
        let has_ln = layer.ln1_weight.is_some();
        if has_attn && has_ffn && has_ln {
            loaded_layers += 1;
        } else {
            println!("  Layer {i}: INCOMPLETE (attn={has_attn}, ffn={has_ffn}, ln={has_ln})");
        }
    }
    println!("  Complete layers: {loaded_layers}/{}", config.num_layers);

    if !weights.unmatched.is_empty() {
        println!("  Unmatched tensors ({}):", weights.unmatched.len());
        for w in &weights.unmatched {
            println!("    {} {:?}", w.name, w.original_shape);
        }
    }
}

/// Index of all tensor names and shapes (for inspection without loading to GPU).
pub fn inspect_safetensors(
    paths: &[impl AsRef<Path>],
) -> Result<Vec<(String, Vec<usize>, String)>> {
    let mut entries = Vec::new();

    for path in paths {
        let path = path.as_ref();
        let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let tensors = safetensors::SafeTensors::deserialize(&data)
            .with_context(|| format!("parsing {}", path.display()))?;

        for (name, view) in tensors.tensors() {
            entries.push((
                name.clone(),
                view.shape().to_vec(),
                format!("{:?}", view.dtype()),
            ));
        }
    }

    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── f16 conversion ──────────────────────────────────────────

    #[test]
    fn f16_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn f16_negative_zero() {
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
    }

    #[test]
    fn f16_one() {
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn f16_negative_one() {
        assert!((f16_to_f32(0xBC00) + 1.0).abs() < 1e-7);
    }

    #[test]
    fn f16_half() {
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn f16_infinity() {
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C00).is_sign_positive());
    }

    #[test]
    fn f16_negative_infinity() {
        assert!(f16_to_f32(0xFC00).is_infinite());
        assert!(f16_to_f32(0xFC00).is_sign_negative());
    }

    #[test]
    fn f16_nan() {
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn f16_subnormal() {
        let val = f16_to_f32(0x0001);
        assert!(val > 0.0);
        assert!(val < 1e-6);
    }

    // ── bf16 conversion ─────────────────────────────────────────

    #[test]
    fn bf16_zero() {
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn bf16_one() {
        assert!((bf16_to_f32(0x3F80) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn bf16_negative_one() {
        assert!((bf16_to_f32(0xBF80) + 1.0).abs() < 1e-7);
    }

    #[test]
    fn bf16_infinity() {
        assert!(bf16_to_f32(0x7F80).is_infinite());
    }

    #[test]
    fn bf16_nan() {
        assert!(bf16_to_f32(0x7FC0).is_nan());
    }

    // ── layer index extraction ──────────────────────────────────

    #[test]
    fn extract_gpt2_layer() {
        assert_eq!(extract_layer_index("h.0.attn.c_attn.weight"), Some(0));
        assert_eq!(extract_layer_index("h.11.mlp.c_fc.bias"), Some(11));
    }

    #[test]
    fn extract_llama_layer() {
        assert_eq!(
            extract_layer_index("model.layers.0.self_attn.q_proj.weight"),
            Some(0)
        );
        assert_eq!(
            extract_layer_index("model.layers.31.mlp.down_proj.weight"),
            Some(31)
        );
    }

    #[test]
    fn extract_no_layer() {
        assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
        assert_eq!(extract_layer_index("lm_head.weight"), None);
    }

    // ── weight organization ─────────────────────────────────────

    #[test]
    fn organize_handles_empty_weights() {
        let config = crate::model_config::TransformerConfig {
            num_layers: 2,
            ..crate::model_config::TransformerConfig::default_gpt2()
        };
        let model = organize_weights(vec![], &config);
        assert!(model.token_embedding.is_none());
        assert!(model.position_embedding.is_none());
        assert_eq!(model.layers.len(), 2);
        assert!(model.unmatched.is_empty());
    }
}
