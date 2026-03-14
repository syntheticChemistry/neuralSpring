// SPDX-License-Identifier: AGPL-3.0-or-later

//! Parse `HuggingFace` `config.json` into typed transformer configurations.
//!
//! Supports GPT-2, Llama, Mistral, Phi, and other common architectures
//! by extracting the key dimensions needed for inference.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Unified transformer configuration extracted from HF `config.json`.
///
/// Different architectures use different field names; this struct
/// normalizes them into a common representation.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub head_dim: usize,
    pub layer_norm_eps: f64,
    pub activation: Activation,
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    #[default]
    Gelu,
    GeluNew,
    Relu,
    Silu,
    Swish,
    Mish,
}

impl Activation {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "gelu_new" | "gelu_fast" | "gelu_pytorch_tanh" => Self::GeluNew,
            "relu" => Self::Relu,
            "silu" | "swiglu" => Self::Silu,
            "swish" => Self::Swish,
            "mish" => Self::Mish,
            _ => Self::Gelu,
        }
    }
}

/// Raw HF config.json with all optional fields.
#[derive(Debug, Deserialize)]
struct RawConfig {
    model_type: Option<String>,
    // GPT-2 / generic
    vocab_size: Option<usize>,
    n_embd: Option<usize>,
    hidden_size: Option<usize>,
    n_layer: Option<usize>,
    num_hidden_layers: Option<usize>,
    n_head: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    n_inner: Option<usize>,
    intermediate_size: Option<usize>,
    n_positions: Option<usize>,
    max_position_embeddings: Option<usize>,
    layer_norm_epsilon: Option<f64>,
    rms_norm_eps: Option<f64>,
    activation_function: Option<String>,
    hidden_act: Option<String>,
    tie_word_embeddings: Option<bool>,
    head_dim: Option<usize>,
}

impl TransformerConfig {
    /// Load and parse a HF `config.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_json(&content)
    }

    /// Parse from JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: RawConfig = serde_json::from_str(json).context("parsing config.json")?;

        let model_type = raw.model_type.unwrap_or_else(|| "unknown".to_string());
        let hidden_size = raw.hidden_size.or(raw.n_embd).unwrap_or(768);
        let num_layers = raw.num_hidden_layers.or(raw.n_layer).unwrap_or(12);
        let num_heads = raw.num_attention_heads.or(raw.n_head).unwrap_or(12);
        let num_kv_heads = raw.num_key_value_heads.unwrap_or(num_heads);
        let intermediate_size = raw
            .intermediate_size
            .or(raw.n_inner)
            .unwrap_or(hidden_size * 4);
        let max_position_embeddings = raw
            .max_position_embeddings
            .or(raw.n_positions)
            .unwrap_or(2048);
        let head_dim = raw.head_dim.unwrap_or(hidden_size / num_heads);
        let layer_norm_eps = raw.layer_norm_epsilon.or(raw.rms_norm_eps).unwrap_or(1e-5);
        let activation_str = raw
            .hidden_act
            .or(raw.activation_function)
            .unwrap_or_else(|| "gelu".to_string());

        Ok(Self {
            model_type,
            vocab_size: raw.vocab_size.unwrap_or(50257),
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            intermediate_size,
            max_position_embeddings,
            head_dim,
            layer_norm_eps,
            activation: Activation::from_str(&activation_str),
            tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(true),
        })
    }

    /// GPT-2 small defaults for testing.
    #[must_use]
    pub fn default_gpt2() -> Self {
        Self {
            model_type: "gpt2".into(),
            vocab_size: 50257,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 1024,
            head_dim: 64,
            layer_norm_eps: 1e-5,
            activation: Activation::GeluNew,
            tie_word_embeddings: true,
        }
    }

    /// Total parameters estimate (rough).
    #[must_use]
    pub fn estimated_params(&self) -> usize {
        let embed = self.vocab_size * self.hidden_size;
        let per_layer = 4 * self.hidden_size * self.hidden_size // attention QKV + out
            + 2 * self.hidden_size * self.intermediate_size // FFN up + down
            + 4 * self.hidden_size; // layer norms
        embed + self.num_layers * per_layer
    }

    /// Memory estimate in bytes at f32 precision.
    #[must_use]
    pub fn estimated_memory_f32(&self) -> usize {
        self.estimated_params() * 4
    }
}

#[expect(clippy::cast_precision_loss, reason = "display-only param count")]
impl std::fmt::Display for TransformerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}): {}L / {}H / {}d / {}ffn / ~{:.1}M params",
            self.model_type,
            if self.num_kv_heads == self.num_heads {
                "MHA"
            } else {
                "GQA"
            },
            self.num_layers,
            self.num_heads,
            self.hidden_size,
            self.intermediate_size,
            self.estimated_params() as f64 / 1e6
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gpt2_config() {
        let json = r#"{
            "model_type": "gpt2",
            "vocab_size": 50257,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_inner": 3072,
            "n_positions": 1024,
            "activation_function": "gelu_new",
            "layer_norm_epsilon": 1e-5
        }"#;
        let config = TransformerConfig::from_json(json).unwrap();
        assert_eq!(config.model_type, "gpt2");
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.activation, Activation::GeluNew);
    }

    #[test]
    fn parse_llama_config() {
        let json = r#"{
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu"
        }"#;
        let config = TransformerConfig::from_json(json).unwrap();
        assert_eq!(config.model_type, "llama");
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.activation, Activation::Silu);
    }

    #[test]
    fn default_gpt2_has_correct_dims() {
        let config = TransformerConfig::default_gpt2();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.head_dim, 64);
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn estimated_params_gpt2() {
        let config = TransformerConfig::default_gpt2();
        let params = config.estimated_params();
        assert!(
            params > 80_000_000,
            "GPT-2 should have >80M params, got {params}"
        );
        assert!(
            params < 200_000_000,
            "GPT-2 should have <200M params, got {params}"
        );
    }

    #[test]
    fn estimated_memory_f32() {
        let config = TransformerConfig::default_gpt2();
        let mem = config.estimated_memory_f32();
        assert_eq!(mem, config.estimated_params() * 4);
    }

    #[test]
    fn minimal_json_uses_defaults() {
        let config = TransformerConfig::from_json("{}").unwrap();
        assert_eq!(config.model_type, "unknown");
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.activation, Activation::Gelu);
    }

    #[test]
    fn activation_parsing() {
        assert_eq!(Activation::from_str("gelu"), Activation::Gelu);
        assert_eq!(Activation::from_str("gelu_new"), Activation::GeluNew);
        assert_eq!(Activation::from_str("gelu_fast"), Activation::GeluNew);
        assert_eq!(Activation::from_str("relu"), Activation::Relu);
        assert_eq!(Activation::from_str("silu"), Activation::Silu);
        assert_eq!(Activation::from_str("swiglu"), Activation::Silu);
        assert_eq!(Activation::from_str("swish"), Activation::Swish);
        assert_eq!(Activation::from_str("mish"), Activation::Mish);
        assert_eq!(Activation::from_str("unknown_activation"), Activation::Gelu);
    }

    #[test]
    fn display_format() {
        let config = TransformerConfig::default_gpt2();
        let s = config.to_string();
        assert!(s.contains("gpt2"));
        assert!(s.contains("MHA"));
        assert!(s.contains("12L"));
        assert!(s.contains("12H"));
        assert!(s.contains("768d"));
    }

    #[test]
    fn gqa_display() {
        let mut config = TransformerConfig::default_gpt2();
        config.num_kv_heads = 4;
        let s = config.to_string();
        assert!(s.contains("GQA"));
    }

    #[test]
    fn parse_phi_style_config() {
        let json = r#"{
            "model_type": "phi",
            "vocab_size": 51200,
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "intermediate_size": 8192,
            "max_position_embeddings": 2048,
            "hidden_act": "gelu_new",
            "head_dim": 64,
            "tie_word_embeddings": false
        }"#;
        let config = TransformerConfig::from_json(json).unwrap();
        assert_eq!(config.model_type, "phi");
        assert_eq!(config.head_dim, 64);
        assert!(!config.tie_word_embeddings);
    }
}
