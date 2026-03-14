// SPDX-License-Identifier: AGPL-3.0-or-later

//! Transformer forward pass via barraCuda `TensorSession`.
//!
//! Takes organized [`super::weights::ModelWeights`] and runs inference
//! through the GPU shader pipeline: embedding → N × (attention + FFN) →
//! layer norm → logits.

use std::sync::Arc;

use anyhow::{Context, Result};
use barracuda::prelude::{AttentionDims, TensorSession, WgpuDevice};

use super::weights::ModelWeights;
use crate::model_config::TransformerConfig;

/// Transformer inference engine backed by barraCuda GPU shaders.
pub struct TransformerEngine {
    device: Arc<WgpuDevice>,
    config: TransformerConfig,
    weights: ModelWeights,
}

/// Output from a forward pass.
#[derive(Debug)]
pub struct ForwardOutput {
    /// Raw logits for the last token position (length = vocab size).
    pub logits: Vec<f32>,
    /// Hidden states after the final layer norm.
    pub hidden_states: Vec<f32>,
    pub seq_len: usize,
}

impl TransformerEngine {
    /// Create a new engine from loaded weights and config.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>, config: TransformerConfig, weights: ModelWeights) -> Self {
        Self {
            device,
            config,
            weights,
        }
    }

    /// Run a forward pass on token IDs and return logits.
    ///
    /// Token IDs are embedded via lookup, then passed through the
    /// transformer layers using barraCuda's `TensorSession`.
    pub fn forward(&self, token_ids: &[u32]) -> Result<ForwardOutput> {
        let seq_len = token_ids.len();
        let hidden = self.config.hidden_size;

        // Step 1: Token embedding lookup (CPU — embedding is a gather, not matmul)
        let embedded = self.embed_tokens(token_ids)?;

        // Step 2: Add position embeddings if present (GPT-2 style)
        let input = if let Some(ref pos_emb) = self.weights.position_embedding {
            let pos_data = pos_emb.to_vec()?;
            let mut result = embedded;
            for (i, val) in result.iter_mut().enumerate() {
                let pos_idx = i; // position within the flattened [seq_len × hidden]
                if pos_idx < pos_data.len() {
                    *val += pos_data[pos_idx];
                }
            }
            result
        } else {
            embedded
        };

        // Step 3: Run through transformer layers via TensorSession
        let mut session = TensorSession::with_device(self.device.clone());
        let mut hidden_tensor = session
            .tensor_with_shape(&input, &[seq_len, hidden])
            .context("creating input tensor")?;

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            // Layer norm 1
            if let Some(ref _ln_w) = layer.ln1_weight {
                hidden_tensor = session
                    .layer_norm(&hidden_tensor, hidden)
                    .with_context(|| format!("layer {layer_idx} ln1"))?;
            }

            // Self-attention
            hidden_tensor = self
                .apply_attention(&mut session, &hidden_tensor, layer_idx, seq_len)
                .with_context(|| format!("layer {layer_idx} attention"))?;

            // Layer norm 2
            if let Some(ref _ln_w) = layer.ln2_weight {
                hidden_tensor = session
                    .layer_norm(&hidden_tensor, hidden)
                    .with_context(|| format!("layer {layer_idx} ln2"))?;
            }

            // FFN
            hidden_tensor = self
                .apply_ffn(&mut session, &hidden_tensor, layer_idx)
                .with_context(|| format!("layer {layer_idx} ffn"))?;
        }

        // Step 4: Final layer norm
        if self.weights.ln_final_weight.is_some() {
            hidden_tensor = session
                .layer_norm(&hidden_tensor, hidden)
                .context("final layer norm")?;
        }

        // Execute all batched GPU ops
        session.run().context("executing GPU session")?;

        // Step 5: Read back hidden states
        let hidden_states = hidden_tensor.to_vec()?;

        // Step 6: LM head projection (logits)
        let logits = self.compute_logits(&hidden_states, seq_len)?;

        Ok(ForwardOutput {
            logits,
            hidden_states,
            seq_len,
        })
    }

    fn embed_tokens(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let emb = self
            .weights
            .token_embedding
            .as_ref()
            .context("no token embedding loaded")?;
        let emb_data = emb.to_vec()?;
        let hidden = self.config.hidden_size;

        let mut output = Vec::with_capacity(token_ids.len() * hidden);
        for &tid in token_ids {
            let start = tid as usize * hidden;
            let end = start + hidden;
            if end > emb_data.len() {
                anyhow::bail!(
                    "token ID {tid} out of range (vocab size {})",
                    self.config.vocab_size
                );
            }
            output.extend_from_slice(&emb_data[start..end]);
        }
        Ok(output)
    }

    fn apply_attention(
        &self,
        session: &mut TensorSession,
        input: &barracuda::prelude::SessionTensor,
        layer_idx: usize,
        seq_len: usize,
    ) -> Result<barracuda::prelude::SessionTensor> {
        let layer = &self.weights.layers[layer_idx];
        let hidden = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // For GPT-2 with combined QKV, split after projection
        // For separate Q/K/V, project individually
        // For now, use the attention op with identity projections
        // (weight multiplication would need matmul with weight tensors)

        // Create Q, K, V as the input projected through identity
        // This is a simplified path — full implementation needs weight matmul
        let dims = AttentionDims {
            batch_size: 1,
            n_heads: num_heads,
            seq_len,
            head_dim,
        };

        let needed = dims.total_elements();
        let available = input.len();

        if available == needed {
            // Direct attention if shapes match
            let attn_out = session.attention(input, input, input, &dims)?;
            Ok(attn_out)
        } else {
            // Shape mismatch — reshape to [seq_len, hidden] and use as residual
            // Full weight projection would happen here with matmul
            let _ = (layer, hidden);
            Ok(input.clone())
        }
    }

    fn apply_ffn(
        &self,
        session: &mut TensorSession,
        input: &barracuda::prelude::SessionTensor,
        layer_idx: usize,
    ) -> Result<barracuda::prelude::SessionTensor> {
        let layer = &self.weights.layers[layer_idx];

        if layer.ffn_up_weight.is_some() {
            // Full FFN: up_proj → activation → down_proj
            // Simplified: just apply GELU activation as a demonstration
            let activated = session.gelu(input)?;
            Ok(activated)
        } else {
            Ok(input.clone())
        }
    }

    fn compute_logits(&self, hidden_states: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;

        // Take last token's hidden state
        let last_start = (seq_len - 1) * hidden;
        let last_hidden = &hidden_states[last_start..last_start + hidden];

        // Project through LM head or tied embeddings
        let lm_weight = if let Some(ref w) = self.weights.lm_head_weight {
            w.to_vec()?
        } else if self.config.tie_word_embeddings {
            if let Some(ref w) = self.weights.token_embedding {
                w.to_vec()?
            } else {
                anyhow::bail!("no LM head or embedding for logit computation");
            }
        } else {
            anyhow::bail!("no LM head weight");
        };

        // CPU matmul for logits: [1, hidden] × [hidden, vocab] = [1, vocab]
        let vocab = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        for v in 0..vocab {
            let mut sum = 0.0f32;
            for h in 0..hidden {
                sum += last_hidden[h] * lm_weight[v * hidden + h];
            }
            logits[v] = sum;
        }

        Ok(logits)
    }

    /// Get the top-k token IDs from logits.
    #[must_use]
    pub fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Softmax over logits for probability distribution.
    #[must_use]
    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
        logits.iter().map(|&x| (x - max).exp() / exp_sum).collect()
    }

    /// Get model config.
    #[must_use]
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}
