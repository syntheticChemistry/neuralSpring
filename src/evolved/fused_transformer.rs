// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fused Transformer encoder block — single encoder, single submit.
//!
//! All operations (`layer_norm`, matmul projections, `head_split`, attention,
//! `head_concat`, FFN with GELU, residual adds) are recorded as compute
//! passes in one `CommandEncoder`.  Head-split and head-concat run as
//! GPU shaders (no CPU round-trips).
//!
//! Architecture:
//! ```text
//! x -> LayerNorm -> Q,K,V projections -> head_split
//!   -> batched attention -> head_concat -> Wo projection -> + residual
//!   -> LayerNorm -> FFN1 (GELU) -> FFN2 -> + residual -> output
//! ```
//!
//! **`ToadStool` handoff**: Once `TensorSession` supports the full op
//! set and MHA dispatch is fixed upstream, this can be retired.

#![allow(clippy::cast_possible_truncation, clippy::too_many_lines)]

use super::fused_pipeline::{
    self as fp, Dev, GeluParams, HeadShapeParams, LayerNormParams, MatMulParams, MatmulConfig,
    ShaderCache,
};

/// Pre-built fused Transformer encoder block.
///
/// Fields that appear unused keep GPU buffers alive for bind groups.
#[allow(dead_code)]
pub struct FusedTransformer {
    shaders: ShaderCache,
    device: Dev,
    cfg: TransformerDims,

    // Weight buffers (immutable after construction)
    w_q: wgpu::Buffer,
    w_k: wgpu::Buffer,
    w_v: wgpu::Buffer,
    w_o: wgpu::Buffer,
    w_ff1: wgpu::Buffer,
    b_ff1: wgpu::Buffer,
    w_ff2: wgpu::Buffer,
    b_ff2: wgpu::Buffer,

    // Intermediate buffers
    input_buf: wgpu::Buffer,
    ln1_out: wgpu::Buffer,
    q_proj: wgpu::Buffer,
    k_proj: wgpu::Buffer,
    v_proj: wgpu::Buffer,
    q_split: wgpu::Buffer,
    k_split: wgpu::Buffer,
    v_split: wgpu::Buffer,
    attn_out: wgpu::Buffer,
    attn_concat: wgpu::Buffer,
    wo_out: wgpu::Buffer,
    residual1: wgpu::Buffer,
    ln2_out: wgpu::Buffer,
    ff1_out: wgpu::Buffer,
    ff1_bias_out: wgpu::Buffer,
    ff1_gelu: wgpu::Buffer,
    ff2_out: wgpu::Buffer,
    ff2_bias_out: wgpu::Buffer,
    output_buf: wgpu::Buffer,

    // Param buffers
    ln1_params: wgpu::Buffer,
    ln2_params: wgpu::Buffer,
    mm_q_params: wgpu::Buffer,
    mm_k_params: wgpu::Buffer,
    mm_v_params: wgpu::Buffer,
    mm_wo_params: wgpu::Buffer,
    mm_ff1_params: wgpu::Buffer,
    mm_ff2_params: wgpu::Buffer,
    head_split_params: wgpu::Buffer,
    head_concat_params: wgpu::Buffer,
    attn_params: wgpu::Buffer,
    gelu_params: wgpu::Buffer,

    // Pre-created bind groups
    bg_ln1: wgpu::BindGroup,
    bg_q: wgpu::BindGroup,
    bg_k: wgpu::BindGroup,
    bg_v: wgpu::BindGroup,
    bg_hs_q: wgpu::BindGroup,
    bg_hs_k: wgpu::BindGroup,
    bg_hs_v: wgpu::BindGroup,
    bg_attn: wgpu::BindGroup,
    bg_hc: wgpu::BindGroup,
    bg_wo: wgpu::BindGroup,
    bg_res1: wgpu::BindGroup,
    bg_ln2: wgpu::BindGroup,
    bg_ff1: wgpu::BindGroup,
    bg_ff1_add: wgpu::BindGroup,
    bg_gelu: wgpu::BindGroup,
    bg_ff2: wgpu::BindGroup,
    bg_ff2_add: wgpu::BindGroup,
    bg_res2: wgpu::BindGroup,
    mm_config: MatmulConfig,
}

#[derive(Clone, Copy)]
pub struct TransformerDims {
    pub seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub epsilon: f32,
}

impl TransformerDims {
    const fn d_head(self) -> usize {
        self.d_model / self.n_heads
    }
    const fn sd(self) -> usize {
        self.seq_len * self.d_model
    }
}

/// Weights for one transformer encoder block.
pub struct TransformerWeightsRef<'a> {
    pub w_q: &'a [f32],
    pub w_k: &'a [f32],
    pub w_v: &'a [f32],
    pub w_o: &'a [f32],
    pub w_ff1: &'a [f32],
    pub b_ff1: &'a [f32],
    pub w_ff2: &'a [f32],
    pub b_ff2: &'a [f32],
}

/// Params for the batched attention shader (matches WGSL `Params` struct).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BAttnParams {
    n_heads: u32,
    seq_len: u32,
    d_head: u32,
    scale: f32,
}

impl FusedTransformer {
    /// Build a fused transformer block. Compiles shaders, uploads weights,
    /// allocates all intermediates and bind groups once.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn new(device: Dev, weights: &TransformerWeightsRef<'_>, cfg: TransformerDims) -> Self {
        let shaders = ShaderCache::new(&device);
        let d = cfg.d_model;
        let seq = cfg.seq_len;
        let sd = cfg.sd();
        let d_head = cfg.d_head();
        let n_heads = cfg.n_heads;
        let d_ff = cfg.d_ff;

        // Upload weights
        let w_q = fp::buf_init(&device, weights.w_q, "tf_w_q");
        let w_k = fp::buf_init(&device, weights.w_k, "tf_w_k");
        let w_v = fp::buf_init(&device, weights.w_v, "tf_w_v");
        let w_o = fp::buf_init(&device, weights.w_o, "tf_w_o");
        let w_ff1 = fp::buf_init(&device, weights.w_ff1, "tf_w_ff1");
        let w_ff2 = fp::buf_init(&device, weights.w_ff2, "tf_w_ff2");

        // Broadcast biases to [seq, d_ff] and [seq, d] respectively
        let b_ff1_bcast: Vec<f32> = (0..seq)
            .flat_map(|_| weights.b_ff1.iter().copied())
            .collect();
        let b_ff2_bcast: Vec<f32> = (0..seq)
            .flat_map(|_| weights.b_ff2.iter().copied())
            .collect();
        let b_ff1 = fp::buf_init(&device, &b_ff1_bcast, "tf_b_ff1");
        let b_ff2 = fp::buf_init(&device, &b_ff2_bcast, "tf_b_ff2");

        // Intermediate buffers
        let input_buf = fp::buf_empty(&device, sd, "tf_input");
        let ln1_out = fp::buf_empty(&device, sd, "tf_ln1");
        let q_proj = fp::buf_empty(&device, sd, "tf_q");
        let k_proj = fp::buf_empty(&device, sd, "tf_k");
        let v_proj = fp::buf_empty(&device, sd, "tf_v");
        let q_split = fp::buf_empty(&device, sd, "tf_q_split");
        let k_split = fp::buf_empty(&device, sd, "tf_k_split");
        let v_split = fp::buf_empty(&device, sd, "tf_v_split");
        let attn_out = fp::buf_empty(&device, sd, "tf_attn");
        let attn_concat = fp::buf_empty(&device, sd, "tf_attn_cat");
        let wo_out = fp::buf_empty(&device, sd, "tf_wo");
        let residual1 = fp::buf_empty(&device, sd, "tf_res1");
        let ln2_out = fp::buf_empty(&device, sd, "tf_ln2");
        let ff1_out = fp::buf_empty(&device, seq * d_ff, "tf_ff1");
        let ff1_bias_out = fp::buf_empty(&device, seq * d_ff, "tf_ff1_b");
        let ff1_gelu = fp::buf_empty(&device, seq * d_ff, "tf_gelu");
        let ff2_out = fp::buf_empty(&device, sd, "tf_ff2");
        let ff2_bias_out = fp::buf_empty(&device, sd, "tf_ff2_b");
        let output_buf = fp::buf_empty(&device, sd, "tf_output");

        // Param buffers
        let ln1_params = fp::buf_uniform(
            &device,
            &LayerNormParams {
                size: sd as u32,
                feature_size: d as u32,
                epsilon: cfg.epsilon,
            },
            "tf_ln1_p",
        );

        let ln2_params = fp::buf_uniform(
            &device,
            &LayerNormParams {
                size: sd as u32,
                feature_size: d as u32,
                epsilon: cfg.epsilon,
            },
            "tf_ln2_p",
        );

        let mm_proj = |label: &str| {
            fp::buf_uniform(
                &device,
                &MatMulParams {
                    m: seq as u32,
                    k: d as u32,
                    n: d as u32,
                    _padding: 0,
                },
                label,
            )
        };
        #[allow(clippy::similar_names)]
        let mm_q_params = mm_proj("tf_mm_q_p");
        #[allow(clippy::similar_names)]
        let mm_k_params = mm_proj("tf_mm_k_p");
        #[allow(clippy::similar_names)]
        let mm_v_params = mm_proj("tf_mm_v_p");
        let mm_wo_params = mm_proj("tf_mm_wo_p");

        let mm_ff1_params = fp::buf_uniform(
            &device,
            &MatMulParams {
                m: seq as u32,
                k: d as u32,
                n: d_ff as u32,
                _padding: 0,
            },
            "tf_mm_ff1_p",
        );
        let mm_ff2_params = fp::buf_uniform(
            &device,
            &MatMulParams {
                m: seq as u32,
                k: d_ff as u32,
                n: d as u32,
                _padding: 0,
            },
            "tf_mm_ff2_p",
        );

        let head_split_params = fp::buf_uniform(
            &device,
            &HeadShapeParams {
                seq_len: seq as u32,
                d_model: d as u32,
                n_heads: n_heads as u32,
                d_head: d_head as u32,
            },
            "tf_hs_p",
        );
        let head_concat_params = fp::buf_uniform(
            &device,
            &HeadShapeParams {
                seq_len: seq as u32,
                d_model: d as u32,
                n_heads: n_heads as u32,
                d_head: d_head as u32,
            },
            "tf_hc_p",
        );

        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (d_head as f32).sqrt();
        let attn_params = fp::buf_uniform(
            &device,
            &BAttnParams {
                n_heads: n_heads as u32,
                seq_len: seq as u32,
                d_head: d_head as u32,
                scale,
            },
            "tf_battn_p",
        );

        let gelu_params = fp::buf_uniform(
            &device,
            &GeluParams {
                size: (seq * d_ff) as u32,
            },
            "tf_gelu_p",
        );

        // Bind groups
        let bg_ln1 = fp::bind_group(
            &device,
            &shaders.layer_norm,
            &[&input_buf, &ln1_out, &ln1_params],
            "tf_bg_ln1",
        );

        let mm_config = MatmulConfig::from_device(&device);
        let (proj_pipe, _) =
            fp::select_matmul(&shaders, seq as u32, d as u32, d as u32, &mm_config);

        let bg_q = fp::bind_group(
            &device,
            proj_pipe,
            &[&ln1_out, &w_q, &q_proj, &mm_q_params],
            "tf_bg_q",
        );
        let bg_k = fp::bind_group(
            &device,
            proj_pipe,
            &[&ln1_out, &w_k, &k_proj, &mm_k_params],
            "tf_bg_k",
        );
        let bg_v = fp::bind_group(
            &device,
            proj_pipe,
            &[&ln1_out, &w_v, &v_proj, &mm_v_params],
            "tf_bg_v",
        );

        let bg_hs_q = fp::bind_group(
            &device,
            &shaders.head_split,
            &[&q_proj, &q_split, &head_split_params],
            "tf_bg_hs_q",
        );
        let bg_hs_k = fp::bind_group(
            &device,
            &shaders.head_split,
            &[&k_proj, &k_split, &head_split_params],
            "tf_bg_hs_k",
        );
        let bg_hs_v = fp::bind_group(
            &device,
            &shaders.head_split,
            &[&v_proj, &v_split, &head_split_params],
            "tf_bg_hs_v",
        );

        let bg_attn = fp::bind_group(
            &device,
            &shaders.attention,
            &[&q_split, &k_split, &v_split, &attn_out, &attn_params],
            "tf_bg_attn",
        );

        let bg_hc = fp::bind_group(
            &device,
            &shaders.head_concat,
            &[&attn_out, &attn_concat, &head_concat_params],
            "tf_bg_hc",
        );

        let bg_wo = fp::bind_group(
            &device,
            proj_pipe,
            &[&attn_concat, &w_o, &wo_out, &mm_wo_params],
            "tf_bg_wo",
        );

        // residual1 = input + wo_out
        let bg_res1 = fp::bind_group(
            &device,
            &shaders.add,
            &[&input_buf, &wo_out, &residual1],
            "tf_bg_res1",
        );

        let bg_ln2 = fp::bind_group(
            &device,
            &shaders.layer_norm,
            &[&residual1, &ln2_out, &ln2_params],
            "tf_bg_ln2",
        );

        let (ff1_pipe, _) =
            fp::select_matmul(&shaders, seq as u32, d as u32, d_ff as u32, &mm_config);
        let bg_ff1 = fp::bind_group(
            &device,
            ff1_pipe,
            &[&ln2_out, &w_ff1, &ff1_out, &mm_ff1_params],
            "tf_bg_ff1",
        );

        let bg_ff1_add = fp::bind_group(
            &device,
            &shaders.add,
            &[&ff1_out, &b_ff1, &ff1_bias_out],
            "tf_bg_ff1_add",
        );

        let bg_gelu = fp::bind_group(
            &device,
            &shaders.gelu,
            &[&ff1_bias_out, &ff1_gelu, &gelu_params],
            "tf_bg_gelu",
        );

        let (ff2_pipe, _) =
            fp::select_matmul(&shaders, seq as u32, d_ff as u32, d as u32, &mm_config);
        let bg_ff2 = fp::bind_group(
            &device,
            ff2_pipe,
            &[&ff1_gelu, &w_ff2, &ff2_out, &mm_ff2_params],
            "tf_bg_ff2",
        );

        let bg_ff2_add = fp::bind_group(
            &device,
            &shaders.add,
            &[&ff2_out, &b_ff2, &ff2_bias_out],
            "tf_bg_ff2_add",
        );

        // output = residual1 + ff2_bias_out
        let bg_res2 = fp::bind_group(
            &device,
            &shaders.add,
            &[&residual1, &ff2_bias_out, &output_buf],
            "tf_bg_res2",
        );

        Self {
            shaders,
            device,
            cfg,
            w_q,
            w_k,
            w_v,
            w_o,
            w_ff1,
            b_ff1,
            w_ff2,
            b_ff2,
            input_buf,
            ln1_out,
            q_proj,
            k_proj,
            v_proj,
            q_split,
            k_split,
            v_split,
            attn_out,
            attn_concat,
            wo_out,
            residual1,
            ln2_out,
            ff1_out,
            ff1_bias_out,
            ff1_gelu,
            ff2_out,
            ff2_bias_out,
            output_buf,
            ln1_params,
            ln2_params,
            mm_q_params,
            mm_k_params,
            mm_v_params,
            mm_wo_params,
            mm_ff1_params,
            mm_ff2_params,
            head_split_params,
            head_concat_params,
            attn_params,
            gelu_params,
            bg_ln1,
            bg_q,
            bg_k,
            bg_v,
            bg_hs_q,
            bg_hs_k,
            bg_hs_v,
            bg_attn,
            bg_hc,
            bg_wo,
            bg_res1,
            bg_ln2,
            bg_ff1,
            bg_ff1_add,
            bg_gelu,
            bg_ff2,
            bg_ff2_add,
            bg_res2,
            mm_config,
        }
    }

    /// Upload input and run a fused forward pass.
    ///
    /// Returns output `[seq_len, d_model]` as CPU `Vec<f32>`.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.device
            .queue()
            .write_buffer(&self.input_buf, 0, bytemuck::cast_slice(input));

        let encoder = self.encode();
        fp::submit_and_wait(&self.device, encoder);

        fp::readback(&self.device, &self.output_buf, self.cfg.sd())
    }

    /// Run forward pass without readback (for throughput benchmarking).
    pub fn forward_no_readback(&self, input: &[f32]) {
        self.device
            .queue()
            .write_buffer(&self.input_buf, 0, bytemuck::cast_slice(input));

        let encoder = self.encode();
        self.device.queue().submit(Some(encoder.finish()));
        self.device.device().poll(wgpu::Maintain::Wait);
    }

    /// Encode all compute passes into a single `CommandEncoder`.
    fn encode(&self) -> wgpu::CommandEncoder {
        let seq = self.cfg.seq_len as u32;
        let d = self.cfg.d_model as u32;
        let d_ff = self.cfg.d_ff as u32;
        let sd = (self.cfg.sd()) as u32;
        let n_heads = self.cfg.n_heads as u32;
        let d_head = self.cfg.d_head() as u32;

        let mut enc = fp::new_encoder(&self.device, "fused_transformer");

        let (proj_pipe, proj_disp) = fp::select_matmul(&self.shaders, seq, d, d, &self.mm_config);
        let (ff1_pipe, ff1_disp) = fp::select_matmul(&self.shaders, seq, d, d_ff, &self.mm_config);
        let (ff2_pipe, ff2_disp) = fp::select_matmul(&self.shaders, seq, d_ff, d, &self.mm_config);

        // 1. Layer norm 1
        fp::record_pass(
            &mut enc,
            &self.shaders.layer_norm,
            &self.bg_ln1,
            fp::elementwise_dispatch(seq),
            "ln1",
        );

        // 2. Q/K/V projections (routed matmul)
        fp::record_pass(&mut enc, proj_pipe, &self.bg_q, proj_disp(seq, d), "proj_q");
        fp::record_pass(&mut enc, proj_pipe, &self.bg_k, proj_disp(seq, d), "proj_k");
        fp::record_pass(&mut enc, proj_pipe, &self.bg_v, proj_disp(seq, d), "proj_v");

        // 3. Head split
        fp::record_pass(
            &mut enc,
            &self.shaders.head_split,
            &self.bg_hs_q,
            fp::elementwise_dispatch(sd),
            "hs_q",
        );
        fp::record_pass(
            &mut enc,
            &self.shaders.head_split,
            &self.bg_hs_k,
            fp::elementwise_dispatch(sd),
            "hs_k",
        );
        fp::record_pass(
            &mut enc,
            &self.shaders.head_split,
            &self.bg_hs_v,
            fp::elementwise_dispatch(sd),
            "hs_v",
        );

        // 4. Batched attention
        fp::record_pass(
            &mut enc,
            &self.shaders.attention,
            &self.bg_attn,
            fp::attention_dispatch(d_head, seq, n_heads),
            "attn",
        );

        // 5. Head concat
        fp::record_pass(
            &mut enc,
            &self.shaders.head_concat,
            &self.bg_hc,
            fp::elementwise_dispatch(sd),
            "hc",
        );

        // 6. Output projection (Wo)
        fp::record_pass(
            &mut enc,
            proj_pipe,
            &self.bg_wo,
            proj_disp(seq, d),
            "proj_wo",
        );

        // 7. Residual 1: input + attn_output
        fp::record_pass(
            &mut enc,
            &self.shaders.add,
            &self.bg_res1,
            fp::elementwise_dispatch(sd),
            "res1",
        );

        // 8. Layer norm 2
        fp::record_pass(
            &mut enc,
            &self.shaders.layer_norm,
            &self.bg_ln2,
            fp::elementwise_dispatch(seq),
            "ln2",
        );

        // 9. FFN layer 1: matmul + add + gelu
        fp::record_pass(
            &mut enc,
            ff1_pipe,
            &self.bg_ff1,
            ff1_disp(seq, d_ff),
            "ff1_mm",
        );
        fp::record_pass(
            &mut enc,
            &self.shaders.add,
            &self.bg_ff1_add,
            fp::elementwise_dispatch(seq * d_ff),
            "ff1_add",
        );
        fp::record_pass(
            &mut enc,
            &self.shaders.gelu,
            &self.bg_gelu,
            fp::elementwise_dispatch(seq * d_ff),
            "ff1_gelu",
        );

        // 10. FFN layer 2: matmul + add
        fp::record_pass(&mut enc, ff2_pipe, &self.bg_ff2, ff2_disp(seq, d), "ff2_mm");
        fp::record_pass(
            &mut enc,
            &self.shaders.add,
            &self.bg_ff2_add,
            fp::elementwise_dispatch(sd),
            "ff2_add",
        );

        // 11. Residual 2: residual1 + ffn_output
        fp::record_pass(
            &mut enc,
            &self.shaders.add,
            &self.bg_res2,
            fp::elementwise_dispatch(sd),
            "res2",
        );

        enc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use barracuda::device::WgpuDevice;
    use std::sync::Arc;

    fn test_device() -> Option<Dev> {
        let rt = tokio::runtime::Runtime::new().ok()?;
        rt.block_on(async { WgpuDevice::new().await.ok().map(|d| Arc::new(d) as Dev) })
    }

    #[test]
    fn transformer_dims_helpers() {
        let cfg = TransformerDims {
            seq_len: 8,
            d_model: 32,
            n_heads: 4,
            d_ff: 64,
            epsilon: 1e-5,
        };
        assert_eq!(cfg.d_head(), 8);
        assert_eq!(cfg.sd(), 256);
    }

    #[test]
    fn fused_transformer_output_shape() {
        let Some(device) = test_device() else { return };

        let cfg = TransformerDims {
            seq_len: 4,
            d_model: 8,
            n_heads: 2,
            d_ff: 16,
            epsilon: 1e-5,
        };
        let sd = cfg.sd();
        let d = cfg.d_model;
        let d_ff = cfg.d_ff;

        let weights = TransformerWeightsRef {
            w_q: &vec![0.01_f32; d * d],
            w_k: &vec![0.01_f32; d * d],
            w_v: &vec![0.01_f32; d * d],
            w_o: &vec![0.01_f32; d * d],
            w_ff1: &vec![0.01_f32; d * d_ff],
            b_ff1: &vec![0.0_f32; d_ff],
            w_ff2: &vec![0.01_f32; d_ff * d],
            b_ff2: &vec![0.0_f32; d],
        };

        let tf = FusedTransformer::new(device, &weights, cfg);
        let input = vec![0.5_f32; sd];
        let output = tf.forward(&input);

        assert_eq!(output.len(), sd);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output element {i} is not finite: {v}");
        }
    }
}
