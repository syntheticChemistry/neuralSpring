// SPDX-License-Identifier: AGPL-3.0-only

//! Fused MLP inference — single encoder, single submit.
//!
//! Pre-compiles shaders, pre-allocates all intermediate buffers, and
//! pre-creates all bind groups **once**.  `forward()` records 9 compute
//! passes into one `CommandEncoder` and calls `queue.submit()` once,
//! collapsing 9 dispatches into 1 submission.
//!
//! Architecture: `input(in_dim)` -> `linear(h1)` -> `relu` -> `linear(h2)` -> `relu` -> `linear(out_dim)` -> `softmax`
//!
//! **`ToadStool` handoff**: Once `TensorSession` supports `MatMul` +
//! activation ops, this fused pipeline can be retired.

#![allow(clippy::cast_possible_truncation)]

use super::fused_pipeline::{
    self as fp, Dev, MatMulParams, MatmulConfig, ShaderCache,
};

/// Pre-built fused MLP pipeline with all GPU resources allocated.
///
/// Fields that appear unused keep GPU buffers alive for bind groups.
#[allow(dead_code)]
pub struct FusedMlp {
    shaders: ShaderCache,
    device: Dev,
    mm_config: MatmulConfig,
    // Weights and biases (GPU-resident, immutable after construction)
    w: [wgpu::Buffer; 3],
    b: [wgpu::Buffer; 3],
    // Intermediate buffers (pre-allocated, reused every forward pass)
    input_buf: wgpu::Buffer,
    mm_out: [wgpu::Buffer; 3],
    add_out: [wgpu::Buffer; 2],
    relu_out: [wgpu::Buffer; 2],
    softmax_out: wgpu::Buffer,
    // Matmul param buffers
    mm_params: [wgpu::Buffer; 3],
    // Pre-created bind groups
    bg_mm: [wgpu::BindGroup; 3],
    bg_add: [wgpu::BindGroup; 3],
    bg_relu: [wgpu::BindGroup; 2],
    bg_softmax: wgpu::BindGroup,
    // Dimensions for dispatch
    dims: MlpDims,
}

/// MLP layer dimensions.
#[derive(Clone, Copy)]
pub struct MlpDims {
    pub input: usize,
    pub hidden1: usize,
    pub hidden2: usize,
    pub output: usize,
}

impl FusedMlp {
    /// Build a fused MLP. Compiles shaders, uploads weights, allocates
    /// all intermediates and bind groups once.
    ///
    /// `weights`: `[W0, W1, W2]` as flat f32 slices (row-major).
    /// `biases`: `[b0, b1, b2]` as flat f32 slices.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn new(
        device: Dev,
        weights: [&[f32]; 3],
        biases: [&[f32]; 3],
        dims: MlpDims,
    ) -> Self {
        let shaders = ShaderCache::new(&device);
        let mm_config = fp::MatmulConfig::from_device(&device);

        let w = [
            fp::buf_init(&device, weights[0], "mlp_w0"),
            fp::buf_init(&device, weights[1], "mlp_w1"),
            fp::buf_init(&device, weights[2], "mlp_w2"),
        ];
        let b = [
            fp::buf_init(&device, biases[0], "mlp_b0"),
            fp::buf_init(&device, biases[1], "mlp_b1"),
            fp::buf_init(&device, biases[2], "mlp_b2"),
        ];

        let input_buf = fp::buf_empty(&device, dims.input, "mlp_input");

        // matmul outputs: [1, h1], [1, h2], [1, out]
        let mm_out = [
            fp::buf_empty(&device, dims.hidden1, "mlp_mm0"),
            fp::buf_empty(&device, dims.hidden2, "mlp_mm1"),
            fp::buf_empty(&device, dims.output, "mlp_mm2"),
        ];

        // add outputs: [1, h1], [1, h2] (layer 3 add goes directly to softmax input)
        let add_out = [
            fp::buf_empty(&device, dims.hidden1, "mlp_add0"),
            fp::buf_empty(&device, dims.hidden2, "mlp_add1"),
        ];

        // relu outputs: [1, h1], [1, h2]
        let relu_out = [
            fp::buf_empty(&device, dims.hidden1, "mlp_relu0"),
            fp::buf_empty(&device, dims.hidden2, "mlp_relu1"),
        ];

        // softmax input = mm_out[2] + b[2], output = softmax_out
        let softmax_in = fp::buf_empty(&device, dims.output, "mlp_softmax_in");
        let softmax_out = fp::buf_empty(&device, dims.output, "mlp_softmax_out");

        // Matmul params: each is [M=1, K=in_dim, N=out_dim]
        let layer_dims = [
            (1u32, dims.input as u32, dims.hidden1 as u32),
            (1u32, dims.hidden1 as u32, dims.hidden2 as u32),
            (1u32, dims.hidden2 as u32, dims.output as u32),
        ];
        let mm_params: [wgpu::Buffer; 3] = std::array::from_fn(|i| {
            fp::buf_uniform(
                &device,
                &MatMulParams {
                    m: layer_dims[i].0,
                    k: layer_dims[i].1,
                    n: layer_dims[i].2,
                    _padding: 0,
                },
                &format!("mlp_mm_params_{i}"),
            )
        });

        // Bind groups — router selects naive for M=1 (single sample)
        let (mm0_pipe, _) = fp::select_matmul(&shaders, 1, dims.input as u32, dims.hidden1 as u32, &mm_config);
        let (mm1_pipe, _) = fp::select_matmul(&shaders, 1, dims.hidden1 as u32, dims.hidden2 as u32, &mm_config);
        let (mm2_pipe, _) = fp::select_matmul(&shaders, 1, dims.hidden2 as u32, dims.output as u32, &mm_config);
        let bg_mm = [
            fp::bind_group(&device, mm0_pipe, &[&input_buf, &w[0], &mm_out[0], &mm_params[0]], "mlp_bg_mm0"),
            fp::bind_group(&device, mm1_pipe, &[&relu_out[0], &w[1], &mm_out[1], &mm_params[1]], "mlp_bg_mm1"),
            fp::bind_group(&device, mm2_pipe, &[&relu_out[1], &w[2], &mm_out[2], &mm_params[2]], "mlp_bg_mm2"),
        ];
        // add: (a, b, output)
        let bg_add = [
            fp::bind_group(&device, &shaders.add, &[&mm_out[0], &b[0], &add_out[0]], "mlp_bg_add0"),
            fp::bind_group(&device, &shaders.add, &[&mm_out[1], &b[1], &add_out[1]], "mlp_bg_add1"),
            fp::bind_group(&device, &shaders.add, &[&mm_out[2], &b[2], &softmax_in], "mlp_bg_add2"),
        ];
        // relu: (input, output)
        let bg_relu = [
            fp::bind_group(&device, &shaders.relu, &[&add_out[0], &relu_out[0]], "mlp_bg_relu0"),
            fp::bind_group(&device, &shaders.relu, &[&add_out[1], &relu_out[1]], "mlp_bg_relu1"),
        ];
        // softmax: (input, output)
        let bg_softmax = fp::bind_group(&device, &shaders.softmax, &[&softmax_in, &softmax_out], "mlp_bg_softmax");

        Self {
            shaders,
            device,
            mm_config,
            w,
            b,
            input_buf,
            mm_out,
            add_out,
            relu_out,
            softmax_out,
            mm_params,
            bg_mm,
            bg_add,
            bg_relu,
            bg_softmax,
            dims,
        }
    }

    /// Upload input data and run a fused forward pass.
    ///
    /// Returns the softmax output as a CPU `Vec<f32>`.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Upload input
        self.device.queue().write_buffer(&self.input_buf, 0, bytemuck::cast_slice(input));

        let mut encoder = fp::new_encoder(&self.device, "fused_mlp_forward");

        let (mm0_pipe, mm0_disp) = fp::select_matmul(&self.shaders, 1, self.dims.input as u32, self.dims.hidden1 as u32, &self.mm_config);
        let (mm1_pipe, mm1_disp) = fp::select_matmul(&self.shaders, 1, self.dims.hidden1 as u32, self.dims.hidden2 as u32, &self.mm_config);
        let (mm2_pipe, mm2_disp) = fp::select_matmul(&self.shaders, 1, self.dims.hidden2 as u32, self.dims.output as u32, &self.mm_config);

        // Layer 1: matmul -> add -> relu
        fp::record_pass(&mut encoder, mm0_pipe, &self.bg_mm[0],
            mm0_disp(1, self.dims.hidden1 as u32), "mlp_mm0");
        fp::record_pass(&mut encoder, &self.shaders.add, &self.bg_add[0],
            fp::elementwise_dispatch(self.dims.hidden1 as u32), "mlp_add0");
        fp::record_pass(&mut encoder, &self.shaders.relu, &self.bg_relu[0],
            fp::elementwise_dispatch(self.dims.hidden1 as u32), "mlp_relu0");

        // Layer 2: matmul -> add -> relu
        fp::record_pass(&mut encoder, mm1_pipe, &self.bg_mm[1],
            mm1_disp(1, self.dims.hidden2 as u32), "mlp_mm1");
        fp::record_pass(&mut encoder, &self.shaders.add, &self.bg_add[1],
            fp::elementwise_dispatch(self.dims.hidden2 as u32), "mlp_add1");
        fp::record_pass(&mut encoder, &self.shaders.relu, &self.bg_relu[1],
            fp::elementwise_dispatch(self.dims.hidden2 as u32), "mlp_relu1");

        // Layer 3: matmul -> add -> softmax
        fp::record_pass(&mut encoder, mm2_pipe, &self.bg_mm[2],
            mm2_disp(1, self.dims.output as u32), "mlp_mm2");
        fp::record_pass(&mut encoder, &self.shaders.add, &self.bg_add[2],
            fp::elementwise_dispatch(self.dims.output as u32), "mlp_add2");
        fp::record_pass(&mut encoder, &self.shaders.softmax, &self.bg_softmax,
            fp::elementwise_dispatch(self.dims.output as u32), "mlp_softmax");

        fp::submit_and_wait(&self.device, encoder);

        fp::readback(&self.device, &self.softmax_out, self.dims.output)
    }

    /// Run forward pass without readback (for throughput benchmarking).
    pub fn forward_no_readback(&self, input: &[f32]) {
        self.device.queue().write_buffer(&self.input_buf, 0, bytemuck::cast_slice(input));

        let mut encoder = fp::new_encoder(&self.device, "fused_mlp_no_readback");

        let (mm0_pipe, mm0_disp) = fp::select_matmul(&self.shaders, 1, self.dims.input as u32, self.dims.hidden1 as u32, &self.mm_config);
        let (mm1_pipe, mm1_disp) = fp::select_matmul(&self.shaders, 1, self.dims.hidden1 as u32, self.dims.hidden2 as u32, &self.mm_config);
        let (mm2_pipe, mm2_disp) = fp::select_matmul(&self.shaders, 1, self.dims.hidden2 as u32, self.dims.output as u32, &self.mm_config);

        fp::record_pass(&mut encoder, mm0_pipe, &self.bg_mm[0],
            mm0_disp(1, self.dims.hidden1 as u32), "mm0");
        fp::record_pass(&mut encoder, &self.shaders.add, &self.bg_add[0],
            fp::elementwise_dispatch(self.dims.hidden1 as u32), "add0");
        fp::record_pass(&mut encoder, &self.shaders.relu, &self.bg_relu[0],
            fp::elementwise_dispatch(self.dims.hidden1 as u32), "relu0");

        fp::record_pass(&mut encoder, mm1_pipe, &self.bg_mm[1],
            mm1_disp(1, self.dims.hidden2 as u32), "mm1");
        fp::record_pass(&mut encoder, &self.shaders.add, &self.bg_add[1],
            fp::elementwise_dispatch(self.dims.hidden2 as u32), "add1");
        fp::record_pass(&mut encoder, &self.shaders.relu, &self.bg_relu[1],
            fp::elementwise_dispatch(self.dims.hidden2 as u32), "relu1");

        fp::record_pass(&mut encoder, mm2_pipe, &self.bg_mm[2],
            mm2_disp(1, self.dims.output as u32), "mm2");
        fp::record_pass(&mut encoder, &self.shaders.add, &self.bg_add[2],
            fp::elementwise_dispatch(self.dims.output as u32), "add2");
        fp::record_pass(&mut encoder, &self.shaders.softmax, &self.bg_softmax,
            fp::elementwise_dispatch(self.dims.output as u32), "sm");

        self.device.queue().submit(Some(encoder.finish()));
        self.device.device().poll(wgpu::Maintain::Wait);
    }
}
