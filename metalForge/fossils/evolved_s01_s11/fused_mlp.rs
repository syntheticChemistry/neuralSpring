// SPDX-License-Identifier: AGPL-3.0-or-later

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

#![expect(clippy::cast_possible_truncation, reason = "fossil record — dimension casts in evolved code")]

use super::fused_pipeline::{self as fp, Dev, MatMulParams, MatmulConfig, ShaderCache};

/// Pre-built fused MLP pipeline with all GPU resources allocated.
///
/// Fields that appear unused keep GPU buffers alive for bind groups.
#[expect(dead_code, reason = "fossil record — evolved code preserved for provenance")]
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
    #[expect(clippy::too_many_lines, reason = "fossil benchmark — monolithic benchmark preserved as-is")]
    pub fn new(device: Dev, weights: [&[f32]; 3], biases: [&[f32]; 3], dims: MlpDims) -> Self {
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
        let (mm0_pipe, _) = fp::select_matmul(
            &shaders,
            1,
            dims.input as u32,
            dims.hidden1 as u32,
            &mm_config,
        );
        let (mm1_pipe, _) = fp::select_matmul(
            &shaders,
            1,
            dims.hidden1 as u32,
            dims.hidden2 as u32,
            &mm_config,
        );
        let (mm2_pipe, _) = fp::select_matmul(
            &shaders,
            1,
            dims.hidden2 as u32,
            dims.output as u32,
            &mm_config,
        );
        let bg_mm = [
            fp::bind_group(
                &device,
                mm0_pipe,
                &[&input_buf, &w[0], &mm_out[0], &mm_params[0]],
                "mlp_bg_mm0",
            ),
            fp::bind_group(
                &device,
                mm1_pipe,
                &[&relu_out[0], &w[1], &mm_out[1], &mm_params[1]],
                "mlp_bg_mm1",
            ),
            fp::bind_group(
                &device,
                mm2_pipe,
                &[&relu_out[1], &w[2], &mm_out[2], &mm_params[2]],
                "mlp_bg_mm2",
            ),
        ];
        // add: (a, b, output)
        let bg_add = [
            fp::bind_group(
                &device,
                &shaders.add,
                &[&mm_out[0], &b[0], &add_out[0]],
                "mlp_bg_add0",
            ),
            fp::bind_group(
                &device,
                &shaders.add,
                &[&mm_out[1], &b[1], &add_out[1]],
                "mlp_bg_add1",
            ),
            fp::bind_group(
                &device,
                &shaders.add,
                &[&mm_out[2], &b[2], &softmax_in],
                "mlp_bg_add2",
            ),
        ];
        // relu: (input, output)
        let bg_relu = [
            fp::bind_group(
                &device,
                &shaders.relu,
                &[&add_out[0], &relu_out[0]],
                "mlp_bg_relu0",
            ),
            fp::bind_group(
                &device,
                &shaders.relu,
                &[&add_out[1], &relu_out[1]],
                "mlp_bg_relu1",
            ),
        ];
        // softmax: (input, output)
        let bg_softmax = fp::bind_group(
            &device,
            &shaders.softmax,
            &[&softmax_in, &softmax_out],
            "mlp_bg_softmax",
        );

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

    /// Record all 9 compute passes (3× matmul + 3× add + 2× relu + softmax)
    /// into the given encoder.
    fn record_passes(&self, encoder: &mut wgpu::CommandEncoder) {
        let layer_dims = [
            (self.dims.input, self.dims.hidden1),
            (self.dims.hidden1, self.dims.hidden2),
            (self.dims.hidden2, self.dims.output),
        ];

        for (i, &(k, n)) in layer_dims.iter().enumerate() {
            let (mm_pipe, mm_disp) =
                fp::select_matmul(&self.shaders, 1, k as u32, n as u32, &self.mm_config);
            fp::record_pass(encoder, mm_pipe, &self.bg_mm[i], mm_disp(1, n as u32), "mm");
            fp::record_pass(
                encoder,
                &self.shaders.add,
                &self.bg_add[i],
                fp::elementwise_dispatch(n as u32),
                "add",
            );
            if i < 2 {
                fp::record_pass(
                    encoder,
                    &self.shaders.relu,
                    &self.bg_relu[i],
                    fp::elementwise_dispatch(n as u32),
                    "relu",
                );
            }
        }

        fp::record_pass(
            encoder,
            &self.shaders.softmax,
            &self.bg_softmax,
            fp::elementwise_dispatch(self.dims.output as u32),
            "softmax",
        );
    }

    /// Upload input data and run a fused forward pass.
    ///
    /// Returns the softmax output as a CPU `Vec<f32>`.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.device
            .queue()
            .write_buffer(&self.input_buf, 0, bytemuck::cast_slice(input));

        let mut encoder = fp::new_encoder(&self.device, "fused_mlp_forward");
        self.record_passes(&mut encoder);
        fp::submit_and_wait(&self.device, encoder);
        fp::readback(&self.device, &self.softmax_out, self.dims.output)
    }

    /// Run forward pass without readback (for throughput benchmarking).
    pub fn forward_no_readback(&self, input: &[f32]) {
        self.device
            .queue()
            .write_buffer(&self.input_buf, 0, bytemuck::cast_slice(input));

        let mut encoder = fp::new_encoder(&self.device, "fused_mlp_no_readback");
        self.record_passes(&mut encoder);
        self.device.queue().submit(Some(encoder.finish()));
        self.device.device().poll(wgpu::Maintain::Wait);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use barracuda::device::WgpuDevice;

    fn test_device() -> Option<Dev> {
        let rt = tokio::runtime::Runtime::new().ok()?;
        rt.block_on(async {
            WgpuDevice::new()
                .await
                .ok()
                .map(|d| std::sync::Arc::new(d) as Dev)
        })
    }

    #[test]
    fn fused_mlp_output_sums_to_one() {
        let Some(device) = test_device() else { return };

        let dims = MlpDims {
            input: 4,
            hidden1: 8,
            hidden2: 8,
            output: 3,
        };

        let w0 = vec![0.1_f32; dims.input * dims.hidden1];
        let w1 = vec![0.1_f32; dims.hidden1 * dims.hidden2];
        let w2 = vec![0.1_f32; dims.hidden2 * dims.output];
        let b0 = vec![0.0_f32; dims.hidden1];
        let b1 = vec![0.0_f32; dims.hidden2];
        let b2 = vec![0.0_f32; dims.output];

        let mlp = FusedMlp::new(device, [&w0, &w1, &w2], [&b0, &b1, &b2], dims);

        let input = vec![1.0_f32, 0.5, -0.5, 0.2];
        let output = mlp.forward(&input);

        assert_eq!(output.len(), 3);
        let sum: f32 = output.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "softmax output should sum to ~1.0, got {sum}"
        );
        for (i, &v) in output.iter().enumerate() {
            assert!(v >= 0.0, "softmax element {i} should be >= 0, got {v}");
        }
    }

    #[test]
    fn fused_mlp_dims_correct() {
        let dims = MlpDims {
            input: 10,
            hidden1: 32,
            hidden2: 16,
            output: 5,
        };
        assert_eq!(dims.input, 10);
        assert_eq!(dims.output, 5);
    }
}
