// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: LeNet-5 Conv2d + `MaxPool2d` (Study 003, gT tier).
//!
//! Wires `Tensor::conv2d()` and `Tensor::maxpool2d()` through `BarraCUDA`'s GPU
//! WGSL shaders and validates against CPU reference implementations.
//!
//! The `BarraCUDA` `Conv2D`/`MaxPool2D` ops currently operate on single-channel 2D
//! tensors `\[H, W\]`. This validator exercises them at the primitive level,
//! proving GPU correctness for the core convolution and pooling operations
//! that compose into the full LeNet-5 pipeline.
//!
//! ## Provenance
//!
//! GPU ops: `barracuda::ops::conv2d::Conv2D`, `barracuda::ops::maxpool2d::MaxPool2D`
//! CPU baseline: analytical convolution and max-pooling (f64 reference)

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "validation binary"
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::require;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, max_abs_diff_gpu_vs_cpu};
use std::sync::Arc;

fn cpu_conv2d(
    input: &[f64],
    kernel: &[f64],
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
) -> Vec<f64> {
    let oh = ih - kh + 1;
    let ow = iw - kw + 1;
    let mut out = vec![0.0_f64; oh * ow];
    for i in 0..oh {
        for j in 0..ow {
            let mut sum = 0.0;
            for ki in 0..kh {
                for kj in 0..kw {
                    sum += input[(i + ki) * iw + (j + kj)] * kernel[ki * kw + kj];
                }
            }
            out[i * ow + j] = sum;
        }
    }
    out
}

fn cpu_maxpool2d(input: &[f64], ih: usize, iw: usize, pool: usize, stride: usize) -> Vec<f64> {
    let oh = ih / stride;
    let ow = iw / stride;
    let mut out = vec![f64::NEG_INFINITY; oh * ow];
    for i in 0..oh {
        for j in 0..ow {
            for pi in 0..pool {
                for pj in 0..pool {
                    let r = i * stride + pi;
                    let c = j * stride + pj;
                    if r < ih && c < iw {
                        let val = input[r * iw + c];
                        if val > out[i * ow + j] {
                            out[i * ow + j] = val;
                        }
                    }
                }
            }
        }
    }
    out
}

fn tensor_from(
    data: &[f32],
    shape: Vec<usize>,
    device: &Arc<WgpuDevice>,
) -> Result<Tensor, barracuda::error::BarracudaError> {
    Tensor::from_data(data, shape, device.clone())
}

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            println!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };
    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_gpu_lenet");

    validate_conv2d_basic(&mut h, &device);
    validate_conv2d_edge_detect(&mut h, &device);
    validate_maxpool2d_basic(&mut h, &device);
    validate_conv_then_pool(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

/// 8x8 input, 3x3 kernel → 6x6 output.
fn validate_conv2d_basic(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = neural_spring::rng::Rng::new(42);
    let ih = 8_usize;
    let iw = 8_usize;
    let kh = 3_usize;
    let kw = 3_usize;

    let input_f64: Vec<f64> = (0..ih * iw).map(|_| rng.uniform()).collect();
    let kernel_f64: Vec<f64> = (0..kh * kw).map(|_| rng.uniform() * 0.5).collect();
    let cpu_out = cpu_conv2d(&input_f64, &kernel_f64, ih, iw, kh, kw);

    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();
    let kernel_f32: Vec<f32> = kernel_f64.iter().map(|&x| x as f32).collect();

    let input_t = require!(
        h,
        tensor_from(&input_f32, vec![ih, iw], device),
        "conv2d input"
    );
    let kernel_t = require!(
        h,
        tensor_from(&kernel_f32, vec![kh, kw], device),
        "conv2d kernel"
    );
    let out_t = require!(h, input_t.conv2d(&kernel_t), "conv2d forward");
    let out = require!(h, out_t.to_vec(), "conv2d readback");

    let diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_out);
    h.check_upper(
        &format!("conv2d 8×8 k=3×3: diff={diff:.2e}"),
        diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
    h.check_bool(
        &format!(
            "conv2d output shape: {} elements (expect {})",
            out.len(),
            6 * 6
        ),
        out.len() == 6 * 6,
    );
}

/// Laplacian edge-detection kernel on 16x16 gradient image.
fn validate_conv2d_edge_detect(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let ih = 16_usize;
    let iw = 16_usize;
    let kh = 3_usize;
    let kw = 3_usize;

    let input_f64: Vec<f64> = (0..ih * iw).map(|i| (i as f64) / 256.0).collect();
    let kernel_f64 = vec![-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];
    let cpu_out = cpu_conv2d(&input_f64, &kernel_f64, ih, iw, kh, kw);

    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();
    let kernel_f32: Vec<f32> = kernel_f64.iter().map(|&x| x as f32).collect();

    let input_t = require!(
        h,
        tensor_from(&input_f32, vec![ih, iw], device),
        "edge input"
    );
    let kernel_t = require!(
        h,
        tensor_from(&kernel_f32, vec![kh, kw], device),
        "edge kernel"
    );
    let out_t = require!(h, input_t.conv2d(&kernel_t), "edge forward");
    let out = require!(h, out_t.to_vec(), "edge readback");

    let diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_out);
    h.check_upper(
        &format!("conv2d edge-detect 16×16: diff={diff:.2e}"),
        diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

/// `MaxPool2d`: 8x8 → 4x4 (pool=2, stride=2).
fn validate_maxpool2d_basic(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = neural_spring::rng::Rng::new(77);
    let ih = 8_usize;
    let iw = 8_usize;
    let pool = 2_usize;
    let stride = 2_usize;

    let input_f64: Vec<f64> = (0..ih * iw).map(|_| rng.uniform()).collect();
    let cpu_out = cpu_maxpool2d(&input_f64, ih, iw, pool, stride);

    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();

    let input_t = require!(
        h,
        tensor_from(&input_f32, vec![ih, iw], device),
        "maxpool input"
    );
    let out_t = require!(h, input_t.maxpool2d(pool, stride), "maxpool forward");
    let out = require!(h, out_t.to_vec(), "maxpool readback");

    let diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_out);
    h.check_upper(
        &format!("maxpool2d 8×8 pool=2: diff={diff:.2e}"),
        diff,
        tolerances::TENSOR_EXACT_F32,
    );
    h.check_bool(
        &format!(
            "maxpool2d output shape: {} elements (expect {})",
            out.len(),
            4 * 4
        ),
        out.len() == 4 * 4,
    );
}

/// Conv2d → `ReLU` → `MaxPool2d` pipeline (single-channel `LeNet` layer).
fn validate_conv_then_pool(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = neural_spring::rng::Rng::new(99);
    let ih = 12_usize;
    let iw = 12_usize;
    let kh = 3_usize;
    let kw = 3_usize;

    let input_f64: Vec<f64> = (0..ih * iw).map(|_| rng.uniform()).collect();
    let kernel_f64: Vec<f64> = (0..kh * kw).map(|_| rng.uniform() * 0.3).collect();

    // CPU: conv → relu → pool
    let conv_out = cpu_conv2d(&input_f64, &kernel_f64, ih, iw, kh, kw);
    let conv_h = ih - kh + 1; // 10
    let conv_w = iw - kw + 1; // 10
    let relu_out: Vec<f64> = conv_out.iter().map(|&x| x.max(0.0)).collect();
    let cpu_pool = cpu_maxpool2d(&relu_out, conv_h, conv_w, 2, 2);

    // GPU: conv → relu → pool
    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();
    let kernel_f32: Vec<f32> = kernel_f64.iter().map(|&x| x as f32).collect();

    let input_t = require!(
        h,
        tensor_from(&input_f32, vec![ih, iw], device),
        "pipe input"
    );
    let kernel_t = require!(
        h,
        tensor_from(&kernel_f32, vec![kh, kw], device),
        "pipe kernel"
    );
    let conv_t = require!(h, input_t.conv2d(&kernel_t), "pipe conv2d");
    let relu_t = require!(h, conv_t.relu(), "pipe relu");
    let pool_t = require!(h, relu_t.maxpool2d(2, 2), "pipe maxpool2d");
    let out = require!(h, pool_t.to_vec(), "pipe readback");

    let diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_pool);
    h.check_upper(
        &format!("conv→relu→pool 12×12 k=3: diff={diff:.2e}"),
        diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
    h.check_bool(
        &format!(
            "pipeline output shape: {} elements (expect {})",
            out.len(),
            5 * 5
        ),
        out.len() == 5 * 5,
    );
}

/// Same input → identical GPU Conv2d output.
fn validate_determinism(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let input_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
    let kernel_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];

    let run = || -> Option<Vec<f32>> {
        let input = Tensor::from_data(&input_data, vec![8, 8], device.clone()).ok()?;
        let kernel = Tensor::from_data(&kernel_data, vec![2, 2], device.clone()).ok()?;
        let out = input.conv2d(&kernel).ok()?;
        out.to_vec().ok()
    };

    let Some(r1) = run() else {
        h.check_bool("conv2d determinism run1 failed", false);
        return;
    };
    let Some(r2) = run() else {
        h.check_bool("conv2d determinism run2 failed", false);
        return;
    };

    h.check_bool(
        "conv2d determinism: bit-identical on rerun",
        r1.iter()
            .zip(r2.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
    );
}
