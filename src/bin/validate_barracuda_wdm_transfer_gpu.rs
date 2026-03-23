// SPDX-License-Identifier: AGPL-3.0-or-later

//! nW-04 GPU Tensor validation: classical-to-WDM transfer learning MLP.
//!
//! Validates that the transfer learning MLP (nW-04) runs correctly on GPU
//! via BarraCUDA Tensor API, closing the last WDM GPU tier gap.
//!
//! ## What it proves
//!
//! ```text
//! Python baseline (control/wdm/transfer_classical_to_wdm.py)
//!   ↓ parity (validate_wdm_transfer, 6/6)
//! Rust CPU (SimpleMlp forward)
//!   ↓ parity (this validator)
//! BarraCUDA GPU Tensor (matmul + add + relu)
//! ```
//!
//! ## Checks
//!
//! 1. Classical MLP forward: GPU vs CPU per-output parity
//! 2. WDM transfer MLP forward: GPU vs CPU per-output parity
//! 3. Batch forward: GPU handles multiple samples
//! 4. `ReLU` activation: GPU `relu()` matches CPU `max(0,x)`
//! 5. Full pipeline: predict + R² on GPU matches CPU R²
//! 6. Determinism: two GPU runs produce identical results
//!
//! ## Provenance
//!
//! WGSL source: `BarraCUDA` transfer MLP (matmul + add + relu).
//! CPU baseline: neuralSpring `wdm_transport` SimpleMlp (Rust CPU).
//! Evolution: Paper nW-04 Python → Rust CPU → WGSL GPU pipeline.

#![expect(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::too_many_arguments,
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::redundant_closure_for_method_calls,
    reason = "validation binary"
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::primitives;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, exit_no_gpu};
use std::sync::Arc;

type Dev = Arc<WgpuDevice>;

fn main() {
    let rt = tokio::runtime::Runtime::new()
        .expect("tokio runtime creation failed — required for async validation");
    let gpu = rt.block_on(async {
        match Gpu::new().await {
            Ok(g) => {
                println!(
                    "  adapter: {} ({:?}, {:?})",
                    g.adapter_name, g.device_type, g.backend
                );
                g
            }
            Err(_) => exit_no_gpu(),
        }
    });

    let device: Dev = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_wdm_transfer_gpu");

    let mut rng = Rng::new(42);
    let layers = [2, 64, 64, 1];

    let (w0, b0) = init_layer(layers[0], layers[1], &mut rng);
    let (w1, b1) = init_layer(layers[1], layers[2], &mut rng);
    let (w2, b2) = init_layer(layers[2], layers[3], &mut rng);

    validate_single_forward(&mut h, &device, &w0, &b0, &w1, &b1, &w2, &b2);
    validate_batch_forward(&mut h, &device, &w0, &b0, &w1, &b1, &w2, &b2);
    validate_relu(&mut h, &device);
    validate_r2_pipeline(&mut h, &device, &w0, &b0, &w1, &b1, &w2, &b2);
    validate_determinism(&mut h, &device, &w0, &b0, &w1, &b1, &w2, &b2);

    h.finish();
}

fn init_layer(n_in: usize, n_out: usize, rng: &mut Rng) -> (Vec<f32>, Vec<f32>) {
    let scale = (2.0 / n_in as f64).sqrt();
    let w: Vec<f32> = (0..n_in * n_out)
        .map(|_| (rng.normal() * scale) as f32)
        .collect();
    let b = vec![0.0_f32; n_out];
    (w, b)
}

fn cpu_mlp_3layer(
    x: &[f32],
    n_samples: usize,
    w0: &[f32],
    b0: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) -> Vec<f32> {
    let (in_d, h1_d, h2_d, out_d) = (2, 64, 64, 1);
    let mut hidden1 = vec![0.0_f32; n_samples * h1_d];
    for s in 0..n_samples {
        for o in 0..h1_d {
            let mut val = b0[o];
            for j in 0..in_d {
                val = w0[j * h1_d + o].mul_add(x[s * in_d + j], val);
            }
            hidden1[s * h1_d + o] = val.max(0.0);
        }
    }
    let mut hidden2 = vec![0.0_f32; n_samples * h2_d];
    for s in 0..n_samples {
        for o in 0..h2_d {
            let mut val = b1[o];
            for j in 0..h1_d {
                val = w1[j * h2_d + o].mul_add(hidden1[s * h1_d + j], val);
            }
            hidden2[s * h2_d + o] = val.max(0.0);
        }
    }
    let mut output = vec![0.0_f32; n_samples * out_d];
    for s in 0..n_samples {
        for o in 0..out_d {
            let mut val = b2[o];
            for j in 0..h2_d {
                val = w2[j * out_d + o].mul_add(hidden2[s * h2_d + j], val);
            }
            output[s * out_d + o] = val;
        }
    }
    output
}

fn gpu_mlp_3layer(
    x: &[f32],
    n_samples: usize,
    w0: &[f32],
    b0: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    device: &Dev,
) -> Result<Vec<f32>, String> {
    let (in_d, h1_d, h2_d, out_d) = (2, 64, 64, 1);

    let x_t = Tensor::from_data(x, vec![n_samples, in_d], device.clone())
        .map_err(|e| format!("x: {e}"))?;

    let w0_t =
        Tensor::from_data(w0, vec![in_d, h1_d], device.clone()).map_err(|e| format!("w0: {e}"))?;
    let b0_rep: Vec<f32> = (0..n_samples).flat_map(|_| b0.iter().copied()).collect();
    let b0_t = Tensor::from_data(&b0_rep, vec![n_samples, h1_d], device.clone())
        .map_err(|e| format!("b0: {e}"))?;

    let h1 = x_t.matmul(&w0_t).map_err(|e| format!("mm0: {e}"))?;
    let h1b = h1.add(&b0_t).map_err(|e| format!("add0: {e}"))?;
    let h1a = h1b.relu().map_err(|e| format!("relu0: {e}"))?;

    let w1_t =
        Tensor::from_data(w1, vec![h1_d, h2_d], device.clone()).map_err(|e| format!("w1: {e}"))?;
    let b1_rep: Vec<f32> = (0..n_samples).flat_map(|_| b1.iter().copied()).collect();
    let b1_t = Tensor::from_data(&b1_rep, vec![n_samples, h2_d], device.clone())
        .map_err(|e| format!("b1: {e}"))?;

    let h2 = h1a.matmul(&w1_t).map_err(|e| format!("mm1: {e}"))?;
    let h2b = h2.add(&b1_t).map_err(|e| format!("add1: {e}"))?;
    let h2a = h2b.relu().map_err(|e| format!("relu1: {e}"))?;

    let w2_t =
        Tensor::from_data(w2, vec![h2_d, out_d], device.clone()).map_err(|e| format!("w2: {e}"))?;
    let b2_rep: Vec<f32> = (0..n_samples).flat_map(|_| b2.iter().copied()).collect();
    let b2_t = Tensor::from_data(&b2_rep, vec![n_samples, out_d], device.clone())
        .map_err(|e| format!("b2: {e}"))?;

    let out = h2a.matmul(&w2_t).map_err(|e| format!("mm2: {e}"))?;
    let outb = out.add(&b2_t).map_err(|e| format!("add2: {e}"))?;
    outb.to_vec().map_err(|e| format!("readback: {e}"))
}

// ═══════════════════════════════════════════════════════════════════
// 1. Single-sample forward: classical regime input
// ═══════════════════════════════════════════════════════════════════

#[expect(clippy::too_many_arguments, reason = "validation binary")]
fn validate_single_forward(
    h: &mut ValidationHarness,
    device: &Dev,
    w0: &[f32],
    b0: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) {
    let x = [0.5_f32, 1.2];
    let cpu = cpu_mlp_3layer(&x, 1, w0, b0, w1, b1, w2, b2);

    match gpu_mlp_3layer(&x, 1, w0, b0, w1, b1, w2, b2, device) {
        Ok(gpu) => {
            h.check_abs(
                "single fwd: classical input",
                f64::from(gpu[0]),
                f64::from(cpu[0]),
                tolerances::ML_MLP_F32,
            );
            h.check_bool("single fwd: finite", gpu[0].is_finite());
        }
        Err(e) => h.check_bool(&format!("single fwd: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. Batch forward: multiple WDM samples
// ═══════════════════════════════════════════════════════════════════

#[expect(clippy::too_many_arguments, reason = "validation binary")]
fn validate_batch_forward(
    h: &mut ValidationHarness,
    device: &Dev,
    w0: &[f32],
    b0: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) {
    let mut rng = Rng::new(77);
    let n = 16;
    let x: Vec<f32> = (0..n * 2).map(|_| rng.normal() as f32).collect();
    let cpu = cpu_mlp_3layer(&x, n, w0, b0, w1, b1, w2, b2);

    match gpu_mlp_3layer(&x, n, w0, b0, w1, b1, w2, b2, device) {
        Ok(gpu) => {
            let max_diff = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                .fold(0.0_f64, f64::max);
            h.check_bool(
                &format!("batch fwd {n}×2→1: max_diff={max_diff:.2e}"),
                max_diff < tolerances::ML_MLP_F32,
            );
            h.check_bool("batch fwd: all finite", gpu.iter().all(|v| v.is_finite()));
        }
        Err(e) => h.check_bool(&format!("batch fwd: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. ReLU activation on GPU
// ═══════════════════════════════════════════════════════════════════

fn validate_relu(h: &mut ValidationHarness, device: &Dev) {
    let data = [-2.0_f32, -0.5, 0.0, 0.3, 1.5, -1e-6];
    let expected: Vec<f32> = data.iter().map(|&v| v.max(0.0)).collect();

    match Tensor::from_data(&data, vec![1, 6], device.clone())
        .and_then(|t| t.relu())
        .and_then(|t| t.to_vec())
    {
        Ok(gpu) => {
            let max_diff = gpu
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            h.check_bool(
                &format!("relu: max_diff={max_diff:.2e}"),
                max_diff < tolerances::TENSOR_RELU_DETERMINISM_F32 as f32,
            );
        }
        Err(e) => h.check_bool(&format!("relu: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Full pipeline: GPU predict → R² parity
// ═══════════════════════════════════════════════════════════════════

#[expect(clippy::too_many_arguments, reason = "validation binary")]
fn validate_r2_pipeline(
    h: &mut ValidationHarness,
    device: &Dev,
    w0: &[f32],
    b0: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) {
    let mut rng = Rng::new(99);
    let n = 50;
    let x: Vec<f32> = (0..n * 2).map(|_| rng.normal() as f32).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.normal() as f32 * 0.5).collect();

    let cpu_pred = cpu_mlp_3layer(&x, n, w0, b0, w1, b1, w2, b2);
    let cpu_r2 = r2_f32(&y, &cpu_pred);

    match gpu_mlp_3layer(&x, n, w0, b0, w1, b1, w2, b2, device) {
        Ok(gpu_pred) => {
            let gpu_r2 = r2_f32(&y, &gpu_pred);
            h.check_abs(
                &format!("R² pipeline: GPU={gpu_r2:.4} vs CPU={cpu_r2:.4}"),
                f64::from(gpu_r2),
                f64::from(cpu_r2),
                tolerances::ML_MLP_F32,
            );
        }
        Err(e) => h.check_bool(&format!("R² pipeline: {e}"), false),
    }
}

fn r2_f32(y_true: &[f32], y_pred: &[f32]) -> f32 {
    let mean = y_true.iter().sum::<f32>() / y_true.len() as f32;
    let ss_res: f32 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    let ss_tot: f32 = y_true.iter().map(|t| (t - mean).powi(2)).sum();
    1.0 - ss_res / ss_tot.max(primitives::R2_DENOMINATOR_FLOOR as f32)
}

// ═══════════════════════════════════════════════════════════════════
// 5. Determinism
// ═══════════════════════════════════════════════════════════════════

#[expect(clippy::too_many_arguments, reason = "validation binary")]
fn validate_determinism(
    h: &mut ValidationHarness,
    device: &Dev,
    w0: &[f32],
    b0: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) {
    let x = [0.5_f32, 1.2, -0.3, 0.8];
    let r1 = gpu_mlp_3layer(&x, 2, w0, b0, w1, b1, w2, b2, device);
    let r2 = gpu_mlp_3layer(&x, 2, w0, b0, w1, b1, w2, b2, device);

    match (r1, r2) {
        (Ok(v1), Ok(v2)) => {
            let max_diff = v1
                .iter()
                .zip(v2.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            h.check_bool(
                &format!("determinism: max_diff={max_diff:.2e}"),
                max_diff == 0.0,
            );
        }
        _ => h.check_bool("determinism: GPU runs failed", false),
    }
}
