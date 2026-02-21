// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validates `BarraCUDA` ML inference against `PyTorch`/`NumPy` baselines.
//!
//! Proves that a 3-layer MLP and a transformer encoder block produce the
//! same output on WGSL shaders (CPU or GPU) as `PyTorch`/`NumPy` with
//! identical weights.
//!
//! ## Baselines
//!
//! Weights and expected outputs from `control/ml_inference/`:
//! - `mlp_baseline.json` — MLP: 4 → 64 (`ReLU`) → 64 (`ReLU`) → 10 (softmax)
//! - `transformer_baseline.json` — pre-norm transformer encoder block
//!
//! Provenance: `NumPy` 1.26+, seed=42, Xavier uniform init.
//!
//! ## Usage
//!
//! ```text
//! NEURALSPRING_BACKEND=gpu  cargo run --bin validate_barracuda_ml_inference
//! NEURALSPRING_BACKEND=cpu  cargo run --bin validate_barracuda_ml_inference
//! ```

#![allow(clippy::cast_precision_loss)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::evolved::mha::multi_head_attention_2d;
use neural_spring::gpu::Gpu;
use neural_spring::require;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

type Dev = Arc<WgpuDevice>;

// ═════════════════════════════════════════════════════════════════════════
// JSON structures
// ═════════════════════════════════════════════════════════════════════════

#[derive(serde::Deserialize)]
struct MlpBaseline {
    input: Vec<f32>,
    weights: Vec<Vec<f32>>,
    weight_shapes: Vec<[usize; 2]>,
    biases: Vec<Vec<f32>>,
    output: Vec<f64>,
    predicted_class: usize,
}

#[derive(serde::Deserialize)]
struct TransformerConfig {
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
    seq_len: usize,
    epsilon: f32,
}

#[derive(serde::Deserialize)]
struct TransformerWeightsJson {
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    w_ff1: Vec<f32>,
    b_ff1: Vec<f32>,
    w_ff2: Vec<f32>,
    b_ff2: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct TransformerBaseline {
    config: TransformerConfig,
    input: Vec<f32>,
    input_shape: [usize; 2],
    weights: TransformerWeightsJson,
    output: Vec<f64>,
}

// ═════════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════════

fn readback(t: &Tensor) -> Result<Vec<f32>, barracuda::error::BarracudaError> {
    t.to_vec()
}

fn t(
    data: &[f32],
    shape: Vec<usize>,
    device: &Dev,
) -> Result<Tensor, barracuda::error::BarracudaError> {
    Tensor::from_data(data, shape, device.clone())
}

// ═════════════════════════════════════════════════════════════════════════
// MLP validation
// ═════════════════════════════════════════════════════════════════════════

fn mlp_forward(input: &Tensor, weights: &[Tensor], biases: &[Tensor]) -> Result<Tensor, String> {
    let e = |err: barracuda::error::BarracudaError| err.to_string();

    let h1 = input
        .clone()
        .matmul(&weights[0])
        .map_err(&e)?
        .add(&biases[0])
        .map_err(&e)?
        .relu()
        .map_err(&e)?;

    let h2 = h1
        .matmul(&weights[1])
        .map_err(&e)?
        .add(&biases[1])
        .map_err(&e)?
        .relu()
        .map_err(&e)?;

    // barracuda's buffer pool can return oversized buffers — softmax uses
    // arrayLength() on the physical buffer, not the logical tensor size.
    // Re-upload to force an exact-size buffer.
    let logits = h2
        .matmul(&weights[2])
        .map_err(&e)?
        .add(&biases[2])
        .map_err(&e)?;
    let logit_data = logits.to_vec().map_err(&e)?;
    Tensor::from_data(
        &logit_data,
        logits.shape().to_vec(),
        logits.device().clone(),
    )
    .map_err(&e)?
    .softmax()
    .map_err(&e)
}

fn validate_mlp(h: &mut ValidationHarness, device: &Dev) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/control/ml_inference/mlp_baseline.json"
    );
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            h.check_bool(&format!("MLP baseline load [{e}]"), false);
            return;
        }
    };
    let baseline: MlpBaseline = match serde_json::from_str(&data) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool(&format!("MLP baseline parse [{e}]"), false);
            return;
        }
    };

    let mut weights = Vec::new();
    let mut biases = Vec::new();
    for (i, w_data) in baseline.weights.iter().enumerate() {
        let [rows, cols] = baseline.weight_shapes[i];
        weights.push(require!(
            h,
            t(w_data, vec![rows, cols], device),
            "tensor upload"
        ));
    }
    for b_data in &baseline.biases {
        let cols = b_data.len();
        biases.push(require!(
            h,
            t(b_data, vec![1, cols], device),
            "tensor upload"
        ));
    }

    let input = require!(
        h,
        t(&baseline.input, vec![1, baseline.input.len()], device),
        "tensor upload"
    );
    let result = mlp_forward(&input, &weights, &biases);

    match result {
        Ok(output) => {
            let probs = require!(h, readback(&output), "GPU readback failed");

            // Check predicted class matches
            let predicted = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i);
            h.check_bool(
                "MLP predicted class matches Python",
                predicted == baseline.predicted_class,
            );

            // Check softmax sums to 1
            let prob_sum: f64 = probs.iter().map(|&p| f64::from(p)).sum();
            h.check_abs(
                "MLP softmax sum ≈ 1.0",
                prob_sum,
                1.0,
                tolerances::TENSOR_EXACT_F32,
            );

            // Check all probabilities non-negative
            h.check_bool("MLP all probs ≥ 0", probs.iter().all(|&p| p >= 0.0));

            // Element-wise comparison against Python baseline
            let max_diff: f64 = probs
                .iter()
                .zip(baseline.output.iter())
                .map(|(&o, &e)| (f64::from(o) - e).abs())
                .fold(0.0_f64, f64::max);

            h.check_abs(
                "MLP max diff vs Python",
                max_diff,
                0.0,
                tolerances::ML_MLP_F32,
            );

            // Check a few individual output elements
            for (i, (&observed, &expected)) in
                probs.iter().zip(baseline.output.iter()).enumerate().take(3)
            {
                h.check_abs(
                    &format!("MLP output[{i}]"),
                    f64::from(observed),
                    expected,
                    tolerances::ML_MLP_F32,
                );
            }
        }
        Err(e) => {
            h.check_bool(&format!("MLP forward pass [ERROR: {e}]"), false);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Transformer validation
// ═════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_lines)]
fn validate_transformer(h: &mut ValidationHarness, device: &Dev) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/control/ml_inference/transformer_baseline.json"
    );
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            h.check_bool(&format!("Transformer baseline load [{e}]"), false);
            return;
        }
    };
    let baseline: TransformerBaseline = match serde_json::from_str(&data) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool(&format!("Transformer baseline parse [{e}]"), false);
            return;
        }
    };

    let cfg = &baseline.config;
    let seq = cfg.seq_len;
    let d = cfg.d_model;
    let d_ff = cfg.d_ff;

    let input = require!(
        h,
        t(&baseline.input, baseline.input_shape.to_vec(), device),
        "tensor upload"
    );

    let w_q = require!(
        h,
        t(&baseline.weights.w_q, vec![d, d], device),
        "tensor upload"
    );
    let w_k = require!(
        h,
        t(&baseline.weights.w_k, vec![d, d], device),
        "tensor upload"
    );
    let w_v = require!(
        h,
        t(&baseline.weights.w_v, vec![d, d], device),
        "tensor upload"
    );
    let w_o = require!(
        h,
        t(&baseline.weights.w_o, vec![d, d], device),
        "tensor upload"
    );
    let w_ff1 = require!(
        h,
        t(&baseline.weights.w_ff1, vec![d, d_ff], device),
        "tensor upload"
    );
    let w_ff2 = require!(
        h,
        t(&baseline.weights.w_ff2, vec![d_ff, d], device),
        "tensor upload"
    );

    let b_ff1_row = require!(
        h,
        t(&baseline.weights.b_ff1, vec![1, d_ff], device),
        "tensor upload"
    );
    let b_ff2_row = require!(
        h,
        t(&baseline.weights.b_ff2, vec![1, d], device),
        "tensor upload"
    );

    let result = (|| -> Result<Tensor, String> {
        let e = |err: barracuda::error::BarracudaError| err.to_string();

        let b_ff1 = b_ff1_row.broadcast(vec![seq, d_ff]).map_err(&e)?;
        let b_ff2 = b_ff2_row.broadcast(vec![seq, d]).map_err(&e)?;

        let x = input.reshape(vec![seq, d]).map_err(|e_| e_.to_string())?;

        // Pre-norm attention (evolved MHA: matmul projections + CPU head
        // reshape — native Tensor::multi_head_attention projection shaders
        // hang; documented for ToadStool absorption S-03b)
        let normed1 = x.clone().layer_norm_wgsl(cfg.epsilon).map_err(&e)?;

        let attn_proj =
            multi_head_attention_2d(&normed1, &w_q, &w_k, &w_v, &w_o, cfg.n_heads, device)
                .map_err(&e)?;

        let after_attn = x.add(&attn_proj).map_err(&e)?;

        // Pre-norm FFN
        let normed2 = after_attn
            .clone()
            .layer_norm_wgsl(cfg.epsilon)
            .map_err(&e)?;
        let ffn_hidden = normed2
            .matmul(&w_ff1)
            .map_err(&e)?
            .add(&b_ff1)
            .map_err(&e)?
            .gelu_wgsl()
            .map_err(&e)?;
        let ffn_out = ffn_hidden
            .matmul(&w_ff2)
            .map_err(&e)?
            .add(&b_ff2)
            .map_err(&e)?;

        after_attn.add(&ffn_out).map_err(&e)
    })();

    match result {
        Ok(output) => {
            let out_data = require!(h, readback(&output), "GPU readback failed");

            h.check_bool(
                "Transformer output shape matches",
                output.shape() == [seq, d],
            );

            // Max absolute diff
            let max_diff: f64 = out_data
                .iter()
                .zip(baseline.output.iter())
                .map(|(&o, &e)| (f64::from(o) - e).abs())
                .fold(0.0_f64, f64::max);

            h.check_abs(
                "Transformer max diff vs Python",
                max_diff,
                0.0,
                tolerances::ML_TRANSFORMER_F32,
            );

            // RMS error
            let rms: f64 = {
                let sum_sq: f64 = out_data
                    .iter()
                    .zip(baseline.output.iter())
                    .map(|(&o, &e)| {
                        let delta = f64::from(o) - e;
                        delta * delta
                    })
                    .sum();
                (sum_sq / out_data.len() as f64).sqrt()
            };
            h.check_abs(
                "Transformer RMS diff vs Python",
                rms,
                0.0,
                tolerances::ML_TRANSFORMER_F32,
            );

            // Output norm should be in a reasonable range
            let norm: f64 = out_data
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                .sqrt();
            let expected_norm: f64 = baseline.output.iter().map(|&v| v * v).sum::<f64>().sqrt();
            h.check_rel(
                "Transformer output norm within 10%",
                norm,
                expected_norm,
                0.1,
            );

            // Spot-check first and last elements
            if let (Some(&o_first), Some(&e_first)) = (out_data.first(), baseline.output.first()) {
                h.check_abs(
                    "Transformer output[0]",
                    f64::from(o_first),
                    e_first,
                    tolerances::ML_TRANSFORMER_F32,
                );
            }
            if let (Some(&o_last), Some(&e_last)) = (out_data.last(), baseline.output.last()) {
                h.check_abs(
                    "Transformer output[last]",
                    f64::from(o_last),
                    e_last,
                    tolerances::ML_TRANSFORMER_F32,
                );
            }
        }
        Err(e) => {
            h.check_bool(&format!("Transformer forward pass [ERROR: {e}]"), false);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() {
    let Ok(gpu) = Gpu::new().await else {
        eprintln!("  0/0 checks — skipping gracefully");
        std::process::exit(0);
    };
    eprintln!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );
    let device: Dev = gpu.wgpu_device().clone();

    let label = format!("barracuda_ml_inference[{}]", gpu.adapter_name);
    let mut h = ValidationHarness::new(&label);

    validate_mlp(&mut h, &device);
    validate_transformer(&mut h, &device);

    h.finish();
}
