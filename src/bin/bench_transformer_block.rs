// SPDX-License-Identifier: AGPL-3.0-or-later

//! Transformer encoder block benchmark on `BarraCUDA` Tensor ops.
//!
//! Proves a single pre-norm transformer encoder block — the building
//! block of GPT, BERT, `AlphaFold`, `ViT`, and every modern AI system —
//! runs entirely on WGSL shaders, same math on CPU and GPU.
//!
//! ## Architecture
//!
//! ```text
//! Input [seq_len, d_model]
//!   → LayerNorm → Q,K,V projections → Multi-Head Attention → + Residual
//!   → LayerNorm → FFN1 (GELU) → FFN2 → + Residual
//!   → Output [seq_len, d_model]
//! ```
//!
//! Config: `d_model`=32, `n_heads`=4, `d_ff`=128, `seq_len`=8.
//!
//! Weights from `control/ml_inference/transformer_baseline.json` (seed=42,
//! Xavier init, `NumPy` provenance).
//!
//! ## Usage
//!
//! ```text
//! NEURALSPRING_BACKEND=gpu  cargo run --release --bin bench_transformer_block
//! NEURALSPRING_BACKEND=cpu  cargo run --release --bin bench_transformer_block
//! ```

#![allow(clippy::cast_precision_loss)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::evolved::mha::multi_head_attention_2d;
use std::sync::Arc;
use std::time::{Duration, Instant};

const WARMUP: usize = 5;
const ITERATIONS: usize = 50;

#[derive(serde::Deserialize)]
struct TransformerConfig {
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
    seq_len: usize,
    epsilon: f32,
}

#[derive(serde::Deserialize)]
struct TransformerWeights {
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
    weights: TransformerWeights,
    output: Vec<f64>,
}

struct GpuWeights {
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
    w_ff1: Tensor,
    b_ff1: Tensor,
    w_ff2: Tensor,
    b_ff2: Tensor,
}

type Dev = Arc<WgpuDevice>;

#[allow(clippy::expect_used)]
fn load_baseline() -> TransformerBaseline {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/control/ml_inference/transformer_baseline.json"
    );
    let data = std::fs::read_to_string(path)
        .expect("transformer_baseline.json not found — run generate_baselines.py first");
    serde_json::from_str(&data).expect("invalid transformer_baseline.json")
}

#[allow(clippy::expect_used)]
fn upload_weights(b: &TransformerBaseline, device: &Dev) -> GpuWeights {
    let d = b.config.d_model;
    let d_ff = b.config.d_ff;
    let seq = b.config.seq_len;
    let t = |data: &[f32], shape: Vec<usize>| -> Tensor {
        Tensor::from_data(data, shape, device.clone()).expect("weight upload")
    };

    GpuWeights {
        w_q: t(&b.weights.w_q, vec![d, d]),
        w_k: t(&b.weights.w_k, vec![d, d]),
        w_v: t(&b.weights.w_v, vec![d, d]),
        w_o: t(&b.weights.w_o, vec![d, d]),
        w_ff1: t(&b.weights.w_ff1, vec![d, d_ff]),
        b_ff1: t(&b.weights.b_ff1, vec![1, d_ff])
            .broadcast(vec![seq, d_ff])
            .expect("b_ff1 broadcast"),
        w_ff2: t(&b.weights.w_ff2, vec![d_ff, d]),
        b_ff2: t(&b.weights.b_ff2, vec![1, d])
            .broadcast(vec![seq, d])
            .expect("b_ff2 broadcast"),
    }
}

/// Full pre-norm transformer encoder block forward pass.
///
/// Uses evolved MHA (matmul projections + CPU head reshape).
/// Native `Tensor::multi_head_attention` projection shaders hang;
/// documented for `ToadStool` as S-03b.
fn transformer_forward(
    input: &Tensor,
    w: &GpuWeights,
    cfg: &TransformerConfig,
    device: &Dev,
) -> Result<Tensor, String> {
    let e = |err: barracuda::error::BarracudaError| err.to_string();

    let x = input
        .reshape(vec![cfg.seq_len, cfg.d_model])
        .map_err(|e_| e_.to_string())?;

    let normed1 = x.clone().layer_norm_wgsl(cfg.epsilon).map_err(&e)?;

    let attn_proj = multi_head_attention_2d(
        &normed1,
        &w.w_q,
        &w.w_k,
        &w.w_v,
        &w.w_o,
        cfg.n_heads,
        device,
    )
    .map_err(&e)?;

    let after_attn = x.add(&attn_proj).map_err(&e)?;

    // ── Pre-norm FFN ────────────────────────────────────────────────
    let normed2 = after_attn
        .clone()
        .layer_norm_wgsl(cfg.epsilon)
        .map_err(&e)?;

    let ffn_hidden = normed2
        .matmul(&w.w_ff1)
        .map_err(&e)?
        .add(&w.b_ff1)
        .map_err(&e)?
        .gelu_wgsl()
        .map_err(&e)?;

    let ffn_out = ffn_hidden
        .matmul(&w.w_ff2)
        .map_err(&e)?
        .add(&w.b_ff2)
        .map_err(&e)?;

    after_attn.add(&ffn_out).map_err(&e)
}

#[allow(clippy::expect_used)]
fn readback(t: &Tensor) -> Vec<f32> {
    t.to_vec().expect("GPU readback failed")
}

#[tokio::main]
#[allow(clippy::expect_used)]
async fn main() {
    let dev = match WgpuDevice::new().await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: {e}");
            return;
        }
    };

    let info = dev.adapter_info();
    eprintln!(
        "Transformer Block Benchmark: {} ({:?}, {:?})",
        info.name, info.device_type, info.backend,
    );

    let device: Dev = Arc::new(dev);
    let baseline = load_baseline();
    let cfg = &baseline.config;

    eprintln!(
        "  Config: d_model={}, n_heads={}, d_ff={}, seq_len={}",
        cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.seq_len,
    );

    let input = Tensor::from_data(
        &baseline.input,
        baseline.input_shape.to_vec(),
        device.clone(),
    )
    .expect("input upload");
    let weights = upload_weights(&baseline, &device);

    // Correctness check
    match transformer_forward(&input, &weights, cfg, &device) {
        Ok(output) => {
            let out_data = readback(&output);
            let max_diff: f64 = out_data
                .iter()
                .zip(baseline.output.iter())
                .map(|(&o, &e)| (f64::from(o) - e).abs())
                .fold(0.0_f64, f64::max);
            let rms_diff: f64 = {
                let sum_sq: f64 = out_data
                    .iter()
                    .zip(baseline.output.iter())
                    .map(|(&o, &e)| {
                        let d = f64::from(o) - e;
                        d * d
                    })
                    .sum();
                (sum_sq / out_data.len() as f64).sqrt()
            };

            eprintln!("  Max abs diff vs Python:  {max_diff:.6e}");
            eprintln!("  RMS diff vs Python:      {rms_diff:.6e}");

            if max_diff < 0.05 {
                eprintln!("  Correctness: PASS (max_diff < 0.05)");
            } else {
                eprintln!("  Correctness: FAIL (max_diff = {max_diff:.4e})");
            }
        }
        Err(e) => {
            eprintln!("  Forward pass ERROR: {e}");
            return;
        }
    }

    // Warmup
    for _ in 0..WARMUP {
        let _ = transformer_forward(&input, &weights, cfg, &device);
    }

    // Benchmark
    let mut timings = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let _ = transformer_forward(&input, &weights, cfg, &device);
        timings.push(start.elapsed());
    }

    timings.sort();
    let median = timings[timings.len() / 2];
    let min_t = timings[0];
    let max_t = timings[timings.len() - 1];
    #[allow(clippy::cast_possible_truncation)]
    let mean_t: Duration = timings.iter().sum::<Duration>() / timings.len() as u32;

    eprintln!();
    eprintln!("  Transformer Block Forward ({ITERATIONS} iterations):");
    eprintln!("    Median:  {}", fmt_dur(median));
    eprintln!("    Mean:    {}", fmt_dur(mean_t));
    eprintln!("    Min:     {}", fmt_dur(min_t));
    eprintln!("    Max:     {}", fmt_dur(max_t));

    let throughput = 1_000_000.0 / median.as_micros() as f64;
    eprintln!("    Throughput: {throughput:.0} blocks/sec");
}

fn fmt_dur(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1_000 {
        format!("{us}µs")
    } else if us < 1_000_000 {
        format!("{:.1}ms", us as f64 / 1_000.0)
    } else {
        format!("{:.2}s", us as f64 / 1_000_000.0)
    }
}
