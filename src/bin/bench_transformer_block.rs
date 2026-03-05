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
//! GPU_BACKEND=gpu  cargo run --release --bin bench_transformer_block
//! GPU_BACKEND=cpu  cargo run --release --bin bench_transformer_block
//! ```

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::evolved::mha::multi_head_attention_2d;
use neural_spring::gpu::Gpu;
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

fn load_baseline() -> Result<TransformerBaseline, String> {
    let path =
        neural_spring::validation::baseline_path("control/ml_inference/transformer_baseline.json");
    let file = std::fs::File::open(&path).map_err(|e| {
        format!("transformer_baseline.json not found — run generate_baselines.py first: {e}")
    })?;
    serde_json::from_reader(std::io::BufReader::new(file))
        .map_err(|e| format!("invalid transformer_baseline.json: {e}"))
}

fn upload_weights(b: &TransformerBaseline, device: &Dev) -> Result<GpuWeights, String> {
    let d = b.config.d_model;
    let d_ff = b.config.d_ff;
    let seq = b.config.seq_len;
    let t = |data: &[f32], shape: Vec<usize>, label: &str| -> Result<Tensor, String> {
        Tensor::from_data(data, shape, device.clone())
            .map_err(|e| format!("weight upload {label}: {e}"))
    };

    Ok(GpuWeights {
        w_q: t(&b.weights.w_q, vec![d, d], "w_q")?,
        w_k: t(&b.weights.w_k, vec![d, d], "w_k")?,
        w_v: t(&b.weights.w_v, vec![d, d], "w_v")?,
        w_o: t(&b.weights.w_o, vec![d, d], "w_o")?,
        w_ff1: t(&b.weights.w_ff1, vec![d, d_ff], "w_ff1")?,
        b_ff1: t(&b.weights.b_ff1, vec![1, d_ff], "b_ff1")?
            .broadcast(vec![seq, d_ff])
            .map_err(|e| format!("b_ff1 broadcast: {e}"))?,
        w_ff2: t(&b.weights.w_ff2, vec![d_ff, d], "w_ff2")?,
        b_ff2: t(&b.weights.b_ff2, vec![1, d], "b_ff2")?
            .broadcast(vec![seq, d])
            .map_err(|e| format!("b_ff2 broadcast: {e}"))?,
    })
}

/// Full pre-norm transformer encoder block forward pass.
///
/// Uses evolved MHA (matmul projections + CPU head reshape).
/// Native `Tensor::multi_head_attention` projection shaders hang;
/// documented for `BarraCUDA` as S-03b.
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

#[tokio::main]
async fn main() {
    let Ok(gpu) = Gpu::new().await else {
        eprintln!("SKIP — no adapter");
        return;
    };
    eprintln!(
        "Transformer Block Benchmark: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );

    let device: Dev = gpu.wgpu_device().clone();
    let baseline = match load_baseline() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
    };
    let cfg = &baseline.config;

    eprintln!(
        "  Config: d_model={}, n_heads={}, d_ff={}, seq_len={}",
        cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.seq_len,
    );

    let input = match Tensor::from_data(
        &baseline.input,
        baseline.input_shape.to_vec(),
        device.clone(),
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: input upload: {e}");
            std::process::exit(1);
        }
    };
    let weights = match upload_weights(&baseline, &device) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
    };

    // Correctness check
    match transformer_forward(&input, &weights, cfg, &device) {
        Ok(output) => {
            let out_data = match output.to_vec() {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("  Readback ERROR: {e}");
                    return;
                }
            };
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
    #[expect(clippy::cast_possible_truncation, reason = "validation binary")]
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
