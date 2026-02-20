// SPDX-License-Identifier: AGPL-3.0-only

//! 4-way ML inference benchmark: Python/NumPy, `BarraCUDA` per-op,
//! `BarraCUDA` fused CPU, `BarraCUDA` fused GPU.
//!
//! Demonstrates that fused dispatch eliminates per-op overhead, and
//! that at meaningful model sizes compiled Rust + GPU parallelism
//! dominate interpreted Python.
//!
//! ## Usage
//!
//! ```text
//! NEURALSPRING_BACKEND=cpu  cargo run --release --bin bench_fused_inference
//! NEURALSPRING_BACKEND=gpu  cargo run --release --bin bench_fused_inference
//! ```

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use barracuda::tensor::Tensor;
use neural_spring::evolved::fused_mlp::{FusedMlp, MlpDims};
use neural_spring::evolved::fused_pipeline::Dev;
use neural_spring::evolved::fused_transformer::{
    FusedTransformer, TransformerDims, TransformerWeightsRef,
};
use neural_spring::evolved::mha::multi_head_attention_2d;
use neural_spring::gpu::Gpu;
use std::process::Command;
use std::time::{Duration, Instant};

const WARMUP: usize = 5;
const ITERATIONS: usize = 100;

// ═══════════════════════════════════════════════════════════════════
// Baseline loading (from existing JSON)
// ═══════════════════════════════════════════════════════════════════

#[derive(serde::Deserialize)]
struct MlpBaseline {
    input: Vec<f32>,
    weights: Vec<Vec<f32>>,
    weight_shapes: Vec<[usize; 2]>,
    biases: Vec<Vec<f32>>,
    output: Vec<f64>,
    #[allow(dead_code)]
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

// ═══════════════════════════════════════════════════════════════════
// Per-op forward passes (reused from existing benchmarks)
// ═══════════════════════════════════════════════════════════════════

#[allow(clippy::expect_used)]
fn mlp_forward_per_op(input: &Tensor, weights: &[Tensor], biases: &[Tensor]) -> Tensor {
    let count: usize = input.shape().iter().product();
    let hidden = input.reshape(vec![1, count]).expect("reshape");

    let hidden = hidden
        .matmul(&weights[0])
        .expect("mm0")
        .add(&biases[0])
        .expect("add0")
        .relu()
        .expect("relu0");
    let hidden = hidden
        .matmul(&weights[1])
        .expect("mm1")
        .add(&biases[1])
        .expect("add1")
        .relu()
        .expect("relu1");
    let logits = hidden
        .matmul(&weights[2])
        .expect("mm2")
        .add(&biases[2])
        .expect("add2");
    let data = logits.to_vec().expect("readback");
    Tensor::from_data(&data, logits.shape().to_vec(), logits.device().clone())
        .expect("re-upload")
        .softmax()
        .expect("softmax")
}

#[allow(clippy::expect_used)]
fn transformer_forward_per_op(
    input: &Tensor,
    cfg: &TransformerConfig,
    tw: &TransformerPerOpWeights,
    device: &Dev,
) -> Tensor {
    let reshaped = input
        .reshape(vec![cfg.seq_len, cfg.d_model])
        .expect("reshape");
    let normed1 = reshaped.clone().layer_norm_wgsl(cfg.epsilon).expect("ln1");
    let attn = multi_head_attention_2d(
        &normed1,
        &tw.w_q,
        &tw.w_k,
        &tw.w_v,
        &tw.w_o,
        cfg.n_heads,
        device,
    )
    .expect("mha");
    let after_attn = reshaped.add(&attn).expect("res1");
    let normed2 = after_attn
        .clone()
        .layer_norm_wgsl(cfg.epsilon)
        .expect("ln2");
    let ffn = normed2
        .matmul(&tw.w_ff1)
        .expect("ff1")
        .add(&tw.b_ff1)
        .expect("ff1_add")
        .gelu_wgsl()
        .expect("gelu");
    let ffn_out = ffn
        .matmul(&tw.w_ff2)
        .expect("ff2")
        .add(&tw.b_ff2)
        .expect("ff2_add");
    after_attn.add(&ffn_out).expect("res2")
}

struct TransformerPerOpWeights {
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
    w_ff1: Tensor,
    b_ff1: Tensor,
    w_ff2: Tensor,
    b_ff2: Tensor,
}

// ═══════════════════════════════════════════════════════════════════
// Timing utilities
// ═══════════════════════════════════════════════════════════════════

fn bench_fn<F: FnMut()>(mut f: F) -> BenchResult {
    for _ in 0..WARMUP {
        f();
    }
    let mut timings = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        f();
        timings.push(start.elapsed());
    }
    timings.sort();
    BenchResult {
        median: timings[timings.len() / 2],
        min: timings[0],
        max: timings[timings.len() - 1],
    }
}

struct BenchResult {
    median: Duration,
    min: Duration,
    max: Duration,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "median={} min={} max={}",
            fmt_dur(self.median),
            fmt_dur(self.min),
            fmt_dur(self.max),
        )
    }
}

fn fmt_dur(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1_000 {
        format!("{us}us")
    } else if us < 1_000_000 {
        format!("{:.1}ms", us as f64 / 1_000.0)
    } else {
        format!("{:.2}s", us as f64 / 1_000_000.0)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Python benchmark (subprocess)
// ═══════════════════════════════════════════════════════════════════

fn bench_python() -> Option<(Duration, Duration)> {
    let script = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/control/ml_inference/bench_inference.py"
    );
    let output = Command::new("python3").arg(script).output().ok()?;
    if !output.status.success() {
        eprintln!(
            "  Python benchmark failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut mlp_us = None;
    let mut tf_us = None;
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("MLP_MEDIAN_US=") {
            mlp_us = rest.trim().parse::<f64>().ok();
        }
        if let Some(rest) = line.strip_prefix("TRANSFORMER_MEDIAN_US=") {
            tf_us = rest.trim().parse::<f64>().ok();
        }
    }
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    Some((
        Duration::from_micros(mlp_us? as u64),
        Duration::from_micros(tf_us? as u64),
    ))
}

// ═══════════════════════════════════════════════════════════════════
// Scaled benchmark (random weights at different model sizes)
// ═══════════════════════════════════════════════════════════════════

#[allow(clippy::cast_precision_loss)]
fn random_f32(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32).mul_add(1.0 / (u64::MAX as f32), -0.5) * 2.0
        })
        .collect()
}

struct ScaleConfig {
    name: &'static str,
    input_dim: usize,
    hidden: usize,
    output_dim: usize,
    seq_len: usize,
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
}

const SCALES: &[ScaleConfig] = &[
    ScaleConfig {
        name: "tiny",
        input_dim: 4,
        hidden: 64,
        output_dim: 10,
        seq_len: 8,
        d_model: 32,
        n_heads: 4,
        d_ff: 128,
    },
    ScaleConfig {
        name: "small",
        input_dim: 32,
        hidden: 128,
        output_dim: 32,
        seq_len: 32,
        d_model: 128,
        n_heads: 8,
        d_ff: 512,
    },
    ScaleConfig {
        name: "medium",
        input_dim: 128,
        hidden: 512,
        output_dim: 128,
        seq_len: 64,
        d_model: 256,
        n_heads: 8,
        d_ff: 1024,
    },
];

fn bench_fused_mlp_scaled(device: &Dev, scale: &ScaleConfig) -> BenchResult {
    let dims = MlpDims {
        input: scale.input_dim,
        hidden1: scale.hidden,
        hidden2: scale.hidden,
        output: scale.output_dim,
    };
    let w0 = random_f32(dims.input * dims.hidden1, 42);
    let w1 = random_f32(dims.hidden1 * dims.hidden2, 43);
    let w2 = random_f32(dims.hidden2 * dims.output, 44);
    let b0 = random_f32(dims.hidden1, 45);
    let b1 = random_f32(dims.hidden2, 46);
    let b2 = random_f32(dims.output, 47);
    let input = random_f32(dims.input, 48);

    let mlp = FusedMlp::new(
        device.clone(),
        [&w0[..], &w1[..], &w2[..]],
        [&b0[..], &b1[..], &b2[..]],
        dims,
    );

    bench_fn(|| mlp.forward_no_readback(&input))
}

fn bench_fused_transformer_scaled(device: &Dev, scale: &ScaleConfig) -> BenchResult {
    let cfg = TransformerDims {
        seq_len: scale.seq_len,
        d_model: scale.d_model,
        n_heads: scale.n_heads,
        d_ff: scale.d_ff,
        epsilon: 1e-5,
    };
    let dd = cfg.d_model * cfg.d_model;
    let df1 = cfg.d_model * cfg.d_ff;
    let df2 = cfg.d_ff * cfg.d_model;
    let weights = TransformerWeightsRef {
        w_q: &random_f32(dd, 50),
        w_k: &random_f32(dd, 51),
        w_v: &random_f32(dd, 52),
        w_o: &random_f32(dd, 53),
        w_ff1: &random_f32(df1, 54),
        b_ff1: &random_f32(cfg.d_ff, 55),
        w_ff2: &random_f32(df2, 56),
        b_ff2: &random_f32(cfg.d_model, 57),
    };
    let input = random_f32(cfg.seq_len * cfg.d_model, 58);

    let transformer = FusedTransformer::new(device.clone(), &weights, cfg);

    bench_fn(|| transformer.forward_no_readback(&input))
}

// ═══════════════════════════════════════════════════════════════════
// Validation (correctness check against Python baseline)
// ═══════════════════════════════════════════════════════════════════

fn validate_fused_mlp(device: &Dev, baseline: &MlpBaseline) -> f64 {
    let dims = MlpDims {
        input: baseline.weight_shapes[0][0],
        hidden1: baseline.weight_shapes[0][1],
        hidden2: baseline.weight_shapes[1][1],
        output: baseline.weight_shapes[2][1],
    };
    let mlp = FusedMlp::new(
        device.clone(),
        [
            &baseline.weights[0],
            &baseline.weights[1],
            &baseline.weights[2],
        ],
        [
            &baseline.biases[0],
            &baseline.biases[1],
            &baseline.biases[2],
        ],
        dims,
    );
    let output = mlp.forward(&baseline.input);
    output
        .iter()
        .zip(baseline.output.iter())
        .map(|(&o, &e)| (f64::from(o) - e).abs())
        .fold(0.0_f64, f64::max)
}

fn validate_fused_transformer(device: &Dev, baseline: &TransformerBaseline) -> f64 {
    let cfg = TransformerDims {
        seq_len: baseline.config.seq_len,
        d_model: baseline.config.d_model,
        n_heads: baseline.config.n_heads,
        d_ff: baseline.config.d_ff,
        epsilon: baseline.config.epsilon,
    };
    let weights = TransformerWeightsRef {
        w_q: &baseline.weights.w_q,
        w_k: &baseline.weights.w_k,
        w_v: &baseline.weights.w_v,
        w_o: &baseline.weights.w_o,
        w_ff1: &baseline.weights.w_ff1,
        b_ff1: &baseline.weights.b_ff1,
        w_ff2: &baseline.weights.w_ff2,
        b_ff2: &baseline.weights.b_ff2,
    };
    let transformer = FusedTransformer::new(device.clone(), &weights, cfg);
    let output = transformer.forward(&baseline.input);
    output
        .iter()
        .zip(baseline.output.iter())
        .map(|(&o, &e)| (f64::from(o) - e).abs())
        .fold(0.0_f64, f64::max)
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

#[allow(clippy::expect_used, clippy::too_many_lines)]
#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: {e}");
            return;
        }
    };
    let device = gpu.wgpu_device().clone();

    eprintln!("=== Fused ToadStool Pipeline Benchmark ===");
    eprintln!(
        "  Adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend
    );
    eprintln!();

    // Load baselines
    let mlp_json: MlpBaseline = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/control/ml_inference/mlp_baseline.json"
        ))
        .expect("mlp_baseline.json not found"),
    )
    .expect("invalid mlp_baseline.json");

    let tf_json: TransformerBaseline = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/control/ml_inference/transformer_baseline.json"
        ))
        .expect("transformer_baseline.json not found"),
    )
    .expect("invalid transformer_baseline.json");

    // ── 1. Validation ────────────────────────────────────────────────
    eprintln!("--- Validation (fused vs Python baseline) ---");
    let mlp_diff = validate_fused_mlp(&device, &mlp_json);
    let tf_diff = validate_fused_transformer(&device, &tf_json);
    eprintln!("  Fused MLP max diff:         {mlp_diff:.6e}");
    eprintln!("  Fused Transformer max diff: {tf_diff:.6e}");
    if mlp_diff < 0.01 {
        eprintln!("  MLP:         PASS");
    } else {
        eprintln!("  MLP:         FAIL (max_diff={mlp_diff:.4e})");
    }
    if tf_diff < 0.05 {
        eprintln!("  Transformer: PASS");
    } else {
        eprintln!("  Transformer: FAIL (max_diff={tf_diff:.4e})");
    }
    eprintln!();

    // ── 2. Python baseline benchmark ──────────────────────────────────
    eprintln!("--- Python/NumPy Benchmark ---");
    let python_times = bench_python();
    if let Some((mlp_py, tf_py)) = python_times {
        eprintln!("  MLP:         {}", fmt_dur(mlp_py));
        eprintln!("  Transformer: {}", fmt_dur(tf_py));
    } else {
        eprintln!("  (Python benchmark not available)");
    }
    eprintln!();

    // ── 3. Per-op benchmark (tiny) ───────────────────────────────────
    eprintln!("--- BarraCUDA Per-Op Benchmark (tiny) ---");
    let mlp_w: Vec<Tensor> = mlp_json
        .weights
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let [r, c] = mlp_json.weight_shapes[i];
            Tensor::from_data(w, vec![r, c], device.clone()).expect("upload")
        })
        .collect();
    let mlp_b: Vec<Tensor> = mlp_json
        .biases
        .iter()
        .map(|b| Tensor::from_data(b, vec![1, b.len()], device.clone()).expect("upload"))
        .collect();
    let mlp_input = Tensor::from_data(&mlp_json.input, vec![mlp_json.input.len()], device.clone())
        .expect("upload");

    let per_op_mlp = bench_fn(|| {
        let _ = mlp_forward_per_op(&mlp_input, &mlp_w, &mlp_b);
    });
    eprintln!("  MLP per-op:         {per_op_mlp}");

    let tcfg = &tf_json.config;
    let tw = TransformerPerOpWeights {
        w_q: Tensor::from_data(
            &tf_json.weights.w_q,
            vec![tcfg.d_model, tcfg.d_model],
            device.clone(),
        )
        .expect("upload"),
        w_k: Tensor::from_data(
            &tf_json.weights.w_k,
            vec![tcfg.d_model, tcfg.d_model],
            device.clone(),
        )
        .expect("upload"),
        w_v: Tensor::from_data(
            &tf_json.weights.w_v,
            vec![tcfg.d_model, tcfg.d_model],
            device.clone(),
        )
        .expect("upload"),
        w_o: Tensor::from_data(
            &tf_json.weights.w_o,
            vec![tcfg.d_model, tcfg.d_model],
            device.clone(),
        )
        .expect("upload"),
        w_ff1: Tensor::from_data(
            &tf_json.weights.w_ff1,
            vec![tcfg.d_model, tcfg.d_ff],
            device.clone(),
        )
        .expect("upload"),
        b_ff1: Tensor::from_data(&tf_json.weights.b_ff1, vec![1, tcfg.d_ff], device.clone())
            .expect("upload")
            .broadcast(vec![tcfg.seq_len, tcfg.d_ff])
            .expect("broadcast"),
        w_ff2: Tensor::from_data(
            &tf_json.weights.w_ff2,
            vec![tcfg.d_ff, tcfg.d_model],
            device.clone(),
        )
        .expect("upload"),
        b_ff2: Tensor::from_data(
            &tf_json.weights.b_ff2,
            vec![1, tcfg.d_model],
            device.clone(),
        )
        .expect("upload")
        .broadcast(vec![tcfg.seq_len, tcfg.d_model])
        .expect("broadcast"),
    };
    let tf_input = Tensor::from_data(&tf_json.input, tf_json.input_shape.to_vec(), device.clone())
        .expect("upload");

    let per_op_tf = bench_fn(|| {
        let _ = transformer_forward_per_op(&tf_input, tcfg, &tw, &device);
    });
    eprintln!("  Transformer per-op: {per_op_tf}");
    eprintln!();

    // ── 4. Fused benchmark (tiny — same weights as baseline) ─────────
    eprintln!("--- BarraCUDA Fused Benchmark (tiny, baseline weights) ---");
    let fused_mlp_dims = MlpDims {
        input: mlp_json.weight_shapes[0][0],
        hidden1: mlp_json.weight_shapes[0][1],
        hidden2: mlp_json.weight_shapes[1][1],
        output: mlp_json.weight_shapes[2][1],
    };
    let fused_mlp = FusedMlp::new(
        device.clone(),
        [
            &mlp_json.weights[0][..],
            &mlp_json.weights[1][..],
            &mlp_json.weights[2][..],
        ],
        [
            &mlp_json.biases[0][..],
            &mlp_json.biases[1][..],
            &mlp_json.biases[2][..],
        ],
        fused_mlp_dims,
    );
    let fused_tiny_mlp = bench_fn(|| fused_mlp.forward_no_readback(&mlp_json.input));
    eprintln!("  MLP fused:          {fused_tiny_mlp}");

    let fused_tf_cfg = TransformerDims {
        seq_len: tcfg.seq_len,
        d_model: tcfg.d_model,
        n_heads: tcfg.n_heads,
        d_ff: tcfg.d_ff,
        epsilon: tcfg.epsilon,
    };
    let fused_tf_weights = TransformerWeightsRef {
        w_q: &tf_json.weights.w_q,
        w_k: &tf_json.weights.w_k,
        w_v: &tf_json.weights.w_v,
        w_o: &tf_json.weights.w_o,
        w_ff1: &tf_json.weights.w_ff1,
        b_ff1: &tf_json.weights.b_ff1,
        w_ff2: &tf_json.weights.w_ff2,
        b_ff2: &tf_json.weights.b_ff2,
    };
    let fused_transformer = FusedTransformer::new(device.clone(), &fused_tf_weights, fused_tf_cfg);
    let fused_tiny_tf = bench_fn(|| fused_transformer.forward_no_readback(&tf_json.input));
    eprintln!("  Transformer fused:  {fused_tiny_tf}");
    eprintln!();

    // ── 5. Scaled benchmarks ─────────────────────────────────────────
    eprintln!("--- Scaled Fused Benchmarks (random weights) ---");
    eprintln!(
        "  {:>8}  {:>16}  {:>16}",
        "Scale", "MLP Fused", "Transformer Fused"
    );
    for scale in SCALES {
        let mlp_r = bench_fused_mlp_scaled(&device, scale);
        let tf_r = bench_fused_transformer_scaled(&device, scale);
        eprintln!(
            "  {:>8}  {:>16}  {:>16}",
            scale.name,
            fmt_dur(mlp_r.median),
            fmt_dur(tf_r.median)
        );
    }
    eprintln!();

    // ── 6. Summary table ─────────────────────────────────────────────
    eprintln!("=== Summary (tiny model, median times) ===");
    eprintln!("  {:>20}  {:>10}  {:>10}", "", "MLP", "Transformer");
    if let Some((mlp_py, tf_py)) = python_times {
        eprintln!(
            "  {:>20}  {:>10}  {:>10}",
            "Python/NumPy",
            fmt_dur(mlp_py),
            fmt_dur(tf_py)
        );
    }
    eprintln!(
        "  {:>20}  {:>10}  {:>10}",
        "BarraCUDA per-op",
        fmt_dur(per_op_mlp.median),
        fmt_dur(per_op_tf.median)
    );
    eprintln!(
        "  {:>20}  {:>10}  {:>10}",
        "BarraCUDA fused",
        fmt_dur(fused_tiny_mlp.median),
        fmt_dur(fused_tiny_tf.median)
    );

    let mlp_speedup =
        per_op_mlp.median.as_micros() as f64 / fused_tiny_mlp.median.as_micros().max(1) as f64;
    let tf_speedup =
        per_op_tf.median.as_micros() as f64 / fused_tiny_tf.median.as_micros().max(1) as f64;
    eprintln!();
    eprintln!("  Fused speedup vs per-op: MLP {mlp_speedup:.1}x, Transformer {tf_speedup:.1}x");

    if let Some((mlp_py, _tf_py)) = python_times {
        let vs_python = mlp_py.as_micros() as f64 / fused_tiny_mlp.median.as_micros().max(1) as f64;
        if vs_python > 1.0 {
            eprintln!("  Fused MLP vs Python: {vs_python:.1}x FASTER (CPU beats Python!)");
        } else {
            eprintln!(
                "  Fused MLP vs Python: {:.1}x slower (overhead still dominates at tiny N)",
                1.0 / vs_python
            );
        }
    }
}
