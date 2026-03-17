// SPDX-License-Identifier: AGPL-3.0-or-later

//! MLP inference benchmark on `BarraCUDA` Tensor ops.
//!
//! Proves a real neural network (3-layer MLP with `ReLU` + softmax) runs
//! entirely on WGSL shaders — same math on CPU and GPU.
//!
//! Architecture: input(4) → 64 (`ReLU`) → 64 (`ReLU`) → 10 (softmax)
//!
//! Weights come from `control/ml_inference/mlp_baseline.json` (seed=42,
//! Xavier init, `NumPy` provenance).
//!
//! ## Usage
//!
//! ```text
//! GPU_BACKEND=gpu  cargo run --release --bin bench_mlp_inference
//! GPU_BACKEND=cpu  cargo run --release --bin bench_mlp_inference
//! ```

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use std::sync::Arc;
use std::time::{Duration, Instant};

const WARMUP: usize = 5;
const ITERATIONS: usize = 50;

#[derive(serde::Deserialize)]
struct MlpBaseline {
    input: Vec<f32>,
    weights: Vec<Vec<f32>>,
    weight_shapes: Vec<[usize; 2]>,
    biases: Vec<Vec<f32>>,
    output: Vec<f64>,
    predicted_class: usize,
}

struct MlpWeights {
    w: Vec<Tensor>,
    b: Vec<Tensor>,
}

fn load_baseline() -> Result<MlpBaseline, String> {
    let path = neural_spring::validation::baseline_path("control/ml_inference/mlp_baseline.json");
    let file = std::fs::File::open(&path).map_err(|e| {
        format!("mlp_baseline.json not found — run generate_baselines.py first: {e}")
    })?;
    serde_json::from_reader(std::io::BufReader::new(file))
        .map_err(|e| format!("invalid mlp_baseline.json: {e}"))
}

fn upload_weights(
    baseline: &MlpBaseline,
    device: &Arc<barracuda::device::WgpuDevice>,
) -> Result<MlpWeights, String> {
    let mut w = Vec::with_capacity(baseline.weights.len());
    let mut b = Vec::with_capacity(baseline.biases.len());

    for (i, weights) in baseline.weights.iter().enumerate() {
        let [rows, cols] = baseline.weight_shapes[i];
        let t = Tensor::from_data(weights, vec![rows, cols], device.clone())
            .map_err(|e| format!("weight upload layer {i}: {e}"))?;
        w.push(t);
    }

    for (i, bias) in baseline.biases.iter().enumerate() {
        let cols = bias.len();
        let t = Tensor::from_data(bias, vec![1, cols], device.clone())
            .map_err(|e| format!("bias upload layer {i}: {e}"))?;
        b.push(t);
    }

    Ok(MlpWeights { w, b })
}

fn mlp_forward(input: &Tensor, weights: &MlpWeights) -> Result<Tensor, String> {
    let n: usize = input.shape().iter().product();
    let x = input.reshape(vec![1, n]).map_err(|e| e.to_string())?;

    // Layer 1: linear + ReLU
    let h = x
        .matmul(&weights.w[0])
        .map_err(|e| e.to_string())?
        .add(&weights.b[0])
        .map_err(|e| e.to_string())?
        .relu()
        .map_err(|e| e.to_string())?;

    // Layer 2: linear + ReLU
    let h = h
        .matmul(&weights.w[1])
        .map_err(|e| e.to_string())?
        .add(&weights.b[1])
        .map_err(|e| e.to_string())?
        .relu()
        .map_err(|e| e.to_string())?;

    // Layer 3: linear + softmax
    // NOTE: barracuda's add uses buffer pooling which can return oversized
    // buffers. The softmax shader uses arrayLength() which reads the physical
    // buffer size, not the logical tensor size. We re-upload to force an
    // exact-size buffer. Documented in specs/TOADSTOOL_HANDOFF.md.
    let logits = h
        .matmul(&weights.w[2])
        .map_err(|e| e.to_string())?
        .add(&weights.b[2])
        .map_err(|e| e.to_string())?;
    let logit_data = logits.to_vec().map_err(|e| e.to_string())?;
    Tensor::from_data(
        &logit_data,
        logits.shape().to_vec(),
        logits.device().clone(),
    )
    .map_err(|e| e.to_string())?
    .softmax()
    .map_err(|e| e.to_string())
}

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => g,
        Err(e) => {
            println!("SKIP: {e}");
            return;
        }
    };

    println!(
        "MLP Inference Benchmark: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );

    let device = gpu.wgpu_device().clone();
    let baseline = match load_baseline() {
        Ok(b) => b,
        Err(e) => {
            println!("ERROR: {e}");
            std::process::exit(1);
        }
    };

    let input = match Tensor::from_data(&baseline.input, vec![baseline.input.len()], device.clone())
    {
        Ok(t) => t,
        Err(e) => {
            println!("ERROR: input upload: {e}");
            std::process::exit(1);
        }
    };
    let weights = match upload_weights(&baseline, &device) {
        Ok(w) => w,
        Err(e) => {
            println!("ERROR: {e}");
            std::process::exit(1);
        }
    };

    // Correctness check
    match mlp_forward(&input, &weights) {
        Ok(output) => {
            let probs = match output.to_vec() {
                Ok(p) => p,
                Err(e) => {
                    println!("  Readback ERROR: {e}");
                    return;
                }
            };
            let predicted = probs
                .iter()
                .enumerate()
                .max_by(|a, b| f32::total_cmp(a.1, b.1))
                .map_or(0, |(i, _)| i);

            println!(
                "  Predicted class: {predicted} (expected: {})",
                baseline.predicted_class
            );
            if predicted == baseline.predicted_class {
                println!("  Correctness: PASS");
            } else {
                println!("  Correctness: FAIL (class mismatch)");
            }

            let max_diff: f64 = probs
                .iter()
                .zip(baseline.output.iter())
                .map(|(&p, &e)| (f64::from(p) - e).abs())
                .fold(0.0_f64, f64::max);
            println!("  Max abs diff vs Python: {max_diff:.6e}");
        }
        Err(e) => {
            println!("  Forward pass ERROR: {e}");
            return;
        }
    }

    // Warmup
    for _ in 0..WARMUP {
        let _ = mlp_forward(&input, &weights);
    }

    // Benchmark
    let mut timings = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let _ = mlp_forward(&input, &weights);
        timings.push(start.elapsed());
    }

    timings.sort();
    let median = timings[timings.len() / 2];
    let min_t = timings[0];
    let max_t = timings[timings.len() - 1];
    #[expect(clippy::cast_possible_truncation, reason = "validation binary")]
    let mean_t: Duration = timings.iter().sum::<Duration>() / timings.len() as u32;

    println!();
    println!("  MLP Forward Pass ({ITERATIONS} iterations):");
    println!("    Median:  {}", fmt_dur(median));
    println!("    Mean:    {}", fmt_dur(mean_t));
    println!("    Min:     {}", fmt_dur(min_t));
    println!("    Max:     {}", fmt_dur(max_t));

    let throughput = 1_000_000.0 / median.as_micros() as f64;
    println!("    Throughput: {throughput:.0} inferences/sec");
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
