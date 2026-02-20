// SPDX-License-Identifier: AGPL-3.0-only

//! 3-way scaling benchmark: Python vs `BarraCUDA` CPU vs `BarraCUDA` GPU.
//!
//! Tests from tiny (4→64→64→10) up to xlarge (512→2048→2048→512).
//! The target progression (following `hotSpring`):
//!   **Python (slowest) < `BarraCUDA` CPU < `BarraCUDA` GPU (fastest)**
//!
//! The math lives in the shader. `ToadStool` compiles it:
//! - CPU: WGSL → SPIR-V → LLVM IR → native x86 (llvmpipe)
//! - GPU: WGSL → SPIR-V → Vulkan driver → hardware
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --bin bench_scaling
//! ```

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use neural_spring::evolved::fused_mlp::{FusedMlp, MlpDims};
use neural_spring::evolved::fused_pipeline::Dev;
use neural_spring::evolved::fused_transformer::{
    FusedTransformer, TransformerDims, TransformerWeightsRef,
};
use neural_spring::gpu::Gpu;
use std::process::Command;
use std::time::{Duration, Instant};

const WARMUP: usize = 5;
const ITERATIONS: usize = 100;

struct Scale {
    name: &'static str,
    input: usize,
    hidden: usize,
    output: usize,
    seq: usize,
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
}

const SCALES: &[Scale] = &[
    Scale {
        name: "tiny",
        input: 4,
        hidden: 64,
        output: 10,
        seq: 8,
        d_model: 32,
        n_heads: 4,
        d_ff: 128,
    },
    Scale {
        name: "small",
        input: 32,
        hidden: 128,
        output: 32,
        seq: 32,
        d_model: 128,
        n_heads: 8,
        d_ff: 512,
    },
    Scale {
        name: "medium",
        input: 128,
        hidden: 512,
        output: 128,
        seq: 64,
        d_model: 256,
        n_heads: 8,
        d_ff: 1024,
    },
    Scale {
        name: "large",
        input: 256,
        hidden: 1024,
        output: 256,
        seq: 128,
        d_model: 512,
        n_heads: 8,
        d_ff: 2048,
    },
    Scale {
        name: "xlarge",
        input: 512,
        hidden: 2048,
        output: 512,
        seq: 256,
        d_model: 1024,
        n_heads: 16,
        d_ff: 4096,
    },
];

#[allow(clippy::cast_precision_loss)]
fn random_f32(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32).mul_add(1.0 / (u64::MAX as f32), -0.5) * 0.2
        })
        .collect()
}

fn bench_fn<F: FnMut()>(mut f: F) -> Duration {
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
    timings[timings.len() / 2]
}

fn fmt_us(us: f64) -> String {
    if us <= 0.0 {
        "N/A".to_string()
    } else if us < 1_000.0 {
        format!("{us:.0}us")
    } else if us < 1_000_000.0 {
        format!("{:.1}ms", us / 1_000.0)
    } else {
        format!("{:.2}s", us / 1_000_000.0)
    }
}

fn fmt_ratio(subject_us: f64, baseline_us: f64) -> String {
    if baseline_us <= 0.0 || subject_us <= 0.0 {
        return "—".to_string();
    }
    let r = subject_us / baseline_us;
    if r < 1.0 {
        format!("{:.1}x faster", 1.0 / r)
    } else {
        format!("{r:.1}x slower")
    }
}

struct PythonResults {
    multi_thread: Vec<(String, f64, f64)>,
    single_thread: Vec<(String, f64, f64)>,
}

fn bench_python_scaling() -> PythonResults {
    let script = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/control/ml_inference/bench_scaling.py"
    );
    let output = Command::new("python3").arg(script).output().ok();
    let empty = PythonResults {
        multi_thread: vec![],
        single_thread: vec![],
    };
    let Some(output) = output else { return empty };
    if !output.status.success() {
        return empty;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);

    let parse_set = |prefix: &str| -> Vec<(String, f64, f64)> {
        let mut results = Vec::new();
        for scale_name in &["TINY", "SMALL", "MEDIUM", "LARGE", "XLARGE"] {
            let mlp_key = format!("SCALE_MLP_{prefix}{scale_name}_US=");
            let tf_key = format!("SCALE_TF_{prefix}{scale_name}_US=");
            let mut mlp_us = 0.0;
            let mut tf_us = 0.0;
            for line in stdout.lines() {
                if let Some(rest) = line.strip_prefix(&mlp_key) {
                    mlp_us = rest.trim().parse().unwrap_or(0.0);
                }
                if let Some(rest) = line.strip_prefix(&tf_key) {
                    tf_us = rest.trim().parse().unwrap_or(0.0);
                }
            }
            results.push((scale_name.to_lowercase(), mlp_us, tf_us));
        }
        results
    };

    PythonResults {
        multi_thread: parse_set(""),
        single_thread: parse_set("1T_"),
    }
}

fn bench_fused_mlp(device: &Dev, scale: &Scale) -> Duration {
    let dims = MlpDims {
        input: scale.input,
        hidden1: scale.hidden,
        hidden2: scale.hidden,
        output: scale.output,
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

fn bench_fused_transformer(device: &Dev, scale: &Scale) -> Duration {
    let cfg = TransformerDims {
        seq_len: scale.seq,
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

/// Per-scale results for one backend
struct BackendResults {
    mlp_us: Vec<f64>,
    tf_us: Vec<f64>,
}

fn bench_backend(device: &Dev) -> BackendResults {
    let mut mlp_us = Vec::with_capacity(SCALES.len());
    let mut tf_us = Vec::with_capacity(SCALES.len());
    for scale in SCALES {
        mlp_us.push(bench_fused_mlp(device, scale).as_micros() as f64);
        tf_us.push(bench_fused_transformer(device, scale).as_micros() as f64);
    }
    BackendResults { mlp_us, tf_us }
}

#[allow(clippy::expect_used, clippy::too_many_lines)]
#[tokio::main]
async fn main() {
    eprintln!("=== 3-Way Scaling Benchmark: Python vs BarraCUDA CPU vs GPU ===");
    eprintln!("  Target: Python (slowest) < CPU < GPU (fastest)");
    eprintln!();

    // --- Python baselines ---
    eprintln!("--- [1/3] Python/NumPy (multi-thread + single-thread) ---");
    let py = bench_python_scaling();
    if py.multi_thread.is_empty() {
        eprintln!("  (not available)");
    } else {
        eprintln!("  Done.");
    }

    // --- CPU backend ---
    eprintln!("--- [2/3] BarraCUDA CPU (llvmpipe → LLVM IR → x86) ---");
    let cpu_gpu = Gpu::new_cpu().await;
    let cpu_results = match &cpu_gpu {
        Ok(g) => {
            eprintln!(
                "  Adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            Some(bench_backend(g.wgpu_device()))
        }
        Err(e) => {
            eprintln!("  SKIP: {e}");
            None
        }
    };

    // --- GPU backend ---
    eprintln!("--- [3/3] BarraCUDA GPU (WGSL → SPIR-V → Vulkan) ---");
    let gpu_gpu = Gpu::new_gpu().await;
    let gpu_results = match &gpu_gpu {
        Ok(g) => {
            eprintln!(
                "  Adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            Some(bench_backend(g.wgpu_device()))
        }
        Err(e) => {
            eprintln!("  SKIP: {e}");
            None
        }
    };

    // --- 3-Way MLP Comparison ---
    eprintln!();
    eprintln!("=== MLP Scaling (input→hidden→hidden→output, ReLU+Softmax) ===");
    eprintln!(
        "  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}",
        "Scale", "FLOPs", "Py(1t)", "CPU", "GPU", "CPU/Py", "GPU/Py", "GPU/CPU"
    );
    eprintln!("  {}", "-".repeat(106));

    for (i, scale) in SCALES.iter().enumerate() {
        let h = scale.hidden;
        let flops: usize = 2 * (scale.input * h + h * h + h * scale.output);
        let py_1t = py
            .single_thread
            .iter()
            .find(|(n, _, _)| n == scale.name)
            .map_or(0.0, |(_, m, _)| *m);
        let cpu_us = cpu_results.as_ref().map_or(0.0, |r| r.mlp_us[i]);
        let gpu_us = gpu_results.as_ref().map_or(0.0, |r| r.mlp_us[i]);

        eprintln!(
            "  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}",
            scale.name,
            format!("{flops:>8}"),
            fmt_us(py_1t),
            fmt_us(cpu_us),
            fmt_us(gpu_us),
            fmt_ratio(cpu_us, py_1t),
            fmt_ratio(gpu_us, py_1t),
            fmt_ratio(gpu_us, cpu_us),
        );
    }

    // --- 3-Way Transformer Comparison ---
    eprintln!();
    eprintln!("=== Transformer Scaling (pre-norm encoder block) ===");
    eprintln!(
        "  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}",
        "Scale", "FLOPs", "Py(1t)", "CPU", "GPU", "CPU/Py", "GPU/Py", "GPU/CPU"
    );
    eprintln!("  {}", "-".repeat(106));

    for (i, scale) in SCALES.iter().enumerate() {
        let d = scale.d_model;
        let dff = scale.d_ff;
        let seq = scale.seq;
        let heads = scale.n_heads;
        let d_head = d / heads;
        let flops: usize = 2 * seq * (4 * d * d + 2 * d * dff) + 2 * heads * seq * seq * d_head;
        let py_1t = py
            .single_thread
            .iter()
            .find(|(n, _, _)| n == scale.name)
            .map_or(0.0, |(_, _, t)| *t);
        let cpu_us = cpu_results.as_ref().map_or(0.0, |r| r.tf_us[i]);
        let gpu_us = gpu_results.as_ref().map_or(0.0, |r| r.tf_us[i]);

        eprintln!(
            "  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}",
            scale.name,
            format!("{flops:>8}"),
            fmt_us(py_1t),
            fmt_us(cpu_us),
            fmt_us(gpu_us),
            fmt_ratio(cpu_us, py_1t),
            fmt_ratio(gpu_us, py_1t),
            fmt_ratio(gpu_us, cpu_us),
        );
    }

    // --- Progression check ---
    eprintln!();
    eprintln!("=== Progression Check: Python > CPU > GPU ===");
    eprintln!("  (✓ = target ordering achieved)");
    eprintln!();
    for (i, scale) in SCALES.iter().enumerate() {
        let py_1t_mlp = py
            .single_thread
            .iter()
            .find(|(n, _, _)| n == scale.name)
            .map_or(0.0, |(_, m, _)| *m);
        let py_1t_tf = py
            .single_thread
            .iter()
            .find(|(n, _, _)| n == scale.name)
            .map_or(0.0, |(_, _, t)| *t);
        let cpu_mlp = cpu_results.as_ref().map_or(0.0, |r| r.mlp_us[i]);
        let gpu_mlp = gpu_results.as_ref().map_or(0.0, |r| r.mlp_us[i]);
        let cpu_tf = cpu_results.as_ref().map_or(0.0, |r| r.tf_us[i]);
        let gpu_tf = gpu_results.as_ref().map_or(0.0, |r| r.tf_us[i]);

        let check = |py: f64, cpu: f64, gpu: f64| -> &'static str {
            if py <= 0.0 || cpu <= 0.0 || gpu <= 0.0 {
                return "N/A";
            }
            if gpu < cpu && cpu < py {
                "✓ GPU < CPU < Py"
            } else if gpu < cpu {
                "~ GPU < CPU (CPU still > Py)"
            } else if gpu < py {
                "~ GPU < Py (but CPU > GPU)"
            } else {
                "✗ not yet"
            }
        };

        eprintln!(
            "  {:>8}  MLP: {}  |  TF: {}",
            scale.name,
            check(py_1t_mlp, cpu_mlp, gpu_mlp),
            check(py_1t_tf, cpu_tf, gpu_tf),
        );
    }

    eprintln!();
    eprintln!("=== Analysis ===");
    eprintln!("  The shader IS the math. CPU and GPU execute the same WGSL code.");
    eprintln!("  CPU: WGSL → SPIR-V → LLVM IR → native x86 (llvmpipe, single-core)");
    eprintln!("  GPU: WGSL → SPIR-V → Vulkan driver → RTX hardware (parallel)");
    eprintln!("  Python: interpreter → OpenBLAS (single-thread for Py(1t))");
    eprintln!();
    eprintln!("  BLAS-evolved shader router (DeviceCapabilities-driven):");
    eprintln!("    Tiny M,N:    naive matmul (no shared memory)");
    eprintln!("    CPU:         32x32 tiles, vec4 B-tile, 8x4 micro-kernel, 4x k-unroll");
    eprintln!("    GPU (small): 16x16 shared-memory tiles (high occupancy)");
    eprintln!("    GPU (large): 32x32 double-buffered, 2x2 micro-kernel, vec4, 4x k-unroll");
}
