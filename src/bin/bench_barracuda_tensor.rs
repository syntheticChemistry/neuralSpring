// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-hardware benchmark for `BarraCUDA` Tensor / WGSL ops.
//!
//! Times each op on whichever backend `NEURALSPRING_BACKEND` selects.
//! Run once per backend to compare:
//!
//! ```text
//! NEURALSPRING_BACKEND=gpu  cargo run --release --bin bench_barracuda_tensor
//! NEURALSPRING_BACKEND=cpu  cargo run --release --bin bench_barracuda_tensor
//! ```
//!
//! Reports per-op warm latency (median of N iterations).
//! All ops now use native `BarraCUDA` Tensor APIs (no evolved workarounds).

#![allow(clippy::cast_precision_loss)]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use std::sync::Arc;
use std::time::{Duration, Instant};

const WARMUP: usize = 3;
const ITERATIONS: usize = 20;

#[tokio::main]
async fn main() {
    let Ok(gpu) = Gpu::new().await else {
        eprintln!("SKIP — no adapter");
        return;
    };
    eprintln!(
        "Benchmark: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );
    eprintln!("Warmup: {WARMUP}, iterations: {ITERATIONS}");
    eprintln!();

    let device = gpu.wgpu_device().clone();
    let results = vec![
        bench_op("relu", &device, |dev| {
            let t = mk_tensor(&[256, 256], dev);
            move || {
                let _ = t.clone().relu();
            }
        }),
        bench_op("gelu_wgsl", &device, |dev| {
            let t = mk_tensor(&[256, 256], dev);
            move || {
                let _ = t.clone().gelu_wgsl();
            }
        }),
        bench_op("sigmoid", &device, |dev| {
            let t = mk_tensor(&[256, 256], dev);
            move || {
                let _ = t.clone().sigmoid();
            }
        }),
        bench_op("softmax", &device, |dev| {
            let t = mk_tensor(&[256, 256], dev);
            move || {
                let _ = t.clone().softmax();
            }
        }),
        bench_op("layer_norm_wgsl (stock)", &device, |dev| {
            let t = mk_tensor(&[64, 256], dev);
            #[allow(clippy::cast_possible_truncation)]
            let eps = neural_spring::tolerances::LAYER_NORM_EPS as f32;
            move || {
                let _ = t.clone().layer_norm_wgsl(eps);
            }
        }),
        bench_op("matmul", &device, |dev| {
            let lhs = mk_tensor(&[64, 128], dev);
            let rhs = mk_tensor(&[128, 64], dev);
            move || {
                let _ = lhs.clone().matmul(&rhs);
            }
        }),
        bench_op("add", &device, |dev| {
            let lhs = mk_tensor(&[256, 256], dev);
            let rhs = mk_tensor(&[256, 256], dev);
            move || {
                let _ = lhs.clone().add(&rhs);
            }
        }),
        bench_op("mse_loss", &device, |dev| {
            let pred = mk_tensor(&[256, 256], dev);
            let target = mk_tensor(&[256, 256], dev);
            move || {
                let _ = pred.clone().mse_loss(target.clone());
            }
        }),
        bench_op("layer_norm_wgsl (native)", &device, |dev| {
            let t = mk_tensor(&[64, 256], dev);
            #[allow(clippy::cast_possible_truncation)]
            let eps = neural_spring::tolerances::LAYER_NORM_EPS as f32;
            move || {
                let _ = t.clone().layer_norm_wgsl(eps);
            }
        }),
        bench_op("log_softmax_wgsl (native)", &device, |dev| {
            let t = mk_tensor(&[64, 256], dev);
            move || {
                let _ = t.clone().log_softmax_wgsl();
            }
        }),
    ];

    eprintln!("{:<30} {:>12} {:>12} {:>12}", "op", "median", "min", "max");
    eprintln!("{}", "-".repeat(68));
    for (name, timings) in &results {
        let med = median(timings);
        let min_t = timings.iter().min().copied().unwrap_or_default();
        let max_t = timings.iter().max().copied().unwrap_or_default();
        eprintln!(
            "{:<30} {:>12} {:>12} {:>12}",
            name,
            fmt_dur(med),
            fmt_dur(min_t),
            fmt_dur(max_t),
        );
    }
}

// ── Tensor helper ──────────────────────────────────────────────────────

fn mk_tensor(shape: &[usize], dev: &Arc<barracuda::device::WgpuDevice>) -> Tensor {
    let count: usize = shape.iter().product();
    let data: Vec<f32> = (0..count).map(|i| (i as f32) * 0.001).collect();
    match Tensor::from_data(&data, shape.to_vec(), dev.clone()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("FATAL: mk_tensor GPU alloc: {e}");
            std::process::exit(1);
        }
    }
}

// ── Benchmark harness ──────────────────────────────────────────────────

fn bench_op<F, G>(
    name: &str,
    device: &Arc<barracuda::device::WgpuDevice>,
    setup: F,
) -> (String, Vec<Duration>)
where
    F: FnOnce(&Arc<barracuda::device::WgpuDevice>) -> G,
    G: Fn(),
{
    let op = setup(device);

    for _ in 0..WARMUP {
        op();
    }

    let mut timings = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        op();
        timings.push(start.elapsed());
    }

    (name.to_owned(), timings)
}

// ── Formatting ─────────────────────────────────────────────────────────

fn median(times: &[Duration]) -> Duration {
    let mut sorted: Vec<Duration> = times.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2]
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
