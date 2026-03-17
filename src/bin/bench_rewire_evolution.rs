// SPDX-License-Identifier: AGPL-3.0-or-later

//! Benchmarks rewired GPU ops: new upstream f64 typed ops vs old f32 Tensor paths.
//!
//! Demonstrates the cross-spring precision and performance evolution:
//!
//! | Op | Old Path | New Path | Origin |
//! |----|----------|----------|--------|
//! | Variance | f32 Tensor (mean→sub→sq→mean) | `VarianceF64` (Welford f64) | hotSpring precision |
//! | Pearson | f32 Tensor (3 dispatches) | `CorrelationF64` (single f64 shader) | wetSpring + hotSpring |
//! | Entropy | f32 Tensor (log→mul→sum) | `FusedMapReduceF64` (fused f64 map-reduce) | wetSpring |
//! | `HillGate` f64 | SKIP (NVVM crash) | `pow_f64` polyfill (S-17) | neuralSpring fix |

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "validation binary"
)]

use barracuda::device::WgpuDevice;
use barracuda::ops::correlation_f64_wgsl::CorrelationF64;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;
use barracuda::ops::variance_f64_wgsl::VarianceF64;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use std::sync::Arc;
use std::time::Instant;

const WARMUP: u32 = 3;
const ITERS: u32 = 20;

type BenchResult<T> = Result<T, Box<dyn std::error::Error>>;

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        println!("FAIL: {e}");
        std::process::exit(1);
    }
}

async fn run() -> BenchResult<()> {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            println!(
                "adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(e) => {
            println!("SKIP: {e}");
            return Ok(());
        }
    };
    let dev = Arc::clone(gpu.wgpu_device());

    println!("\n=== Rewire Evolution Benchmark (Session 52) ===");
    println!("  Warmup: {WARMUP}, Iterations: {ITERS}\n");

    let mut rng = Rng::new(42);
    let n = 10_000_usize;
    let data: Vec<f64> = (0..n).map(|_| rng.uniform() * 10.0).collect();
    let data2: Vec<f64> = (0..n).map(|_| rng.uniform() * 10.0).collect();
    let probs: Vec<f64> = {
        let raw: Vec<f64> = (0..n).map(|_| rng.uniform().max(1e-12)).collect();
        let sum: f64 = raw.iter().sum();
        raw.iter().map(|&r| r / sum).collect()
    };

    println!("--- Variance ({n} elements) ---");
    let old = bench_variance_old(&data, &dev)?;
    let new = bench_variance_new(&data, &dev);
    report(
        "Variance (f32 Tensor)",
        old,
        "Variance (f64 Welford)",
        new,
        "hotSpring precision",
    );

    println!("--- Pearson Correlation ({n} elements) ---");
    let old = bench_pearson_old(&data, &data2, &dev)?;
    let new = bench_pearson_new(&data, &data2, &dev)?;
    report(
        "Pearson (f32 Tensor)",
        old,
        "Pearson (f64 CorrelationF64)",
        new,
        "wetSpring + hotSpring",
    );

    println!("--- Shannon Entropy ({n} elements) ---");
    let old = bench_entropy_old(&probs, &dev)?;
    let new = bench_entropy_new(&probs, &dev)?;
    report(
        "Entropy (f32 Tensor)",
        old,
        "Entropy (f64 FusedMapReduce)",
        new,
        "wetSpring fused",
    );

    println!("\n=== Cross-Spring Provenance ===\n");
    println!("  VarianceF64        : hotSpring Welford algorithm → BarraCUDA → neuralSpring");
    println!("  CorrelationF64      : wetSpring bio stats → hotSpring f64 precision → BarraCUDA → neuralSpring");
    println!("  FusedMapReduceF64   : wetSpring fused map-reduce → BarraCUDA → neuralSpring");
    println!("  pow_f64 polyfill    : hotSpring math_f64.wgsl + wetSpring (zero+literal) fix → S-17 HillGate fix");
    println!("  HillGate f64        : neuralSpring metalForge → BarraCUDA (ToadStool absorption) → S-17 pow polyfill");
    Ok(())
}

fn report(old_name: &str, old_us: f64, new_name: &str, new_us: f64, origin: &str) {
    let speedup = old_us / new_us;
    let arrow = if speedup > 1.0 { "↑" } else { "↓" };
    println!("  {old_name:38} : {old_us:8.1} µs");
    println!("  {new_name:38} : {new_us:8.1} µs  ({speedup:.2}× {arrow})  [origin: {origin}]");
    println!();
}

fn bench_variance_old(data: &[f64], dev: &Arc<WgpuDevice>) -> BenchResult<f64> {
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let n = data_f32.len();
    for _ in 0..WARMUP {
        let t = Tensor::from_data(&data_f32, vec![n], dev.clone())?;
        let m = t.mean()?.to_vec()?[0];
        let mv = vec![m; n];
        let mb = Tensor::from_data(&mv, vec![n], dev.clone())?;
        let d = t.sub(&mb)?;
        let _ = d.mul(&d)?.mean()?.to_vec()?;
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let t = Tensor::from_data(&data_f32, vec![n], dev.clone())?;
        let m = t.mean()?.to_vec()?[0];
        let mv = vec![m; n];
        let mb = Tensor::from_data(&mv, vec![n], dev.clone())?;
        let d = t.sub(&mb)?;
        let _ = d.mul(&d)?.mean()?.to_vec()?;
    }
    Ok(start.elapsed().as_micros() as f64 / f64::from(ITERS))
}

fn bench_variance_new(data: &[f64], dev: &Arc<WgpuDevice>) -> f64 {
    let Ok(var_op) = VarianceF64::new(dev.clone()) else {
        return f64::NAN;
    };
    for _ in 0..WARMUP {
        let _ = var_op.variance(data);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = var_op.variance(data);
    }
    start.elapsed().as_micros() as f64 / f64::from(ITERS)
}

fn bench_pearson_old(x: &[f64], y: &[f64], dev: &Arc<WgpuDevice>) -> BenchResult<f64> {
    let x_f32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let y_f32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let n = x_f32.len();
    for _ in 0..WARMUP {
        let xt = Tensor::from_data(&x_f32, vec![n], dev.clone())?;
        let yt = Tensor::from_data(&y_f32, vec![n], dev.clone())?;
        let _ = xt.mul(&yt)?.sum()?.to_vec()?;
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let xt = Tensor::from_data(&x_f32, vec![n], dev.clone())?;
        let yt = Tensor::from_data(&y_f32, vec![n], dev.clone())?;
        let _ = xt.mul(&yt)?.sum()?.to_vec()?;
    }
    Ok(start.elapsed().as_micros() as f64 / f64::from(ITERS))
}

fn bench_pearson_new(x: &[f64], y: &[f64], dev: &Arc<WgpuDevice>) -> BenchResult<f64> {
    let op = CorrelationF64::new(dev.clone()).map_err(|e| format!("CorrelationF64 init: {e}"))?;
    for _ in 0..WARMUP {
        let _ = op.correlation(x, y);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = op.correlation(x, y);
    }
    Ok(start.elapsed().as_micros() as f64 / f64::from(ITERS))
}

fn bench_entropy_old(probs: &[f64], dev: &Arc<WgpuDevice>) -> BenchResult<f64> {
    let p_f32: Vec<f32> = probs.iter().map(|&p| p.max(1e-30) as f32).collect();
    let n = p_f32.len();
    for _ in 0..WARMUP {
        let pl = Tensor::from_data(&p_f32, vec![n], dev.clone())?;
        let pm = Tensor::from_data(&p_f32, vec![n], dev.clone())?;
        let lp = pl.log_wgsl()?;
        let _ = pm.mul(&lp)?.sum()?.to_vec()?;
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let pl = Tensor::from_data(&p_f32, vec![n], dev.clone())?;
        let pm = Tensor::from_data(&p_f32, vec![n], dev.clone())?;
        let lp = pl.log_wgsl()?;
        let _ = pm.mul(&lp)?.sum()?.to_vec()?;
    }
    Ok(start.elapsed().as_micros() as f64 / f64::from(ITERS))
}

fn bench_entropy_new(probs: &[f64], dev: &Arc<WgpuDevice>) -> BenchResult<f64> {
    let op =
        FusedMapReduceF64::new(dev.clone()).map_err(|e| format!("FusedMapReduceF64 init: {e}"))?;
    for _ in 0..WARMUP {
        let _ = op.shannon_entropy(probs);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = op.shannon_entropy(probs);
    }
    Ok(start.elapsed().as_micros() as f64 / f64::from(ITERS))
}
