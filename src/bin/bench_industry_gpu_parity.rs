// SPDX-License-Identifier: AGPL-3.0-or-later

//! Industry GPU benchmark parity: `BarraCUDA` WGSL vs `cuBLAS`/`cuDNN`/`cuFFT`.
//!
//! Runs `BarraCUDA` GPU operations at the same scales as the Python/CUDA
//! control scripts in `control/industry_gpu/`, then invokes each Python
//! script (which calls cuBLAS, cuDNN, cuFFT via `PyTorch`) and compares
//! timings and numerical results.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release --bin bench_industry_gpu_parity
//! cargo run --release --bin bench_industry_gpu_parity -- --with-python
//! ```
//!
//! `--with-python` invokes the CUDA control scripts on the same GPU.
//! Without it, only `BarraCUDA` timings are reported.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "benchmark binary with index → f32/f64 conversions"
)]

use barracuda::device::WgpuDevice;
use barracuda::ops::fft::{Fft1D, Rfft};
use barracuda::ops::mha::MultiHeadAttention;
use barracuda::tensor::Tensor;
use neural_spring::validation::{baseline_path, bench_median};
use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;

const WARMUP: usize = 50;
const ITERATIONS: usize = 200;

// ═══════════════════════════════════════════════════════════════════════
// Entry type
// ═══════════════════════════════════════════════════════════════════════

struct IndustryEntry {
    category: &'static str,
    kernel: String,
    scale: String,
    barracuda_us: f64,
    cuda_us: Option<f64>,
}

// ═══════════════════════════════════════════════════════════════════════
// Tensor helpers
// ═══════════════════════════════════════════════════════════════════════

fn mk_tensor(shape: &[usize], dev: &Arc<WgpuDevice>) -> Tensor {
    let count: usize = shape.iter().product();
    let data: Vec<f32> = (0..count).map(|i| (i as f32) * 0.001).collect();
    Tensor::from_data(&data, shape.to_vec(), dev.clone()).unwrap_or_else(|e| {
        println!("FATAL: mk_tensor failed: {e}");
        std::process::exit(1);
    })
}

fn mk_randn(shape: &[usize], dev: &Arc<WgpuDevice>, seed_offset: u64) -> Tensor {
    let count: usize = shape.iter().product();
    let mut rng = neural_spring::rng::Rng::new(42 + seed_offset);
    let data: Vec<f32> = (0..count).map(|_| rng.normal() as f32).collect();
    Tensor::from_data(&data, shape.to_vec(), dev.clone()).unwrap_or_else(|e| {
        println!("FATAL: mk_randn failed: {e}");
        std::process::exit(1);
    })
}

// ═══════════════════════════════════════════════════════════════════════
// cuBLAS GEMM parity — Tensor::matmul
// ═══════════════════════════════════════════════════════════════════════

fn bench_gemm(results: &mut Vec<IndustryEntry>, device: &Arc<WgpuDevice>) {
    let sgemm_scales: &[usize] = &[64, 128, 256, 512, 1024, 2048];

    for &n in sgemm_scales {
        let a = mk_tensor(&[n, n], device);
        let b = mk_tensor(&[n, n], device);

        let us = bench_median(WARMUP, ITERATIONS, || {
            let _ = a.matmul_ref(&b);
        });

        println!("  SGEMM {n}×{n}: {us:.1} µs");
        results.push(IndustryEntry {
            category: "cuBLAS",
            kernel: format!("SGEMM_{n}"),
            scale: format!("{n}×{n}"),
            barracuda_us: us,
            cuda_us: None,
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════
// cuDNN ops parity — softmax, layer_norm, gelu, sigmoid
// ═══════════════════════════════════════════════════════════════════════

fn bench_cudnn_ops(results: &mut Vec<IndustryEntry>, device: &Arc<WgpuDevice>) {
    let softmax_sizes: &[usize] = &[64, 256, 1024, 4096];
    for &n in softmax_sizes {
        let t = mk_tensor(&[n], device);
        let us = bench_median(WARMUP, ITERATIONS, || {
            let _ = t.clone().softmax();
        });
        println!("  Softmax {n}: {us:.1} µs");
        results.push(IndustryEntry {
            category: "cuDNN",
            kernel: format!("SOFTMAX_{n}"),
            scale: format!("{n}"),
            barracuda_us: us,
            cuda_us: None,
        });
    }

    let layernorm_shapes: &[(usize, usize)] = &[(32, 128), (64, 256), (128, 512)];
    #[expect(clippy::cast_possible_truncation, reason = "tolerance → f32")]
    let eps = neural_spring::tolerances::LAYER_NORM_EPS as f32;
    for &(m, n) in layernorm_shapes {
        let t = mk_tensor(&[m, n], device);
        let us = bench_median(WARMUP, ITERATIONS, || {
            let _ = t.clone().layer_norm_wgsl(eps);
        });
        println!("  LayerNorm {m}×{n}: {us:.1} µs");
        results.push(IndustryEntry {
            category: "cuDNN",
            kernel: format!("LAYERNORM_{m}x{n}"),
            scale: format!("{m}×{n}"),
            barracuda_us: us,
            cuda_us: None,
        });
    }

    let gelu_sizes: &[usize] = &[1024, 4096, 16384];
    for &n in gelu_sizes {
        let t = mk_tensor(&[n], device);
        let us = bench_median(WARMUP, ITERATIONS, || {
            let _ = t.clone().gelu_wgsl();
        });
        println!("  GELU {n}: {us:.1} µs");
        results.push(IndustryEntry {
            category: "cuDNN",
            kernel: format!("GELU_{n}"),
            scale: format!("{n}"),
            barracuda_us: us,
            cuda_us: None,
        });
    }

    let sigmoid_sizes: &[usize] = &[1024, 4096];
    for &n in sigmoid_sizes {
        let t = mk_tensor(&[n], device);
        let us = bench_median(WARMUP, ITERATIONS, || {
            let _ = t.clone().sigmoid();
        });
        println!("  Sigmoid {n}: {us:.1} µs");
        results.push(IndustryEntry {
            category: "cuDNN",
            kernel: format!("SIGMOID_{n}"),
            scale: format!("{n}"),
            barracuda_us: us,
            cuda_us: None,
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════
// cuFFT parity — Fft1D, Rfft
// ═══════════════════════════════════════════════════════════════════════

fn bench_fft(results: &mut Vec<IndustryEntry>, device: &Arc<WgpuDevice>) {
    let fft_sizes: &[usize] = &[256, 1024, 4096, 16384, 65536];

    for &n in fft_sizes {
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let us = bench_median(WARMUP, ITERATIONS, || {
            if let Ok(t) = Tensor::from_data(&data, vec![n], device.clone()) {
                if let Ok(fft) = Fft1D::new(t, n as u32) {
                    let _ = fft.execute();
                }
            }
        });
        println!("  FFT {n}: {us:.1} µs");
        results.push(IndustryEntry {
            category: "cuFFT",
            kernel: format!("FFT_{n}"),
            scale: format!("{n}"),
            barracuda_us: us,
            cuda_us: None,
        });
    }

    for &n in fft_sizes {
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let us = bench_median(WARMUP, ITERATIONS, || {
            if let Ok(t) = Tensor::from_data(&data, vec![n], device.clone()) {
                if let Ok(rfft) = Rfft::new(t, n as u32) {
                    let _ = rfft.execute();
                }
            }
        });
        println!("  RFFT {n}: {us:.1} µs");
        results.push(IndustryEntry {
            category: "cuFFT",
            kernel: format!("RFFT_{n}"),
            scale: format!("{n}"),
            barracuda_us: us,
            cuda_us: None,
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════
// FlashAttention / MHA parity
// ═══════════════════════════════════════════════════════════════════════

fn bench_mha(results: &mut Vec<IndustryEntry>, device: &Arc<WgpuDevice>) {
    let configs: &[(usize, usize, usize)] = &[(32, 64, 4), (64, 128, 8), (128, 256, 8)];

    for &(seq, d_model, n_heads) in configs {
        let q = mk_randn(&[1, seq, d_model], device, 0);
        let k = mk_randn(&[1, seq, d_model], device, 1);
        let v = mk_randn(&[1, seq, d_model], device, 2);

        let w_q = mk_randn(&[d_model, d_model], device, 3);
        let w_k = mk_randn(&[d_model, d_model], device, 4);
        let w_v = mk_randn(&[d_model, d_model], device, 5);
        let w_o = mk_randn(&[d_model, d_model], device, 6);

        let us = bench_median(WARMUP.min(10), ITERATIONS.min(50), || {
            if let Ok(mha) = MultiHeadAttention::new(
                q.clone(),
                k.clone(),
                v.clone(),
                w_q.clone(),
                w_k.clone(),
                w_v.clone(),
                w_o.clone(),
                n_heads,
            ) {
                let _ = mha.execute();
            }
        });

        let label = format!("{seq}×{d_model}×{n_heads}");
        println!("  MHA {label}: {us:.1} µs");
        results.push(IndustryEntry {
            category: "MHA",
            kernel: format!("MHA_{seq}x{d_model}x{n_heads}"),
            scale: label,
            barracuda_us: us,
            cuda_us: None,
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Python control script runner
// ═══════════════════════════════════════════════════════════════════════

fn run_python_industry(script_rel: &str) -> HashMap<String, f64> {
    let script = baseline_path(script_rel);
    let mut timings = HashMap::new();

    if !script.exists() {
        println!("    [skip] {}: not found", script.display());
        return timings;
    }

    let python = std::env::var("NEURALSPRING_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let output = Command::new(&python).arg(&script).output();

    match output {
        Ok(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            for line in stdout.lines() {
                if let Some(idx) = line.find("_US=") {
                    let tag = &line[..idx];
                    let val_str = &line[idx + 4..];
                    if let Ok(v) = val_str.trim().parse::<f64>() {
                        timings.insert(tag.to_string(), v);
                    }
                }
            }
        }
        Ok(o) => {
            println!("    [fail] {script_rel}: exit {}", o.status);
            let stderr = String::from_utf8_lossy(&o.stderr);
            for line in stderr.lines().take(5) {
                println!("      {line}");
            }
        }
        Err(e) => println!("    [skip] {script_rel}: {e}"),
    }

    timings
}

fn match_python_timings(results: &mut [IndustryEntry], timings: &HashMap<String, f64>) {
    for entry in results.iter_mut() {
        let cublas_tag = match entry.category {
            "cuBLAS" => format!("CUBLAS_{}", entry.kernel),
            "cuDNN" => format!("CUDNN_{}", entry.kernel),
            "cuFFT" => format!("CUFFT_{}", entry.kernel),
            "MHA" => entry.kernel.clone(),
            _ => continue,
        };
        if let Some(&us) = timings.get(&cublas_tag) {
            entry.cuda_us = Some(us);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Summary table
// ═══════════════════════════════════════════════════════════════════════

fn print_summary(results: &[IndustryEntry], adapter: &str) {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  INDUSTRY GPU PARITY — BarraCUDA WGSL vs cuBLAS/cuDNN/cuFFT (PyTorch/CUDA)                     ║");
    println!("║  Adapter: {adapter:<84}║");
    println!("║  Warmup: {WARMUP}, Iterations: {ITERATIONS}                                                                          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "{:<10} {:<20} {:>12} {:>14} {:>14} {:>10}",
        "Library", "Kernel", "Scale", "BarraCUDA µs", "CUDA µs", "Ratio"
    );
    println!("{}", "─".repeat(84));

    let mut wins = 0_u32;
    let mut losses = 0_u32;
    let mut comparable = 0_u32;

    for e in results {
        let cuda_str = e
            .cuda_us
            .map_or_else(|| "—".to_string(), |v| format!("{v:.1}"));
        let ratio_str = e.cuda_us.map_or_else(
            || "—".to_string(),
            |cuda| {
                let r = e.barracuda_us / cuda;
                comparable += 1;
                if r < 1.0 {
                    wins += 1;
                } else {
                    losses += 1;
                }
                format!("{r:.2}×")
            },
        );
        println!(
            "{:<10} {:<20} {:>12} {:>14.1} {:>14} {:>10}",
            e.category, e.kernel, e.scale, e.barracuda_us, cuda_str, ratio_str
        );
    }

    println!("{}", "─".repeat(84));
    if comparable > 0 {
        println!("  BarraCUDA faster: {wins}/{comparable}, CUDA faster: {losses}/{comparable}");
        println!("  Ratio < 1.0 = BarraCUDA wins, > 1.0 = CUDA wins");
    }
    println!();

    // Machine-readable stdout
    println!("category\tkernel\tscale\tbarracuda_us\tcuda_us\tratio");
    for e in results {
        let cuda_str = e
            .cuda_us
            .map_or_else(|| "—".to_string(), |v| format!("{v:.1}"));
        let ratio_str = e
            .cuda_us
            .map_or_else(|| "—".to_string(), |c| format!("{:.2}", e.barracuda_us / c));
        println!(
            "{}\t{}\t{}\t{:.1}\t{}\t{}",
            e.category, e.kernel, e.scale, e.barracuda_us, cuda_str, ratio_str
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() {
    let gpu = neural_spring::validation::gpu_or_exit().await;
    let adapter = format!(
        "{} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend
    );
    let device = gpu.wgpu_device().clone();
    let with_python = std::env::args().any(|a| a == "--with-python");

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  neuralSpring — Industry GPU Parity Benchmark                ║");
    println!("║  BarraCUDA WGSL vs cuBLAS/cuDNN/cuFFT/FlashAttention        ║");
    println!("║  Adapter: {:<50}║", gpu.adapter_name);
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    let mut results: Vec<IndustryEntry> = Vec::new();

    println!("── cuBLAS GEMM (Tensor::matmul) ──");
    bench_gemm(&mut results, &device);
    println!();

    println!("── cuDNN ops (softmax, layernorm, gelu, sigmoid) ──");
    bench_cudnn_ops(&mut results, &device);
    println!();

    println!("── cuFFT (Fft1D, Rfft) ──");
    bench_fft(&mut results, &device);
    println!();

    println!("── FlashAttention / MHA ──");
    bench_mha(&mut results, &device);
    println!();

    if with_python {
        println!("── Running Python/CUDA control scripts ──");
        let scripts = [
            "control/industry_gpu/bench_cublas_gemm.py",
            "control/industry_gpu/bench_cudnn_ops.py",
            "control/industry_gpu/bench_cufft.py",
            "control/industry_gpu/bench_flash_attention.py",
        ];
        let mut all_timings = HashMap::new();
        for script in &scripts {
            println!("  Running {script} ...");
            let t = run_python_industry(script);
            println!("    → {} timing(s)", t.len());
            all_timings.extend(t);
        }
        match_python_timings(&mut results, &all_timings);
    } else {
        println!("(run with --with-python to compare against CUDA)");
    }
    println!();

    print_summary(&results, &adapter);
}
