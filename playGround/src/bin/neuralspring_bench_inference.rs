// SPDX-License-Identifier: AGPL-3.0-or-later

//! barraCuda/WGSL inference benchmark — direct comparison with PyTorch/CUDA.
//!
//! Benchmarks the same operations and model forward passes as
//! `bench/pytorch_baseline.py`, outputting results in compatible format.
//!
//! Two dispatch modes:
//! - **cold** (default): new `TensorSession` per call — measures pipeline
//!   compilation + dispatch (worst case, equivalent to no caching)
//! - **hot** (`--hot`): reuse `TensorSession` with pre-compiled pipelines
//!   — measures pure kernel dispatch (comparable to PyTorch/CUDA)

#![expect(clippy::pedantic, clippy::unwrap_used, reason = "benchmark binary")]

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use barracuda::prelude::{AttentionDims, TensorSession, WgpuDevice};
use clap::Parser;

use neuralspring_playground::hf_hub::{self, HfHub};
use neuralspring_playground::inference::{transformer::TransformerEngine, weights};
use neuralspring_playground::model_config::TransformerConfig;
use neuralspring_playground::secrets::Secrets;

#[derive(Parser)]
#[command(
    name = "neuralspring-bench-inference",
    about = "barraCuda/WGSL inference benchmark — compare with PyTorch/CUDA"
)]
struct Cli {
    /// HuggingFace model ID for forward pass benchmark
    #[arg(long, default_value = "openai-community/gpt2")]
    model: String,

    /// Warmup iterations
    #[arg(long, default_value = "50")]
    warmup: usize,

    /// Benchmark iterations
    #[arg(long, default_value = "200")]
    iters: usize,

    /// Sequence length for benchmarks
    #[arg(long, default_value = "128")]
    seq_len: usize,

    /// Output as JSON
    #[arg(long)]
    json: bool,

    /// Only benchmark individual ops
    #[arg(long)]
    ops_only: bool,

    /// Only benchmark forward pass
    #[arg(long)]
    forward_only: bool,

    /// Hot mode: reuse TensorSession (pre-compiled pipelines)
    /// Without this flag, each iteration creates a fresh session (cold dispatch)
    #[arg(long)]
    hot: bool,

    /// Override HuggingFace token
    #[arg(long)]
    hf_token: Option<String>,
}

struct BenchResult {
    name: String,
    shape: String,
    median_us: f64,
}

fn bench_fn<F: FnMut()>(mut f: F, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }

    let mut timings: Vec<f64> = (0..iters)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_nanos() as f64 / 1000.0
        })
        .collect();

    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    timings[timings.len() / 2]
}

fn bench_ops(
    device: &Arc<WgpuDevice>,
    hidden: usize,
    seq_len: usize,
    num_heads: usize,
    warmup: usize,
    iters: usize,
    hot: bool,
) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let head_dim = hidden / num_heads;

    // --- MatMul ---
    {
        let a_data: Vec<f32> = (0..seq_len * hidden)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let b_data: Vec<f32> = (0..hidden * hidden)
            .map(|i| (i as f32 * 0.002).cos())
            .collect();

        let median = if hot {
            let mut session = TensorSession::with_device(device.clone());
            bench_fn(
                || {
                    session.reset();
                    let a = session
                        .tensor_with_shape(&a_data, &[seq_len, hidden])
                        .unwrap();
                    let b = session
                        .tensor_with_shape(&b_data, &[hidden, hidden])
                        .unwrap();
                    let _c = session.matmul(&a, &b).unwrap();
                    session.run().unwrap();
                },
                warmup,
                iters,
            )
        } else {
            bench_fn(
                || {
                    let mut session = TensorSession::with_device(device.clone());
                    let a = session
                        .tensor_with_shape(&a_data, &[seq_len, hidden])
                        .unwrap();
                    let b = session
                        .tensor_with_shape(&b_data, &[hidden, hidden])
                        .unwrap();
                    let _c = session.matmul(&a, &b).unwrap();
                    session.run().unwrap();
                },
                warmup,
                iters,
            )
        };

        results.push(BenchResult {
            name: "matmul".into(),
            shape: format!("[{seq_len},{hidden}] x [{hidden},{hidden}]"),
            median_us: median,
        });
    }

    // --- Layer Norm ---
    {
        let x_data: Vec<f32> = (0..seq_len * hidden)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        let median = if hot {
            let mut session = TensorSession::with_device(device.clone());
            bench_fn(
                || {
                    session.reset();
                    let x = session
                        .tensor_with_shape(&x_data, &[seq_len, hidden])
                        .unwrap();
                    let _y = session.layer_norm(&x, hidden).unwrap();
                    session.run().unwrap();
                },
                warmup,
                iters,
            )
        } else {
            bench_fn(
                || {
                    let mut session = TensorSession::with_device(device.clone());
                    let x = session
                        .tensor_with_shape(&x_data, &[seq_len, hidden])
                        .unwrap();
                    let _y = session.layer_norm(&x, hidden).unwrap();
                    session.run().unwrap();
                },
                warmup,
                iters,
            )
        };

        results.push(BenchResult {
            name: "layer_norm".into(),
            shape: format!("[{seq_len},{hidden}]"),
            median_us: median,
        });
    }

    // --- GELU ---
    {
        let x_data: Vec<f32> = (0..seq_len * hidden)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        let median = if hot {
            let mut session = TensorSession::with_device(device.clone());
            bench_fn(
                || {
                    session.reset();
                    let x = session
                        .tensor_with_shape(&x_data, &[seq_len, hidden])
                        .unwrap();
                    let _y = session.gelu(&x).unwrap();
                    session.run().unwrap();
                },
                warmup,
                iters,
            )
        } else {
            bench_fn(
                || {
                    let mut session = TensorSession::with_device(device.clone());
                    let x = session
                        .tensor_with_shape(&x_data, &[seq_len, hidden])
                        .unwrap();
                    let _y = session.gelu(&x).unwrap();
                    session.run().unwrap();
                },
                warmup,
                iters,
            )
        };

        results.push(BenchResult {
            name: "gelu".into(),
            shape: format!("[{seq_len},{hidden}]"),
            median_us: median,
        });
    }

    // --- Softmax ---
    {
        let x_data: Vec<f32> = (0..seq_len * hidden)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        let median = if hot {
            let mut session = TensorSession::with_device(device.clone());
            bench_fn(
                || {
                    session.reset();
                    let x = session
                        .tensor_with_shape(&x_data, &[seq_len, hidden])
                        .unwrap();
                    let _y = session.softmax(&x).unwrap();
                    session.run().unwrap();
                },
                warmup,
                iters,
            )
        } else {
            bench_fn(
                || {
                    let mut session = TensorSession::with_device(device.clone());
                    let x = session
                        .tensor_with_shape(&x_data, &[seq_len, hidden])
                        .unwrap();
                    let _y = session.softmax(&x).unwrap();
                    session.run().unwrap();
                },
                warmup,
                iters,
            )
        };

        results.push(BenchResult {
            name: "softmax".into(),
            shape: format!("[{seq_len},{hidden}]"),
            median_us: median,
        });
    }

    // --- Scaled Dot-Product Attention ---
    {
        let total = num_heads * seq_len * head_dim;
        let q_data: Vec<f32> = (0..total).map(|i| (i as f32 * 0.001).sin()).collect();
        let k_data: Vec<f32> = (0..total).map(|i| (i as f32 * 0.002).cos()).collect();
        let v_data: Vec<f32> = (0..total).map(|i| (i as f32 * 0.003).sin()).collect();

        let dims = AttentionDims {
            batch_size: 1,
            n_heads: num_heads,
            seq_len,
            head_dim,
        };

        let shape_arr = [1, num_heads, seq_len, head_dim];

        let median = if hot {
            let mut session = TensorSession::with_device(device.clone());
            bench_fn(
                || {
                    session.reset();
                    let q = session.tensor_with_shape(&q_data, &shape_arr).unwrap();
                    let k = session.tensor_with_shape(&k_data, &shape_arr).unwrap();
                    let v = session.tensor_with_shape(&v_data, &shape_arr).unwrap();
                    let _out = session.attention(&q, &k, &v, &dims).unwrap();
                    session.run().unwrap();
                },
                warmup,
                iters,
            )
        } else {
            bench_fn(
                || {
                    let mut session = TensorSession::with_device(device.clone());
                    let q = session.tensor_with_shape(&q_data, &shape_arr).unwrap();
                    let k = session.tensor_with_shape(&k_data, &shape_arr).unwrap();
                    let v = session.tensor_with_shape(&v_data, &shape_arr).unwrap();
                    let _out = session.attention(&q, &k, &v, &dims).unwrap();
                    session.run().unwrap();
                },
                warmup,
                iters,
            )
        };

        results.push(BenchResult {
            name: "sdpa".into(),
            shape: format!("B=1, H={num_heads}, S={seq_len}, D={head_dim}"),
            median_us: median,
        });
    }

    results
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    eprintln!("Initializing barraCuda GPU device...");
    let device = Arc::new(WgpuDevice::new().await.context("creating GPU device")?);
    let adapter_info = device.adapter_info();

    let hidden = 768;
    let num_heads = 12;
    let mode = if cli.hot { "hot" } else { "cold" };

    if cli.json {
        print_json(&cli, &device, adapter_info, hidden, num_heads, mode).await?;
    } else {
        print_human(&cli, &device, adapter_info, hidden, num_heads, mode).await?;
    }

    Ok(())
}

async fn print_human(
    cli: &Cli,
    device: &Arc<WgpuDevice>,
    adapter_info: &wgpu::AdapterInfo,
    hidden: usize,
    num_heads: usize,
    mode: &str,
) -> Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("barraCuda/WGSL Benchmark ({mode} dispatch)");
    println!("GPU: {} ({:?})", adapter_info.name, adapter_info.backend);
    if mode == "hot" {
        println!("Mode: HOT — reusing TensorSession (pre-compiled pipelines)");
    } else {
        println!("Mode: COLD — new TensorSession per call (includes pipeline compilation)");
    }
    println!("{}\n", "=".repeat(60));

    if !cli.forward_only {
        let ops = bench_ops(
            device,
            hidden,
            cli.seq_len,
            num_heads,
            cli.warmup,
            cli.iters,
            cli.hot,
        );
        println!("Individual Operations:");
        println!("  {:<20} {:<45} {:>12}", "Operation", "Shape", "Median µs");
        println!("  {} {} {}", "-".repeat(20), "-".repeat(45), "-".repeat(12));
        for r in &ops {
            println!("  {:<20} {:<45} {:>10.1}µs", r.name, r.shape, r.median_us);
        }
    }

    if !cli.ops_only {
        println!("\nForward Pass ({}):", cli.model);
        match bench_forward(cli, device).await {
            Ok(fwd) => {
                println!("  Model: {}", fwd.config_summary);
                println!("  Load time: {:.2}s", fwd.load_time_s);
                println!("  Seq length: {}", fwd.seq_len);
                println!("  Median latency: {:.2}ms", fwd.median_us / 1000.0);
                println!(
                    "  Throughput: {:.0} tokens/s",
                    fwd.seq_len as f64 / (fwd.median_us / 1e6)
                );
            }
            Err(e) => println!("  Error: {e}"),
        }
    }

    println!("\n---");
    println!("Compare with: python3 playGround/bench/pytorch_baseline.py --device cuda");
    println!("Run with --hot to use pre-compiled pipelines (closer to PyTorch dispatch)");

    Ok(())
}

async fn print_json(
    cli: &Cli,
    device: &Arc<WgpuDevice>,
    adapter_info: &wgpu::AdapterInfo,
    hidden: usize,
    num_heads: usize,
    mode: &str,
) -> Result<()> {
    let mut output = serde_json::json!({
        "framework": "barracuda",
        "device": format!("{:?}", adapter_info.backend),
        "gpu_name": adapter_info.name,
        "wgpu_backend": format!("{:?}", adapter_info.backend),
        "dispatch_mode": mode,
    });

    if !cli.forward_only {
        let ops = bench_ops(
            device,
            hidden,
            cli.seq_len,
            num_heads,
            cli.warmup,
            cli.iters,
            cli.hot,
        );
        let ops_json: serde_json::Map<String, serde_json::Value> = ops
            .into_iter()
            .map(|r| {
                (
                    r.name,
                    serde_json::json!({
                        "shape": r.shape,
                        "median_us": r.median_us,
                    }),
                )
            })
            .collect();
        output["ops"] = serde_json::Value::Object(ops_json);
    }

    if !cli.ops_only {
        match bench_forward(cli, device).await {
            Ok(fwd) => {
                output["forward"] = serde_json::json!({
                    "model": cli.model,
                    "config": fwd.config_summary,
                    "load_time_s": fwd.load_time_s,
                    "forward_pass": {
                        "seq_len": fwd.seq_len,
                        "median_us": fwd.median_us,
                        "median_ms": fwd.median_us / 1000.0,
                        "throughput_tokens_per_sec": fwd.seq_len as f64 / (fwd.median_us / 1e6),
                    }
                });
            }
            Err(e) => {
                output["forward"] = serde_json::json!({ "error": format!("{e}") });
            }
        }
    }

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

struct ForwardResult {
    config_summary: String,
    load_time_s: f64,
    seq_len: usize,
    median_us: f64,
}

async fn bench_forward(cli: &Cli, device: &Arc<WgpuDevice>) -> Result<ForwardResult> {
    let hf_token = cli.hf_token.clone().or_else(|| {
        Secrets::load_default()
            .ok()
            .and_then(|s| s.huggingface_token)
    });

    let cache_dir = hf_hub::default_cache_dir();
    let hub = HfHub::new(hf_token.as_deref(), cache_dir)?;

    eprintln!("Downloading {}...", cli.model);
    let files = hub.download_model(&cli.model).await?;
    let config_path = files.config.context("no config.json")?;
    let config = TransformerConfig::from_file(&config_path)?;
    let config_summary = config.to_string();

    eprintln!("Loading weights to GPU...");
    let t0 = Instant::now();
    let raw_weights = weights::load_safetensors(&files.safetensors, device)?;
    let model_weights = weights::organize_weights(raw_weights, &config);
    let load_time_s = t0.elapsed().as_secs_f64();

    let engine = TransformerEngine::new(device.clone(), config, model_weights);
    let token_ids: Vec<u32> = (0..cli.seq_len as u32).map(|i| i % 50257).collect();

    let fwd_warmup = cli.warmup.min(10);
    let fwd_iters = cli.iters.min(50);
    eprintln!("Warmup ({fwd_warmup} iters)...");
    for _ in 0..fwd_warmup {
        let _ = engine.forward(&token_ids);
    }

    eprintln!("Benchmarking ({fwd_iters} iters)...");
    let mut timings: Vec<f64> = Vec::with_capacity(fwd_iters);
    for _ in 0..fwd_iters {
        let t0 = Instant::now();
        let _ = engine.forward(&token_ids);
        timings.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }

    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = timings[timings.len() / 2];

    Ok(ForwardResult {
        config_summary,
        load_time_s,
        seq_len: cli.seq_len,
        median_us,
    })
}
