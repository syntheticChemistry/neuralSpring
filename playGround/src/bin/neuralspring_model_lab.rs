// SPDX-License-Identifier: AGPL-3.0-or-later

//! Model Lab: download HuggingFace models and run inference through
//! barraCuda's WGSL shader pipeline.
//!
//! This binary demonstrates the ecoPrimals approach to model inference:
//! take standard HuggingFace weights (safetensors), load them onto GPU
//! via barraCuda, and run the forward pass through sovereign WGSL shaders
//! instead of PyTorch/CUDA.

#![expect(clippy::pedantic, reason = "playground binary — iterating rapidly")]

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use neuralspring_playground::hf_hub::{self, HfHub};
use neuralspring_playground::inference::transformer::TransformerEngine;
use neuralspring_playground::inference::weights;
use neuralspring_playground::model_config::TransformerConfig;
use neuralspring_playground::secrets::Secrets;

#[derive(Parser)]
#[command(
    name = "neuralspring-model-lab",
    about = "Download HuggingFace models and run inference through barraCuda WGSL shaders"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Override HuggingFace API token
    #[arg(long, global = true)]
    hf_token: Option<String>,

    /// Override model cache directory
    #[arg(long, global = true)]
    cache_dir: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Show info about a HuggingFace model
    Info {
        /// Model ID (e.g., "openai-community/gpt2", "meta-llama/Llama-2-7b-hf")
        model_id: String,
    },

    /// Download a model's weights and config
    Download {
        /// Model ID
        model_id: String,
    },

    /// Inspect safetensors files: list tensor names, shapes, dtypes
    Inspect {
        /// Model ID (downloads if not cached) or path to safetensors file
        source: String,
    },

    /// Load model weights onto GPU and show summary
    Load {
        /// Model ID (downloads if not cached)
        model_id: String,
    },

    /// Run a forward pass with token IDs
    Forward {
        /// Model ID
        model_id: String,

        /// Token IDs (comma-separated)
        #[arg(long)]
        tokens: String,

        /// Show top-k predictions
        #[arg(long, default_value = "10")]
        top_k: usize,
    },

    /// List cached models
    Cached,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    let hf_token = cli.hf_token.or_else(|| {
        Secrets::load_default()
            .ok()
            .and_then(|s| s.huggingface_token)
    });

    let cache_dir = cli.cache_dir.unwrap_or_else(hf_hub::default_cache_dir);

    let hub = HfHub::new(hf_token.as_deref(), cache_dir.clone())?;

    match cli.command {
        Commands::Info { model_id } => cmd_info(&hub, &model_id).await,
        Commands::Download { model_id } => cmd_download(&hub, &model_id).await,
        Commands::Inspect { source } => cmd_inspect(&hub, &source).await,
        Commands::Load { model_id } => cmd_load(&hub, &model_id).await,
        Commands::Forward {
            model_id,
            tokens,
            top_k,
        } => cmd_forward(&hub, &model_id, &tokens, top_k).await,
        Commands::Cached => cmd_cached(&cache_dir),
    }
}

async fn cmd_info(hub: &HfHub, model_id: &str) -> Result<()> {
    println!("Fetching model info for {model_id}...");
    let info = hub.model_info(model_id).await?;

    println!("Model: {}", info.model_id);
    println!("SHA: {}", info.sha);
    if let Some(ref tag) = info.pipeline_tag {
        println!("Pipeline: {tag}");
    }
    if let Some(ref lib) = info.library_name {
        println!("Library: {lib}");
    }

    let safetensors: Vec<_> = info
        .siblings
        .iter()
        .filter(|s| s.filename.ends_with(".safetensors"))
        .collect();
    println!("\nSafetensors files ({}):", safetensors.len());
    for f in &safetensors {
        println!("  {}", f.filename);
    }

    let has_config = info.siblings.iter().any(|s| s.filename == "config.json");
    let has_tokenizer = info.siblings.iter().any(|s| s.filename == "tokenizer.json");
    println!("\nconfig.json: {}", if has_config { "yes" } else { "no" });
    println!(
        "tokenizer.json: {}",
        if has_tokenizer { "yes" } else { "no" }
    );

    Ok(())
}

async fn cmd_download(hub: &HfHub, model_id: &str) -> Result<()> {
    println!("Downloading {model_id}...\n");
    let files = hub.download_model(model_id).await?;

    if let Some(ref p) = files.config {
        println!("  config.json: {}", p.display());
    }
    for p in &files.safetensors {
        println!("  safetensors: {}", p.display());
    }
    if let Some(ref p) = files.tokenizer {
        println!("  tokenizer: {}", p.display());
    }

    if let Some(ref config_path) = files.config {
        let config = TransformerConfig::from_file(config_path)?;
        println!("\n{config}");
        println!(
            "  Estimated memory (f32): {:.1} MB",
            config.estimated_memory_f32() as f64 / 1e6
        );
    }

    println!("\nDownload complete. Ready for 'load' or 'forward'.");
    Ok(())
}

async fn cmd_inspect(hub: &HfHub, source: &str) -> Result<()> {
    let paths: Vec<PathBuf> = if source.contains('/') && !std::path::Path::new(source).exists() {
        // Looks like a model ID — download first
        println!("Downloading {source} for inspection...\n");
        let files = hub.download_model(source).await?;
        files.safetensors
    } else {
        vec![PathBuf::from(source)]
    };

    let entries = weights::inspect_safetensors(&paths)?;
    println!("Tensors ({}):", entries.len());
    println!("{:<60} {:>20} {:>8}", "Name", "Shape", "Dtype");
    println!("{}", "-".repeat(90));
    for (name, shape, dtype) in &entries {
        let shape_str = format!("{shape:?}");
        println!("{name:<60} {shape_str:>20} {dtype:>8}");
    }
    Ok(())
}

async fn cmd_load(hub: &HfHub, model_id: &str) -> Result<()> {
    let files = hub.download_model(model_id).await?;

    let config_path = files.config.context("no config.json found")?;
    let config = TransformerConfig::from_file(&config_path)?;
    println!("{config}\n");

    println!("Initializing GPU device...");
    let device = Arc::new(
        barracuda::prelude::WgpuDevice::new()
            .await
            .context("creating GPU device")?,
    );
    println!("GPU: {:?}", device.adapter_info());

    println!("Loading weights to GPU...");
    let raw_weights = weights::load_safetensors(&files.safetensors, &device)?;
    println!("Loaded {} tensors to GPU", raw_weights.len());

    let model_weights = weights::organize_weights(raw_weights, &config);
    weights::print_weight_summary(&model_weights, &config);

    Ok(())
}

async fn cmd_forward(hub: &HfHub, model_id: &str, tokens_str: &str, top_k: usize) -> Result<()> {
    let token_ids: Vec<u32> = tokens_str
        .split(',')
        .map(|s| s.trim().parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .context("parsing token IDs (comma-separated integers)")?;

    println!("Input tokens: {token_ids:?} (len={})", token_ids.len());

    let files = hub.download_model(model_id).await?;
    let config_path = files.config.context("no config.json found")?;
    let config = TransformerConfig::from_file(&config_path)?;
    println!("{config}\n");

    println!("Initializing GPU device...");
    let device = Arc::new(
        barracuda::prelude::WgpuDevice::new()
            .await
            .context("creating GPU device")?,
    );
    println!("GPU: {:?}\n", device.adapter_info());

    println!("Loading weights...");
    let raw_weights = weights::load_safetensors(&files.safetensors, &device)?;
    let model_weights = weights::organize_weights(raw_weights, &config);
    weights::print_weight_summary(&model_weights, &config);

    println!("\nRunning forward pass through barraCuda shaders...");
    let engine = TransformerEngine::new(device, config, model_weights);
    let output = engine.forward(&token_ids)?;

    println!("Output logits: {} values", output.logits.len());
    println!("\nTop-{top_k} predictions:");
    let top = TransformerEngine::top_k(&output.logits, top_k);
    let probs = TransformerEngine::softmax(&output.logits);
    for (rank, (token_id, logit)) in top.iter().enumerate() {
        let prob = probs[*token_id];
        println!(
            "  #{}: token={token_id:<6} logit={logit:>8.3} prob={prob:.4}",
            rank + 1
        );
    }

    Ok(())
}

fn cmd_cached(cache_dir: &std::path::Path) -> Result<()> {
    if !cache_dir.exists() {
        println!("No cached models (cache dir does not exist).");
        return Ok(());
    }

    println!("Cached models in {}:", cache_dir.display());
    let mut found = false;
    for entry in std::fs::read_dir(cache_dir)?.flatten() {
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            let name = entry.file_name();
            let model_dir = name.to_string_lossy().replace("--", "/");
            let safetensor_count = std::fs::read_dir(entry.path())
                .ok()
                .map(|entries| {
                    entries
                        .flatten()
                        .filter(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
                        .count()
                })
                .unwrap_or(0);

            let has_config = entry.path().join("config.json").exists();
            println!(
                "  {model_dir} ({safetensor_count} safetensors, config={})",
                if has_config { "yes" } else { "no" }
            );
            found = true;
        }
    }

    if !found {
        println!("  (empty)");
    }

    Ok(())
}
