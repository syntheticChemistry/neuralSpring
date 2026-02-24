// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: taxonomy Naive Bayes FC via `barracuda::ops::bio::TaxonomyFcGpu`.
//!
//! Validates upstream GPU taxonomy classification against CPU reference.
//! Used for wetSpring metagenomics parity.
//!
//! ## Provenance
//!
//! Upstream: `barracuda::ops::bio::taxonomy_fc::TaxonomyFcGpu`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_arguments
)]

use barracuda::ops::bio::taxonomy_fc::TaxonomyFcGpu;
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

fn cpu_taxonomy_fc(
    log_probs: &[f64],
    log_priors: &[f64],
    features: &[u32],
    n_queries: usize,
    n_taxa: usize,
    n_features: usize,
) -> Vec<f64> {
    let mut scores = vec![0.0_f64; n_queries * n_taxa];
    for q in 0..n_queries {
        for t in 0..n_taxa {
            let mut s = log_priors[t];
            for f in 0..n_features {
                if features[q * n_features + f] == 1 {
                    s += log_probs[t * n_features + f];
                }
            }
            scores[q * n_taxa + t] = s;
        }
    }
    scores
}

fn create_f64_buffer(device: &wgpu::Device, data: &[f64], label: &str) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn gpu_taxonomy_fc(
    gpu: &Gpu,
    op: &TaxonomyFcGpu,
    log_probs: &[f64],
    log_priors: &[f64],
    features: &[u32],
    n_queries: u32,
    n_taxa: u32,
    n_features: u32,
) -> Result<Vec<f64>, String> {
    let device = gpu.device();

    let log_probs_buf = create_f64_buffer(device, log_probs, "log_probs");
    let log_priors_buf = create_f64_buffer(device, log_priors, "log_priors");
    let features_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("features"),
        contents: bytemuck::cast_slice(features),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let scores_size = (n_queries as usize) * (n_taxa as usize);
    let scores_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scores"),
        size: (scores_size * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    op.dispatch(
        &log_probs_buf,
        &log_priors_buf,
        &features_buf,
        &scores_buf,
        n_queries,
        n_taxa,
        n_features,
    );

    gpu.read_buffer_f64(&scores_buf, scores_size)
}

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    if !gpu.capabilities.supports_f64 {
        eprintln!("  SKIP: f64 shader support required for TaxonomyFcGpu");
        eprintln!("  0/0 checks — skipping gracefully");
        std::process::exit(0);
    }

    let device = gpu.wgpu_device().clone();
    let op = TaxonomyFcGpu::new(device);

    let mut h = ValidationHarness::new("upstream_taxonomy");

    validate_simple_classify(&mut h, &gpu, &op);
    validate_all_features_present(&mut h, &gpu, &op);
    validate_determinism(&mut h, &gpu, &op);

    h.finish();
}

fn validate_simple_classify(h: &mut ValidationHarness, gpu: &Gpu, op: &TaxonomyFcGpu) {
    let n_queries = 3_usize;
    let n_taxa = 2_usize;
    let n_features = 4_usize;

    // log_probs [n_taxa × n_features]
    let log_probs: Vec<f64> = vec![
        -0.5, -0.3, -0.2, -0.8, // taxon 0
        -0.4, -0.6, -0.1, -0.7, // taxon 1
    ];
    let log_priors: Vec<f64> = vec![-0.7, -0.5];
    // features [n_queries × n_features] binary
    let features: Vec<u32> = vec![
        1, 0, 1, 0, // query 0: features 0,2 present
        0, 1, 0, 1, // query 1: features 1,3 present
        1, 1, 1, 1, // query 2: all present
    ];

    let cpu_scores = cpu_taxonomy_fc(
        &log_probs,
        &log_priors,
        &features,
        n_queries,
        n_taxa,
        n_features,
    );

    match gpu_taxonomy_fc(
        gpu,
        op,
        &log_probs,
        &log_priors,
        &features,
        n_queries as u32,
        n_taxa as u32,
        n_features as u32,
    ) {
        Ok(gpu_scores) => {
            let max_err = cpu_scores
                .iter()
                .zip(gpu_scores.iter())
                .map(|(&c, &g)| (c - g).abs())
                .fold(0.0_f64, f64::max);
            h.check_abs(
                "simple classify: max |GPU - CPU| < 1e-10",
                max_err,
                0.0,
                tolerances::GPU_F64_EXACT,
            );
        }
        Err(e) => {
            h.check_bool(&format!("simple classify: dispatch failed — {e}"), false);
        }
    }
}

fn validate_all_features_present(h: &mut ValidationHarness, gpu: &Gpu, op: &TaxonomyFcGpu) {
    let n_queries = 2_usize;
    let n_taxa = 3_usize;
    let n_features = 4_usize;

    let log_probs: Vec<f64> = (0..n_taxa * n_features)
        .map(|i| -0.1 * (i as f64 + 1.0))
        .collect();
    let log_priors: Vec<f64> = vec![-0.5, -0.3, -0.7];
    let features: Vec<u32> = vec![1; n_queries * n_features];

    let cpu_scores = cpu_taxonomy_fc(
        &log_probs,
        &log_priors,
        &features,
        n_queries,
        n_taxa,
        n_features,
    );

    match gpu_taxonomy_fc(
        gpu,
        op,
        &log_probs,
        &log_priors,
        &features,
        n_queries as u32,
        n_taxa as u32,
        n_features as u32,
    ) {
        Ok(gpu_scores) => {
            let max_err = cpu_scores
                .iter()
                .zip(gpu_scores.iter())
                .map(|(&c, &g)| (c - g).abs())
                .fold(0.0_f64, f64::max);
            h.check_abs(
                "all features present: score = log_prior + sum(log_probs)",
                max_err,
                0.0,
                tolerances::GPU_F64_EXACT,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("all features present: dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu, op: &TaxonomyFcGpu) {
    let n_queries = 4_usize;
    let n_taxa = 3_usize;
    let n_features = 5_usize;

    let log_probs: Vec<f64> = (0..n_taxa * n_features)
        .map(|i| -0.2 * (i as f64))
        .collect();
    let log_priors: Vec<f64> = vec![-0.4, -0.6, -0.8];
    let features: Vec<u32> = vec![1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0];

    let r1 = gpu_taxonomy_fc(
        gpu,
        op,
        &log_probs,
        &log_priors,
        &features,
        n_queries as u32,
        n_taxa as u32,
        n_features as u32,
    );
    let r2 = gpu_taxonomy_fc(
        gpu,
        op,
        &log_probs,
        &log_priors,
        &features,
        n_queries as u32,
        n_taxa as u32,
        n_features as u32,
    );

    match (r1, r2) {
        (Ok(s1), Ok(s2)) => {
            let identical = s1
                .iter()
                .zip(s2.iter())
                .all(|(a, b)| (a - b).abs() < tolerances::ZERO_DETECTION);
            h.check_bool("determinism: two runs identical", identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}
