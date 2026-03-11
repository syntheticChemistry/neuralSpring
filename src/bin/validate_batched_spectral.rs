// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exp 104: `BatchedComputeDispatch` for spectral analysis across composition
//! experiments.
//!
//! Batches IPR + participation number + spectral radius computations across all
//! 5 composition experiments in a single dispatch sequence, measuring throughput
//! vs sequential execution.

#![expect(clippy::expect_used, reason = "binary entry point")]

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::nucleus_pipeline::{execute_composition_pipeline, execute_composition_pipeline_gpu};
use std::time::Instant;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let dispatcher = rt.block_on(Dispatcher::new());

    println!("═══ Exp 104: Batched Spectral Analysis ═══");
    println!("Backend: {}", dispatcher.backend());
    println!("Adapter: {}", dispatcher.adapter_name());
    println!();

    println!("── Sequential (CPU-only pipeline) ──");
    let seq_start = Instant::now();
    let seq_report = execute_composition_pipeline();
    let seq_us = seq_start.elapsed().as_secs_f64() * 1_000_000.0;
    assert!(seq_report.all_passed(), "sequential pipeline should pass");
    println!("  Total: {seq_us:.1}µs ({} stages)", seq_report.total_stages);
    for result in &seq_report.execution.results {
        println!(
            "    {}: {:.1}µs {:?}",
            result.stage_id, result.elapsed_us, result.actual_substrate
        );
    }

    println!();
    println!("── GPU-Dispatch pipeline ──");
    let gpu_start = Instant::now();
    let gpu_report = execute_composition_pipeline_gpu(&dispatcher);
    let gpu_us = gpu_start.elapsed().as_secs_f64() * 1_000_000.0;
    assert!(gpu_report.all_passed(), "GPU pipeline should pass");
    println!(
        "  Total: {gpu_us:.1}µs ({} stages, substrate: {})",
        gpu_report.total_stages, gpu_report.substrate_used
    );
    println!(
        "  GPU stages: {} | CPU stages: {}",
        gpu_report.gpu_stages, gpu_report.cpu_stages
    );
    for result in &gpu_report.execution.results {
        println!(
            "    {}: {:.1}µs {:?}",
            result.stage_id, result.elapsed_us, result.actual_substrate
        );
    }

    println!();
    let speedup = seq_us / gpu_us.max(0.001);
    println!("Speedup: {speedup:.2}× (sequential vs GPU-dispatch)");

    println!();
    println!("── Cross-experiment spectral summary ──");
    let mut spectral_data: Vec<(&str, f64, f64)> = Vec::new();
    for result in &gpu_report.execution.results {
        if let neural_spring_forge::graph::StageOutput::Map(m) = &result.output {
            if let Some(&ipr) = m.get("mean_ipr") {
                let sr = m.get("spectral_radius").copied().unwrap_or(0.0);
                spectral_data.push((&result.stage_id, ipr, sr));
            }
        }
    }
    for (name, ipr, sr) in &spectral_data {
        println!("  {name}: mean_ipr={ipr:.4} spectral_radius={sr:.4}");
    }

    println!();
    println!("✓ Exp 104 complete");
}
