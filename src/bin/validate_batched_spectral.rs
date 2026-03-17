// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exp 104: `BatchedComputeDispatch` for spectral analysis across composition
//! experiments.
//!
//! Batches IPR + participation number + spectral radius computations across all
//! 5 composition experiments in a single dispatch sequence, measuring throughput
//! vs sequential execution.

#![expect(clippy::expect_used, reason = "binary entry point")]

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::nucleus_pipeline::{
    execute_composition_pipeline, execute_composition_pipeline_gpu,
};
use neural_spring::validation::ValidationHarness;
use std::time::Instant;

fn main() {
    let mut h = ValidationHarness::new("batched_spectral");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let dispatcher = rt.block_on(Dispatcher::new());

    println!("═══ Exp 104: Batched Spectral Analysis ═══");
    println!("Backend: {}", dispatcher.backend());
    println!("Adapter: {}", dispatcher.adapter_name());

    println!("\n── Sequential (CPU-only pipeline) ──");
    let seq_start = Instant::now();
    let seq_report = execute_composition_pipeline();
    let seq_us = seq_start.elapsed().as_secs_f64() * 1_000_000.0;
    h.check_bool("sequential pipeline passes", seq_report.all_passed());
    println!(
        "  Total: {seq_us:.1}µs ({} stages)",
        seq_report.total_stages
    );
    for result in &seq_report.execution.results {
        println!(
            "    {}: {:.1}µs {:?}",
            result.stage_id, result.elapsed_us, result.actual_substrate
        );
    }

    println!("\n── GPU-Dispatch pipeline ──");
    let gpu_start = Instant::now();
    let gpu_report = execute_composition_pipeline_gpu(&dispatcher);
    let gpu_us = gpu_start.elapsed().as_secs_f64() * 1_000_000.0;
    h.check_bool("GPU pipeline passes", gpu_report.all_passed());
    println!(
        "  Total: {gpu_us:.1}µs ({} stages, substrate: {})",
        gpu_report.total_stages, gpu_report.substrate_used
    );

    for result in &gpu_report.execution.results {
        h.check_bool(&format!("stage {} passes", result.stage_id), result.success);
        println!(
            "    {}: {:.1}µs {:?}",
            result.stage_id, result.elapsed_us, result.actual_substrate
        );
    }

    let speedup = seq_us / gpu_us.max(0.001);
    println!("\nSpeedup: {speedup:.2}× (sequential vs GPU-dispatch)");

    println!("\n── Cross-experiment spectral summary ──");
    for result in &gpu_report.execution.results {
        if let neural_spring_forge::graph::StageOutput::Map(m) = &result.output {
            if let Some(&ipr) = m.get("mean_ipr") {
                let sr = m.get("spectral_radius").copied().unwrap_or(0.0);
                h.check_bool(&format!("{} IPR > 0", result.stage_id), ipr > 0.0);
                println!(
                    "  {}: mean_ipr={ipr:.4} spectral_radius={sr:.4}",
                    result.stage_id
                );
            }
        }
    }

    h.finish();
}
