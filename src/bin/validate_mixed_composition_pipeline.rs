// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exp 106: Mixed-hardware composition pipeline.
//!
//! Runs the full `composition_pipeline()` with GPU eigensolve + CPU bio stages
//! + GPU `attention_anderson`, measuring transfer costs and substrate decisions.
//!
//! This proves the NUCLEUS executor can orchestrate mixed GPU/CPU DAGs.

#![expect(clippy::expect_used, reason = "binary entry point")]

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::nucleus_pipeline::{
    execute_composition_pipeline, execute_composition_pipeline_gpu,
};
use neural_spring::validation::ValidationHarness;
use neural_spring_forge::graph::StageOutput;
use std::time::Instant;

fn main() {
    let mut h = ValidationHarness::new("mixed_composition_pipeline");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let dispatcher = rt.block_on(Dispatcher::new());

    eprintln!("═══ Exp 106: Mixed-Hardware Composition Pipeline ═══");
    eprintln!("Backend: {}", dispatcher.backend());
    eprintln!("Adapter: {}", dispatcher.adapter_name());

    eprintln!("\n── Phase 1: CPU-only baseline ──");
    let cpu_start = Instant::now();
    let cpu_report = execute_composition_pipeline();
    let cpu_us = cpu_start.elapsed().as_secs_f64() * 1_000_000.0;
    h.check_bool("CPU pipeline passes", cpu_report.all_passed());
    eprintln!(
        "  Total: {cpu_us:.1}µs | substrate: {}",
        cpu_report.substrate_used
    );
    eprintln!(
        "  GPU stages: {} | CPU stages: {}",
        cpu_report.gpu_stages, cpu_report.cpu_stages
    );

    eprintln!("\n── Phase 2: Mixed GPU/CPU pipeline ──");
    let mixed_start = Instant::now();
    let mixed_report = execute_composition_pipeline_gpu(&dispatcher);
    let mixed_us = mixed_start.elapsed().as_secs_f64() * 1_000_000.0;
    h.check_bool("mixed pipeline passes", mixed_report.all_passed());
    eprintln!(
        "  Total: {mixed_us:.1}µs | substrate: {}",
        mixed_report.substrate_used
    );
    eprintln!(
        "  GPU stages: {} | CPU stages: {}",
        mixed_report.gpu_stages, mixed_report.cpu_stages
    );

    eprintln!("\n── Per-stage breakdown ──");
    for result in &mixed_report.execution.results {
        h.check_bool(&format!("stage {} passes", result.stage_id), result.success);
        eprintln!(
            "  {:<25} {:>10.1}µs {:>14?} {}",
            result.stage_id,
            result.elapsed_us,
            result.actual_substrate,
            if result.success { "PASS" } else { "FAIL" }
        );
    }

    eprintln!("\n── Transfer cost analysis ──");
    let gpu_stage_us: f64 = mixed_report
        .execution
        .results
        .iter()
        .filter(|r| {
            matches!(
                r.actual_substrate,
                neural_spring_forge::mixed::MixedSubstrate::GpuOnly
            )
        })
        .map(|r| r.elapsed_us)
        .sum();
    let cpu_stage_us: f64 = mixed_report
        .execution
        .results
        .iter()
        .filter(|r| {
            matches!(
                r.actual_substrate,
                neural_spring_forge::mixed::MixedSubstrate::CpuOnly
            )
        })
        .map(|r| r.elapsed_us)
        .sum();
    let total_us = mixed_report.total_us();
    let overhead_us = total_us - gpu_stage_us - cpu_stage_us;
    eprintln!(
        "  GPU: {gpu_stage_us:.1}µs | CPU: {cpu_stage_us:.1}µs | overhead: {overhead_us:.1}µs"
    );

    eprintln!("\n── Output shapes ──");
    for result in &mixed_report.execution.results {
        let desc = match &result.output {
            StageOutput::Map(m) => format!("{} keys", m.len()),
            StageOutput::Vector(v) => format!("{} elements", v.len()),
            StageOutput::Scalar(s) => format!("scalar={s:.4}"),
            StageOutput::Empty => "empty".to_string(),
        };
        eprintln!("  {}: {desc}", result.stage_id);
    }

    let speedup = cpu_us / mixed_us.max(0.001);
    eprintln!("\n── Summary ──");
    eprintln!("  CPU-only:  {cpu_us:.1}µs");
    eprintln!("  Mixed:     {mixed_us:.1}µs");
    eprintln!("  Speedup:   {speedup:.2}×");

    h.finish();
}
