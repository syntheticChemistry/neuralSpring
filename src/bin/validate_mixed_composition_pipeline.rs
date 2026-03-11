// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exp 106: Mixed-hardware composition pipeline.
//!
//! Runs the full `composition_pipeline()` with GPU eigensolve + CPU bio stages
//! + GPU `attention_anderson`, measuring transfer costs and substrate decisions.
//!
//! This proves the NUCLEUS executor can orchestrate mixed GPU/CPU DAGs.

#![expect(clippy::expect_used, reason = "binary entry point")]
#![expect(clippy::too_many_lines, reason = "mixed pipeline validation binary")]

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::nucleus_pipeline::{
    execute_composition_pipeline, execute_composition_pipeline_gpu,
};
use neural_spring_forge::graph::StageOutput;
use std::time::Instant;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let dispatcher = rt.block_on(Dispatcher::new());

    println!("═══ Exp 106: Mixed-Hardware Composition Pipeline ═══");
    println!("Backend: {}", dispatcher.backend());
    println!("Adapter: {}", dispatcher.adapter_name());
    println!();

    println!("── Phase 1: CPU-only baseline ──");
    let cpu_start = Instant::now();
    let cpu_report = execute_composition_pipeline();
    let cpu_us = cpu_start.elapsed().as_secs_f64() * 1_000_000.0;
    assert!(cpu_report.all_passed());
    println!("  Total: {cpu_us:.1}µs | substrate: {}", cpu_report.substrate_used);
    println!("  GPU stages: {} | CPU stages: {}", cpu_report.gpu_stages, cpu_report.cpu_stages);
    println!();

    println!("── Phase 2: Mixed GPU/CPU pipeline ──");
    let mixed_start = Instant::now();
    let mixed_report = execute_composition_pipeline_gpu(&dispatcher);
    let mixed_us = mixed_start.elapsed().as_secs_f64() * 1_000_000.0;
    assert!(mixed_report.all_passed(), "mixed pipeline should pass");
    println!(
        "  Total: {mixed_us:.1}µs | substrate: {}",
        mixed_report.substrate_used
    );
    println!(
        "  GPU stages: {} | CPU stages: {}",
        mixed_report.gpu_stages, mixed_report.cpu_stages
    );
    println!();

    println!("── Per-stage breakdown ──");
    println!("  {:<25} {:>10} {:>14} {:>8}", "Stage", "Time (µs)", "Substrate", "Status");
    println!("  {}", "─".repeat(60));
    for result in &mixed_report.execution.results {
        println!(
            "  {:<25} {:>10.1} {:>14?} {:>8}",
            result.stage_id,
            result.elapsed_us,
            result.actual_substrate,
            if result.success { "PASS" } else { "FAIL" }
        );
    }
    println!();

    println!("── Transfer cost analysis ──");
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

    println!("  GPU stage time: {gpu_stage_us:.1}µs");
    println!("  CPU stage time: {cpu_stage_us:.1}µs");
    println!("  Overhead/transfer: {overhead_us:.1}µs");
    println!("  GPU fraction: {:.1}%", gpu_stage_us / total_us.max(0.001) * 100.0);
    println!();

    println!("── Output validation ──");
    let mut all_valid = true;
    for result in &mixed_report.execution.results {
        let desc = match &result.output {
            StageOutput::Map(m) => format!("{} keys", m.len()),
            StageOutput::Vector(v) => format!("{} elements", v.len()),
            StageOutput::Scalar(s) => format!("scalar={s:.4}"),
            StageOutput::Empty => "empty".to_string(),
        };
        let ok = result.success;
        if !ok {
            all_valid = false;
        }
        println!("  {}: {} {}", result.stage_id, desc, if ok { "✓" } else { "✗" });
    }
    println!();

    let speedup = cpu_us / mixed_us.max(0.001);
    println!("── Summary ──");
    println!("  CPU-only:  {cpu_us:.1}µs");
    println!("  Mixed:     {mixed_us:.1}µs");
    println!("  Speedup:   {speedup:.2}×");
    println!("  All valid: {all_valid}");
    println!();

    if all_valid {
        println!("✓ Exp 106 complete — mixed-hardware composition pipeline validated");
    } else {
        println!("✗ Some stages failed");
        std::process::exit(1);
    }
}
