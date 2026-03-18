// SPDX-License-Identifier: AGPL-3.0-or-later

//! Kokkos/GPU parity dashboard scenario builder.
//!
//! Visualizes the performance gap between barraCuda (wgpu) and Kokkos-CUDA
//! baselines from groundSpring. For each GPU operation, shows:
//!
//! - **Bar chart**: median dispatch time vs Kokkos baseline
//! - **Gauge**: overhead ratio (barraCuda / Kokkos)
//! - `TimeSeries`: scaling behavior across problem sizes
//!
//! This is the "are we fast enough?" dashboard — the data a PI or GPU
//! engineer would use to decide whether wgpu is viable at production scale.
//!
//! ## Data source
//!
//! Baseline Kokkos numbers are from `groundSpring` benchmark suite.
//! barraCuda numbers come from `src/bin/bench_kokkos_parity.rs` output.
//! Until both are run on the same machine, we use documented reference
//! values from groundSpring V100 handoff.

use crate::visualization::types::{NeuralScenario, ScenarioEdge, ThresholdRange};

use super::{bar, edge, gauge, heatmap, node, scaffold, timeseries};

/// Provenance level for benchmark values, tracking data maturity.
#[derive(Clone, Copy)]
enum ProvenanceLevel {
    /// Estimated from handoff notes — not from matched-hardware runs.
    Estimated,
    /// Measured on matching hardware with documented methodology.
    #[expect(dead_code, reason = "will be used as benchmarks graduate")]
    Measured,
}

struct OpBenchmark {
    name: &'static str,
    pattern: &'static str,
    kokkos_us: f64,
    barracuda_us: f64,
    provenance: ProvenanceLevel,
    problem_size: u32,
}

/// Build the Kokkos parity dashboard scenario.
///
/// Nodes:
/// - `parity_overview`: aggregate gap chart + overhead gauge
/// - `parallel_for_ops`: `parallel_for` pattern operations (element-wise)
/// - `parallel_reduce_ops`: `parallel_reduce` pattern operations (reductions)
/// - `domain_ops`: domain-specific complex operations
#[expect(
    clippy::too_many_lines,
    reason = "scenario builder — single cohesive builder"
)]
#[must_use]
pub fn kokkos_parity_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Kokkos GPU Parity Dashboard",
        "Performance comparison: barraCuda (wgpu) vs Kokkos-CUDA baselines. \
         Isolates dispatch overhead from raw compute — the critical question \
         for wgpu viability at production scale.",
    );

    let benchmarks = reference_benchmarks();

    let estimated_count = benchmarks
        .iter()
        .filter(|b| matches!(b.provenance, ProvenanceLevel::Estimated))
        .count();
    if estimated_count > 0 {
        use std::fmt::Write;
        let _ = write!(
            s.description,
            " [{estimated_count}/{} benchmarks are estimated, not from matched-hardware runs.]",
            benchmarks.len()
        );
    }

    // ── Overview node ────────────────────────────────────────────────────

    let names: Vec<String> = benchmarks.iter().map(|b| b.name.to_string()).collect();

    let overhead_ratios: Vec<f64> = benchmarks
        .iter()
        .map(|b| {
            if b.kokkos_us > 0.0 {
                b.barracuda_us / b.kokkos_us
            } else {
                1.0
            }
        })
        .collect();
    #[expect(
        clippy::cast_precision_loss,
        reason = "benchmark count fits in f64 mantissa"
    )]
    let mean_overhead = overhead_ratios.iter().sum::<f64>() / overhead_ratios.len().max(1) as f64;
    let max_overhead = overhead_ratios
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let gap_labels: Vec<String> = names.iter().map(|n| format!("{n} gap")).collect();
    let gap_values: Vec<f64> = overhead_ratios.iter().map(|r| (r - 1.0) * 100.0).collect();

    let hm_labels = vec![
        "Kokkos".to_string(),
        crate::primal_names::display::BARRACUDA.to_string(),
    ];
    let hm_values: Vec<f64> = benchmarks
        .iter()
        .flat_map(|b| [b.kokkos_us, b.barracuda_us])
        .collect();

    s.ecosystem.primals.push(node(
        "parity_overview",
        "GPU Parity Overview",
        "dashboard",
        0.0,
        0.0,
        &["benchmark.kokkos_parity", "benchmark.gpu_overhead"],
        vec![
            heatmap(
                "timing-comparison",
                "Dispatch Time: Kokkos vs barraCuda (µs)",
                names,
                hm_labels,
                hm_values,
                "µs",
            ),
            bar(
                "overhead-gap",
                "Overhead Above Kokkos Baseline (%)",
                gap_labels,
                gap_values,
                "%",
            ),
            gauge(
                "mean-overhead",
                "Mean Overhead Ratio",
                mean_overhead,
                0.5,
                5.0,
                "×",
                [0.9, 1.5],
                [1.5, 3.0],
            ),
            gauge(
                "max-overhead",
                "Worst-Case Overhead",
                max_overhead,
                0.5,
                10.0,
                "×",
                [1.0, 2.0],
                [2.0, 5.0],
            ),
        ],
        vec![
            ThresholdRange {
                label: "At parity (<1.1×)".into(),
                min: 0.0,
                max: 1.1,
                status: "normal".into(),
            },
            ThresholdRange {
                label: "Acceptable (<2×)".into(),
                min: 1.1,
                max: 2.0,
                status: "warning".into(),
            },
            ThresholdRange {
                label: "Needs work (>2×)".into(),
                min: 2.0,
                max: f64::INFINITY,
                status: "critical".into(),
            },
        ],
    ));

    // ── Parallel-for operations ──────────────────────────────────────────

    let pfor: Vec<&OpBenchmark> = benchmarks
        .iter()
        .filter(|b| b.pattern == "parallel_for")
        .collect();
    let pfor_names: Vec<String> = pfor.iter().map(|b| b.name.to_string()).collect();
    let pfor_kokkos: Vec<f64> = pfor.iter().map(|b| b.kokkos_us).collect();
    let pfor_barracuda: Vec<f64> = pfor.iter().map(|b| b.barracuda_us).collect();

    let base_problem_size = pfor.first().map_or(65536.0, |b| f64::from(b.problem_size));
    let scaling_sizes: Vec<f64> = vec![1024.0, 4096.0, 16384.0, 65536.0, 262_144.0];
    let scaling_times: Vec<f64> = scaling_sizes
        .iter()
        .map(|&sz| {
            let base = pfor.first().map_or(100.0, |b| b.barracuda_us);
            base * (sz / base_problem_size).sqrt()
        })
        .collect();

    s.ecosystem.primals.push(node(
        "parallel_for_ops",
        "Element-wise Operations (parallel_for)",
        "benchmark",
        400.0,
        0.0,
        &["benchmark.parallel_for"],
        vec![
            bar(
                "pfor-kokkos",
                "Kokkos Baseline (µs)",
                pfor_names.clone(),
                pfor_kokkos,
                "µs",
            ),
            bar(
                "pfor-barracuda",
                "barraCuda Dispatch (µs)",
                pfor_names,
                pfor_barracuda,
                "µs",
            ),
            timeseries(
                "pfor-scaling",
                "Dispatch Scaling vs Problem Size",
                "Problem size (elements)",
                "Dispatch time (µs)",
                "µs",
                scaling_sizes,
                scaling_times,
            ),
        ],
        vec![],
    ));

    // ── Parallel-reduce operations ───────────────────────────────────────

    let preduce: Vec<&OpBenchmark> = benchmarks
        .iter()
        .filter(|b| b.pattern == "parallel_reduce")
        .collect();
    let preduce_names: Vec<String> = preduce.iter().map(|b| b.name.to_string()).collect();
    let preduce_overhead: Vec<f64> = preduce
        .iter()
        .map(|b| {
            if b.kokkos_us > 0.0 {
                b.barracuda_us / b.kokkos_us
            } else {
                1.0
            }
        })
        .collect();

    s.ecosystem.primals.push(node(
        "parallel_reduce_ops",
        "Reduction Operations (parallel_reduce)",
        "benchmark",
        200.0,
        300.0,
        &["benchmark.parallel_reduce"],
        vec![bar(
            "preduce-overhead",
            "Overhead Ratio vs Kokkos (×)",
            preduce_names,
            preduce_overhead,
            "× baseline",
        )],
        vec![],
    ));

    // ── Domain operations ────────────────────────────────────────────────

    let domain: Vec<&OpBenchmark> = benchmarks
        .iter()
        .filter(|b| b.pattern == "domain")
        .collect();
    let domain_names: Vec<String> = domain.iter().map(|b| b.name.to_string()).collect();
    let domain_kokkos: Vec<f64> = domain.iter().map(|b| b.kokkos_us).collect();
    let domain_barracuda: Vec<f64> = domain.iter().map(|b| b.barracuda_us).collect();

    s.ecosystem.primals.push(node(
        "domain_ops",
        "Domain-Specific Operations",
        "benchmark",
        400.0,
        300.0,
        &["benchmark.domain_specific"],
        vec![
            bar(
                "domain-kokkos",
                "Kokkos Baseline (µs)",
                domain_names.clone(),
                domain_kokkos,
                "µs",
            ),
            bar(
                "domain-barracuda",
                "barraCuda Dispatch (µs)",
                domain_names,
                domain_barracuda,
                "µs",
            ),
        ],
        vec![],
    ));

    let edges = vec![
        edge(
            "parity_overview",
            "parallel_for_ops",
            "overview → parallel_for detail",
        ),
        edge(
            "parity_overview",
            "parallel_reduce_ops",
            "overview → parallel_reduce detail",
        ),
        edge("parity_overview", "domain_ops", "overview → domain detail"),
    ];

    (s, edges)
}

/// Reference benchmark values from groundSpring V100 handoff.
///
/// Format: `(operation, parallel_pattern, kokkos_µs, barracuda_µs, N)`.
///
/// **Provenance**: `Estimated` — derived from groundSpring V100
/// Kokkos-CUDA handoff notes and barraCuda RTX 4070 Vulkan dispatch
/// measurements. These values are NOT from matched-hardware runs.
///
/// **To graduate to `Measured`**: run `bench_kokkos_parity` on a system
/// with both a Kokkos-CUDA build and a barraCuda wgpu adapter on the
/// same GPU, then update each entry's `provenance` to `Measured`.
///
/// **Source**: groundSpring V100 handoff (Mar 2026), barraCuda RTX 4070.
///
/// **Debt**: Galaxy, BLAST+, GATK, and other industry-standard tool
/// benchmarks not yet included.  See `specs/INDUSTRY_TOOL_GAP_ANALYSIS.md`.
fn reference_benchmarks() -> Vec<OpBenchmark> {
    let p = ProvenanceLevel::Estimated;
    vec![
        OpBenchmark {
            name: "BatchFitness",
            pattern: "parallel_for",
            kokkos_us: 45.0,
            barracuda_us: 82.0,
            problem_size: 65536,
            provenance: p,
        },
        OpBenchmark {
            name: "PairwiseHamming",
            pattern: "parallel_for",
            kokkos_us: 120.0,
            barracuda_us: 195.0,
            problem_size: 65536,
            provenance: p,
        },
        OpBenchmark {
            name: "PairwiseJaccard",
            pattern: "parallel_for",
            kokkos_us: 130.0,
            barracuda_us: 210.0,
            problem_size: 65536,
            provenance: p,
        },
        OpBenchmark {
            name: "PairwiseL2",
            pattern: "parallel_for",
            kokkos_us: 110.0,
            barracuda_us: 180.0,
            problem_size: 65536,
            provenance: p,
        },
        OpBenchmark {
            name: "LocusVariance",
            pattern: "parallel_reduce",
            kokkos_us: 85.0,
            barracuda_us: 150.0,
            problem_size: 65536,
            provenance: p,
        },
        OpBenchmark {
            name: "SpatialPayoff",
            pattern: "parallel_reduce",
            kokkos_us: 95.0,
            barracuda_us: 160.0,
            problem_size: 4096,
            provenance: p,
        },
        OpBenchmark {
            name: "HillGate",
            pattern: "domain",
            kokkos_us: 200.0,
            barracuda_us: 320.0,
            problem_size: 65536,
            provenance: p,
        },
        OpBenchmark {
            name: "SmithWaterman",
            pattern: "domain",
            kokkos_us: 350.0,
            barracuda_us: 520.0,
            problem_size: 1024,
            provenance: p,
        },
        OpBenchmark {
            name: "MultiObjFitness",
            pattern: "parallel_for",
            kokkos_us: 55.0,
            barracuda_us: 95.0,
            problem_size: 65536,
            provenance: p,
        },
    ]
}
