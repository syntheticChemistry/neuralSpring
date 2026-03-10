// SPDX-License-Identifier: AGPL-3.0-or-later

//! Industry tool coverage scenario builder.
//!
//! Generates a comprehensive view of which scientific tools the ecosystem
//! has reached parity with, what's in progress, and what's missing — the
//! "big picture" dashboard for ecosystem evolution planning.
//!
//! Data is sourced from `specs/INDUSTRY_TOOL_GAP_ANALYSIS.md` and reflects
//! actual implementation status across neuralSpring, barraCuda, and other
//! Springs.

use crate::visualization::types::{NeuralScenario, ScenarioEdge, ThresholdRange};

use super::{bar, edge, gauge, heatmap, node, scaffold};

struct ToolEntry {
    name: &'static str,
    domain: &'static str,
    status: Status,
    owner: &'static str,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Done,
    InProgress,
    Scoped,
    Missing,
    Deferred,
}

impl Status {
    const fn as_f64(self) -> f64 {
        match self {
            Self::Done => 1.0,
            Self::InProgress => 0.7,
            Self::Scoped => 0.4,
            Self::Missing => 0.0,
            Self::Deferred => -0.2,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Done => "Done",
            Self::InProgress => "In progress",
            Self::Scoped => "Scoped",
            Self::Missing => "Missing",
            Self::Deferred => "Deferred",
        }
    }
}

/// Build the industry coverage scenario.
///
/// Nodes:
/// - `coverage_overview`: heatmap of domain × tool status
/// - `domain_progress`: per-domain completion percentage bars
/// - `implementation_detail`: what's done, in progress, missing
#[expect(
    clippy::too_many_lines,
    reason = "scenario builder — single cohesive builder"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "tool/domain counts fit in f64 mantissa"
)]
#[must_use]
pub fn industry_coverage_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Industry Tool Coverage",
        "Ecosystem parity with scientific tools: what's done, what's being built, \
         and what's missing — the roadmap dashboard for closing the gap",
    );

    let tools = tool_inventory();
    let domains = unique_domains(&tools);

    // ── Coverage heatmap ─────────────────────────────────────────────────

    let tool_names: Vec<String> = tools.iter().map(|t| t.name.to_string()).collect();
    let domain_labels = vec!["Status".to_string()];
    let status_values: Vec<f64> = tools.iter().map(|t| t.status.as_f64()).collect();

    s.ecosystem.primals.push(node(
        "coverage_overview",
        "Tool Coverage Heatmap",
        "dashboard",
        0.0,
        0.0,
        &["industry.coverage", "industry.gap_analysis"],
        vec![
            heatmap(
                "tool-status-heatmap",
                "Tool Parity Status (1.0=Done, 0.7=Progress, 0.4=Scoped, 0=Missing)",
                tool_names,
                domain_labels,
                status_values,
                "readiness",
            ),
            gauge(
                "overall-coverage",
                "Overall Coverage",
                compute_overall_coverage(&tools),
                0.0,
                100.0,
                "%",
                [60.0, 90.0],
                [30.0, 60.0],
            ),
        ],
        vec![
            ThresholdRange {
                label: "Complete (>80%)".into(),
                min: 80.0,
                max: 100.0,
                status: "normal".into(),
            },
            ThresholdRange {
                label: "Progressing (40-80%)".into(),
                min: 40.0,
                max: 80.0,
                status: "warning".into(),
            },
            ThresholdRange {
                label: "Early (<40%)".into(),
                min: 0.0,
                max: 40.0,
                status: "critical".into(),
            },
        ],
    ));

    // ── Per-domain progress ──────────────────────────────────────────────

    let mut domain_names = Vec::new();
    let mut domain_pcts = Vec::new();
    for domain in &domains {
        let domain_tools: Vec<&ToolEntry> = tools.iter().filter(|t| t.domain == *domain).collect();
        let done = domain_tools
            .iter()
            .filter(|t| t.status == Status::Done)
            .count();
        let total = domain_tools.len();
        let pct = if total > 0 {
            (done as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        domain_names.push((*domain).to_string());
        domain_pcts.push(pct);
    }

    s.ecosystem.primals.push(node(
        "domain_progress",
        "Per-Domain Completion",
        "analysis",
        350.0,
        0.0,
        &["industry.domain_progress"],
        vec![bar(
            "domain-completion",
            "Completion by Domain (%)",
            domain_names,
            domain_pcts,
            "%",
        )],
        vec![],
    ));

    // ── Status breakdown ─────────────────────────────────────────────────

    let status_categories = [
        Status::Done,
        Status::InProgress,
        Status::Scoped,
        Status::Missing,
        Status::Deferred,
    ];
    let status_names: Vec<String> = status_categories.iter().map(|s| s.label().into()).collect();
    let status_counts: Vec<f64> = status_categories
        .iter()
        .map(|target| tools.iter().filter(|t| t.status == *target).count() as f64)
        .collect();

    let owner_names: Vec<String> = unique_owners(&tools)
        .into_iter()
        .map(String::from)
        .collect();
    let owner_counts: Vec<f64> = owner_names
        .iter()
        .map(|owner| tools.iter().filter(|t| t.owner == owner.as_str()).count() as f64)
        .collect();

    s.ecosystem.primals.push(node(
        "implementation_detail",
        "Implementation Status Breakdown",
        "analysis",
        175.0,
        300.0,
        &["industry.status_breakdown"],
        vec![
            bar(
                "status-breakdown",
                "Tools by Status",
                status_names,
                status_counts,
                "tools",
            ),
            bar(
                "owner-distribution",
                "Tool Ownership by Primal",
                owner_names,
                owner_counts,
                "tools",
            ),
        ],
        vec![],
    ));

    let edges = vec![
        edge(
            "coverage_overview",
            "domain_progress",
            "overview → domain drill-down",
        ),
        edge(
            "coverage_overview",
            "implementation_detail",
            "overview → status breakdown",
        ),
    ];

    (s, edges)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "tool counts fit in f64 mantissa"
)]
fn compute_overall_coverage(tools: &[ToolEntry]) -> f64 {
    let actionable: Vec<&ToolEntry> = tools
        .iter()
        .filter(|t| t.status != Status::Deferred)
        .collect();
    if actionable.is_empty() {
        return 0.0;
    }
    let done = actionable
        .iter()
        .filter(|t| t.status == Status::Done)
        .count();
    (done as f64 / actionable.len() as f64) * 100.0
}

fn unique_domains(tools: &[ToolEntry]) -> Vec<&'static str> {
    let mut seen = Vec::new();
    for t in tools {
        if !seen.contains(&t.domain) {
            seen.push(t.domain);
        }
    }
    seen
}

fn unique_owners(tools: &[ToolEntry]) -> Vec<&'static str> {
    let mut seen = Vec::new();
    for t in tools {
        if !seen.contains(&t.owner) {
            seen.push(t.owner);
        }
    }
    seen
}

/// Current ecosystem tool inventory from `specs/INDUSTRY_TOOL_GAP_ANALYSIS.md`.
fn tool_inventory() -> Vec<ToolEntry> {
    vec![
        // ── Streaming parsers (neuralSpring) ─────────────────────────────
        ToolEntry {
            name: "FASTQ parser",
            domain: "I/O",
            status: Status::Done,
            owner: "neuralSpring",
        },
        ToolEntry {
            name: "FASTA parser",
            domain: "I/O",
            status: Status::Done,
            owner: "neuralSpring",
        },
        ToolEntry {
            name: "VCF parser",
            domain: "I/O",
            status: Status::Done,
            owner: "neuralSpring",
        },
        ToolEntry {
            name: "mzML parser",
            domain: "I/O",
            status: Status::Deferred,
            owner: "wetSpring",
        },
        ToolEntry {
            name: "SAM/BAM parser",
            domain: "I/O",
            status: Status::Missing,
            owner: "neuralSpring",
        },
        // ── Sequence search / alignment ──────────────────────────────────
        ToolEntry {
            name: "BLAST-like search (CPU)",
            domain: "Alignment",
            status: Status::Done,
            owner: "neuralSpring",
        },
        ToolEntry {
            name: "BLAST-like search (GPU)",
            domain: "Alignment",
            status: Status::Scoped,
            owner: "barraCuda",
        },
        ToolEntry {
            name: "Smith-Waterman GPU",
            domain: "Alignment",
            status: Status::Done,
            owner: "barraCuda",
        },
        ToolEntry {
            name: "BLOSUM62 substitution",
            domain: "Alignment",
            status: Status::Scoped,
            owner: "barraCuda",
        },
        // ── MSA / HMM ───────────────────────────────────────────────────
        ToolEntry {
            name: "HMM forward/Viterbi",
            domain: "HMM/MSA",
            status: Status::Done,
            owner: "neuralSpring",
        },
        ToolEntry {
            name: "HMM GPU (f64)",
            domain: "HMM/MSA",
            status: Status::Done,
            owner: "barraCuda",
        },
        ToolEntry {
            name: "JackHMMER equiv",
            domain: "HMM/MSA",
            status: Status::Scoped,
            owner: "neuralSpring",
        },
        ToolEntry {
            name: "MMseqs2 clustering",
            domain: "HMM/MSA",
            status: Status::Scoped,
            owner: "neuralSpring",
        },
        // ── GPU primitives (barraCuda) ───────────────────────────────────
        ToolEntry {
            name: "Tensor operations",
            domain: "Linear Algebra",
            status: Status::Done,
            owner: "barraCuda",
        },
        ToolEntry {
            name: "Pairwise distances",
            domain: "Linear Algebra",
            status: Status::Done,
            owner: "barraCuda",
        },
        ToolEntry {
            name: "BatchFitness",
            domain: "Evolutionary",
            status: Status::Done,
            owner: "barraCuda",
        },
        ToolEntry {
            name: "HillGate activation",
            domain: "Evolutionary",
            status: Status::Done,
            owner: "barraCuda",
        },
        ToolEntry {
            name: "MultiObj fitness",
            domain: "Evolutionary",
            status: Status::Done,
            owner: "barraCuda",
        },
        // ── Protein structure ────────────────────────────────────────────
        ToolEntry {
            name: "Protein folding (primitives)",
            domain: "Structure",
            status: Status::Done,
            owner: "neuralSpring",
        },
        ToolEntry {
            name: "AlphaFold MSA pipeline",
            domain: "Structure",
            status: Status::Scoped,
            owner: "neuralSpring",
        },
        // ── Pipeline / compute ───────────────────────────────────────────
        ToolEntry {
            name: "Pipeline DAG (pipeline_graph)",
            domain: "Compute",
            status: Status::Done,
            owner: "toadStool",
        },
        ToolEntry {
            name: "GPU capability discovery",
            domain: "Compute",
            status: Status::Done,
            owner: "toadStool",
        },
        ToolEntry {
            name: "Sovereign WGSL compilation",
            domain: "Compute",
            status: Status::Done,
            owner: "coralReef",
        },
        // ── Benchmarking ─────────────────────────────────────────────────
        ToolEntry {
            name: "Kokkos parity harness",
            domain: "Benchmarks",
            status: Status::Done,
            owner: "neuralSpring",
        },
        ToolEntry {
            name: "Python CPU baselines",
            domain: "Benchmarks",
            status: Status::Done,
            owner: "neuralSpring",
        },
        ToolEntry {
            name: "Kokkos-CUDA baselines",
            domain: "Benchmarks",
            status: Status::InProgress,
            owner: "groundSpring",
        },
        // ── Chromatography (wetSpring) ───────────────────────────────────
        ToolEntry {
            name: "Chromeleon equiv",
            domain: "Chromatography",
            status: Status::Deferred,
            owner: "wetSpring",
        },
        ToolEntry {
            name: "DADA2 equiv",
            domain: "Metagenomics",
            status: Status::Missing,
            owner: "wetSpring",
        },
        ToolEntry {
            name: "QIIME2 pipeline",
            domain: "Metagenomics",
            status: Status::Missing,
            owner: "wetSpring",
        },
    ]
}
