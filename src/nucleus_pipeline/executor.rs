// SPDX-License-Identifier: AGPL-3.0-or-later

//! Node phase: walk a [`PipelineGraph`](neural_spring_forge::graph::PipelineGraph), dispatch stages, record provenance.

use neural_spring_forge::graph::{PipelineExecution, PipelineGraph, StageResult};
use neural_spring_forge::mixed::MixedSubstrate;

use crate::gpu_dispatch::Dispatcher;

use super::dispatch::{dispatch_capability, dispatch_capability_gpu};
use super::error::PipelineError;
use super::report::PipelineReport;

/// Execute the composition pipeline locally (CPU path).
///
/// Runs the 6-stage composition pipeline DAG:
/// `eigensolve` → `digester_anderson` / `isomorphic_reservoir` → ... → `attention_anderson`
///
/// Each stage calls the real neuralSpring module and records timing + outputs.
///
/// # Errors
///
/// Returns [`PipelineError`] if the composition graph is malformed.
pub fn execute_composition_pipeline() -> Result<PipelineReport, PipelineError> {
    let graph = neural_spring_forge::graph::composition_pipeline();
    execute_graph(&graph)
}

/// Execute the composition pipeline with GPU dispatch where substrate permits.
///
/// GPU-capable stages (`GpuOnly`, `GpuPreferred`) route through the `Dispatcher`,
/// which falls back to CPU if no GPU is available. CPU-only stages use direct
/// function calls. Provenance records actual execution substrate per stage.
///
/// # Errors
///
/// Returns [`PipelineError`] if the composition graph is malformed.
pub fn execute_composition_pipeline_gpu(
    dispatcher: &Dispatcher,
) -> Result<PipelineReport, PipelineError> {
    let graph = neural_spring_forge::graph::composition_pipeline();
    execute_graph_gpu(&graph, dispatcher)
}

/// Execute any `PipelineGraph` by resolving capabilities to local functions.
///
/// Tower phase: validate graph and compute topological order.
/// Node phase: dispatch each stage to local computation.
/// Nest phase: record provenance (substrate, timing, outputs).
///
/// # Errors
///
/// Returns [`PipelineError::CyclicGraph`] if topological sort fails, or
/// [`PipelineError::MissingStage`] if a stage ID from the order is absent.
pub fn execute_graph(graph: &PipelineGraph) -> Result<PipelineReport, PipelineError> {
    let order = graph
        .execute_order()
        .ok_or_else(|| PipelineError::CyclicGraph {
            pipeline: graph.name.clone(),
        })?;

    let mut exec = PipelineExecution::new(&graph.name);

    for stage_id in &order {
        let stage = graph
            .stage(stage_id)
            .ok_or_else(|| PipelineError::MissingStage {
                stage_id: stage_id.clone(),
                pipeline: graph.name.clone(),
            })?;

        let start = std::time::Instant::now();
        let (success, output) = dispatch_capability(&stage.capability);
        let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;

        exec.record(StageResult {
            stage_id: stage_id.clone(),
            success,
            elapsed_us,
            actual_substrate: stage.substrate,
            output,
        });
    }

    Ok(PipelineReport {
        pipeline_name: graph.name.clone(),
        substrate_used: "CPU".to_string(),
        total_stages: order.len(),
        gpu_stages: 0,
        cpu_stages: order.len(),
        execution: exec,
    })
}

/// Execute a `PipelineGraph` with GPU dispatch for eligible stages.
///
/// Stages marked `GpuOnly` or `GpuPreferred` are dispatched through the
/// `Dispatcher`, which routes to GPU when available and falls back to CPU.
/// Stages marked `CpuOnly` always use direct function calls.
///
/// Provenance records the actual substrate used per stage:
/// - `GpuOnly`/`GpuPreferred` with GPU available → original `MixedSubstrate` tag
/// - `GpuOnly`/`GpuPreferred` without GPU → `MixedSubstrate::CpuOnly` (fallback)
/// - `CpuOnly` stages → `MixedSubstrate::CpuOnly`
///
/// # Errors
///
/// Returns [`PipelineError::CyclicGraph`] if topological sort fails, or
/// [`PipelineError::MissingStage`] if a stage ID from the order is absent.
pub fn execute_graph_gpu(
    graph: &PipelineGraph,
    dispatcher: &Dispatcher,
) -> Result<PipelineReport, PipelineError> {
    let order = graph
        .execute_order()
        .ok_or_else(|| PipelineError::CyclicGraph {
            pipeline: graph.name.clone(),
        })?;

    let mut exec = PipelineExecution::new(&graph.name);
    let mut gpu_count = 0_usize;
    let mut cpu_count = 0_usize;

    for stage_id in &order {
        let stage = graph
            .stage(stage_id)
            .ok_or_else(|| PipelineError::MissingStage {
                stage_id: stage_id.clone(),
                pipeline: graph.name.clone(),
            })?;

        let use_gpu = matches!(
            stage.substrate,
            MixedSubstrate::GpuOnly | MixedSubstrate::GpuPreferred
        );

        let start = std::time::Instant::now();
        let (success, output, actual_substrate) = if use_gpu && dispatcher.has_gpu() {
            let (s, o) = dispatch_capability_gpu(&stage.capability, dispatcher);
            gpu_count += 1;
            (s, o, stage.substrate)
        } else {
            let (s, o) = dispatch_capability(&stage.capability);
            cpu_count += 1;
            (s, o, MixedSubstrate::CpuOnly)
        };
        let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;

        exec.record(StageResult {
            stage_id: stage_id.clone(),
            success,
            elapsed_us,
            actual_substrate,
            output,
        });
    }

    let substrate_label = if gpu_count > 0 && cpu_count > 0 {
        format!("Mixed (GPU:{gpu_count} CPU:{cpu_count})")
    } else if gpu_count > 0 {
        "GPU".to_string()
    } else {
        "CPU".to_string()
    };

    Ok(PipelineReport {
        pipeline_name: graph.name.clone(),
        substrate_used: substrate_label,
        total_stages: order.len(),
        gpu_stages: gpu_count,
        cpu_stages: cpu_count,
        execution: exec,
    })
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test assertions use expect for clear messages"
)]
mod tests {
    use super::*;
    use neural_spring_forge::graph::{PipelineGraph, StageNode, StageOutput};
    use neural_spring_forge::mixed::MixedSubstrate;

    use crate::gpu_dispatch::Dispatcher;
    use crate::nucleus_pipeline::PipelineError;

    #[test]
    fn composition_pipeline_executes_all_stages() {
        let report = execute_composition_pipeline().expect("composition pipeline");
        assert!(
            report.all_passed(),
            "all stages should pass: {:?}",
            report
                .execution
                .results
                .iter()
                .filter(|r| !r.success)
                .map(|r| &r.stage_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(report.total_stages, 6);
    }

    #[test]
    fn composition_pipeline_respects_topo_order() {
        let report = execute_composition_pipeline().expect("composition pipeline");
        let ids: Vec<&str> = report
            .execution
            .results
            .iter()
            .map(|r| r.stage_id.as_str())
            .collect();

        let pos = |id: &str| -> usize {
            ids.iter()
                .position(|&s| s == id)
                .unwrap_or_else(|| panic!("{id} should be in results"))
        };

        assert!(
            pos("eigensolve") < pos("digester_anderson"),
            "eigensolve before digester"
        );
        assert!(
            pos("eigensolve") < pos("isomorphic_reservoir"),
            "eigensolve before isomorphic"
        );
        assert!(
            pos("eigensolve") < pos("attention_anderson"),
            "eigensolve before attention"
        );
        assert!(
            pos("digester_anderson") < pos("wdm_ensemble_qs"),
            "digester before wdm_qs"
        );
    }

    #[test]
    fn composition_pipeline_records_timing() {
        let report = execute_composition_pipeline().expect("composition pipeline");
        assert!(report.total_us() > 0.0, "pipeline should take > 0µs");
        for result in &report.execution.results {
            assert!(
                result.elapsed_us > 0.0,
                "{} should have timing",
                result.stage_id
            );
        }
    }

    #[test]
    fn composition_pipeline_outputs_are_populated() {
        let report = execute_composition_pipeline().expect("composition pipeline");
        for result in &report.execution.results {
            match &result.output {
                StageOutput::Map(m) => {
                    assert!(!m.is_empty(), "{} map should be non-empty", result.stage_id);
                }
                StageOutput::Vector(v) => {
                    assert!(!v.is_empty(), "{} vec should be non-empty", result.stage_id);
                }
                _ => panic!("{} should produce Map or Vector output", result.stage_id),
            }
        }
    }

    #[test]
    fn substrate_provenance_is_recorded() {
        let report = execute_composition_pipeline().expect("composition pipeline");
        for result in &report.execution.results {
            match result.stage_id.as_str() {
                "eigensolve" | "attention_anderson" => {
                    assert_eq!(
                        result.actual_substrate,
                        MixedSubstrate::GpuOnly,
                        "{} should be GpuOnly",
                        result.stage_id
                    );
                }
                _ => {
                    assert_eq!(
                        result.actual_substrate,
                        MixedSubstrate::CpuOnly,
                        "{} should be CpuOnly",
                        result.stage_id
                    );
                }
            }
        }
    }

    #[test]
    fn gpu_pipeline_cpu_fallback() {
        let dispatcher = Dispatcher::cpu_only();
        let report = execute_composition_pipeline_gpu(&dispatcher).expect("gpu pipeline");
        assert!(report.all_passed(), "all stages pass on CPU fallback");
        assert_eq!(report.total_stages, 6);
        assert_eq!(report.gpu_stages, 0, "CPU-only dispatcher → no GPU stages");
        assert_eq!(report.cpu_stages, 6);
        assert_eq!(report.substrate_used, "CPU");
    }

    #[test]
    fn execute_graph_empty_runs_zero_stages() {
        let graph = PipelineGraph::new("empty");
        let report = execute_graph(&graph).expect("empty graph");
        assert_eq!(report.total_stages, 0);
        assert!(report.total_us().abs() < f64::EPSILON);
        assert!(report.execution.results.is_empty());
        assert!(
            !report.all_passed(),
            "all_passed is false when no stages ran (see PipelineExecution::all_passed)"
        );
        assert_eq!(report.substrate_used, "CPU");
    }

    #[test]
    fn execute_graph_unknown_capability_marks_failure() {
        let mut graph = PipelineGraph::new("bad-cap");
        graph.add_stage(StageNode {
            id: "only".to_string(),
            capability: "science.not_a_real_capability".to_string(),
            substrate: MixedSubstrate::CpuOnly,
            label: "noop".to_string(),
        });
        let report = execute_graph(&graph).expect("single-stage graph");
        assert!(!report.all_passed());
        assert_eq!(report.total_stages, 1);
    }

    #[tokio::test]
    async fn gpu_composition_pipeline_substrate_when_available() {
        let dispatcher = Dispatcher::new().await;
        let report = execute_composition_pipeline_gpu(&dispatcher).expect("gpu pipeline");
        assert!(report.all_passed());
        if dispatcher.has_gpu() {
            assert!(
                report.substrate_used.starts_with("Mixed (GPU:")
                    || report.substrate_used == "GPU"
                    || report.substrate_used == "CPU",
                "unexpected substrate label: {}",
                report.substrate_used
            );
            assert!(
                report.gpu_stages >= 2,
                "expected GpuOnly eigensolve + attention on GPU"
            );
        } else {
            assert_eq!(report.substrate_used, "CPU");
        }
    }

    #[tokio::test]
    async fn gpu_only_single_stage_uses_gpu_label_when_device_present() {
        let Ok(gpu) = crate::gpu::Gpu::new().await else {
            return;
        };
        let dispatcher = Dispatcher::from_gpu(gpu);
        if !dispatcher.has_gpu() {
            return;
        }
        let mut graph = PipelineGraph::new("gpu-only-test");
        graph.add_stage(StageNode {
            id: "eig".to_string(),
            capability: "science.eigensolve".to_string(),
            substrate: MixedSubstrate::GpuOnly,
            label: "eigensolve".to_string(),
        });
        let report = execute_graph_gpu(&graph, &dispatcher).expect("gpu single-stage");
        assert!(report.all_passed());
        assert_eq!(report.substrate_used, "GPU");
        assert_eq!(report.gpu_stages, 1);
        assert_eq!(report.cpu_stages, 0);
    }

    #[test]
    fn cyclic_graph_returns_error() {
        let mut graph = PipelineGraph::new("cycle-test");
        graph.add_stage(StageNode {
            id: "a".to_string(),
            capability: "science.eigensolve".to_string(),
            substrate: MixedSubstrate::CpuOnly,
            label: "a".to_string(),
        });
        graph.add_stage(StageNode {
            id: "b".to_string(),
            capability: "science.eigensolve".to_string(),
            substrate: MixedSubstrate::CpuOnly,
            label: "b".to_string(),
        });
        graph.add_edge("a", "b");
        graph.add_edge("b", "a");
        let err = execute_graph(&graph).expect_err("should detect cycle");
        assert!(
            matches!(err, PipelineError::CyclicGraph { .. }),
            "expected CyclicGraph, got {err:?}"
        );
    }
}
