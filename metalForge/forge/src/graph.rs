// SPDX-License-Identifier: AGPL-3.0-or-later

//! biomeOS pipeline graph — DAG-based stage coordination.
//!
//! Models multi-stage compute pipelines as directed acyclic graphs where:
//! - **Nodes** are capability-addressed stages (resolved via biomeOS)
//! - **Edges** are data-flow dependencies between stages
//!
//! ## Design
//!
//! Graphs are defined declaratively (struct literals or config) and executed
//! in topological order. Each stage specifies a `capability` string that
//! biomeOS resolves to a primal at runtime, enabling dynamic routing.
//!
//! ## Absorption
//!
//! `ToadStool` can absorb this into `barracuda::pipeline::graph` for
//! ecosystem-wide DAG-based dispatch.

use std::collections::HashMap;

use super::mixed::MixedSubstrate;

/// A single stage in a pipeline graph.
#[derive(Debug, Clone)]
pub struct StageNode {
    /// Unique identifier for this stage (e.g. `"eigensolve"`, `"ipr"`).
    pub id: String,
    /// biomeOS capability string (e.g. `"science.eigensolve"`).
    pub capability: String,
    /// Preferred execution substrate.
    pub substrate: MixedSubstrate,
    /// Human-readable label for visualization.
    pub label: String,
}

/// Result of executing a single stage.
#[derive(Debug, Clone)]
pub struct StageResult {
    /// Stage identifier.
    pub stage_id: String,
    /// Whether the stage completed successfully.
    pub success: bool,
    /// Execution time in microseconds.
    pub elapsed_us: f64,
    /// Which substrate was actually used.
    pub actual_substrate: MixedSubstrate,
    /// Opaque output data (JSON-serializable in practice).
    pub output: StageOutput,
}

/// Opaque output from a completed stage.
#[derive(Debug, Clone)]
pub enum StageOutput {
    /// Scalar result (e.g. entropy value).
    Scalar(f64),
    /// Vector result (e.g. eigenvalues).
    Vector(Vec<f64>),
    /// Named map of values.
    Map(HashMap<String, f64>),
    /// No output (side-effect only).
    Empty,
}

/// Directed acyclic graph of pipeline stages.
///
/// Stages are stored by ID, edges encode data-flow dependencies.
/// [`PipelineGraph::execute_order`] returns a topological sort.
#[derive(Debug, Clone)]
pub struct PipelineGraph {
    /// Pipeline name for logging and provenance.
    pub name: String,
    stages: Vec<StageNode>,
    /// Edges: `(from_id, to_id)` — `from` must complete before `to`.
    edges: Vec<(String, String)>,
}

impl PipelineGraph {
    /// Create a new empty pipeline graph.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            stages: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a stage to the graph.
    pub fn add_stage(&mut self, stage: StageNode) {
        self.stages.push(stage);
    }

    /// Add a dependency edge: `from` must complete before `to`.
    pub fn add_edge(&mut self, from: &str, to: &str) {
        self.edges.push((from.to_string(), to.to_string()));
    }

    /// Number of stages.
    #[must_use]
    pub const fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Number of dependency edges.
    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get a stage by ID.
    #[must_use]
    pub fn stage(&self, id: &str) -> Option<&StageNode> {
        self.stages.iter().find(|s| s.id == id)
    }

    /// Compute topological execution order via Kahn's algorithm.
    ///
    /// Returns stage IDs in a valid execution order, or `None` if the
    /// graph contains a cycle (which violates the DAG invariant).
    #[must_use]
    pub fn execute_order(&self) -> Option<Vec<String>> {
        let ids: Vec<&str> = self.stages.iter().map(|s| s.id.as_str()).collect();
        let mut in_degree: HashMap<&str, usize> = ids.iter().map(|id| (*id, 0)).collect();
        let mut adjacency: HashMap<&str, Vec<&str>> =
            ids.iter().map(|id| (*id, Vec::new())).collect();

        for (from, to) in &self.edges {
            if let Some(neighbors) = adjacency.get_mut(from.as_str()) {
                neighbors.push(to.as_str());
            }
            if let Some(deg) = in_degree.get_mut(to.as_str()) {
                *deg += 1;
            }
        }

        let mut queue: Vec<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        queue.sort_unstable();

        let mut order = Vec::with_capacity(self.stages.len());

        while let Some(node) = queue.pop() {
            order.push(node.to_string());
            if let Some(neighbors) = adjacency.get(node) {
                for &neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(neighbor);
                            queue.sort_unstable();
                        }
                    }
                }
            }
        }

        if order.len() == self.stages.len() {
            Some(order)
        } else {
            None
        }
    }

    /// Validate the graph structure.
    ///
    /// Checks:
    /// 1. All edge endpoints reference existing stages
    /// 2. No duplicate stage IDs
    /// 3. Graph is a DAG (no cycles)
    ///
    /// # Errors
    ///
    /// Returns a description of the first structural issue found.
    pub fn validate(&self) -> Result<(), String> {
        let mut seen_ids = std::collections::HashSet::new();
        for stage in &self.stages {
            if !seen_ids.insert(&stage.id) {
                return Err(format!("duplicate stage ID: {}", stage.id));
            }
        }

        for (from, to) in &self.edges {
            if !seen_ids.contains(from) {
                return Err(format!("edge references unknown stage: {from}"));
            }
            if !seen_ids.contains(to) {
                return Err(format!("edge references unknown stage: {to}"));
            }
        }

        if self.execute_order().is_none() {
            return Err("graph contains a cycle".to_string());
        }

        Ok(())
    }
}

/// Track execution results across all stages of a pipeline.
#[derive(Debug)]
pub struct PipelineExecution {
    /// Pipeline name (from graph).
    pub pipeline_name: String,
    /// Per-stage results in execution order.
    pub results: Vec<StageResult>,
}

impl PipelineExecution {
    /// Create a new execution tracker for a pipeline.
    #[must_use]
    pub fn new(pipeline_name: &str) -> Self {
        Self {
            pipeline_name: pipeline_name.to_string(),
            results: Vec::new(),
        }
    }

    /// Record a stage result.
    pub fn record(&mut self, result: StageResult) {
        self.results.push(result);
    }

    /// Whether all recorded stages passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        !self.results.is_empty() && self.results.iter().all(|r| r.success)
    }

    /// Total execution time in microseconds.
    #[must_use]
    pub fn total_elapsed_us(&self) -> f64 {
        self.results.iter().map(|r| r.elapsed_us).sum()
    }

    /// Count of completed stages.
    #[must_use]
    pub const fn completed_count(&self) -> usize {
        self.results.len()
    }

    /// Count of failed stages.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.results.iter().filter(|r| !r.success).count()
    }
}

/// Build the canonical neuralSpring spectral analysis pipeline graph.
///
/// ```text
/// eigensolve → ipr  → entropy
///            ↘ lsr ↗
/// ```
///
/// Stage 1: Eigensolve (GPU or CPU) — produces eigenvalues
/// Stage 2a: IPR — inverse participation ratio from eigenvalues
/// Stage 2b: LSR — level spacing ratio from eigenvalues
/// Stage 3: Entropy — Shannon entropy from spectral statistics
#[must_use]
pub fn spectral_pipeline() -> PipelineGraph {
    let mut g = PipelineGraph::new("neuralSpring spectral analysis");

    g.add_stage(StageNode {
        id: "eigensolve".to_string(),
        capability: "science.eigensolve".to_string(),
        substrate: MixedSubstrate::GpuOnly,
        label: "Eigensolve (Hermitian)".to_string(),
    });
    g.add_stage(StageNode {
        id: "ipr".to_string(),
        capability: "science.ipr".to_string(),
        substrate: MixedSubstrate::CpuOnly,
        label: "Inverse Participation Ratio".to_string(),
    });
    g.add_stage(StageNode {
        id: "lsr".to_string(),
        capability: "science.level_spacing_ratio".to_string(),
        substrate: MixedSubstrate::CpuOnly,
        label: "Level Spacing Ratio".to_string(),
    });
    g.add_stage(StageNode {
        id: "entropy".to_string(),
        capability: "science.shannon_entropy".to_string(),
        substrate: MixedSubstrate::CpuOnly,
        label: "Shannon Entropy".to_string(),
    });

    g.add_edge("eigensolve", "ipr");
    g.add_edge("eigensolve", "lsr");
    g.add_edge("ipr", "entropy");
    g.add_edge("lsr", "entropy");

    g
}

/// Build a population genetics pipeline graph.
///
/// ```text
/// allele_freq → nucleotide_div → fst → entropy
/// ```
#[must_use]
pub fn population_genetics_pipeline() -> PipelineGraph {
    let mut g = PipelineGraph::new("population genetics pipeline");

    g.add_stage(StageNode {
        id: "allele_freq".to_string(),
        capability: "science.allele_frequencies".to_string(),
        substrate: MixedSubstrate::CpuOnly,
        label: "Allele Frequencies".to_string(),
    });
    g.add_stage(StageNode {
        id: "nucleotide_div".to_string(),
        capability: "science.nucleotide_diversity".to_string(),
        substrate: MixedSubstrate::CpuOnly,
        label: "Nucleotide Diversity (π)".to_string(),
    });
    g.add_stage(StageNode {
        id: "fst".to_string(),
        capability: "science.pairwise_fst".to_string(),
        substrate: MixedSubstrate::CpuOnly,
        label: "Pairwise FST".to_string(),
    });
    g.add_stage(StageNode {
        id: "entropy".to_string(),
        capability: "science.shannon_entropy".to_string(),
        substrate: MixedSubstrate::CpuOnly,
        label: "Shannon Entropy".to_string(),
    });

    g.add_edge("allele_freq", "nucleotide_div");
    g.add_edge("nucleotide_div", "fst");
    g.add_edge("fst", "entropy");

    g
}

/// Build a protein folding pipeline graph (`EvoFormer` → Structure Module).
///
/// ```text
/// evoformer → structure_module → folding_health
/// ```
#[must_use]
pub fn folding_pipeline() -> PipelineGraph {
    let mut g = PipelineGraph::new("protein folding pipeline");

    g.add_stage(StageNode {
        id: "evoformer".to_string(),
        capability: "science.evoformer_block".to_string(),
        substrate: MixedSubstrate::GpuOnly,
        label: "EvoFormer Block".to_string(),
    });
    g.add_stage(StageNode {
        id: "structure_module".to_string(),
        capability: "science.structure_module".to_string(),
        substrate: MixedSubstrate::GpuOnly,
        label: "Structure Module (IPA)".to_string(),
    });
    g.add_stage(StageNode {
        id: "folding_health".to_string(),
        capability: "science.folding_health".to_string(),
        substrate: MixedSubstrate::CpuOnly,
        label: "Folding Health Report".to_string(),
    });

    g.add_edge("evoformer", "structure_module");
    g.add_edge("structure_module", "folding_health");

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_pipeline_is_valid() {
        let g = spectral_pipeline();
        assert!(g.validate().is_ok());
        assert_eq!(g.stage_count(), 4);
        assert_eq!(g.edge_count(), 4);
    }

    #[test]
    fn spectral_pipeline_topo_order() {
        let g = spectral_pipeline();
        let order = g.execute_order().expect("DAG should have valid topo order");
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], "eigensolve");
        assert_eq!(order.last().expect("non-empty topo order"), "entropy");
    }

    #[test]
    fn popgen_pipeline_is_valid() {
        let g = population_genetics_pipeline();
        assert!(g.validate().is_ok());
        assert_eq!(g.stage_count(), 4);
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn popgen_pipeline_linear_order() {
        let g = population_genetics_pipeline();
        let order = g.execute_order().expect("DAG should have valid topo order");
        assert_eq!(
            order,
            vec!["allele_freq", "nucleotide_div", "fst", "entropy"]
        );
    }

    #[test]
    fn folding_pipeline_is_valid() {
        let g = folding_pipeline();
        assert!(g.validate().is_ok());
        assert_eq!(g.stage_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn folding_pipeline_topo_order() {
        let g = folding_pipeline();
        let order = g.execute_order().unwrap();
        assert_eq!(
            order,
            vec!["evoformer", "structure_module", "folding_health"]
        );
    }

    #[test]
    fn empty_graph_is_valid() {
        let g = PipelineGraph::new("empty");
        assert!(g.validate().is_ok());
        assert_eq!(g.stage_count(), 0);
        assert_eq!(g.execute_order(), Some(vec![]));
    }

    #[test]
    fn cycle_detected() {
        let mut g = PipelineGraph::new("cycle");
        g.add_stage(StageNode {
            id: "a".into(),
            capability: "x".into(),
            substrate: MixedSubstrate::CpuOnly,
            label: "A".into(),
        });
        g.add_stage(StageNode {
            id: "b".into(),
            capability: "y".into(),
            substrate: MixedSubstrate::CpuOnly,
            label: "B".into(),
        });
        g.add_edge("a", "b");
        g.add_edge("b", "a");
        assert!(g.validate().is_err());
        assert!(g.execute_order().is_none());
    }

    #[test]
    fn duplicate_stage_id_rejected() {
        let mut g = PipelineGraph::new("dup");
        g.add_stage(StageNode {
            id: "x".into(),
            capability: "c".into(),
            substrate: MixedSubstrate::CpuOnly,
            label: "X".into(),
        });
        g.add_stage(StageNode {
            id: "x".into(),
            capability: "d".into(),
            substrate: MixedSubstrate::CpuOnly,
            label: "X2".into(),
        });
        let err = g.validate().unwrap_err();
        assert!(err.contains("duplicate"));
    }

    #[test]
    fn edge_to_unknown_stage_rejected() {
        let mut g = PipelineGraph::new("bad edge");
        g.add_stage(StageNode {
            id: "a".into(),
            capability: "x".into(),
            substrate: MixedSubstrate::CpuOnly,
            label: "A".into(),
        });
        g.add_edge("a", "nonexistent");
        let err = g.validate().unwrap_err();
        assert!(err.contains("unknown"));
    }

    #[test]
    fn stage_lookup() {
        let g = spectral_pipeline();
        let ipr = g.stage("ipr").expect("spectral pipeline has ipr stage");
        assert_eq!(ipr.capability, "science.ipr");
        assert!(g.stage("nonexistent").is_none());
    }

    #[test]
    fn pipeline_execution_tracking() {
        let mut exec = PipelineExecution::new("test");
        assert!(!exec.all_passed());
        assert_eq!(exec.completed_count(), 0);

        exec.record(StageResult {
            stage_id: "a".into(),
            success: true,
            elapsed_us: 100.0,
            actual_substrate: MixedSubstrate::CpuOnly,
            output: StageOutput::Scalar(42.0),
        });
        exec.record(StageResult {
            stage_id: "b".into(),
            success: true,
            elapsed_us: 200.0,
            actual_substrate: MixedSubstrate::GpuOnly,
            output: StageOutput::Empty,
        });

        assert!(exec.all_passed());
        assert_eq!(exec.completed_count(), 2);
        assert_eq!(exec.failed_count(), 0);
        assert!((exec.total_elapsed_us() - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pipeline_execution_with_failure() {
        let mut exec = PipelineExecution::new("test");
        exec.record(StageResult {
            stage_id: "a".into(),
            success: true,
            elapsed_us: 50.0,
            actual_substrate: MixedSubstrate::CpuOnly,
            output: StageOutput::Empty,
        });
        exec.record(StageResult {
            stage_id: "b".into(),
            success: false,
            elapsed_us: 10.0,
            actual_substrate: MixedSubstrate::GpuOnly,
            output: StageOutput::Empty,
        });

        assert!(!exec.all_passed());
        assert_eq!(exec.failed_count(), 1);
    }

    #[test]
    fn diamond_dag_valid() {
        let mut g = PipelineGraph::new("diamond");
        for id in ["a", "b", "c", "d"] {
            g.add_stage(StageNode {
                id: id.into(),
                capability: format!("cap.{id}"),
                substrate: MixedSubstrate::CpuOnly,
                label: id.to_uppercase(),
            });
        }
        g.add_edge("a", "b");
        g.add_edge("a", "c");
        g.add_edge("b", "d");
        g.add_edge("c", "d");
        assert!(g.validate().is_ok());
        let order = g
            .execute_order()
            .expect("diamond DAG should have valid topo order");
        assert_eq!(order[0], "a");
        assert_eq!(order.last().expect("non-empty topo order"), "d");
    }

    #[test]
    fn stage_output_variants() {
        let scalar = StageOutput::Scalar(3.14);
        assert!(matches!(scalar, StageOutput::Scalar(v) if (v - 3.14).abs() < 1e-10));

        let vec = StageOutput::Vector(vec![1.0, 2.0, 3.0]);
        assert!(matches!(vec, StageOutput::Vector(ref v) if v.len() == 3));

        let mut map = HashMap::new();
        map.insert("ipr".to_string(), 0.25);
        let map_out = StageOutput::Map(map);
        assert!(matches!(map_out, StageOutput::Map(ref m) if m.contains_key("ipr")));
    }
}
