// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(
    clippy::cast_precision_loss,
    reason = "small graph counts → f64 for ValidationHarness checks"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "graph stage counts fit in u32"
)]

//! biomeOS Graph Coordination Validator
//!
//! Validates the DAG-based pipeline graph infrastructure:
//!   1. Graph construction and structural validation
//!   2. Topological sort correctness
//!   3. Cycle detection
//!   4. Pipeline execution tracking
//!   5. Canonical pipeline definitions (spectral, popgen, folding)
//!   6. Mixed-substrate stage routing
//!   7. Graph-driven dispatch simulation
//!
//! ## Provenance
//!
//! Validation class: Infrastructure (graph coordination layer)
//! No Python baseline — validates graph algorithms and pipeline definitions
//! against analytical expectations (topo order, DAG invariants, stage counts).

#![expect(
    clippy::too_many_lines,
    reason = "validation binary — comprehensive graph coordination tests"
)]

use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring_forge::graph::{
    self, PipelineExecution, PipelineGraph, StageNode, StageOutput, StageResult,
};
use neural_spring_forge::mixed::MixedSubstrate;

fn main() {
    let mut h = ValidationHarness::new("biomeOS Graph Coordination");

    // ═══════════════════════════════════════════════════════════════════
    // 1. Spectral pipeline structure
    // ═══════════════════════════════════════════════════════════════════
    let spectral = graph::spectral_pipeline();
    h.check_abs(
        "spectral.stage_count",
        spectral.stage_count() as f64,
        4.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "spectral.edge_count",
        spectral.edge_count() as f64,
        4.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "spectral.validate",
        if spectral.validate().is_ok() {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    let spectral_order = spectral.execute_order();
    let spectral_order_valid = spectral_order.as_ref().is_some_and(|o| {
        o.len() == 4 && o[0] == "eigensolve" && o.last().is_some_and(|l| l == "entropy")
    });
    h.check_abs(
        "spectral.topo_order_valid",
        if spectral_order_valid { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    let eigensolve = spectral.stage("eigensolve");
    h.check_abs(
        "spectral.eigensolve_capability",
        if eigensolve.is_some_and(|s| s.capability == "science.eigensolve") {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "spectral.eigensolve_substrate_gpu",
        if eigensolve.is_some_and(|s| s.substrate == MixedSubstrate::GpuOnly) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 2. Population genetics pipeline structure
    // ═══════════════════════════════════════════════════════════════════
    let popgen = graph::population_genetics_pipeline();
    h.check_abs(
        "popgen.stage_count",
        popgen.stage_count() as f64,
        4.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "popgen.edge_count",
        popgen.edge_count() as f64,
        3.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "popgen.validate",
        if popgen.validate().is_ok() { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    let popgen_order = popgen.execute_order();
    let popgen_linear = popgen_order
        .as_ref()
        .is_some_and(|o| o == &["allele_freq", "nucleotide_div", "fst", "entropy"]);
    h.check_abs(
        "popgen.linear_order",
        if popgen_linear { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 3. Folding pipeline structure
    // ═══════════════════════════════════════════════════════════════════
    let folding = graph::folding_pipeline();
    h.check_abs(
        "folding.stage_count",
        folding.stage_count() as f64,
        3.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "folding.validate",
        if folding.validate().is_ok() { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    let folding_order = folding.execute_order();
    let folding_correct = folding_order
        .as_ref()
        .is_some_and(|o| o == &["evoformer", "structure_module", "folding_health"]);
    h.check_abs(
        "folding.order_correct",
        if folding_correct { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 4. Cycle detection
    // ═══════════════════════════════════════════════════════════════════
    let mut cyclic = PipelineGraph::new("cycle_test");
    cyclic.add_stage(StageNode {
        id: "a".into(),
        capability: "x".into(),
        substrate: MixedSubstrate::CpuOnly,
        label: "A".into(),
    });
    cyclic.add_stage(StageNode {
        id: "b".into(),
        capability: "y".into(),
        substrate: MixedSubstrate::CpuOnly,
        label: "B".into(),
    });
    cyclic.add_edge("a", "b");
    cyclic.add_edge("b", "a");
    h.check_abs(
        "cycle.detected",
        if cyclic.validate().is_err() { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "cycle.topo_returns_none",
        if cyclic.execute_order().is_none() {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 5. Diamond DAG (parallel branches merge)
    // ═══════════════════════════════════════════════════════════════════
    let mut diamond = PipelineGraph::new("diamond");
    for id in ["root", "left", "right", "sink"] {
        diamond.add_stage(StageNode {
            id: id.into(),
            capability: format!("cap.{id}"),
            substrate: MixedSubstrate::CpuOnly,
            label: id.to_uppercase(),
        });
    }
    diamond.add_edge("root", "left");
    diamond.add_edge("root", "right");
    diamond.add_edge("left", "sink");
    diamond.add_edge("right", "sink");
    h.check_abs(
        "diamond.valid",
        if diamond.validate().is_ok() { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    let diamond_order = diamond.execute_order().unwrap_or_default();
    let diamond_root_first = diamond_order.first().is_some_and(|f| f == "root");
    let diamond_sink_last = diamond_order.last().is_some_and(|l| l == "sink");
    h.check_abs(
        "diamond.root_first",
        if diamond_root_first { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "diamond.sink_last",
        if diamond_sink_last { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 6. Pipeline execution tracking
    // ═══════════════════════════════════════════════════════════════════
    let mut exec = PipelineExecution::new("test_pipeline");

    let order = spectral.execute_order().unwrap_or_default();
    for (i, stage_id) in order.iter().enumerate() {
        let stage = spectral.stage(stage_id);
        exec.record(StageResult {
            stage_id: stage_id.clone(),
            success: true,
            elapsed_us: (i as f64 + 1.0) * 100.0,
            actual_substrate: stage.map_or(MixedSubstrate::CpuOnly, |s| s.substrate),
            output: StageOutput::Scalar(f64::from(i as u32)),
        });
    }

    h.check_abs(
        "exec.all_passed",
        if exec.all_passed() { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "exec.completed_count",
        exec.completed_count() as f64,
        4.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "exec.failed_count",
        exec.failed_count() as f64,
        0.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "exec.total_elapsed_us",
        exec.total_elapsed_us(),
        1000.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 7. Execution with failure
    // ═══════════════════════════════════════════════════════════════════
    let mut exec_fail = PipelineExecution::new("fail_test");
    exec_fail.record(StageResult {
        stage_id: "ok_stage".into(),
        success: true,
        elapsed_us: 50.0,
        actual_substrate: MixedSubstrate::CpuOnly,
        output: StageOutput::Empty,
    });
    exec_fail.record(StageResult {
        stage_id: "bad_stage".into(),
        success: false,
        elapsed_us: 10.0,
        actual_substrate: MixedSubstrate::GpuOnly,
        output: StageOutput::Empty,
    });
    h.check_abs(
        "exec_fail.not_all_passed",
        if exec_fail.all_passed() { 0.0 } else { 1.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "exec_fail.failed_count",
        exec_fail.failed_count() as f64,
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 8. Stage output variants
    // ═══════════════════════════════════════════════════════════════════
    let pi_approx = std::f64::consts::PI;
    let scalar = StageOutput::Scalar(pi_approx);
    h.check_abs(
        "output.scalar",
        if matches!(scalar, StageOutput::Scalar(v) if (v - pi_approx).abs() < tolerances::CROSS_LANGUAGE) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    let vector = StageOutput::Vector(vec![1.0, 2.0, 3.0]);
    h.check_abs(
        "output.vector_len",
        if matches!(&vector, StageOutput::Vector(v) if v.len() == 3) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 9. Empty graph edge case
    // ═══════════════════════════════════════════════════════════════════
    let empty = PipelineGraph::new("empty");
    h.check_abs(
        "empty.valid",
        if empty.validate().is_ok() { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "empty.topo_empty",
        if empty.execute_order() == Some(vec![]) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 10. Graph-driven dispatch simulation
    // ═══════════════════════════════════════════════════════════════════
    let popgen_sim = graph::population_genetics_pipeline();
    let sim_order = popgen_sim.execute_order().unwrap_or_default();
    let mut sim_exec = PipelineExecution::new("popgen simulation");

    let sim_values = [0.3, 0.01, 0.15, 2.5];
    for (i, stage_id) in sim_order.iter().enumerate() {
        let stage = popgen_sim.stage(stage_id);
        sim_exec.record(StageResult {
            stage_id: stage_id.clone(),
            success: true,
            elapsed_us: 50.0 * (i as f64 + 1.0),
            actual_substrate: stage.map_or(MixedSubstrate::CpuOnly, |s| s.substrate),
            output: StageOutput::Scalar(sim_values[i]),
        });
    }
    h.check_abs(
        "sim.all_passed",
        if sim_exec.all_passed() { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "sim.stage_count",
        sim_exec.completed_count() as f64,
        4.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 11. Duplicate stage ID rejection
    // ═══════════════════════════════════════════════════════════════════
    let mut dup = PipelineGraph::new("duplicate_test");
    dup.add_stage(StageNode {
        id: "x".into(),
        capability: "c1".into(),
        substrate: MixedSubstrate::CpuOnly,
        label: "X1".into(),
    });
    dup.add_stage(StageNode {
        id: "x".into(),
        capability: "c2".into(),
        substrate: MixedSubstrate::CpuOnly,
        label: "X2".into(),
    });
    h.check_abs(
        "dup.rejected",
        if dup.validate().is_err() { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 12. Bad edge reference rejection
    // ═══════════════════════════════════════════════════════════════════
    let mut bad_edge = PipelineGraph::new("bad_edge_test");
    bad_edge.add_stage(StageNode {
        id: "a".into(),
        capability: "x".into(),
        substrate: MixedSubstrate::CpuOnly,
        label: "A".into(),
    });
    bad_edge.add_edge("a", "nonexistent");
    h.check_abs(
        "bad_edge.rejected",
        if bad_edge.validate().is_err() {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    h.finish();
}
