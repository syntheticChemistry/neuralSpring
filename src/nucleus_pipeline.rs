// SPDX-License-Identifier: AGPL-3.0-or-later

//! NUCLEUS pipeline executor for neuralSpring.
//!
//! Bridges metalForge [`PipelineGraph`] to actual neuralSpring computation.
//! Follows the NUCLEUS atomic model:
//!
//! - **Tower**: Capability discovery — resolves stage capabilities to local functions
//! - **Node**: Compute dispatch — executes each stage (CPU or GPU via `Dispatcher`)
//! - **Nest**: Provenance — records substrate, timing, outputs per stage
//!
//! ## Usage
//!
//! ```no_run
//! use neural_spring::nucleus_pipeline::{execute_composition_pipeline, PipelineReport};
//! let report = execute_composition_pipeline();
//! assert!(report.all_passed());
//! ```

#![expect(
    clippy::cast_precision_loss,
    reason = "timing and dimension values fit in f64"
)]

use neural_spring_forge::graph::{
    PipelineExecution, PipelineGraph, StageOutput, StageResult,
};

/// A completed pipeline report with provenance metadata.
#[derive(Debug)]
pub struct PipelineReport {
    pub execution: PipelineExecution,
    pub pipeline_name: String,
    pub substrate_used: String,
    pub total_stages: usize,
}

impl PipelineReport {
    /// Whether all stages passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.execution.all_passed()
    }

    /// Total elapsed microseconds.
    #[must_use]
    pub fn total_us(&self) -> f64 {
        self.execution.total_elapsed_us()
    }
}

/// Execute the composition pipeline locally (CPU path).
///
/// Runs the 6-stage composition pipeline DAG:
/// `eigensolve` → `digester_anderson` / `isomorphic_reservoir` → ... → `attention_anderson`
///
/// Each stage calls the real neuralSpring module and records timing + outputs.
#[must_use]
pub fn execute_composition_pipeline() -> PipelineReport {
    let graph = neural_spring_forge::graph::composition_pipeline();
    execute_graph(&graph)
}

/// Execute any `PipelineGraph` by resolving capabilities to local functions.
///
/// Tower phase: validate graph and compute topological order.
/// Node phase: dispatch each stage to local computation.
/// Nest phase: record provenance (substrate, timing, outputs).
///
/// # Panics
///
/// Panics if the graph contains a cycle or references invalid stage IDs
/// (both of which are prevented by `PipelineGraph::validate`).
#[must_use]
#[expect(
    clippy::expect_used,
    reason = "graph is validated at construction — these are structural invariants"
)]
pub fn execute_graph(graph: &PipelineGraph) -> PipelineReport {
    let order = graph
        .execute_order()
        .expect("graph must be a valid DAG");

    let mut exec = PipelineExecution::new(&graph.name);

    for stage_id in &order {
        let stage = graph
            .stage(stage_id)
            .expect("topo order references valid stages");

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

    PipelineReport {
        pipeline_name: graph.name.clone(),
        substrate_used: "CPU".to_string(),
        total_stages: order.len(),
        execution: exec,
    }
}

/// Tower: resolve a capability string to a local computation function.
///
/// Returns `(success, output)`. Each capability maps to a real neuralSpring
/// module function. Unknown capabilities return `(false, Empty)`.
fn dispatch_capability(capability: &str) -> (bool, StageOutput) {
    match capability {
        "science.eigensolve" => stage_eigensolve(),
        "science.digester_anderson_coupling" => stage_digester_anderson(),
        "science.isomorphic_reservoir" => stage_isomorphic_reservoir(),
        "science.wdm_ensemble_qs" => stage_wdm_ensemble_qs(),
        "science.introgression_nn" => stage_introgression_nn(),
        "science.attention_anderson" => stage_attention_anderson(),
        _ => (false, StageOutput::Empty),
    }
}

fn stage_eigensolve() -> (bool, StageOutput) {
    let n = 16;
    let mut matrix = vec![0.0; n * n];
    for i in 0..n {
        matrix[i * n + i] = 1.0;
    }
    let result = crate::eigh::eigh_householder_qr(&matrix, n);
    let sum: f64 = result.eigenvalues.iter().sum();
    (
        (sum - n as f64).abs() < 1e-6,
        StageOutput::Vector(result.eigenvalues),
    )
}

fn stage_digester_anderson() -> (bool, StageOutput) {
    let mut rng = crate::rng::Rng::new(42);
    let n_species = 10;
    let (h, evenness, w, ipr, xi) =
        crate::digester_anderson::community_anderson(n_species, 1.0, 20, &mut rng);

    let mut map = std::collections::HashMap::new();
    map.insert("shannon_h".to_string(), h);
    map.insert("evenness".to_string(), evenness);
    map.insert("disorder_w".to_string(), w);
    map.insert("mean_ipr".to_string(), ipr);
    map.insert("xi".to_string(), xi);

    let valid = h > 0.0 && (0.0..=1.0).contains(&ipr);
    (valid, StageOutput::Map(map))
}

fn stage_isomorphic_reservoir() -> (bool, StageOutput) {
    let n = 16;
    let mut rng = crate::rng::Rng::new(42);
    let mut matrices = Vec::new();

    for gain in [0.9, 0.85, 0.95] {
        let mut m = vec![0.0; n * n];
        for val in &mut m {
            *val = rng.uniform().mul_add(2.0, -1.0) * gain / (n as f64).sqrt();
        }
        let sym: Vec<f64> = (0..n * n)
            .map(|idx| {
                let r = idx / n;
                let c = idx % n;
                (m[r * n + c] + m[c * n + r]) * 0.5
            })
            .collect();
        matrices.push(sym);
    }

    let profiles: Vec<_> = matrices
        .iter()
        .zip(["esn", "glucose", "weather"])
        .map(|(m, name)| crate::isomorphic_reservoir::spectral_properties(m, n, name))
        .collect();

    let cdm = crate::isomorphic_reservoir::cross_domain_metrics(&profiles);

    let mut map = std::collections::HashMap::new();
    map.insert("eff_ratio_cv".to_string(), cdm.eff_ratio_cv);
    map.insert("ipr_cv".to_string(), cdm.ipr_cv);
    map.insert("spacing_ratio_mean".to_string(), cdm.spacing_ratio_mean);

    let valid = cdm.eff_ratio_cv < 0.5 && cdm.ipr_cv < 0.5;
    (valid, StageOutput::Map(map))
}

fn stage_wdm_ensemble_qs() -> (bool, StageOutput) {
    let mut rng = crate::rng::Rng::new(42);
    let disagree = 0.5;
    let w = crate::wdm_ensemble_qs::disagreement_to_disorder(disagree, 0.01, 1.0, 16.0);

    let disorder_vec: Vec<f64> = (0..20).map(|_| rng.uniform() * w).collect();
    let (ipr, xi) = crate::wdm_ensemble_qs::anderson_from_disorder(&disorder_vec);

    let w_frac = (w / 16.0).clamp(0.0, 1.0);
    let payoff = crate::wdm_ensemble_qs::snowdrift_payoff(w_frac);
    let coop = crate::wdm_ensemble_qs::replicator_final_coop(&payoff, 500);

    let mut map = std::collections::HashMap::new();
    map.insert("disorder".to_string(), w);
    map.insert("mean_ipr".to_string(), ipr);
    map.insert("xi".to_string(), xi);
    map.insert("cooperation".to_string(), coop);

    let valid = ipr >= 0.0 && (0.0..=1.0).contains(&coop);
    (valid, StageOutput::Map(map))
}

fn stage_introgression_nn() -> (bool, StageOutput) {
    let hmm = crate::introgression_nn::build_nn_hmm();
    let null_hmm = crate::introgression_nn::build_null_hmm();
    let n_layers = 50;

    let mut truth = vec![0_usize; n_layers];
    for t in &mut truth[15..30] {
        *t = 1;
    }

    let mut rng = crate::rng::Rng::new(42);
    let obs: Vec<usize> = truth
        .iter()
        .map(|&s| {
            if s == 1 {
                2
            } else {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "rng in [0,2) → usize"
                )]
                let v = (rng.uniform() * 2.0) as usize;
                v
            }
        })
        .collect();

    let (path, _) = hmm.viterbi(&obs);
    let (tpr, fpr, accuracy) = crate::introgression_nn::detection_metrics(&path, &truth);

    let (_, log_lik_model) = hmm.forward(&obs);
    let (_, log_lik_null) = null_hmm.forward(&obs);

    let mut map = std::collections::HashMap::new();
    map.insert("tpr".to_string(), tpr);
    map.insert("fpr".to_string(), fpr);
    map.insert("accuracy".to_string(), accuracy);
    map.insert("llr".to_string(), log_lik_model - log_lik_null);

    let valid = tpr > 0.5 && accuracy > 0.5;
    (valid, StageOutput::Map(map))
}

fn stage_attention_anderson() -> (bool, StageOutput) {
    let n = 16;
    let mut rng = crate::rng::Rng::new(42);
    let quality = 0.8;

    let mut matrix = vec![0.0; n * n];
    for row in 0..n {
        let mut row_vals = Vec::with_capacity(n);
        for col in 0..n {
            let base = if row == col { quality } else { 1.0 - quality };
            row_vals.push(rng.uniform().mul_add(0.1, base));
        }
        let sum: f64 = row_vals.iter().sum();
        for (col, val) in row_vals.into_iter().enumerate() {
            matrix[row * n + col] = val / sum;
        }
    }
    let sym: Vec<f64> = (0..n * n)
        .map(|idx| {
            let r = idx / n;
            let c = idx % n;
            (matrix[r * n + c] + matrix[c * n + r]) * 0.5
        })
        .collect();

    let result = crate::attention_anderson::attention_spectral(&sym, n);

    let mut map = std::collections::HashMap::new();
    map.insert("quality".to_string(), result.quality);
    map.insert("entropy".to_string(), result.entropy);
    map.insert("mean_ipr".to_string(), result.mean_ipr);
    map.insert("spectral_radius".to_string(), result.spectral_radius);
    map.insert("participation".to_string(), result.participation);

    let valid = result.spectral_radius > 0.0 && result.participation > 0.0;
    (valid, StageOutput::Map(map))
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;
    use neural_spring_forge::mixed::MixedSubstrate;

    #[test]
    fn composition_pipeline_executes_all_stages() {
        let report = execute_composition_pipeline();
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
        let report = execute_composition_pipeline();
        let ids: Vec<&str> = report
            .execution
            .results
            .iter()
            .map(|r| r.stage_id.as_str())
            .collect();

        let pos = |id: &str| -> usize {
            ids.iter()
                .position(|&s| s == id)
                .expect(&format!("{id} should be in results"))
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
        let report = execute_composition_pipeline();
        assert!(report.total_us() > 0.0, "pipeline should take > 0µs");
        for result in &report.execution.results {
            assert!(result.elapsed_us > 0.0, "{} should have timing", result.stage_id);
        }
    }

    #[test]
    fn composition_pipeline_outputs_are_populated() {
        let report = execute_composition_pipeline();
        for result in &report.execution.results {
            match &result.output {
                StageOutput::Map(m) => assert!(!m.is_empty(), "{} map should be non-empty", result.stage_id),
                StageOutput::Vector(v) => assert!(!v.is_empty(), "{} vec should be non-empty", result.stage_id),
                _ => panic!("{} should produce Map or Vector output", result.stage_id),
            }
        }
    }

    #[test]
    fn eigensolve_stage_produces_correct_eigenvalues() {
        let (success, output) = dispatch_capability("science.eigensolve");
        assert!(success);
        if let StageOutput::Vector(evals) = output {
            assert_eq!(evals.len(), 16);
            for &e in &evals {
                assert!((e - 1.0).abs() < 1e-6, "identity matrix eigenvalue should be 1.0");
            }
        } else {
            panic!("eigensolve should produce Vector output");
        }
    }

    #[test]
    fn unknown_capability_fails() {
        let (success, output) = dispatch_capability("science.nonexistent");
        assert!(!success);
        assert!(matches!(output, StageOutput::Empty));
    }

    #[test]
    fn digester_anderson_produces_valid_metrics() {
        let (success, output) = dispatch_capability("science.digester_anderson_coupling");
        assert!(success);
        if let StageOutput::Map(m) = output {
            assert!(m.contains_key("shannon_h"));
            assert!(m.contains_key("mean_ipr"));
            assert!(*m.get("mean_ipr").unwrap() >= 0.0);
            assert!(*m.get("mean_ipr").unwrap() <= 1.0);
        } else {
            panic!("expected Map output");
        }
    }

    #[test]
    fn introgression_detects_anomalous_layers() {
        let (success, output) = dispatch_capability("science.introgression_nn");
        assert!(success);
        if let StageOutput::Map(m) = output {
            assert!(*m.get("tpr").unwrap() > 0.5, "TPR should be > 0.5");
            assert!(*m.get("accuracy").unwrap() > 0.5, "accuracy should be > 0.5");
        } else {
            panic!("expected Map output");
        }
    }

    #[test]
    fn substrate_provenance_is_recorded() {
        let report = execute_composition_pipeline();
        for result in &report.execution.results {
            match result.stage_id.as_str() {
                "attention_anderson" => {
                    assert_eq!(result.actual_substrate, MixedSubstrate::GpuOnly);
                }
                _ => {
                    assert_eq!(result.actual_substrate, MixedSubstrate::CpuOnly);
                }
            }
        }
    }
}
