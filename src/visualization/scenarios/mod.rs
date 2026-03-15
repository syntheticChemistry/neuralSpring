// SPDX-License-Identifier: AGPL-3.0-or-later

//! Per-domain petalTongue scenario builders for neuralSpring.
//!
//! Each builder calls real neuralSpring computation and wraps outputs in
//! [`super::types::DataChannel`] / [`super::types::ScenarioNode`] / [`super::types::NeuralScenario`] so petalTongue
//! can render them directly.

mod attention_anderson;
mod combiners;
mod coordination;
mod digester_anderson;
mod folding;
mod game_theory;
mod glucose;
mod hmm;
mod immunological;
mod industry_coverage;
mod introgression_nn;
mod isomorphic_reservoir;
mod kokkos_parity;
mod loss_landscape;
mod population;
mod provenance;
pub(crate) mod scaffold;
mod search_results;
mod spectral;
mod streaming_io;
mod training;
mod wdm;
mod wdm_ensemble_qs;

pub use attention_anderson::attention_anderson_study;
pub use combiners::{composition_study, full_study, scenario_with_edges_json};
pub use coordination::coordination_study;
pub use digester_anderson::digester_anderson_study;
pub use folding::folding_study;
pub use game_theory::game_theory_study;
pub use glucose::glucose_study;
pub use hmm::hmm_study;
pub use immunological::immunological_study;
pub use industry_coverage::industry_coverage_study;
pub use introgression_nn::introgression_nn_study;
pub use isomorphic_reservoir::isomorphic_reservoir_study;
pub use kokkos_parity::kokkos_parity_study;
pub use loss_landscape::loss_landscape_study;
pub use population::population_study;
pub use provenance::provenance_study;
pub use search_results::search_study;
pub use spectral::spectral_study;
pub use streaming_io::streaming_io_study;
pub use training::training_study;
pub use wdm::wdm_study;
pub use wdm_ensemble_qs::wdm_ensemble_qs_study;

// Re-export scaffold helpers for sub-modules.
pub(crate) use scaffold::{
    bar, distribution, edge, fieldmap, gauge, heatmap, node, scaffold, scatter3d, spectrum,
    timeseries,
};

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;
    use crate::visualization::types::DataChannel;

    fn assert_study_invariants(
        scenario: &crate::visualization::types::NeuralScenario,
        edges: &[crate::visualization::types::ScenarioEdge],
        expected_node_ids: &[&str],
        min_edge_count: usize,
    ) {
        let nodes = &scenario.ecosystem.primals;
        assert_eq!(nodes.len(), expected_node_ids.len(), "node count mismatch");
        for n in nodes {
            assert!(n.health <= 100, "node {} health {} > 100", n.id, n.health);
        }
        let ids: std::collections::HashSet<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids.len(), nodes.len(), "duplicate node IDs");
        for id in expected_node_ids {
            assert!(ids.contains(id), "expected node id {id} not found");
        }
        assert!(
            edges.len() >= min_edge_count,
            "expected >= {min_edge_count} edges, got {}",
            edges.len()
        );
    }

    fn assert_json_roundtrips(
        scenario: &crate::visualization::types::NeuralScenario,
        edges: &[crate::visualization::types::ScenarioEdge],
    ) {
        let json = scenario_with_edges_json(scenario, edges);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON must be valid");
        assert!(parsed.get("name").is_some());
        assert!(parsed.get("ecosystem").is_some());
        if !edges.is_empty() {
            assert!(parsed.get("edges").is_some());
            assert!(parsed["edges"].is_array());
        }
    }

    #[test]
    fn spectral_study_structure() {
        let (scenario, edges) = spectral_study();
        assert_study_invariants(
            &scenario,
            &edges,
            &["anderson_sweep", "spectral_analysis", "hessian_eigen"],
            2,
        );
    }

    #[test]
    fn spectral_study_capabilities() {
        let (scenario, _) = spectral_study();
        let caps: std::collections::HashSet<String> = scenario
            .ecosystem
            .primals
            .iter()
            .flat_map(|n| n.capabilities.clone())
            .collect();
        assert!(caps.contains("science.anderson_localization"));
        assert!(caps.contains("science.spectral_analysis"));
        assert!(caps.contains("science.hessian_eigen"));
    }

    #[test]
    fn spectral_study_json_roundtrips() {
        let (scenario, edges) = spectral_study();
        assert_json_roundtrips(&scenario, &edges);
    }

    #[test]
    fn spectral_study_has_timeseries_and_spectrum() {
        let (scenario, _) = spectral_study();
        let channels: Vec<&DataChannel> = scenario
            .ecosystem
            .primals
            .iter()
            .flat_map(|n| n.data_channels.iter())
            .collect();

        let has_ts = channels
            .iter()
            .any(|c| matches!(c, DataChannel::TimeSeries { .. }));
        let has_sp = channels
            .iter()
            .any(|c| matches!(c, DataChannel::Spectrum { .. }));
        let has_ga = channels
            .iter()
            .any(|c| matches!(c, DataChannel::Gauge { .. }));
        assert!(has_ts, "spectral study should have TimeSeries");
        assert!(has_sp, "spectral study should have Spectrum");
        assert!(has_ga, "spectral study should have Gauge");
    }

    #[test]
    fn training_study_structure() {
        let (scenario, edges) = training_study();
        assert_study_invariants(&scenario, &edges, &["training_trajectory"], 0);
    }

    #[test]
    fn training_study_multi_channel() {
        let (scenario, _) = training_study();
        let node = &scenario.ecosystem.primals[0];
        assert!(
            node.data_channels.len() >= 4,
            "training node should have >= 4 channels, got {}",
            node.data_channels.len()
        );
    }

    #[test]
    fn training_study_json_roundtrips() {
        let (scenario, edges) = training_study();
        assert_json_roundtrips(&scenario, &edges);
    }

    #[test]
    fn coordination_study_structure() {
        let (scenario, edges) = coordination_study();
        assert_study_invariants(&scenario, &edges, &["agent_coordination"], 0);
    }

    #[test]
    fn coordination_study_has_scatter3d() {
        let (scenario, _) = coordination_study();
        let channels: Vec<&DataChannel> = scenario
            .ecosystem
            .primals
            .iter()
            .flat_map(|n| n.data_channels.iter())
            .collect();
        let has_s3d = channels
            .iter()
            .any(|c| matches!(c, DataChannel::Scatter3D { .. }));
        assert!(has_s3d, "coordination study should have Scatter3D");
    }

    #[test]
    fn coordination_study_json_roundtrips() {
        let (scenario, edges) = coordination_study();
        assert_json_roundtrips(&scenario, &edges);
    }

    #[test]
    fn provenance_study_structure() {
        let (scenario, _edges) = provenance_study();
        assert!(!scenario.ecosystem.primals.is_empty());
        let node = &scenario.ecosystem.primals[0];
        assert_eq!(node.id, "shader_provenance");
    }

    #[test]
    fn provenance_study_json_roundtrips() {
        let (scenario, edges) = provenance_study();
        assert_json_roundtrips(&scenario, &edges);
    }

    #[test]
    fn folding_study_structure() {
        let (scenario, edges) = folding_study();
        assert_study_invariants(&scenario, &edges, &["folding_primitives"], 0);
    }

    #[test]
    fn folding_study_has_all_primitives() {
        let (scenario, _) = folding_study();
        let node = &scenario.ecosystem.primals[0];
        let bar_channel = node
            .data_channels
            .iter()
            .find(|c| matches!(c, DataChannel::Bar { .. }));
        assert!(
            bar_channel.is_some(),
            "folding study should have Bar channel"
        );
        if let Some(DataChannel::Bar {
            categories, values, ..
        }) = bar_channel
        {
            assert_eq!(categories.len(), 14, "should have 14 folding primitives");
            assert!(values.iter().all(|&v| (v - 1.0).abs() < f64::EPSILON));
        }
    }

    #[test]
    fn folding_study_json_roundtrips() {
        let (scenario, edges) = folding_study();
        assert_json_roundtrips(&scenario, &edges);
    }

    #[test]
    fn search_study_structure() {
        let (scenario, edges) = search_study();
        assert_study_invariants(
            &scenario,
            &edges,
            &["search_pipeline", "kmer_index", "hit_analysis"],
            2,
        );
    }

    #[test]
    fn search_study_has_bar_and_gauge() {
        let (scenario, _) = search_study();
        let channels: Vec<&DataChannel> = scenario
            .ecosystem
            .primals
            .iter()
            .flat_map(|n| n.data_channels.iter())
            .collect();
        let has_bar = channels
            .iter()
            .any(|c| matches!(c, DataChannel::Bar { .. }));
        let has_gauge = channels
            .iter()
            .any(|c| matches!(c, DataChannel::Gauge { .. }));
        assert!(has_bar, "search study should have Bar channel");
        assert!(has_gauge, "search study should have Gauge channel");
    }

    #[test]
    fn search_study_json_roundtrips() {
        let (scenario, edges) = search_study();
        assert_json_roundtrips(&scenario, &edges);
    }

    #[test]
    fn streaming_io_study_structure() {
        let (scenario, edges) = streaming_io_study();
        assert_study_invariants(
            &scenario,
            &edges,
            &["fastq_quality", "fasta_lengths", "vcf_variants"],
            2,
        );
    }

    #[test]
    fn streaming_io_study_has_distribution() {
        let (scenario, _) = streaming_io_study();
        let channels: Vec<&DataChannel> = scenario
            .ecosystem
            .primals
            .iter()
            .flat_map(|n| n.data_channels.iter())
            .collect();
        let has_dist = channels
            .iter()
            .any(|c| matches!(c, DataChannel::Distribution { .. }));
        assert!(has_dist, "streaming I/O study should have Distribution");
    }

    #[test]
    fn streaming_io_study_json_roundtrips() {
        let (scenario, edges) = streaming_io_study();
        assert_json_roundtrips(&scenario, &edges);
    }

    #[test]
    fn kokkos_parity_study_structure() {
        let (scenario, edges) = kokkos_parity_study();
        assert_study_invariants(
            &scenario,
            &edges,
            &[
                "parity_overview",
                "parallel_for_ops",
                "parallel_reduce_ops",
                "domain_ops",
            ],
            3,
        );
    }

    #[test]
    fn kokkos_parity_study_has_heatmap() {
        let (scenario, _) = kokkos_parity_study();
        let channels: Vec<&DataChannel> = scenario
            .ecosystem
            .primals
            .iter()
            .flat_map(|n| n.data_channels.iter())
            .collect();
        let has_hm = channels
            .iter()
            .any(|c| matches!(c, DataChannel::Heatmap { .. }));
        assert!(has_hm, "Kokkos parity should have Heatmap");
    }

    #[test]
    fn kokkos_parity_study_json_roundtrips() {
        let (scenario, edges) = kokkos_parity_study();
        assert_json_roundtrips(&scenario, &edges);
    }

    #[test]
    fn industry_coverage_study_structure() {
        let (scenario, edges) = industry_coverage_study();
        assert_study_invariants(
            &scenario,
            &edges,
            &[
                "coverage_overview",
                "domain_progress",
                "implementation_detail",
            ],
            2,
        );
    }

    #[test]
    fn industry_coverage_study_has_gauge() {
        let (scenario, _) = industry_coverage_study();
        let channels: Vec<&DataChannel> = scenario
            .ecosystem
            .primals
            .iter()
            .flat_map(|n| n.data_channels.iter())
            .collect();
        let has_gauge = channels
            .iter()
            .any(|c| matches!(c, DataChannel::Gauge { .. }));
        assert!(has_gauge, "industry coverage should have Gauge");
    }

    #[test]
    fn industry_coverage_study_json_roundtrips() {
        let (scenario, edges) = industry_coverage_study();
        assert_json_roundtrips(&scenario, &edges);
    }

    #[test]
    fn digester_anderson_study_structure() {
        let (scenario, edges) = digester_anderson_study();
        assert_study_invariants(
            &scenario,
            &edges,
            &["digester_community", "anderson_coupling", "esn_accuracy"],
            2,
        );
    }

    #[test]
    fn digester_anderson_study_has_timeseries() {
        let (scenario, _) = digester_anderson_study();
        let channels: Vec<&DataChannel> = scenario
            .ecosystem
            .primals
            .iter()
            .flat_map(|n| n.data_channels.iter())
            .collect();
        let has_ts = channels
            .iter()
            .any(|c| matches!(c, DataChannel::TimeSeries { .. }));
        assert!(has_ts, "digester×Anderson should have TimeSeries");
    }

    #[test]
    fn digester_anderson_study_json_roundtrips() {
        let (scenario, edges) = digester_anderson_study();
        assert_json_roundtrips(&scenario, &edges);
    }

    #[test]
    fn isomorphic_reservoir_study_structure() {
        let (scenario, edges) = isomorphic_reservoir_study();
        assert_study_invariants(
            &scenario,
            &edges,
            &["reservoir_spectral", "universality_metrics"],
            1,
        );
    }

    #[test]
    fn isomorphic_reservoir_study_has_spectrum_and_bar() {
        let (scenario, _) = isomorphic_reservoir_study();
        let channels: Vec<&DataChannel> = scenario
            .ecosystem
            .primals
            .iter()
            .flat_map(|n| n.data_channels.iter())
            .collect();
        let has_sp = channels
            .iter()
            .any(|c| matches!(c, DataChannel::Spectrum { .. }));
        let has_bar = channels
            .iter()
            .any(|c| matches!(c, DataChannel::Bar { .. }));
        assert!(has_sp, "isomorphic reservoir should have Spectrum");
        assert!(has_bar, "isomorphic reservoir should have Bar");
    }

    #[test]
    fn isomorphic_reservoir_study_json_roundtrips() {
        let (scenario, edges) = isomorphic_reservoir_study();
        assert_json_roundtrips(&scenario, &edges);
    }

    #[test]
    fn wdm_ensemble_qs_study_structure() {
        let (scenario, edges) = wdm_ensemble_qs_study();
        assert_study_invariants(
            &scenario,
            &edges,
            &["ensemble_disagreement", "anderson_phase", "qs_dynamics"],
            2,
        );
    }

    #[test]
    fn wdm_ensemble_qs_study_has_gauge() {
        let (scenario, _) = wdm_ensemble_qs_study();
        let channels: Vec<&DataChannel> = scenario
            .ecosystem
            .primals
            .iter()
            .flat_map(|n| n.data_channels.iter())
            .collect();
        let has_gauge = channels
            .iter()
            .any(|c| matches!(c, DataChannel::Gauge { .. }));
        assert!(has_gauge, "WDM ensemble QS should have Gauge");
    }

    #[test]
    fn wdm_ensemble_qs_study_json_roundtrips() {
        let (scenario, edges) = wdm_ensemble_qs_study();
        assert_json_roundtrips(&scenario, &edges);
    }

    #[test]
    fn introgression_nn_study_structure() {
        let (scenario, edges) = introgression_nn_study();
        assert_study_invariants(&scenario, &edges, &["nn_observations", "hmm_detection"], 1);
    }

    #[test]
    fn introgression_nn_study_has_heatmap_and_gauge() {
        let (scenario, _) = introgression_nn_study();
        let channels: Vec<&DataChannel> = scenario
            .ecosystem
            .primals
            .iter()
            .flat_map(|n| n.data_channels.iter())
            .collect();
        let has_hm = channels
            .iter()
            .any(|c| matches!(c, DataChannel::Heatmap { .. }));
        let has_gauge = channels
            .iter()
            .any(|c| matches!(c, DataChannel::Gauge { .. }));
        assert!(has_hm, "introgression NN should have Heatmap");
        assert!(has_gauge, "introgression NN should have Gauge");
    }

    #[test]
    fn introgression_nn_study_json_roundtrips() {
        let (scenario, edges) = introgression_nn_study();
        assert_json_roundtrips(&scenario, &edges);
    }

    #[test]
    fn attention_anderson_study_structure() {
        let (scenario, edges) = attention_anderson_study();
        assert_study_invariants(
            &scenario,
            &edges,
            &["attention_quality", "spectral_localization"],
            1,
        );
    }

    #[test]
    fn attention_anderson_study_has_spectrum() {
        let (scenario, _) = attention_anderson_study();
        let channels: Vec<&DataChannel> = scenario
            .ecosystem
            .primals
            .iter()
            .flat_map(|n| n.data_channels.iter())
            .collect();
        let has_sp = channels
            .iter()
            .any(|c| matches!(c, DataChannel::Spectrum { .. }));
        assert!(has_sp, "attention Anderson should have Spectrum");
    }

    #[test]
    fn attention_anderson_study_json_roundtrips() {
        let (scenario, edges) = attention_anderson_study();
        assert_json_roundtrips(&scenario, &edges);
    }
}
