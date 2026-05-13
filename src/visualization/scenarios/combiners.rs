// SPDX-License-Identifier: AGPL-3.0-or-later

//! Study combiners that merge per-domain scenarios into multi-track graphs.
//!
//! [`full_study`] merges all 21 tracks with cross-track edges.
//! [`composition_study`] merges the 5 novel composition experiments.
//! [`scenario_with_edges_json`] serializes any scenario + edges to JSON.

use super::super::types::{NeuralScenario, ScenarioEdge};
#[cfg(feature = "barracuda")]
use super::scaffold::{edge, scaffold};
#[cfg(not(feature = "barracuda"))]
use super::scaffold::scaffold;
#[cfg(not(feature = "barracuda"))]
use super::{
    folding_study, game_theory_study, hmm_study, immunological_study, industry_coverage_study,
    introgression_nn_study, kokkos_parity_study, population_study, search_study, streaming_io_study,
};
#[cfg(feature = "barracuda")]
use super::{
    attention_anderson_study, coordination_study, digester_anderson_study, folding_study,
    game_theory_study, glucose_study, hmm_study, immunological_study, industry_coverage_study,
    introgression_nn_study, isomorphic_reservoir_study, kokkos_parity_study, loss_landscape_study,
    population_study, provenance_study, search_study, spectral_study, streaming_io_study,
    training_study, wdm_ensemble_qs_study, wdm_study,
};

fn merge_tracks(
    scenario: &mut NeuralScenario,
    tracks: Vec<(NeuralScenario, Vec<ScenarioEdge>)>,
    offsets: &[(f64, f64)],
) -> Vec<ScenarioEdge> {
    let mut all_edges = Vec::new();
    for ((track, mut edges), offset) in tracks.into_iter().zip(offsets) {
        for mut n in track.ecosystem.primals {
            n.position.x += offset.0;
            n.position.y += offset.1;
            scenario.ecosystem.primals.push(n);
        }
        all_edges.append(&mut edges);
    }
    all_edges
}

/// Build a combined all-tracks scenario for the complete neuralSpring study.
///
/// Merges all 21 tracks into a single graph with cross-track edges: the
/// original 16 tracks plus 5 novel composition experiments.
#[must_use]
pub fn full_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    #[cfg(feature = "barracuda")]
    {
        let tracks: Vec<(NeuralScenario, Vec<ScenarioEdge>)> = vec![
            spectral_study(),
            training_study(),
            coordination_study(),
            provenance_study(),
            folding_study(),
            hmm_study(),
            game_theory_study(),
            wdm_study(),
            glucose_study(),
            immunological_study(),
            population_study(),
            loss_landscape_study(),
            search_study(),
            streaming_io_study(),
            kokkos_parity_study(),
            industry_coverage_study(),
            digester_anderson_study(),
            isomorphic_reservoir_study(),
            wdm_ensemble_qs_study(),
            introgression_nn_study(),
            attention_anderson_study(),
        ];

        let mut s = scaffold(
            "neuralSpring Complete Study",
            "All 21 tracks: 16 original + 5 novel compositions (Digester×Anderson, \
             Isomorphic Reservoir, WDM Ensemble QS, HMM Introgression NN, Attention Anderson)",
        );

        let offsets: [(f64, f64); 21] = [
            (0.0, 0.0), (0.0, 500.0), (600.0, 0.0), (600.0, 500.0),
            (300.0, 800.0), (900.0, 0.0), (900.0, 500.0), (1200.0, 0.0),
            (1200.0, 500.0), (1500.0, 0.0), (1500.0, 500.0), (300.0, 1200.0),
            (1800.0, 0.0), (1800.0, 500.0), (2100.0, 0.0), (2100.0, 500.0),
            (0.0, 1600.0), (600.0, 1600.0), (1200.0, 1600.0),
            (1800.0, 1600.0), (2400.0, 1600.0),
        ];

        let mut all_edges = merge_tracks(&mut s, tracks, &offsets);
        all_edges.extend(cross_track_edges());

        return (s, all_edges);
    }

    #[cfg(not(feature = "barracuda"))]
    {
        let tracks: Vec<(NeuralScenario, Vec<ScenarioEdge>)> = vec![
            folding_study(),
            hmm_study(),
            game_theory_study(),
            immunological_study(),
            population_study(),
            search_study(),
            streaming_io_study(),
            kokkos_parity_study(),
            industry_coverage_study(),
            introgression_nn_study(),
        ];

        let mut s = scaffold(
            "neuralSpring Complete Study (IPC / CPU tier)",
            "Subset of study tracks built without linking BarraCUDA (no spectral, training WDM, or GPU shaders).",
        );

        let offsets: [(f64, f64); 9] = [
            (300.0, 800.0), (900.0, 0.0), (900.0, 500.0),
            (1500.0, 0.0), (1500.0, 500.0), (1800.0, 0.0),
            (1800.0, 500.0), (2100.0, 0.0), (2100.0, 500.0),
        ];

        let all_edges = merge_tracks(&mut s, tracks, &offsets);
        (s, all_edges)
    }
}

/// Build the composition-only study: all 5 novel experiments in one graph.
///
/// Shows the isomorphic connections between composition experiments and links
/// back to the foundational spectral and HMM tracks they compose.
#[must_use]
pub fn composition_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    #[cfg(feature = "barracuda")]
    {
        let tracks: Vec<(NeuralScenario, Vec<ScenarioEdge>)> = vec![
            digester_anderson_study(),
            isomorphic_reservoir_study(),
            wdm_ensemble_qs_study(),
            introgression_nn_study(),
            attention_anderson_study(),
        ];

        let mut s = scaffold(
            "neuralSpring Novel Compositions",
            "5 composition experiments: Digester×Anderson coupling, Isomorphic reservoir \
             ensemble, WDM ensemble quorum sensing, HMM introgression on NN layers, \
             Attention Anderson spectral analysis",
        );

        let offsets: [(f64, f64); 5] = [
            (0.0, 0.0),
            (600.0, 0.0),
            (1200.0, 0.0),
            (0.0, 600.0),
            (600.0, 600.0),
        ];

        let mut all_edges = merge_tracks(&mut s, tracks, &offsets);
        all_edges.extend(composition_cross_edges());

        return (s, all_edges);
    }

    #[cfg(not(feature = "barracuda"))]
    {
        let tracks = vec![introgression_nn_study()];
        let mut s = scaffold(
            "neuralSpring Novel Compositions (IPC / CPU tier)",
            "Subset of composition tracks available without BarraCUDA: introgression NN only.",
        );

        let all_edges = merge_tracks(&mut s, tracks, &[(0.0, 0.0)]);
        (s, all_edges)
    }
}

/// Edges linking composition experiments to each other via shared physics.
#[cfg(feature = "barracuda")]
fn composition_cross_edges() -> Vec<ScenarioEdge> {
    vec![
        edge(
            "anderson_coupling",
            "anderson_phase",
            "shared Anderson physics",
        ),
        edge(
            "reservoir_spectral",
            "attention_quality",
            "shared spectral analysis",
        ),
        edge(
            "nn_observations",
            "reservoir_spectral",
            "NN weights → spectral universality",
        ),
        edge(
            "ensemble_disagreement",
            "digester_community",
            "surrogate variance ↔ community diversity",
        ),
    ]
}

/// Inter-track edges that connect nodes across different study tracks.
#[cfg(feature = "barracuda")]
fn cross_track_edges() -> Vec<ScenarioEdge> {
    vec![
        edge(
            "spectral_analysis",
            "training_trajectory",
            "spectral diagnostics → training",
        ),
        edge(
            "spectral_analysis",
            "agent_coordination",
            "spectral theory → multi-agent",
        ),
        edge(
            "spectral_analysis",
            "folding_primitives",
            "spectral metrics inform folding",
        ),
        edge(
            "shader_provenance",
            "spectral_analysis",
            "shaders implement spectral ops",
        ),
        edge(
            "hessian_analysis",
            "spectral_analysis",
            "loss Hessian ↔ weight spectra",
        ),
        edge(
            "hmm_forward",
            "meta_pop",
            "phylogenetics → population structure",
        ),
        edge(
            "replicator_dynamics",
            "qs_cooperation",
            "classical → spatial dynamics",
        ),
        edge(
            "glucose_prediction",
            "training_trajectory",
            "LSTM training → glucose forecast",
        ),
        edge(
            "immuno_anderson",
            "wdm_transport",
            "Anderson localization ↔ WDM transport",
        ),
        edge(
            "fastq_quality",
            "search_pipeline",
            "parsed reads → BLAST search",
        ),
        edge(
            "fasta_lengths",
            "kmer_index",
            "FASTA database → k-mer index",
        ),
        edge(
            "parity_overview",
            "coverage_overview",
            "GPU performance → industry readiness",
        ),
        edge(
            "anderson_sweep",
            "digester_community",
            "Anderson theory → digester coupling",
        ),
        edge(
            "spectral_analysis",
            "reservoir_spectral",
            "spectral methods → isomorphic thesis",
        ),
        edge(
            "wdm_transport",
            "ensemble_disagreement",
            "WDM surrogates → ensemble QS",
        ),
        edge(
            "hmm_forward",
            "nn_observations",
            "HMM phylo → NN introgression",
        ),
        edge(
            "spectral_analysis",
            "attention_quality",
            "Anderson theory → attention spectral",
        ),
    ]
}

/// Serialize a scenario + edges to pretty JSON.
///
/// Edges are merged into the scenario's `edges` field for a single JSON output.
///
/// # Panics
///
/// Cannot panic — all types implement `Serialize`.
#[must_use]
pub fn scenario_with_edges_json(scenario: &NeuralScenario, edges: &[ScenarioEdge]) -> String {
    let mut merged = scenario.clone();
    merged.edges.extend_from_slice(edges);
    serde_json::to_string_pretty(&merged).unwrap_or_default()
}

#[cfg(all(test, feature = "barracuda"))]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    fn full_study_all_nodes() {
        let (scenario, edges) = full_study();
        let ids: std::collections::HashSet<&str> = scenario
            .ecosystem
            .primals
            .iter()
            .map(|n| n.id.as_str())
            .collect();

        assert!(ids.contains("anderson_sweep"));
        assert!(ids.contains("spectral_analysis"));
        assert!(ids.contains("hessian_eigen"));
        assert!(ids.contains("training_trajectory"));
        assert!(ids.contains("agent_coordination"));
        assert!(ids.contains("shader_provenance"));
        assert!(ids.contains("folding_primitives"));
        assert!(ids.contains("search_pipeline"));
        assert!(ids.contains("kmer_index"));
        assert!(ids.contains("fastq_quality"));
        assert!(ids.contains("fasta_lengths"));
        assert!(ids.contains("vcf_variants"));
        assert!(ids.contains("parity_overview"));
        assert!(ids.contains("coverage_overview"));

        assert_eq!(
            ids.len(),
            scenario.ecosystem.primals.len(),
            "no duplicate IDs"
        );
        assert!(
            edges.len() >= 17,
            "full study should have >= 17 edges (12 original + 5 composition cross)"
        );
    }

    #[test]
    fn full_study_cross_track_edges() {
        let (_, edges) = full_study();
        let edge_pairs: std::collections::HashSet<(&str, &str)> = edges
            .iter()
            .map(|e| (e.from.as_str(), e.to.as_str()))
            .collect();
        assert!(
            edge_pairs.contains(&("spectral_analysis", "training_trajectory")),
            "cross-track: spectral → training"
        );
        assert!(
            edge_pairs.contains(&("spectral_analysis", "agent_coordination")),
            "cross-track: spectral → coordination"
        );
        assert!(
            edge_pairs.contains(&("shader_provenance", "spectral_analysis")),
            "cross-track: provenance → spectral"
        );
    }

    #[test]
    fn full_study_json_roundtrips() {
        let (scenario, edges) = full_study();
        let json = scenario_with_edges_json(&scenario, &edges);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON must be valid");
        assert!(parsed.get("name").is_some());
        assert!(parsed.get("ecosystem").is_some());
    }

    #[test]
    fn full_study_includes_composition_nodes() {
        let (scenario, _) = full_study();
        let ids: std::collections::HashSet<&str> = scenario
            .ecosystem
            .primals
            .iter()
            .map(|n| n.id.as_str())
            .collect();

        assert!(
            ids.contains("digester_community"),
            "full study has digester"
        );
        assert!(
            ids.contains("reservoir_spectral"),
            "full study has reservoir"
        );
        assert!(
            ids.contains("ensemble_disagreement"),
            "full study has ensemble"
        );
        assert!(
            ids.contains("nn_observations"),
            "full study has introgression"
        );
        assert!(
            ids.contains("attention_quality"),
            "full study has attention"
        );
    }

    #[test]
    fn composition_study_structure() {
        let (scenario, edges) = composition_study();
        let ids: std::collections::HashSet<&str> = scenario
            .ecosystem
            .primals
            .iter()
            .map(|n| n.id.as_str())
            .collect();

        assert!(ids.contains("digester_community"));
        assert!(ids.contains("reservoir_spectral"));
        assert!(ids.contains("ensemble_disagreement"));
        assert!(ids.contains("nn_observations"));
        assert!(ids.contains("attention_quality"));
        assert_eq!(
            ids.len(),
            scenario.ecosystem.primals.len(),
            "no duplicate IDs"
        );
        assert!(
            edges.len() >= 10,
            "composition study should have >= 10 edges"
        );
    }

    #[test]
    fn composition_study_json_roundtrips() {
        let (scenario, edges) = composition_study();
        let json = scenario_with_edges_json(&scenario, &edges);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON must be valid");
        assert!(parsed.get("name").is_some());
        assert!(parsed.get("ecosystem").is_some());
    }

    #[test]
    fn scenario_with_edges_json_valid() {
        let (scenario, edges) = super::super::spectral_study();
        let json = scenario_with_edges_json(&scenario, &edges);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert!(parsed["name"].as_str().is_some());
        assert!(parsed["ecosystem"]["primals"].is_array());
        assert!(parsed["edges"].is_array());
        assert_eq!(parsed["edges"].as_array().expect("edges").len(), 2);
    }
}
