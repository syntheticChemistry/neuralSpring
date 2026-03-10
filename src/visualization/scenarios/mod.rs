// SPDX-License-Identifier: AGPL-3.0-or-later

//! Per-domain petalTongue scenario builders for neuralSpring.
//!
//! Each builder calls real neuralSpring computation and wraps outputs in
//! [`DataChannel`] / [`ScenarioNode`] / [`NeuralScenario`] so petalTongue
//! can render them directly.

mod attention_anderson;
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
mod search_results;
mod spectral;
mod streaming_io;
mod training;
mod wdm;
mod wdm_ensemble_qs;

use super::types::{
    Animations, CapReqs, DataChannel, Ecosystem, NeuralApi, NeuralScenario, Performance, Position,
    ScenarioEdge, ScenarioNode, SensoryConfig, ThresholdRange, UiConfig,
};

pub use attention_anderson::attention_anderson_study;
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

fn scaffold(name: &str, description: &str) -> NeuralScenario {
    NeuralScenario {
        name: name.into(),
        description: description.into(),
        version: "1.0.0".into(),
        mode: "research".into(),
        sensory_config: SensoryConfig {
            required_capabilities: CapReqs {
                outputs: vec!["visual".into()],
                inputs: vec![],
            },
            optional_capabilities: CapReqs {
                outputs: vec!["audio".into()],
                inputs: vec!["pointer".into(), "keyboard".into()],
            },
            complexity_hint: "standard".into(),
        },
        ui_config: UiConfig {
            theme: crate::config::PETALTONGUE_THEME.into(),
            animations: Animations {
                enabled: true,
                breathing_nodes: true,
                connection_pulses: true,
                smooth_transitions: true,
                celebration_effects: false,
            },
            performance: Performance {
                target_fps: 60,
                vsync: true,
                hardware_acceleration: true,
            },
            show_panels: None,
            awakening_enabled: Some(true),
            initial_zoom: None,
        },
        ecosystem: Ecosystem { primals: vec![] },
        neural_api: NeuralApi { enabled: false },
        edges: Vec::new(),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal helper — all args have clear roles"
)]
pub(crate) fn gauge(
    id: &str,
    label: &str,
    value: f64,
    min: f64,
    max: f64,
    unit: &str,
    normal: [f64; 2],
    warn: [f64; 2],
) -> DataChannel {
    DataChannel::Gauge {
        id: id.into(),
        label: label.into(),
        value,
        min,
        max,
        unit: unit.into(),
        normal_range: normal,
        warning_range: warn,
    }
}

pub(crate) fn timeseries(
    id: &str,
    label: &str,
    x_label: &str,
    y_label: &str,
    unit: &str,
    xs: Vec<f64>,
    ys: Vec<f64>,
) -> DataChannel {
    DataChannel::TimeSeries {
        id: id.into(),
        label: label.into(),
        x_label: x_label.into(),
        y_label: y_label.into(),
        unit: unit.into(),
        x_values: xs,
        y_values: ys,
    }
}

pub(crate) fn bar(
    id: &str,
    label: &str,
    cats: Vec<String>,
    vals: Vec<f64>,
    unit: &str,
) -> DataChannel {
    DataChannel::Bar {
        id: id.into(),
        label: label.into(),
        categories: cats,
        values: vals,
        unit: unit.into(),
    }
}

pub(crate) fn spectrum(
    id: &str,
    label: &str,
    unit: &str,
    frequencies: Vec<f64>,
    amplitudes: Vec<f64>,
) -> DataChannel {
    DataChannel::Spectrum {
        id: id.into(),
        label: label.into(),
        frequencies,
        amplitudes,
        unit: unit.into(),
    }
}

pub(crate) fn scatter3d(
    id: &str,
    label: &str,
    unit: &str,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    point_labels: Vec<String>,
) -> DataChannel {
    DataChannel::Scatter3D {
        id: id.into(),
        label: label.into(),
        x,
        y,
        z,
        point_labels,
        unit: unit.into(),
    }
}

pub(crate) fn heatmap(
    id: &str,
    label: &str,
    x_labels: Vec<String>,
    y_labels: Vec<String>,
    values: Vec<f64>,
    unit: &str,
) -> DataChannel {
    DataChannel::Heatmap {
        id: id.into(),
        label: label.into(),
        x_labels,
        y_labels,
        values,
        unit: unit.into(),
    }
}

pub(crate) fn distribution(
    id: &str,
    label: &str,
    unit: &str,
    values: Vec<f64>,
    mean: f64,
    std: f64,
    comparison_value: f64,
) -> DataChannel {
    DataChannel::Distribution {
        id: id.into(),
        label: label.into(),
        unit: unit.into(),
        values,
        mean,
        std,
        comparison_value,
    }
}

pub(crate) fn fieldmap(
    id: &str,
    label: &str,
    grid_x: Vec<f64>,
    grid_y: Vec<f64>,
    values: Vec<f64>,
    unit: &str,
) -> DataChannel {
    DataChannel::FieldMap {
        id: id.into(),
        label: label.into(),
        grid_x,
        grid_y,
        values,
        unit: unit.into(),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal helper — all args have clear roles"
)]
pub(crate) fn node(
    id: &str,
    name: &str,
    node_type: &str,
    x: f64,
    y: f64,
    caps: &[&str],
    channels: Vec<DataChannel>,
    thresholds: Vec<ThresholdRange>,
) -> ScenarioNode {
    ScenarioNode {
        id: id.into(),
        name: name.into(),
        node_type: node_type.into(),
        family: crate::config::PRIMAL_FAMILY.into(),
        status: "healthy".into(),
        health: 100,
        confidence: 95,
        position: Position { x, y },
        capabilities: caps.iter().map(|s| (*s).into()).collect(),
        data_channels: channels,
        thresholds,
    }
}

pub(crate) fn edge(from: &str, to: &str, label: &str) -> ScenarioEdge {
    ScenarioEdge {
        from: from.into(),
        to: to.into(),
        edge_type: "data-flow".into(),
        label: label.into(),
    }
}

// ---------------------------------------------------------------------------
// Full Study (all 5 tracks combined)
// ---------------------------------------------------------------------------

/// Build a combined all-tracks scenario for the complete neuralSpring study.
///
/// Merges all 21 tracks into a single graph with cross-track edges: the
/// original 16 tracks plus 5 novel composition experiments.
#[must_use]
pub fn full_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
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
        (0.0, 0.0),
        (0.0, 500.0),
        (600.0, 0.0),
        (600.0, 500.0),
        (300.0, 800.0),
        (900.0, 0.0),
        (900.0, 500.0),
        (1200.0, 0.0),
        (1200.0, 500.0),
        (1500.0, 0.0),
        (1500.0, 500.0),
        (300.0, 1200.0),
        (1800.0, 0.0),
        (1800.0, 500.0),
        (2100.0, 0.0),
        (2100.0, 500.0),
        // Composition experiments — row below
        (0.0, 1600.0),
        (600.0, 1600.0),
        (1200.0, 1600.0),
        (1800.0, 1600.0),
        (2400.0, 1600.0),
    ];

    let mut all_edges = Vec::new();
    for ((track, mut edges), offset) in tracks.into_iter().zip(offsets) {
        for mut n in track.ecosystem.primals {
            n.position.x += offset.0;
            n.position.y += offset.1;
            s.ecosystem.primals.push(n);
        }
        all_edges.append(&mut edges);
    }

    all_edges.extend(cross_track_edges());

    (s, all_edges)
}

/// Build the composition-only study: all 5 novel experiments in one graph.
///
/// Shows the isomorphic connections between composition experiments and links
/// back to the foundational spectral and HMM tracks they compose.
#[must_use]
pub fn composition_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
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

    let mut all_edges = Vec::new();
    for ((track, mut edges), offset) in tracks.into_iter().zip(offsets) {
        for mut n in track.ecosystem.primals {
            n.position.x += offset.0;
            n.position.y += offset.1;
            s.ecosystem.primals.push(n);
        }
        all_edges.append(&mut edges);
    }

    all_edges.extend(composition_cross_edges());

    (s, all_edges)
}

/// Edges linking composition experiments to each other via shared physics.
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
        // Composition experiments ↔ foundation tracks
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;

    fn assert_study_invariants(
        scenario: &NeuralScenario,
        edges: &[ScenarioEdge],
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

    fn assert_json_roundtrips(scenario: &NeuralScenario, edges: &[ScenarioEdge]) {
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
        assert!(edges.len() >= 17, "full study should have >= 17 edges (12 original + 5 composition cross)");
    }

    #[test]
    fn full_study_cross_track_edges() {
        let (_, edges) = full_study();
        let edge_pairs: std::collections::HashSet<(String, String)> = edges
            .iter()
            .map(|e| (e.from.clone(), e.to.clone()))
            .collect();
        assert!(
            edge_pairs.contains(&("spectral_analysis".into(), "training_trajectory".into())),
            "cross-track: spectral → training"
        );
        assert!(
            edge_pairs.contains(&("spectral_analysis".into(), "agent_coordination".into())),
            "cross-track: spectral → coordination"
        );
        assert!(
            edge_pairs.contains(&("shader_provenance".into(), "spectral_analysis".into())),
            "cross-track: provenance → spectral"
        );
    }

    #[test]
    fn full_study_json_roundtrips() {
        let (scenario, edges) = full_study();
        assert_json_roundtrips(&scenario, &edges);
    }

    #[test]
    fn scaffold_structure() {
        let (scenario, _) = spectral_study();
        assert!(!scenario.name.is_empty());
        assert!(!scenario.description.is_empty());
        assert_eq!(scenario.version, "1.0.0");
        assert_eq!(scenario.mode, "research");
        assert!(scenario.ui_config.theme.contains("neural"));
        assert!(!scenario.neural_api.enabled);
    }

    #[test]
    fn scenario_with_edges_json_valid() {
        let (scenario, edges) = spectral_study();
        let json = scenario_with_edges_json(&scenario, &edges);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert!(parsed["name"].as_str().is_some());
        assert!(parsed["ecosystem"]["primals"].is_array());
        assert!(parsed["edges"].is_array());
        assert_eq!(parsed["edges"].as_array().expect("edges").len(), 2);
    }

    #[test]
    fn gauge_produces_gauge_channel() {
        let ch = super::gauge(
            "g1",
            "Test",
            50.0,
            0.0,
            100.0,
            "u",
            [20.0, 80.0],
            [10.0, 20.0],
        );
        assert!(matches!(ch, DataChannel::Gauge { .. }));
    }

    #[test]
    fn timeseries_produces_timeseries_channel() {
        let ch = super::timeseries("ts1", "T", "X", "Y", "u", vec![1.0], vec![2.0]);
        assert!(matches!(ch, DataChannel::TimeSeries { .. }));
    }

    #[test]
    fn bar_produces_bar_channel() {
        let ch = super::bar("b1", "B", vec!["A".into()], vec![1.0], "u");
        assert!(matches!(ch, DataChannel::Bar { .. }));
    }

    #[test]
    fn spectrum_produces_spectrum_channel() {
        let ch = super::spectrum("s1", "S", "u", vec![1.0], vec![2.0]);
        assert!(matches!(ch, DataChannel::Spectrum { .. }));
    }

    #[test]
    fn scatter3d_produces_scatter3d_channel() {
        let ch = super::scatter3d(
            "sc1",
            "SC",
            "u",
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec!["a".into()],
        );
        assert!(matches!(ch, DataChannel::Scatter3D { .. }));
    }

    #[test]
    fn node_produces_scenario_node() {
        let n = super::node("n1", "N", "compute", 10.0, 20.0, &["cap1"], vec![], vec![]);
        assert_eq!(n.id, "n1");
        assert_eq!(n.family, crate::config::PRIMAL_FAMILY);
        assert_eq!(n.health, 100);
    }

    #[test]
    fn edge_produces_scenario_edge() {
        let e = super::edge("a", "b", "test");
        assert_eq!(e.from, "a");
        assert_eq!(e.to, "b");
        assert_eq!(e.edge_type, "data-flow");
    }

    // ── New scenario builder tests (S138+) ───────────────────────────────

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

    // ── Composition experiment tests (S143) ────────────────────────────────

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
        assert_study_invariants(
            &scenario,
            &edges,
            &["nn_observations", "hmm_detection"],
            1,
        );
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
        assert_eq!(ids.len(), scenario.ecosystem.primals.len(), "no duplicate IDs");
        assert!(edges.len() >= 10, "composition study should have >= 10 edges");
    }

    #[test]
    fn composition_study_json_roundtrips() {
        let (scenario, edges) = composition_study();
        assert_json_roundtrips(&scenario, &edges);
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

        assert!(ids.contains("digester_community"), "full study has digester");
        assert!(ids.contains("reservoir_spectral"), "full study has reservoir");
        assert!(ids.contains("ensemble_disagreement"), "full study has ensemble");
        assert!(ids.contains("nn_observations"), "full study has introgression");
        assert!(ids.contains("attention_quality"), "full study has attention");
    }
}
