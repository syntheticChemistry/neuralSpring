// SPDX-License-Identifier: AGPL-3.0-or-later

//! petalTongue-compatible type definitions for neuralSpring visualization.
//!
//! Mirrors healthSpring's schema (`barracuda/src/visualization/types.rs`)
//! and extends [`DataChannel`] with all 8 petalTongue `DataBinding` types.
//! neuralSpring uses `Spectrum`, `Heatmap`, and `Scatter3D` for eigenvalue
//! spectra, attention matrices, and phase-space plots.

use serde::Serialize;

/// A typed data channel attached to a scenario node.
///
/// petalTongue renders each variant with a domain-appropriate chart:
/// `TimeSeries` → line chart, `Gauge` → arc/bar, `Spectrum` → frequency
/// plot, `Heatmap` → color grid, `Scatter3D` → 3D point cloud.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "channel_type")]
pub enum DataChannel {
    /// Time series or parametric curve: paired samples for a line chart.
    #[serde(rename = "timeseries")]
    TimeSeries {
        /// Unique identifier for this channel.
        id: String,
        /// Human-readable display name for the chart.
        label: String,
        /// Label for the horizontal axis.
        x_label: String,
        /// Label for the vertical axis.
        y_label: String,
        /// Measurement unit string (e.g. "eV", "Hz").
        unit: String,
        /// Independent variable coordinates (e.g. time or step index).
        x_values: Vec<f64>,
        /// Dependent samples aligned with `x_values`.
        y_values: Vec<f64>,
    },
    /// Histogram or distribution with summary statistics and a reference overlay.
    #[serde(rename = "distribution")]
    Distribution {
        /// Unique identifier for this channel.
        id: String,
        /// Human-readable display name for the chart.
        label: String,
        /// Measurement unit string (e.g. "eV", "Hz").
        unit: String,
        /// Sample values or bin heights (semantics depend on binding).
        values: Vec<f64>,
        /// Mean of the distribution.
        mean: f64,
        /// Standard deviation of the distribution.
        std: f64,
        /// Reference value for overlay (e.g. baseline or target).
        comparison_value: f64,
    },
    /// Categorical bar chart: one value per category.
    #[serde(rename = "bar")]
    Bar {
        /// Unique identifier for this channel.
        id: String,
        /// Human-readable display name for the chart.
        label: String,
        /// Category names for the categorical axis.
        categories: Vec<String>,
        /// Bar heights or values per category.
        values: Vec<f64>,
        /// Measurement unit string (e.g. "eV", "Hz").
        unit: String,
    },
    /// Gauge: current reading against scale and threshold bands.
    #[serde(rename = "gauge")]
    Gauge {
        /// Unique identifier for this channel.
        id: String,
        /// Human-readable display name for the gauge.
        label: String,
        /// Current value shown on the gauge.
        value: f64,
        /// Lower end of the gauge scale.
        min: f64,
        /// Upper end of the gauge scale.
        max: f64,
        /// Measurement unit string (e.g. "eV", "Hz").
        unit: String,
        /// Inclusive [lo, hi] range considered normal operation.
        normal_range: [f64; 2],
        /// Inclusive [lo, hi] range that triggers warning styling.
        warning_range: [f64; 2],
    },
    /// 2D color grid with labeled rows and columns.
    #[serde(rename = "heatmap")]
    Heatmap {
        /// Unique identifier for this channel.
        id: String,
        /// Human-readable display name for the heatmap.
        label: String,
        /// Column labels (horizontal axis).
        x_labels: Vec<String>,
        /// Row labels (vertical axis).
        y_labels: Vec<String>,
        /// Flattened row-major matrix of cell values.
        values: Vec<f64>,
        /// Measurement unit string (e.g. "eV", "Hz").
        unit: String,
    },
    /// 3D point cloud with optional short labels per point.
    #[serde(rename = "scatter3d")]
    Scatter3D {
        /// Unique identifier for this channel.
        id: String,
        /// Human-readable display name for the chart.
        label: String,
        /// X coordinates for each point.
        x: Vec<f64>,
        /// Y coordinates for each point.
        y: Vec<f64>,
        /// Z coordinates for each point.
        z: Vec<f64>,
        /// Per-point labels aligned with `x`/`y`/`z`.
        point_labels: Vec<String>,
        /// Measurement unit string (e.g. "eV", "Hz").
        unit: String,
    },
    /// Scalar field sampled on a 2D grid (heatmap or contour style).
    #[serde(rename = "fieldmap")]
    FieldMap {
        /// Unique identifier for this channel.
        id: String,
        /// Human-readable display name for the field view.
        label: String,
        /// Horizontal grid coordinates.
        grid_x: Vec<f64>,
        /// Vertical grid coordinates.
        grid_y: Vec<f64>,
        /// Field samples on the `grid_x` × `grid_y` mesh (flattened).
        values: Vec<f64>,
        /// Measurement unit string (e.g. "eV", "Hz").
        unit: String,
    },
    /// Frequency-domain spectrum: amplitudes at each frequency.
    #[serde(rename = "spectrum")]
    Spectrum {
        /// Unique identifier for this channel.
        id: String,
        /// Human-readable display name for the spectrum.
        label: String,
        /// Frequency bin centers or line positions.
        frequencies: Vec<f64>,
        /// Amplitude or power at each frequency.
        amplitudes: Vec<f64>,
        /// Measurement unit string (e.g. "eV", "Hz").
        unit: String,
    },
}

/// Quality threshold for petalTongue's threshold coloring.
///
/// Unlike healthSpring's `ClinicalRange`, neuralSpring thresholds indicate
/// spectral/ML quality levels (e.g. Extended/Critical/Localized phase).
#[derive(Debug, Clone, Serialize)]
pub struct ThresholdRange {
    /// Human-readable label for this band (e.g. quality tier or phase).
    pub label: String,
    /// Lower bound of the inclusive range.
    pub min: f64,
    /// Upper bound of the inclusive range.
    pub max: f64,
    /// Operational status string for styling (e.g. phase name).
    pub status: String,
}

/// A node in the scenario graph.
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioNode {
    /// Unique identifier for this node.
    pub id: String,
    /// Display name for the node.
    pub name: String,
    /// Type discriminator for the node (serialized as `type`).
    #[serde(rename = "type")]
    pub node_type: String,
    /// Grouping or domain family for layout and styling.
    pub family: String,
    /// Operational status string.
    pub status: String,
    /// Health score (0–100).
    pub health: u8,
    /// Confidence score (0–100).
    pub confidence: u8,
    /// 2D layout position on the graph canvas.
    pub position: Position,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    /// Capability identifiers this node advertises.
    pub capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    /// Data streams bound to this node for charts.
    pub data_channels: Vec<DataChannel>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    /// Quality threshold bands for spectral/ML coloring.
    pub thresholds: Vec<ThresholdRange>,
}

/// 2D coordinates for laying out a node on the scenario graph.
#[derive(Debug, Clone, Serialize)]
pub struct Position {
    /// Horizontal position in layout space.
    pub x: f64,
    /// Vertical position in layout space.
    pub y: f64,
}

/// An edge in the scenario graph.
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioEdge {
    /// Source node id.
    pub from: String,
    /// Target node id.
    pub to: String,
    /// Relationship type (e.g. data flow or dependency).
    pub edge_type: String,
    /// Human-readable display name for the edge.
    pub label: String,
}

/// Complete scenario — petalTongue-compatible with neuralSpring extensions.
#[derive(Debug, Clone, Serialize)]
pub struct NeuralScenario {
    /// Scenario title.
    pub name: String,
    /// Longer description for tooltips or documentation.
    pub description: String,
    /// Schema or bundle version string.
    pub version: String,
    /// Run or visualization mode identifier.
    pub mode: String,
    /// Capability and scheduling requirements for sensory pipelines.
    pub sensory_config: SensoryConfig,
    /// Theme, animation, performance, and panel options for the viewer.
    pub ui_config: UiConfig,
    /// Primal nodes and embedded graph state for the ecosystem.
    pub ecosystem: Ecosystem,
    /// Whether the neural/API integration layer is enabled.
    pub neural_api: NeuralApi,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    /// Directed edges between scenario nodes (empty if omitted).
    pub edges: Vec<ScenarioEdge>,
}

/// Top-level ecosystem container for primal scenario nodes.
#[derive(Debug, Clone, Serialize)]
pub struct Ecosystem {
    /// Primal nodes in this ecosystem.
    pub primals: Vec<ScenarioNode>,
}

/// Sensory pipeline requirements: capabilities and scheduling hint.
#[derive(Debug, Clone, Serialize)]
pub struct SensoryConfig {
    /// Required output/input capability bindings.
    pub required_capabilities: CapReqs,
    /// Optional output/input capability bindings.
    pub optional_capabilities: CapReqs,
    /// Scheduling hint for workload complexity.
    pub complexity_hint: String,
}

/// Output and input capability lists for a binding requirement.
#[derive(Debug, Clone, Serialize)]
pub struct CapReqs {
    /// Advertised or required output capability ids.
    pub outputs: Vec<String>,
    /// Advertised or required input capability ids.
    pub inputs: Vec<String>,
}

/// Viewer UI: theme, motion, performance, and layout toggles.
#[derive(Debug, Clone, Serialize)]
pub struct UiConfig {
    /// Color theme identifier for the viewer.
    pub theme: String,
    /// Feature flags for motion and transition effects.
    pub animations: Animations,
    /// Frame pacing and hardware acceleration settings.
    pub performance: Performance,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Per-panel visibility overrides when set.
    pub show_panels: Option<ShowPanels>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Enables ecosystem awakening animation when `Some(true)`.
    pub awakening_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Initial zoom level string when set.
    pub initial_zoom: Option<String>,
}

/// Panel visibility for petalTongue scenario config.
#[expect(
    clippy::struct_excessive_bools,
    reason = "matches petalTongue JSON schema — each field serializes as a named boolean key"
)]
#[derive(Debug, Clone, Serialize)]
pub struct ShowPanels {
    /// Whether the left sidebar panel is shown.
    pub left_sidebar: bool,
    /// Whether the right sidebar panel is shown.
    pub right_sidebar: bool,
    /// Whether the top menu bar is shown.
    pub top_menu: bool,
    /// Whether the system dashboard panel is shown.
    pub system_dashboard: bool,
    /// Whether the audio panel is shown.
    pub audio_panel: bool,
    /// Whether the trust dashboard panel is shown.
    pub trust_dashboard: bool,
    /// Whether the proprioception panel is shown.
    pub proprioception: bool,
    /// Whether the graph statistics panel is shown.
    pub graph_stats: bool,
}

/// Viewer animation toggles for motion and visual effects.
#[expect(clippy::struct_excessive_bools, reason = "matches petalTongue schema")]
#[derive(Debug, Clone, Serialize)]
pub struct Animations {
    /// Whether animation effects are enabled globally.
    pub enabled: bool,
    /// Whether nodes use a breathing motion effect.
    pub breathing_nodes: bool,
    /// Whether connection edges show pulse animations.
    pub connection_pulses: bool,
    /// Whether view transitions use smooth interpolation.
    pub smooth_transitions: bool,
    /// Whether milestone celebration effects are enabled.
    pub celebration_effects: bool,
}

/// Target frame rate and hardware acceleration for the viewer.
#[derive(Debug, Clone, Serialize)]
pub struct Performance {
    /// Desired frames per second for rendering.
    pub target_fps: u32,
    /// Whether vertical sync is enabled.
    pub vsync: bool,
    /// Whether hardware-accelerated rendering is enabled.
    pub hardware_acceleration: bool,
}

/// Flags for enabling the neural/API integration layer in the scenario.
#[derive(Debug, Clone, Serialize)]
pub struct NeuralApi {
    /// Whether the neural/API integration layer is enabled.
    pub enabled: bool,
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test serialization roundtrips")]
mod tests {
    use super::*;

    #[test]
    fn timeseries_serializes_correctly() {
        let ch = DataChannel::TimeSeries {
            id: "ts1".into(),
            label: "Loss curve".into(),
            x_label: "Epoch".into(),
            y_label: "Loss".into(),
            unit: "dimensionless".into(),
            x_values: vec![0.0, 1.0, 2.0],
            y_values: vec![1.0, 0.5, 0.25],
        };
        let json = serde_json::to_value(&ch).expect("serialize");
        assert_eq!(json["channel_type"], "timeseries");
        assert_eq!(json["id"], "ts1");
        assert_eq!(json["x_values"].as_array().expect("array").len(), 3);
    }

    #[test]
    fn gauge_serializes_with_ranges() {
        let ch = DataChannel::Gauge {
            id: "g1".into(),
            label: "Temperature".into(),
            value: 42.0,
            min: 0.0,
            max: 100.0,
            unit: "C".into(),
            normal_range: [20.0, 60.0],
            warning_range: [60.0, 80.0],
        };
        let json = serde_json::to_value(&ch).expect("serialize");
        assert_eq!(json["channel_type"], "gauge");
        assert_eq!(json["value"], 42.0);
    }

    #[test]
    fn heatmap_serializes_grid() {
        let ch = DataChannel::Heatmap {
            id: "h1".into(),
            label: "Attention".into(),
            x_labels: vec!["a".into(), "b".into()],
            y_labels: vec!["c".into(), "d".into()],
            values: vec![1.0, 2.0, 3.0, 4.0],
            unit: "weight".into(),
        };
        let json = serde_json::to_value(&ch).expect("serialize");
        assert_eq!(json["channel_type"], "heatmap");
        assert_eq!(json["values"].as_array().expect("array").len(), 4);
    }

    #[test]
    fn scatter3d_serializes() {
        let ch = DataChannel::Scatter3D {
            id: "s3d".into(),
            label: "Phase space".into(),
            x: vec![1.0],
            y: vec![2.0],
            z: vec![3.0],
            point_labels: vec!["origin".into()],
            unit: "eV".into(),
        };
        let json = serde_json::to_value(&ch).expect("serialize");
        assert_eq!(json["channel_type"], "scatter3d");
    }

    #[test]
    fn spectrum_serializes() {
        let ch = DataChannel::Spectrum {
            id: "sp1".into(),
            label: "Eigenvalues".into(),
            frequencies: vec![1.0, 2.0, 3.0],
            amplitudes: vec![0.5, 0.8, 0.3],
            unit: "eV".into(),
        };
        let json = serde_json::to_value(&ch).expect("serialize");
        assert_eq!(json["channel_type"], "spectrum");
    }

    #[test]
    fn threshold_range_construction() {
        let tr = ThresholdRange {
            label: "Extended".into(),
            min: 0.0,
            max: 1.0,
            status: "healthy".into(),
        };
        assert_eq!(tr.label, "Extended");
        assert!(tr.max > tr.min);
    }

    #[test]
    fn scenario_node_skips_empty_vecs() {
        let node = ScenarioNode {
            id: "n1".into(),
            name: "Test".into(),
            node_type: "primal".into(),
            family: "test".into(),
            status: "healthy".into(),
            health: 100,
            confidence: 95,
            position: Position { x: 0.0, y: 0.0 },
            capabilities: vec![],
            data_channels: vec![],
            thresholds: vec![],
        };
        let json = serde_json::to_value(&node).expect("serialize");
        assert!(json.get("capabilities").is_none());
        assert!(json.get("data_channels").is_none());
        assert!(json.get("thresholds").is_none());
    }

    #[test]
    fn neural_api_toggle() {
        let api = NeuralApi { enabled: true };
        let json = serde_json::to_value(&api).expect("serialize");
        assert_eq!(json["enabled"], true);
    }
}
