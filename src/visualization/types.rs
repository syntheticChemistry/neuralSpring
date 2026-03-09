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
    #[serde(rename = "timeseries")]
    TimeSeries {
        id: String,
        label: String,
        x_label: String,
        y_label: String,
        unit: String,
        x_values: Vec<f64>,
        y_values: Vec<f64>,
    },
    #[serde(rename = "distribution")]
    Distribution {
        id: String,
        label: String,
        unit: String,
        values: Vec<f64>,
        mean: f64,
        std: f64,
        comparison_value: f64,
    },
    #[serde(rename = "bar")]
    Bar {
        id: String,
        label: String,
        categories: Vec<String>,
        values: Vec<f64>,
        unit: String,
    },
    #[serde(rename = "gauge")]
    Gauge {
        id: String,
        label: String,
        value: f64,
        min: f64,
        max: f64,
        unit: String,
        normal_range: [f64; 2],
        warning_range: [f64; 2],
    },
    #[serde(rename = "heatmap")]
    Heatmap {
        id: String,
        label: String,
        x_labels: Vec<String>,
        y_labels: Vec<String>,
        values: Vec<f64>,
        unit: String,
    },
    #[serde(rename = "scatter3d")]
    Scatter3D {
        id: String,
        label: String,
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<f64>,
        point_labels: Vec<String>,
        unit: String,
    },
    #[serde(rename = "fieldmap")]
    FieldMap {
        id: String,
        label: String,
        grid_x: Vec<f64>,
        grid_y: Vec<f64>,
        values: Vec<f64>,
        unit: String,
    },
    #[serde(rename = "spectrum")]
    Spectrum {
        id: String,
        label: String,
        frequencies: Vec<f64>,
        amplitudes: Vec<f64>,
        unit: String,
    },
}

/// Quality threshold for petalTongue's threshold coloring.
///
/// Unlike healthSpring's `ClinicalRange`, neuralSpring thresholds indicate
/// spectral/ML quality levels (e.g. Extended/Critical/Localized phase).
#[derive(Debug, Clone, Serialize)]
pub struct ThresholdRange {
    pub label: String,
    pub min: f64,
    pub max: f64,
    pub status: String,
}

/// A node in the scenario graph.
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioNode {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub node_type: String,
    pub family: String,
    pub status: String,
    pub health: u8,
    pub confidence: u8,
    pub position: Position,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub data_channels: Vec<DataChannel>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub thresholds: Vec<ThresholdRange>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

/// An edge in the scenario graph.
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioEdge {
    pub from: String,
    pub to: String,
    pub edge_type: String,
    pub label: String,
}

/// Complete scenario — petalTongue-compatible with neuralSpring extensions.
#[derive(Debug, Clone, Serialize)]
pub struct NeuralScenario {
    pub name: String,
    pub description: String,
    pub version: String,
    pub mode: String,
    pub sensory_config: SensoryConfig,
    pub ui_config: UiConfig,
    pub ecosystem: Ecosystem,
    pub neural_api: NeuralApi,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub edges: Vec<ScenarioEdge>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Ecosystem {
    pub primals: Vec<ScenarioNode>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SensoryConfig {
    pub required_capabilities: CapReqs,
    pub optional_capabilities: CapReqs,
    pub complexity_hint: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CapReqs {
    pub outputs: Vec<String>,
    pub inputs: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UiConfig {
    pub theme: String,
    pub animations: Animations,
    pub performance: Performance,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_panels: Option<ShowPanels>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub awakening_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_zoom: Option<String>,
}

/// Panel visibility for petalTongue scenario config.
#[expect(
    clippy::struct_excessive_bools,
    reason = "matches petalTongue JSON schema — each field serializes as a named boolean key"
)]
#[derive(Debug, Clone, Serialize)]
pub struct ShowPanels {
    pub left_sidebar: bool,
    pub right_sidebar: bool,
    pub top_menu: bool,
    pub system_dashboard: bool,
    pub audio_panel: bool,
    pub trust_dashboard: bool,
    pub proprioception: bool,
    pub graph_stats: bool,
}

#[expect(clippy::struct_excessive_bools, reason = "matches petalTongue schema")]
#[derive(Debug, Clone, Serialize)]
pub struct Animations {
    pub enabled: bool,
    pub breathing_nodes: bool,
    pub connection_pulses: bool,
    pub smooth_transitions: bool,
    pub celebration_effects: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Performance {
    pub target_fps: u32,
    pub vsync: bool,
    pub hardware_acceleration: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct NeuralApi {
    pub enabled: bool,
}
