// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reusable scaffold helpers for building petalTongue scenarios.
//!
//! Thin constructors that map domain-specific data into [`super::super::types::DataChannel`],
//! [`super::super::types::ScenarioNode`], and [`super::super::types::ScenarioEdge`] types.  Used by every per-domain
//! scenario builder and the study combiners.

use super::super::types::{
    Animations, CapReqs, DataChannel, Ecosystem, NeuralApi, NeuralScenario, Performance, Position,
    ScenarioEdge, ScenarioNode, SensoryConfig, ThresholdRange, UiConfig,
};

pub fn scaffold(name: &str, description: &str) -> NeuralScenario {
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
pub fn gauge(
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

pub fn timeseries(
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

pub fn bar(id: &str, label: &str, cats: Vec<String>, vals: Vec<f64>, unit: &str) -> DataChannel {
    DataChannel::Bar {
        id: id.into(),
        label: label.into(),
        categories: cats,
        values: vals,
        unit: unit.into(),
    }
}

pub fn spectrum(
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

pub fn scatter3d(
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

pub fn heatmap(
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

pub fn distribution(
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

#[cfg(feature = "barracuda")]
pub fn fieldmap(
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
pub fn node(
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

pub fn edge(from: &str, to: &str, label: &str) -> ScenarioEdge {
    ScenarioEdge {
        from: from.into(),
        to: to.into(),
        edge_type: "data-flow".into(),
        label: label.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_structure() {
        let scenario = scaffold("test", "desc");
        assert!(!scenario.name.is_empty());
        assert!(!scenario.description.is_empty());
        assert_eq!(scenario.version, "1.0.0");
        assert_eq!(scenario.mode, "research");
        assert!(scenario.ui_config.theme.contains("neural"));
        assert!(!scenario.neural_api.enabled);
    }

    #[test]
    fn gauge_produces_gauge_channel() {
        let ch = gauge(
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
        let ch = timeseries("ts1", "T", "X", "Y", "u", vec![1.0], vec![2.0]);
        assert!(matches!(ch, DataChannel::TimeSeries { .. }));
    }

    #[test]
    fn bar_produces_bar_channel() {
        let ch = bar("b1", "B", vec!["A".into()], vec![1.0], "u");
        assert!(matches!(ch, DataChannel::Bar { .. }));
    }

    #[test]
    fn spectrum_produces_spectrum_channel() {
        let ch = spectrum("s1", "S", "u", vec![1.0], vec![2.0]);
        assert!(matches!(ch, DataChannel::Spectrum { .. }));
    }

    #[test]
    fn scatter3d_produces_scatter3d_channel() {
        let ch = scatter3d(
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
        let n = node("n1", "N", "compute", 10.0, 20.0, &["cap1"], vec![], vec![]);
        assert_eq!(n.id, "n1");
        assert_eq!(n.family, crate::config::PRIMAL_FAMILY);
        assert_eq!(n.health, 100);
    }

    #[test]
    fn edge_produces_scenario_edge() {
        let e = edge("a", "b", "test");
        assert_eq!(e.from, "a");
        assert_eq!(e.to, "b");
        assert_eq!(e.edge_type, "data-flow");
    }
}
