// SPDX-License-Identifier: AGPL-3.0-or-later

//! Warm Dense Matter scenario builder (nW-01..05).
//!
//! Produces 2 nodes: transport coefficient sweep and phase-space
//! scatter. Uses the WDM surrogate MLP for real predictions across
//! a grid of (`log_rho`, `log_T`) conditions.

#![expect(
    clippy::cast_precision_loss,
    reason = "grid size to f64 for gauge value"
)]

use crate::rng::Rng;
use crate::visualization::types::{NeuralScenario, ScenarioEdge};
use crate::wdm_transport::TransportSurrogate;

use super::{edge, gauge, node, scaffold, scatter3d, timeseries};

type ScatterData = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>);

/// Build the WDM transport surrogate scenario.
///
/// Nodes:
/// - `wdm_transport`: `D*` vs temperature sweep + phase scatter
/// - `wdm_phase`: 3D scatter of (`log_rho`, `log_T`, `D*`) predictions
#[must_use]
pub fn wdm_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Warm Dense Matter Surrogates",
        "Transport coefficients (D*, η*, λ*) from MLP surrogates across WDM regimes",
    );

    let json_path =
        crate::validation::baseline_path("control/wdm/transport_surrogate_baseline.json");

    let surrogate = json_path
        .exists()
        .then(|| {
            let file = std::fs::File::open(&json_path).ok()?;
            let reader = std::io::BufReader::new(file);
            crate::wdm_transport::load_transport_from_reader(reader).ok()
        })
        .flatten();

    let log_rhos = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
    let log_ts = vec![3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0];
    let z_star = 1.0;

    let (d_stars, scatter_rho, scatter_t, scatter_d, scatter_labels) =
        surrogate.as_ref().map_or_else(
            || build_synthetic(&log_rhos, &log_ts),
            |surr| build_from_surrogate(surr, &log_rhos, &log_ts, z_star),
        );

    s.ecosystem.primals.push(node(
        "wdm_transport",
        "WDM Transport Coefficients",
        "compute",
        0.0,
        0.0,
        &["science.wdm_transport"],
        vec![
            timeseries(
                "d-star-vs-temp",
                "D* vs Temperature (log_ρ=0)",
                "log₁₀(T/K)",
                "D* (reduced)",
                "dimensionless",
                log_ts.clone(),
                d_stars,
            ),
            gauge(
                "wdm-grid-points",
                "Phase-Space Grid Points",
                (log_rhos.len() * log_ts.len()) as f64,
                0.0,
                200.0,
                "count",
                [10.0, 100.0],
                [100.0, 200.0],
            ),
        ],
        vec![],
    ));

    s.ecosystem.primals.push(node(
        "wdm_phase",
        "WDM Phase-Space Scatter",
        "compute",
        400.0,
        0.0,
        &["science.wdm_phase_diagram"],
        vec![scatter3d(
            "wdm-phase-scatter",
            "Transport in (ρ, T, D*) Space",
            "dimensionless",
            scatter_rho,
            scatter_t,
            scatter_d,
            scatter_labels,
        )],
        vec![],
    ));

    let edges = vec![edge("wdm_transport", "wdm_phase", "sweep → phase diagram")];
    (s, edges)
}

fn build_from_surrogate(
    surr: &TransportSurrogate,
    log_rhos: &[f64],
    log_ts: &[f64],
    z_star: f64,
) -> ScatterData {
    let d_stars: Vec<f64> = log_ts
        .iter()
        .map(|&lt| surr.predict(0.0, lt, z_star).0)
        .collect();

    let mut scatter_rho = Vec::new();
    let mut scatter_t = Vec::new();
    let mut scatter_d = Vec::new();
    let mut scatter_labels = Vec::new();

    for &lr in log_rhos {
        for &lt in log_ts {
            let (d, _, _) = surr.predict(lr, lt, z_star);
            scatter_rho.push(lr);
            scatter_t.push(lt);
            scatter_d.push(d);
            scatter_labels.push(format!("ρ={lr:.1},T={lt:.1}"));
        }
    }
    (d_stars, scatter_rho, scatter_t, scatter_d, scatter_labels)
}

fn build_synthetic(log_rhos: &[f64], log_ts: &[f64]) -> ScatterData {
    let mut rng = Rng::new(42);

    let d_stars: Vec<f64> = log_ts
        .iter()
        .map(|&lt| 0.1f64.mul_add((lt - 3.0).powi(2), rng.uniform() * 0.05))
        .collect();

    let mut scatter_rho = Vec::new();
    let mut scatter_t = Vec::new();
    let mut scatter_d = Vec::new();
    let mut scatter_labels = Vec::new();

    for &lr in log_rhos {
        for &lt in log_ts {
            let d = 0.1f64.mul_add(
                (lt - 3.0).powi(2),
                (-0.05f64).mul_add(lr, rng.uniform() * 0.02),
            );
            scatter_rho.push(lr);
            scatter_t.push(lt);
            scatter_d.push(d);
            scatter_labels.push(format!("ρ={lr:.1},T={lt:.1}"));
        }
    }
    (d_stars, scatter_rho, scatter_t, scatter_d, scatter_labels)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;
    use crate::visualization::types::DataChannel;

    #[test]
    fn wdm_study_has_two_nodes_and_edge() {
        let (scenario, edges) = wdm_study();
        assert_eq!(scenario.ecosystem.primals.len(), 2);
        let ids: Vec<&str> = scenario
            .ecosystem
            .primals
            .iter()
            .map(|n| n.id.as_str())
            .collect();
        assert!(ids.contains(&"wdm_transport"));
        assert!(ids.contains(&"wdm_phase"));
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].from, "wdm_transport");
        assert_eq!(edges[0].to, "wdm_phase");
    }

    #[test]
    fn wdm_study_channels_include_timeseries_scatter_gauge() {
        let (scenario, _) = wdm_study();
        let channels: Vec<&DataChannel> = scenario
            .ecosystem
            .primals
            .iter()
            .flat_map(|n| n.data_channels.iter())
            .collect();
        assert!(
            channels
                .iter()
                .any(|c| matches!(c, DataChannel::TimeSeries { .. })),
            "expected TimeSeries for D* vs T"
        );
        assert!(
            channels
                .iter()
                .any(|c| matches!(c, DataChannel::Scatter3D { .. })),
            "expected Scatter3D phase diagram"
        );
        assert!(
            channels
                .iter()
                .any(|c| matches!(c, DataChannel::Gauge { .. })),
            "expected grid-point gauge"
        );
    }

    #[test]
    fn wdm_study_grid_point_gauge_matches_phase_space_size() {
        let (scenario, _) = wdm_study();
        let transport = scenario
            .ecosystem
            .primals
            .iter()
            .find(|n| n.id == "wdm_transport")
            .expect("transport node");
        let gauge = transport
            .data_channels
            .iter()
            .find_map(|c| {
                if let DataChannel::Gauge { value, .. } = c {
                    Some(*value)
                } else {
                    None
                }
            })
            .expect("grid gauge");
        assert!((gauge - 70.0).abs() < f64::EPSILON, "7×10 grid points");
    }

    #[test]
    fn wdm_study_synthetic_fallback_is_deterministic() {
        let (s1, _) = wdm_study();
        let (s2, _) = wdm_study();
        let scatter_values = |scenario: &crate::visualization::types::NeuralScenario| {
            scenario
                .ecosystem
                .primals
                .iter()
                .find(|n| n.id == "wdm_phase")
                .and_then(|n| {
                    n.data_channels.iter().find_map(|c| {
                        if let DataChannel::Scatter3D { z, .. } = c {
                            Some(z.clone())
                        } else {
                            None
                        }
                    })
                })
        };
        let z1 = scatter_values(&s1).expect("scatter z");
        let z2 = scatter_values(&s2).expect("scatter z");
        assert_eq!(z1.len(), 70);
        assert_eq!(z1, z2, "seeded synthetic path must be reproducible");
    }

    #[test]
    fn wdm_study_scenario_metadata() {
        let (scenario, _) = wdm_study();
        assert_eq!(scenario.name, "Warm Dense Matter Surrogates");
        assert!(scenario.description.contains("Transport coefficients"));
    }

    #[test]
    fn wdm_study_timeseries_has_ten_points() {
        let (scenario, _) = wdm_study();
        let transport = scenario
            .ecosystem
            .primals
            .iter()
            .find(|n| n.id == "wdm_transport")
            .expect("transport node");
        let ts = transport
            .data_channels
            .iter()
            .find_map(|c| {
                if let DataChannel::TimeSeries { y_values, .. } = c {
                    Some(y_values.len())
                } else {
                    None
                }
            })
            .expect("timeseries");
        assert_eq!(ts, 10, "log_T grid has 10 temperature points");
    }

    #[test]
    fn wdm_study_scatter_labels_contain_rho_and_temp() {
        let (scenario, _) = wdm_study();
        let phase = scenario
            .ecosystem
            .primals
            .iter()
            .find(|n| n.id == "wdm_phase")
            .expect("phase node");
        let labels = phase
            .data_channels
            .iter()
            .find_map(|c| {
                if let DataChannel::Scatter3D { point_labels, .. } = c {
                    Some(point_labels.clone())
                } else {
                    None
                }
            })
            .expect("scatter labels");
        assert_eq!(labels.len(), 70);
        assert!(labels[0].starts_with("ρ="));
        assert!(labels[0].contains("T="));
    }

    #[test]
    fn wdm_study_scatter_d_values_finite_and_positive() {
        let (scenario, _) = wdm_study();
        let z = scenario
            .ecosystem
            .primals
            .iter()
            .find(|n| n.id == "wdm_phase")
            .and_then(|n| {
                n.data_channels.iter().find_map(|c| {
                    if let DataChannel::Scatter3D { z, .. } = c {
                        Some(z.clone())
                    } else {
                        None
                    }
                })
            })
            .expect("scatter z");
        assert!(z.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[test]
    fn wdm_study_node_capabilities() {
        let (scenario, _) = wdm_study();
        let transport = scenario
            .ecosystem
            .primals
            .iter()
            .find(|n| n.id == "wdm_transport")
            .expect("transport");
        assert!(
            transport
                .capabilities
                .iter()
                .any(|c| c == "science.wdm_transport")
        );
        let phase = scenario
            .ecosystem
            .primals
            .iter()
            .find(|n| n.id == "wdm_phase")
            .expect("phase");
        assert!(
            phase
                .capabilities
                .iter()
                .any(|c| c == "science.wdm_phase_diagram")
        );
    }

    #[test]
    fn wdm_study_edge_label() {
        let (_, edges) = wdm_study();
        assert_eq!(edges[0].label, "sweep → phase diagram");
    }

    #[test]
    fn wdm_study_surrogate_path_when_baseline_present() {
        let baseline =
            crate::validation::baseline_path("control/wdm/transport_surrogate_baseline.json");
        if !baseline.exists() {
            return;
        }
        let (scenario, _) = wdm_study();
        let z = scenario
            .ecosystem
            .primals
            .iter()
            .find(|n| n.id == "wdm_phase")
            .and_then(|n| {
                n.data_channels.iter().find_map(|c| {
                    if let DataChannel::Scatter3D { z, .. } = c {
                        Some(z.clone())
                    } else {
                        None
                    }
                })
            })
            .expect("scatter");
        let (scenario2, _) = wdm_study();
        let z2 = scenario2
            .ecosystem
            .primals
            .iter()
            .find(|n| n.id == "wdm_phase")
            .and_then(|n| {
                n.data_channels.iter().find_map(|c| {
                    if let DataChannel::Scatter3D { z, .. } = c {
                        Some(z.clone())
                    } else {
                        None
                    }
                })
            })
            .expect("scatter2");
        assert_eq!(z, z2, "surrogate path must be deterministic");
    }
}
