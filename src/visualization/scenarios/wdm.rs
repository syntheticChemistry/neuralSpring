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
