// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-spring shader provenance scenario builder.
//!
//! Visualizes the shader dependency graph across springs and the
//! distribution of shaders by category.

#![expect(
    clippy::cast_precision_loss,
    reason = "shader count fits comfortably in f64 mantissa"
)]

use crate::visualization::types::{NeuralScenario, ScenarioEdge};

use super::{bar, gauge, node, scaffold};

/// Build the shader provenance study scenario.
///
/// Produces 1 node showing shader categories as a bar chart and total
/// counts as gauges.  Edges represent cross-spring shader dependencies.
#[must_use]
pub fn provenance_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Cross-Spring Shader Provenance",
        "Shader dependency graph and category distribution across ecoPrimal springs",
    );

    let shaders = barracuda::shaders::provenance::cross_spring_shaders();
    let matrix = barracuda::shaders::provenance::cross_spring_matrix();

    let mut category_counts: std::collections::BTreeMap<String, f64> =
        std::collections::BTreeMap::new();
    for shader in &shaders {
        *category_counts
            .entry(format!("{}", shader.category))
            .or_default() += 1.0;
    }

    let categories: Vec<String> = category_counts.keys().cloned().collect();
    let counts: Vec<f64> = category_counts.values().copied().collect();

    s.ecosystem.primals.push(node(
        "shader_provenance",
        "Shader Provenance",
        "metadata",
        0.0,
        0.0,
        &[
            "science.cross_spring_provenance",
            "science.cross_spring_benchmark",
        ],
        vec![
            bar(
                "shader-categories",
                "Shaders by Category",
                categories,
                counts,
                "count",
            ),
            gauge(
                "total-shaders",
                "Total Tracked Shaders",
                shaders.len() as f64,
                0.0,
                100.0,
                "count",
                [0.0, 50.0],
                [50.0, 80.0],
            ),
            gauge(
                "cross-spring-edges",
                "Cross-Spring Dependencies",
                matrix.len() as f64,
                0.0,
                50.0,
                "count",
                [0.0, 20.0],
                [20.0, 40.0],
            ),
        ],
        vec![],
    ));

    let mut edges = Vec::new();
    for (from, to) in matrix.keys() {
        edges.push(ScenarioEdge {
            from: format!("{from}"),
            to: format!("{to}"),
            edge_type: "shader-dependency".into(),
            label: "shared shaders".into(),
        });
    }

    (s, edges)
}
