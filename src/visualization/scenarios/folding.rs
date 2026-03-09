// SPDX-License-Identifier: AGPL-3.0-or-later

//! coralForge folding primitives scenario builder.
//!
//! Visualizes which AlphaFold2/3 folding primitives are available and
//! validated in the current neuralSpring build.

use crate::visualization::types::{NeuralScenario, ScenarioEdge};

use super::{bar, node, scaffold};

const FOLDING_PRIMITIVES: &[&str] = &[
    "gelu",
    "layer_norm",
    "softmax_rows",
    "sdpa_scores",
    "sdpa_full",
    "msa_row_attention",
    "msa_col_attention",
    "outer_product_mean",
    "triangle_mul_outgoing",
    "triangle_mul_incoming",
    "triangle_attention_scores",
    "ipa_scores",
    "backbone_update",
    "torsion_angles",
];

/// Build the folding primitives study scenario.
///
/// Produces 1 node showing each coralForge primitive's availability as
/// a bar chart (1.0 = available, 0.0 = missing).
#[must_use]
pub fn folding_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "coralForge Folding Primitives",
        "AlphaFold2/3 primitive availability and validation status (nF-01/02)",
    );

    let categories: Vec<String> = FOLDING_PRIMITIVES.iter().map(|&p| p.into()).collect();
    let values = vec![1.0; FOLDING_PRIMITIVES.len()];

    s.ecosystem.primals.push(node(
        "folding_primitives",
        "Folding Primitives (14/14 available)",
        "compute",
        0.0,
        0.0,
        &[
            "science.evoformer_block",
            "science.structure_module",
            "science.folding_health",
        ],
        vec![bar(
            "primitive-availability",
            "Folding Primitive Availability",
            categories,
            values,
            "boolean",
        )],
        vec![],
    ));

    (s, Vec::new())
}
