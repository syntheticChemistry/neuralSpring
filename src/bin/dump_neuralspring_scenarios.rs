// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dump neuralSpring visualization scenarios to JSON files.
//!
//! Generates scenario JSON in `sandbox/scenarios/` for offline rendering
//! by petalTongue: `petaltongue ui --scenario sandbox/scenarios/neuralspring-full-study.json`

#![expect(
    clippy::expect_used,
    reason = "standalone utility binary — panics on I/O failure are acceptable"
)]

use neural_spring::visualization::{
    coordination_study, folding_study, full_study, provenance_study, scenario_with_edges_json,
    spectral_study, training_study,
};

fn main() {
    let dir = "sandbox/scenarios";
    std::fs::create_dir_all(dir).expect("create sandbox/scenarios/");

    let studies: Vec<(&str, _)> = vec![
        ("neuralspring-spectral-study", spectral_study()),
        ("neuralspring-training-study", training_study()),
        ("neuralspring-coordination-study", coordination_study()),
        ("neuralspring-provenance-study", provenance_study()),
        ("neuralspring-folding-study", folding_study()),
        ("neuralspring-full-study", full_study()),
    ];

    for (name, (scenario, edges)) in &studies {
        let json = scenario_with_edges_json(scenario, edges);
        let path = format!("{dir}/{name}.json");
        std::fs::write(&path, &json).unwrap_or_else(|e| panic!("write {path}: {e}"));
        println!(
            "{path}: {} nodes, {} edges, {} bytes",
            scenario.ecosystem.primals.len(),
            edges.len(),
            json.len()
        );
    }

    println!("\nAll scenarios written to {dir}/");
    println!("Render with: petaltongue ui --scenario {dir}/neuralspring-full-study.json");
}
