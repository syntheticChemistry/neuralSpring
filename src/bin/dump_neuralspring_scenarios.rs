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
    attention_anderson_study, composition_study, coordination_study, digester_anderson_study,
    folding_study, full_study, game_theory_study, glucose_study, hmm_study,
    immunological_study, introgression_nn_study, isomorphic_reservoir_study,
    loss_landscape_study, population_study, provenance_study, scenario_with_edges_json,
    spectral_study, training_study, wdm_ensemble_qs_study, wdm_study,
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
        ("neuralspring-hmm-study", hmm_study()),
        ("neuralspring-game-theory-study", game_theory_study()),
        ("neuralspring-wdm-study", wdm_study()),
        ("neuralspring-glucose-study", glucose_study()),
        ("neuralspring-immunological-study", immunological_study()),
        ("neuralspring-population-study", population_study()),
        ("neuralspring-loss-landscape-study", loss_landscape_study()),
        ("neuralspring-digester-anderson", digester_anderson_study()),
        ("neuralspring-isomorphic-reservoir", isomorphic_reservoir_study()),
        ("neuralspring-wdm-ensemble-qs", wdm_ensemble_qs_study()),
        ("neuralspring-introgression-nn", introgression_nn_study()),
        ("neuralspring-attention-anderson", attention_anderson_study()),
        ("neuralspring-compositions", composition_study()),
        ("neuralspring-complete-study", full_study()),
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

    println!(
        "\nAll {count} scenarios written to {dir}/",
        count = studies.len()
    );
    println!("Render with: petaltongue ui --scenario {dir}/neuralspring-complete-study.json");
}
