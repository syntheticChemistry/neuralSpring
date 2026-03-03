// SPDX-License-Identifier: AGPL-3.0-or-later

//! `ToadStool` S86 rewire validation — nautilus absorption + `DriftMonitor` evolution.
//!
//! Validates the migration from `bingocube_nautilus` to `barracuda::nautilus`
//! (`ToadStool` S80 absorption) and confirms all nautilus APIs work correctly
//! with the `BarraCUDA`-native implementation.
//!
//! ## What changed (`ToadStool` S79 → S86)
//!
//! - `barracuda::nautilus` absorbed from `bingoCube` (S80): 7 files, 22 tests
//! - `DriftMonitor::record` signature: `(epoch, pop_size, mean, best)` → `(&GenerationRecord, pop_size)`
//! - `DriftMonitor.history` → `DriftMonitor.ne_s_history`
//! - `DriftMonitor.consecutive_drift` removed (computed inline in `is_drifting()`)
//! - `BatchedEncoder` added for multi-op GPU pipelines
//! - `ComputeDispatch`: 76 → 144 ops (+68 migrations)
//! - `blake3` dependency added for Nautilus board hashing
//!
//! ```text
//! cargo run --release --bin validate_toadstool_s86_rewire
//! ```

#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "validation binary — direct field access on known-good test data"
)]

use barracuda::nautilus::{
    BetaObservation, DriftMonitor, EvolutionConfig, GenerationRecord, InstanceId, NautilusBrain,
    NautilusBrainConfig, NautilusShell, SelectionMethod, ShellConfig,
};
use neural_spring::nautilus_bridge::SpectralNautilusBridge;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn validate_nautilus_types(h: &mut ValidationHarness) {
    eprintln!("\n── barracuda::nautilus types (absorbed from bingoCube) ──");

    let config = NautilusBrainConfig::default();
    let brain = NautilusBrain::new(config, "s86-test");
    h.check_bool("NautilusBrain creates from default config", true);
    h.check_bool("NautilusBrain starts untrained", !brain.trained);
    h.check_bool(
        "NautilusBrain starts with 0 observations",
        brain.observations.is_empty(),
    );

    let shell_config = ShellConfig::default();
    let origin = InstanceId("s86-shell".to_string());
    let shell = NautilusShell::from_seed(shell_config, origin, 42);
    h.check_bool("NautilusShell creates from seed", true);
    h.check_bool(
        "NautilusShell has population",
        !shell.population.boards.is_empty(),
    );

    let evo_config = EvolutionConfig {
        mutation_rate: 0.1,
        crossover_rate: 0.5,
        selection: SelectionMethod::Tournament(3),
    };
    h.check_abs(
        "EvolutionConfig mutation_rate",
        evo_config.mutation_rate,
        0.1,
        tolerances::EXACT_F64,
    );
}

fn validate_drift_monitor(h: &mut ValidationHarness) {
    eprintln!("\n── DriftMonitor S86 API (GenerationRecord) ──");

    let mut drift = DriftMonitor::default();
    h.check_bool("DriftMonitor starts empty", drift.ne_s_history.is_empty());
    h.check_bool("DriftMonitor not drifting when empty", !drift.is_drifting());

    let gen = GenerationRecord {
        generation: 0,
        mean_fitness: 0.5,
        best_fitness: 0.8,
        pop_size: 100,
        origin: InstanceId("s86-test".to_string()),
        training_size: 10,
    };
    drift.record(&gen, 100);
    h.check_bool(
        "DriftMonitor records ne_s entry",
        drift.ne_s_history.len() == 1,
    );

    let ne_s = drift.ne_s_history[0];
    let expected = (100.0 * 0.8) / (1.0 + 0.8);
    h.check_abs(
        "DriftMonitor ne_s calculation",
        ne_s,
        expected,
        tolerances::CROSS_LANGUAGE,
    );

    for i in 1..10 {
        let gen_low = GenerationRecord {
            generation: i,
            mean_fitness: 0.001,
            best_fitness: 0.002,
            pop_size: 100,
            origin: InstanceId("s86-test".to_string()),
            training_size: 10,
        };
        drift.record(&gen_low, 100);
    }

    h.check_bool(
        "DriftMonitor detects drift with low fitness",
        drift.is_drifting(),
    );
}

fn validate_bridge_absorption(h: &mut ValidationHarness) {
    eprintln!("\n── SpectralNautilusBridge (barracuda::nautilus) ──");

    let mut bridge = SpectralNautilusBridge::new("s86-bridge");
    h.check_bool("Bridge creates with barracuda::nautilus", true);
    h.check_bool(
        "Bridge starts with 0 observations",
        bridge.observation_count() == 0,
    );

    for i in 0..8 {
        let w = f64::from(i).mul_add(0.5, 2.0);
        let lsr = if w < 4.0 { 0.53 } else { 0.39 };
        bridge.observe_spectral(w, lsr, 0.1 / w, w * 0.3, 0.02 * w);
    }
    h.check_bool(
        "Bridge accumulates 8 observations",
        bridge.observation_count() == 8,
    );

    let mse = bridge.train();
    h.check_bool("Bridge trains successfully", mse.is_some());
    h.check_bool("Bridge is trained after train()", bridge.is_trained());

    let pred = bridge.predict(3.0);
    h.check_bool("Bridge predicts after training", pred.is_some());
    if let Some((ipr_s, bw, lsr)) = pred {
        h.check_bool("Prediction ipr_s is finite", ipr_s.is_finite());
        h.check_bool("Prediction bw is finite", bw.is_finite());
        h.check_bool("Prediction lsr is finite", lsr.is_finite());
    }

    let scored = bridge.screen_candidates(&[2.0, 3.0, 4.0, 5.0]);
    h.check_bool("Bridge screens 4 candidates", scored.len() == 4);

    let json = bridge.to_json().expect("serialization should not fail");
    let restored =
        SpectralNautilusBridge::from_json(&json).expect("deserialization should not fail");
    h.check_bool(
        "JSON roundtrip preserves observation count",
        restored.observation_count() == bridge.observation_count(),
    );
    h.check_bool(
        "JSON roundtrip preserves trained state",
        restored.is_trained() == bridge.is_trained(),
    );
}

fn validate_beta_observation(h: &mut ValidationHarness) {
    eprintln!("\n── BetaObservation struct (field compatibility) ──");

    let obs = BetaObservation {
        beta: 5.5,
        plaquette: 0.58,
        cg_iters: 120.0,
        acceptance: 0.75,
        delta_h_abs: 0.01,
        quenched_plaq: Some(0.60),
        quenched_plaq_var: Some(0.003),
        anderson_r: Some(0.42),
        anderson_lambda_min: Some(-2.1),
    };

    h.check_abs("BetaObservation.beta", obs.beta, 5.5, tolerances::EXACT_F64);
    h.check_abs(
        "BetaObservation.plaquette",
        obs.plaquette,
        0.58,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "BetaObservation.anderson_r",
        obs.anderson_r.unwrap(),
        0.42,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "BetaObservation.anderson_lambda_min",
        obs.anderson_lambda_min.unwrap(),
        -2.1,
        tolerances::EXACT_F64,
    );
}

fn main() {
    eprintln!("=== ToadStool S86 Rewire Validation ===\n");
    eprintln!("Pin: f97fc2ae → 2fee1969 (S79→S86, 7 commits)");
    eprintln!("Key: bingocube_nautilus → barracuda::nautilus (absorption)");
    eprintln!("ComputeDispatch: 76 → 144 ops (+68 migrations)");

    let mut h = ValidationHarness::new("toadstool_s86_rewire");

    validate_nautilus_types(&mut h);
    validate_drift_monitor(&mut h);
    validate_bridge_absorption(&mut h);
    validate_beta_observation(&mut h);

    h.finish();
}
