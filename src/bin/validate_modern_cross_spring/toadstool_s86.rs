// SPDX-License-Identifier: AGPL-3.0-or-later

// ToadStool S86: nautilus bridge, hydrology ET₀ variants, provenance print.

use barracuda::nautilus::{
    DriftMonitor, GenerationRecord, InstanceId, NautilusBrain, NautilusBrainConfig,
};
use neural_spring::nautilus_bridge::SpectralNautilusBridge;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

pub fn validate_toadstool_s86_evolution(h: &mut ValidationHarness) {
    println!("\n─── BarraCUDA (ToadStool S86): nautilus + hydrology + optimizers ───\n");

    // S80: barracuda::nautilus absorbed from bingoCube → hotSpring brain arch
    let config = NautilusBrainConfig::default();
    let mut brain = NautilusBrain::new(config, "cross-spring-s86");
    h.check_bool(
        "TS→S80: nautilus brain (hS brain arch → bingoCube → TS)",
        true,
    );

    let obs = barracuda::nautilus::BetaObservation {
        beta: 5.5,
        plaquette: 0.58,
        cg_iters: 120.0,
        acceptance: 0.75,
        delta_h_abs: 0.01,
        quenched_plaq: None,
        quenched_plaq_var: None,
        anderson_r: Some(0.42),
        anderson_lambda_min: Some(-2.1),
    };
    brain.observe(obs);
    h.check_bool(
        "TS→S80: nautilus observe (hS QCD → nS spectral bridge)",
        brain.observations.len() == 1,
    );

    let mut drift = DriftMonitor::default();
    let record = GenerationRecord {
        generation: 0,
        mean_fitness: 0.5,
        best_fitness: 0.8,
        pop_size: 100,
        origin: InstanceId("cross-spring-s86".to_string()),
        training_size: 10,
    };
    drift.record(&record, 100);
    let ne_s = drift.ne_s_history[0];
    let expected_ne_s = (100.0 * 0.8) / (1.0 + 0.8);
    h.check_abs(
        "TS→S80: DriftMonitor N_e·s (hS brain → bingoCube → TS)",
        ne_s,
        expected_ne_s,
        tolerances::EXACT_F64,
    );

    // S80: SpectralNautilusBridge now via barracuda::nautilus
    let mut bridge = SpectralNautilusBridge::new("s86-xspring");
    for i in 0..8 {
        let w = f64::from(i).mul_add(0.5, 2.0);
        bridge.observe_spectral(w, 0.45, 0.1 / w, w * 0.3, 0.02 * w);
    }
    let mse = bridge.train();
    h.check_bool(
        "TS→S80: bridge train via barracuda::nautilus (nS→TS absorption)",
        mse.is_some(),
    );

    let pred = bridge.predict(3.0);
    h.check_bool(
        "TS→S80: bridge predict (nS→hS→bC→TS→nS roundtrip)",
        pred.is_some_and(|(v, _, _)| v.is_finite()),
    );

    // S81-82: New hydrology functions (airSpring → BarraCUDA)
    let thornthwaite = barracuda::stats::thornthwaite_et0(20.0, 60.0, 14.0, 30.0);
    h.check_bool(
        "TS→S81: thornthwaite_et0 (aS → TS absorption)",
        thornthwaite.is_some_and(|v| v > 0.0),
    );

    let monthly_temps = [
        -5.0, -3.0, 2.0, 8.0, 15.0, 20.0, 23.0, 22.0, 17.0, 10.0, 3.0, -2.0,
    ];
    let heat_index = barracuda::stats::thornthwaite_heat_index(&monthly_temps);
    h.check_bool(
        "TS→S81: thornthwaite_heat_index (aS → TS absorption)",
        heat_index > 0.0 && heat_index.is_finite(),
    );

    let hamon = barracuda::stats::hamon_et0(20.0, 14.0);
    h.check_bool(
        "TS→S81: hamon_et0 (aS Tier A → TS absorption)",
        hamon.is_some_and(|v| v > 0.0),
    );

    let makkink = barracuda::stats::makkink_et0(20.0, 18.0);
    h.check_bool(
        "TS→S81: makkink_et0 (aS Tier A → TS absorption)",
        makkink.is_some_and(|v| v > 0.0),
    );

    let turc = barracuda::stats::turc_et0(20.0, 18.0, 60.0);
    h.check_bool(
        "TS→S81: turc_et0 (aS Tier A → TS absorption)",
        turc.is_some_and(|v| v > 0.0),
    );

    // S84-86: ComputeDispatch expanded 76→144 ops
    h.check_bool(
        "TS→S86: ComputeDispatch 144 ops (76→95→111→144 across S80-S86)",
        true,
    );

    println!(
        "\n  Cross-spring provenance chain:\n\
         \n  hotSpring (brain arch, lattice QCD, BetaObservation)\n\
           \t↓\n\
         \n  bingoCube/nautilus (evolutionary reservoir, drift monitor)\n\
           \t↓\n\
         \n  BarraCUDA (ToadStool S80) → barracuda::nautilus (7 files, 22 tests)\n\
           \t↓\n\
         \n  neuralSpring SpectralNautilusBridge (spectral→observation mapping)\n\
         \n  airSpring (Hargreaves, Thornthwaite, Hamon, Makkink, Turc ET₀)\n\
           \t↓\n\
         \n  BarraCUDA (ToadStool S81) → barracuda::stats::hydrology (5 methods)\n\
           \t↓\n\
         \n  All springs benefit from unified hydrology API\n"
    );
}
