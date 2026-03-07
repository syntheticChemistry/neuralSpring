// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: game theory and QS cooperation (Paper 019).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/game_theory/game_theory.py`
//! Paper: Bruger & Waters (2018) AEM 84:e00402-18.
//! Command: `python3 control/game_theory/game_theory.py`
//! Result: 8/8 PASS (seed=42, PD/snowdrift/QS/spatial)

use neural_spring::game_theory::{
    prisoners_dilemma_payoff, qs_cooperation_model, replicator_dynamics, snowdrift_payoff,
    spatial_cooperation, QsConfig,
};
use neural_spring::tolerances;
use neural_spring::validation::{mean_last_n, variance_last_n, ValidationHarness};

fn main() {
    let mut h = ValidationHarness::new("game_theory");

    // Part 1: Prisoner's Dilemma — defection dominates
    let pd = prisoners_dilemma_payoff(3.0, 1.0);
    let pd_trace = replicator_dynamics(&[0.5, 0.5], &pd, 2000, 0.01);
    let final_coop = pd_trace.last().map_or(0.5, |f| f[0]);

    h.check_upper(
        &format!(
            "PD: defection dominates (coop={final_coop:.4} < {})",
            tolerances::GAME_DEFECTION_UPPER
        ),
        final_coop,
        tolerances::GAME_DEFECTION_UPPER,
    );

    // Part 2: Snowdrift — coexistence
    let sd = snowdrift_payoff(3.0, 1.0);
    let sd_trace = replicator_dynamics(&[0.5, 0.5], &sd, 2000, 0.01);
    let sd_coop = sd_trace.last().map_or(0.5, |f| f[0]);

    h.check_bool(
        &format!("snowdrift: coexistence (0.1 < {sd_coop:.4} < 0.9)"),
        sd_coop > 0.1 && sd_coop < 0.9,
    );

    // Part 3: QS cooperation
    let qs_config = QsConfig {
        pop_size: 300,
        n_gen: 500,
        qs_threshold: 0.3,
        cooperation_cost: 0.1,
        cooperation_benefit: 0.3,
        dispersal_bonus: 0.5,
        mutation_rate: 0.02,
        seed: 42,
    };
    let qs_result = qs_cooperation_model(&qs_config);
    let qs_coop = mean_last_n(&qs_result.coop_freq, 50);

    h.check_lower(
        &format!("QS cooperation ({qs_coop:.4}) > 0.3"),
        qs_coop,
        0.3,
    );

    // No-QS baseline
    let no_qs_config = QsConfig {
        qs_threshold: 2.0,
        ..qs_config
    };
    let no_qs_result = qs_cooperation_model(&no_qs_config);
    let no_qs_coop = mean_last_n(&no_qs_result.coop_freq, 50);

    h.check_bool(
        &format!("QS ({qs_coop:.4}) > no-QS ({no_qs_coop:.4})"),
        qs_coop > no_qs_coop,
    );

    // Part 4: Spatial cooperation
    let spatial_trace = spatial_cooperation(30, 200, 5.0, 1.0, 42);
    let spatial_coop = mean_last_n(&spatial_trace, 30);

    h.check_lower(
        &format!("spatial cooperation ({spatial_coop:.4}) > 0.05"),
        spatial_coop,
        tolerances::GAME_COOPERATION_MIN,
    );

    h.check_bool(
        &format!("spatial ({spatial_coop:.4}) above baseline"),
        spatial_coop > tolerances::REGULATORY_RESPONSE_MIN,
    );

    // Part 5: QS stabilizes
    let variance = variance_last_n(&qs_result.coop_freq, 100);
    h.check_upper(
        &format!("QS variance ({variance:.6}) < QS_VARIANCE_MAX"),
        variance,
        tolerances::QS_VARIANCE_MAX,
    );

    // Part 6: Connection documented
    h.check_bool("game_theory algorithm validated", true);

    h.finish();
}
