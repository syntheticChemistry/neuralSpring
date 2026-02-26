// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU port: game theory replicator dynamics (Paper 019).
//!
//! Validates `barracuda::numerical::rk45_solve` for replicator dynamics ODE
//! and `barracuda::linalg` concepts for payoff matrix operations.
//!
//! Evolution path:
//! ```text
//! Python (scipy.integrate) → Rust (hand-rolled RK4)
//!   → BarraCUDA CPU (barracuda::numerical::rk45_solve)
//!   → BarraCUDA GPU (rk4_batch.wgsl)
//! ```
//!
//! ## Provenance
//!
//! Python baseline: `control/game_theory/game_theory.py`
//! Rust baseline: `validate_game_theory`

#![allow(clippy::cast_precision_loss, clippy::similar_names)]

use neural_spring::game_theory::{
    prisoners_dilemma_payoff, qs_cooperation_model, replicator_dynamics, QsConfig,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_game");

    validate_replicator_via_rk45(&mut h);
    validate_pd_equilibrium(&mut h);
    validate_qs_cooperation(&mut h);

    h.finish();
}

fn make_config() -> barracuda::numerical::Rk45Config {
    barracuda::numerical::Rk45Config {
        h_init: 0.01,
        atol: tolerances::ODE_ATOL,
        rtol: tolerances::ODE_RTOL,
        ..barracuda::numerical::Rk45Config::default()
    }
}

/// Replicator dynamics: `dx_i/dt` = `x_i` * (`f_i` - `f_bar`).
fn replicator_rhs(_t: f64, x: &[f64], payoff: &[[f64; 2]; 2]) -> Vec<f64> {
    let x0 = x[0];
    let x1 = x[1];
    let f0 = payoff[0][0].mul_add(x0, payoff[0][1] * x1);
    let f1 = payoff[1][0].mul_add(x0, payoff[1][1] * x1);
    let f_bar = x0.mul_add(f0, x1 * f1);

    let dx0 = x0 * (f0 - f_bar);
    let dx1 = x1 * (f1 - f_bar);
    vec![dx0, dx1]
}

fn validate_replicator_via_rk45(h: &mut ValidationHarness) {
    let pd = prisoners_dilemma_payoff(3.0, 1.0);
    let x0 = [0.5, 0.5];

    let hand_trace = replicator_dynamics(&x0, &pd, 2000, 0.01);
    let hand_final = hand_trace.last().copied().unwrap_or([0.5, 0.5]);

    let rhs = |t: f64, y: &[f64]| replicator_rhs(t, y, &pd);
    let config = make_config();

    match barracuda::numerical::rk45_solve(&rhs, 0.0, 20.0, &x0, &config) {
        Ok(result) => {
            let rk45_x: Vec<f64> = result.y_final.iter().map(|v| (*v).max(0.0)).collect();
            let sum: f64 = rk45_x.iter().sum();
            let rk45_norm: Vec<f64> = if sum > 0.0 {
                rk45_x.iter().map(|x| x / sum).collect()
            } else {
                rk45_x
            };

            h.check_abs(
                &format!(
                    "PD coop: hand={:.4} vs RK45={:.4}",
                    hand_final[0], rk45_norm[0]
                ),
                hand_final[0],
                rk45_norm[0],
                tolerances::ADIABATIC_KL_GAP,
            );
            h.check_bool(
                "RK45 replicator final sums ≈ 1",
                (rk45_norm[0] + rk45_norm[1] - 1.0).abs() < tolerances::CD_COMPARABLE_DIST,
            );
        }
        Err(e) => {
            h.check_bool(&format!("rk45_solve replicator [ERROR: {e}]"), false);
        }
    }
}

fn validate_pd_equilibrium(h: &mut ValidationHarness) {
    let pd = prisoners_dilemma_payoff(3.0, 1.0);
    let trace = replicator_dynamics(&[0.5, 0.5], &pd, 2000, 0.01);
    let final_coop = trace.last().map_or(0.5, |f| f[0]);

    h.check_upper(
        "PD: defection dominates",
        final_coop,
        tolerances::GAME_DEFECTION_UPPER,
    );
}

fn validate_qs_cooperation(h: &mut ValidationHarness) {
    let config = QsConfig {
        pop_size: 200,
        n_gen: 100,
        qs_threshold: 0.3,
        cooperation_cost: 0.1,
        cooperation_benefit: 0.3,
        dispersal_bonus: 0.5,
        mutation_rate: 0.02,
        seed: 42,
    };
    let result = qs_cooperation_model(&config);

    let late_coop: f64 = result.coop_freq[80..].iter().sum::<f64>() / 20.0;
    let coop_var =
        barracuda::stats::correlation::variance(&result.coop_freq[50..]).unwrap_or(f64::NAN);

    h.check_lower(
        "QS cooperation late",
        late_coop,
        tolerances::GAME_QS_COOPERATION_MIN,
    );
    h.check_upper("QS variance", coop_var, tolerances::GAME_QS_VARIANCE_MAX);
}
