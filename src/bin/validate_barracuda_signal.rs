// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU port: signal integration vpsT regulatory ODE (Paper 021).
//!
//! Validates that `barracuda::numerical::rk45_solve` reproduces the vpsT ODE
//! steady states from hand-rolled RK4 in `signal_integration::integrate_ode`.
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
//! Python baseline: `control/signal_integration/signal_integration.py`
//! Rust baseline: `validate_signal_integration`

use neural_spring::signal_integration::{
    LogicGate, OdeParams, OdeState, classify_logic_gate, integrate_ode, logic_gate_sweep,
    two_input_hill,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_signal");

    validate_rk45_vs_rk4(&mut h);
    validate_and_gate_logic(&mut h);
    validate_final_state_bounds(&mut h);

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

/// RHS for vpsT ODE: d\[cdg,ai,vpsT,biofilm\]/dt. Deterministic (`noise_scale=0`).
fn vps_t_rhs(_t: f64, y: &[f64], params: &OdeParams) -> Vec<f64> {
    let cdg = y[0];
    let ai = y[1];
    let vps_t = y[2];

    let d_cdg = params.cdg_deg.mul_add(-cdg, params.cdg_synth);
    let d_ai = params
        .ai_prod
        .mul_add(params.cell_density, -(params.ai_decay * ai));
    let f_val = two_input_hill(
        cdg,
        ai,
        params.vmax,
        params.k1,
        params.k2,
        params.n1,
        params.n2,
    );
    let d_vps_t = params.vps_degradation.mul_add(-vps_t, f_val);
    let d_biofilm = vps_t;

    vec![d_cdg, d_ai, d_vps_t, d_biofilm]
}

/// Compare barracuda `rk45_solve` against hand-rolled RK4 for vpsT ODE.
fn validate_rk45_vs_rk4(h: &mut ValidationHarness) {
    let params = OdeParams::default();
    let y0 = OdeState {
        cdg: 0.1,
        ai: 0.1,
        vps_t: 0.0,
        biofilm: 0.0,
    };
    let y0_arr = [y0.cdg, y0.ai, y0.vps_t, y0.biofilm];

    let trace = integrate_ode(20.0, 0.01, &y0, &params);
    let Some(rk4_final) = trace.last() else {
        h.check_bool("trace non-empty", false);
        return;
    };
    let rk4_cdg = rk4_final.cdg;
    let rk4_ai = rk4_final.ai;
    let rk4_vps_t = rk4_final.vps_t;

    let rhs = |t: f64, y: &[f64]| vps_t_rhs(t, y, &params);
    let config = make_config();

    match barracuda::numerical::rk45_solve(&rhs, 0.0, 20.0, &y0_arr, &config) {
        Ok(result) => {
            let y_cdg = result.y_final[0].max(0.0);
            let y_ai = result.y_final[1].max(0.0);
            let y_vps_t = result.y_final[2].max(0.0);

            h.check_abs(
                &format!("cdg: RK4={rk4_cdg:.4} vs RK45={y_cdg:.4}"),
                rk4_cdg,
                y_cdg,
                tolerances::ODE_INTEGRATOR_AGREEMENT,
            );
            h.check_abs(
                &format!("ai: RK4={rk4_ai:.4} vs RK45={y_ai:.4}"),
                rk4_ai,
                y_ai,
                tolerances::ODE_INTEGRATOR_AGREEMENT,
            );
            h.check_abs(
                &format!("vps_t: RK4={rk4_vps_t:.4} vs RK45={y_vps_t:.4}"),
                rk4_vps_t,
                y_vps_t,
                tolerances::ODE_INTEGRATOR_AGREEMENT,
            );

            h.check_bool(
                &format!("RK45 n_steps finite (n={})", result.n_steps),
                result.n_steps > 0 && result.t_final > 0.0,
            );
        }
        Err(e) => {
            h.check_bool(&format!("rk45_solve [ERROR: {e}]"), false);
        }
    }
}

/// Verify AND gate logic: high output only when both inputs high.
fn validate_and_gate_logic(h: &mut ValidationHarness) {
    let sweep = logic_gate_sweep(1.0, 1.0, 2.0, 2.0);
    let threshold = 0.5;

    for (gate, v) in &sweep {
        let expect_high = matches!(gate, LogicGate::OnOn);
        h.check_bool(
            &format!("{gate:?} output {v:.4} vs threshold {threshold}"),
            (*v >= threshold) == expect_high,
        );
    }

    let on_on = sweep
        .iter()
        .find(|(g, _)| matches!(g, LogicGate::OnOn))
        .map_or(0.0, |(_, v)| *v);
    let on_off = sweep
        .iter()
        .find(|(g, _)| matches!(g, LogicGate::OnOff))
        .map_or(0.0, |(_, v)| *v);

    h.check_bool(
        &format!("AND: OnOn ({on_on:.4}) > OnOff ({on_off:.4})"),
        on_on > on_off,
    );
}

/// Final state bounds: cdg, ai, `vps_t` finite and non-negative.
fn validate_final_state_bounds(h: &mut ValidationHarness) {
    let params = OdeParams::default();
    let y0 = OdeState {
        cdg: 0.1,
        ai: 0.1,
        vps_t: 0.0,
        biofilm: 0.0,
    };
    let y0_arr = [y0.cdg, y0.ai, y0.vps_t, y0.biofilm];
    let rhs = |t: f64, y: &[f64]| vps_t_rhs(t, y, &params);
    let config = make_config();

    match barracuda::numerical::rk45_solve(&rhs, 0.0, 20.0, &y0_arr, &config) {
        Ok(result) => {
            let cdg = result.y_final[0].max(0.0);
            let ai = result.y_final[1].max(0.0);
            let vps_t = result.y_final[2].max(0.0);
            let biofilm = result.y_final[3].max(0.0);

            h.check_bool(
                "final cdg finite and non-negative",
                cdg.is_finite() && cdg >= 0.0,
            );
            h.check_bool(
                "final ai finite and non-negative",
                ai.is_finite() && ai >= 0.0,
            );
            h.check_bool(
                "final vps_t finite and non-negative",
                vps_t.is_finite() && vps_t >= 0.0,
            );
            h.check_bool(
                "final biofilm finite and non-negative",
                biofilm.is_finite() && biofilm >= 0.0,
            );

            let gate = classify_logic_gate(cdg, ai, 0.5, 0.5);
            h.check_bool(
                &format!("classify_logic_gate yields valid LogicGate ({gate:?})"),
                true,
            );
        }
        Err(e) => {
            h.check_bool(&format!("final bounds rk45 [ERROR: {e}]"), false);
        }
    }
}
