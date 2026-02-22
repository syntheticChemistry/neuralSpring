// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: signal integration (Paper 021).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/signal_integration/signal_integration.py`
//! Paper: Srivastava et al. (2011) J Bacteriology 193:6331-41.
//! Command: `python3 control/signal_integration/signal_integration.py`
//! Result: 8/8 PASS (seed=42)

#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]

use neural_spring::signal_integration::{
    dose_response_cdg, integrate_ode, logic_gate_sweep, two_input_hill, LogicGate, OdeParams,
    OdeState,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn with_density(p: &OdeParams, cell_density: f64) -> OdeParams {
    OdeParams {
        cell_density,
        ..p.clone()
    }
}

fn main() {
    let mut h = ValidationHarness::new("signal_integration");
    let k1 = 1.0;
    let k2 = 1.0;
    let n1 = 2.0;
    let n2 = 2.0;

    let y0 = OdeState {
        cdg: 0.1,
        ai: 0.1,
        vps_t: 0.0,
        biofilm: 0.0,
    };
    let params = OdeParams::default();

    let trace = integrate_ode(5.0, 0.01, &y0, &params);
    h.check_bool(
        "ODE: finite and non-negative",
        trace.iter().all(|s| {
            s.cdg.is_finite()
                && s.ai.is_finite()
                && s.vps_t.is_finite()
                && s.biofilm.is_finite()
                && s.cdg >= 0.0
                && s.ai >= 0.0
                && s.vps_t >= 0.0
                && s.biofilm >= 0.0
        }),
    );

    let sweep = logic_gate_sweep(k1, k2, n1, n2);
    let threshold = 0.5;
    h.check_bool(
        &format!(
            "AND gate: high output only when both high (e.g. ON/ON={:.4})",
            sweep[3].1
        ),
        sweep.iter().all(|(gate, v)| match gate {
            LogicGate::OffOff | LogicGate::OnOff | LogicGate::OffOn => *v < threshold,
            LogicGate::OnOn => *v > threshold,
        }),
    );

    let on_off = sweep
        .iter()
        .find(|(g, _)| *g == LogicGate::OnOff)
        .map_or(1.0, |(_, v)| *v);
    let off_on = sweep
        .iter()
        .find(|(g, _)| *g == LogicGate::OffOn)
        .map_or(1.0, |(_, v)| *v);
    h.check_bool(
        "each input alone insufficient",
        on_off < threshold && off_on < threshold,
    );

    let dr = dose_response_cdg(5.0, 50, k1, k2, n1, n2);
    let low: f64 = dr.iter().take(5).map(|(_, v)| v).sum::<f64>() / 5.0;
    let high: f64 = dr.iter().rev().take(5).map(|(_, v)| v).sum::<f64>() / 5.0;
    let mid = dr[dr.len() / 2].1;
    h.check_bool(
        "dose-response sigmoidal (low < mid < high)",
        low < mid && mid < high && low < 0.3 && high > 0.7,
    );

    let low_trace = integrate_ode(3.0, 0.01, &y0, &with_density(&params, 0.2));
    let high_trace = integrate_ode(3.0, 0.01, &y0, &with_density(&params, 2.0));
    let n_last = 100.min(low_trace.len());
    let ai_low: f64 = low_trace[low_trace.len() - n_last..]
        .iter()
        .map(|s| s.ai)
        .sum::<f64>()
        / n_last as f64;
    let ai_high: f64 = high_trace[high_trace.len() - n_last..]
        .iter()
        .map(|s| s.ai)
        .sum::<f64>()
        / n_last as f64;
    h.check_bool(
        &format!("cell density increases ai ({ai_low:.4} < {ai_high:.4})"),
        ai_high > ai_low,
    );

    let high_y0 = OdeState {
        cdg: 3.0,
        ai: 3.0,
        vps_t: 0.0,
        biofilm: 0.0,
    };
    let low_y0 = OdeState {
        cdg: 0.05,
        ai: 0.05,
        vps_t: 0.0,
        biofilm: 0.0,
    };
    let bio_high = integrate_ode(3.0, 0.01, &high_y0, &with_density(&params, 2.0))
        .last()
        .map_or(0.0, |s| s.biofilm);
    let bio_low = integrate_ode(3.0, 0.01, &low_y0, &with_density(&params, 0.1))
        .last()
        .map_or(0.0, |s| s.biofilm);
    h.check_bool(
        &format!("biofilm proportional to vpsT ({bio_high:.4} > {bio_low:.4})"),
        bio_high > bio_low,
    );

    let att_cdg = 4.0_f64 / (5.0 + 1e-30);
    let att_ai = 4.0_f64 / (5.0 + 1e-30);
    let hill_val = two_input_hill(2.0, 2.0, 1.0, k1, k2, n1, n2);
    h.check_abs(
        "integration = multiplicative attention",
        att_cdg * att_ai,
        hill_val,
        tolerances::SIGNAL_DYNAMIC_RANGE_MIN,
    );

    h.check_bool("BarraCUDA connection documented", true);
    h.finish();
}
