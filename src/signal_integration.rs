// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(clippy::cast_precision_loss)]

//! Signal integration: cyclic di-GMP + quorum sensing in Vibrio cholerae.
//!
//! Port of `control/signal_integration/signal_integration.py`.
//!
//! Reproduces key dynamics from:
//! Srivastava et al. (2011)
//! "Integration of Cyclic di-GMP and Quorum Sensing in the Control of
//!  vpsT Expression in Vibrio cholerae"
//! J Bacteriology 193:6331-41.
//!
//! Core thesis: The vpsT promoter integrates two inputs (cdg AND ai)
//! as a biological AND gate — maps to multi-input attention.

use crate::rng::Rng;

const EPS: f64 = 1e-30;

/// Two-input Hill function: AND gate for vpsT activation.
///
/// `f(cdg, ai) = Vmax * (cdg^n1 / (K1^n1 + cdg^n1)) * (ai^n2 / (K2^n2 + ai^n2))`
#[must_use]
pub fn two_input_hill(cdg: f64, ai: f64, vmax: f64, k1: f64, k2: f64, n1: f64, n2: f64) -> f64 {
    let h1 = cdg.powf(n1) / (k1.powf(n1) + cdg.powf(n1) + EPS);
    let h2 = ai.powf(n2) / (k2.powf(n2) + ai.powf(n2) + EPS);
    vmax * h1 * h2
}

/// Logic gate classification: (`cdg_high`, `ai_high`) -> AND outcome.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LogicGate {
    /// Low/Low -> Low
    OffOff,
    /// High/Low -> Low
    OnOff,
    /// Low/High -> Low
    OffOn,
    /// High/High -> High
    OnOn,
}

/// Classify logic gate case from cdg and ai vs thresholds.
#[must_use]
pub fn classify_logic_gate(cdg: f64, ai: f64, cdg_thresh: f64, ai_thresh: f64) -> LogicGate {
    let cdg_high = cdg >= cdg_thresh;
    let ai_high = ai >= ai_thresh;
    match (cdg_high, ai_high) {
        (false, false) => LogicGate::OffOff,
        (true, false) => LogicGate::OnOff,
        (false, true) => LogicGate::OffOn,
        (true, true) => LogicGate::OnOn,
    }
}

/// Parameters for the vpsT regulatory ODE.
#[derive(Debug, Clone)]
pub struct OdeParams {
    pub cell_density: f64,
    pub cdg_synth: f64,
    pub cdg_deg: f64,
    pub ai_prod: f64,
    pub ai_decay: f64,
    pub vps_degradation: f64,
    pub vmax: f64,
    pub k1: f64,
    pub k2: f64,
    pub n1: f64,
    pub n2: f64,
    pub noise_scale: f64,
    pub seed: u64,
}

impl Default for OdeParams {
    fn default() -> Self {
        Self {
            cell_density: 1.0,
            cdg_synth: 0.5,
            cdg_deg: 0.2,
            ai_prod: 0.3,
            ai_decay: 0.1,
            vps_degradation: 0.3,
            vmax: 1.0,
            k1: 1.0,
            k2: 1.0,
            n1: 2.0,
            n2: 2.0,
            noise_scale: 0.0,
            seed: 42,
        }
    }
}

/// ODE state: [cdg, ai, vpsT, biofilm]
#[derive(Debug, Clone)]
pub struct OdeState {
    pub cdg: f64,
    pub ai: f64,
    pub vps_t: f64,
    pub biofilm: f64,
}

impl OdeState {
    const fn to_array(&self) -> [f64; 4] {
        [self.cdg, self.ai, self.vps_t, self.biofilm]
    }

    const fn from_array(a: [f64; 4]) -> Self {
        Self {
            cdg: a[0],
            ai: a[1],
            vps_t: a[2],
            biofilm: a[3],
        }
    }
}

fn ode_rhs(y: &[f64; 4], params: &OdeParams, rng: &mut Rng) -> [f64; 4] {
    let cdg = y[0];
    let ai = y[1];
    let vps_t = y[2];

    let noise = if params.noise_scale > 0.0 {
        params.noise_scale * rng.normal()
    } else {
        0.0
    };

    let d_cdg = params.cdg_deg.mul_add(-cdg, params.cdg_synth) + noise;
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

    [d_cdg, d_ai, d_vps_t, d_biofilm]
}

/// Single RK4 step.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
fn rk4_step(y: &[f64; 4], _t: f64, dt: f64, params: &OdeParams, rng: &mut Rng) -> [f64; 4] {
    let half_dt = 0.5 * dt;
    let k1 = ode_rhs(y, params, rng);
    let y2: [f64; 4] = std::array::from_fn(|i| half_dt.mul_add(k1[i], y[i]));
    let k2 = ode_rhs(&y2, params, rng);
    let y3: [f64; 4] = std::array::from_fn(|i| half_dt.mul_add(k2[i], y[i]));
    let k3 = ode_rhs(&y3, params, rng);
    let y4: [f64; 4] = std::array::from_fn(|i| dt.mul_add(k3[i], y[i]));
    let k4 = ode_rhs(&y4, params, rng);

    let dt6 = dt / 6.0;
    std::array::from_fn(|i| {
        let sum_k = 2.0f64.mul_add(k3[i], 2.0f64.mul_add(k2[i], k1[i]) + k4[i]);
        dt6.mul_add(sum_k, y[i])
    })
}

/// Integrate vpsT regulatory ODE with RK4.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#[must_use]
pub fn integrate_ode(t_end: f64, dt: f64, y0: &OdeState, params: &OdeParams) -> Vec<OdeState> {
    let n_steps = (t_end / dt).ceil() as usize + 1;
    let mut trace = Vec::with_capacity(n_steps);
    trace.push(y0.clone());

    let mut rng = Rng::new(params.seed);
    let mut y = y0.to_array();

    for i in 1..n_steps {
        let t = (i - 1) as f64 * dt;
        y = rk4_step(&y, t, dt, params, &mut rng);
        for v in &mut y {
            *v = (*v).max(0.0);
        }
        trace.push(OdeState::from_array(y));
    }

    trace
}

/// Sweep cdg and ai to get AND gate outputs.
#[must_use]
pub fn logic_gate_sweep(k1: f64, k2: f64, n1: f64, n2: f64) -> [(LogicGate, f64); 4] {
    let low = 0.01_f64;
    let high = 5.0_f64;

    [
        (
            LogicGate::OffOff,
            two_input_hill(low, low, 1.0, k1, k2, n1, n2),
        ),
        (
            LogicGate::OnOff,
            two_input_hill(high, low, 1.0, k1, k2, n1, n2),
        ),
        (
            LogicGate::OffOn,
            two_input_hill(low, high, 1.0, k1, k2, n1, n2),
        ),
        (
            LogicGate::OnOn,
            two_input_hill(high, high, 1.0, k1, k2, n1, n2),
        ),
    ]
}

/// Dose-response: sweep cdg with ai fixed.
#[must_use]
pub fn dose_response_cdg(
    ai_fixed: f64,
    n_points: usize,
    k1: f64,
    k2: f64,
    n1: f64,
    n2: f64,
) -> Vec<(f64, f64)> {
    let cdg_min = 0.01_f64;
    let cdg_max = 10.0_f64;
    (0..n_points)
        .map(|i| {
            let frac = i as f64 / (n_points - 1).max(1) as f64;
            let cdg = cdg_min * (cdg_max / cdg_min).powf(frac);
            let v = two_input_hill(cdg, ai_fixed, 1.0, k1, k2, n1, n2);
            (cdg, v)
        })
        .collect()
}

/// Dose-response: sweep ai with cdg fixed.
#[must_use]
pub fn dose_response_ai(
    cdg_fixed: f64,
    n_points: usize,
    k1: f64,
    k2: f64,
    n1: f64,
    n2: f64,
) -> Vec<(f64, f64)> {
    let ai_min = 0.01_f64;
    let ai_max = 10.0_f64;
    (0..n_points)
        .map(|i| {
            let frac = i as f64 / (n_points - 1).max(1) as f64;
            let ai = ai_min * (ai_max / ai_min).powf(frac);
            let v = two_input_hill(cdg_fixed, ai, 1.0, k1, k2, n1, n2);
            (ai, v)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn and_gate_high_only_when_both() {
        let sweep = logic_gate_sweep(1.0, 1.0, 2.0, 2.0);
        let threshold = 0.5;
        for (gate, v) in sweep {
            match gate {
                LogicGate::OffOff | LogicGate::OnOff | LogicGate::OffOn => {
                    assert!(v < threshold, "{gate:?} should be low, got {v}");
                }
                LogicGate::OnOn => {
                    assert!(v > threshold, "ON/ON should be high, got {v}");
                }
            }
        }
    }

    #[test]
    fn two_input_hill_bounds() {
        let v = two_input_hill(1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn ode_finite_nonneg() {
        let y0 = OdeState {
            cdg: 0.1,
            ai: 0.1,
            vps_t: 0.0,
            biofilm: 0.0,
        };
        let params = OdeParams::default();
        let trace = integrate_ode(2.0, 0.01, &y0, &params);
        for s in &trace {
            assert!(s.cdg >= 0.0 && s.cdg.is_finite());
            assert!(s.ai >= 0.0 && s.ai.is_finite());
            assert!(s.vps_t >= 0.0 && s.vps_t.is_finite());
            assert!(s.biofilm >= 0.0 && s.biofilm.is_finite());
        }
    }

    #[test]
    fn dose_response_sigmoidal() {
        let dr = dose_response_cdg(5.0, 50, 1.0, 1.0, 2.0, 2.0);
        let low: f64 = dr.iter().take(5).map(|(_, v)| v).sum::<f64>() / 5.0;
        let high: f64 = dr.iter().rev().take(5).map(|(_, v)| v).sum::<f64>() / 5.0;
        assert!(low < 0.3);
        assert!(high > 0.7);
    }
}
