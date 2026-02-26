// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: Information flow analysis (baseCamp nS-02).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## baseCamp Sub-thesis 02
//!
//! Information Flow as Wave Propagation in Neural Lattices.
//! Experiments nS-201 through nS-206.
//!
//! ## Provenance
//!
//! No Python baseline — these are novel experiments. Validated against
//! analytical known-values (information theory, Anderson diagnostics).

#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]

use neural_spring::information_flow::{
    attention_spectral_analysis, attention_to_hamiltonian, depth_scale, gate_disorder_parameter,
    gate_saturation, information_ipr, jacobian_spectral_radius, mlp_signal_propagation,
};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("information_flow");
    let mut rng = Rng::new(42);

    // ── nS-201: Depth scale for constant variance ────────────────────

    let constant_var = vec![1.0; 10];
    let xi_const = depth_scale(&constant_var);
    h.check_bool(
        "Constant variance: infinite depth scale",
        xi_const.is_infinite(),
    );

    // ── nS-201: Depth scale for decaying signal ──────────────────────

    let decaying_var: Vec<f64> = (0..10).map(|i| (-0.5 * f64::from(i)).exp()).collect();
    let xi_decay = depth_scale(&decaying_var);
    h.check_bool(
        "Decaying variance: finite depth scale",
        xi_decay > 0.0 && xi_decay < 10.0,
    );

    // ── nS-202: Gate disorder zero for constant gates ────────────────

    let constant_gates = vec![0.5; 100];
    let w_constant = gate_disorder_parameter(&constant_gates);
    h.check_abs(
        "Constant gates: zero disorder",
        w_constant,
        0.0,
        tolerances::EXACT_F64,
    );

    // ── nS-202: Gate disorder positive for spread gates ──────────────

    let spread_gates: Vec<f64> = (0..100).map(|_| rng.uniform()).collect();
    let w_spread = gate_disorder_parameter(&spread_gates);
    h.check_bool("Spread gates: positive disorder", w_spread > 0.0);

    // ── nS-202: Gate saturation ──────────────────────────────────────

    let gates = vec![0.01, 0.99, 0.5, 0.02, 0.98, 0.5];
    let sat = gate_saturation(&gates, 0.05);
    h.check_abs(
        "Gate saturation = 4/6",
        sat,
        4.0 / 6.0,
        tolerances::EXACT_F64,
    );

    // ── nS-203: Information IPR bounds ───────────────────────────────

    let uniform_act = vec![1.0; 16];
    let ipr_uniform = information_ipr(&uniform_act);
    h.check_bool("Uniform activation: IPR > 0", ipr_uniform > 0.0);

    let mut localized_act = vec![0.0; 16];
    localized_act[0] = 1.0;
    let ipr_localized = information_ipr(&localized_act);
    h.check_bool("Localized > uniform IPR", ipr_localized > ipr_uniform);

    // ── nS-204: Attention Hamiltonian symmetric ──────────────────────

    let n = 8;
    let attention: Vec<f64> = (0..n * n).map(|_| rng.uniform()).collect();
    let ham = attention_to_hamiltonian(&attention, n);
    let symmetric = (0..n)
        .all(|i| (0..n).all(|j| (ham[i * n + j] - ham[j * n + i]).abs() < tolerances::EXACT_F64));
    h.check_bool("Attention Hamiltonian is symmetric", symmetric);

    // ── nS-204: Attention spectral analysis finite ───────────────────

    let attn_result = attention_spectral_analysis(&attention, n);
    h.check_bool(
        "Attention eigenvalues all finite",
        attn_result.eigenvalues.iter().all(|&ev| ev.is_finite()),
    );
    h.check_bool("Attention IPR positive", attn_result.mean_ipr > 0.0);
    h.check_bool(
        "Attention level spacing in [0, 1]",
        (0.0..=1.0 + tolerances::EXACT_F64).contains(&attn_result.level_spacing_ratio),
    );

    // ── nS-201: MLP signal propagation ───────────────────────────────

    let input: Vec<f64> = (0..8).map(|_| rng.normal()).collect();
    let w1: Vec<f64> = (0..8 * 8).map(|_| rng.normal() * 0.5).collect();
    let w2: Vec<f64> = (0..8 * 8).map(|_| rng.normal() * 0.5).collect();
    let variances = mlp_signal_propagation(&input, &[&w1, &w2], &[8, 8]);

    h.check_bool(
        "MLP produces 3 variance measurements (input + 2 layers)",
        variances.len() == 3,
    );
    h.check_bool("Input variance positive", variances[0] > 0.0);

    // ── nS-206: Jacobian spectral radius ─────────────────────────────

    let small_n = 4;
    let weights: Vec<f64> = (0..small_n * small_n).map(|_| rng.normal() * 0.3).collect();
    let pre_act: Vec<f64> = (0..small_n).map(|_| rng.normal()).collect();
    let rho = jacobian_spectral_radius(&weights, &pre_act, small_n);
    h.check_bool(
        "Jacobian spectral radius finite and non-negative",
        rho.is_finite() && rho >= 0.0,
    );

    // ── nS-205: Hill activation analysis of LSTM-like gates ───────────

    let gate_values: Vec<f64> = (0..50)
        .map(|i| {
            let x = f64::from(i) / 49.0;
            1.0 / (1.0 + (-10.0 * (x - 0.5)).exp())
        })
        .collect();
    let sat_steep = gate_saturation(&gate_values, 0.05);
    h.check_bool("nS-205: steep sigmoid saturates", sat_steep > 0.0);

    let gentle_gates: Vec<f64> = (0..50)
        .map(|i| {
            let x = f64::from(i) / 49.0;
            1.0 / (1.0 + (-(x - 0.5)).exp())
        })
        .collect();
    let sat_gentle = gate_saturation(&gentle_gates, 0.05);
    h.check_bool(
        "nS-205: steep sigmoid has more saturation than gentle",
        sat_steep >= sat_gentle,
    );

    let disorder_steep = gate_disorder_parameter(&gate_values);
    let disorder_gentle = gate_disorder_parameter(&gentle_gates);
    h.check_bool(
        "nS-205: steeper gates produce higher disorder",
        disorder_steep >= disorder_gentle - tolerances::GATE_DISORDER_COMPARISON,
    );

    // ── nS-206: Edge-of-chaos systematic sigma_w sweep ───────────────

    let sweep_n = 4;
    let mut rho_values = Vec::new();
    for sigma_idx in 0_u64..5 {
        let sigma = 0.3f64.mul_add(sigma_idx as f64, 0.1);
        let sweep_w: Vec<f64> = (0..sweep_n * sweep_n)
            .map(|i| {
                let mut prng = Rng::new(42 + 10000 * sigma_idx + i as u64);
                prng.normal() * sigma
            })
            .collect();
        let sweep_pre: Vec<f64> = (0..sweep_n)
            .map(|i| {
                let mut prng = Rng::new(42 + 20000 + i as u64);
                prng.normal()
            })
            .collect();
        rho_values.push(jacobian_spectral_radius(&sweep_w, &sweep_pre, sweep_n));
    }
    h.check_bool(
        "nS-206: spectral radius finite for all sigma_w",
        rho_values.iter().all(|r| r.is_finite()),
    );
    h.check_bool(
        "nS-206: spectral radius increases with sigma_w",
        *rho_values.last().unwrap_or(&0.0)
            >= rho_values.first().unwrap_or(&f64::INFINITY)
                - tolerances::SPECTRAL_RADIUS_SWEEP_SLACK,
    );

    // ── nS-203: Layer-by-layer IPR trajectory ────────────────────────

    let multi_input: Vec<f64> = (0..8).map(|_| rng.normal()).collect();
    let multi_w1: Vec<f64> = (0..8 * 8).map(|_| rng.normal() * 0.5).collect();
    let multi_w2: Vec<f64> = (0..8 * 8).map(|_| rng.normal() * 0.5).collect();
    let multi_w3: Vec<f64> = (0..8 * 8).map(|_| rng.normal() * 0.5).collect();
    let multi_variances =
        mlp_signal_propagation(&multi_input, &[&multi_w1, &multi_w2, &multi_w3], &[8, 8, 8]);
    h.check_bool(
        "nS-203: 4 variance measurements (input + 3 layers)",
        multi_variances.len() == 4,
    );
    h.check_bool(
        "nS-203: all layer variances positive",
        multi_variances.iter().all(|&v| v > 0.0),
    );

    // ── Determinism ──────────────────────────────────────────────────

    let r1 = attention_spectral_analysis(&attention, n);
    let r2 = attention_spectral_analysis(&attention, n);
    h.check_bool(
        "Attention spectral analysis deterministic",
        r1.eigenvalues == r2.eigenvalues,
    );

    h.finish();
}
