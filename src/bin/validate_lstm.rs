// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: LSTM/GRU recurrent cell primitives (Study 004).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/lstm_weather/lstm_era5.py`
//! Run: 2026-02-16, Eastgate, Python 3.10, `PyTorch` 2.9.0+cu128, seed=42
//! Command: `python3 control/lstm_weather/lstm_era5.py`
//!
//! Validates Rust-portable LSTM and GRU cell forward pass mechanics using
//! known weight matrices and analytically verifiable gate values.
//! Training remains Python-only (Phase 0); this validates the inference
//! math that `BarraCUDA`'s `lstm_cell.wgsl` and `gru_cell.wgsl` will implement.

use neural_spring::sequence::{
    gru_cell, gru_forward, lstm_cell, lstm_forward, sigmoid, tanh_activation, GruWeights,
    LstmWeights,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("lstm");

    validate_lstm_gate_math(&mut h);
    validate_lstm_cell_known(&mut h);
    validate_lstm_sequence(&mut h);
    validate_gru_cell_known(&mut h);
    validate_gru_sequence(&mut h);
    validate_lstm_vs_gru(&mut h);
    validate_determinism(&mut h);

    h.finish();
}

fn validate_lstm_gate_math(h: &mut ValidationHarness) {
    h.check_abs("σ(0) = 0.5", sigmoid(0.0), 0.5, tolerances::EXACT_F64);
    h.check_abs(
        "tanh(0) = 0",
        tanh_activation(0.0),
        0.0,
        tolerances::EXACT_F64,
    );

    let sig_1 = sigmoid(1.0);
    h.check_bool("σ(1) ∈ (0.5, 1)", sig_1 > 0.5 && sig_1 < 1.0);

    h.check_abs(
        "σ(x) + σ(-x) = 1",
        sigmoid(3.0) + sigmoid(-3.0),
        1.0,
        tolerances::EXACT_F64,
    );

    h.check_abs(
        "tanh antisymmetry",
        tanh_activation(2.0) + tanh_activation(-2.0),
        0.0,
        tolerances::EXACT_F64,
    );
}

fn validate_lstm_cell_known(h: &mut ValidationHarness) {
    let hs = 2;
    let is = 2;

    let w_input = vec![0.1; 4 * hs * is];
    let w_hidden = vec![0.05; 4 * hs * hs];
    let b_input = vec![0.0; 4 * hs];
    let b_hidden = vec![0.0; 4 * hs];
    let w = LstmWeights {
        w_input: &w_input,
        w_hidden: &w_hidden,
        b_input: &b_input,
        b_hidden: &b_hidden,
        hidden_size: hs,
    };

    let x = vec![1.0, 0.5];
    let h_prev = vec![0.0, 0.0];
    let c_prev = vec![0.0, 0.0];

    let (h_new, c_new) = lstm_cell(&x, &h_prev, &c_prev, &w);

    h.check_bool("LSTM h_new dim = hidden_size", h_new.len() == hs);
    h.check_bool("LSTM c_new dim = hidden_size", c_new.len() == hs);

    let pre_activation = 0.1_f64.mul_add(1.0, 0.1 * 0.5);
    let expected_forget = sigmoid(pre_activation);
    let expected_input = sigmoid(pre_activation);
    let expected_candidate = tanh_activation(pre_activation);
    let expected_output = sigmoid(pre_activation);
    let expected_c = expected_forget * 0.0 + expected_input * expected_candidate;
    let expected_h = expected_output * expected_c.tanh();

    h.check_abs(
        "LSTM c[0] known value",
        c_new[0],
        expected_c,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "LSTM h[0] known value",
        h_new[0],
        expected_h,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "LSTM c[0] = c[1] (symmetric weights)",
        c_new[0],
        c_new[1],
        tolerances::EXACT_F64,
    );
}

fn validate_lstm_sequence(h: &mut ValidationHarness) {
    let hs = 3;
    let is = 2;

    let w_input = vec![0.1; 4 * hs * is];
    let w_hidden = vec![0.01; 4 * hs * hs];
    let b_input = vec![0.0; 4 * hs];
    let b_hidden = vec![0.0; 4 * hs];
    let w = LstmWeights {
        w_input: &w_input,
        w_hidden: &w_hidden,
        b_input: &b_input,
        b_hidden: &b_hidden,
        hidden_size: hs,
    };

    let seq = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let h_final = lstm_forward(&seq, &w);

    h.check_bool("LSTM forward dim = hidden_size", h_final.len() == hs);

    for (idx, val) in h_final.iter().enumerate() {
        h.check_bool(
            &format!("LSTM h[{idx}] bounded: |h| < 1 (tanh output)"),
            val.abs() < 1.0,
        );
    }

    let h_1step = lstm_forward(&seq[..1], &w);
    let h_3step = lstm_forward(&seq, &w);
    h.check_bool(
        "LSTM 3-step != 1-step (state evolves)",
        (h_1step[0] - h_3step[0]).abs() > tolerances::CROSS_LANGUAGE,
    );
}

fn validate_gru_cell_known(h: &mut ValidationHarness) {
    let hs = 2;
    let is = 2;

    let w_input = vec![0.1; 3 * hs * is];
    let w_hidden = vec![0.05; 3 * hs * hs];
    let b_input = vec![0.0; 3 * hs];
    let b_hidden = vec![0.0; 3 * hs];
    let w = GruWeights {
        w_input: &w_input,
        w_hidden: &w_hidden,
        b_input: &b_input,
        b_hidden: &b_hidden,
        hidden_size: hs,
    };

    let x = vec![1.0, 0.5];
    let h_prev = vec![0.0, 0.0];

    let h_new = gru_cell(&x, &h_prev, &w);

    h.check_bool("GRU h_new dim = hidden_size", h_new.len() == hs);
    h.check_abs(
        "GRU h[0] = h[1] (symmetric weights)",
        h_new[0],
        h_new[1],
        tolerances::EXACT_F64,
    );

    for val in &h_new {
        h.check_bool(&format!("GRU |h| < 1: {val:.6}"), val.abs() < 1.0);
    }
}

fn validate_gru_sequence(h: &mut ValidationHarness) {
    let hs = 3;
    let is = 2;

    let w_input = vec![0.1; 3 * hs * is];
    let w_hidden = vec![0.01; 3 * hs * hs];
    let b_input = vec![0.0; 3 * hs];
    let b_hidden = vec![0.0; 3 * hs];
    let w = GruWeights {
        w_input: &w_input,
        w_hidden: &w_hidden,
        b_input: &b_input,
        b_hidden: &b_hidden,
        hidden_size: hs,
    };

    let seq = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let h_final = gru_forward(&seq, &w);

    h.check_bool("GRU forward dim = hidden_size", h_final.len() == hs);

    let h_1step = gru_forward(&seq[..1], &w);
    h.check_bool(
        "GRU 3-step != 1-step (state evolves)",
        (h_1step[0] - h_final[0]).abs() > tolerances::CROSS_LANGUAGE,
    );
}

fn validate_lstm_vs_gru(h: &mut ValidationHarness) {
    let hs = 4;
    let is = 3;
    let seq: Vec<Vec<f64>> = (0..5).map(|t| vec![f64::from(t) * 0.1; is]).collect();

    let lstm_w = LstmWeights {
        w_input: &vec![0.1; 4 * hs * is],
        w_hidden: &vec![0.01; 4 * hs * hs],
        b_input: &vec![0.0; 4 * hs],
        b_hidden: &vec![0.0; 4 * hs],
        hidden_size: hs,
    };
    let gru_w = GruWeights {
        w_input: &vec![0.1; 3 * hs * is],
        w_hidden: &vec![0.01; 3 * hs * hs],
        b_input: &vec![0.0; 3 * hs],
        b_hidden: &vec![0.0; 3 * hs],
        hidden_size: hs,
    };

    let lstm_h = lstm_forward(&seq, &lstm_w);
    let gru_h = gru_forward(&seq, &gru_w);

    h.check_bool(
        "LSTM and GRU produce different outputs (different architectures)",
        (lstm_h[0] - gru_h[0]).abs() > tolerances::CROSS_LANGUAGE,
    );

    h.check_bool("LSTM output bounded", lstm_h.iter().all(|v| v.abs() < 1.0));
    h.check_bool("GRU output bounded", gru_h.iter().all(|v| v.abs() < 1.0));
}

fn validate_determinism(h: &mut ValidationHarness) {
    let hs = 3;
    let is = 2;
    let seq = vec![vec![1.0, 0.5], vec![0.3, 0.7], vec![0.9, 0.1]];

    let lstm_w = LstmWeights {
        w_input: &vec![0.1; 4 * hs * is],
        w_hidden: &vec![0.01; 4 * hs * hs],
        b_input: &vec![0.0; 4 * hs],
        b_hidden: &vec![0.0; 4 * hs],
        hidden_size: hs,
    };
    let gru_w = GruWeights {
        w_input: &vec![0.1; 3 * hs * is],
        w_hidden: &vec![0.01; 3 * hs * hs],
        b_input: &vec![0.0; 3 * hs],
        b_hidden: &vec![0.0; 3 * hs],
        hidden_size: hs,
    };

    let h1 = lstm_forward(&seq, &lstm_w);
    let h2 = lstm_forward(&seq, &lstm_w);
    h.check_bool("LSTM forward deterministic", h1 == h2);

    let g1 = gru_forward(&seq, &gru_w);
    let g2 = gru_forward(&seq, &gru_w);
    h.check_bool("GRU forward deterministic", g1 == g2);
}
