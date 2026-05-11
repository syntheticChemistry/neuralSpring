// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sequence forecasting primitives.
//!
//! Rust implementations of time-series operations used in Exp 003
//! (LSTM/GRU weather forecasting) and Study 004 (ERA5 weather).
//!
//! ## `BarraCUDA` Target
//!
//! - `lstm_cell.wgsl` — gate computations (forget, input, cell, output)
//! - `gru_cell.wgsl` — reset, update, candidate gates
//! - `gemm_f64.wgsl` — weight matrices

use std::f64::consts::PI;

/// Create input/target pairs for sequence forecasting.
///
/// Given `data` of length N, produces:
/// - inputs: windows of `seq_len` consecutive values
/// - targets: the value at offset `horizon` after each window
///
/// Matches Python `create_sequences(data, seq_len, horizon)`.
#[must_use]
pub fn create_sequences(data: &[f64], seq_len: usize, horizon: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = data.len();

    (seq_len..=(n.saturating_sub(horizon)))
        .filter(|&i| i + horizon - 1 < n)
        .map(|i| (data[i - seq_len..i].to_vec(), data[i + horizon - 1]))
        .unzip()
}

/// Persistence forecast: predict the last observed value for each window.
///
/// For multivariate input `(batch, seq_len, features)`, returns the last
/// timestep's first feature. For univariate `(batch, seq_len)`, returns
/// the last timestep.
#[must_use]
pub fn persistence_forecast(windows: &[Vec<f64>]) -> Vec<f64> {
    windows
        .iter()
        .map(|w| w.last().copied().unwrap_or_default())
        .collect()
}

/// Seasonal climatology: sinusoidal temperature model for Michigan.
///
/// `tmax(doy) = 8.5 + 15.0 * sin(2π(doy - 100) / 365)`
///
/// Matches the Python `generate_synthetic_weather` seasonal component.
#[must_use]
pub fn seasonal_tmax(doy: u32) -> f64 {
    let doy_f = f64::from(doy % 365);
    15.0f64.mul_add((2.0 * PI * (doy_f - 100.0) / 365.0).sin(), 8.5)
}

/// Sigmoid activation (used in LSTM/GRU gates).
///
/// Delegates to [`crate::primitives::sigmoid`] (numerically stable).
#[cfg(feature = "barracuda")]
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    crate::primitives::sigmoid(x)
}

/// Tanh activation (used in LSTM cell state).
#[must_use]
pub fn tanh_activation(x: f64) -> f64 {
    x.tanh()
}

/// Packed LSTM weight/bias parameters.
///
/// Mirrors `PyTorch`'s `nn.LSTMCell` layout: 4 gates (forget, input,
/// candidate, output) packed in `[4*hidden, input]` and `[4*hidden, hidden]`.
pub struct LstmWeights<'a> {
    /// Input-to-hidden weights `[4*hidden_size, input_size]` row-major.
    pub w_input: &'a [f64],
    /// Hidden-to-hidden weights `[4*hidden_size, hidden_size]` row-major.
    pub w_hidden: &'a [f64],
    /// Input-to-hidden bias `[4*hidden_size]`.
    pub b_input: &'a [f64],
    /// Hidden-to-hidden bias `[4*hidden_size]`.
    pub b_hidden: &'a [f64],
    /// Hidden dimension.
    pub hidden_size: usize,
}

/// LSTM cell forward pass.
///
/// Computes one timestep of an LSTM cell given input `x`, previous hidden
/// state `h_prev`, and previous cell state `c_prev`.
///
/// Gate equations (`PyTorch` convention — concatenated `[h, x]` input):
///   - `f = sigmoid(W_f · [h, x] + b_f)`  (forget)
///   - `i = sigmoid(W_i · [h, x] + b_i)`  (input)
///   - `g = tanh(W_g · [h, x] + b_g)`     (candidate)
///   - `o = sigmoid(W_o · [h, x] + b_o)`  (output)
///   - `c = f * c_prev + i * g`
///   - `h = o * tanh(c)`
///
/// Returns `(h_new, c_new)`.
#[cfg(feature = "barracuda")]
#[must_use]
pub fn lstm_cell(
    x: &[f64],
    h_prev: &[f64],
    c_prev: &[f64],
    w: &LstmWeights<'_>,
) -> (Vec<f64>, Vec<f64>) {
    let input_size = x.len();
    let hs = w.hidden_size;

    let gates: Vec<f64> = (0..4 * hs)
        .map(|gate_idx| {
            let input_row = &w.w_input[gate_idx * input_size..(gate_idx + 1) * input_size];
            let hidden_row = &w.w_hidden[gate_idx * hs..(gate_idx + 1) * hs];
            let base = w.b_input[gate_idx] + w.b_hidden[gate_idx];
            let input_dot: f64 = input_row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            let hidden_dot: f64 = hidden_row
                .iter()
                .zip(h_prev.iter())
                .map(|(a, b)| a * b)
                .sum();
            base + input_dot + hidden_dot
        })
        .collect();

    let mut h_new = vec![0.0; hs];
    let mut c_new = vec![0.0; hs];

    for idx in 0..hs {
        let forget = sigmoid(gates[idx]);
        let inp = sigmoid(gates[hs + idx]);
        let cand = tanh_activation(gates[2 * hs + idx]);
        let out = sigmoid(gates[3 * hs + idx]);

        c_new[idx] = forget.mul_add(c_prev[idx], inp * cand);
        h_new[idx] = out * c_new[idx].tanh();
    }

    (h_new, c_new)
}

/// Packed GRU weight/bias parameters.
///
/// Mirrors `PyTorch`'s `nn.GRUCell` layout: 3 gates (reset, update, new)
/// packed in `[3*hidden, input]` and `[3*hidden, hidden]`.
pub struct GruWeights<'a> {
    /// Input-to-hidden weights `[3*hidden_size, input_size]` row-major.
    pub w_input: &'a [f64],
    /// Hidden-to-hidden weights `[3*hidden_size, hidden_size]` row-major.
    pub w_hidden: &'a [f64],
    /// Input-to-hidden bias `[3*hidden_size]`.
    pub b_input: &'a [f64],
    /// Hidden-to-hidden bias `[3*hidden_size]`.
    pub b_hidden: &'a [f64],
    /// Hidden dimension.
    pub hidden_size: usize,
}

/// GRU cell forward pass.
///
/// Gate equations (`PyTorch` convention):
///   - `r = sigmoid(W_ir · x + b_ir + W_hr · h + b_hr)`  (reset)
///   - `z = sigmoid(W_iz · x + b_iz + W_hz · h + b_hz)`  (update)
///   - `n = tanh(W_in · x + b_in + r * (W_hn · h + b_hn))` (new)
///   - `h' = (1 - z) * n + z * h`
///
/// Returns `h_new`.
#[cfg(feature = "barracuda")]
#[must_use]
pub fn gru_cell(x: &[f64], h_prev: &[f64], w: &GruWeights<'_>) -> Vec<f64> {
    let input_size = x.len();
    let hs = w.hidden_size;

    let compute_proj = |gate_idx: usize| -> (f64, f64) {
        let input_row = &w.w_input[gate_idx * input_size..(gate_idx + 1) * input_size];
        let hidden_row = &w.w_hidden[gate_idx * hs..(gate_idx + 1) * hs];
        let input_proj: f64 = w.b_input[gate_idx]
            + input_row
                .iter()
                .zip(x.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        let hidden_proj: f64 = w.b_hidden[gate_idx]
            + hidden_row
                .iter()
                .zip(h_prev.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        (input_proj, hidden_proj)
    };

    (0..hs)
        .map(|idx| {
            let (ih_r, hh_r) = compute_proj(idx);
            let (ih_z, hh_z) = compute_proj(hs + idx);
            let (ih_n, hh_n) = compute_proj(2 * hs + idx);
            let reset = sigmoid(ih_r + hh_r);
            let update = sigmoid(ih_z + hh_z);
            let new_gate = tanh_activation(reset.mul_add(hh_n, ih_n));
            (1.0 - update).mul_add(new_gate, update * h_prev[idx])
        })
        .collect()
}

/// Process a full sequence through an LSTM, returning final hidden state.
#[cfg(feature = "barracuda")]
#[must_use]
pub fn lstm_forward(sequence: &[Vec<f64>], w: &LstmWeights<'_>) -> Vec<f64> {
    let mut h = vec![0.0; w.hidden_size];
    let mut c = vec![0.0; w.hidden_size];
    for x in sequence {
        let (h_new, c_new) = lstm_cell(x, &h, &c, w);
        h = h_new;
        c = c_new;
    }
    h
}

/// Process a full sequence through a GRU, returning final hidden state.
#[cfg(feature = "barracuda")]
#[must_use]
pub fn gru_forward(sequence: &[Vec<f64>], w: &GruWeights<'_>) -> Vec<f64> {
    let mut h = vec![0.0; w.hidden_size];
    for x in sequence {
        h = gru_cell(x, &h, w);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;
    use approx::assert_relative_eq;

    #[test]
    fn create_sequences_basic() {
        let data: Vec<f64> = (0..10).map(f64::from).collect();
        let (inputs, targets) = create_sequences(&data, 3, 1);
        assert_eq!(inputs[0], vec![0.0, 1.0, 2.0]);
        assert_relative_eq!(targets[0], 3.0);
        assert_eq!(inputs.len(), targets.len());
    }

    #[test]
    fn create_sequences_horizon() {
        let data: Vec<f64> = (0..20).map(f64::from).collect();
        let (inputs, targets) = create_sequences(&data, 5, 3);
        assert_eq!(inputs[0], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_relative_eq!(targets[0], 7.0);
    }

    #[test]
    fn persistence_returns_last() {
        let windows = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let pred = persistence_forecast(&windows);
        assert_relative_eq!(pred[0], 3.0);
        assert_relative_eq!(pred[1], 6.0);
    }

    #[test]
    fn seasonal_tmax_summer_peak() {
        let summer = seasonal_tmax(190);
        let winter = seasonal_tmax(10);
        assert!(
            summer > winter,
            "summer ({summer}) should be warmer than winter ({winter})"
        );
        assert!(summer > 20.0, "summer should be > 20°C");
        assert!(winter < 0.0, "winter should be < 0°C");
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn sigmoid_bounds() {
        assert_relative_eq!(sigmoid(0.0), 0.5);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn sigmoid_symmetry() {
        let x = 2.5;
        assert_relative_eq!(
            sigmoid(x) + sigmoid(-x),
            1.0,
            epsilon = tolerances::ZERO_DETECTION
        );
    }

    #[test]
    fn tanh_bounds() {
        assert_relative_eq!(tanh_activation(0.0), 0.0);
        assert!(tanh_activation(100.0) > 0.99);
        assert!(tanh_activation(-100.0) < -0.99);
    }

    #[test]
    fn sequence_ops_deterministic() {
        let data: Vec<f64> = (0..100).map(f64::from).collect();
        let (inp1, tgt1) = create_sequences(&data, 14, 1);
        let (inp2, tgt2) = create_sequences(&data, 14, 1);
        assert_eq!(inp1, inp2, "create_sequences must be bit-identical");
        assert_eq!(tgt1, tgt2, "create_sequences targets must be bit-identical");

        let pred1 = persistence_forecast(&inp1);
        let pred2 = persistence_forecast(&inp2);
        assert_eq!(pred1, pred2, "persistence must be bit-identical");
    }

    #[test]
    fn seasonal_tmax_deterministic() {
        let run1: Vec<f64> = (0..365).map(seasonal_tmax).collect();
        let run2: Vec<f64> = (0..365).map(seasonal_tmax).collect();
        assert_eq!(run1, run2, "seasonal_tmax must be bit-identical");
    }

    #[cfg(feature = "barracuda")]
    fn make_lstm_weights(
        hs: usize,
        is: usize,
        val: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            vec![val; 4 * hs * is],
            vec![val; 4 * hs * hs],
            vec![0.0; 4 * hs],
            vec![0.0; 4 * hs],
        )
    }

    #[cfg(feature = "barracuda")]
    fn make_gru_weights(
        hs: usize,
        is: usize,
        val: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            vec![val; 3 * hs * is],
            vec![val; 3 * hs * hs],
            vec![0.0; 3 * hs],
            vec![0.0; 3 * hs],
        )
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn lstm_cell_zero_input() {
        let hs = 2;
        let (wi, wh, bi, bh) = make_lstm_weights(hs, 3, 0.0);
        let w = LstmWeights {
            w_input: &wi,
            w_hidden: &wh,
            b_input: &bi,
            b_hidden: &bh,
            hidden_size: hs,
        };

        let (h_new, c_new) = lstm_cell(&[0.0; 3], &[0.0; 2], &[0.0; 2], &w);

        assert_eq!(h_new.len(), hs);
        assert_eq!(c_new.len(), hs);
        for &v in &c_new {
            assert_relative_eq!(v, 0.0, epsilon = tolerances::ZERO_DETECTION);
        }
        for &v in &h_new {
            assert_relative_eq!(v, 0.0, epsilon = tolerances::ZERO_DETECTION);
        }
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn lstm_cell_nonzero_bias() {
        let wi = vec![0.0; 4];
        let wh = vec![0.0; 4];
        let bi = vec![1.0, 1.0, 0.0, 0.0];
        let bh = vec![0.0; 4];
        let w = LstmWeights {
            w_input: &wi,
            w_hidden: &wh,
            b_input: &bi,
            b_hidden: &bh,
            hidden_size: 1,
        };

        let (h_new, c_new) = lstm_cell(&[1.0], &[0.0], &[0.0], &w);

        assert_relative_eq!(c_new[0], 0.0, epsilon = tolerances::EXACT_F64);
        assert_relative_eq!(h_new[0], 0.0, epsilon = tolerances::EXACT_F64);
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn gru_cell_zero_input() {
        let hs = 2;
        let (wi, wh, bi, bh) = make_gru_weights(hs, 3, 0.0);
        let w = GruWeights {
            w_input: &wi,
            w_hidden: &wh,
            b_input: &bi,
            b_hidden: &bh,
            hidden_size: hs,
        };

        let h_new = gru_cell(&[0.0; 3], &[0.0; 2], &w);
        assert_eq!(h_new.len(), hs);
        for &v in &h_new {
            assert_relative_eq!(v, 0.0, epsilon = tolerances::ZERO_DETECTION);
        }
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn lstm_forward_sequence() {
        let hs = 2;
        let (wi, wh, bi, bh) = make_lstm_weights(hs, 1, 0.1);
        let w = LstmWeights {
            w_input: &wi,
            w_hidden: &wh,
            b_input: &bi,
            b_hidden: &bh,
            hidden_size: hs,
        };
        let seq = vec![vec![1.0], vec![2.0], vec![3.0]];

        let h1 = lstm_forward(&seq, &w);
        let h2 = lstm_forward(&seq, &w);
        assert_eq!(h1, h2, "lstm_forward must be deterministic");
        assert_eq!(h1.len(), hs);
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn gru_forward_sequence() {
        let hs = 2;
        let (wi, wh, bi, bh) = make_gru_weights(hs, 1, 0.1);
        let w = GruWeights {
            w_input: &wi,
            w_hidden: &wh,
            b_input: &bi,
            b_hidden: &bh,
            hidden_size: hs,
        };
        let seq = vec![vec![1.0], vec![2.0], vec![3.0]];

        let h1 = gru_forward(&seq, &w);
        let h2 = gru_forward(&seq, &w);
        assert_eq!(h1, h2, "gru_forward must be deterministic");
        assert_eq!(h1.len(), hs);
    }
}
