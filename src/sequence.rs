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
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let n = data.len();

    for i in seq_len..=(n.saturating_sub(horizon)) {
        if i + horizon - 1 < n {
            inputs.push(data[i - seq_len..i].to_vec());
            targets.push(data[i + horizon - 1]);
        }
    }

    (inputs, targets)
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
/// Matches the Python `generate_michigan_weather` seasonal component.
#[must_use]
pub fn seasonal_tmax(doy: u32) -> f64 {
    let doy_f = f64::from(doy % 365);
    15.0f64.mul_add((2.0 * PI * (doy_f - 100.0) / 365.0).sin(), 8.5)
}

/// Sigmoid activation (used in LSTM/GRU gates).
///
/// σ(x) = 1 / (1 + exp(-x))
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Tanh activation (used in LSTM cell state).
#[must_use]
pub fn tanh_activation(x: f64) -> f64 {
    x.tanh()
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn sigmoid_bounds() {
        assert_relative_eq!(sigmoid(0.0), 0.5);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn sigmoid_symmetry() {
        let x = 2.5;
        assert_relative_eq!(sigmoid(x) + sigmoid(-x), 1.0, epsilon = 1e-15);
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
}
