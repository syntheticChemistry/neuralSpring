// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paper 026: LSTM blood glucose prediction with horizon limit analysis.
//!
//! Port of Chuna (2020) "Setting Limits on Neural Network's Predictive
//! Capacity in T1D Blood Glucose Concentration" (medRxiv 2020.08.04.20117812).
//!
//! Validates that LSTM prediction accuracy degrades with forecast horizon,
//! with autocorrelation decay τ ≈ 1.5–3 hrs setting the fundamental limit.
//! Same LSTM primitives as Exp 003 (weather), Exp 009 (ERA5), nW-03 (S(q,ω)).
//!
//! ## Python Baseline Provenance
//!
//! | Metric | Value | Source |
//! |--------|-------|--------|
//! | R²(5min) | 0.9629 | `control/glucose_prediction/glucose_prediction.py`, seed=42 |
//! | R²(30min) | 0.7790 | same |
//! | R²(60min) | 0.4698 | same |
//! | R²(120min) | 0.2159 | same |
//! | R²(240min) | 0.1641 | same |
//! | τ (autocorrelation) | 1.5 hrs | same |
//!
//! ## Architecture
//!
//! LSTM(input_size=1, hidden=24) reservoir → pooled features
//! [mean, std, last] → per-horizon linear readout → glucose (mg/dL).
//!
//! ## Reference
//!
//! Chuna (2020), medRxiv 2020.08.04.20117812

#![expect(
    clippy::doc_markdown,
    reason = "Paper 026 table and citation formatting in module docs"
)]

#[cfg(feature = "barracuda")]
/// Ridge regression regularization strength for LSTM readout.
///
/// Small enough to not bias predictions, large enough to stabilize
/// the Cholesky solve for the ill-conditioned feature matrices
/// produced by LSTM hidden-state pooling.
pub(crate) const RIDGE_ALPHA: f64 = 1e-3;

/// Per-horizon prediction results.
#[derive(Debug, Clone)]
pub struct HorizonResult {
    /// Forecast horizon in CGM samples (steps ahead).
    pub horizon_steps: usize,
    /// Forecast horizon in minutes (steps × sample interval).
    pub horizon_minutes: usize,
    /// Coefficient of determination for the LSTM predictor.
    pub r2_lstm: f64,
    /// Root mean squared error for the LSTM predictor (mg/dL).
    pub rmse_lstm: f64,
    /// R² for the naive persistence baseline (last value).
    pub r2_persistence: f64,
    /// RMSE for the persistence baseline (mg/dL).
    pub rmse_persistence: f64,
    /// Percent improvement of LSTM RMSE over persistence.
    pub lstm_improvement_pct: f64,
}

/// Trained glucose predictor for a single horizon.
#[derive(Debug, Clone)]
pub struct GlucoseReadout {
    /// Linear readout weight vector on pooled LSTM features.
    pub w_out: Vec<f64>,
    /// Linear readout bias term.
    pub b_out: f64,
}

/// Complete glucose prediction model (multi-horizon).
#[derive(Debug, Clone)]
pub struct GlucosePredictor {
    /// LSTM input weight matrix (flattened).
    pub w_i: Vec<f64>,
    /// LSTM hidden recurrent weight matrix (flattened).
    pub w_h: Vec<f64>,
    /// LSTM input bias vector.
    pub b_i: Vec<f64>,
    /// LSTM hidden bias vector.
    pub b_h: Vec<f64>,
    /// LSTM hidden state width.
    pub hidden_size: usize,
    /// Input window length (past CGM samples).
    pub seq_len: usize,
    /// CGM training mean used for normalization (mg/dL).
    pub cgm_mean: f64,
    /// CGM training standard deviation used for normalization.
    pub cgm_std: f64,
    /// Per-horizon linear readouts `(horizon_steps, readout)`.
    pub readouts: Vec<(usize, GlucoseReadout)>,
}

mod analysis;
mod cgm;
mod experiment;

#[cfg(test)]
mod tests;

pub use analysis::*;
pub use cgm::*;
pub use experiment::*;
