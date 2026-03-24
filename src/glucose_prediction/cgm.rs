// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic continuous glucose monitor (CGM) traces and sliding-window sequence pairs.
//!
//! Constants and helpers for the T1D-style simulator used by Paper 026 (`generate_synthetic_cgm`)
//! and by the LSTM experiment (`create_sequences`).

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "CGM simulation maps discrete steps to hours and meal indices with intentional casts"
)]

use std::f64::consts::PI;

use crate::rng::Rng;

/// LSTM washout period: discard the first N hidden states before pooling.
pub const WASHOUT: usize = 4;
const BASAL_GLUCOSE: f64 = 120.0;
const NOISE_STD: f64 = 8.0;
/// Steps at which autocorrelation noise is calibrated (also used as `estimate_tau` fallback lag).
pub const ACOR_DECAY_STEPS: usize = 36;
const INSULIN_DECAY_RATE: f64 = 0.02;
/// CGM sampling interval in minutes (standard 5-minute CGM cadence).
pub const DT_MINUTES: f64 = 5.0;
const SAMPLES_PER_DAY: usize = 288;

/// Generate synthetic CGM trace capturing T1D statistical structure.
///
/// Models basal glucose, circadian variation (dawn phenomenon), three
/// daily meals with postprandial spikes and insulin decay, plus
/// Ornstein-Uhlenbeck autocorrelated noise with τ ≈ 3 hrs.
#[must_use]
pub fn generate_synthetic_cgm(n_days: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let n = n_days * SAMPLES_PER_DAY;
    let mut glucose = vec![0.0_f64; n];

    for (t, g) in glucose.iter_mut().enumerate() {
        let hours = (t % SAMPLES_PER_DAY) as f64 * DT_MINUTES / 60.0;
        let dawn = 8.0 * (-0.5 * ((hours - 5.0) / 1.5).powi(2)).exp();
        let circadian = 3.0f64.mul_add((2.0 * PI * hours / 24.0).sin(), dawn);
        *g = BASAL_GLUCOSE + circadian;
    }

    let meal_times_hr = [7.0_f64, 12.0, 18.0];
    let meal_sizes = [50.0_f64, 65.0, 55.0];

    for day in 0..n_days {
        for (&mt, &ms) in meal_times_hr.iter().zip(meal_sizes.iter()) {
            let jitter_hr = rng.normal_params(0.0, 0.3);
            let jitter_size = rng.normal_params(0.0, 8.0);
            let meal_step = day * SAMPLES_PER_DAY + ((mt + jitter_hr) * 60.0 / DT_MINUTES) as usize;
            if meal_step < n {
                let amp = ms + jitter_size;
                for k in 0..48.min(n - meal_step) {
                    let decay = (-INSULIN_DECAY_RATE * k as f64).exp();
                    let rise = 1.0 - (-0.15 * k as f64).exp();
                    glucose[meal_step + k] += amp * rise * decay;
                }
            }
        }
    }

    let alpha = (-1.0 / ACOR_DECAY_STEPS as f64).exp();
    let sigma_scale = (1.0 - alpha * alpha).sqrt();
    let mut noise_prev = rng.normal_params(0.0, NOISE_STD);
    glucose[0] += noise_prev;

    for g in glucose.iter_mut().skip(1) {
        let noise = alpha.mul_add(noise_prev, sigma_scale * rng.normal_params(0.0, NOISE_STD));
        noise_prev = noise;
        *g += noise;
    }

    for g in &mut glucose {
        *g = g.clamp(40.0, 400.0);
    }

    glucose
}

/// Create (input_window, target) pairs for forecasting.
#[must_use]
pub fn create_sequences(data: &[f64], seq_len: usize, horizon: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = data.len();
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for i in seq_len..=(n.saturating_sub(horizon)) {
        if i + horizon - 1 < n {
            inputs.push(data[i - seq_len..i].to_vec());
            targets.push(data[i + horizon - 1]);
        }
    }
    (inputs, targets)
}
