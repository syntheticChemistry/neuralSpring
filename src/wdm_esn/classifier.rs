// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU reference ESN classifier and JSON deserialization.

#![expect(
    clippy::doc_markdown,
    reason = "domain-specific terms (log_rho, log_T) are clearer without backticks"
)]

use serde::Deserialize;

use super::argmax_f64;

/// Input normalization for (log_rho, log_T).
#[derive(Debug, Clone)]
pub struct EsnNormalization {
    /// Per-input mean for `(log_rho, log_T)` normalization.
    pub x_mean: [f64; 2],
    /// Per-input standard deviation for `(log_rho, log_T)` normalization.
    pub x_std: [f64; 2],
}

/// Trained ESN regime classifier (CPU reference path).
#[derive(Debug, Clone)]
pub struct EsnClassifier {
    /// Input weights `[reservoir_size, 2]` flattened row-major.
    pub w_in: Vec<f64>,
    /// Reservoir weights `[reservoir_size, reservoir_size]` flattened row-major.
    pub w_res: Vec<f64>,
    /// Reservoir bias `[reservoir_size]`.
    pub b_res: Vec<f64>,
    /// Output weights `[reservoir_size, n_classes]` flattened row-major.
    pub w_out: Vec<f64>,
    /// Output bias `[n_classes]`.
    pub b_out: Vec<f64>,
    /// Reservoir size.
    pub reservoir_size: usize,
    /// Number of classes.
    pub n_classes: usize,
    /// Input normalization.
    pub norm: EsnNormalization,
}

impl EsnClassifier {
    /// Classify (log_rho, log_T) → regime label (0=Classical, 1=WDM, 2=Degenerate).
    ///
    /// Returns (predicted_label, raw_scores).
    #[must_use]
    pub fn classify(&self, log_rho: f64, log_t: f64) -> (usize, Vec<f64>) {
        let rs = self.reservoir_size;

        let x0 = (log_rho - self.norm.x_mean[0]) / self.norm.x_std[0];
        let x1 = (log_t - self.norm.x_mean[1]) / self.norm.x_std[1];

        let mut h: Vec<f64> = (0..rs)
            .map(|i| {
                (self.w_in[i * 2].mul_add(x0, self.w_in[i * 2 + 1] * x1) + self.b_res[i]).tanh()
            })
            .collect();

        let h_prev = h.clone();
        for (i, h_val) in h.iter_mut().enumerate() {
            let input_proj = self.w_in[i * 2].mul_add(x0, self.w_in[i * 2 + 1] * x1);
            let res_proj: f64 = h_prev
                .iter()
                .enumerate()
                .map(|(j, hj)| self.w_res[i * rs + j] * hj)
                .sum();
            *h_val = (input_proj + res_proj + self.b_res[i]).tanh();
        }

        let nc = self.n_classes;
        let mut scores = self.b_out.clone();
        for (j, h_val) in h.iter().enumerate() {
            for (s, score) in scores.iter_mut().enumerate() {
                *score = self.w_out[j * nc + s].mul_add(*h_val, *score);
            }
        }

        let label = argmax_f64(&scores);
        (label, scores)
    }
}

#[derive(Deserialize)]
struct EsnWeightsJson {
    normalization: NormJson,
    weights: WeightsJson,
}

#[derive(Deserialize)]
struct NormJson {
    x_mean: [f64; 2],
    x_std: [f64; 2],
}

#[derive(Deserialize)]
struct WeightsJson {
    reservoir_size: usize,
    n_classes: usize,
    #[serde(rename = "W_in")]
    w_in: Vec<f64>,
    #[serde(rename = "W_res")]
    w_res: Vec<f64>,
    b_res: Vec<f64>,
    #[serde(rename = "W_out")]
    w_out: Vec<f64>,
    b_out: Vec<f64>,
}

/// Load an [`EsnClassifier`] from the Python baseline JSON.
///
/// Uses typed deserialization instead of manual `serde_json::Value` traversal.
///
/// # Errors
///
/// Returns `Err` if the JSON structure is unexpected.
pub fn load_esn_from_json(json_str: &str) -> Result<EsnClassifier, String> {
    let parsed: EsnWeightsJson =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

    Ok(EsnClassifier {
        w_in: parsed.weights.w_in,
        w_res: parsed.weights.w_res,
        b_res: parsed.weights.b_res,
        w_out: parsed.weights.w_out,
        b_out: parsed.weights.b_out,
        reservoir_size: parsed.weights.reservoir_size,
        n_classes: parsed.weights.n_classes,
        norm: EsnNormalization {
            x_mean: parsed.normalization.x_mean,
            x_std: parsed.normalization.x_std,
        },
    })
}
