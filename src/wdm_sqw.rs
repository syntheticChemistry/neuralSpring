// SPDX-License-Identifier: AGPL-3.0-or-later

//! WDM S(q,ω) peak predictor: LSTM reservoir on density fluctuation time series.
//!
//! nW-03: Processes synthetic MD density fluctuation time series through
//! an LSTM reservoir and predicts plasmon peak frequency (ω) and Landau
//! damping rate (γ) in reduced units.
//!
//! ## Python Baseline Provenance
//!
//! | Metric | Value | Source |
//! |--------|-------|--------|
//! | R²(ω) | 0.9797 | `control/wdm/sqw_peak_predictor.py`, seed=42 |
//! | R²(γ) | 0.9835 | same |
//! | RMSE   | 0.1381 | same |
//!
//! ## Architecture
//!
//! LSTM(input_size=1, hidden=32) reservoir → pooled features
//! [mean, std, last] → linear readout → (ω_reduced, γ_reduced).
//!
//! ## Reference
//!
//! Hansen & McDonald, "Theory of Simple Liquids" (2013)
//! Gregori et al., PRE 67, 026412 (2003)

#![expect(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    reason = "domain-specific numeric patterns"
)]

use crate::sequence::{lstm_cell, LstmWeights};

const WASHOUT: usize = 4;

/// Normalization parameters for time series input/output.
#[derive(Debug, Clone)]
pub struct SqwNormalization {
    pub series_mean: f64,
    pub series_std: f64,
    pub y_mean: [f64; 2],
    pub y_std: [f64; 2],
}

/// Trained S(q,ω) peak predictor.
#[derive(Debug, Clone)]
pub struct SqwPredictor {
    /// LSTM input-to-hidden weights `[4*hs, 1]` flattened.
    pub w_i: Vec<f64>,
    /// LSTM hidden-to-hidden weights `[4*hs, hs]` flattened.
    pub w_h: Vec<f64>,
    /// LSTM input bias `[4*hs]`.
    pub b_i: Vec<f64>,
    /// LSTM hidden bias `[4*hs]`.
    pub b_h: Vec<f64>,
    /// Output weights `[3*hs, 2]` flattened (row-major).
    pub w_out: Vec<f64>,
    /// Output bias `[2]`.
    pub b_out: [f64; 2],
    /// Hidden size.
    pub hidden_size: usize,
    /// Normalization parameters.
    pub norm: SqwNormalization,
}

impl SqwPredictor {
    /// Predict (ω_reduced, γ_reduced) from a raw time series.
    ///
    /// Normalizes input, runs LSTM, pools hidden states, applies
    /// linear readout, and denormalizes output.
    #[must_use]
    pub fn predict(&self, time_series: &[f64]) -> (f64, f64) {
        let hs = self.hidden_size;

        let normalized: Vec<f64> = time_series
            .iter()
            .map(|&v| (v - self.norm.series_mean) / self.norm.series_std)
            .collect();

        let lstm_w = LstmWeights {
            w_input: &self.w_i,
            w_hidden: &self.w_h,
            b_input: &self.b_i,
            b_hidden: &self.b_h,
            hidden_size: hs,
        };

        let mut h = vec![0.0; hs];
        let mut c = vec![0.0; hs];
        let mut all_h = Vec::with_capacity(normalized.len());

        for val in &normalized {
            let (h_new, c_new) = lstm_cell(&[*val], &h, &c, &lstm_w);
            h = h_new;
            c = c_new;
            all_h.push(h.clone());
        }

        let valid_h = &all_h[WASHOUT..];
        let n_valid = valid_h.len() as f64;

        let mut h_mean = vec![0.0; hs];
        for state in valid_h {
            for (m, s) in h_mean.iter_mut().zip(state.iter()) {
                *m += s;
            }
        }
        for m in &mut h_mean {
            *m /= n_valid;
        }

        let mut h_std = vec![0.0; hs];
        for state in valid_h {
            for (j, s) in state.iter().enumerate() {
                h_std[j] += (s - h_mean[j]).powi(2);
            }
        }
        for s in &mut h_std {
            *s = (*s / n_valid).sqrt();
        }

        let h_last = &all_h[all_h.len() - 1];

        let out_dim = 2;
        let weight_feat_dim = self.w_out.len() / out_dim;

        let features: Vec<f64> = if weight_feat_dim >= 3 * hs {
            let mut f = Vec::with_capacity(3 * hs);
            f.extend_from_slice(&h_mean);
            f.extend_from_slice(&h_std);
            f.extend_from_slice(h_last);
            f
        } else {
            h_last.clone()
        };

        let mut output = [self.b_out[0], self.b_out[1]];
        for (j, feat_val) in features.iter().enumerate() {
            output[0] = self.w_out[j * 2].mul_add(*feat_val, output[0]);
            output[1] = self.w_out[j * 2 + 1].mul_add(*feat_val, output[1]);
        }

        let omega = output[0].mul_add(self.norm.y_std[0], self.norm.y_mean[0]);
        let gamma = output[1].mul_add(self.norm.y_std[1], self.norm.y_mean[1]);

        (omega, gamma)
    }
}

/// Load an [`SqwPredictor`] from the Python baseline JSON.
///
/// # Errors
///
/// Returns `Err` if the JSON structure is unexpected.
pub fn load_sqw_from_json(json_str: &str) -> Result<SqwPredictor, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

    let norm_data = parsed
        .get("normalization")
        .ok_or("Missing 'normalization'")?;

    let norm = SqwNormalization {
        series_mean: norm_data
            .get("series_mean")
            .or_else(|| norm_data.get("spec_mean"))
            .and_then(serde_json::Value::as_f64)
            .ok_or("Missing series_mean / spec_mean")?,
        series_std: norm_data
            .get("series_std")
            .or_else(|| norm_data.get("spec_std"))
            .and_then(serde_json::Value::as_f64)
            .ok_or("Missing series_std / spec_std")?,
        y_mean: parse_f64_array2(norm_data, "y_mean")?,
        y_std: parse_f64_array2(norm_data, "y_std")?,
    };

    let w = parsed.get("weights").ok_or("Missing 'weights'")?;

    let hs = usize::try_from(
        w.get("hidden_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or("Missing hidden_size")?,
    )
    .map_err(|e| format!("hidden_size: {e}"))?;

    let w_i = parse_f64_vec(w, "W_i")?;
    let w_h = parse_f64_vec(w, "W_h")?;
    let b_i = parse_f64_vec(w, "b_i")?;
    let b_h = parse_f64_vec(w, "b_h")?;
    let w_out = parse_f64_vec(w, "W_out")?;
    let b_out_vec = parse_f64_vec(w, "b_out")?;

    if b_out_vec.len() < 2 {
        return Err("b_out needs 2 elements".to_string());
    }

    if w_i.len() != 4 * hs {
        return Err(format!("W_i length {} != 4*hs={}", w_i.len(), 4 * hs));
    }
    if w_h.len() != 4 * hs * hs {
        return Err(format!(
            "W_h length {} != 4*hs*hs={}",
            w_h.len(),
            4 * hs * hs
        ));
    }

    Ok(SqwPredictor {
        w_i,
        w_h,
        b_i,
        b_h,
        w_out,
        b_out: [b_out_vec[0], b_out_vec[1]],
        hidden_size: hs,
        norm,
    })
}

fn parse_f64_array2(obj: &serde_json::Value, key: &str) -> Result<[f64; 2], String> {
    let arr = obj
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("Missing '{key}'"))?;
    if arr.len() < 2 {
        return Err(format!("'{key}' needs 2 elements"));
    }
    Ok([
        arr[0].as_f64().ok_or_else(|| format!("{key}[0] not f64"))?,
        arr[1].as_f64().ok_or_else(|| format!("{key}[1] not f64"))?,
    ])
}

fn parse_f64_vec(obj: &serde_json::Value, key: &str) -> Result<Vec<f64>, String> {
    obj.get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("Missing '{key}'"))?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| format!("{key} element not f64")))
        .collect()
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions use expect for clarity")]
mod tests {
    use super::*;

    fn tiny_predictor() -> SqwPredictor {
        let hs = 2;
        SqwPredictor {
            w_i: vec![0.1; 4 * hs],
            w_h: vec![0.01; 4 * hs * hs],
            b_i: vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            b_h: vec![0.0; 4 * hs],
            w_out: vec![0.1; 3 * hs * 2],
            b_out: [0.0, 0.0],
            hidden_size: hs,
            norm: SqwNormalization {
                series_mean: 0.0,
                series_std: 1.0,
                y_mean: [1.5, 0.1],
                y_std: [0.7, 0.05],
            },
        }
    }

    #[test]
    fn predict_deterministic() {
        let p = tiny_predictor();
        let ts: Vec<f64> = (0..16).map(|i| (f64::from(i) * 0.3).cos()).collect();
        let (o1, g1) = p.predict(&ts);
        let (o2, g2) = p.predict(&ts);
        assert!((o1 - o2).abs() < f64::EPSILON);
        assert!((g1 - g2).abs() < f64::EPSILON);
    }

    #[test]
    fn predict_finite() {
        let p = tiny_predictor();
        let ts: Vec<f64> = (0..16).map(|i| (f64::from(i) * 0.5).sin()).collect();
        let (o, g) = p.predict(&ts);
        assert!(o.is_finite(), "omega must be finite");
        assert!(g.is_finite(), "gamma must be finite");
    }

    #[test]
    fn predict_different_signals_differ() {
        let p = tiny_predictor();
        let ts1: Vec<f64> = (0..16).map(|i| (f64::from(i) * 0.3).cos()).collect();
        let ts2: Vec<f64> = (0..16).map(|i| (f64::from(i) * 2.0).cos()).collect();
        let (o1, _) = p.predict(&ts1);
        let (o2, _) = p.predict(&ts2);
        assert!(
            (o1 - o2).abs() > 1e-10,
            "different signals should give different predictions"
        );
    }

    #[test]
    fn load_roundtrip() {
        let json = r#"{
            "normalization": {
                "series_mean": 0.0, "series_std": 1.0,
                "y_mean": [1.5, 0.1], "y_std": [0.7, 0.05]
            },
            "weights": {
                "hidden_size": 2,
                "W_i": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                "W_h": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                         0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                "b_i": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "b_h": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "W_out": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                "b_out": [0.0, 0.0]
            }
        }"#;
        let p = load_sqw_from_json(json).expect("valid JSON should parse");
        let ts: Vec<f64> = (0..16).map(|i| (f64::from(i) * 0.3).cos()).collect();
        let (o, g) = p.predict(&ts);
        assert!(o.is_finite());
        assert!(g.is_finite());
    }

    #[test]
    fn load_invalid_json() {
        assert!(load_sqw_from_json("not json").is_err());
    }

    #[test]
    fn load_missing_weights() {
        let json = r#"{"normalization": {"series_mean": 0, "series_std": 1, "y_mean": [0,0], "y_std": [1,1]}}"#;
        assert!(load_sqw_from_json(json).is_err());
    }
}
