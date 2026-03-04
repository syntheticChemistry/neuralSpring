// SPDX-License-Identifier: AGPL-3.0-or-later

//! WDM Transport Surrogate: MLP inference for Stanton-Murillo coefficients.
//!
//! nW-01: Predicts reduced transport coefficients D*, η*, λ* from
//! (log_rho, log_T, Z*) for partially ionized WDM plasmas.
//!
//! ## Python Baseline Provenance
//!
//! | Metric | Value | Source |
//! |--------|-------|--------|
//! | R²(D*) | 0.9832 | `control/wdm/transport_surrogate.py`, seed=42 |
//! | R²(η*) | 0.9698 | same |
//! | R²(λ*) | 0.9908 | same |
//! | RMSE   | 0.1402 | same |
//!
//! ## Reference
//!
//! Stanton & Murillo, PRE 93, 043203 (2016)
//!
//! ## Evolution path
//!
//! ```text
//! Stanton-Murillo model → Python MLP → Rust MLP → BarraCUDA GPU → Pure GPU
//! ```

#![expect(
    clippy::doc_markdown,
    reason = "domain terms (Stanton-Murillo, MLP) are not crate links"
)]

use barracuda::nn::simple_mlp::{Activation, DenseLayer};
use barracuda::nn::SimpleMlp;

/// Normalization parameters for 3-input/3-output MLP.
#[derive(Debug, Clone)]
pub struct Normalization3 {
    pub x_mean: [f64; 3],
    pub x_std: [f64; 3],
    pub y_mean: [f64; 3],
    pub y_std: [f64; 3],
}

/// Trained transport surrogate predicting (D*, η*, λ*).
///
/// Wraps [`barracuda::nn::SimpleMlp`] with domain-specific normalization
/// and log-space output transform. Rewired from local MLP forward pass
/// to upstream `SimpleMlp::forward` (Session 121, barraCuda v0.3.1).
#[derive(Debug, Clone)]
pub struct TransportSurrogate {
    pub mlp: SimpleMlp,
    pub norm: Normalization3,
}

impl TransportSurrogate {
    /// MLP forward pass: input is `[log10(rho), log10(T), Z*]`.
    ///
    /// Returns `(D_star, eta_star, lambda_star)` in reduced units,
    /// converted from log-space predictions.
    #[must_use]
    pub fn predict(&self, log_rho: f64, log_t: f64, z_star: f64) -> (f64, f64, f64) {
        let x0 = (log_rho - self.norm.x_mean[0]) / self.norm.x_std[0];
        let x1 = (log_t - self.norm.x_mean[1]) / self.norm.x_std[1];
        let x2 = (z_star - self.norm.x_mean[2]) / self.norm.x_std[2];

        let raw = self.mlp.forward(&[x0, x1, x2]);

        let d_log = raw[0].mul_add(self.norm.y_std[0], self.norm.y_mean[0]);
        let eta_log = raw[1].mul_add(self.norm.y_std[1], self.norm.y_mean[1]);
        let lam_log = raw[2].mul_add(self.norm.y_std[2], self.norm.y_mean[2]);

        (
            10.0_f64.powf(d_log),
            10.0_f64.powf(eta_log),
            10.0_f64.powf(lam_log),
        )
    }
}

/// Load a [`TransportSurrogate`] from the Python baseline JSON.
///
/// Parses `transport_surrogate_baseline.json` produced by
/// `control/wdm/transport_surrogate.py`.
///
/// # Errors
///
/// Returns `Err` if the JSON structure is unexpected.
pub fn load_transport_from_json(json_str: &str) -> Result<TransportSurrogate, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

    let norm_data = parsed
        .get("normalization")
        .ok_or("Missing 'normalization'")?;

    let norm = Normalization3 {
        x_mean: parse_f64_array3(norm_data, "x_mean")?,
        x_std: parse_f64_array3(norm_data, "x_std")?,
        y_mean: parse_f64_array3(norm_data, "y_mean")?,
        y_std: parse_f64_array3(norm_data, "y_std")?,
    };

    let weights_data = parsed
        .get("weights")
        .and_then(serde_json::Value::as_array)
        .ok_or("Missing 'weights'")?;

    let n_layers = weights_data.len();
    let mut dense_layers = Vec::with_capacity(n_layers);

    for (i, layer_data) in weights_data.iter().enumerate() {
        let w_flat: Vec<f64> = layer_data
            .get("weights")
            .and_then(serde_json::Value::as_array)
            .ok_or("Missing layer weights")?
            .iter()
            .filter_map(serde_json::Value::as_f64)
            .collect();
        let bias: Vec<f64> = layer_data
            .get("bias")
            .and_then(serde_json::Value::as_array)
            .ok_or("Missing layer bias")?
            .iter()
            .filter_map(serde_json::Value::as_f64)
            .collect();
        let in_f = usize::try_from(
            layer_data
                .get("in_features")
                .and_then(serde_json::Value::as_u64)
                .ok_or("Missing in_features")?,
        )
        .map_err(|e| format!("in_features: {e}"))?;
        let out_f = usize::try_from(
            layer_data
                .get("out_features")
                .and_then(serde_json::Value::as_u64)
                .ok_or("Missing out_features")?,
        )
        .map_err(|e| format!("out_features: {e}"))?;

        let weight = (0..out_f)
            .map(|row| w_flat[row * in_f..(row + 1) * in_f].to_vec())
            .collect();

        let is_last = i == n_layers - 1;
        dense_layers.push(DenseLayer {
            weight,
            bias,
            activation: if is_last {
                Activation::Identity
            } else {
                Activation::Relu
            },
        });
    }

    Ok(TransportSurrogate {
        mlp: SimpleMlp::new(dense_layers),
        norm,
    })
}

fn parse_f64_array3(obj: &serde_json::Value, key: &str) -> Result<[f64; 3], String> {
    let arr = obj
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("Missing '{key}'"))?;
    if arr.len() < 3 {
        return Err(format!("'{key}' needs 3 elements, got {}", arr.len()));
    }
    Ok([
        arr[0].as_f64().ok_or_else(|| format!("{key}[0] not f64"))?,
        arr[1].as_f64().ok_or_else(|| format!("{key}[1] not f64"))?,
        arr[2].as_f64().ok_or_else(|| format!("{key}[2] not f64"))?,
    ])
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn test_surrogate() -> TransportSurrogate {
        TransportSurrogate {
            mlp: SimpleMlp::new(vec![
                DenseLayer {
                    weight: vec![
                        vec![0.1, 0.2, 0.3],
                        vec![0.4, 0.5, 0.6],
                        vec![0.7, 0.8, 0.9],
                        vec![1.0, 1.1, 1.2],
                    ],
                    bias: vec![0.0, 0.0, 0.0, 0.0],
                    activation: Activation::Relu,
                },
                DenseLayer {
                    weight: vec![
                        vec![0.1, 0.2, 0.3, 0.4],
                        vec![0.5, 0.6, 0.7, 0.8],
                        vec![0.9, 1.0, 1.1, 1.2],
                    ],
                    bias: vec![0.0, 0.0, 0.0],
                    activation: Activation::Identity,
                },
            ]),
            norm: Normalization3 {
                x_mean: [0.0, 6.0, 7.0],
                x_std: [1.0, 1.0, 3.5],
                y_mean: [-1.0, -1.3, -0.4],
                y_std: [0.5, 0.4, 0.3],
            },
        }
    }

    #[test]
    fn predict_deterministic() {
        let s = test_surrogate();
        let (d1, e1, l1) = s.predict(0.5, 5.0, 6.0);
        let (d2, e2, l2) = s.predict(0.5, 5.0, 6.0);
        assert!((d1 - d2).abs() < f64::EPSILON);
        assert!((e1 - e2).abs() < f64::EPSILON);
        assert!((l1 - l2).abs() < f64::EPSILON);
    }

    #[test]
    fn predict_finite_output() {
        let s = test_surrogate();
        for &(lr, lt, z) in &[(0.5, 5.0, 1.0), (-1.0, 8.0, 13.0), (1.7, 4.0, 6.0)] {
            let (d, e, l) = s.predict(lr, lt, z);
            assert!(d.is_finite(), "D* not finite for ({lr}, {lt}, {z})");
            assert!(e.is_finite(), "η* not finite for ({lr}, {lt}, {z})");
            assert!(l.is_finite(), "λ* not finite for ({lr}, {lt}, {z})");
        }
    }

    #[test]
    fn predict_positive_output() {
        let s = test_surrogate();
        let (d, e, l) = s.predict(0.5, 6.0, 5.0);
        assert!(d > 0.0, "Transport coefficients must be positive");
        assert!(e > 0.0, "Transport coefficients must be positive");
        assert!(l > 0.0, "Transport coefficients must be positive");
    }

    #[test]
    fn load_roundtrip() {
        let json = r#"{
            "normalization": {
                "x_mean": [0.0, 6.0, 7.0],
                "x_std": [1.0, 1.0, 3.5],
                "y_mean": [-1.0, -1.3, -0.4],
                "y_std": [0.5, 0.4, 0.3]
            },
            "weights": [
                {"weights": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                 "bias": [0.0, 0.0, 0.0, 0.0],
                 "in_features": 3, "out_features": 4},
                {"weights": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                 "bias": [0.0, 0.0, 0.0],
                 "in_features": 4, "out_features": 3}
            ]
        }"#;
        let surr = load_transport_from_json(json).expect("valid JSON should parse");
        assert_eq!(surr.mlp.layers.len(), 2);
        let (d, e, l) = surr.predict(0.5, 6.0, 5.0);
        assert!(d.is_finite());
        assert!(e.is_finite());
        assert!(l.is_finite());
    }

    #[test]
    fn load_missing_normalization() {
        let json = r#"{"weights": []}"#;
        assert!(load_transport_from_json(json).is_err());
    }

    #[test]
    fn load_short_normalization_array() {
        let json = r#"{"normalization": {"x_mean": [0.0, 1.0], "x_std": [1.0], "y_mean": [0.0], "y_std": [1.0]}, "weights": []}"#;
        assert!(load_transport_from_json(json).is_err());
    }

    #[test]
    fn load_invalid_json() {
        assert!(load_transport_from_json("not json").is_err());
    }
}
