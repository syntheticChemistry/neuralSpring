// SPDX-License-Identifier: AGPL-3.0-or-later

//! WDM EOS surrogate: MLP inference for FPEOS table predictions.
//!
//! nW-02: Trains on Militzer FPEOS tables (PRE 103, 013203) to predict
//! pressure P(rho, T) and energy E(rho, T) for H, He, C in the warm
//! dense matter regime.
//!
//! ## Python Baseline Provenance
//!
//! | Metric | Value | Source |
//! |--------|-------|--------|
//! | H R²(P) | 0.9989 | `control/wdm/eos_surrogate.py`, seed=42 |
//! | He R²(P) | 0.9896 | same |
//! | C R²(P) | 0.9854 | same |
//! | All R²(E) | >0.70 | same |
//!
//! ## Evolution path
//!
//! ```text
//! FPEOS tables (Militzer) → Python MLP → Rust MLP → BarraCUDA GPU → Pure GPU
//! ```

use barracuda::nn::SimpleMlp;
use barracuda::nn::simple_mlp::{Activation, DenseLayer};

/// Normalization parameters for MLP input/output.
#[derive(Debug, Clone)]
pub struct Normalization {
    /// Mean of `[log10(rho), log10(T)]` inputs used for z-scoring.
    pub x_mean: [f64; 2],
    /// Standard deviation of `[log10(rho), log10(T)]` inputs.
    pub x_std: [f64; 2],
    /// Mean of normalized log-pressure and log-energy outputs.
    pub y_mean: [f64; 2],
    /// Standard deviation of normalized log-pressure and log-energy outputs.
    pub y_std: [f64; 2],
}

/// Trained EOS surrogate for one element.
///
/// Wraps [`barracuda::nn::SimpleMlp`] with domain-specific normalization
/// and signed-log output transform. Rewired from local MLP forward pass
/// to upstream `SimpleMlp::forward` (Session 121, barraCuda v0.3.1).
#[derive(Debug, Clone)]
pub struct EosSurrogate {
    /// Element symbol (e.g. H, He, C) this surrogate was trained for.
    pub element: String,
    /// Trained MLP weights for pressure and energy prediction.
    pub mlp: SimpleMlp,
    /// Input/output normalization parameters for the MLP.
    pub norm: Normalization,
}

impl EosSurrogate {
    /// MLP forward pass: input is `[log10(rho), log10(T)]`.
    ///
    /// Returns `(P_predicted, E_predicted)` in original units.
    #[must_use]
    pub fn predict(&self, rho: f64, temperature: f64) -> (f64, f64) {
        let guard = crate::tolerances::LOG_ZERO_GUARD;
        let log_rho = (rho + guard).log10();
        let log_t = (temperature + guard).log10();

        let x0 = (log_rho - self.norm.x_mean[0]) / self.norm.x_std[0];
        let x1 = (log_t - self.norm.x_mean[1]) / self.norm.x_std[1];

        let raw = self.mlp.forward(&[x0, x1]);

        let log_pres = raw[0].mul_add(self.norm.y_std[0], self.norm.y_mean[0]);
        let log_eng = raw[1].mul_add(self.norm.y_std[1], self.norm.y_mean[1]);

        let pres = log_pres.signum() * 10.0_f64.powf(log_pres.abs());
        let eng = log_eng.signum() * 10.0_f64.powf(log_eng.abs());

        (pres, eng)
    }
}

/// Load an [`EosSurrogate`] from the Python baseline JSON.
///
/// Parses the `eos_surrogate_baseline.json` produced by
/// `control/wdm/eos_surrogate.py`. Converts flat row-major weights to
/// [`barracuda::nn::SimpleMlp`] `DenseLayer` format (2D `Vec<Vec<f64>>`).
///
/// # Errors
///
/// Returns `Err` if the JSON cannot be parsed or the element is missing.
pub fn load_surrogate_from_json(json_str: &str, element: &str) -> Result<EosSurrogate, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

    let elem_data = parsed
        .get("elements")
        .and_then(|e| e.get(element))
        .ok_or_else(|| format!("Element '{element}' not found in baseline"))?;

    let norm_data = elem_data
        .get("normalization")
        .ok_or("Missing 'normalization'")?;

    let norm = Normalization {
        x_mean: parse_f64_array(norm_data, "x_mean")?,
        x_std: parse_f64_array(norm_data, "x_std")?,
        y_mean: parse_f64_array(norm_data, "y_mean")?,
        y_std: parse_f64_array(norm_data, "y_std")?,
    };

    let weights_data = elem_data
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

    Ok(EosSurrogate {
        element: element.to_string(),
        mlp: SimpleMlp::new(dense_layers),
        norm,
    })
}

fn parse_f64_array(obj: &serde_json::Value, key: &str) -> Result<[f64; 2], String> {
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

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions use expect for clarity")]
mod tests {
    use super::*;

    fn test_surrogate() -> EosSurrogate {
        EosSurrogate {
            element: "test".to_string(),
            mlp: SimpleMlp::new(vec![
                DenseLayer {
                    weight: vec![
                        vec![0.1, 0.2],
                        vec![0.3, 0.4],
                        vec![0.5, 0.6],
                        vec![0.7, 0.8],
                    ],
                    bias: vec![0.0, 0.0, 0.0, 0.0],
                    activation: Activation::Relu,
                },
                DenseLayer {
                    weight: vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.5, 0.6, 0.7, 0.8]],
                    bias: vec![0.0, 0.0],
                    activation: Activation::Identity,
                },
            ]),
            norm: Normalization {
                x_mean: [0.0, 4.0],
                x_std: [1.0, 1.0],
                y_mean: [0.0, 0.0],
                y_std: [1.0, 1.0],
            },
        }
    }

    #[test]
    fn surrogate_predict_deterministic() {
        let surr = test_surrogate();
        let (p1, e1) = surr.predict(1.0, 10000.0);
        let (p2, e2) = surr.predict(1.0, 10000.0);
        assert!(
            (p1 - p2).abs() < f64::EPSILON,
            "predict must be deterministic"
        );
        assert!(
            (e1 - e2).abs() < f64::EPSILON,
            "predict must be deterministic"
        );
    }

    #[test]
    fn surrogate_predict_finite_output() {
        let surr = test_surrogate();
        for &(rho, t) in &[(1.0, 1e5), (0.01, 1e3), (100.0, 1e7), (1e-10, 1.0)] {
            let (p, e) = surr.predict(rho, t);
            assert!(p.is_finite(), "P not finite for rho={rho}, T={t}");
            assert!(e.is_finite(), "E not finite for rho={rho}, T={t}");
        }
    }

    #[test]
    fn surrogate_predict_zero_inputs_safe() {
        let surr = test_surrogate();
        let (p, e) = surr.predict(0.0, 0.0);
        assert!(p.is_finite(), "P must be finite for zero inputs");
        assert!(e.is_finite(), "E must be finite for zero inputs");
    }

    #[test]
    fn surrogate_predict_very_small_inputs() {
        let surr = test_surrogate();
        let (p, e) = surr.predict(1e-300, 1e-300);
        assert!(p.is_finite(), "P must be finite for tiny inputs");
        assert!(e.is_finite(), "E must be finite for tiny inputs");
    }

    #[test]
    fn surrogate_predict_large_inputs() {
        let surr = test_surrogate();
        let (p, e) = surr.predict(1e10, 1e10);
        assert!(p.is_finite(), "P must be finite for large inputs");
        assert!(e.is_finite(), "E must be finite for large inputs");
    }

    #[test]
    fn surrogate_normalization_identity() {
        let surr = EosSurrogate {
            element: "id".to_string(),
            mlp: SimpleMlp::new(vec![DenseLayer {
                weight: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                bias: vec![0.0, 0.0],
                activation: Activation::Identity,
            }]),
            norm: Normalization {
                x_mean: [0.0, 0.0],
                x_std: [1.0, 1.0],
                y_mean: [0.0, 0.0],
                y_std: [1.0, 1.0],
            },
        };
        let (p, e) = surr.predict(10.0, 10.0);
        assert!(p.is_finite());
        assert!(e.is_finite());
    }

    #[test]
    fn surrogate_relu_clips_negative() {
        let surr = EosSurrogate {
            element: "relu".to_string(),
            mlp: SimpleMlp::new(vec![
                DenseLayer {
                    weight: vec![vec![-1.0, 0.0], vec![0.0, -1.0]],
                    bias: vec![0.0, 0.0],
                    activation: Activation::Relu,
                },
                DenseLayer {
                    weight: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                    bias: vec![0.0, 0.0],
                    activation: Activation::Identity,
                },
            ]),
            norm: Normalization {
                x_mean: [0.0, 0.0],
                x_std: [1.0, 1.0],
                y_mean: [0.0, 0.0],
                y_std: [1.0, 1.0],
            },
        };
        let (p, e) = surr.predict(10.0, 10.0);
        assert!(
            p.abs() <= 1.0 + f64::EPSILON,
            "ReLU should clip negative hidden activations, got P={p}"
        );
        assert!(
            e.abs() <= 1.0 + f64::EPSILON,
            "ReLU should clip negative hidden activations, got E={e}"
        );
    }

    fn valid_json() -> &'static str {
        r#"{
            "elements": {
                "H": {
                    "normalization": {
                        "x_mean": [0.0, 4.0],
                        "x_std": [1.0, 1.0],
                        "y_mean": [0.0, 0.0],
                        "y_std": [1.0, 1.0]
                    },
                    "weights": [
                        {"weights": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                         "bias": [0.0, 0.0, 0.0, 0.0],
                         "in_features": 2, "out_features": 4},
                        {"weights": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                         "bias": [0.0, 0.0],
                         "in_features": 4, "out_features": 2}
                    ]
                }
            }
        }"#
    }

    #[test]
    fn load_surrogate_valid_json() {
        let surr = load_surrogate_from_json(valid_json(), "H").expect("valid JSON should parse");
        assert_eq!(surr.element, "H");
        assert_eq!(surr.mlp.input_size(), Some(2));
        assert_eq!(surr.mlp.output_size(), Some(2));
    }

    #[test]
    fn load_surrogate_roundtrip() {
        let surr = load_surrogate_from_json(valid_json(), "H").expect("roundtrip parse");
        let (p, e) = surr.predict(1.0, 10000.0);
        assert!(p.is_finite());
        assert!(e.is_finite());
    }

    #[test]
    fn load_surrogate_missing_element() {
        let result = load_surrogate_from_json(valid_json(), "Xe");
        assert!(result.is_err());
        let err = result.expect_err("already asserted is_err");
        assert!(
            err.contains("not found"),
            "error should mention missing element, got: {err}"
        );
    }

    #[test]
    fn load_surrogate_invalid_json() {
        let result = load_surrogate_from_json("not json", "H");
        assert!(result.is_err());
    }

    #[test]
    fn load_surrogate_missing_normalization() {
        let json = r#"{"elements": {"H": {"weights": []}}}"#;
        let result = load_surrogate_from_json(json, "H");
        assert!(result.is_err());
    }

    #[test]
    fn load_surrogate_short_normalization_array() {
        let json = r#"{"elements": {"H": {
            "normalization": {"x_mean": [0.0], "x_std": [1.0], "y_mean": [0.0], "y_std": [1.0]},
            "weights": []
        }}}"#;
        let result = load_surrogate_from_json(json, "H");
        assert!(result.is_err());
    }

    #[test]
    fn parse_f64_array_rejects_non_float() {
        let val: serde_json::Value =
            serde_json::from_str(r#"{"arr": ["a", "b"]}"#).expect("test JSON parses");
        let result = parse_f64_array(&val, "arr");
        assert!(result.is_err());
    }
}
