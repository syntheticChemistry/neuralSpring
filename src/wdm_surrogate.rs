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

/// Normalization parameters for MLP input/output.
#[derive(Debug, Clone)]
pub struct Normalization {
    pub x_mean: [f64; 2],
    pub x_std: [f64; 2],
    pub y_mean: [f64; 2],
    pub y_std: [f64; 2],
}

/// MLP layer weights (row-major) and biases.
#[derive(Debug, Clone)]
pub struct MlpLayer {
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
    pub in_features: usize,
    pub out_features: usize,
}

/// Trained EOS surrogate for one element.
#[derive(Debug, Clone)]
pub struct EosSurrogate {
    pub element: String,
    pub layers: Vec<MlpLayer>,
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

        let mut activations = vec![x0, x1];

        for (i, layer) in self.layers.iter().enumerate() {
            let mut output = layer.bias.clone();
            for (row, out_val) in output.iter_mut().enumerate() {
                for (col, act_val) in activations.iter().enumerate() {
                    *out_val =
                        layer.weights[row * layer.in_features + col].mul_add(*act_val, *out_val);
                }
            }
            if i < self.layers.len() - 1 {
                for v in &mut output {
                    *v = v.max(0.0);
                }
            }
            activations = output;
        }

        let log_pres_norm = activations[0];
        let log_eng_norm = activations[1];

        let log_pres = log_pres_norm.mul_add(self.norm.y_std[0], self.norm.y_mean[0]);
        let log_eng = log_eng_norm.mul_add(self.norm.y_std[1], self.norm.y_mean[1]);

        let pres = log_pres.signum() * 10.0_f64.powf(log_pres.abs());
        let eng = log_eng.signum() * 10.0_f64.powf(log_eng.abs());

        (pres, eng)
    }
}

/// Load an `EosSurrogate` from the Python baseline JSON.
///
/// Parses the `eos_surrogate_baseline.json` produced by
/// `control/wdm/eos_surrogate.py`.
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

    let mut layers = Vec::new();
    for layer_data in weights_data {
        let w: Vec<f64> = layer_data
            .get("weights")
            .and_then(serde_json::Value::as_array)
            .ok_or("Missing layer weights")?
            .iter()
            .filter_map(serde_json::Value::as_f64)
            .collect();
        let b: Vec<f64> = layer_data
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

        layers.push(MlpLayer {
            weights: w,
            bias: b,
            in_features: in_f,
            out_features: out_f,
        });
    }

    Ok(EosSurrogate {
        element: element.to_string(),
        layers,
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
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn test_surrogate() -> EosSurrogate {
        EosSurrogate {
            element: "test".to_string(),
            layers: vec![
                MlpLayer {
                    weights: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    bias: vec![0.0, 0.0, 0.0, 0.0],
                    in_features: 2,
                    out_features: 4,
                },
                MlpLayer {
                    weights: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    bias: vec![0.0, 0.0],
                    in_features: 4,
                    out_features: 2,
                },
            ],
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
            layers: vec![MlpLayer {
                weights: vec![1.0, 0.0, 0.0, 1.0],
                bias: vec![0.0, 0.0],
                in_features: 2,
                out_features: 2,
            }],
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
            layers: vec![
                MlpLayer {
                    weights: vec![-1.0, 0.0, 0.0, -1.0],
                    bias: vec![0.0, 0.0],
                    in_features: 2,
                    out_features: 2,
                },
                MlpLayer {
                    weights: vec![1.0, 0.0, 0.0, 1.0],
                    bias: vec![0.0, 0.0],
                    in_features: 2,
                    out_features: 2,
                },
            ],
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
        assert_eq!(surr.layers.len(), 2);
        assert_eq!(surr.layers[0].in_features, 2);
        assert_eq!(surr.layers[0].out_features, 4);
        assert_eq!(surr.layers[1].in_features, 4);
        assert_eq!(surr.layers[1].out_features, 2);
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
