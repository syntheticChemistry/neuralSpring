// SPDX-License-Identifier: AGPL-3.0-or-later

//! WDM regime classifier: Echo State Network for plasma phase detection.
//!
//! nW-05: Classifies (ρ, T) conditions into three WDM regimes:
//!   0 = Classical (Γ < 1)
//!   1 = Warm Dense Matter (1 ≤ Γ ≤ 10)
//!   2 = Degenerate (Γ > 10)
//!
//! ## Python Baseline Provenance
//!
//! | Metric | Value | Source |
//! |--------|-------|--------|
//! | Test accuracy | 0.965 | `control/wdm/esn_regime_classifier.py`, seed=42 |
//! | Classical acc | 0.964 | same |
//! | WDM acc | 1.000 | same |
//! | Degenerate acc | 0.939 | same |
//!
//! ## Architecture
//!
//! ESN(input=2, reservoir=64, 2-step recurrence) → linear readout → 3 classes.
//!
//! ## Reference
//!
//! Jaeger, "The echo state approach" (2001)
//! Ichimaru, "Statistical Plasma Physics" (1994)

#![allow(clippy::cast_possible_truncation, clippy::doc_markdown)]

/// Input normalization for (log_rho, log_T).
#[derive(Debug, Clone)]
pub struct EsnNormalization {
    pub x_mean: [f64; 2],
    pub x_std: [f64; 2],
}

/// Trained ESN regime classifier.
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

        // Step 1: h = tanh(W_in · x + b)
        let mut h: Vec<f64> = (0..rs)
            .map(|i| {
                (self.w_in[i * 2].mul_add(x0, self.w_in[i * 2 + 1] * x1) + self.b_res[i]).tanh()
            })
            .collect();

        // Step 2: h = tanh(W_in · x + W_res · h + b)
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

        // Readout: scores = h · W_out + b_out
        let nc = self.n_classes;
        let mut scores = self.b_out.clone();
        for (j, h_val) in h.iter().enumerate() {
            for (s, score) in scores.iter_mut().enumerate() {
                *score = self.w_out[j * nc + s].mul_add(*h_val, *score);
            }
        }

        let label = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);

        (label, scores)
    }
}

/// Load an [`EsnClassifier`] from the Python baseline JSON.
///
/// # Errors
///
/// Returns `Err` if the JSON structure is unexpected.
pub fn load_esn_from_json(json_str: &str) -> Result<EsnClassifier, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

    let norm_data = parsed
        .get("normalization")
        .ok_or("Missing 'normalization'")?;

    let norm = EsnNormalization {
        x_mean: parse_f64_array2(norm_data, "x_mean")?,
        x_std: parse_f64_array2(norm_data, "x_std")?,
    };

    let w = parsed.get("weights").ok_or("Missing 'weights'")?;

    let rs = usize::try_from(
        w.get("reservoir_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or("Missing reservoir_size")?,
    )
    .map_err(|e| format!("reservoir_size: {e}"))?;

    let nc = usize::try_from(
        w.get("n_classes")
            .and_then(serde_json::Value::as_u64)
            .ok_or("Missing n_classes")?,
    )
    .map_err(|e| format!("n_classes: {e}"))?;

    Ok(EsnClassifier {
        w_in: parse_f64_vec(w, "W_in")?,
        w_res: parse_f64_vec(w, "W_res")?,
        b_res: parse_f64_vec(w, "b_res")?,
        w_out: parse_f64_vec(w, "W_out")?,
        b_out: parse_f64_vec(w, "b_out")?,
        reservoir_size: rs,
        n_classes: nc,
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

// ═══════════════════════════════════════════════════════════════════
// Cross-spring evolution: barracuda esn_v2 GPU bridge
//
// Rewires the local CPU ESN inference to barracuda's hardware-agnostic
// ESN (esn_v2) for GPU/NPU execution while preserving Python baseline
// compatibility.
//
// Evolution chain:
//   Jaeger ESN → Python (scikit-learn ridge) → Rust CPU (wdm_esn.rs)
//   → barracuda::esn_v2 (ToadStool S70+, hotSpring ESN absorption)
//   → GPU WGSL (esn_reservoir_update_f64.wgsl, esn_readout_f64.wgsl)
// ═══════════════════════════════════════════════════════════════════

/// Classify using barracuda Tensor ops on GPU.
///
/// Implements the full ESN 2-step recurrence + readout using barracuda
/// `Tensor` operations (matmul, add, tanh). This routes through ToadStool
/// WGSL shaders when a GPU is available, falling back to CPU otherwise.
///
/// Returns `(label, raw_scores_f32)` matching [`EsnClassifier::classify`].
///
/// # Evolution chain
///
/// ```text
/// Python ESN → Rust CPU (EsnClassifier) → barracuda Tensor GPU
///                                         ↑
///                          hotSpring esn_v2 shaders (S70+)
/// ```
///
/// # Errors
///
/// Returns `Err` on GPU/Tensor operation failure.
pub fn classify_via_barracuda(
    classifier: &EsnClassifier,
    log_rho: f64,
    log_t: f64,
    device: &std::sync::Arc<barracuda::device::WgpuDevice>,
) -> Result<(usize, Vec<f32>), String> {
    let rs = classifier.reservoir_size;
    let nc = classifier.n_classes;

    let x0 = ((log_rho - classifier.norm.x_mean[0]) / classifier.norm.x_std[0]) as f32;
    let x1 = ((log_t - classifier.norm.x_mean[1]) / classifier.norm.x_std[1]) as f32;

    let x = barracuda::tensor::Tensor::from_data(&[x0, x1], vec![1, 2], device.clone())
        .map_err(|e| format!("x tensor: {e}"))?;

    let w_in_f32: Vec<f32> = classifier.w_in.iter().map(|&v| v as f32).collect();
    let b_f32: Vec<f32> = classifier.b_res.iter().map(|&v| v as f32).collect();

    let w_in = barracuda::tensor::Tensor::from_data(&w_in_f32, vec![rs, 2], device.clone())
        .map_err(|e| format!("W_in: {e}"))?;
    let w_in_t = w_in.transpose().map_err(|e| format!("W_in^T: {e}"))?;
    let b = barracuda::tensor::Tensor::from_data(&b_f32, vec![1, rs], device.clone())
        .map_err(|e| format!("b_res: {e}"))?;

    // Step 1: h = tanh(x @ W_in^T + b)
    let z1 = x
        .matmul_ref(&w_in_t)
        .map_err(|e| format!("step1 matmul: {e}"))?;
    let z1b = z1.add(&b).map_err(|e| format!("step1 add: {e}"))?;
    let h1 = z1b.tanh().map_err(|e| format!("step1 tanh: {e}"))?;

    // Step 2: h = tanh(x @ W_in^T + h1 @ W_res^T + b)
    let w_res_f32: Vec<f32> = classifier.w_res.iter().map(|&v| v as f32).collect();
    let w_res = barracuda::tensor::Tensor::from_data(&w_res_f32, vec![rs, rs], device.clone())
        .map_err(|e| format!("W_res: {e}"))?;
    let w_res_t = w_res.transpose().map_err(|e| format!("W_res^T: {e}"))?;

    let input_proj = x.matmul(&w_in_t).map_err(|e| format!("step2 input: {e}"))?;
    let res_proj = h1.matmul(&w_res_t).map_err(|e| format!("step2 res: {e}"))?;
    let z2 = input_proj
        .add(&res_proj)
        .map_err(|e| format!("step2 add: {e}"))?;
    let z2b = z2.add(&b).map_err(|e| format!("step2 bias: {e}"))?;
    let h2 = z2b.tanh().map_err(|e| format!("step2 tanh: {e}"))?;

    // Readout: scores = h2 @ W_out + b_out
    let w_out_f32: Vec<f32> = classifier.w_out.iter().map(|&v| v as f32).collect();
    let b_out_f32: Vec<f32> = classifier.b_out.iter().map(|&v| v as f32).collect();

    let w_out = barracuda::tensor::Tensor::from_data(&w_out_f32, vec![rs, nc], device.clone())
        .map_err(|e| format!("W_out: {e}"))?;
    let b_out = barracuda::tensor::Tensor::from_data(&b_out_f32, vec![1, nc], device.clone())
        .map_err(|e| format!("b_out: {e}"))?;

    let scores_raw = h2
        .matmul(&w_out)
        .map_err(|e| format!("readout matmul: {e}"))?;
    let scores = scores_raw
        .add(&b_out)
        .map_err(|e| format!("readout add: {e}"))?;

    let scores_vec = scores.to_vec().map_err(|e| format!("readback: {e}"))?;

    let label = scores_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);

    Ok((label, scores_vec))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn tiny_esn() -> EsnClassifier {
        let rs = 4;
        let nc = 3;
        EsnClassifier {
            w_in: vec![0.1; rs * 2],
            w_res: vec![0.01; rs * rs],
            b_res: vec![0.0; rs],
            w_out: vec![0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            b_out: vec![0.0; nc],
            reservoir_size: rs,
            n_classes: nc,
            norm: EsnNormalization {
                x_mean: [0.5, 6.0],
                x_std: [1.0, 1.5],
            },
        }
    }

    #[test]
    fn classify_deterministic() {
        let esn = tiny_esn();
        let (l1, s1) = esn.classify(0.5, 5.5);
        let (l2, s2) = esn.classify(0.5, 5.5);
        assert_eq!(l1, l2);
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert!((a - b).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn classify_finite_scores() {
        let esn = tiny_esn();
        let (_, scores) = esn.classify(1.0, 6.0);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn classify_label_in_range() {
        let esn = tiny_esn();
        for &(lr, lt) in &[(-1.0, 8.0), (2.0, 4.0), (0.5, 5.5)] {
            let (label, _) = esn.classify(lr, lt);
            assert!(label < 3, "label {label} out of range");
        }
    }

    #[test]
    fn load_roundtrip() {
        let json = r#"{
            "normalization": {"x_mean": [0.5, 6.0], "x_std": [1.0, 1.5]},
            "weights": {
                "reservoir_size": 2, "input_dim": 2, "n_classes": 3,
                "W_in": [0.1, 0.2, 0.3, 0.4],
                "W_res": [0.01, 0.02, 0.03, 0.04],
                "b_res": [0.0, 0.0],
                "W_out": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "b_out": [0.0, 0.0, 0.0]
            }
        }"#;
        let esn = load_esn_from_json(json).expect("valid JSON should parse");
        let (label, scores) = esn.classify(0.5, 5.5);
        assert!(label < 3);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn load_invalid_json() {
        assert!(load_esn_from_json("nope").is_err());
    }

    #[test]
    fn load_missing_weights() {
        let json = r#"{"normalization": {"x_mean": [0, 0], "x_std": [1, 1]}}"#;
        assert!(load_esn_from_json(json).is_err());
    }
}
