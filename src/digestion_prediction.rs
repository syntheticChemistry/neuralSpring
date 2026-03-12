// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paper 027: ML prediction of anaerobic digestion performance.
//!
//! Port of Wang et al. (2020) "Prediction of anaerobic digestion
//! performance and identification of critical operational parameters
//! using machine learning algorithms" (Bioresour Technol 298:122495).
//!
//! Validates that ESN reservoir computing predicts biogas yield from
//! operational parameters (temperature, pH, OLR, HRT, VS/TS). Same
//! ESN architecture as nW-05 (WDM regime classifier), different
//! domain (bioprocess engineering) — proves isomorphic thesis.
//!
//! ## Python Baseline Provenance
//!
//! | Metric | Value | Source |
//! |--------|-------|--------|
//! | R²(train) | 0.9203 | `control/digestion_prediction/digestion_prediction.py`, seed=42 |
//! | R²(test) | 0.8528 | same |
//! | RMSE(train) | 5.77 mL/gVS | same |
//! | RMSE(test) | 8.37 mL/gVS | same |
//!
//! ## Architecture
//!
//! ESN(input=5, reservoir=512, 2-step recurrence) → linear readout
//! → methane yield (mL CH₄/gVS).
//!
//! ## Reference
//!
//! Wang et al. (2020), Bioresour Technol 298:122495
//! Liao lab (ADREC, MSU BAE)

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::suboptimal_flops,
    reason = "domain-specific numeric patterns and bioprocess simulation"
)]

use crate::rng::Rng;

// ═══════════════════════════════════════════════════════════════════
// Process model constants (matching Python baseline)
// ═══════════════════════════════════════════════════════════════════

const Y_BASE: f64 = 150.0;
const W_T: f64 = 60.0;
const W_PH: f64 = 40.0;
const W_OLR: f64 = 50.0;
const W_HRT: f64 = 60.0;
const W_VS: f64 = 30.0;
const W_T_OLR: f64 = 25.0;

const MESO_CENTER: f64 = 35.0;
const MESO_SIGMA: f64 = 6.0;
const THERMO_CENTER: f64 = 55.0;
const THERMO_SIGMA: f64 = 6.0;
const PH_CENTER: f64 = 7.2;
const PH_SIGMA: f64 = 1.0;
const K_OLR: f64 = 2.0;
const OLR_INHIBITION: f64 = 0.15;
const TAU_HRT: f64 = 10.0;

const NOISE_STD: f64 = 5.0;
const INPUT_DIM: usize = 5;

// ═══════════════════════════════════════════════════════════════════
// Process model
// ═══════════════════════════════════════════════════════════════════

/// Dual Gaussian temperature response (mesophilic + thermophilic).
#[must_use]
pub fn temperature_response(t: f64) -> f64 {
    0.7 * (-0.5 * ((t - MESO_CENTER) / MESO_SIGMA).powi(2)).exp()
        + 0.3 * (-0.5 * ((t - THERMO_CENTER) / THERMO_SIGMA).powi(2)).exp()
}

/// Gaussian pH response centered at 7.2.
#[must_use]
pub fn ph_response(ph: f64) -> f64 {
    (-0.5 * ((ph - PH_CENTER) / PH_SIGMA).powi(2)).exp()
}

/// Monod saturation with substrate inhibition for OLR.
#[must_use]
pub fn olr_response(olr: f64) -> f64 {
    (olr / (K_OLR + olr)) * (-OLR_INHIBITION * olr).exp()
}

/// Exponential approach to complete conversion for HRT.
#[must_use]
pub fn hrt_response(hrt: f64) -> f64 {
    1.0 - (-hrt / TAU_HRT).exp()
}

/// Compute expected methane yield (mL CH₄/gVS) from operational parameters.
///
/// Additive process model with T×OLR interaction term.
#[must_use]
pub fn biogas_yield(t: f64, ph: f64, olr: f64, hrt: f64, vs_ts: f64) -> f64 {
    let f_t = temperature_response(t);
    let f_ph = ph_response(ph);
    let f_olr = olr_response(olr);
    let f_hrt = hrt_response(hrt);
    let f_vs = vs_ts / 100.0;
    Y_BASE
        + W_T * f_t
        + W_PH * f_ph
        + W_OLR * f_olr
        + W_HRT * f_hrt
        + W_VS * f_vs
        + W_T_OLR * f_t * f_olr
}

// ═══════════════════════════════════════════════════════════════════
// Data generation
// ═══════════════════════════════════════════════════════════════════

/// Synthetic digester operational conditions and observed yield.
#[derive(Debug, Clone)]
pub struct DigesterSample {
    pub temperature: f64,
    pub ph: f64,
    pub olr: f64,
    pub hrt: f64,
    pub vs_ts: f64,
    pub yield_true: f64,
    pub yield_observed: f64,
}

/// Generate synthetic digester dataset matching Python baseline.
#[must_use]
pub fn generate_dataset(n_samples: usize, seed: u64) -> Vec<DigesterSample> {
    let mut rng = Rng::new(seed);
    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let t = 20.0 + rng.uniform() * 40.0;
        let ph = 5.5 + rng.uniform() * 3.0;
        let olr = 0.5 + rng.uniform() * 7.5;
        let hrt = 5.0 + rng.uniform() * 35.0;
        let vs_ts = 50.0 + rng.uniform() * 40.0;
        let y_true = biogas_yield(t, ph, olr, hrt, vs_ts);
        let noise = rng.normal() * NOISE_STD;
        let y_obs = (y_true + noise).max(0.0);
        samples.push(DigesterSample {
            temperature: t,
            ph,
            olr,
            hrt,
            vs_ts,
            yield_true: y_true,
            yield_observed: y_obs,
        });
    }
    samples
}

// ═══════════════════════════════════════════════════════════════════
// ESN predictor
// ═══════════════════════════════════════════════════════════════════

/// Normalization parameters for input features and target.
#[derive(Debug, Clone)]
pub struct DigestionNormalization {
    pub x_mean: [f64; INPUT_DIM],
    pub x_std: [f64; INPUT_DIM],
    pub y_mean: f64,
    pub y_std: f64,
}

/// ESN regression predictor for biogas yield.
#[derive(Debug, Clone)]
pub struct DigestionPredictor {
    pub reservoir_size: usize,
    pub w_in: Vec<f64>,
    pub w_res: Vec<f64>,
    pub b_res: Vec<f64>,
    pub w_out: Vec<f64>,
    pub norm: DigestionNormalization,
}

impl DigestionPredictor {
    /// Predict methane yield for given operational parameters.
    ///
    /// Runs the 2-step ESN recurrence and linear readout, then
    /// denormalizes to mL CH₄/gVS.
    #[must_use]
    pub fn predict(&self, t: f64, ph: f64, olr: f64, hrt: f64, vs_ts: f64) -> f64 {
        let rs = self.reservoir_size;

        let x = [
            (t - self.norm.x_mean[0]) / self.norm.x_std[0],
            (ph - self.norm.x_mean[1]) / self.norm.x_std[1],
            (olr - self.norm.x_mean[2]) / self.norm.x_std[2],
            (hrt - self.norm.x_mean[3]) / self.norm.x_std[3],
            (vs_ts - self.norm.x_mean[4]) / self.norm.x_std[4],
        ];

        // Step 1: h = tanh(W_in @ x + b_res)
        let mut h: Vec<f64> = (0..rs)
            .map(|i| {
                let mut sum = self.b_res[i];
                for (j, &xj) in x.iter().enumerate() {
                    sum += self.w_in[i * INPUT_DIM + j] * xj;
                }
                sum.tanh()
            })
            .collect();

        // Step 2: h = tanh(W_in @ x + W_res @ h_prev + b_res)
        let h_prev = h.clone();
        for (i, h_val) in h.iter_mut().enumerate() {
            let mut sum = self.b_res[i];
            for (j, &xj) in x.iter().enumerate() {
                sum += self.w_in[i * INPUT_DIM + j] * xj;
            }
            for (k, &hk) in h_prev.iter().enumerate() {
                sum += self.w_res[i * rs + k] * hk;
            }
            *h_val = sum.tanh();
        }

        // Readout: y_norm = h · w_out
        let y_norm: f64 = h.iter().zip(&self.w_out).map(|(&hi, &wi)| hi * wi).sum();
        y_norm.mul_add(self.norm.y_std, self.norm.y_mean)
    }

    /// Return the raw reservoir state for a given input (for GPU comparison).
    #[must_use]
    pub fn reservoir_state(&self, t: f64, ph: f64, olr: f64, hrt: f64, vs_ts: f64) -> Vec<f64> {
        let rs = self.reservoir_size;
        let x = [
            (t - self.norm.x_mean[0]) / self.norm.x_std[0],
            (ph - self.norm.x_mean[1]) / self.norm.x_std[1],
            (olr - self.norm.x_mean[2]) / self.norm.x_std[2],
            (hrt - self.norm.x_mean[3]) / self.norm.x_std[3],
            (vs_ts - self.norm.x_mean[4]) / self.norm.x_std[4],
        ];

        let mut h: Vec<f64> = (0..rs)
            .map(|i| {
                let mut sum = self.b_res[i];
                for (j, &xj) in x.iter().enumerate() {
                    sum += self.w_in[i * INPUT_DIM + j] * xj;
                }
                sum.tanh()
            })
            .collect();

        let h_prev = h.clone();
        for (i, h_val) in h.iter_mut().enumerate() {
            let mut sum = self.b_res[i];
            for (j, &xj) in x.iter().enumerate() {
                sum += self.w_in[i * INPUT_DIM + j] * xj;
            }
            for (k, &hk) in h_prev.iter().enumerate() {
                sum += self.w_res[i * rs + k] * hk;
            }
            *h_val = sum.tanh();
        }
        h
    }
}

// ═══════════════════════════════════════════════════════════════════
// Metrics
// ═══════════════════════════════════════════════════════════════════

/// Coefficient of determination (R²).
#[must_use]
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len() as f64;
    let mean = y_true.iter().sum::<f64>() / n;
    let ss_tot: f64 = y_true.iter().map(|&y| (y - mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(&yt, &yp)| (yt - yp).powi(2))
        .sum();
    1.0 - ss_res / ss_tot
}

/// Root mean squared error.
#[must_use]
pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len() as f64;
    let mse: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(&yt, &yp)| (yt - yp).powi(2))
        .sum::<f64>()
        / n;
    mse.sqrt()
}

// ═══════════════════════════════════════════════════════════════════
// JSON loading
// ═══════════════════════════════════════════════════════════════════

/// Reference prediction from the Python baseline.
#[derive(Debug, Clone)]
pub struct ReferencePrediction {
    pub desc: String,
    pub inputs: [f64; INPUT_DIM],
    pub predicted: f64,
    pub analytical: f64,
    pub reservoir_state: Vec<f64>,
}

/// Baseline loaded from the Python JSON.
#[derive(Debug, Clone)]
pub struct DigestionBaseline {
    pub predictor: DigestionPredictor,
    pub r2_train: f64,
    pub r2_test: f64,
    pub rmse_train: f64,
    pub rmse_test: f64,
    pub reference_predictions: Vec<ReferencePrediction>,
}

/// Load the digestion prediction baseline from JSON.
///
/// # Errors
///
/// Returns `Err` if JSON parsing fails or required fields are missing.
pub fn load_digestion_from_json(json_str: &str) -> Result<DigestionBaseline, String> {
    let v: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse: {e}"))?;

    let norm_obj = v.get("normalization").ok_or("missing normalization")?;
    let x_mean: Vec<f64> = norm_obj["x_mean"]
        .as_array()
        .ok_or("bad x_mean")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0))
        .collect();
    let x_std: Vec<f64> = norm_obj["x_std"]
        .as_array()
        .ok_or("bad x_std")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(1.0))
        .collect();
    let y_mean = norm_obj["y_mean"].as_f64().ok_or("bad y_mean")?;
    let y_std = norm_obj["y_std"].as_f64().ok_or("bad y_std")?;

    let esn = v.get("esn_config").ok_or("missing esn_config")?;
    let reservoir_size = esn["reservoir_size"].as_u64().ok_or("bad reservoir_size")? as usize;

    let w = v.get("weights").ok_or("missing weights")?;
    let parse_vec = |key: &str| -> Result<Vec<f64>, String> {
        w[key]
            .as_array()
            .ok_or_else(|| format!("bad {key}"))?
            .iter()
            .map(|v| v.as_f64().ok_or_else(|| format!("non-f64 in {key}")))
            .collect()
    };

    let w_in = parse_vec("W_in")?;
    let w_res = parse_vec("W_res")?;
    let b_res = parse_vec("b_res")?;
    let w_out = parse_vec("w_out")?;

    let metrics = v.get("metrics").ok_or("missing metrics")?;
    let r2_train = metrics["r2_train"].as_f64().ok_or("bad r2_train")?;
    let r2_test = metrics["r2_test"].as_f64().ok_or("bad r2_test")?;
    let rmse_train = metrics["rmse_train"].as_f64().ok_or("bad rmse_train")?;
    let rmse_test = metrics["rmse_test"].as_f64().ok_or("bad rmse_test")?;

    let refs = v
        .get("reference_predictions")
        .and_then(|a| a.as_array())
        .ok_or("missing reference_predictions")?;
    let reference_predictions: Vec<ReferencePrediction> = refs
        .iter()
        .map(|r| {
            let inputs_arr: Vec<f64> = r["inputs"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect();
            let mut inputs = [0.0; INPUT_DIM];
            for (i, &val) in inputs_arr.iter().take(INPUT_DIM).enumerate() {
                inputs[i] = val;
            }
            let res_state: Vec<f64> = r["reservoir_state"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect();
            ReferencePrediction {
                desc: r["desc"].as_str().unwrap_or("").to_string(),
                inputs,
                predicted: r["predicted"].as_f64().unwrap_or(0.0),
                analytical: r["analytical"].as_f64().unwrap_or(0.0),
                reservoir_state: res_state,
            }
        })
        .collect();

    let norm = DigestionNormalization {
        x_mean: [x_mean[0], x_mean[1], x_mean[2], x_mean[3], x_mean[4]],
        x_std: [x_std[0], x_std[1], x_std[2], x_std[3], x_std[4]],
        y_mean,
        y_std,
    };

    Ok(DigestionBaseline {
        predictor: DigestionPredictor {
            reservoir_size,
            w_in,
            w_res,
            b_res,
            w_out,
            norm,
        },
        r2_train,
        r2_test,
        rmse_train,
        rmse_test,
        reference_predictions,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn test_temperature_response_mesophilic_peak() {
        let f = temperature_response(35.0);
        assert!(f > 0.65, "mesophilic peak should be > 0.65, got {f}");
        assert!(f < 0.75, "mesophilic peak should be < 0.75, got {f}");
    }

    #[test]
    fn test_temperature_response_thermophilic_peak() {
        let f = temperature_response(55.0);
        assert!(f > 0.25, "thermophilic peak should be > 0.25, got {f}");
        assert!(f < 0.35, "thermophilic peak should be < 0.35, got {f}");
    }

    #[test]
    fn test_ph_response_optimum() {
        let f = ph_response(7.2);
        assert!(
            (f - 1.0).abs() < tolerances::EXACT_F64,
            "pH optimum should be ~1.0, got {f}"
        );
    }

    #[test]
    fn test_ph_response_acidic() {
        let f = ph_response(5.5);
        assert!(f < 0.30, "pH 5.5 should have low response, got {f}");
    }

    #[test]
    fn test_olr_response_saturation() {
        let low = olr_response(0.5);
        let mid = olr_response(3.0);
        let high = olr_response(8.0);
        assert!(mid > low, "OLR 3 > OLR 0.5");
        assert!(high < mid, "OLR 8 < OLR 3 (inhibition)");
    }

    #[test]
    fn test_hrt_response_approach() {
        let short = hrt_response(5.0);
        let long = hrt_response(40.0);
        assert!(short < long, "longer HRT → higher conversion");
        assert!(long > 0.95, "40d HRT should be near complete");
    }

    #[test]
    fn test_biogas_yield_mesophilic_optimum() {
        let y = biogas_yield(35.0, 7.2, 3.0, 20.0, 75.0);
        assert!(y > 250.0, "optimum yield should be > 250, got {y}");
        assert!(y < 400.0, "optimum yield should be < 400, got {y}");
    }

    #[test]
    fn test_biogas_yield_low_ph_reduces() {
        let y_opt = biogas_yield(35.0, 7.2, 3.0, 20.0, 75.0);
        let y_low = biogas_yield(35.0, 5.5, 3.0, 20.0, 75.0);
        assert!(y_low < y_opt, "low pH should reduce yield");
    }

    #[test]
    fn test_generate_dataset_deterministic() {
        let d1 = generate_dataset(10, 42);
        let d2 = generate_dataset(10, 42);
        for (a, b) in d1.iter().zip(&d2) {
            assert!(
                (a.yield_observed - b.yield_observed).abs() < tolerances::EXACT_F64,
                "dataset should be deterministic"
            );
        }
    }

    #[test]
    fn test_r2_score_perfect() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r2 = r2_score(&y, &y);
        assert!((r2 - 1.0).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn test_rmse_zero() {
        let y = vec![1.0, 2.0, 3.0];
        let e = rmse(&y, &y);
        assert!(e < tolerances::EXACT_F64);
    }
}
