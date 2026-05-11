// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 098: WDM Surrogate Ensemble Quorum Sensing.
//!
//! Novel composition of nS-05 (game theory, Anderson QS) with nW-01..05
//! (WDM surrogates). Surrogate disagreement maps to Anderson disorder,
//! and QS cooperation dynamics respond to ensemble confidence.
//!
//! Composes:
//! - [`crate::game_theory`] — replicator dynamics, snowdrift payoffs
//! - [`crate::anderson_localization`] — 1D Anderson Hamiltonian, IPR

#[cfg(feature = "barracuda")]
use crate::anderson_localization;
#[cfg(feature = "barracuda")]
use crate::rng::Rng;
use crate::tolerances;

/// Per-temperature-slice Anderson localization result.
#[derive(Debug, Clone)]
pub struct SliceResult {
    /// Index of this slice on the temperature grid.
    pub temp_idx: usize,
    /// Mean Anderson disorder strength W for this slice.
    pub mean_w: f64,
    /// Mean inverse participation ratio across states in this slice.
    pub mean_ipr: f64,
    /// Localization length estimate ξ for this slice.
    pub xi: f64,
}

/// Full baseline loaded from Python JSON.
#[derive(Debug, Clone)]
pub struct EnsembleBaseline {
    /// Number of density grid points in the surrogate sweep.
    pub n_rho: usize,
    /// Number of temperature steps in the surrogate sweep.
    pub n_temp: usize,
    /// Number of WDM surrogate models in the ensemble.
    pub n_surrogates: usize,
    /// Mean surrogate disagreement (coefficient of variation).
    pub disagreement_mean: f64,
    /// Standard deviation of surrogate disagreement.
    pub disagreement_std: f64,
    /// Mean mapped Anderson disorder field W.
    pub w_field_mean: f64,
    /// Standard deviation of the mapped disorder field W.
    pub w_field_std: f64,
    /// Pearson correlation between W and localization length ξ.
    pub r_w_xi: f64,
    /// Mean quorum-sensing cooperation at low W (replicator steady state).
    pub mean_coop_low_w: f64,
    /// Mean quorum-sensing cooperation at high W (replicator steady state).
    pub mean_coop_high_w: f64,
    /// Per-temperature Anderson localization results for the baseline.
    pub slices: Vec<SliceResult>,
    /// Reference 1D disorder profile used for Anderson comparisons.
    pub reference_disorder: Vec<f64>,
}

/// Map disagreement (coefficient of variation) to Anderson disorder.
#[must_use]
pub fn disagreement_to_disorder(disagreement: f64, d_min: f64, d_max: f64, w_scale: f64) -> f64 {
    let d_range = (d_max - d_min).max(tolerances::EXACT_F64);
    let d_norm = (disagreement - d_min) / d_range;
    d_norm * w_scale
}

/// Snowdrift payoff matrix parameterized by disorder fraction.
///
/// Low disorder (agreement) yields high net benefit → cooperation stable.
/// High disorder (disagreement) yields low net benefit → defection wins.
#[must_use]
pub fn snowdrift_payoff(w_frac: f64) -> [[f64; 2]; 2] {
    let b = 3.0;
    let c = 4.0_f64.mul_add(w_frac, 1.0);
    [[b - c / 2.0, b - c], [b, 0.0]]
}

/// Run replicator dynamics for `n_steps` with given payoff.
#[must_use]
pub fn replicator_final_coop(payoff: &[[f64; 2]; 2], n_steps: usize) -> f64 {
    let dt = 0.01_f64;
    let mut freq_c = 0.5;
    for _ in 0..n_steps {
        let freq_d = 1.0 - freq_c;
        let f_c = payoff[0][0].mul_add(freq_c, payoff[0][1] * freq_d);
        let f_d = payoff[1][0].mul_add(freq_c, payoff[1][1] * freq_d);
        let f_bar = freq_c.mul_add(f_c, freq_d * f_d);
        let dx = freq_c * (f_c - f_bar);
        freq_c = dt.mul_add(dx, freq_c).clamp(0.0, 1.0);
    }
    freq_c
}

/// Compute Anderson localization on a 1D disorder field.
///
/// Returns `(mean_ipr, localization_length)`.
#[cfg(feature = "barracuda")]
#[must_use]
pub fn anderson_from_disorder(disorder: &[f64]) -> (f64, f64) {
    let n = disorder.len();
    let t_hop = 1.0;
    let mut rng = Rng::new(0);
    let h = anderson_localization::anderson_hamiltonian_random(n, t_hop, 0.0, &mut rng);

    let mut h_with_disorder = h;
    for i in 0..n {
        h_with_disorder[i * n + i] = disorder[i];
    }

    let result = crate::eigh::eigh_householder_qr(&h_with_disorder, n);

    let mut iprs = Vec::with_capacity(n);
    for k in 0..n {
        let psi: Vec<f64> = (0..n).map(|i| result.eigenvectors[i * n + k]).collect();
        iprs.push(anderson_localization::ipr(&psi));
    }

    let mean_ipr = iprs.iter().sum::<f64>() / n as f64;
    let xi = if mean_ipr > tolerances::EXACT_F64 {
        1.0 / (n as f64 * mean_ipr)
    } else {
        n as f64
    };
    (mean_ipr, xi)
}

///
/// # Errors
///
/// Returns `Err` if the JSON structure is unexpected.
pub fn load_ensemble_from_json(json_str: &str) -> Result<EnsembleBaseline, String> {
    let v: serde_json::Value = serde_json::from_str(json_str).map_err(|e| format!("parse: {e}"))?;

    let grid = &v["grid"];
    let d_stats = &v["disagreement_stats"];
    let w_stats = &v["W_field_stats"];
    let coupling = &v["coupling"];
    let qs = &v["qs_dynamics"];

    let slices = v["slice_results"]
        .as_array()
        .ok_or("missing slice_results")?
        .iter()
        .map(|s| {
            Ok(SliceResult {
                temp_idx: s["temp_idx"].as_u64().ok_or("idx")? as usize,
                mean_w: s["mean_W"].as_f64().ok_or("W")?,
                mean_ipr: s["mean_ipr"].as_f64().ok_or("ipr")?,
                xi: s["xi"].as_f64().ok_or("xi")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    let reference_disorder = v["reference_disorder"]
        .as_array()
        .ok_or("missing ref")?
        .iter()
        .map(|x| x.as_f64().ok_or_else(|| "f64".to_string()))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(EnsembleBaseline {
        n_rho: grid["n_rho"].as_u64().ok_or("n_rho")? as usize,
        n_temp: grid["n_temp"].as_u64().ok_or("n_temp")? as usize,
        n_surrogates: v["n_surrogates"].as_u64().ok_or("n_surr")? as usize,
        disagreement_mean: d_stats["mean"].as_f64().ok_or("d_mean")?,
        disagreement_std: d_stats["std"].as_f64().ok_or("d_std")?,
        w_field_mean: w_stats["mean"].as_f64().ok_or("w_mean")?,
        w_field_std: w_stats["std"].as_f64().ok_or("w_std")?,
        r_w_xi: coupling["r_W_xi"].as_f64().ok_or("r")?,
        mean_coop_low_w: qs["mean_coop_low_W"].as_f64().ok_or("c_low")?,
        mean_coop_high_w: qs["mean_coop_high_W"].as_f64().ok_or("c_high")?,
        slices,
        reference_disorder,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disagreement_to_disorder() {
        use crate::tolerances::CROSS_LANGUAGE;
        assert!((disagreement_to_disorder(0.5, 0.0, 1.0, 20.0) - 10.0).abs() < CROSS_LANGUAGE);
        assert!((disagreement_to_disorder(0.0, 0.0, 1.0, 20.0) - 0.0).abs() < CROSS_LANGUAGE);
        assert!((disagreement_to_disorder(1.0, 0.0, 1.0, 20.0) - 20.0).abs() < CROSS_LANGUAGE);
    }

    #[test]
    fn test_snowdrift_payoff() {
        let p = snowdrift_payoff(0.0);
        assert!(
            (p[0][0] - 2.5).abs() < crate::tolerances::CROSS_LANGUAGE,
            "b - c/2 = 3 - 0.5"
        );
        assert!(
            (p[0][1] - 2.0).abs() < crate::tolerances::CROSS_LANGUAGE,
            "b - c = 3 - 1"
        );
    }

    #[test]
    fn test_replicator_snowdrift() {
        let p = snowdrift_payoff(0.0);
        let fc = replicator_final_coop(&p, 1000);
        assert!(fc > 0.3, "snowdrift low cost → cooperation survives");

        let p_hard = snowdrift_payoff(0.8);
        let fc_hard = replicator_final_coop(&p_hard, 1000);
        assert!(fc > fc_hard, "higher cost → less cooperation");
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn test_anderson_disorder_localizes() {
        let mut rng = crate::rng::Rng::new(99);
        let n = 32;
        let low: Vec<f64> = (0..n).map(|_| rng.uniform() * 0.5).collect();
        let high: Vec<f64> = (0..n).map(|_| rng.uniform() * 15.0).collect();
        let (ipr_low, _) = anderson_from_disorder(&low);
        let (ipr_high, _) = anderson_from_disorder(&high);

        assert!(
            ipr_high > ipr_low,
            "strong disorder → higher IPR (more localized): low={ipr_low:.4}, high={ipr_high:.4}"
        );
    }
}
