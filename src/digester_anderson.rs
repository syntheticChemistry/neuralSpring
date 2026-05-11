// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 096: Digester Community–Performance Coupling via Anderson-ESN.
//!
//! Novel composition of Paper 027 (ESN digestion prediction) and Paper 023
//! (Anderson localization). Tests whether microbial community disorder W
//! predicts ESN yield prediction quality.
//!
//! Composes:
//! - [`crate::digestion_prediction`]: process model, ESN architecture
//! - [`crate::anderson_localization`]: Hamiltonian, IPR, disorder sweep
//! - [`crate::metrics`]: R², RMSE
//!
//! ## Scientific Hypothesis
//!
//! Communities with high Anderson disorder W have fragmented QS signaling,
//! leading to less stable metabolic coordination and noisier biogas yield.
//! The ESN predicts yield worse for high-W communities: Pearson r(W, R²) < 0.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::suboptimal_flops,
    clippy::many_single_char_names,
    reason = "domain-specific numeric patterns, coupling simulation, and Marsaglia-Tsang standard notation (d, c, x, v, u)"
)]

#[cfg(feature = "barracuda")]
use crate::anderson_localization;
use crate::digestion_prediction;
use crate::rng::Rng;
use crate::tolerances;

const W_MAX: f64 = 20.0;
const RECURRENCE_STEPS: usize = 2;

/// Minimum localization length clamp for noise computation.
///
/// Prevents division by zero in [`noise_from_xi`] when ξ → 0.
/// Domain-appropriate: smallest physically meaningful localization
/// in a lattice is ~1 site spacing, far above 0.01.
const XI_FLOOR: f64 = 0.01;

/// Community profile with diversity and Anderson properties.
#[derive(Debug, Clone)]
pub struct CommunityProfile {
    /// Community index.
    pub id: usize,
    /// Species abundance parameter (Dirichlet concentration).
    pub alpha: f64,
    /// Number of species in the community.
    pub n_species: usize,
    /// Shannon diversity index H′.
    pub shannon_h: f64,
    /// Pielou evenness J = H′ / ln(S).
    pub evenness: f64,
    /// Anderson disorder strength W.
    pub disorder_w: f64,
    /// Mean inverse participation ratio (IPR) over the disorder sweep.
    pub mean_ipr: f64,
    /// Anderson localization length ξ.
    pub loc_length_xi: f64,
    /// Environmental noise standard deviation on observed yield.
    pub noise_std: f64,
    /// Test-set coefficient of determination R².
    pub r2_test: f64,
    /// Test-set root mean squared error.
    pub rmse_test: f64,
}

/// Coupling metrics between Anderson disorder and ESN prediction quality.
#[derive(Debug, Clone)]
pub struct CouplingMetrics {
    /// Pearson correlation between disorder W and test R².
    pub pearson_w_r2: f64,
    /// Pearson correlation between localization length ξ and test R².
    pub pearson_xi_r2: f64,
    /// Pearson correlation between mean IPR and test R².
    pub pearson_ipr_r2: f64,
    /// Average test R² pooled over all communities.
    pub pooled_r2_test: f64,
}

/// ESN predictor loaded from the coupling baseline JSON.
#[derive(Debug, Clone)]
pub struct CouplingPredictor {
    /// ESN reservoir state dimension.
    pub reservoir_size: usize,
    /// Input-to-reservoir weight matrix (flattened).
    pub w_in: Vec<f64>,
    /// Reservoir recurrent weight matrix (flattened row-major).
    pub w_res: Vec<f64>,
    /// Reservoir bias vector.
    pub b_res: Vec<f64>,
    /// Linear readout weights from reservoir to scalar output.
    pub w_out: Vec<f64>,
    /// Per-input feature means for normalization.
    pub x_mean: [f64; 5],
    /// Per-input feature standard deviations for normalization.
    pub x_std: [f64; 5],
    /// Target mean for denormalizing the readout.
    pub y_mean: f64,
    /// Target scale for denormalizing the readout.
    pub y_std: f64,
}

impl CouplingPredictor {
    /// ESN inference: normalize, drive reservoir, readout, denormalize.
    #[must_use]
    pub fn predict(&self, t: f64, ph: f64, olr: f64, hrt: f64, vs_ts: f64) -> f64 {
        let rs = self.reservoir_size;
        let x = [
            (t - self.x_mean[0]) / self.x_std[0],
            (ph - self.x_mean[1]) / self.x_std[1],
            (olr - self.x_mean[2]) / self.x_std[2],
            (hrt - self.x_mean[3]) / self.x_std[3],
            (vs_ts - self.x_mean[4]) / self.x_std[4],
        ];

        let mut h = vec![0.0_f64; rs];
        for (i, h_val) in h.iter_mut().enumerate() {
            let mut dot = self.b_res[i];
            for (j, &xj) in x.iter().enumerate() {
                dot += self.w_in[i * 5 + j] * xj;
            }
            *h_val = dot.tanh();
        }

        for _ in 1..RECURRENCE_STEPS {
            let h_prev = h.clone();
            for (i, h_val) in h.iter_mut().enumerate() {
                let mut dot = self.b_res[i];
                for (j, &xj) in x.iter().enumerate() {
                    dot += self.w_in[i * 5 + j] * xj;
                }
                for (j, &hp) in h_prev.iter().enumerate() {
                    dot += self.w_res[i * rs + j] * hp;
                }
                *h_val = dot.tanh();
            }
        }

        let mut y_norm = 0.0_f64;
        for (i, &hi) in h.iter().enumerate() {
            y_norm += self.w_out[i] * hi;
        }
        y_norm.mul_add(self.y_std, self.y_mean)
    }

    /// Return reservoir state for a given input (for GPU parity checks).
    #[must_use]
    pub fn reservoir_state(&self, t: f64, ph: f64, olr: f64, hrt: f64, vs_ts: f64) -> Vec<f64> {
        let rs = self.reservoir_size;
        let x = [
            (t - self.x_mean[0]) / self.x_std[0],
            (ph - self.x_mean[1]) / self.x_std[1],
            (olr - self.x_mean[2]) / self.x_std[2],
            (hrt - self.x_mean[3]) / self.x_std[3],
            (vs_ts - self.x_mean[4]) / self.x_std[4],
        ];

        let mut h = vec![0.0_f64; rs];
        for (i, h_val) in h.iter_mut().enumerate() {
            let mut dot = self.b_res[i];
            for (j, &xj) in x.iter().enumerate() {
                dot += self.w_in[i * 5 + j] * xj;
            }
            *h_val = dot.tanh();
        }

        for _ in 1..RECURRENCE_STEPS {
            let h_prev = h.clone();
            for (i, h_val) in h.iter_mut().enumerate() {
                let mut dot = self.b_res[i];
                for (j, &xj) in x.iter().enumerate() {
                    dot += self.w_in[i * 5 + j] * xj;
                }
                for (j, &hp) in h_prev.iter().enumerate() {
                    dot += self.w_res[i * rs + j] * hp;
                }
                *h_val = dot.tanh();
            }
        }

        h
    }
}

/// Map Shannon evenness to Anderson disorder parameter W.
///
/// Even community (evenness → 1): low W → extended QS → stable yield.
/// Uneven community (evenness → 0): high W → localized QS → noisy yield.
#[must_use]
pub fn evenness_to_disorder(evenness: f64) -> f64 {
    W_MAX * (1.0 - evenness)
}

/// Compute community noise from localization length.
///
/// Localized (small ξ) → more noise. Extended (large ξ) → less noise.
#[must_use]
pub fn noise_from_xi(xi: f64) -> f64 {
    let base = 2.0;
    let scale = 2.0;
    let cap = 15.0;
    (base + scale / xi.max(XI_FLOOR)).min(cap)
}

/// Generate Dirichlet-distributed abundances.
///
/// Uses the Gamma distribution trick: draw Gamma(α,1) for each species,
/// then normalize. Low α → dominated, high α → even.
#[must_use]
pub fn dirichlet_abundances(n_species: usize, alpha: f64, rng: &mut Rng) -> Vec<f64> {
    let mut raw: Vec<f64> = (0..n_species).map(|_| gamma_variate(alpha, rng)).collect();
    let sum: f64 = raw.iter().sum();
    if sum > tolerances::LOG_ZERO_GUARD {
        for v in &mut raw {
            *v /= sum;
        }
    }
    raw
}

/// Gamma(α, 1) variate via Marsaglia-Tsang method.
fn gamma_variate(alpha: f64, rng: &mut Rng) -> f64 {
    if alpha < 1.0 {
        let u = rng.uniform();
        return gamma_variate(alpha + 1.0, rng) * u.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = rng.normal();
        let v = (1.0 + c * x).powi(3);
        if v > 0.0 {
            let u = rng.uniform();
            let half_x2 = 0.5 * x * x;
            if u < 1.0 - 0.0331 * x * x * x * x || u.ln() < half_x2 + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }
}

/// Shannon diversity index: `H' = -Σ(p_i * ln(p_i))`.
///
/// Delegates to `barracuda::stats::shannon_from_frequencies` via
/// [`crate::primitives::shannon_entropy`] (absorbed upstream S64).
#[cfg(feature = "barracuda")]
#[must_use]
pub fn shannon_diversity(abundances: &[f64]) -> f64 {
    crate::primitives::shannon_entropy(abundances)
}

/// Generate a community's Anderson properties.
#[cfg(feature = "barracuda")]
#[must_use]
pub fn community_anderson(
    n_species: usize,
    alpha: f64,
    lattice_size: usize,
    rng: &mut Rng,
) -> (f64, f64, f64, f64, f64) {
    let abundances = dirichlet_abundances(n_species, alpha, rng);
    let h_prime = shannon_diversity(&abundances);
    let h_max = (n_species as f64).ln();
    let evenness = if h_max > tolerances::EXACT_F64 {
        h_prime / h_max
    } else {
        0.0
    };
    let w = evenness_to_disorder(evenness);

    let mipr = anderson_localization::disorder_sweep(lattice_size, 1.0, &[w], rng)[0];
    let xi = if mipr > tolerances::EXACT_F64 {
        1.0 / (lattice_size as f64 * mipr)
    } else {
        lattice_size as f64
    };

    (h_prime, evenness, w, mipr, xi)
}

/// Generate digester data for a community with given noise level.
#[must_use]
pub fn generate_community_data(
    n_samples: usize,
    noise_std: f64,
    rng: &mut Rng,
) -> Vec<(f64, f64, f64, f64, f64, f64)> {
    let mut data = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let t = 20.0 + rng.uniform() * 40.0;
        let ph = 5.5 + rng.uniform() * 3.0;
        let olr = 0.5 + rng.uniform() * 7.5;
        let hrt = 5.0 + rng.uniform() * 35.0;
        let vs_ts = 50.0 + rng.uniform() * 40.0;
        let y_true = digestion_prediction::biogas_yield(t, ph, olr, hrt, vs_ts);
        let noise = rng.normal() * noise_std;
        let y_obs = (y_true + noise).max(0.0);
        data.push((t, ph, olr, hrt, vs_ts, y_obs));
    }
    data
}

/// Re-export centralized Pearson correlation wrapper.
#[cfg(feature = "barracuda")]
pub use crate::primitives::pearson_r;

/// Load coupling baseline from Python JSON.
///
/// # Errors
///
/// Returns `Err` if the JSON structure is unexpected.
pub fn load_coupling_from_json(json_str: &str) -> Result<CouplingBaseline, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

    let esn = &parsed["esn"];
    let rs: usize = esn["reservoir_size"]
        .as_u64()
        .ok_or("missing reservoir_size")? as usize;

    let w_in = flatten_2d(esn["w_in"].as_array().ok_or("missing w_in")?)?;
    let w_res = flatten_2d(esn["w_res"].as_array().ok_or("missing w_res")?)?;
    let b_res = flatten_1d(esn["b_res"].as_array().ok_or("missing b_res")?)?;
    let w_out = flatten_1d(esn["w_out"].as_array().ok_or("missing w_out")?)?;

    let x_mean_v = flatten_1d(esn["x_mean"].as_array().ok_or("missing x_mean")?)?;
    let x_std_v = flatten_1d(esn["x_std"].as_array().ok_or("missing x_std")?)?;

    let mut x_mean = [0.0_f64; 5];
    let mut x_std = [0.0_f64; 5];
    x_mean.copy_from_slice(&x_mean_v[..5]);
    x_std.copy_from_slice(&x_std_v[..5]);

    let predictor = CouplingPredictor {
        reservoir_size: rs,
        w_in,
        w_res,
        b_res,
        w_out,
        x_mean,
        x_std,
        y_mean: esn["y_mean"].as_f64().ok_or("missing y_mean")?,
        y_std: esn["y_std"].as_f64().ok_or("missing y_std")?,
    };

    let coupling = &parsed["coupling"];
    let metrics = CouplingMetrics {
        pearson_w_r2: coupling["pearson_w_r2"]
            .as_f64()
            .ok_or("missing pearson_w_r2")?,
        pearson_xi_r2: coupling["pearson_xi_r2"]
            .as_f64()
            .ok_or("missing pearson_xi_r2")?,
        pearson_ipr_r2: coupling["pearson_ipr_r2"]
            .as_f64()
            .ok_or("missing pearson_ipr_r2")?,
        pooled_r2_test: coupling["pooled_r2_test"]
            .as_f64()
            .ok_or("missing pooled_r2_test")?,
    };

    let communities = parsed["communities"]
        .as_array()
        .ok_or("missing communities")?
        .iter()
        .map(|c| {
            Ok(CommunityProfile {
                id: c["id"].as_u64().ok_or("missing id")? as usize,
                alpha: c["alpha"].as_f64().ok_or("missing alpha")?,
                n_species: c["n_species"].as_u64().ok_or("missing n_species")? as usize,
                shannon_h: c["shannon_h"].as_f64().ok_or("missing shannon_h")?,
                evenness: c["evenness"].as_f64().ok_or("missing evenness")?,
                disorder_w: c["disorder_w"].as_f64().ok_or("missing disorder_w")?,
                mean_ipr: c["mean_ipr"].as_f64().ok_or("missing mean_ipr")?,
                loc_length_xi: c["loc_length_xi"].as_f64().ok_or("missing xi")?,
                noise_std: c["noise_std"].as_f64().ok_or("missing noise_std")?,
                r2_test: c["r2_test"].as_f64().ok_or("missing r2_test")?,
                rmse_test: c["rmse_test"].as_f64().ok_or("missing rmse_test")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    let reference_predictions = parsed["reference_predictions"]
        .as_array()
        .ok_or("missing reference_predictions")?
        .iter()
        .map(|rp| {
            let inp = rp["input"].as_array().ok_or("missing input")?;
            let mut input = [0.0_f64; 5];
            for (i, v) in inp.iter().enumerate().take(5) {
                input[i] = v.as_f64().ok_or("expected f64 in input")?;
            }
            Ok(ReferencePrediction {
                input,
                esn_yield: rp["esn_yield"].as_f64().ok_or("missing esn_yield")?,
                analytical_yield: rp["analytical_yield"]
                    .as_f64()
                    .ok_or("missing analytical_yield")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(CouplingBaseline {
        predictor,
        metrics,
        communities,
        reference_predictions,
        lattice_size: parsed["lattice_size"]
            .as_u64()
            .ok_or("missing lattice_size")? as usize,
    })
}

/// Reference prediction for Rust parity checking.
#[derive(Debug, Clone)]
pub struct ReferencePrediction {
    /// Raw operational feature vector (T, pH, OLR, HRT, VS/TS).
    pub input: [f64; 5],
    /// ESN-predicted methane yield.
    pub esn_yield: f64,
    /// Analytical methane yield from the process model.
    pub analytical_yield: f64,
}

/// Complete coupling baseline loaded from JSON.
#[derive(Debug, Clone)]
pub struct CouplingBaseline {
    /// Trained ESN predictor for yield.
    pub predictor: CouplingPredictor,
    /// Disorder–prediction-quality coupling statistics.
    pub metrics: CouplingMetrics,
    /// One profile per simulated microbial community.
    pub communities: Vec<CommunityProfile>,
    /// Per-community reference predictions for parity checks.
    pub reference_predictions: Vec<ReferencePrediction>,
    /// Spatial dimension of the Anderson localization lattice.
    pub lattice_size: usize,
}

fn flatten_2d(arr: &[serde_json::Value]) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    for row in arr {
        for val in row.as_array().ok_or("expected 2D array")? {
            out.push(val.as_f64().ok_or("expected f64")?);
        }
    }
    Ok(out)
}

fn flatten_1d(arr: &[serde_json::Value]) -> Result<Vec<f64>, String> {
    arr.iter()
        .map(|v| v.as_f64().ok_or_else(|| "expected f64".to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evenness_to_disorder_boundaries() {
        use crate::tolerances::CROSS_LANGUAGE;
        assert!((evenness_to_disorder(0.0) - W_MAX).abs() < CROSS_LANGUAGE);
        assert!(evenness_to_disorder(1.0).abs() < CROSS_LANGUAGE);
        assert!((evenness_to_disorder(0.5) - W_MAX / 2.0).abs() < CROSS_LANGUAGE);
    }

    #[test]
    fn test_noise_from_xi() {
        let noise_high = noise_from_xi(0.05);
        let noise_low = noise_from_xi(1.0);
        assert!(noise_high > noise_low, "localized → more noise");
        assert!(noise_high <= 15.0, "capped at 15");
    }

    #[test]
    fn test_dirichlet_sums_to_one() {
        let mut rng = Rng::new(42);
        let abundances = dirichlet_abundances(20, 0.5, &mut rng);
        let sum: f64 = abundances.iter().sum();
        assert!(
            (sum - 1.0).abs() < crate::tolerances::CROSS_LANGUAGE,
            "abundances sum to 1"
        );
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn test_dirichlet_evenness_ordering() {
        let mut rng_low = Rng::new(42);
        let a_low = dirichlet_abundances(20, 0.1, &mut rng_low);
        let h_low = shannon_diversity(&a_low);

        let mut rng_high = Rng::new(42);
        let a_high = dirichlet_abundances(20, 10.0, &mut rng_high);
        let h_high = shannon_diversity(&a_high);

        assert!(h_high > h_low, "high α → higher diversity");
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn test_shannon_diversity_uniform() {
        let n = 10;
        let uniform: Vec<f64> = vec![1.0 / n as f64; n];
        let h = shannon_diversity(&uniform);
        let expected = (n as f64).ln();
        assert!((h - expected).abs() < crate::tolerances::CROSS_LANGUAGE);
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn test_community_anderson_produces_valid_range() {
        let mut rng = Rng::new(42);
        let (h, ev, w, ipr, xi) = community_anderson(20, 5.0, 32, &mut rng);
        assert!(h > 0.0, "diversity > 0");
        assert!((0.0..=1.0).contains(&ev), "evenness in [0,1]");
        assert!((0.0..=W_MAX).contains(&w), "W in [0, W_MAX]");
        assert!(ipr > 0.0, "IPR > 0");
        assert!(xi > 0.0, "ξ > 0");
    }

    #[test]
    fn test_generate_community_data_count() {
        let mut rng = Rng::new(42);
        let data = generate_community_data(100, 5.0, &mut rng);
        assert_eq!(data.len(), 100);
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn test_pearson_r_perfect_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_r(&x, &y);
        assert!(
            (r - 1.0).abs() < crate::tolerances::CROSS_LANGUAGE,
            "perfect positive correlation"
        );
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn test_pearson_r_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r = pearson_r(&x, &y);
        assert!(
            (r - (-1.0)).abs() < crate::tolerances::CROSS_LANGUAGE,
            "perfect negative correlation"
        );
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn test_disorder_coupling_direction() {
        let mut rng = Rng::new(42);
        let (_, _, w_low, _, xi_low) = community_anderson(20, 10.0, 32, &mut rng);
        let mut rng2 = Rng::new(43);
        let (_, _, w_high, _, xi_high) = community_anderson(20, 0.1, 32, &mut rng2);
        assert!(w_high > w_low, "low α → high disorder");
        assert!(xi_high < xi_low, "high disorder → short ξ");
    }

    #[test]
    fn test_gamma_variate_positive() {
        let mut rng = Rng::new(42);
        for alpha in &[0.1, 0.5, 1.0, 2.0, 5.0] {
            let v = gamma_variate(*alpha, &mut rng);
            assert!(v > 0.0, "Gamma variate positive for α={alpha}");
        }
    }
}
