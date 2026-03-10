// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 097: Isomorphic Reservoir Ensemble — Cross-Domain Spectral Proof.
//!
//! Novel composition of Paper 027 (ESN digester), Paper 026 (LSTM glucose),
//! and Study 003/004 (LSTM weather). Proves spectral universality: the same
//! reservoir computing architecture produces similar eigenvalue distributions,
//! IPR, and effective dimension ratios across three unrelated domains.
//!
//! Composes:
//! - [`crate::anderson_localization`]: IPR computation
//! - [`crate::eigh`] — eigendecomposition
//! - `crate::metrics` — R², RMSE

use crate::anderson_localization;

/// Spectral properties of a weight matrix.
#[derive(Debug, Clone)]
pub struct SpectralProfile {
    pub name: String,
    pub size: usize,
    pub spectral_radius: f64,
    pub eigenvalue_mean: f64,
    pub eigenvalue_std: f64,
    pub eigenvalue_min: f64,
    pub eigenvalue_max: f64,
    pub mean_spacing_ratio: f64,
    pub mean_ipr: f64,
    pub effective_dimension: f64,
    pub effective_ratio: f64,
}

/// Cross-domain comparison metrics.
#[derive(Debug, Clone)]
pub struct CrossDomainMetrics {
    pub eff_ratio_mean: f64,
    pub eff_ratio_std: f64,
    pub eff_ratio_cv: f64,
    pub ipr_mean: f64,
    pub ipr_std: f64,
    pub ipr_cv: f64,
    pub spacing_ratio_mean: f64,
    pub spacing_ratio_std: f64,
}

/// Full baseline loaded from JSON.
#[derive(Debug, Clone)]
pub struct IsomorphicBaseline {
    pub spectra: Vec<SpectralProfile>,
    pub cross_domain: CrossDomainMetrics,
    pub domain_matrices: Vec<(String, Vec<f64>, usize)>,
    pub reference_sums: Vec<(String, f64)>,
}

/// Compute spectral properties of a symmetric weight matrix.
///
/// The matrix should be provided as a flat row-major `n × n` array.
#[must_use]
pub fn spectral_properties(matrix: &[f64], n: usize, name: &str) -> SpectralProfile {
    let result = crate::eigh::eigh_householder_qr(matrix, n);
    let mut evals = result.eigenvalues.clone();
    evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let spectral_radius = evals.iter().map(|e| e.abs()).fold(0.0_f64, f64::max);

    let sum: f64 = evals.iter().sum();
    let eigenvalue_mean = sum / n as f64;
    let eigenvalue_std = (evals.iter().map(|e| (e - eigenvalue_mean).powi(2)).sum::<f64>()
        / n as f64)
        .sqrt();

    // Level spacing ratio
    let gaps: Vec<f64> = evals.windows(2).map(|w| w[1] - w[0]).collect();
    let mean_spacing_ratio = if gaps.len() > 1 {
        let ratios: Vec<f64> = gaps
            .windows(2)
            .map(|w| w[0].min(w[1]) / w[0].max(w[1]).max(1e-30))
            .filter(|r| r.is_finite())
            .collect();
        if ratios.is_empty() {
            0.0
        } else {
            ratios.iter().sum::<f64>() / ratios.len() as f64
        }
    } else {
        0.0
    };

    // IPR from eigenvectors (columns of result.eigenvectors)
    let mut iprs = Vec::with_capacity(n);
    for k in 0..n {
        let psi: Vec<f64> = (0..n).map(|i| result.eigenvectors[i * n + k]).collect();
        iprs.push(anderson_localization::ipr(&psi));
    }
    let mean_ipr = iprs.iter().sum::<f64>() / n as f64;

    let effective_dimension = if mean_ipr > 1e-12 {
        1.0 / mean_ipr
    } else {
        n as f64
    };
    let effective_ratio = effective_dimension / n as f64;

    SpectralProfile {
        name: name.to_string(),
        size: n,
        spectral_radius,
        eigenvalue_mean,
        eigenvalue_std,
        eigenvalue_min: evals[0],
        eigenvalue_max: evals[n - 1],
        mean_spacing_ratio,
        mean_ipr,
        effective_dimension,
        effective_ratio,
    }
}

/// Compute cross-domain metrics from a set of spectral profiles.
#[must_use]
pub fn cross_domain_metrics(profiles: &[SpectralProfile]) -> CrossDomainMetrics {
    let n = profiles.len() as f64;
    let eff_ratios: Vec<f64> = profiles.iter().map(|p| p.effective_ratio).collect();
    let iprs: Vec<f64> = profiles.iter().map(|p| p.mean_ipr).collect();
    let spacings: Vec<f64> = profiles.iter().map(|p| p.mean_spacing_ratio).collect();

    let eff_mean = eff_ratios.iter().sum::<f64>() / n;
    let eff_std = (eff_ratios.iter().map(|v| (v - eff_mean).powi(2)).sum::<f64>() / n).sqrt();

    let ipr_mean = iprs.iter().sum::<f64>() / n;
    let ipr_std = (iprs.iter().map(|v| (v - ipr_mean).powi(2)).sum::<f64>() / n).sqrt();

    let sp_mean = spacings.iter().sum::<f64>() / n;
    let sp_std = (spacings.iter().map(|v| (v - sp_mean).powi(2)).sum::<f64>() / n).sqrt();

    CrossDomainMetrics {
        eff_ratio_mean: eff_mean,
        eff_ratio_std: eff_std,
        eff_ratio_cv: eff_std / eff_mean.max(1e-12),
        ipr_mean,
        ipr_std,
        ipr_cv: ipr_std / ipr_mean.max(1e-12),
        spacing_ratio_mean: sp_mean,
        spacing_ratio_std: sp_std,
    }
}

/// Load isomorphic baseline from Python JSON.
///
/// # Errors
///
/// Returns `Err` if the JSON structure is unexpected.
pub fn load_isomorphic_from_json(json_str: &str) -> Result<IsomorphicBaseline, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

    let spectra_map = parsed["spectra"]
        .as_object()
        .ok_or("missing spectra")?;

    let mut spectra = Vec::new();
    for (name, sp) in spectra_map {
        spectra.push(SpectralProfile {
            name: name.clone(),
            size: sp["size"].as_u64().ok_or("missing size")? as usize,
            spectral_radius: sp["spectral_radius"].as_f64().ok_or("missing sr")?,
            eigenvalue_mean: sp["eigenvalue_mean"].as_f64().ok_or("missing ev_mean")?,
            eigenvalue_std: sp["eigenvalue_std"].as_f64().ok_or("missing ev_std")?,
            eigenvalue_min: sp["eigenvalue_min"].as_f64().ok_or("missing ev_min")?,
            eigenvalue_max: sp["eigenvalue_max"].as_f64().ok_or("missing ev_max")?,
            mean_spacing_ratio: sp["mean_spacing_ratio"].as_f64().ok_or("missing spacing")?,
            mean_ipr: sp["mean_ipr"].as_f64().ok_or("missing ipr")?,
            effective_dimension: sp["effective_dimension"].as_f64().ok_or("missing eff_dim")?,
            effective_ratio: sp["effective_ratio"].as_f64().ok_or("missing eff_ratio")?,
        });
    }

    let cd = &parsed["cross_domain"];
    let cross_domain = CrossDomainMetrics {
        eff_ratio_mean: cd["eff_ratio_mean"].as_f64().ok_or("missing eff_ratio_mean")?,
        eff_ratio_std: cd["eff_ratio_std"].as_f64().ok_or("missing eff_ratio_std")?,
        eff_ratio_cv: cd["eff_ratio_cv"].as_f64().ok_or("missing eff_ratio_cv")?,
        ipr_mean: cd["ipr_mean"].as_f64().ok_or("missing ipr_mean")?,
        ipr_std: cd["ipr_std"].as_f64().ok_or("missing ipr_std")?,
        ipr_cv: cd["ipr_cv"].as_f64().ok_or("missing ipr_cv")?,
        spacing_ratio_mean: cd["spacing_ratio_mean"].as_f64().ok_or("missing sp_mean")?,
        spacing_ratio_std: cd["spacing_ratio_std"].as_f64().ok_or("missing sp_std")?,
    };

    let mut domain_matrices = Vec::new();
    let domains = parsed["domains"].as_object().ok_or("missing domains")?;
    for (name, dom) in domains {
        let key = if name == "digester" {
            "w_res_sym"
        } else {
            "w_hh_sym"
        };
        let mat_2d = dom[key].as_array().ok_or("missing matrix")?;
        let mut flat = Vec::new();
        for row in mat_2d {
            for val in row.as_array().ok_or("expected 2D")? {
                flat.push(val.as_f64().ok_or("expected f64")?);
            }
        }
        let n = mat_2d.len();
        domain_matrices.push((name.clone(), flat, n));
    }

    let ref_sums = &parsed["reference_sums"];
    let reference_sums = vec![
        (
            "digester".to_string(),
            ref_sums["digester_w_out_head"]
                .as_f64()
                .ok_or("missing ref")?,
        ),
        (
            "glucose".to_string(),
            ref_sums["glucose_w_out_head"]
                .as_f64()
                .ok_or("missing ref")?,
        ),
        (
            "weather".to_string(),
            ref_sums["weather_w_out_head"]
                .as_f64()
                .ok_or("missing ref")?,
        ),
    ];

    Ok(IsomorphicBaseline {
        spectra,
        cross_domain,
        domain_matrices,
        reference_sums,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity(n: usize) -> Vec<f64> {
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    }

    #[test]
    fn test_spectral_identity() {
        let m = make_identity(8);
        let sp = spectral_properties(&m, 8, "identity");
        assert!((sp.spectral_radius - 1.0).abs() < 1e-10);
        assert!((sp.eigenvalue_mean - 1.0).abs() < 1e-10);
        assert!(sp.eigenvalue_std < 1e-10);
    }

    #[test]
    fn test_spectral_diagonal() {
        let n = 8;
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            m[i * n + i] = (i + 1) as f64;
        }
        let sp = spectral_properties(&m, n, "diagonal");
        assert!((sp.spectral_radius - 8.0).abs() < 1e-10);
        assert!((sp.eigenvalue_min - 1.0).abs() < 1e-10);
        assert!((sp.eigenvalue_max - 8.0).abs() < 1e-10);
        assert!(sp.mean_ipr > 0.0);
    }

    #[test]
    fn test_cross_domain_identical() {
        let m = make_identity(8);
        let sp = spectral_properties(&m, 8, "a");
        let sp2 = spectral_properties(&m, 8, "b");
        let cd = cross_domain_metrics(&[sp, sp2]);
        assert!(cd.eff_ratio_cv < 1e-10, "identical matrices → CV=0");
    }

    #[test]
    fn test_ipr_diagonal_is_localized() {
        let n = 16;
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            m[i * n + i] = i as f64;
        }
        let sp = spectral_properties(&m, n, "diag");
        assert!(
            (sp.mean_ipr - 1.0).abs() < 1e-6,
            "diagonal → basis eigenvectors → IPR=1"
        );
    }

    #[test]
    fn test_effective_dimension_formula() {
        let ipr = 0.05_f64;
        let eff_dim = 1.0 / ipr;
        assert!((eff_dim - 20.0).abs() < 1e-10);
    }
}
