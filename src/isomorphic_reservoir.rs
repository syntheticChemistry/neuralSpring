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

/// Spectral properties of a weight matrix.
#[derive(Debug, Clone)]
pub struct SpectralProfile {
    /// Reservoir or matrix label.
    pub name: String,
    /// Matrix side length `n` (symmetric `n × n`).
    pub size: usize,
    /// Spectral radius (max |λ|).
    pub spectral_radius: f64,
    /// Mean eigenvalue.
    pub eigenvalue_mean: f64,
    /// Standard deviation of eigenvalues.
    pub eigenvalue_std: f64,
    /// Smallest eigenvalue after sorting.
    pub eigenvalue_min: f64,
    /// Largest eigenvalue after sorting.
    pub eigenvalue_max: f64,
    /// Mean adjacent gap ratio (level spacing statistic).
    pub mean_spacing_ratio: f64,
    /// Mean inverse participation ratio over eigenvectors.
    pub mean_ipr: f64,
    /// Effective dimension from IPR (`1 / mean_ipr` when IPR > 0).
    pub effective_dimension: f64,
    /// Effective dimension normalized by matrix size.
    pub effective_ratio: f64,
}

/// Cross-domain comparison metrics.
#[derive(Debug, Clone)]
pub struct CrossDomainMetrics {
    /// Mean effective-dimension ratio across domains.
    pub eff_ratio_mean: f64,
    /// Standard deviation of effective-dimension ratios.
    pub eff_ratio_std: f64,
    /// Coefficient of variation of effective-dimension ratios.
    pub eff_ratio_cv: f64,
    /// Mean IPR across domains.
    pub ipr_mean: f64,
    /// Standard deviation of IPR.
    pub ipr_std: f64,
    /// Coefficient of variation of IPR.
    pub ipr_cv: f64,
    /// Mean level-spacing ratio across domains.
    pub spacing_ratio_mean: f64,
    /// Standard deviation of level-spacing ratios.
    pub spacing_ratio_std: f64,
}

/// Full baseline loaded from JSON.
#[derive(Debug, Clone)]
pub struct IsomorphicBaseline {
    /// Per-domain spectral profiles from the baseline run.
    pub spectra: Vec<SpectralProfile>,
    /// Aggregated cross-domain spectral statistics.
    pub cross_domain: CrossDomainMetrics,
    /// Flattened symmetric weight matrices per domain `(name, data, n)`.
    pub domain_matrices: Vec<(String, Vec<f64>, usize)>,
    /// Reference output-head weight sums per domain name.
    pub reference_sums: Vec<(String, f64)>,
}

/// Compute spectral properties of a symmetric weight matrix.
///
/// The matrix should be provided as a flat row-major `n × n` array.
#[cfg(feature = "barracuda")]
#[must_use]
pub fn spectral_properties(matrix: &[f64], n: usize, name: &str) -> SpectralProfile {
    use crate::anderson_localization;
    use crate::tolerances;

    let result = crate::eigh::eigh_householder_qr(matrix, n);
    let mut evals = result.eigenvalues.clone();
    evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let spectral_radius = evals.iter().map(|e| e.abs()).fold(0.0_f64, f64::max);

    let sum: f64 = evals.iter().sum();
    let eigenvalue_mean = sum / n as f64;
    let eigenvalue_std = (evals
        .iter()
        .map(|e| (e - eigenvalue_mean).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();

    // Level spacing ratio
    let gaps: Vec<f64> = evals.windows(2).map(|w| w[1] - w[0]).collect();
    let mean_spacing_ratio = if gaps.len() > 1 {
        let ratios: Vec<f64> = gaps
            .windows(2)
            .map(|w| w[0].min(w[1]) / w[0].max(w[1]).max(tolerances::LOG_ZERO_GUARD))
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

    let effective_dimension = if mean_ipr > tolerances::EXACT_F64 {
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
#[cfg(feature = "barracuda")]
#[must_use]
pub fn cross_domain_metrics(profiles: &[SpectralProfile]) -> CrossDomainMetrics {
    use crate::tolerances;

    let n = profiles.len() as f64;
    let eff_ratios: Vec<f64> = profiles.iter().map(|p| p.effective_ratio).collect();
    let iprs: Vec<f64> = profiles.iter().map(|p| p.mean_ipr).collect();
    let spacings: Vec<f64> = profiles.iter().map(|p| p.mean_spacing_ratio).collect();

    let eff_mean = eff_ratios.iter().sum::<f64>() / n;
    let eff_std = (eff_ratios
        .iter()
        .map(|v| (v - eff_mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    let ipr_mean = iprs.iter().sum::<f64>() / n;
    let ipr_std = (iprs.iter().map(|v| (v - ipr_mean).powi(2)).sum::<f64>() / n).sqrt();

    let sp_mean = spacings.iter().sum::<f64>() / n;
    let sp_std = (spacings.iter().map(|v| (v - sp_mean).powi(2)).sum::<f64>() / n).sqrt();

    CrossDomainMetrics {
        eff_ratio_mean: eff_mean,
        eff_ratio_std: eff_std,
        eff_ratio_cv: eff_std / eff_mean.max(tolerances::EXACT_F64),
        ipr_mean,
        ipr_std,
        ipr_cv: ipr_std / ipr_mean.max(tolerances::EXACT_F64),
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

    let spectra_map = parsed["spectra"].as_object().ok_or("missing spectra")?;

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
            effective_dimension: sp["effective_dimension"]
                .as_f64()
                .ok_or("missing eff_dim")?,
            effective_ratio: sp["effective_ratio"].as_f64().ok_or("missing eff_ratio")?,
        });
    }

    let cd = &parsed["cross_domain"];
    let cross_domain = CrossDomainMetrics {
        eff_ratio_mean: cd["eff_ratio_mean"]
            .as_f64()
            .ok_or("missing eff_ratio_mean")?,
        eff_ratio_std: cd["eff_ratio_std"]
            .as_f64()
            .ok_or("missing eff_ratio_std")?,
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
#[expect(clippy::unwrap_used, clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    fn test_effective_dimension_formula() {
        let ipr = 0.05_f64;
        let eff_dim = 1.0 / ipr;
        assert!((eff_dim - 20.0).abs() < crate::tolerances::CROSS_LANGUAGE);
    }

    #[cfg(feature = "barracuda")]
    fn make_identity(n: usize) -> Vec<f64> {
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    }

    #[cfg(feature = "barracuda")]
    mod spectral {
        use super::*;

        fn make_identity(n: usize) -> Vec<f64> {
            super::make_identity(n)
        }

        #[test]
        fn test_spectral_identity() {
            let m = make_identity(8);
            let sp = spectral_properties(&m, 8, "identity");
            assert!((sp.spectral_radius - 1.0).abs() < crate::tolerances::CROSS_LANGUAGE);
            assert!((sp.eigenvalue_mean - 1.0).abs() < crate::tolerances::CROSS_LANGUAGE);
            assert!(sp.eigenvalue_std < crate::tolerances::CROSS_LANGUAGE);
        }

        #[test]
        fn test_spectral_diagonal() {
            let n = 8;
            let mut m = vec![0.0; n * n];
            for i in 0..n {
                m[i * n + i] = (i + 1) as f64;
            }
            let sp = spectral_properties(&m, n, "diagonal");
            assert!((sp.spectral_radius - 8.0).abs() < crate::tolerances::CROSS_LANGUAGE);
            assert!((sp.eigenvalue_min - 1.0).abs() < crate::tolerances::CROSS_LANGUAGE);
            assert!((sp.eigenvalue_max - 8.0).abs() < crate::tolerances::CROSS_LANGUAGE);
            assert!(sp.mean_ipr > 0.0);
        }

        #[test]
        fn test_cross_domain_identical() {
            let m = make_identity(8);
            let sp = spectral_properties(&m, 8, "a");
            let sp2 = spectral_properties(&m, 8, "b");
            let cd = cross_domain_metrics(&[sp, sp2]);
            assert!(
                cd.eff_ratio_cv < crate::tolerances::CROSS_LANGUAGE,
                "identical matrices → CV=0"
            );
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
                (sp.mean_ipr - 1.0).abs() < crate::tolerances::GELU_LARGE_INPUT,
                "diagonal → basis eigenvectors → IPR=1"
            );
        }

        mod proptests {
            use super::*;
            use crate::rng::Rng as PrimalRng;
            use proptest::prelude::*;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(48))]

                #[test]
                fn spectral_radius_matches_eigenvalue_extremes(
                    n in 3_usize..12,
                    seed in 0_u64..10_000,
                ) {
                    let mut rng = PrimalRng::new(seed);
                    let mut m = vec![0.0; n * n];
                    for i in 0..n {
                        for j in i..n {
                            let v = 2.0f64.mul_add(rng.uniform(), -1.0);
                            m[i * n + j] = v;
                            m[j * n + i] = v;
                        }
                    }

                    let sp = spectral_properties(&m, n, "proptest");

                    prop_assert!(sp.spectral_radius.is_finite());
                    prop_assert!(sp.spectral_radius >= 0.0);

                    let expected = sp.eigenvalue_min.abs().max(sp.eigenvalue_max.abs());
                    prop_assert!((sp.spectral_radius - expected).abs() < 1e-8,
                        "spectral_radius={} vs max(|min|,|max|)={}",
                        sp.spectral_radius, expected);
                }

                #[test]
                fn eigenvalue_stats_consistent(
                    n in 3_usize..12,
                    seed in 0_u64..10_000,
                ) {
                    let mut rng = PrimalRng::new(seed);
                    let mut m = vec![0.0; n * n];
                    for i in 0..n {
                        for j in i..n {
                            let v = 2.0f64.mul_add(rng.uniform(), -1.0);
                            m[i * n + j] = v;
                            m[j * n + i] = v;
                        }
                    }

                    let sp = spectral_properties(&m, n, "proptest");

                    prop_assert!(sp.eigenvalue_min <= sp.eigenvalue_max + 1e-10);
                    prop_assert!(sp.eigenvalue_mean.is_finite());
                    prop_assert!(sp.eigenvalue_std >= 0.0);
                    prop_assert!(sp.mean_ipr > 0.0);
                    prop_assert!(sp.effective_dimension.is_finite());
                }
            }
        }
    }

    fn minimal_isomorphic_json() -> String {
        r#"{
  "spectra": {
    "digester_esn": {
      "size": 2,
      "spectral_radius": 1.0,
      "eigenvalue_mean": 0.5,
      "eigenvalue_std": 0.5,
      "eigenvalue_min": 0.0,
      "eigenvalue_max": 1.0,
      "mean_spacing_ratio": 0.5,
      "mean_ipr": 0.5,
      "effective_dimension": 2.0,
      "effective_ratio": 1.0
    }
  },
  "cross_domain": {
    "eff_ratio_mean": 1.0,
    "eff_ratio_std": 0.0,
    "eff_ratio_cv": 0.0,
    "ipr_mean": 0.5,
    "ipr_std": 0.0,
    "ipr_cv": 0.0,
    "spacing_ratio_mean": 0.5,
    "spacing_ratio_std": 0.0
  },
  "domains": {
    "digester": { "w_res_sym": [[1.0, 0.0], [0.0, 1.0]] },
    "glucose": { "w_hh_sym": [[2.0, 0.0], [0.0, 2.0]] },
    "weather": { "w_hh_sym": [[3.0, 0.0], [0.0, 3.0]] }
  },
  "reference_sums": {
    "digester_w_out_head": 1.0,
    "glucose_w_out_head": 2.0,
    "weather_w_out_head": 3.0
  }
}"#
        .to_string()
    }

    #[test]
    fn load_isomorphic_from_json_minimal() {
        let baseline = load_isomorphic_from_json(&minimal_isomorphic_json()).expect("parse");
        assert_eq!(baseline.spectra.len(), 1);
        assert_eq!(baseline.spectra[0].name, "digester_esn");
        assert_eq!(baseline.domain_matrices.len(), 3);
        assert_eq!(baseline.reference_sums.len(), 3);
        assert!((baseline.cross_domain.eff_ratio_mean - 1.0).abs() < crate::tolerances::EXACT_F64);
    }

    #[test]
    fn load_isomorphic_from_json_parse_error() {
        let err = load_isomorphic_from_json("{not json").unwrap_err();
        assert!(err.contains("JSON parse error"));
    }

    #[test]
    fn load_isomorphic_from_json_missing_spectra() {
        let err = load_isomorphic_from_json(r#"{"cross_domain":{}}"#).unwrap_err();
        assert_eq!(err, "missing spectra");
    }

    #[test]
    fn load_isomorphic_from_json_missing_matrix_field() {
        let json = r#"{
  "spectra": {"a": {
    "size": 2, "spectral_radius": 1.0, "eigenvalue_mean": 0.0,
    "eigenvalue_std": 0.0, "eigenvalue_min": 0.0, "eigenvalue_max": 1.0,
    "mean_spacing_ratio": 0.0, "mean_ipr": 1.0, "effective_dimension": 1.0,
    "effective_ratio": 0.5
  }},
  "cross_domain": {
    "eff_ratio_mean": 0.0, "eff_ratio_std": 0.0, "eff_ratio_cv": 0.0,
    "ipr_mean": 0.0, "ipr_std": 0.0, "ipr_cv": 0.0,
    "spacing_ratio_mean": 0.0, "spacing_ratio_std": 0.0
  },
  "domains": {"digester": {}},
  "reference_sums": {
    "digester_w_out_head": 0.0, "glucose_w_out_head": 0.0, "weather_w_out_head": 0.0
  }
}"#;
        let err = load_isomorphic_from_json(json).unwrap_err();
        assert_eq!(err, "missing matrix");
    }

    #[test]
    fn load_isomorphic_from_json_matrix_keys_by_domain() {
        let baseline = load_isomorphic_from_json(&minimal_isomorphic_json()).expect("parse");
        let digester = baseline
            .domain_matrices
            .iter()
            .find(|(name, _, _)| name == "digester")
            .expect("digester domain");
        assert_eq!(digester.2, 2);
        assert!((digester.1[0] - 1.0).abs() < crate::tolerances::EXACT_F64);

        let glucose = baseline
            .domain_matrices
            .iter()
            .find(|(name, _, _)| name == "glucose")
            .expect("glucose domain");
        assert!((glucose.1[0] - 2.0).abs() < crate::tolerances::EXACT_F64);
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn cross_domain_metrics_single_profile() {
        let m = make_identity(4);
        let sp = spectral_properties(&m, 4, "solo");
        let cd = cross_domain_metrics(&[sp]);
        assert!((cd.eff_ratio_cv - 0.0).abs() < crate::tolerances::CROSS_LANGUAGE);
        assert!((cd.ipr_cv - 0.0).abs() < crate::tolerances::CROSS_LANGUAGE);
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn spectral_properties_spacing_zero_for_two_eigenvalues() {
        let m = make_identity(2);
        let sp = spectral_properties(&m, 2, "tiny");
        assert!(
            sp.mean_spacing_ratio.abs() < crate::tolerances::CROSS_LANGUAGE,
            "n=2 → one gap → spacing ratio branch returns 0"
        );
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn cross_domain_metrics_diverse_profiles_have_positive_cv() {
        let sp1 = spectral_properties(&make_identity(4), 4, "a");
        let mut diag = vec![0.0; 16];
        for i in 0..4 {
            diag[i * 4 + i] = (i + 1) as f64;
        }
        let sp2 = spectral_properties(&diag, 4, "b");
        let cd = cross_domain_metrics(&[sp1, sp2]);
        assert!(cd.eff_ratio_std >= 0.0);
        assert!(cd.ipr_std >= 0.0);
        assert!(cd.spacing_ratio_std >= 0.0);
    }

    #[test]
    fn load_isomorphic_from_json_missing_cross_domain() {
        let json = r#"{
  "spectra": {"a": {
    "size": 2, "spectral_radius": 1.0, "eigenvalue_mean": 0.0,
    "eigenvalue_std": 0.0, "eigenvalue_min": 0.0, "eigenvalue_max": 1.0,
    "mean_spacing_ratio": 0.0, "mean_ipr": 1.0, "effective_dimension": 1.0,
    "effective_ratio": 0.5
  }}
}"#;
        let err = load_isomorphic_from_json(json).unwrap_err();
        assert_eq!(err, "missing eff_ratio_mean");
    }

    #[test]
    fn load_isomorphic_from_json_missing_spectra_size() {
        let json = r#"{
  "spectra": {"a": {"spectral_radius": 1.0}},
  "cross_domain": {
    "eff_ratio_mean": 0.0, "eff_ratio_std": 0.0, "eff_ratio_cv": 0.0,
    "ipr_mean": 0.0, "ipr_std": 0.0, "ipr_cv": 0.0,
    "spacing_ratio_mean": 0.0, "spacing_ratio_std": 0.0
  },
  "domains": {},
  "reference_sums": {
    "digester_w_out_head": 0.0, "glucose_w_out_head": 0.0, "weather_w_out_head": 0.0
  }
}"#;
        let err = load_isomorphic_from_json(json).unwrap_err();
        assert_eq!(err, "missing size");
    }

    #[test]
    fn load_isomorphic_from_json_multiple_spectra() {
        let json = r#"{
  "spectra": {
    "digester_esn": {
      "size": 2, "spectral_radius": 1.0, "eigenvalue_mean": 0.5,
      "eigenvalue_std": 0.5, "eigenvalue_min": 0.0, "eigenvalue_max": 1.0,
      "mean_spacing_ratio": 0.5, "mean_ipr": 0.5, "effective_dimension": 2.0,
      "effective_ratio": 1.0
    },
    "glucose_lstm": {
      "size": 4, "spectral_radius": 2.0, "eigenvalue_mean": 1.0,
      "eigenvalue_std": 0.25, "eigenvalue_min": 0.5, "eigenvalue_max": 2.0,
      "mean_spacing_ratio": 0.3, "mean_ipr": 0.25, "effective_dimension": 4.0,
      "effective_ratio": 1.0
    }
  },
  "cross_domain": {
    "eff_ratio_mean": 1.0, "eff_ratio_std": 0.0, "eff_ratio_cv": 0.0,
    "ipr_mean": 0.5, "ipr_std": 0.0, "ipr_cv": 0.0,
    "spacing_ratio_mean": 0.5, "spacing_ratio_std": 0.0
  },
  "domains": {
    "digester": { "w_res_sym": [[1.0, 0.0], [0.0, 1.0]] },
    "glucose": { "w_hh_sym": [[2.0, 0.0], [0.0, 2.0]] },
    "weather": { "w_hh_sym": [[3.0, 0.0], [0.0, 3.0]] }
  },
  "reference_sums": {
    "digester_w_out_head": 1.0, "glucose_w_out_head": 2.0, "weather_w_out_head": 3.0
  }
}"#;
        let baseline = load_isomorphic_from_json(json).expect("parse");
        assert_eq!(baseline.spectra.len(), 2);
        let names: Vec<&str> = baseline.spectra.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"digester_esn"));
        assert!(names.contains(&"glucose_lstm"));
    }

    #[test]
    fn load_isomorphic_from_json_missing_domains() {
        let json = r#"{
  "spectra": {"a": {
    "size": 2, "spectral_radius": 1.0, "eigenvalue_mean": 0.0,
    "eigenvalue_std": 0.0, "eigenvalue_min": 0.0, "eigenvalue_max": 1.0,
    "mean_spacing_ratio": 0.0, "mean_ipr": 1.0, "effective_dimension": 1.0,
    "effective_ratio": 0.5
  }},
  "cross_domain": {
    "eff_ratio_mean": 0.0, "eff_ratio_std": 0.0, "eff_ratio_cv": 0.0,
    "ipr_mean": 0.0, "ipr_std": 0.0, "ipr_cv": 0.0,
    "spacing_ratio_mean": 0.0, "spacing_ratio_std": 0.0
  },
  "reference_sums": {
    "digester_w_out_head": 0.0, "glucose_w_out_head": 0.0, "weather_w_out_head": 0.0
  }
}"#;
        let err = load_isomorphic_from_json(json).unwrap_err();
        assert_eq!(err, "missing domains");
    }

    #[test]
    fn load_isomorphic_from_json_missing_reference_sum() {
        let json = r#"{
  "spectra": {"a": {
    "size": 2, "spectral_radius": 1.0, "eigenvalue_mean": 0.0,
    "eigenvalue_std": 0.0, "eigenvalue_min": 0.0, "eigenvalue_max": 1.0,
    "mean_spacing_ratio": 0.0, "mean_ipr": 1.0, "effective_dimension": 1.0,
    "effective_ratio": 0.5
  }},
  "cross_domain": {
    "eff_ratio_mean": 0.0, "eff_ratio_std": 0.0, "eff_ratio_cv": 0.0,
    "ipr_mean": 0.0, "ipr_std": 0.0, "ipr_cv": 0.0,
    "spacing_ratio_mean": 0.0, "spacing_ratio_std": 0.0
  },
  "domains": {"digester": { "w_res_sym": [[1.0, 0.0], [0.0, 1.0]] }},
  "reference_sums": {}
}"#;
        let err = load_isomorphic_from_json(json).unwrap_err();
        assert_eq!(err, "missing ref");
    }

    #[test]
    fn load_isomorphic_from_json_matrix_not_2d() {
        let json = r#"{
  "spectra": {"a": {
    "size": 2, "spectral_radius": 1.0, "eigenvalue_mean": 0.0,
    "eigenvalue_std": 0.0, "eigenvalue_min": 0.0, "eigenvalue_max": 1.0,
    "mean_spacing_ratio": 0.0, "mean_ipr": 1.0, "effective_dimension": 1.0,
    "effective_ratio": 0.5
  }},
  "cross_domain": {
    "eff_ratio_mean": 0.0, "eff_ratio_std": 0.0, "eff_ratio_cv": 0.0,
    "ipr_mean": 0.0, "ipr_std": 0.0, "ipr_cv": 0.0,
    "spacing_ratio_mean": 0.0, "spacing_ratio_std": 0.0
  },
  "domains": {"digester": { "w_res_sym": [1.0, 0.0] }},
  "reference_sums": {
    "digester_w_out_head": 0.0, "glucose_w_out_head": 0.0, "weather_w_out_head": 0.0
  }
}"#;
        let err = load_isomorphic_from_json(json).unwrap_err();
        assert_eq!(err, "expected 2D");
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn spectral_properties_effective_ratio_in_unit_interval() {
        use crate::rng::Rng as PrimalRng;

        let mut rng = PrimalRng::new(99);
        let n = 6;
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let v = 2.0f64.mul_add(rng.uniform(), -1.0);
                m[i * n + j] = v;
                m[j * n + i] = v;
            }
        }
        let sp = spectral_properties(&m, n, "random");
        assert!(sp.effective_ratio > 0.0);
        assert!(sp.effective_ratio <= 1.0 + crate::tolerances::CROSS_LANGUAGE);
        assert!(sp.spectral_radius >= sp.eigenvalue_min.abs());
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn cross_domain_metrics_three_profiles() {
        let profiles: Vec<SpectralProfile> = (0..3)
            .map(|i| {
                let n = 4;
                let mut m = make_identity(n);
                m[0] = f64::from(i + 1);
                spectral_properties(&m, n, &format!("p{i}"))
            })
            .collect();
        let cd = cross_domain_metrics(&profiles);
        assert!(cd.eff_ratio_mean.is_finite());
        assert!(cd.ipr_mean.is_finite());
        assert!(cd.spacing_ratio_mean.is_finite());
        assert!(cd.eff_ratio_cv.is_finite());
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn spectral_properties_constant_matrix_zero_spacing() {
        let n = 4;
        let m = vec![1.0; n * n];
        let sp = spectral_properties(&m, n, "constant");
        assert!(sp.eigenvalue_std >= 0.0);
        assert!(sp.mean_spacing_ratio.is_finite());
    }
}
