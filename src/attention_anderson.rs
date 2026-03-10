// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 100: Anderson Spectral Analysis of Attention Weight Matrices.
//!
//! Novel composition of nS-01 (`eigh_f64`, `BatchIprGpu`) with
//! `coralForge` attention concepts. Attention quality correlates with
//! Anderson localization properties of symmetrized attention matrices.
//!
//! Composes:
//! - [`crate::anderson_localization`] — IPR computation
//! - [`crate::eigh`] — eigendecomposition
//! - [`crate::information_flow`] — attention spectral analysis concept

use crate::anderson_localization;

/// Per-configuration spectral result.
#[derive(Debug, Clone)]
pub struct AttentionSpectralResult {
    pub quality: f64,
    pub entropy: f64,
    pub spectral_radius: f64,
    pub mean_ipr: f64,
    pub participation: f64,
    pub xi: f64,
    pub eigenvalue_spread: f64,
}

/// Baseline loaded from Python JSON.
#[derive(Debug, Clone)]
pub struct AttentionAndersonBaseline {
    pub seq_len: usize,
    pub n_configs: usize,
    pub results: Vec<AttentionSpectralResult>,
    pub r_quality_entropy: f64,
    pub r_quality_ipr: f64,
    pub r_quality_xi: f64,
    pub r_entropy_ipr: f64,
    pub reference_matrix: Vec<f64>,
    pub reference_n: usize,
}

/// Compute spectral properties of a symmetrized attention matrix.
///
/// Input: flat row-major `n x n` symmetric matrix.
#[must_use]
pub fn attention_spectral(matrix: &[f64], n: usize) -> AttentionSpectralResult {
    let result = crate::eigh::eigh_householder_qr(matrix, n);
    let mut evals = result.eigenvalues.clone();
    evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let spectral_radius = evals.iter().map(|e| e.abs()).fold(0.0_f64, f64::max);
    let eigenvalue_spread = evals[n - 1] - evals[0];

    let mut iprs = Vec::with_capacity(n);
    for k in 0..n {
        let psi: Vec<f64> = (0..n).map(|i| result.eigenvectors[i * n + k]).collect();
        iprs.push(anderson_localization::ipr(&psi));
    }
    let mean_ipr = iprs.iter().sum::<f64>() / n as f64;
    let participation = if mean_ipr > 1e-12 {
        1.0 / mean_ipr
    } else {
        n as f64
    };
    let xi = participation / n as f64;

    AttentionSpectralResult {
        quality: 0.0,
        entropy: 0.0,
        spectral_radius,
        mean_ipr,
        participation,
        xi,
        eigenvalue_spread,
    }
}

/// Pearson correlation.
#[must_use]
pub fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    barracuda::stats::correlation::pearson_correlation(x, y).unwrap_or(0.0)
}

/// Load baseline from Python JSON.
///
/// # Errors
///
/// Returns `Err` if JSON structure is unexpected.
pub fn load_attention_anderson_from_json(
    json_str: &str,
) -> Result<AttentionAndersonBaseline, String> {
    let v: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("parse: {e}"))?;

    let results = v["results"]
        .as_array()
        .ok_or("missing results")?
        .iter()
        .map(|r| {
            Ok(AttentionSpectralResult {
                quality: r["quality"].as_f64().ok_or("quality")?,
                entropy: r["entropy"].as_f64().ok_or("entropy")?,
                spectral_radius: r["spectral_radius"].as_f64().ok_or("sr")?,
                mean_ipr: r["mean_ipr"].as_f64().ok_or("ipr")?,
                participation: r["participation"].as_f64().ok_or("part")?,
                xi: r["xi"].as_f64().ok_or("xi")?,
                eigenvalue_spread: r["eigenvalue_spread"].as_f64().ok_or("spread")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    let corr = &v["correlations"];
    let seq_len = v["seq_len"].as_u64().ok_or("seq_len")? as usize;

    let ref_mat_2d = v["reference_matrix"]
        .as_array()
        .ok_or("missing ref matrix")?;
    let mut ref_flat = Vec::new();
    for row in ref_mat_2d {
        for val in row.as_array().ok_or("2D")? {
            ref_flat.push(val.as_f64().ok_or("f64")?);
        }
    }
    let ref_n = ref_mat_2d.len();

    Ok(AttentionAndersonBaseline {
        seq_len,
        n_configs: v["n_configs"].as_u64().ok_or("n_configs")? as usize,
        results,
        r_quality_entropy: corr["r_quality_entropy"].as_f64().ok_or("r_qe")?,
        r_quality_ipr: corr["r_quality_ipr"].as_f64().ok_or("r_qi")?,
        r_quality_xi: corr["r_quality_xi"].as_f64().ok_or("r_qx")?,
        r_entropy_ipr: corr["r_entropy_ipr"].as_f64().ok_or("r_ei")?,
        reference_matrix: ref_flat,
        reference_n: ref_n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_spectral_identity() {
        let n = 8;
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0 / n as f64;
        }
        let sp = attention_spectral(&m, n);
        assert!(sp.spectral_radius > 0.0);
        assert!(sp.mean_ipr > 0.0);
    }

    #[test]
    fn test_attention_spectral_uniform() {
        let n = 8;
        let val = 1.0 / n as f64;
        let m = vec![val; n * n];
        let sp = attention_spectral(&m, n);
        assert!(sp.eigenvalue_spread.is_finite());
    }

    #[test]
    fn test_pearson_self() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r = pearson_r(&x, &x);
        assert!((r - 1.0).abs() < 1e-10);
    }
}
