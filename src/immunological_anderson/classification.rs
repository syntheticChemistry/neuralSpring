// SPDX-License-Identifier: AGPL-3.0-or-later

//! AD state classification, Pielou evenness, and dimensional promotion.
//!
//! Maps cytokine-derived Anderson parameters to skin disease regimes and
//! converts cell-type heterogeneity into effective disorder for localization.

use super::AdSkinState;

/// Dimensional promotion metric.
///
/// Quantifies how barrier disruption changes effective dimension from
/// baseline 2D (intact epidermis) toward 3D (breached). The inverse of
/// Paper 06's tillage dimensional collapse.
#[must_use]
pub fn dimensional_promotion(intact_fraction: f64, baseline_d: f64, target_d: f64) -> f64 {
    let breach_fraction = 1.0 - intact_fraction.clamp(0.0, 1.0);
    breach_fraction.mul_add(target_d - baseline_d, baseline_d)
}

/// AD disease state classifier output.
#[derive(Debug, Clone)]
pub struct AdClassification {
    /// Inferred AD skin Anderson regime.
    pub state: AdSkinState,
    /// Classifier confidence in the state assignment.
    pub confidence: f64,
    /// Disagreement between tissue-compartment heads.
    pub compartment_disagreement: f64,
    /// Effective dimension from cytokine spectral analysis.
    pub effective_dimension: f64,
}

/// Classify AD skin state from cytokine-derived Anderson parameters.
///
/// Uses level spacing ratio and effective dimension to determine the
/// Anderson regime of cytokine propagation.
#[must_use]
pub fn classify_ad_state(
    level_spacing_ratio: f64,
    effective_dimension: f64,
    is_treated: bool,
) -> AdSkinState {
    const R_GOE: f64 = 0.5307;

    if is_treated && level_spacing_ratio < R_GOE {
        return AdSkinState::Treated;
    }

    if effective_dimension < 2.5 && level_spacing_ratio < R_GOE {
        AdSkinState::Healthy
    } else if effective_dimension > 2.7 && level_spacing_ratio > R_GOE {
        AdSkinState::Chronic
    } else if level_spacing_ratio > R_GOE {
        AdSkinState::Flare
    } else {
        AdSkinState::Healthy
    }
}

/// Pielou evenness index for cell-type heterogeneity -> disorder W.
///
/// `J = H' / ln(S)` where `H' = -sum(p_i * ln(p_i))`, `S` = number of
/// cell types. `J = 1.0` means perfectly even (maximum disorder), `J -> 0`
/// means dominated by one type (minimum disorder).
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "cell type count (small usize) → f64 for log normalization"
)]
pub fn pielou_evenness(cell_fractions: &[f64]) -> f64 {
    let s = cell_fractions.len();
    if s <= 1 {
        return 0.0;
    }
    let h_prime: f64 = cell_fractions
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    let h_max = (s as f64).ln();
    if h_max == 0.0 { 0.0 } else { h_prime / h_max }
}

/// Map Pielou evenness to Anderson disorder W.
///
/// In the Anderson model, `W/t` determines localization. Higher evenness
/// (more diverse cell populations) maps to higher disorder.
#[must_use]
pub fn evenness_to_disorder(evenness: f64, w_scale: f64) -> f64 {
    evenness * w_scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensional_promotion() {
        let fully_intact = dimensional_promotion(1.0, 2.0, 3.0);
        assert!((fully_intact - 2.0).abs() < 1e-10, "intact = baseline 2D");

        let fully_breached = dimensional_promotion(0.0, 2.0, 3.0);
        assert!((fully_breached - 3.0).abs() < 1e-10, "breached = target 3D");

        let half_breached = dimensional_promotion(0.5, 2.0, 3.0);
        assert!((half_breached - 2.5).abs() < 1e-10, "half = midpoint");
    }

    #[test]
    fn test_classify_ad_state() {
        assert_eq!(classify_ad_state(0.40, 2.0, false), AdSkinState::Healthy);
        assert_eq!(classify_ad_state(0.60, 2.8, false), AdSkinState::Chronic);
        assert_eq!(classify_ad_state(0.60, 2.6, false), AdSkinState::Flare);
        assert_eq!(classify_ad_state(0.40, 2.6, true), AdSkinState::Treated);
    }

    #[test]
    fn test_pielou_evenness() {
        let even = pielou_evenness(&[0.25, 0.25, 0.25, 0.25]);
        assert!((even - 1.0).abs() < 1e-10, "perfectly even = 1.0");

        let dominated = pielou_evenness(&[0.97, 0.01, 0.01, 0.01]);
        assert!(dominated < 0.3, "dominated should be low");
    }

    #[test]
    fn test_evenness_to_disorder() {
        let w = evenness_to_disorder(0.8, 10.0);
        assert!((w - 8.0).abs() < 1e-10);
    }
}
