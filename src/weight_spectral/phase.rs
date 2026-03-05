// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spectral phase classification from level spacing ratio.
//!
//! Maps the continuous level spacing ratio to a qualitative label using
//! Anderson localization theory.

/// GOE (Gaussian Orthogonal Ensemble) expected level spacing ratio.
pub const GOE_LEVEL_SPACING: f64 = 0.530_95;

/// Poisson expected level spacing ratio (localized regime).
pub const POISSON_LEVEL_SPACING: f64 = 0.386_29;

/// Discrete spectral phase classification.
///
/// Maps the continuous level spacing ratio to a qualitative label using
/// Anderson localization theory. Thresholds derived from GOE (0.531) and
/// Poisson (0.386) ensembles with a critical window around the Anderson
/// transition point.
///
/// Cross-spring evolution: hotSpring `proxy.rs` → `BarraCUDA` → neuralSpring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralPhase {
    /// GOE-like: delocalized eigenstates, good generalization.
    Extended,
    /// Near Anderson transition: mixed localization character.
    Critical,
    /// Poisson-like: localized eigenstates, memorization risk.
    Localized,
}

impl std::fmt::Display for SpectralPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Extended => write!(f, "extended"),
            Self::Critical => write!(f, "critical"),
            Self::Localized => write!(f, "localized"),
        }
    }
}

/// Classify level spacing ratio into a spectral phase.
///
/// Thresholds: Extended ≥ 0.48, Critical ∈ [0.42, 0.48), Localized < 0.42.
/// These bracket the GOE–Poisson crossover with a narrow critical window.
#[must_use]
pub fn classify_phase(lsr: f64) -> SpectralPhase {
    if lsr >= 0.48 {
        SpectralPhase::Extended
    } else if lsr >= 0.42 {
        SpectralPhase::Critical
    } else {
        SpectralPhase::Localized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_classification_extended() {
        assert_eq!(classify_phase(0.53), SpectralPhase::Extended);
        assert_eq!(classify_phase(0.48), SpectralPhase::Extended);
    }

    #[test]
    fn phase_classification_critical() {
        assert_eq!(classify_phase(0.45), SpectralPhase::Critical);
        assert_eq!(classify_phase(0.42), SpectralPhase::Critical);
    }

    #[test]
    fn phase_classification_localized() {
        assert_eq!(classify_phase(0.38), SpectralPhase::Localized);
        assert_eq!(classify_phase(0.0), SpectralPhase::Localized);
    }

    #[test]
    fn phase_display() {
        assert_eq!(format!("{}", SpectralPhase::Extended), "extended");
        assert_eq!(format!("{}", SpectralPhase::Critical), "critical");
        assert_eq!(format!("{}", SpectralPhase::Localized), "localized");
    }
}
