// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario registry types — tracks, tiers, metadata, and the registry.
//!
//! Adapted from `primalSpring/ecoPrimal/src/validation/scenarios/registry.rs`
//! for neuralSpring's domain tracks.

use std::fmt;

/// Validation tier — determines what infrastructure is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Pure Rust, no IPC — runs anywhere without primals.
    Rust,
    /// Live NUCLEUS required — uses `CompositionContext` with deployed primals.
    Live,
    /// Both tiers: structural checks first, then live verification.
    Both,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rust => write!(f, "rust"),
            Self::Live => write!(f, "live"),
            Self::Both => write!(f, "both"),
        }
    }
}

/// Validation track — neuralSpring's domain areas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Track {
    /// Spectral analysis and Anderson localization.
    SpectralAnalysis,
    /// NUCLEUS composition parity.
    NucleusComposition,
    /// Inference pipeline (Squirrel-mediated).
    InferencePipeline,
    /// GPU parity (barraCuda Rust vs WGSL).
    GpuParity,
    /// Cross-spring integration.
    CrossSpring,
    /// Provenance and lineage.
    Provenance,
}

impl fmt::Display for Track {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SpectralAnalysis => write!(f, "spectral-analysis"),
            Self::NucleusComposition => write!(f, "nucleus-composition"),
            Self::InferencePipeline => write!(f, "inference-pipeline"),
            Self::GpuParity => write!(f, "gpu-parity"),
            Self::CrossSpring => write!(f, "cross-spring"),
            Self::Provenance => write!(f, "provenance"),
        }
    }
}

/// Metadata for a validation scenario.
#[derive(Debug, Clone)]
pub struct ScenarioMeta {
    /// Unique scenario identifier (e.g. `"nucleus_composition"`).
    pub id: &'static str,
    /// Which domain track this scenario belongs to.
    pub track: Track,
    /// Required infrastructure tier.
    pub tier: Tier,
    /// Source crate or binary this scenario was absorbed from.
    pub provenance_crate: &'static str,
    /// When the scenario was absorbed (ISO 8601 date).
    pub provenance_date: &'static str,
    /// Human-readable description.
    pub description: &'static str,
    /// Approximate number of checks.
    pub check_count: usize,
}

/// A runnable validation scenario.
pub struct Scenario {
    /// Scenario metadata.
    pub meta: ScenarioMeta,
    /// Tier 1 (Rust-only) validation function.
    pub run_rust: Option<fn(&mut primalspring::validation::ValidationResult)>,
    /// Tier 2 (Live) validation function using `CompositionContext`.
    pub run_live: Option<
        fn(
            &mut primalspring::composition::CompositionContext,
            &mut primalspring::validation::ValidationResult,
        ),
    >,
}

/// Registry of all known scenarios.
pub struct ScenarioRegistry {
    scenarios: Vec<Scenario>,
}

impl ScenarioRegistry {
    /// Create a new registry from a list of scenarios.
    #[must_use]
    pub fn new(scenarios: Vec<Scenario>) -> Self {
        Self { scenarios }
    }

    /// All registered scenarios.
    #[must_use]
    pub fn all(&self) -> &[Scenario] {
        &self.scenarios
    }

    /// Filter scenarios by track.
    #[must_use]
    pub fn by_track(&self, track: Track) -> Vec<&Scenario> {
        self.scenarios
            .iter()
            .filter(|s| s.meta.track == track)
            .collect()
    }

    /// Filter scenarios by tier.
    #[must_use]
    pub fn by_tier(&self, tier: Tier) -> Vec<&Scenario> {
        self.scenarios
            .iter()
            .filter(|s| s.meta.tier == tier)
            .collect()
    }

    /// Find a scenario by id.
    #[must_use]
    pub fn find(&self, id: &str) -> Option<&Scenario> {
        self.scenarios.iter().find(|s| s.meta.id == id)
    }

    /// Total number of registered scenarios.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scenarios.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scenarios.is_empty()
    }

    /// List all scenario IDs.
    #[must_use]
    pub fn ids(&self) -> Vec<&'static str> {
        self.scenarios.iter().map(|s| s.meta.id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_display() {
        assert_eq!(format!("{}", Tier::Rust), "rust");
        assert_eq!(format!("{}", Tier::Live), "live");
        assert_eq!(format!("{}", Tier::Both), "both");
    }

    #[test]
    fn track_display() {
        assert_eq!(format!("{}", Track::SpectralAnalysis), "spectral-analysis");
        assert_eq!(
            format!("{}", Track::NucleusComposition),
            "nucleus-composition"
        );
    }

    #[test]
    fn empty_registry() {
        let reg = ScenarioRegistry::new(vec![]);
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.ids().is_empty());
    }
}
