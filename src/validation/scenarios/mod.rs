// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation scenarios — eukaryotic evolution of `validate_*` binaries.
//!
//! Each scenario absorbs the essential validation logic from a
//! pre-extinction binary into a reusable, composable module with
//! [`ScenarioMeta`] provenance and two-tier execution (Rust / Live).
//!
//! ## Usage
//!
//! ```ignore
//! let registry = build_registry();
//! for scenario in registry.all() {
//!     println!("{}: {} checks", scenario.meta.id, scenario.meta.check_count);
//! }
//! ```

pub mod registry;
mod s_composition_evolution;
mod s_compute_dispatch;
mod s_cross_gate_dispatch;
mod s_gpu_parity;
mod s_inference_composition;
mod s_nest_commit;
mod s_nucleus_composition;
mod s_nucleus_tower;
mod s_schema_standard;
mod s_science_composition;
mod s_signal_dispatch;

pub use registry::{Scenario, ScenarioMeta, ScenarioRegistry, Tier, Track};

/// Build the full scenario registry with all absorbed scenarios.
#[must_use]
pub fn build_registry() -> ScenarioRegistry {
    ScenarioRegistry::new(vec![
        s_nucleus_composition::SCENARIO,
        s_inference_composition::SCENARIO,
        s_science_composition::SCENARIO,
        s_nucleus_tower::SCENARIO,
        s_compute_dispatch::SCENARIO,
        s_composition_evolution::SCENARIO,
        s_signal_dispatch::SCENARIO,
        s_nest_commit::SCENARIO,
        s_schema_standard::SCENARIO,
        s_gpu_parity::SCENARIO,
        s_cross_gate_dispatch::SCENARIO,
    ])
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    fn registry_has_11_scenarios() {
        let reg = build_registry();
        assert_eq!(reg.len(), 11);
    }

    #[test]
    fn all_scenarios_have_unique_ids() {
        let reg = build_registry();
        let ids = reg.ids();
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len(), "duplicate scenario IDs");
    }

    #[test]
    fn find_by_id() {
        let reg = build_registry();
        let s = reg.find("nucleus_composition");
        assert!(s.is_some());
        assert_eq!(s.unwrap().meta.track, Track::NucleusComposition);
    }

    #[test]
    fn filter_by_track() {
        let reg = build_registry();
        let nucleus = reg.by_track(Track::NucleusComposition);
        assert!(
            nucleus.len() >= 2,
            "should have nucleus_composition + nucleus_tower + compute_dispatch"
        );
        let cross_gate = reg.by_track(Track::CrossGate);
        assert_eq!(cross_gate.len(), 1, "should have cross_gate_dispatch");
    }

    #[test]
    fn all_check_counts_positive() {
        let reg = build_registry();
        for s in reg.all() {
            assert!(
                s.meta.check_count > 0,
                "scenario {} has 0 checks",
                s.meta.id
            );
        }
    }
}
