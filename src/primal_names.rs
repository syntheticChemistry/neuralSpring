// SPDX-License-Identifier: AGPL-3.0-or-later

//! Known primal name constants for capability-based discovery.
//!
//! Springs discover primals at runtime via capability probing and
//! `biomeos::find_socket()`. These constants eliminate hardcoded string
//! literals when referring to other primals in socket lookups, composition
//! status, and graph node identifiers.
//!
//! Primal code only has self-knowledge (see [`crate::niche`]); these names
//! are discovery hints, not compile-time coupling. Discovery always tries
//! capability probing first — name hints are the last resort.

/// Hardware discovery and GPU compute orchestration.
pub const TOADSTOOL: &str = "toadstool";

/// Security primal (Ed25519 signing, encryption, key generation).
pub const BEARDOG: &str = "beardog";

/// Network discovery mesh (IPC routing, socket registry).
pub const SONGBIRD: &str = "songbird";

/// Data storage and retrieval primal.
pub const NESTGATE: &str = "nestgate";

/// AI narration and ecology interpretation primal.
pub const SQUIRREL: &str = "squirrel";

/// Sovereign shader compiler.
pub const CORALREEF: &str = "coralreef";

/// DAG session management (provenance trio).
pub const RHIZOCRYPT: &str = "rhizocrypt";

/// Immutable ledger / certificate primal (provenance trio).
///
/// Reserved for future provenance trio wiring — no active IPC routing yet.
pub const LOAMSPINE: &str = "loamspine";

/// Provenance braids / attribution primal (provenance trio).
///
/// Reserved for future provenance trio wiring — no active IPC routing yet.
pub const SWEETGRASS: &str = "sweetgrass";

/// Visualization / interactive exploration primal.
pub const PETALTONGUE: &str = "petaltongue";

/// Pure math / GPU compute (WGSL shaders, tensor ops, stats).
pub const BARRACUDA: &str = "barracuda";

/// Defensive network security primal (metadata-only reconnaissance, threat detection).
pub const SKUNKBAT: &str = "skunkbat";

/// `biomeOS` orchestrator.
pub const BIOMEOS: &str = "biomeos";

/// Display names for presentation contexts (dashboards, reports, handoffs).
///
/// These are the canonical mixed-case names as used in prose and UI,
/// distinct from the lowercase discovery hints above.
pub mod display {
    /// Display name for the `barraCuda` primal.
    pub const BARRACUDA: &str = "barraCuda";
    /// Display name for the `toadStool` primal.
    pub const TOADSTOOL: &str = "toadStool";
    /// Display name for the `coralReef` primal.
    pub const CORALREEF: &str = "coralReef";
    /// Display name for the `BearDog` primal.
    pub const BEARDOG: &str = "BearDog";
    /// Display name for the `Songbird` primal.
    pub const SONGBIRD: &str = "Songbird";
    /// Display name for the `Squirrel` primal.
    pub const SQUIRREL: &str = "Squirrel";
    /// Display name for the `petalTongue` primal.
    pub const PETALTONGUE: &str = "petalTongue";
    /// Display name for the `biomeOS` orchestrator.
    pub const BIOMEOS: &str = "biomeOS";
    /// Display name for the `NestGate` primal.
    pub const NESTGATE: &str = "NestGate";
    /// Display name for the `skunkBat` primal.
    pub const SKUNKBAT: &str = "skunkBat";
    /// Display name for the `rhizoCrypt` primal.
    pub const RHIZOCRYPT: &str = "rhizoCrypt";
    /// Display name for the `loamSpine` primal.
    pub const LOAMSPINE: &str = "loamSpine";
    /// Display name for the `sweetGrass` primal.
    pub const SWEETGRASS: &str = "sweetGrass";

    /// Display name for the neuralSpring spring.
    pub const NEURALSPRING: &str = "neuralSpring";
    /// Display name for the wetSpring spring.
    pub const WETSPRING: &str = "wetSpring";
    /// Display name for the hotSpring spring.
    pub const HOTSPRING: &str = "hotSpring";
    /// Display name for the groundSpring spring.
    pub const GROUNDSPRING: &str = "groundSpring";
    /// Display name for the airSpring spring.
    pub const AIRSPRING: &str = "airSpring";
    /// Display name for the healthSpring spring.
    pub const HEALTHSPRING: &str = "healthSpring";
    /// Display name for the ludoSpring spring.
    pub const LUDOSPRING: &str = "ludoSpring";
}

/// Provenance trio capability domains (used in `capability.call`).
pub mod domains {
    /// `rhizoCrypt` DAG session domain.
    pub const DAG: &str = "dag";
    /// `loamSpine` commit/certificate domain.
    pub const COMMIT: &str = "commit";
    /// `sweetGrass` provenance/attribution domain.
    pub const PROVENANCE: &str = "provenance";
    /// `toadStool` compute dispatch domain.
    pub const COMPUTE: &str = "compute";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_names_are_lowercase() {
        for name in [
            TOADSTOOL,
            BEARDOG,
            SONGBIRD,
            NESTGATE,
            SQUIRREL,
            CORALREEF,
            BARRACUDA,
            RHIZOCRYPT,
            LOAMSPINE,
            SWEETGRASS,
            PETALTONGUE,
            SKUNKBAT,
            BIOMEOS,
        ] {
            assert_eq!(name, name.to_lowercase(), "{name} must be lowercase");
        }
    }

    #[test]
    fn domains_are_lowercase() {
        for d in [
            domains::DAG,
            domains::COMMIT,
            domains::PROVENANCE,
            domains::COMPUTE,
        ] {
            assert_eq!(d, d.to_lowercase(), "{d} must be lowercase");
        }
    }
}
