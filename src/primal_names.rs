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
pub const LOAMSPINE: &str = "loamspine";

/// Provenance braids / attribution primal (provenance trio).
pub const SWEETGRASS: &str = "sweetgrass";

/// Visualization / interactive exploration primal.
pub const PETALTONGUE: &str = "petaltongue";

/// Pure math / GPU compute (WGSL shaders, tensor ops, stats).
pub const BARRACUDA: &str = "barracuda";

/// biomeOS orchestrator.
pub const BIOMEOS: &str = "biomeos";

/// Display names for presentation contexts (dashboards, reports, handoffs).
///
/// These are the canonical mixed-case names as used in prose and UI,
/// distinct from the lowercase discovery hints above.
pub mod display {
    /// Display name for the `barraCuda` primal.
    pub const BARRACUDA: &str = "barraCuda";
    /// Display name for the toadstool primal.
    pub const TOADSTOOL: &str = "toadStool";
    /// Display name for the coralreef primal.
    pub const CORALREEF: &str = "coralReef";
    /// Display name for the neuralSpring primal.
    pub const NEURALSPRING: &str = "neuralSpring";
    /// Display name for the wetSpring primal.
    pub const WETSPRING: &str = "wetSpring";
    /// Display name for the hotSpring primal.
    pub const HOTSPRING: &str = "hotSpring";
    /// Display name for the groundSpring primal.
    pub const GROUNDSPRING: &str = "groundSpring";
    /// Display name for the airSpring primal.
    pub const AIRSPRING: &str = "airSpring";
    /// Display name for the squirrel primal.
    pub const SQUIRREL: &str = "Squirrel";
    /// Display name for the petaltongue primal.
    pub const PETALTONGUE: &str = "petalTongue";
    /// Display name for the biomeOS primal.
    pub const BIOMEOS: &str = "biomeOS";
    /// Display name for the nestgate primal.
    pub const NESTGATE: &str = "NestGate";
}

/// Provenance trio capability domains (used in `capability.call`).
pub mod domains {
    /// rhizoCrypt DAG session domain.
    pub const DAG: &str = "dag";
    /// loamSpine commit/certificate domain.
    pub const COMMIT: &str = "commit";
    /// sweetGrass provenance/attribution domain.
    pub const PROVENANCE: &str = "provenance";
    /// toadStool compute dispatch domain.
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
