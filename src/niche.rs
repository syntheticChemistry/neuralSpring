// SPDX-License-Identifier: AGPL-3.0-or-later

//! Niche deployment self-knowledge for neuralSpring.
//!
//! A Spring is a niche validation domain — not a primal. It proves that
//! scientific Python baselines can be faithfully ported to sovereign
//! Rust + GPU compute using the ecoPrimals stack. The niche deploys as
//! a biomeOS graph (`graphs/neuralspring_deploy.toml`) that composes
//! real primals (`BearDog`, `Songbird`, `ToadStool`, etc.). The proto-nucleate
//! entry lives in `primalSpring/graphs/downstream/downstream_manifest.toml`.
//!
//! This module holds the niche's self-knowledge:
//! - Capability table (what the niche exposes via biomeOS)
//! - Semantic mappings (capability domain → science methods)
//! - Operation dependencies (parallelization hints for Pathway Learner)
//! - Cost estimates (scheduling hints for biomeOS)
//!
//! # Evolution
//!
//! The transitional `neuralspring_primal` binary exposes these capabilities
//! via a JSON-RPC server. The final form is graph-only deployment where
//! biomeOS orchestrates the niche directly from deploy graphs.

/// Niche identity (lowercase, used in IPC and deploy graphs).
pub const NICHE_NAME: &str = "neuralspring";

// ═══════════════════════════════════════════════════════════════════
// NUCLEUS bonding policy (proto-nucleate: neuralspring_inference)
// ═══════════════════════════════════════════════════════════════════

/// Bond type for inter-atomic composition.
///
/// `Metallic` indicates shared electron density — all primals in the
/// composition share a common trust domain and can freely exchange
/// capability calls without per-call authentication overhead.
pub const BOND_TYPE: &str = "Metallic";

/// Trust model governing cross-atomic boundaries.
///
/// `InternalNucleus` means all traffic stays within the NUCLEUS perimeter;
/// no external (internet-facing) endpoints are exposed. Tower Atomic
/// provides the BTSP encryption boundary.
pub const TRUST_MODEL: &str = "InternalNucleus";

/// Encryption tiers per atomic boundary.
///
/// | Atomic | Tier |
/// |--------|------|
/// | Tower (BearDog + Songbird) | Full BTSP encryption |
/// | Node (compute primals) | Delegated — Tower establishes session |
/// | Nest (storage primals) | Delegated — Tower establishes session |
/// | Meta (biomeOS, Squirrel) | Delegated — Tower establishes session |
pub const ENCRYPTION_TIER_TOWER: &str = "full";
/// Encryption tier for Node Atomic boundary.
pub const ENCRYPTION_TIER_NODE: &str = "tower_delegated";
/// Encryption tier for Nest Atomic boundary.
pub const ENCRYPTION_TIER_NEST: &str = "tower_delegated";
/// Encryption tier for Meta-tier boundary.
pub const ENCRYPTION_TIER_META: &str = "tower_delegated";

/// All capabilities this niche exposes to biomeOS.
///
/// Mirrors `config::ALL_CAPABILITIES` exactly — the test suite validates
/// parity between this array, the config constant, and the TOML registry.
pub const CAPABILITIES: &[&str] = &[
    // ── Science domain ──
    "science.spectral_analysis",
    "science.anderson_localization",
    "science.hessian_eigen",
    "science.agent_coordination",
    "science.ipr",
    "science.disorder_sweep",
    "science.training_trajectory",
    "science.evoformer_block",
    "science.structure_module",
    "science.folding_health",
    "science.gpu_dispatch",
    "science.cross_spring_provenance",
    "science.cross_spring_benchmark",
    "science.precision_routing",
    // ── Health probes (DEPLOYMENT_VALIDATION_STANDARD triad) ──
    "health.liveness",
    "health.readiness",
    "health.check",
    // ── Provenance trio (biomeOS composition) ──
    "provenance.begin",
    "provenance.record",
    "provenance.complete",
    "provenance.status",
    // ── Inference (Squirrel composition — proto-nucleate inference.*) ──
    "inference.complete",
    "inference.embed",
    "inference.models",
    // ── Cross-primal ──
    "primal.forward",
    "primal.discover",
    // ── Niche deployment (biomeOS graph composition) ──
    "capability.list",
    // ── Compute offload (Node Atomic) ──
    "compute.offload",
    // ── Identity + MCP (T4 discovery, composition pattern) ──
    "identity.get",
    "mcp.tools.list",
    // ── biomeOS v3.51 composition surface ──
    "composition.status",
    "method.register",
    // ── Security audit (skunkBat JH-5 forwarding) ──
    "security.audit_log",
];

/// Operation dependency hints for biomeOS Pathway Learner parallelization.
///
/// Maps each capability to the data inputs it requires, enabling the
/// Pathway Learner to determine which operations can run in parallel.
#[must_use]
pub fn operation_dependencies() -> serde_json::Value {
    serde_json::json!({
        "science.spectral_analysis":    ["hamiltonian_matrix"],
        "science.anderson_localization": ["dimension", "disorder_strength", "lattice_size"],
        "science.hessian_eigen":        ["loss_function", "parameters"],
        "science.agent_coordination":   ["agent_count", "strategy_matrix"],
        "science.ipr":                  ["eigenvectors"],
        "science.disorder_sweep":       ["disorder_range", "samples"],
        "science.training_trajectory":  ["model_weights", "training_config"],
        "science.evoformer_block":      ["msa_features", "pair_features"],
        "science.structure_module":     ["single_repr", "pair_repr"],
        "science.folding_health":       ["predicted_structure"],
        "science.gpu_dispatch":         ["operation", "tensors"],
        "science.cross_spring_provenance": ["experiment_id"],
        "science.cross_spring_benchmark":  ["benchmark_suite"],
        "science.precision_routing":    ["operation", "precision_hint"],
        "inference.complete":          ["prompt", "max_tokens", "temperature"],
        "inference.embed":             ["text"],
        "inference.models":            [],
        "health.liveness":             [],
        "health.readiness":            [],
        "provenance.begin":    ["experiment_name"],
        "provenance.record":   ["session_id", "step_data"],
        "provenance.complete": ["session_id"],
        "provenance.status":   [],
        "primal.forward":      ["capability", "params"],
        "primal.discover":     [],
        "capability.list":     [],
        "compute.offload":     ["operation", "tensors"],
        "health.check":        [],
        "identity.get":        [],
        "mcp.tools.list":      [],
    })
}

/// Cost estimates for biomeOS scheduling (reference hardware).
///
/// RTX 4070 12 GB + i9-12900K. Latency is p50 for representative inputs.
#[must_use]
pub fn cost_estimates() -> serde_json::Value {
    serde_json::json!({
        "science.spectral_analysis":       { "latency_ms": 5.0,   "cpu": "medium", "gpu": "preferred", "memory_bytes": 8192 },
        "science.anderson_localization":   { "latency_ms": 10.0,  "cpu": "medium", "gpu": "preferred", "memory_bytes": 16_384 },
        "science.hessian_eigen":           { "latency_ms": 50.0,  "cpu": "high",   "gpu": "preferred", "memory_bytes": 65_536 },
        "science.agent_coordination":      { "latency_ms": 2.0,   "cpu": "low",    "memory_bytes": 4096 },
        "science.ipr":                     { "latency_ms": 1.0,   "cpu": "low",    "gpu": "preferred", "memory_bytes": 2048 },
        "science.disorder_sweep":          { "latency_ms": 100.0, "cpu": "high",   "gpu": "required",  "memory_bytes": 131_072 },
        "science.training_trajectory":     { "latency_ms": 200.0, "cpu": "high",   "gpu": "preferred", "memory_bytes": 262_144 },
        "science.evoformer_block":         { "latency_ms": 50.0,  "cpu": "high",   "gpu": "required",  "memory_bytes": 524_288 },
        "science.structure_module":        { "latency_ms": 30.0,  "cpu": "high",   "gpu": "required",  "memory_bytes": 131_072 },
        "science.folding_health":          { "latency_ms": 5.0,   "cpu": "low",    "memory_bytes": 4096 },
        "science.gpu_dispatch":            { "latency_ms": 1.0,   "cpu": "low",    "gpu": "required",  "memory_bytes": 1024 },
        "science.cross_spring_provenance": { "latency_ms": 10.0,  "cpu": "low",    "memory_bytes": 2048 },
        "science.cross_spring_benchmark":  { "latency_ms": 500.0, "cpu": "high",   "gpu": "preferred", "memory_bytes": 524_288 },
        "science.precision_routing":       { "latency_ms": 0.5,   "cpu": "low",    "memory_bytes": 256 },
        "inference.complete":              { "latency_ms": 500.0, "cpu": "high",   "gpu": "preferred", "memory_bytes": 2_097_152 },
        "inference.embed":                 { "latency_ms": 100.0, "cpu": "medium", "gpu": "preferred", "memory_bytes": 1_048_576 },
        "inference.models":                { "latency_ms": 1.0,   "cpu": "none",   "memory_bytes": 256 },
        "health.liveness":                 { "latency_ms": 0.1,   "cpu": "none",   "memory_bytes": 64 },
        "health.readiness":                { "latency_ms": 0.2,   "cpu": "none",   "memory_bytes": 128 },
        "provenance.begin":    { "latency_ms": 10.0, "cpu": "low", "memory_bytes": 512 },
        "provenance.record":   { "latency_ms": 5.0,  "cpu": "low", "memory_bytes": 1024 },
        "provenance.complete": { "latency_ms": 50.0, "cpu": "medium", "memory_bytes": 2048 },
        "provenance.status":   { "latency_ms": 1.0,  "cpu": "none", "memory_bytes": 256 },
        "primal.forward":      { "latency_ms": 10.0, "cpu": "low", "memory_bytes": 2048 },
        "primal.discover":     { "latency_ms": 1.0,  "cpu": "none", "memory_bytes": 512 },
        "capability.list":     { "latency_ms": 0.1,  "cpu": "none", "memory_bytes": 256 },
        "compute.offload":     { "latency_ms": 5.0,  "cpu": "low", "gpu": "preferred", "memory_bytes": 4096 },
        "health.check":        { "latency_ms": 0.2,  "cpu": "none", "memory_bytes": 128 },
        "identity.get":        { "latency_ms": 0.1,  "cpu": "none", "memory_bytes": 256 },
        "mcp.tools.list":      { "latency_ms": 0.2,  "cpu": "none", "memory_bytes": 512 },
    })
}

/// Semantic mappings for the science capability domain.
///
/// Maps short names (used in biomeOS routing) to fully-qualified
/// capability strings, enabling `capability.call` routing.
#[must_use]
pub fn science_semantic_mappings() -> serde_json::Value {
    serde_json::json!({
        "spectral_analysis":       "science.spectral_analysis",
        "anderson_localization":   "science.anderson_localization",
        "hessian_eigen":           "science.hessian_eigen",
        "agent_coordination":      "science.agent_coordination",
        "ipr":                     "science.ipr",
        "disorder_sweep":          "science.disorder_sweep",
        "training_trajectory":     "science.training_trajectory",
        "evoformer_block":         "science.evoformer_block",
        "structure_module":        "science.structure_module",
        "folding_health":          "science.folding_health",
        "gpu_dispatch":            "science.gpu_dispatch",
        "cross_spring_provenance": "science.cross_spring_provenance",
        "cross_spring_benchmark":  "science.cross_spring_benchmark",
        "precision_routing":       "science.precision_routing",
    })
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
mod tests {
    use super::*;
    use crate::config;

    #[test]
    fn capabilities_are_not_empty() {
        assert!(!CAPABILITIES.is_empty());
    }

    #[test]
    fn capabilities_follow_semantic_naming() {
        for cap in CAPABILITIES {
            assert!(
                cap.contains('.'),
                "capability '{cap}' should follow domain.operation format"
            );
        }
    }

    #[test]
    fn science_capabilities_match_config() {
        for cap in config::ALL_CAPABILITIES {
            assert!(
                CAPABILITIES.contains(cap),
                "config capability '{cap}' missing from niche CAPABILITIES"
            );
        }
    }

    #[test]
    fn operation_dependencies_is_object() {
        let deps = operation_dependencies();
        assert!(deps.is_object());
    }

    #[test]
    fn cost_estimates_is_object() {
        let costs = cost_estimates();
        assert!(costs.is_object());
    }

    #[test]
    fn science_mappings_cover_all_science_capabilities() {
        let mappings = science_semantic_mappings();
        let map = mappings.as_object().unwrap();
        for cap in config::ALL_CAPABILITIES {
            if let Some(short) = cap.strip_prefix("science.") {
                assert!(
                    map.contains_key(short),
                    "science capability '{cap}' (key '{short}') missing from semantic mappings"
                );
            }
        }
    }

    #[test]
    fn niche_name_matches_convention() {
        assert_eq!(NICHE_NAME, "neuralspring");
        assert!(NICHE_NAME.chars().all(|c| c.is_ascii_lowercase()));
    }

    #[test]
    fn bonding_policy_declared() {
        assert_eq!(BOND_TYPE, "Metallic");
        assert_eq!(TRUST_MODEL, "InternalNucleus");
        assert_eq!(ENCRYPTION_TIER_TOWER, "full");
        assert_eq!(ENCRYPTION_TIER_NODE, "tower_delegated");
        assert_eq!(ENCRYPTION_TIER_NEST, "tower_delegated");
        assert_eq!(ENCRYPTION_TIER_META, "tower_delegated");
    }

    #[test]
    fn inference_capabilities_present() {
        assert!(
            CAPABILITIES.contains(&"inference.complete"),
            "niche must advertise inference.complete"
        );
        assert!(
            CAPABILITIES.contains(&"inference.embed"),
            "niche must advertise inference.embed"
        );
        assert!(
            CAPABILITIES.contains(&"inference.models"),
            "niche must advertise inference.models"
        );
    }

    #[test]
    fn evoformer_folding_capabilities_present() {
        assert!(
            CAPABILITIES.contains(&"science.evoformer_block"),
            "niche must advertise science.evoformer_block"
        );
        assert!(
            CAPABILITIES.contains(&"science.structure_module"),
            "niche must advertise science.structure_module"
        );
        assert!(
            CAPABILITIES.contains(&"science.folding_health"),
            "niche must advertise science.folding_health"
        );
    }

    #[test]
    fn composition_and_security_capabilities_present() {
        assert!(
            CAPABILITIES.contains(&"composition.status"),
            "niche must advertise composition.status (biomeOS v3.51)"
        );
        assert!(
            CAPABILITIES.contains(&"method.register"),
            "niche must advertise method.register (biomeOS v3.51)"
        );
        assert!(
            CAPABILITIES.contains(&"security.audit_log"),
            "niche must advertise security.audit_log (skunkBat JH-5)"
        );
    }
}
