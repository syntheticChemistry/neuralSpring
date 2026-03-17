// SPDX-License-Identifier: AGPL-3.0-or-later

//! Niche deployment self-knowledge for neuralSpring.
//!
//! A Spring is a niche validation domain — not a primal. It proves that
//! scientific Python baselines can be faithfully ported to sovereign
//! Rust + GPU compute using the ecoPrimals stack. The niche deploys as
//! a biomeOS graph (`graphs/neuralspring_deploy.toml`) that composes
//! real primals (`BearDog`, `Songbird`, `ToadStool`, etc.).
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

/// All capabilities this niche exposes to biomeOS.
///
/// Re-exports `config::ALL_CAPABILITIES` and adds niche-infrastructure
/// capabilities (provenance, data forwarding, compute offload).
pub const CAPABILITIES: &[&str] = &[
    // ── Science domain (from config::ALL_CAPABILITIES) ──
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
    // ── Health probes (biomeOS/Kubernetes pattern) ──
    "health.liveness",
    "health.readiness",
    // ── Provenance trio (biomeOS composition) ──
    "provenance.begin",
    "provenance.record",
    "provenance.complete",
    "provenance.status",
    // ── Cross-primal ──
    "primal.forward",
    "primal.discover",
    // ── Niche deployment (biomeOS graph composition) ──
    "capability.list",
    // ── Compute offload (Node Atomic) ──
    "compute.offload",
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
        "health.liveness":             [],
        "health.readiness":            [],
        "provenance.begin":    ["experiment_name"],
        "provenance.record":   ["session_id", "step_data"],
        "provenance.complete": ["session_id"],
        "provenance.status":   [],
    })
}

/// Cost estimates for biomeOS scheduling (measured on Eastgate hardware).
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
        "health.liveness":                 { "latency_ms": 0.1,   "cpu": "none",   "memory_bytes": 64 },
        "health.readiness":                { "latency_ms": 0.2,   "cpu": "none",   "memory_bytes": 128 },
        "provenance.begin":    { "latency_ms": 10.0, "cpu": "low", "memory_bytes": 512 },
        "provenance.record":   { "latency_ms": 5.0,  "cpu": "low", "memory_bytes": 1024 },
        "provenance.complete": { "latency_ms": 50.0, "cpu": "medium", "memory_bytes": 2048 },
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
}
