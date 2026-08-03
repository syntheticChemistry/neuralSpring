// SPDX-License-Identifier: AGPL-3.0-or-later

//! Proto-nucleate graph definitions and bonding policy for composition validation.

use crate::primal_names;

/// Proto-nucleate node descriptor for composition validation.
#[derive(Debug, Clone)]
pub struct ProtoNucleateNode {
    /// Node name (lowercase discovery hint).
    pub name: &'static str,
    /// Primary capability used for `by_capability` discovery.
    pub by_capability: &'static str,
    /// Whether this node is required for the composition to be valid.
    pub required: bool,
}

/// The proto-nucleate graph for neuralSpring inference composition.
///
/// Derived from `primalSpring/graphs/downstream/downstream_manifest.toml`
/// `[[downstream]] spring_name = "neuralspring"`.
///
/// The upstream manifest entry defines:
///   fragments:  `["tower_atomic", "node_atomic", "meta_tier"]`
///   `depends_on`: `["beardog", "songbird", "coralreef", "toadstool", "barracuda", "squirrel"]`
///
/// This function returns one node per `depends_on` primal plus biomeOS (the
/// orchestrator). `NestGate` is NOT in the proto-nucleate `depends_on` — it
/// appears in the richer `spring_deploy_manifest.toml` graph instead.
#[must_use]
pub fn inference_proto_nucleate_nodes() -> Vec<ProtoNucleateNode> {
    vec![
        ProtoNucleateNode {
            name: primal_names::BIOMEOS,
            by_capability: "graph.deploy",
            required: false,
        },
        // Tower Atomic
        ProtoNucleateNode {
            name: primal_names::BEARDOG,
            by_capability: "security",
            required: false,
        },
        ProtoNucleateNode {
            name: primal_names::SONGBIRD,
            by_capability: "discovery",
            required: false,
        },
        // Node Atomic
        ProtoNucleateNode {
            name: primal_names::CORALREEF,
            by_capability: "shader.compile.wgsl",
            required: false,
        },
        ProtoNucleateNode {
            name: primal_names::TOADSTOOL,
            by_capability: "compute.dispatch.submit",
            required: false,
        },
        ProtoNucleateNode {
            name: primal_names::BARRACUDA,
            by_capability: "tensor.matmul",
            required: false,
        },
        // Meta-tier
        ProtoNucleateNode {
            name: primal_names::SQUIRREL,
            by_capability: "ai.query",
            required: false,
        },
    ]
}

/// The `validation_capabilities` from the upstream proto-nucleate manifest.
///
/// These are the IPC methods that the primal proof must exercise: each one
/// is a capability that the NUCLEUS composition exposes and that neuralSpring's
/// science depends on. The primal proof calls these via IPC and compares
/// results against Python/Rust baselines.
///
/// Source: `downstream_manifest.toml` `[[downstream]] spring_name = "neuralspring"`
pub const PROTO_NUCLEATE_VALIDATION_CAPABILITIES: &[&str] = &[
    "tensor.matmul",
    "tensor.create",
    "compute.dispatch",
    "inference.complete",
    "inference.embed",
    "stats.mean",
    "crypto.hash",
];

/// Bonding policy for NUCLEUS composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondType {
    /// Shared trust domain — all primals freely exchange capabilities.
    Metallic,
    /// Different families — limited cross-call surface.
    Ionic,
    /// Same family seed — full meld routing.
    Covalent,
    /// Read-only ephemeral connections.
    Weak,
}

impl std::fmt::Display for BondType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Metallic => f.write_str("Metallic"),
            Self::Ionic => f.write_str("Ionic"),
            Self::Covalent => f.write_str("Covalent"),
            Self::Weak => f.write_str("Weak"),
        }
    }
}

/// Skip-aware exit code: 0 = pass, 1 = fail, 2 = all skipped.
///
/// Follows primalSpring's `exit_code_skip_aware` pattern — composition
/// validators that find no live primals exit 2 (honest skip), not 0.
#[must_use]
pub const fn exit_code_skip_aware(passed: usize, failed: usize, skipped: usize) -> i32 {
    if failed > 0 {
        1
    } else if passed > 0 {
        0
    } else if skipped > 0 {
        2
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_code_all_pass() {
        assert_eq!(exit_code_skip_aware(5, 0, 0), 0);
    }

    #[test]
    fn exit_code_any_fail() {
        assert_eq!(exit_code_skip_aware(3, 1, 0), 1);
    }

    #[test]
    fn exit_code_all_skip() {
        assert_eq!(exit_code_skip_aware(0, 0, 5), 2);
    }

    #[test]
    fn exit_code_pass_and_skip() {
        assert_eq!(exit_code_skip_aware(2, 0, 3), 0);
    }

    #[test]
    fn exit_code_empty() {
        assert_eq!(exit_code_skip_aware(0, 0, 0), 1);
    }

    #[test]
    fn proto_nucleate_nodes_match_upstream_manifest() {
        let nodes = inference_proto_nucleate_nodes();
        let names: Vec<&str> = nodes.iter().map(|n| n.name).collect();
        // depends_on from downstream_manifest.toml:
        assert!(names.contains(&"beardog"));
        assert!(names.contains(&"songbird"));
        assert!(names.contains(&"coralreef"));
        assert!(names.contains(&"toadstool"));
        assert!(names.contains(&"barracuda"));
        assert!(names.contains(&"squirrel"));
        // biomeOS is the orchestrator (not in depends_on but always present)
        assert!(names.contains(&"biomeos"));
        // nestgate is NOT in the proto-nucleate depends_on
        assert!(!names.contains(&"nestgate"));
    }

    #[test]
    fn validation_capabilities_match_upstream_manifest() {
        assert_eq!(PROTO_NUCLEATE_VALIDATION_CAPABILITIES.len(), 7);
        assert!(PROTO_NUCLEATE_VALIDATION_CAPABILITIES.contains(&"tensor.matmul"));
        assert!(PROTO_NUCLEATE_VALIDATION_CAPABILITIES.contains(&"inference.complete"));
        assert!(PROTO_NUCLEATE_VALIDATION_CAPABILITIES.contains(&"crypto.hash"));
    }

    #[test]
    fn bond_type_display() {
        assert_eq!(BondType::Metallic.to_string(), "Metallic");
        assert_eq!(BondType::Ionic.to_string(), "Ionic");
        assert_eq!(BondType::Covalent.to_string(), "Covalent");
        assert_eq!(BondType::Weak.to_string(), "Weak");
    }

    #[test]
    fn proto_nucleate_nodes_all_optional() {
        for node in inference_proto_nucleate_nodes() {
            assert!(!node.required, "{} should be optional", node.name);
            assert!(!node.by_capability.is_empty());
        }
    }
}
