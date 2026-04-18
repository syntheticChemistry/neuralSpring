// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Proto-nucleate graph composition validator.
//!
//! Validates that neuralSpring's inference composition graph can resolve
//! all declared primal nodes via capability-based discovery. This is the
//! NUCLEUS-layer validation target: Rust + Python baselines validate
//! science correctness, this binary validates primal composition patterns.
//!
//! ## Exit codes
//!
//! - 0: All discovered primals pass composition checks.
//! - 1: One or more composition checks failed.
//! - 2: No primals discovered (honest skip — not a failure, but nothing
//!   could be validated).
//!
//! ## Provenance
//!
//! Proto-nucleate: `primalSpring/graphs/downstream/downstream_manifest.toml` `[[downstream]]` `spring_name` = "neuralspring"
//! Version: v1.1.0 (2026-03-23)

use neural_spring::niche;
use neural_spring::validation::ValidationHarness;
use neural_spring::validation::composition::{self, BondType, DiscoveryResult, ProtoNucleateNode};
use std::time::Duration;

const IPC_TIMEOUT: Duration = Duration::from_secs(5);

fn main() {
    let mut h = ValidationHarness::new("nucleus_composition");
    let mut discovered = 0_usize;
    let mut skipped = 0_usize;

    let nodes = composition::inference_proto_nucleate_nodes();

    println!(
        "═══ NUCLEUS Composition Validator: {} proto-nucleate nodes ═══",
        nodes.len()
    );

    // ── Phase 1: Validate niche self-knowledge ──

    println!("\n── Phase 1: Niche bonding policy ──");

    h.check_bool("niche declares bond type", !niche::BOND_TYPE.is_empty());
    h.check_bool("niche declares trust model", !niche::TRUST_MODEL.is_empty());
    h.check_bool(
        "bond type is Metallic (proto-nucleate spec)",
        niche::BOND_TYPE == "Metallic",
    );
    h.check_bool(
        "encryption Tower = full",
        niche::ENCRYPTION_TIER_TOWER == "full",
    );
    h.check_bool(
        "encryption Node = tower_delegated",
        niche::ENCRYPTION_TIER_NODE == "tower_delegated",
    );
    h.check_bool(
        "encryption Nest = tower_delegated",
        niche::ENCRYPTION_TIER_NEST == "tower_delegated",
    );
    h.check_bool(
        "encryption Meta = tower_delegated",
        niche::ENCRYPTION_TIER_META == "tower_delegated",
    );

    // ── Phase 2: Capability coverage ──

    println!("\n── Phase 2: Capability coverage ──");

    let caps = niche::CAPABILITIES;

    h.check_bool(
        "niche advertises health.liveness",
        caps.contains(&"health.liveness"),
    );
    h.check_bool(
        "niche advertises capability.list",
        caps.contains(&"capability.list"),
    );
    h.check_bool(
        "niche advertises inference.complete",
        caps.contains(&"inference.complete"),
    );
    h.check_bool(
        "niche advertises inference.embed",
        caps.contains(&"inference.embed"),
    );
    h.check_bool(
        "niche advertises inference.models",
        caps.contains(&"inference.models"),
    );

    // ── Phase 3: Proto-nucleate node discovery ──

    println!("\n── Phase 3: Proto-nucleate node discovery ──");

    for node in &nodes {
        validate_node(&mut h, node, &mut discovered, &mut skipped);
    }

    // ── Phase 4: Bonding validation ──

    println!("\n── Phase 4: Bonding validation ──");

    validate_bonding(&mut h);

    // ── Phase 5: Summary ──

    let failed = h.total_count() - h.passed_count();

    println!();
    println!("Composition: {discovered} primals discovered, {skipped} skipped (not running)");

    let exit = composition::exit_code_skip_aware(h.passed_count(), failed, skipped);
    if exit == 2 {
        println!("SKIP: no primals available for live composition validation");
        println!(
            "{}/{} checks — honest skip (exit 2)",
            h.passed_count(),
            h.total_count()
        );
        h.emit_to_sink(&mut neural_spring::validation::StdoutSink);
        std::process::exit(2);
    }

    h.finish();
}

fn validate_node(
    h: &mut ValidationHarness,
    node: &ProtoNucleateNode,
    discovered: &mut usize,
    skipped: &mut usize,
) {
    let label_prefix = format!("proto-nucleate/{}", node.name);

    match composition::discover_primal_socket(node.name) {
        DiscoveryResult::Found(path) => {
            *discovered += 1;
            h.check_bool(&format!("{label_prefix}: socket discovered"), true);

            match composition::probe_liveness(&path, IPC_TIMEOUT) {
                Ok(()) => {
                    h.check_bool(&format!("{label_prefix}: health.liveness"), true);
                }
                Err(e) => {
                    h.check_bool(&format!("{label_prefix}: health.liveness ({e})"), false);
                    return;
                }
            }

            match composition::probe_capabilities(&path, IPC_TIMEOUT) {
                Ok(caps) => {
                    let has_cap = caps.iter().any(|c| c == node.by_capability);
                    h.check_bool(
                        &format!(
                            "{label_prefix}: advertises {} ({} caps total)",
                            node.by_capability,
                            caps.len()
                        ),
                        has_cap,
                    );
                }
                Err(e) => {
                    h.check_bool(&format!("{label_prefix}: capabilities.list ({e})"), false);
                }
            }
        }
        DiscoveryResult::NotFound { .. } => {
            *skipped += 1;
            println!(
                "  SKIP {}: not running (by_capability: {})",
                node.name, node.by_capability
            );
        }
    }
}

fn validate_bonding(h: &mut ValidationHarness) {
    h.check_bool(
        "bonding: declared bond type parses",
        parse_bond_type(niche::BOND_TYPE).is_some(),
    );

    if let Some(bond) = parse_bond_type(niche::BOND_TYPE) {
        match bond {
            BondType::Metallic => {
                h.check_bool(
                    "bonding: metallic requires internal trust model",
                    niche::TRUST_MODEL == "InternalNucleus",
                );
                h.check_bool(
                    "bonding: metallic requires tower full encryption",
                    niche::ENCRYPTION_TIER_TOWER == "full",
                );
            }
            BondType::Covalent => {
                h.check_bool("bonding: covalent requires shared family seed", true);
            }
            BondType::Ionic | BondType::Weak => {
                h.check_bool("bonding: ionic/weak — limited cross-call surface", true);
            }
        }
    }
}

fn parse_bond_type(s: &str) -> Option<BondType> {
    match s {
        "Metallic" => Some(BondType::Metallic),
        "Covalent" => Some(BondType::Covalent),
        "Ionic" => Some(BondType::Ionic),
        "Weak" => Some(BondType::Weak),
        _ => None,
    }
}
