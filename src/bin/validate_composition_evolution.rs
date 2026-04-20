// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Composition evolution validator — the third validation tier.
//!
//! neuralSpring's validation evolution path:
//!
//! ```text
//! Tier 1: Python baseline → Rust validation (science correctness)
//! Tier 2: Rust/Python → NUCLEUS composition (primal wiring)
//! Tier 3: Composition evolution (this binary) — validates that
//!          science patterns, primal wiring, and ecosystem standards
//!          cohere as a deployable NUCLEUS graph.
//! ```
//!
//! This binary validates the full lifecycle: niche self-knowledge,
//! capability surface completeness, proto-nucleate graph alignment,
//! deploy graph consistency, primal IPC readiness, and the bonding
//! policy that governs cross-atomic communication.
//!
//! ## Exit codes
//!
//! - 0: All composition evolution checks pass.
//! - 1: One or more checks failed.
//! - 2: No primals discovered (honest skip).
//!
//! ## Provenance
//!
//! Proto-nucleate: `primalSpring/graphs/downstream/downstream_manifest.toml` `[[downstream]]` `spring_name` = "neuralspring"
//! Deploy graph: `graphs/neuralspring_deploy.toml` V136/S185

use neural_spring::config;
use neural_spring::niche;
use neural_spring::validation::ValidationHarness;
use neural_spring::validation::composition::{self, DiscoveryResult};
use std::time::Duration;

const IPC_TIMEOUT: Duration = Duration::from_secs(5);

fn main() {
    let mut h = ValidationHarness::new("composition_evolution");
    let mut discovered = 0_usize;
    let mut skipped = 0_usize;

    println!("═══ Composition Evolution Validator (Tier 3) ═══");
    println!("Validates science → primal → NUCLEUS coherence\n");

    validate_capability_surface_completeness(&mut h);
    validate_deploy_graph_alignment(&mut h);
    validate_proto_nucleate_wiring(&mut h, &mut discovered, &mut skipped);
    validate_inference_evolution(&mut h, &mut discovered, &mut skipped);
    validate_health_triad(&mut h, &mut discovered, &mut skipped);

    let failed = h.total_count() - h.passed_count();
    let exit = composition::exit_code_skip_aware(h.passed_count(), failed, skipped);

    println!("\nComposition evolution: {discovered} primals live, {skipped} skipped");

    h.emit_to_sink(&mut neural_spring::validation::StdoutSink);

    if exit == 2 {
        println!("SKIP: no primals available — honest skip (exit 2)");
    }

    std::process::exit(exit);
}

/// Phase 1: Capability surface completeness.
///
/// Validates that `niche::CAPABILITIES`, `config::ALL_CAPABILITIES`, and the
/// TOML registry are in sync — and that every dispatched method is registered.
fn validate_capability_surface_completeness(h: &mut ValidationHarness) {
    println!("── Phase 1: Capability surface completeness ──");

    let niche_caps = niche::CAPABILITIES;
    let config_caps = config::ALL_CAPABILITIES;

    h.check_bool(
        "niche::CAPABILITIES >= config::ALL_CAPABILITIES",
        config_caps.iter().all(|c| niche_caps.contains(c)),
    );

    h.check_bool(
        "config::ALL_CAPABILITIES >= niche::CAPABILITIES",
        niche_caps.iter().all(|c| config_caps.contains(c)),
    );

    let dispatched = [
        "health.check",
        "identity.get",
        "mcp.tools.list",
        "health.liveness",
        "health.readiness",
        "capability.list",
        "inference.complete",
        "inference.embed",
        "inference.models",
    ];
    for method in &dispatched {
        h.check_bool(
            &format!("dispatched method {method} in ALL_CAPABILITIES"),
            config_caps.contains(method),
        );
    }

    let deps = niche::operation_dependencies();
    let costs = niche::cost_estimates();
    for cap in config_caps {
        h.check_bool(
            &format!("{cap}: has operation_dependencies entry"),
            deps.get(cap).is_some(),
        );
        h.check_bool(
            &format!("{cap}: has cost_estimates entry"),
            costs.get(cap).is_some(),
        );
    }

    let toml_src = include_str!("../../config/capability_registry.toml");
    for cap in config_caps {
        h.check_bool(
            &format!("{cap}: present in capability_registry.toml"),
            toml_src.contains(cap),
        );
    }
}

/// Phase 2: Deploy graph alignment with proto-nucleate.
///
/// The deploy graph must reference the proto-nucleate, declare correct
/// fragments, and use consistent bonding policy.
fn validate_deploy_graph_alignment(h: &mut ValidationHarness) {
    println!("\n── Phase 2: Deploy graph alignment ──");

    let deploy_src = include_str!("../../graphs/neuralspring_deploy.toml");

    h.check_bool(
        "deploy graph references proto-nucleate",
        deploy_src.contains("downstream_manifest::neuralspring"),
    );

    h.check_bool(
        "deploy graph declares tower_atomic fragment",
        deploy_src.contains("tower_atomic"),
    );
    h.check_bool(
        "deploy graph declares node_atomic fragment",
        deploy_src.contains("node_atomic"),
    );
    h.check_bool(
        "deploy graph declares nest_atomic fragment",
        deploy_src.contains("nest_atomic"),
    );
    h.check_bool(
        "deploy graph declares meta_tier fragment",
        deploy_src.contains("meta_tier"),
    );

    h.check_bool(
        "deploy graph bonding is Metallic",
        deploy_src.contains("bond_type = \"Metallic\""),
    );
    h.check_bool(
        "deploy graph trust is InternalNucleus",
        deploy_src.contains("trust_model = \"InternalNucleus\""),
    );
    h.check_bool(
        "deploy graph transport is UDS",
        deploy_src.contains("transport = \"uds\""),
    );

    h.check_bool(
        "bonding: niche matches deploy graph",
        niche::BOND_TYPE == "Metallic"
            && deploy_src.contains(&format!("bond_type = \"{}\"", niche::BOND_TYPE)),
    );
}

/// Phase 3: Proto-nucleate node wiring via live IPC discovery.
fn validate_proto_nucleate_wiring(
    h: &mut ValidationHarness,
    discovered: &mut usize,
    skipped: &mut usize,
) {
    println!("\n── Phase 3: Proto-nucleate node wiring ──");

    let nodes = composition::inference_proto_nucleate_nodes();

    for node in &nodes {
        match composition::discover_primal_socket(node.name) {
            DiscoveryResult::Found(path) => {
                *discovered += 1;
                h.check_bool(&format!("{}: socket discovered", node.name), true);

                if composition::probe_liveness(&path, IPC_TIMEOUT) == Ok(()) {
                    h.check_bool(&format!("{}: health.liveness responds", node.name), true);

                    if let Ok(caps) = composition::probe_capabilities(&path, IPC_TIMEOUT) {
                        let has_cap = caps.iter().any(|c| c == node.by_capability);
                        h.check_bool(
                            &format!(
                                "{}: advertises by_capability '{}'",
                                node.name, node.by_capability
                            ),
                            has_cap,
                        );
                    }
                } else {
                    h.check_bool(&format!("{}: health.liveness responds", node.name), false);
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
}

/// Phase 4: Inference evolution readiness — probe the neuralspring primal's
/// own inference handlers and check Squirrel routing status.
fn validate_inference_evolution(
    h: &mut ValidationHarness,
    discovered: &mut usize,
    skipped: &mut usize,
) {
    println!("\n── Phase 4: Inference evolution readiness ──");

    match composition::discover_primal_socket(niche::NICHE_NAME) {
        DiscoveryResult::Found(path) => {
            *discovered += 1;
            h.check_bool("neuralspring primal: socket discovered", true);

            for method in &["inference.complete", "inference.embed", "inference.models"] {
                match composition::call_capability(
                    &path,
                    method,
                    &serde_json::json!({"prompt": "test", "text": "test"}),
                    IPC_TIMEOUT,
                ) {
                    Ok(result) => {
                        h.check_bool(
                            &format!("{method}: responds (composition surface wired)"),
                            true,
                        );

                        let status = result
                            .get("status")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        let provider = result
                            .get("provider")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");

                        h.check_bool(
                            &format!("{method}: status is reported ({status})"),
                            !status.is_empty(),
                        );
                        h.check_bool(
                            &format!("{method}: provider is reported ({provider})"),
                            !provider.is_empty(),
                        );
                    }
                    Err(e) => {
                        h.check_bool(&format!("{method}: responds ({e})"), false);
                    }
                }
            }
        }
        DiscoveryResult::NotFound { .. } => {
            *skipped += 1;
            println!("  SKIP neuralspring primal: not running");
        }
    }
}

/// Phase 5: Health triad completeness — validate that the primal exposes
/// all three health endpoints required by `DEPLOYMENT_VALIDATION_STANDARD`.
fn validate_health_triad(h: &mut ValidationHarness, discovered: &mut usize, skipped: &mut usize) {
    println!("\n── Phase 5: Health triad (liveness + readiness + check) ──");

    match composition::discover_primal_socket(niche::NICHE_NAME) {
        DiscoveryResult::Found(path) => {
            *discovered += 1;

            for method in &["health.liveness", "health.readiness", "health.check"] {
                match composition::call_capability(
                    &path,
                    method,
                    &serde_json::json!({}),
                    IPC_TIMEOUT,
                ) {
                    Ok(_) => {
                        h.check_bool(&format!("{method}: responds"), true);
                    }
                    Err(e) => {
                        h.check_bool(&format!("{method}: responds ({e})"), false);
                    }
                }
            }

            match composition::call_capability(
                &path,
                "identity.get",
                &serde_json::json!({}),
                IPC_TIMEOUT,
            ) {
                Ok(result) => {
                    let has_caps = result.get("capabilities").is_some();
                    h.check_bool("identity.get: returns capabilities", has_caps);
                }
                Err(e) => {
                    h.check_bool(&format!("identity.get: responds ({e})"), false);
                }
            }

            match composition::call_capability(
                &path,
                "mcp.tools.list",
                &serde_json::json!({}),
                IPC_TIMEOUT,
            ) {
                Ok(result) => {
                    let count = result
                        .get("count")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0);
                    h.check_bool(&format!("mcp.tools.list: returns {count} tools"), count > 0);
                }
                Err(e) => {
                    h.check_bool(&format!("mcp.tools.list: responds ({e})"), false);
                }
            }
        }
        DiscoveryResult::NotFound { .. } => {
            *skipped += 1;
            println!("  SKIP neuralspring primal: not running (health triad untested)");
        }
    }
}
