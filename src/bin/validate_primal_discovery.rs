// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Capability-based primal discovery validator.
//!
//! Validates that this spring can discover ecosystem primals by capability
//! (not by identity) following the `by_capability` routing rule from the
//! proto-nucleate graph.
//!
//! This validator exercises the 5-tier discovery order:
//! 1. `$BIOMEOS_ORCHESTRATOR_SOCKET` override
//! 2. `$XDG_RUNTIME_DIR/biomeos/{primal}*.sock`
//! 3. `/tmp/biomeos/{primal}*.sock`
//! 4. `$XDG_RUNTIME_DIR/{primal}/*.sock` (legacy)
//! 5. `/tmp/{primal}-*.sock` (legacy)
//!
//! ## Exit codes
//!
//! - 0: All discoverable primals respond to probes.
//! - 1: A discovered primal failed probing.
//! - 2: No primals found (honest skip).

use neural_spring::config;
use neural_spring::niche;
use neural_spring::primal_names;
use neural_spring::validation::ValidationHarness;
use neural_spring::validation::composition::{self, DiscoveryResult};
use std::time::Duration;

const IPC_TIMEOUT: Duration = Duration::from_secs(5);

struct DiscoveryTarget {
    name: &'static str,
    by_capability: &'static str,
}

const DISCOVERY_TARGETS: &[DiscoveryTarget] = &[
    DiscoveryTarget {
        name: primal_names::TOADSTOOL,
        by_capability: "compute.dispatch.submit",
    },
    DiscoveryTarget {
        name: primal_names::CORALREEF,
        by_capability: "shader.compile.wgsl",
    },
    DiscoveryTarget {
        name: primal_names::SQUIRREL,
        by_capability: "ai.query",
    },
    DiscoveryTarget {
        name: primal_names::BEARDOG,
        by_capability: "security",
    },
    DiscoveryTarget {
        name: primal_names::SONGBIRD,
        by_capability: "discovery",
    },
    DiscoveryTarget {
        name: primal_names::NESTGATE,
        by_capability: "storage.retrieve",
    },
    DiscoveryTarget {
        name: primal_names::BIOMEOS,
        by_capability: "graph.deploy",
    },
];

fn main() {
    let mut h = ValidationHarness::new("primal_discovery");
    let mut discovered = 0_usize;
    let mut skipped = 0_usize;

    println!("═══ Primal Discovery Validator ═══");

    // ── Phase 1: Self-knowledge checks ──

    println!("\n── Phase 1: Self-knowledge ──");

    h.check_bool("niche identity is non-empty", !niche::NICHE_NAME.is_empty());
    h.check_bool(
        "config domain is non-empty",
        !config::PRIMAL_DOMAIN.is_empty(),
    );
    h.check_bool(
        "capabilities list is non-empty",
        !niche::CAPABILITIES.is_empty(),
    );
    h.check_bool(
        "capabilities include health.liveness (discoverable)",
        niche::CAPABILITIES.contains(&"health.liveness"),
    );
    h.check_bool(
        "capabilities include capability.list (discoverable)",
        niche::CAPABILITIES.contains(&"capability.list"),
    );

    // ── Phase 2: by_capability discovery sweep ──

    println!(
        "\n── Phase 2: Discovery sweep ({} targets) ──",
        DISCOVERY_TARGETS.len()
    );

    for target in DISCOVERY_TARGETS {
        let prefix = format!("discover/{}", target.name);

        match composition::discover_primal_socket(target.name) {
            DiscoveryResult::Found(path) => {
                discovered += 1;
                h.check_bool(&format!("{prefix}: socket found"), true);

                match composition::probe_liveness(&path, IPC_TIMEOUT) {
                    Ok(()) => {
                        h.check_bool(&format!("{prefix}: health.liveness"), true);
                    }
                    Err(e) => {
                        h.check_bool(&format!("{prefix}: health.liveness ({e})"), false);
                        continue;
                    }
                }

                match composition::probe_capabilities(&path, IPC_TIMEOUT) {
                    Ok(caps) => {
                        let has = caps.iter().any(|c| c == target.by_capability);
                        h.check_bool(
                            &format!(
                                "{prefix}: by_capability '{}' ({} total)",
                                target.by_capability,
                                caps.len()
                            ),
                            has,
                        );
                    }
                    Err(e) => {
                        h.check_bool(&format!("{prefix}: capabilities.list ({e})"), false);
                    }
                }
            }
            DiscoveryResult::NotFound { searched, .. } => {
                skipped += 1;
                println!(
                    "  SKIP {}: not found (searched {} dirs, by_capability: {})",
                    target.name,
                    searched.len(),
                    target.by_capability,
                );
            }
        }
    }

    // ── Summary ──

    let failed = h.total_count() - h.passed_count();
    println!();
    println!(
        "Discovery: {discovered}/{} primals found, {skipped} skipped",
        DISCOVERY_TARGETS.len()
    );

    let exit = composition::exit_code_skip_aware(h.passed_count(), failed, skipped);

    h.emit_to_sink(&mut neural_spring::validation::StdoutSink);

    if exit == 2 {
        println!("SKIP: no primals available for discovery validation — honest skip (exit 2)");
    }

    std::process::exit(exit);
}
