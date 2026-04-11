// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Inference capability chain composition validator.
//!
//! Validates the `inference.*` capability chain end-to-end:
//! neuralSpring → Squirrel → provider (Ollama fallback).
//!
//! This is the composition-layer validation for neuralSpring's primary
//! ecosystem contribution: AI inference for all springs. The evolution
//! path is:
//!
//! 1. Squirrel discovers neuralSpring as inference provider
//! 2. neuralSpring routes to local WGSL ML (when ready) or Ollama fallback
//! 3. Any spring with Squirrel in its composition gains inference.*
//!
//! ## Exit codes
//!
//! - 0: Inference chain validated (all checks pass).
//! - 1: Chain broken (one or more checks failed).
//! - 2: No inference providers available (honest skip).

use neural_spring::niche;
use neural_spring::primal_names;
use neural_spring::validation::ValidationHarness;
use neural_spring::validation::composition::{self, DiscoveryResult};
use std::time::Duration;

const IPC_TIMEOUT: Duration = Duration::from_secs(10);

fn main() {
    let mut h = ValidationHarness::new("inference_composition");
    let mut skipped = 0_usize;

    println!("═══ Inference Composition Validator ═══");

    validate_capability_registration(&mut h);
    validate_squirrel_chain(&mut h, &mut skipped);

    finish_with_skip_aware(&h, skipped);
}

fn validate_capability_registration(h: &mut ValidationHarness) {
    println!("\n── Phase 1: Capability registration ──");

    h.check_bool(
        "niche advertises inference.complete",
        niche::CAPABILITIES.contains(&"inference.complete"),
    );
    h.check_bool(
        "niche advertises inference.embed",
        niche::CAPABILITIES.contains(&"inference.embed"),
    );
    h.check_bool(
        "niche advertises inference.models",
        niche::CAPABILITIES.contains(&"inference.models"),
    );

    let deps = niche::operation_dependencies();
    h.check_bool(
        "inference.complete has operation dependencies",
        deps.get("inference.complete").is_some(),
    );
    h.check_bool(
        "inference.embed has operation dependencies",
        deps.get("inference.embed").is_some(),
    );
    h.check_bool(
        "inference.models has operation dependencies",
        deps.get("inference.models").is_some(),
    );

    let costs = niche::cost_estimates();
    h.check_bool(
        "inference.complete has cost estimate",
        costs.get("inference.complete").is_some(),
    );
}

fn validate_squirrel_chain(h: &mut ValidationHarness, skipped: &mut usize) {
    println!("\n── Phase 2: Squirrel discovery ──");

    match composition::discover_primal_socket(primal_names::SQUIRREL) {
        DiscoveryResult::Found(path) => {
            h.check_bool("squirrel: socket discovered", true);

            match composition::probe_liveness(&path, IPC_TIMEOUT) {
                Ok(()) => {
                    h.check_bool("squirrel: health.liveness", true);
                }
                Err(e) => {
                    h.check_bool(&format!("squirrel: health.liveness ({e})"), false);
                    return;
                }
            }

            match composition::probe_capabilities(&path, IPC_TIMEOUT) {
                Ok(caps) => {
                    h.check_bool(
                        "squirrel: advertises ai.query",
                        caps.iter().any(|c| c == "ai.query"),
                    );
                }
                Err(e) => {
                    h.check_bool(&format!("squirrel: capabilities.list ({e})"), false);
                }
            }

            validate_inference_models_probe(h, skipped);
        }
        DiscoveryResult::NotFound { .. } => {
            *skipped += 1;
            println!("  SKIP squirrel: not running (inference chain unavailable)");
        }
    }
}

fn validate_inference_models_probe(h: &mut ValidationHarness, skipped: &mut usize) {
    println!("\n── Phase 3: inference.models probe ──");

    match composition::discover_primal_socket(niche::NICHE_NAME) {
        DiscoveryResult::Found(ns_path) => {
            h.check_bool("neuralspring: socket discovered", true);

            match composition::call_capability(
                &ns_path,
                "inference.models",
                &serde_json::json!({}),
                IPC_TIMEOUT,
            ) {
                Ok(result) => {
                    let has_models = result.get("models").and_then(|v| v.as_array()).is_some();
                    h.check_bool("inference.models returns model list", has_models);

                    let status = result
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    h.check_bool(
                        &format!("inference.models status: {status}"),
                        !status.is_empty(),
                    );
                }
                Err(e) => {
                    h.check_bool(&format!("inference.models call ({e})"), false);
                }
            }
        }
        DiscoveryResult::NotFound { .. } => {
            *skipped += 1;
            println!("  SKIP neuralspring primal: not running");
        }
    }
}

fn finish_with_skip_aware(h: &ValidationHarness, skipped: usize) -> ! {
    let failed = h.total_count() - h.passed_count();
    let exit = composition::exit_code_skip_aware(h.passed_count(), failed, skipped);

    h.emit_to_sink(&mut neural_spring::validation::StdoutSink);

    if exit == 2 {
        println!("SKIP: no inference providers available — honest skip (exit 2)");
    }

    std::process::exit(exit);
}
