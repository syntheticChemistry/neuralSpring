// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Science composition parity validator — Rust→IPC (Tier 3).
//!
//! This is the composition evolution of validation:
//!
//! ```text
//! Tier 1: Python baseline → Rust validation (cross-language parity)
//! Tier 2: Rust CPU → GPU validation (cross-substrate parity)
//! Tier 3: Rust direct → IPC round-trip (composition parity) ← THIS BINARY
//! ```
//!
//! For each science capability exposed on the neuralSpring primal's JSON-RPC
//! surface, this binary:
//! 1. Computes the expected result via direct Rust library calls (same seed,
//!    same parameters — deterministic).
//! 2. Calls the same capability via JSON-RPC over a Unix socket.
//! 3. Compares the IPC response to the Rust baseline within documented
//!    tolerances (should be exact or within `SPECIAL_FUNCTION_F64`).
//!
//! If the primal is not running, exit 2 (honest skip). If any parity check
//! fails, exit 1. If all checks pass, exit 0.
//!
//! ## Provenance
//!
//! Rust baselines: computed deterministically in `validation::composition::science_baselines()`
//! IPC surface: `neuralspring_primal` JSON-RPC (`science.*` methods)

use neural_spring::niche;
use neural_spring::validation::ValidationHarness;
use neural_spring::validation::composition::{
    self, DiscoveryResult, ScienceBaseline, call_capability, discover_primal_socket,
    science_baselines,
};
use std::path::Path;
use std::time::Duration;

const IPC_TIMEOUT: Duration = Duration::from_secs(10);

fn main() {
    let mut h = ValidationHarness::new("science_composition_parity");

    println!("═══ Science Composition Parity Validator (Tier 3) ═══");
    println!("Validates: Rust direct == IPC round-trip for science.*\n");

    let socket = match discover_primal_socket(niche::NICHE_NAME) {
        DiscoveryResult::Found(path) => {
            println!("  neuralspring primal: discovered at {}", path.display());
            h.check_bool("neuralspring primal: socket discovered", true);
            path
        }
        DiscoveryResult::NotFound { searched, .. } => {
            println!("  SKIP: neuralspring primal not running");
            println!("  Searched: {searched:?}");
            println!("\nSKIP: no primal available — honest skip (exit 2)");
            std::process::exit(2);
        }
    };

    if composition::probe_liveness(&socket, IPC_TIMEOUT).is_err() {
        h.check_bool("neuralspring primal: health.liveness", false);
        println!("FAIL: primal socket exists but liveness probe failed");
        std::process::exit(1);
    }
    h.check_bool("neuralspring primal: health.liveness", true);

    let baselines = science_baselines();
    println!(
        "\n── Science parity checks ({} baselines) ──\n",
        baselines.len()
    );

    for baseline in &baselines {
        validate_science_parity(&mut h, &socket, baseline);
    }

    let failed = h.total_count() - h.passed_count();
    let exit = composition::exit_code_skip_aware(h.passed_count(), failed, 0);

    h.emit_to_sink(&mut neural_spring::validation::StdoutSink);
    std::process::exit(exit);
}

fn validate_science_parity(h: &mut ValidationHarness, socket: &Path, baseline: &ScienceBaseline) {
    println!("  {}: calling via IPC...", baseline.method);

    let result = match call_capability(socket, baseline.method, &baseline.params, IPC_TIMEOUT) {
        Ok(r) => r,
        Err(e) => {
            h.check_bool(
                &format!("{}: IPC call succeeded ({e})", baseline.method),
                false,
            );
            return;
        }
    };
    h.check_bool(&format!("{}: IPC call succeeded", baseline.method), true);

    if baseline.method == "science.disorder_sweep" {
        validate_disorder_sweep_parity(h, baseline, &result);
        return;
    }

    for &(key, expected) in &baseline.expected {
        let observed = result
            .get(key)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(f64::NAN);

        h.check_abs(
            &format!(
                "{}.{key}: Rust={expected:.8e} IPC={observed:.8e}",
                baseline.method
            ),
            observed,
            expected,
            baseline.tolerance,
        );
    }
}

fn validate_disorder_sweep_parity(
    h: &mut ValidationHarness,
    baseline: &ScienceBaseline,
    result: &serde_json::Value,
) {
    let ipr_values = result
        .get("ipr_values")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_f64)
                .collect::<Vec<_>>()
        });

    let Some(iprs) = ipr_values else {
        h.check_bool("science.disorder_sweep: ipr_values present", false);
        return;
    };
    h.check_bool("science.disorder_sweep: ipr_values present", true);

    for (i, &(key, expected)) in baseline.expected.iter().enumerate() {
        let observed = iprs.get(i).copied().unwrap_or(f64::NAN);
        h.check_abs(
            &format!("science.disorder_sweep.{key}: Rust={expected:.8e} IPC={observed:.8e}"),
            observed,
            expected,
            baseline.tolerance,
        );
    }
}
