// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: GPU parity — CPU vs GPU math equivalence for all 6 pipeline stages.
//!
//! Tier 1 (Rust): structural checks that all 6 stages have GPU dispatch arms
//! and are tagged with GPU-capable substrates in the composition graph.
//! Tier 2 (Live): not applicable — GPU parity is validated locally via Dispatcher.

use super::registry::{Scenario, ScenarioMeta, Tier, Track};

use primalspring::validation::ValidationResult;

const GPU_STAGE_CAPABILITIES: [&str; 6] = [
    "science.eigensolve",
    "science.attention_anderson",
    "science.digester_anderson_coupling",
    "science.isomorphic_reservoir",
    "science.wdm_ensemble_qs",
    "science.introgression_nn",
];

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "gpu_parity",
        track: Track::GpuParity,
        tier: Tier::Rust,
        provenance_crate: "neuralSpring",
        provenance_date: "2026-05-17",
        description: "CPU vs GPU math parity for all 6 science pipeline stages",
        check_count: 12,
    },
    run_rust: Some(run_rust),
    run_live: None,
};

fn run_rust(v: &mut ValidationResult) {
    v.section("GPU Parity — Tier 1 (structural coverage)");

    let dispatch_src = include_str!("../../nucleus_pipeline/dispatch.rs");

    for cap in &GPU_STAGE_CAPABILITIES {
        let stage_name = cap.strip_prefix("science.").unwrap_or(cap);
        let gpu_fn = format!("stage_{stage_name}_gpu");
        let has_gpu_fn = dispatch_src.contains(&gpu_fn);
        let label = format!("gpu_parity:struct:gpu_fn_exists:{cap}");
        let detail = if has_gpu_fn {
            format!("{gpu_fn}() present")
        } else {
            format!("{gpu_fn}() MISSING")
        };
        v.check_bool(&label, has_gpu_fn, &detail);
    }

    let gpu_dispatch_section = dispatch_src
        .find("dispatch_capability_gpu")
        .map_or("", |start| &dispatch_src[start..]);

    for cap in &GPU_STAGE_CAPABILITIES {
        let routed = gpu_dispatch_section.contains(&format!("\"{cap}\""));
        let label = format!("gpu_parity:struct:gpu_dispatch_routed:{cap}");
        v.check_bool(
            &label,
            routed,
            if routed {
                "routed in dispatch_capability_gpu"
            } else {
                "NOT routed in dispatch_capability_gpu"
            },
        );
    }
}
