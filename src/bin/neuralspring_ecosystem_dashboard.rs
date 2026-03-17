// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ecosystem dashboard for neuralSpring — the "big picture" view.
//!
//! Renders all 16 scenario tracks to petalTongue, then streams live
//! updates for the Kokkos parity gauges and search pipeline statistics.
//!
//! This is the entry point for a scientist, clinician, or anyone with a
//! GPU and some curiosity who wants to see: What can this system do?
//! What's fast? What's being built? Where are the gaps?
//!
//! ## Usage
//!
//! ```text
//! # Start petalTongue
//! petaltongue ui
//!
//! # Render everything
//! cargo run --bin neuralspring_ecosystem_dashboard
//!
//! # Or dump scenarios to files (no petalTongue required)
//! MODE=dump cargo run --bin neuralspring_ecosystem_dashboard
//! ```
//!
//! ## Modes
//!
//! - **live** (default): Push to petalTongue via IPC, then stream updates
//! - **dump**: Write scenario JSON files to `sandbox/scenarios/`

use neural_spring::visualization::ipc_push::PetalTonguePushClient;
use neural_spring::visualization::stream::StreamSession;
use neural_spring::visualization::{
    full_study, industry_coverage_study, kokkos_parity_study, scenario_with_edges_json,
    search_study, streaming_io_study,
};

fn main() {
    let mode = std::env::var("MODE").unwrap_or_else(|_| "live".into());

    println!("neuralSpring Ecosystem Dashboard");
    println!("  mode: {mode}");
    println!();

    match mode.as_str() {
        "dump" => dump_scenarios(),
        _ => live_dashboard(),
    }
}

fn dump_scenarios() {
    let dir = std::path::Path::new("sandbox/scenarios");
    std::fs::create_dir_all(dir).ok();

    let scenarios: Vec<(&str, _)> = vec![
        ("neuralspring-full-study", full_study()),
        ("neuralspring-search", search_study()),
        ("neuralspring-streaming-io", streaming_io_study()),
        ("neuralspring-kokkos-parity", kokkos_parity_study()),
        ("neuralspring-industry-coverage", industry_coverage_study()),
    ];

    for (name, (scenario, edges)) in &scenarios {
        let json = scenario_with_edges_json(scenario, edges);
        let path = dir.join(format!("{name}.json"));
        match std::fs::write(&path, &json) {
            Ok(()) => println!("  wrote {}", path.display()),
            Err(err) => println!("  FAIL {}: {err}", path.display()),
        }
    }

    println!("\nDone. Load in petalTongue:");
    println!("  petaltongue ui --scenario sandbox/scenarios/neuralspring-full-study.json");
}

fn live_dashboard() {
    let client = match PetalTonguePushClient::discover() {
        Ok(c) => {
            println!("Discovered petalTongue via IPC");
            c
        }
        Err(err) => {
            println!("petalTongue not found ({err}) — running in headless mode");
            PetalTonguePushClient::headless()
        }
    };

    let (scenario, _edges) = full_study();
    let session = match StreamSession::start(
        client,
        "ecosystem-dashboard",
        "neuralSpring Ecosystem Dashboard — 16 Tracks",
        &scenario,
    ) {
        Ok(s) => s,
        Err(err) => {
            println!("Initial render failed ({err}), continuing headless");
            StreamSession::resume(PetalTonguePushClient::headless(), "ecosystem-dashboard")
        }
    };

    println!("Rendered full scenario (16 tracks) to petalTongue\n");

    println!("Tracks:");
    println!("  Spectral Analysis         | Training Metrics");
    println!("  Multi-Agent Coordination  | Shader Provenance");
    println!("  Protein Folding           | HMM Phylogenetics");
    println!("  Evolutionary Game Theory  | Warm Dense Matter");
    println!("  Blood Glucose Prediction  | Immunological Anderson");
    println!("  Meta-Population Dynamics  | Loss Landscape");
    println!("  Sequence Search Pipeline  | Streaming I/O Quality");
    println!("  Kokkos GPU Parity         | Industry Tool Coverage");
    println!();

    println!("Streaming live gauge updates...\n");

    let updates = [
        ("seed-count", 42.0, "Search: seed hits"),
        ("hit-count", 8.0, "Search: reported hits"),
        ("mean-overhead", 1.65, "Kokkos: mean overhead ratio"),
        ("max-overhead", 2.1, "Kokkos: worst-case overhead"),
        ("overall-coverage", 45.0, "Industry: overall coverage %"),
        ("read-count", 20.0, "Streaming: reads parsed"),
        ("seq-count", 6.0, "Streaming: FASTA sequences"),
        ("variant-count", 12.0, "Streaming: VCF variants"),
    ];

    for (binding_id, value, label) in &updates {
        match session.set_gauge(binding_id, *value) {
            Ok(()) => println!("  {label}: {value}"),
            Err(_) => println!("  {label}: {value} (headless)"),
        }
    }

    let stats = session.stats();
    println!("\nSession stats:");
    println!("  messages: {}", stats.messages_sent);
    println!("  bytes:    {}", stats.bytes_sent);
    println!("  errors:   {}", stats.errors);
    println!("  uptime:   {}ms", stats.uptime_ms);
}
