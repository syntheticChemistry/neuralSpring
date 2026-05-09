// SPDX-License-Identifier: AGPL-3.0-or-later

//! neuralSpring UniBin — eukaryotic single-binary deployment.
//!
//! Consolidates certification, validation, serve, status, and version
//! into a single binary with clap subcommands.
//!
//! ## Subcommands
//!
//! - `certify` — run certification layers (L0-L3)
//! - `validate` — run validation scenarios (filter by track/tier/id)
//! - `serve` — start the JSON-RPC IPC server
//! - `status` — show capability discovery summary
//! - `version` — print version and exit

mod cli;

use clap::Parser;
use cli::{Cli, Commands};
use log::info;

use neural_spring::certification;
use neural_spring::validation::scenarios::{self, Tier, Track};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Certify { layer, bare } => cmd_certify(if bare { 0 } else { layer }),
        Commands::Validate {
            track,
            scenario,
            tier,
            list,
        } => cmd_validate(track, scenario, tier, list),
        Commands::Serve => cmd_serve(),
        Commands::Status => cmd_status(),
        Commands::Version => cmd_version(),
    }
}

fn cmd_certify(max_layer: u8) {
    let result = certification::certify(max_layer);
    std::process::exit(result.exit_code_skip_aware());
}

fn cmd_validate(
    track_filter: Option<String>,
    scenario_id: Option<String>,
    tier_filter: Option<String>,
    list: bool,
) {
    let registry = scenarios::build_registry();

    if list {
        println!("Available scenarios ({} total):", registry.len());
        for s in registry.all() {
            println!(
                "  {:<30} track={:<25} tier={:<5} checks={}",
                s.meta.id, s.meta.track, s.meta.tier, s.meta.check_count
            );
        }
        return;
    }

    let mut v = primalspring::validation::ValidationResult::new("neuralSpring validation");
    let mut ctx =
        primalspring::composition::CompositionContext::from_live_discovery_with_fallback();

    let track_match = track_filter.as_ref().map(|t| parse_track(t));
    let tier_match = tier_filter.as_ref().map(|t| parse_tier(t));

    let mut ran = 0;
    for s in registry.all() {
        if let Some(ref sid) = scenario_id {
            if s.meta.id != sid.as_str() {
                continue;
            }
        }
        if let Some(Some(ref track)) = track_match {
            if s.meta.track != *track {
                continue;
            }
        }
        if let Some(Some(ref tier)) = tier_match {
            if s.meta.tier != *tier {
                continue;
            }
        }

        info!("Running scenario: {} ({})", s.meta.id, s.meta.track);

        if let Some(run_rust) = s.run_rust {
            if tier_match.is_none()
                || matches!(tier_match, Some(Some(Tier::Rust)) | Some(Some(Tier::Both)))
            {
                run_rust(&mut v);
            }
        }
        if let Some(run_live) = s.run_live {
            if tier_match.is_none()
                || matches!(tier_match, Some(Some(Tier::Live)) | Some(Some(Tier::Both)))
            {
                run_live(&mut ctx, &mut v);
            }
        }
        ran += 1;
    }

    if ran == 0 {
        eprintln!("No matching scenarios found.");
        std::process::exit(1);
    }

    v.finish();
    std::process::exit(v.exit_code_skip_aware());
}

fn cmd_serve() {
    eprintln!("neuralspring serve: use `neuralspring` binary directly for IPC server mode.");
    eprintln!("The serve subcommand will absorb the primal server in the next evolution wave.");
    std::process::exit(0);
}

fn cmd_status() {
    let client = neural_spring::ipc::IpcMathClient::discover();
    let report = client.probe_all();
    let alive = report.alive_count();

    println!("neuralSpring v{VERSION} — NUCLEUS Status");
    println!("─────────────────────────────────");
    println!(
        "  barraCuda:  {}",
        status_icon(report.is_alive(neural_spring::ipc::PrimalSlot::Barracuda))
    );
    println!(
        "  toadStool:  {}",
        status_icon(report.is_alive(neural_spring::ipc::PrimalSlot::Toadstool))
    );
    println!(
        "  BearDog:    {}",
        status_icon(report.is_alive(neural_spring::ipc::PrimalSlot::Beardog))
    );
    println!(
        "  Squirrel:   {}",
        status_icon(report.is_alive(neural_spring::ipc::PrimalSlot::Squirrel))
    );
    println!(
        "  coralReef:  {}",
        status_icon(report.is_alive(neural_spring::ipc::PrimalSlot::Coralreef))
    );
    println!("─────────────────────────────────");
    println!("  Alive: {alive}/5");

    let registry = scenarios::build_registry();
    println!("  Scenarios: {}", registry.len());
    println!("  Certification layers: 0-{}", certification::MAX_LAYER);
}

fn cmd_version() {
    println!("neuralspring {VERSION}");
}

fn status_icon(alive: bool) -> &'static str {
    if alive { "ONLINE" } else { "offline" }
}

fn parse_track(s: &str) -> Option<Track> {
    match s {
        "spectral-analysis" => Some(Track::SpectralAnalysis),
        "nucleus-composition" => Some(Track::NucleusComposition),
        "inference-pipeline" => Some(Track::InferencePipeline),
        "gpu-parity" => Some(Track::GpuParity),
        "cross-spring" => Some(Track::CrossSpring),
        "provenance" => Some(Track::Provenance),
        _ => None,
    }
}

fn parse_tier(s: &str) -> Option<Tier> {
    match s {
        "rust" => Some(Tier::Rust),
        "live" => Some(Tier::Live),
        "both" => Some(Tier::Both),
        _ => None,
    }
}
