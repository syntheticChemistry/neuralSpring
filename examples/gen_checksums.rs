// SPDX-License-Identifier: AGPL-3.0-or-later

//! Generate the CHECKSUMS manifest for `neuralspring_guidestone` P3 verification.
//!
//! Usage: `cargo run --example gen_checksums --features guidestone > validation/CHECKSUMS`

fn main() {
    let files: &[&str] = &[
        "src/bin/neuralspring_guidestone.rs",
        "src/tolerances/mod.rs",
        "src/tolerances/registry.rs",
        "src/tolerances/gpu.rs",
        "src/tolerances/training.rs",
        "src/tolerances/evolutionary.rs",
        "src/provenance/mod.rs",
        "src/provenance/experiments.rs",
        "src/provenance/references.rs",
        "src/validation/mod.rs",
        "src/validation/composition.rs",
        "src/rng.rs",
        "config/capability_registry.toml",
        "control/tolerances.py",
        "Cargo.toml",
    ];

    let root = std::path::Path::new(".");
    let manifest = primalspring::checksums::generate_manifest(root, files);
    println!("# neuralSpring guideStone CHECKSUMS — BLAKE3");
    println!("# Generated: {}", chrono_free_date());
    println!("# Files: {}", files.len());
    println!("#");
    println!("# Verify: primalspring::checksums::verify_manifest()");
    println!("{manifest}");
}

fn chrono_free_date() -> String {
    std::process::Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map_or_else(|| "unknown".to_owned(), |s| s.trim().to_owned())
}
