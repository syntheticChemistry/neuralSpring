// SPDX-License-Identifier: AGPL-3.0-only

//! Meta-validation binary: runs all `validate_*` binaries and aggregates results.
//!
//! Imitates the hotSpring `validate_all` pattern: each sub-binary runs
//! independently and reports exit 0 (pass) or 1 (fail). This binary
//! aggregates and reports the overall status.

use std::process::{self, Command};

const BINARIES: &[&str] = &[
    // neuralSpring-native validation
    "validate_surrogate",
    "validate_transformer",
    "validate_metrics",
    // BarraCUDA CPU primitive validation
    "validate_barracuda_stats",
    "validate_barracuda_linalg",
    "validate_barracuda_special",
    "validate_barracuda_optimize",
    "validate_barracuda_precision",
    "validate_barracuda_tensor",
    "validate_barracuda_tensor_f64",
    "validate_barracuda_quantized",
    "validate_barracuda_linalg_ext",
    "validate_barracuda_ml_inference",
];

fn main() {
    println!("=== neural-spring validate_all ===\n");

    let mut total_pass = 0_u32;
    let mut total_fail = 0_u32;

    for &name in BINARIES {
        print!("Running {name}... ");

        let result = Command::new("cargo")
            .args(["run", "--release", "--bin", name])
            .output();

        match result {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                if output.status.success() {
                    println!("PASS");
                    total_pass += 1;
                } else {
                    println!("FAIL");
                    total_fail += 1;
                }

                for line in stdout.lines() {
                    println!("    {line}");
                }
                for line in stderr.lines() {
                    if !line.contains("Compiling") && !line.contains("Finished") {
                        println!("    {line}");
                    }
                }
                println!();
            }
            Err(e) => {
                println!("ERROR: {e}");
                total_fail += 1;
            }
        }
    }

    let total = total_pass + total_fail;
    println!("=== validate_all: {total_pass}/{total} binaries PASS, {total_fail} FAIL ===");

    if total_fail > 0 {
        process::exit(1);
    }
}
