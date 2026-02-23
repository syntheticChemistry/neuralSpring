// SPDX-License-Identifier: AGPL-3.0-or-later

//! Meta-validation binary: runs all `validate_*` binaries and aggregates results.
//!
//! Imitates the hotSpring `validate_all` pattern: each sub-binary runs
//! independently and reports exit 0 (pass) or 1 (fail). This binary
//! aggregates and reports the overall status.

use std::process::{self, Command};

const BINARIES: &[&str] = &[
    // neuralSpring-native validation (Phase 0)
    "validate_surrogate",
    "validate_transformer",
    "validate_metrics",
    "validate_lenet",
    "validate_transfer",
    "validate_isomorphic",
    "validate_lstm",
    "validate_quantized",
    "validate_sequence",
    // Phase 0++ paper reproduction validation
    "validate_counterdiabatic",
    "validate_modes",
    "validate_eco_dynamics",
    "validate_directed_evolution",
    "validate_hmm",
    "validate_game_theory",
    "validate_regulatory_network",
    "validate_signal_integration",
    "validate_swarm_robotics",
    "validate_sate_alignment",
    "validate_introgression",
    "validate_spectral_commutativity",
    "validate_anderson_localization",
    "validate_pangenome_selection",
    "validate_meta_population",
    "validate_pinn",
    "validate_deeponet",
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
    // BarraCUDA CPU ports (Phase 0++ modules → barracuda math)
    "validate_barracuda_spectral",
    "validate_barracuda_anderson",
    "validate_barracuda_regulatory",
    "validate_barracuda_signal",
    "validate_barracuda_hmm",
    "validate_barracuda_introgression",
    "validate_barracuda_counterdiabatic",
    "validate_barracuda_modes",
    "validate_barracuda_eco",
    "validate_barracuda_directed",
    "validate_barracuda_swarm",
    "validate_barracuda_sate",
    "validate_barracuda_game",
    "validate_barracuda_pangenome",
    "validate_barracuda_meta_pop",
    "validate_barracuda_pinn",
    "validate_barracuda_deeponet",
    // BarraCUDA extended validation
    "validate_barracuda_fft",
    "validate_barracuda_logsumexp",
    // GPU shader validation (Phase 3c)
    "validate_gpu_hmm_forward",
    "validate_gpu_batch_fitness",
    "validate_gpu_rk4",
    // Pure GPU pipeline + cross-dispatch (Phase 3d)
    "validate_gpu_stateful_pipeline",
    "validate_gpu_pure_workload",
    "validate_cross_dispatch",
    // GPU shader validation — expanded coverage (Phase 3c+)
    "validate_gpu_pangenome",   // Paper 024 — Jaccard distance
    "validate_gpu_meta_pop",    // Paper 025 — locus variance
    "validate_gpu_game_theory", // Paper 019 — spatial payoff stencil
    "validate_gpu_anderson",    // Papers 022-023 — batch IPR
    "validate_gpu_sate",        // Paper 017 — pairwise Hamming
    "validate_gpu_modes",       // Paper 012 — pairwise L2 distance
    "validate_gpu_directed",    // Paper 014 — multi-objective fitness
    "validate_gpu_swarm",       // Paper 015 — neural net forward
    "validate_gpu_signal",      // Paper 021 — Hill function gate
    // Cross-dispatch (Phase 3d+)
    "validate_cross_dispatch_genomics",
    "validate_cross_dispatch_extended",
    "validate_cross_dispatch_phase4e",
    // Pure GPU end-to-end pipelines (Phase 4b)
    "validate_gpu_pipeline_hmm",
    "validate_gpu_pipeline_ecology",
    "validate_gpu_pipeline_spectral",
    "validate_gpu_pipeline_genomics",
    "validate_gpu_pipeline_modes",
    "validate_gpu_pipeline_directed",
    "validate_gpu_pipeline_signal",
    // GPU PRNG (Phase 4c)
    "validate_gpu_prng",
    // ToadStool issue resolution (Phase 4d)
    "validate_eigh_accuracy",
    "validate_mha_gpu",
    // BarraCUDA CPU: Phase 0/0+ S-15-safe validators
    "validate_barracuda_sequence",
    "validate_barracuda_lenet",
    "validate_barracuda_lstm",
    "validate_barracuda_surrogate",
    "validate_barracuda_transfer",
    // BarraCUDA GPU tensor validation (Phase 5a)
    "validate_barracuda_gpu_spectral",
    "validate_barracuda_gpu_eco",
    "validate_barracuda_gpu_hmm",
    "validate_barracuda_gpu_fitness",
    "validate_barracuda_gpu_nn",
    "validate_barracuda_gpu_pairwise",
    "validate_barracuda_gpu_anderson",
    "validate_barracuda_gpu_modes",
    "validate_barracuda_gpu_directed",
    "validate_barracuda_gpu_swarm",
    "validate_barracuda_gpu_game",
    "validate_barracuda_gpu_introgression",
    "validate_barracuda_gpu_regulatory",
    "validate_barracuda_gpu_signal",
    "validate_barracuda_gpu_meta_pop",
    "validate_barracuda_gpu_transformer",
    // GPU Pipeline: expanded coverage (Phase 4b+)
    "validate_gpu_pipeline_fitness",
    "validate_gpu_pipeline_eco",
    "validate_gpu_pipeline_swarm",
    "validate_gpu_pipeline_sate",
    "validate_gpu_pipeline_regulatory",
    "validate_gpu_pipeline_meta_pop",
    // Cross-dispatch: expanded coverage (Phase 3d++)
    "validate_cross_dispatch_hmm",
    "validate_cross_dispatch_ode",
    // Upstream wrapper + parity validation (Phase 5c)
    "validate_barracuda_bio_ops",
    "validate_barracuda_hmm_f64",
    "validate_barracuda_spectral_theory",
    // Session 43: new WGSL shader validators
    "validate_gpu_logsumexp",
    "validate_gpu_stencil",
    "validate_gpu_rk45",
    "validate_gpu_wright_fisher",
    // Session 43: upstream wrapper validators
    "validate_gpu_gillespie",
    "validate_upstream_taxonomy",
    "validate_upstream_kmer",
    "validate_upstream_unifrac",
    "validate_barracuda_chi_squared",
    // Session 43: parity + dispatch validators
    "validate_cpu_gpu_parity",
    "validate_toadstool_dispatch",
    "validate_mixed_dispatch",
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
