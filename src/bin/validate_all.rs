// SPDX-License-Identifier: AGPL-3.0-or-later

//! Meta-validation binary: runs all `validate_*` binaries and aggregates results.
//!
//! Imitates the hotSpring `validate_all` pattern: each sub-binary runs
//! independently and reports exit 0 (pass) or 1 (fail). This binary
//! aggregates and reports the overall status.
//!
//! ## Provenance
//!
//! Meta-validation runner: aggregates all `validate_*` binaries.
//! No standalone validation.

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
    "validate_gpu_pangenome",    // Paper 024 — Jaccard distance
    "validate_gpu_meta_pop",     // Paper 025 — locus variance
    "validate_gpu_game_theory",  // Paper 019 — spatial payoff stencil
    "validate_gpu_anderson",     // Papers 022-023 — batch IPR
    "validate_gpu_sate",         // Paper 017 — pairwise Hamming
    "validate_gpu_modes",        // Paper 012 — pairwise L2 distance
    "validate_gpu_directed",     // Paper 014 — multi-objective fitness
    "validate_gpu_swarm",        // Paper 015 — neural net forward
    "validate_gpu_signal",       // Paper 021 — Hill function gate (polyfill)
    "validate_hillgate_f64_fix", // S-17 — HillGate f64 pow() fix proof
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
    // BarraCUDA issue resolution (Phase 4d)
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
    // Session 44: stochastic GPU pipelines + gap closure
    "validate_gpu_pipeline_wright_fisher",
    "validate_gpu_pipeline_gillespie",
    "validate_barracuda_gpu_lenet",
    "validate_barracuda_transformer",
    // Session 45: GPU promotion — all CPU math → GPU dispatch
    "validate_gpu_promotion",
    // Session 46: Phase B — HMM backward/Viterbi, meta-pop, game theory, Hill GPU
    "validate_gpu_phase_b",
    // Session 66: Phase C — HMM chain, FST, introgression, inter-pop AF variance GPU
    "validate_gpu_phase_c",
    // Session 67: CPU math parity — Rust vs Python/NumPy cross-language validation
    "validate_cpu_math_parity",
    // Session 50: baseCamp — Biophysical AI Interpretability
    "validate_weight_spectral",
    // S99: nS-01 Paper A real pretrained weight spectral analysis (safetensors)
    "validate_weight_spectral_real",
    "validate_information_flow",
    "validate_loss_landscape",
    "validate_neural_pgm",
    "validate_agent_coordination",
    // Session 54: baseCamp pure GPU workload validation
    "validate_basecamp_gpu",
    "validate_basecamp_gpu_pure",
    // Session 55: BarraCUDA CPU vs GPU dispatch + metalForge mixed hardware
    "validate_compute_dispatch",
    "validate_mixed_hardware",
    // Session 56: Dispatcher baseCamp, parity proof, metalForge PCIe
    "validate_basecamp_dispatch",
    "validate_barracuda_parity",
    "validate_metalforge_pcie",
    // Session 58: cross-spring evolution benchmark + GpuDriverProfile
    "validate_cross_spring_evolution",
    // Session 74: pure GPU all-domains workload validation
    "validate_gpu_pure_workload_all",
    // Session 74: metalForge cross-system dispatch (GPU→NPU→CPU)
    "validate_cross_system_dispatch",
    // WDM Surrogate Extensions (nW-01..nW-05)
    "validate_wdm_eos",
    "validate_barracuda_wdm_eos",
    "validate_barracuda_wdm_transport",
    "validate_barracuda_wdm_esn",
    "validate_barracuda_wdm_sqw",
    "validate_wdm_transport",
    "validate_wdm_transfer",
    "validate_wdm_sqw",
    "validate_wdm_esn",
    // coralForge — Evoformer primitives (nF-01 Phase B)
    // Exp-050: Training trajectory spectral analysis (Paper A)
    "validate_training_trajectory",
    // Exp-052: Hessian eigenanalysis at trained minima (Paper D)
    "validate_hessian_eigenanalysis",
    // Exp-053: Anderson multi-agent coordination (Paper C)
    "validate_anderson_multiagent",
    // GPU tier: Exp-050/052/053 publication experiments
    "validate_barracuda_training_trajectory",
    "validate_barracuda_hessian_eigen",
    "validate_barracuda_anderson_multiagent",
    // Pure GPU pipeline + metalForge: publication experiments
    "validate_publication_gpu_pipeline",
    // Publication mixed-hardware (NUCLEUS atomics)
    "validate_publication_mixed_hardware",
    // NUCLEUS compute dispatch (Tower → Node → Nest)
    "validate_nucleus_compute_dispatch",
    // BarraCUDA spectral absorption readiness
    "validate_toadstool_spectral_absorption",
    // Phase 4 WGSL shader validation (direct dispatch)
    "validate_gpu_shader_phase4",
    // ToadStool streaming spectral pipeline
    "validate_streaming_spectral_pipeline",
    // coralForge — Evoformer primitives (nF-01 Phase B)
    "validate_coral_forge",
    "validate_coral_forge_gpu",
    "validate_coral_forge_gpu_pipeline",
    // nF-02 AlphaFold2 full Evoformer block + BarraCUDA CPU
    "validate_alphafold2_evoformer",
    // nF-03 AlphaFold3 diffusion + Pairformer + confidence heads
    "validate_alphafold3_diffusion",
    "validate_alphafold3_pairformer",
    "validate_alphafold3_confidence",
    "validate_barracuda_alphafold3_confidence_gpu",
    "validate_barracuda_alphafold2",
    // Phase B gap closure: ODE batch GPU + FST + introgression HMM chain
    "validate_gpu_ode_batch",
    "validate_gpu_phase_b_extended",
    // Modern rewire + cross-spring provenance benchmark
    "bench_modern_rewire",
    // BarraCUDA CPU parity + performance (Python/NumPy vs pure Rust)
    "validate_barracuda_cpu_bench",
    // BarraCUDA dispatch parity (CPU ↔ GPU same math)
    "validate_barracuda_dispatch_parity",
    // Mixed-hardware dispatch (NPU/GPU/CPU substrate routing + PCIe bridge)
    "validate_mixed_hardware_dispatch",
    // Portability tier benchmark (CPU → GPU parity + ToadStool streaming proof)
    "bench_portability_tiers",
    // S91: Modern cross-spring evolution — BarraCUDA S68 universal precision,
    // provenance tracking across all 5 springs, bio+stats+precision validation
    "validate_modern_cross_spring",
    // S96: WDM + AlphaFold3 dispatch parity (CPU ↔ GPU) + metalForge routing + NUCLEUS
    "validate_wdm_alphafold_dispatch",
    // S97: Pure GPU pipeline for WDM + coralForge (Tensor API, scalar-only readback)
    "validate_gpu_pure_wdm_coral",
    // S97: WDM nW-04 transfer learning GPU Tensor parity
    "validate_barracuda_wdm_transfer_gpu",
    // S97b: coralForge Evoformer dispatch + metalForge mixed (nF-01/nF-02)
    "validate_coral_forge_dispatch",
    // S97c: nF-03 AlphaFold3 BarraCUDA CPU tier (closes bC 2/3 → 3/3)
    "validate_barracuda_alphafold3",
    // S97c: WDM+coralForge CPU↔GPU domain parity (BarraCUDA portability proof)
    "validate_wdm_coral_parity",
    // S97c: metalForge mixed-hardware WDM+coralForge (NUCLEUS atomics + PCIe bypass)
    "validate_metalforge_wdm_coral",
    // S97d: BarraCUDA S70+++ cross-spring evolution (matmul_ref, SimpleMlp, stats::evolution,
    // stats::jackknife, stats::hydrology, diversity::chao1_classic — provenance tracking)
    "validate_toadstool_s70_evolution",
    // S101: BarraCUDA S71 GPU stats parity (KimuraGpu, HistogramGpu, upstream shader bugs)
    "validate_toadstool_s71_gpu_stats",
    // S102: Nautilus Shell cross-spring bridge (hotSpring brain arch → neuralSpring spectral)
    "validate_nautilus_bridge",
    // S98: nF-03 AlphaFold3 diffusion GPU Tensor tier (forward/DDPM/DDIM/SE(3)/FFN)
    "validate_alphafold3_diffusion_gpu",
    // S98: nF-03 AlphaFold3 Pairformer GPU Tensor tier (conditioning/TriMul/TriAttn/FFN/block)
    "validate_alphafold3_pairformer_gpu",
    // S105: MultiHeadEsn + NPU export + typed JSON deserialization
    "validate_multi_head_esn",
    // S105: TrainingMonitor brain-inspired FSM + DriftMonitor
    "validate_training_monitor",
    // S105: baseCamp Paper 12 — immunological Anderson (Py 20/20, Rs 53/53)
    "validate_immunological_anderson",
    // S105: baseCamp GPU promotions (weight_spectral matmul, loss Hessian, PGM HMM, agent L2)
    "validate_barracuda_basecamp",
    // S107: baseCamp Paper 12 extended — Gonzales deep modeling, 3D lattice, Fajgenbaum MATRIX
    "validate_immunological_anderson_extended",
    // S108: ToadStool S79 cross-spring provenance rewire
    "validate_toadstool_s79_rewire",
    // S112: ToadStool S86 rewire — nautilus absorbed into barracuda::nautilus, DriftMonitor API
    "validate_toadstool_s86_rewire",
    // S115: ToadStool ComputeDispatch evolution (Dispatcher↔barracuda::dispatch bridge)
    "validate_compute_dispatch_evolution",
    // S116: ToadStool S87 sync — deep debt, CPU ungating, error evolution, gpu_helpers refactor
    "validate_toadstool_s87_sync",
    // S116: ToadStool S93 / barraCuda standalone extraction validation
    "validate_toadstool_s93_barracuda_extraction",
    // S117: Cross-spring shader evolution — provenance tracking, all springs → BarraCUDA convergence
    "validate_cross_spring_shader_evolution",
    // S115: NUCLEUS PCIe bypass + mixed-pipeline (Tower→Node→Nest + GPU↔NPU↔CPU)
    "validate_nucleus_pcie_mixed_pipeline",
    // S121: SimpleMlp rewire (WDM surrogates) + HMM Viterbi f64 ComputeDispatch
    "validate_barracuda_s121_rewire",
    // S123: Paper 026 (Chuna LSTM blood glucose prediction) — Py→Rs→bC→gT
    "validate_glucose_prediction",
    "validate_barracuda_glucose_prediction",
    // S126: ToadStool S94b + wgpu 28 + BarraCUDA v0.3.3 fused op absorption
    "validate_toadstool_s94b_wgpu28",
    // biomeOS graph coordination (DAG pipeline, topo sort, execution tracking)
    "validate_biomeos_graph",
    // petalTongue visualization (scenarios, streaming, mock IPC roundtrip)
    "validate_petaltongue_scenarios",
];

/// Feature-gated validation binaries: `(name, feature)`.
///
/// These require `cargo run --release --features <feature> --bin <name>`.
const FEATURE_BINARIES: &[(&str, &str)] = &[
    // NUCLEUS Tower integration (JSON-RPC + folding + discovery)
    ("validate_nucleus_tower", "primal"),
    // biomeOS spectral pipeline (primal RPC: health, IPR, disorder, spectral)
    ("validate_biomeos_spectral", "primal"),
];

fn run_binary(name: &str, features: Option<&str>) -> (bool, String, String) {
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--release"]);
    if let Some(feat) = features {
        cmd.args(["--features", feat]);
    }
    cmd.args(["--bin", name]);

    match cmd.output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            (output.status.success(), stdout, stderr)
        }
        Err(e) => (false, String::new(), format!("ERROR: {e}")),
    }
}

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

    // Feature-gated binaries
    for &(name, feature) in FEATURE_BINARIES {
        print!("Running {name} (--features {feature})... ");

        let (success, stdout, stderr) = run_binary(name, Some(feature));
        if success {
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

    let total = total_pass + total_fail;
    println!("=== validate_all: {total_pass}/{total} binaries PASS, {total_fail} FAIL ===");

    if total_fail > 0 {
        process::exit(1);
    }
}
