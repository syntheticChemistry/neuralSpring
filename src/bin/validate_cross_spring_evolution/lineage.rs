// SPDX-License-Identifier: AGPL-3.0-or-later

//! Long-form cross-spring evolution lineage report (stdout).

pub fn report_cross_spring_lineage() {
    println!("\n=== Cross-Spring Evolution Lineage ===\n");
    println!("hotSpring \u{2192} BarraCUDA precision layer:");
    println!("  \u{2022} df64_core.wgsl (double-float f32-pair emulation)");
    println!("  \u{2022} pow_f64 polyfill (transcendental workaround \u{2192} S-17 RESOLVED)");
    println!("  \u{2022} Fp64Strategy (Native/Hybrid detection)");
    println!("  \u{2022} DeviceCapabilities (hardware-adaptive dispatch)");
    println!("  \u{2022} Taylor-series sin/cos (7-term + Cody-Waite)");
    println!("  \u{2022} Lanczos eigensolver (lattice QCD heritage)");
    println!();
    println!("wetSpring \u{2192} BarraCUDA bio+spectral layer:");
    println!("  \u{2022} HMM forward/backward (phylogenetics)");
    println!(
        "  \u{2022} 5 ODE bio systems (Capacitor, Cooperation, MultiSignal, Bistable, PhageDefense)"
    );
    println!("  \u{2022} NMF (non-negative matrix factorization)");
    println!("  \u{2022} Anderson localization (3d_correlated, sweep_averaged, find_w_c)");
    println!("  \u{2022} Ridge regression (ESN readout)");
    println!(
        "  \u{2022} fst_variance_decomposition (population genetics F-statistics)  [S72 rewire]"
    );
    println!();
    println!("neuralSpring \u{2192} BarraCUDA validation+ops layer:");
    println!("  \u{2022} ValidationHarness + exit_no_gpu + require! macro");
    println!("  \u{2022} batch_fitness_eval, pairwise_l2, pairwise_hamming/jaccard");
    println!("  \u{2022} spatial_payoff, hill_gate, multi_obj_fitness");
    println!("  \u{2022} eigh_householder_qr, batch_ipr, swarm_nn");
    println!("  \u{2022} 4-tier matmul KernelRouter");
    println!("  \u{2022} empirical_spectral_density, marchenko_pastur_bounds (S54)");
    println!("  \u{2022} effective_rank (S54), gelu_dispatch + hmm_forward_dispatch (S52)");
    println!();
    println!("S72 cross-spring rewiring:");
    println!(
        "  \u{2022} argmax_dim(axis) \u{2192} Viterbi psi extraction (was CPU loop, now upstream)"
    );
    println!(
        "  \u{2022} softmax_dim(axis) \u{2192} Dispatcher::softmax_row_wise (was manual per-row)"
    );
    println!("  \u{2022} fst_variance_decomposition \u{2192} fst_single_locus + pairwise_fst_full");
    println!(
        "  \u{2022} All 17 shortcomings RESOLVED upstream (S-14/15/16 at a4996b34, S-17 at c82c23d1)"
    );
    println!();
    println!("airSpring \u{2192} BarraCUDA stats+regression layer:");
    println!(
        "  \u{2022} mae, rmse, r_squared, nash_sutcliffe, index_of_agreement [S64\u{2013}S66]"
    );
    println!("  \u{2022} fit_linear, fit_quadratic, fit_exponential, fit_logarithmic [S66]");
    println!("  \u{2022} hydrology (hargreaves, soil_water_balance) [S66]");
    println!();
    println!("S78 cross-spring rewiring (neuralSpring \u{2192} BarraCUDA via ToadStool S66):");
    println!("  \u{2022} metrics::mae \u{2192} barracuda::stats::mae (airSpring origin)");
    println!(
        "  \u{2022} primitives::shannon_entropy \u{2192} barracuda::stats::shannon_from_frequencies (wetSpring origin)"
    );
    println!(
        "  \u{2022} primitives::hill_activation/repression \u{2192} barracuda::stats::hill (wetSpring+hotSpring origin)"
    );
    println!(
        "  \u{2022} modes::l2_distance \u{2192} barracuda::dispatch::l2_distance_dispatch (neuralSpring origin)"
    );
    println!(
        "  \u{2022} modes::complexity_metric \u{2192} barracuda::stats::fit_linear (airSpring origin)"
    );
    println!(
        "  \u{2022} 9 metalForge shaders aligned to compile_shader_df64 convention (hotSpring origin)"
    );
    println!();
    println!("All springs \u{2192} ToadStool (GPU sovereign pipeline):");
    println!("  \u{2022} 633+ WGSL shaders (cross-spring evolved, S66 Wave 5)");
    println!("  \u{2022} domain_ops dispatch \u{2014} 9 methods rewired (S58: 7, S59: +2)");
    println!("  \u{2022} stats/linalg \u{2014} 3 library functions rewired (S59)");
    println!(
        "  \u{2022} S72 \u{2014} 4 new rewires (softmax_row_wise, fst_single_locus, fst_full, argmax_dim)"
    );
    println!("  \u{2022} S76 \u{2014} 2 rewires (pearson_correlation)");
    println!("  \u{2022} S78 \u{2014} 6 rewires (mae, shannon, hill x2, l2_distance, fit_linear)");
    println!(
        "  \u{2022} S91 \u{2014} 2 rewires (primal matmul_2d/3d \u{2192} matmul_dispatch, compile_shader_universal)"
    );
    println!("  \u{2022} Total: 44 functions + 6 shader sources rewired");
    println!("  \u{2022} DeviceCapabilities (this benchmark validates detection)");
}
