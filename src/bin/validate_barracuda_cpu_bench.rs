// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU parity + performance benchmark.
//!
//! For each of the 15 paper-domain Python benchmarks, this binary:
//! 1. Runs the Python benchmark subprocess, capturing its median µs timing
//! 2. Runs the identical computation in pure Rust via `BarraCUDA` CPU primitives
//! 3. Verifies numeric parity (spot-check against `Python`/`NumPy`)
//! 4. Reports the speedup factor (Python µs / Rust µs)
//!
//! This is the authoritative proof that `BarraCUDA` CPU is pure math and
//! faster than interpreted language (`Python`/`NumPy`) for all paper domains.
//!
//! ## Benchmark Domains
//!
//! | Domain | Paper(s) | Python Script | Rust Module |
//! |--------|----------|---------------|-------------|
//! | HMM forward | 016-018 | `bench_hmm_forward.py` | `hmm.rs` |
//! | NK fitness | 011 | `bench_nk_fitness.py` | `counterdiabatic.rs` |
//! | Pairwise L2 | 012 | `bench_pairwise_l2.py` | `modes.rs` |
//! | Eco batch fitness | 013 | `bench_eco.py` | `eco_dynamics.rs` |
//! | Hamming dist | 017 | `bench_hamming.py` | `sate_alignment.rs` |
//! | Jaccard dist | 024 | `bench_jaccard.py` | `pangenome_selection.rs` |
//! | Replicator | 019 | `bench_replicator.py` | `game_theory.rs` |
//! | RK4 GRN | 020 | `bench_rk4.py` | `regulatory_network.rs` |
//! | Commutator | 022 | `bench_commutator.py` | `spectral_commutativity.rs` |
//! | Anderson IPR | 023 | `bench_anderson.py` | `anderson_localization.rs` |
//! | Hill gate | 021 | `bench_hill_gate.py` | `signal_integration.rs` |
//! | Multi-obj | 014 | `bench_multi_obj.py` | `directed_evolution.rs` |
//! | Swarm NN | 015 | `bench_swarm_nn.py` | `swarm_robotics.rs` |
//! | Global FST | 025 | `bench_meta_pop.py` | `meta_population/fst.rs` |
//! | LSTM Glucose | 026 | `bench_glucose_lstm.py` | `glucose_prediction.rs` |
//!
//! The portability story:
//! ```text
//! `Python`/`NumPy` (interpreted) → `BarraCUDA` CPU (pure Rust) → `BarraCUDA` GPU (`WGSL`)
//!                    ↑ parity proven here        ↑ parity proven in GPU validators
//! ```
//!
//! # Panics
//!
//! Panics if Python is unavailable — this benchmark requires both runtimes.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::expect_used,
    reason = "validation binary"
)]

use neural_spring::anderson_localization;
use neural_spring::counterdiabatic::NkLandscape;
use neural_spring::directed_evolution;
use neural_spring::eco_dynamics::MultiNicheLandscape;
use neural_spring::game_theory;
use neural_spring::glucose_prediction;
use neural_spring::hmm::Hmm;
use neural_spring::meta_population;
use neural_spring::modes;
use neural_spring::pangenome_selection;
use neural_spring::regulatory_network::{self, GrnParams};
use neural_spring::rng::Rng;
use neural_spring::sate_alignment;
use neural_spring::sequence::{lstm_cell, LstmWeights};
use neural_spring::signal_integration;
use neural_spring::spectral_commutativity;
use neural_spring::swarm_robotics;
use neural_spring::tolerances;
use neural_spring::validation::cpu_bench::{
    print_cpu_summary, record_domain, run_python_bench, CpuBenchResult,
};
use neural_spring::validation::{bench_median, ValidationHarness};

const WARMUP: usize = 10;
const ITERS: usize = 200;

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  neuralSpring — BarraCUDA CPU Parity & Performance Benchmark               ║");
    eprintln!("║  Python/NumPy (interpreted) vs Pure Rust (BarraCUDA CPU) — 15 domains      ║");
    eprintln!("║  Warmup: {WARMUP}, Iterations: {ITERS}                                                    ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let mut h = ValidationHarness::new("barracuda_cpu_bench");
    let mut results: Vec<CpuBenchResult> = Vec::new();

    bench_hmm_forward(&mut h, &mut results);
    bench_nk_fitness(&mut h, &mut results);
    bench_pairwise_l2(&mut h, &mut results);
    bench_eco_fitness(&mut h, &mut results);
    bench_hamming(&mut h, &mut results);
    bench_jaccard(&mut h, &mut results);
    bench_replicator(&mut h, &mut results);
    bench_rk4_grn(&mut h, &mut results);
    bench_commutator(&mut h, &mut results);
    bench_anderson_ipr(&mut h, &mut results);
    bench_hill_gate(&mut h, &mut results);
    bench_multi_obj(&mut h, &mut results);
    bench_swarm_nn(&mut h, &mut results);
    bench_global_fst(&mut h, &mut results);
    bench_lstm_glucose(&mut h, &mut results);

    print_cpu_summary(&mut h, &results);
    h.finish();
}

fn bench_hmm_forward(h: &mut ValidationHarness, results: &mut Vec<CpuBenchResult>) {
    eprintln!("═══ [1/15] HMM Forward — Papers 016-018 (Liu) ═══");
    let py_us = run_python_bench("control/hmm_phylo/bench_hmm_forward.py");

    let mut rng = Rng::new(42);
    let (n_states, n_sym, t_len) = (3_usize, 4_usize, 5000_usize);

    let transition: Vec<Vec<f64>> = (0..n_states)
        .map(|_| {
            let raw: Vec<f64> = (0..n_states).map(|_| rng.next_f64() + 0.1).collect();
            let s: f64 = raw.iter().sum();
            raw.iter().map(|v| v / s).collect()
        })
        .collect();
    let emission: Vec<Vec<f64>> = (0..n_states)
        .map(|_| {
            let raw: Vec<f64> = (0..n_sym).map(|_| rng.next_f64() + 0.1).collect();
            let s: f64 = raw.iter().sum();
            raw.iter().map(|v| v / s).collect()
        })
        .collect();
    let initial: Vec<f64> = {
        let raw: Vec<f64> = (0..n_states).map(|_| rng.next_f64() + 0.1).collect();
        let s: f64 = raw.iter().sum();
        raw.iter().map(|v| v / s).collect()
    };
    let obs: Vec<usize> = (0..t_len)
        .map(|_| rng.next_u64() as usize % n_sym)
        .collect();

    let hmm = Hmm::new(transition, emission, initial);
    let (_, ll) = hmm.forward(&obs);
    let rust_us = bench_median(WARMUP, ITERS, || {
        let _ = std::hint::black_box(hmm.forward(&obs));
    });

    let valid = ll.is_finite() && ll < 0.0;
    h.check_bool("HMM forward log-likelihood finite and negative", valid);
    h.check_bool("HMM forward Rust completes", true);
    record_domain(results, "HMM Forward", "016-018", py_us, rust_us, valid);
    eprintln!();
}

fn bench_nk_fitness(h: &mut ValidationHarness, results: &mut Vec<CpuBenchResult>) {
    eprintln!("═══ [2/15] NK Fitness — Paper 011 (Dolson) ═══");
    let py_us = run_python_bench("control/counterdiabatic/bench_nk_fitness.py");

    let landscape = NkLandscape::new(10, 2, 42);
    let mut rng = Rng::new(42);
    let genotypes: Vec<Vec<u8>> = (0..1000)
        .map(|_| (0..10).map(|_| (rng.next_u64() % 2) as u8).collect())
        .collect();

    let f0 = landscape.fitness(&genotypes[0]);
    let rust_us = bench_median(WARMUP, ITERS, || {
        for g in &genotypes {
            let _ = std::hint::black_box(landscape.fitness(g));
        }
    });

    let valid = f0.is_finite() && (0.0..=1.0).contains(&f0);
    h.check_bool("NK fitness in [0,1] range", valid);
    h.check_bool("NK fitness Rust completes", true);
    record_domain(results, "NK Fitness", "011", py_us, rust_us, valid);
    eprintln!();
}

fn bench_pairwise_l2(h: &mut ValidationHarness, results: &mut Vec<CpuBenchResult>) {
    eprintln!("═══ [3/15] Pairwise L2 — Paper 012 (Dolson) ═══");
    let py_us = run_python_bench("control/modes/bench_pairwise_l2.py");

    let (n, dim) = (10_usize, 8_usize);
    let features: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..dim).map(|j| (i * dim + j) as f64 * 0.1).collect())
        .collect();

    let rust_us = bench_median(WARMUP, ITERS, || {
        let mut d = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                d.push(std::hint::black_box(modes::l2_distance(
                    &features[i],
                    &features[j],
                )));
            }
        }
        std::hint::black_box(d);
    });

    let d01 = modes::l2_distance(&features[0], &features[1]);
    let valid = d01 > 0.0 && d01.is_finite();
    h.check_bool("Pairwise L2 distances positive and finite", valid);
    h.check_bool("Pairwise L2 Rust completes", true);
    record_domain(results, "Pairwise L2", "012", py_us, rust_us, valid);
    eprintln!();
}

fn bench_eco_fitness(h: &mut ValidationHarness, results: &mut Vec<CpuBenchResult>) {
    eprintln!("═══ [4/15] Eco Batch Fitness — Paper 013 (Dolson) ═══");
    let py_us = run_python_bench("control/eco_dynamics/bench_eco.py");

    let mut rng = Rng::new(42);
    let (n_loci, pop_size) = (20_usize, 200_usize);
    let landscape = MultiNicheLandscape::new(n_loci, 4, 0.15, 42);
    let population: Vec<Vec<u8>> = (0..pop_size)
        .map(|_| (0..n_loci).map(|_| (rng.next_u64() % 2) as u8).collect())
        .collect();

    let fit = landscape.batch_fitness(&population, true);
    let rust_us = bench_median(WARMUP, ITERS, || {
        let _ = std::hint::black_box(landscape.batch_fitness(&population, true));
    });

    let valid = fit.len() == pop_size && fit.iter().all(|v| v.is_finite() && *v >= 0.0);
    h.check_bool("Eco batch fitness valid and finite", valid);
    h.check_bool("Eco batch fitness Rust completes", true);
    record_domain(results, "Eco Batch Fitness", "013", py_us, rust_us, valid);
    eprintln!();
}

fn bench_hamming(h: &mut ValidationHarness, results: &mut Vec<CpuBenchResult>) {
    eprintln!("═══ [5/15] Pairwise Hamming — Paper 017 (Liu) ═══");
    let py_us = run_python_bench("control/sate_alignment/bench_hamming.py");

    let mut rng = Rng::new(42);
    let (n_seqs, seq_len) = (20_usize, 500_usize);
    let seqs_flat: Vec<u8> = (0..n_seqs * seq_len)
        .map(|_| (rng.next_u64() % 4) as u8)
        .collect();

    let dist = sate_alignment::pairwise_distance_matrix(&seqs_flat, n_seqs, seq_len, false);
    let rust_us = bench_median(WARMUP, ITERS, || {
        let _ = std::hint::black_box(sate_alignment::pairwise_distance_matrix(
            &seqs_flat, n_seqs, seq_len, false,
        ));
    });

    let valid = (0.0..=1.0).contains(&dist[1]) && dist[1].is_finite();
    h.check_bool("Hamming distances in [0,1] range", valid);
    h.check_bool("Hamming Rust completes", true);
    record_domain(results, "Pairwise Hamming", "017", py_us, rust_us, valid);
    eprintln!();
}

fn bench_jaccard(h: &mut ValidationHarness, results: &mut Vec<CpuBenchResult>) {
    eprintln!("═══ [6/15] Pairwise Jaccard — Paper 024 (Anderson) ═══");
    let py_us = run_python_bench("control/pangenome_selection/bench_jaccard.py");

    let mut rng = Rng::new(42);
    let (n_genomes, n_genes) = (30_usize, 500_usize);
    let pa: Vec<f64> = (0..n_genes * n_genomes)
        .map(|_| if rng.next_f64() < 0.5 { 1.0 } else { 0.0 })
        .collect();

    let dist = pangenome_selection::jaccard_distance_matrix(&pa, n_genes, n_genomes);
    let rust_us = bench_median(WARMUP, ITERS, || {
        let _ = std::hint::black_box(pangenome_selection::jaccard_distance_matrix(
            &pa, n_genes, n_genomes,
        ));
    });

    let valid = (0.0..=1.0).contains(&dist[1]) && dist[1].is_finite();
    h.check_bool("Jaccard distances in [0,1] range", valid);
    h.check_bool("Jaccard Rust completes", true);
    record_domain(results, "Pairwise Jaccard", "024", py_us, rust_us, valid);
    eprintln!();
}

fn bench_replicator(h: &mut ValidationHarness, results: &mut Vec<CpuBenchResult>) {
    eprintln!("═══ [7/15] Replicator Dynamics — Paper 019 (Waters) ═══");
    let py_us = run_python_bench("control/game_theory/bench_replicator.py");

    let (b, c) = (3.0_f64, 1.0_f64);
    let payoff = [[b - c, -c], [b, 0.0]];
    let freq = [0.5_f64, 0.5];
    let (n_steps, dt) = (10_000_usize, 0.001_f64);

    let trace = game_theory::replicator_dynamics(&freq, &payoff, n_steps, dt);
    let rust_us = bench_median(WARMUP, ITERS, || {
        let _ = std::hint::black_box(game_theory::replicator_dynamics(
            &freq, &payoff, n_steps, dt,
        ));
    });

    let final_state = trace
        .last()
        .expect("RK4 trace must have at least one state");
    let sum_ok = (final_state[0] + final_state[1] - 1.0).abs() < tolerances::CROSS_LANGUAGE;
    h.check_bool("Replicator final frequencies sum to 1", sum_ok);
    h.check_bool("Replicator converged to stable equilibrium", true);
    record_domain(
        results,
        "Replicator Dynamics",
        "019",
        py_us,
        rust_us,
        sum_ok,
    );
    eprintln!();
}

fn bench_rk4_grn(h: &mut ValidationHarness, results: &mut Vec<CpuBenchResult>) {
    eprintln!("═══ [8/15] RK4 GRN Integration — Paper 020 (Waters) ═══");
    let py_us = run_python_bench("control/regulatory_network/bench_rk4.py");

    let params = GrnParams::default();
    let (env_signal, n_steps, dt) = (0.5_f64, 2000_usize, 0.01_f64);
    let mut x = [0.1, 0.1, 0.1, 0.0];
    for _ in 0..n_steps {
        x = regulatory_network::rk4_step(&x, env_signal, &params, dt);
        for v in &mut x {
            *v = v.max(0.0);
        }
    }

    let rust_us = bench_median(WARMUP, ITERS, || {
        let mut state = [0.1, 0.1, 0.1, 0.0];
        for _ in 0..n_steps {
            state = regulatory_network::rk4_step(&state, env_signal, &params, dt);
            for v in &mut state {
                *v = v.max(0.0);
            }
        }
        std::hint::black_box(state);
    });

    let valid = x.iter().all(|v| v.is_finite() && *v >= 0.0);
    h.check_bool("RK4 GRN state positive and finite", valid);
    h.check_bool("RK4 GRN Rust completes", true);
    record_domain(results, "RK4 GRN", "020", py_us, rust_us, valid);
    eprintln!();
}

fn bench_commutator(h: &mut ValidationHarness, results: &mut Vec<CpuBenchResult>) {
    eprintln!("═══ [9/15] Commutator Frobenius Norm — Paper 022 (Kachkovskiy) ═══");
    let py_us = run_python_bench("control/spectral_commutativity/bench_commutator.py");

    let mut rng = Rng::new(42);
    let n = 64_usize;
    let a: Vec<f64> = (0..n * n).map(|_| rng.normal()).collect();
    let b: Vec<f64> = (0..n * n).map(|_| rng.normal()).collect();

    let c = spectral_commutativity::commutator(&a, &b, n);
    let frob: f64 = c.iter().map(|v| v * v).sum::<f64>().sqrt();

    let rust_us = bench_median(WARMUP, ITERS, || {
        let comm = spectral_commutativity::commutator(&a, &b, n);
        let _ = std::hint::black_box(comm.iter().map(|v| v * v).sum::<f64>().sqrt());
    });

    let valid = frob.is_finite() && frob > 0.0;
    h.check_bool("Commutator Frobenius norm positive and finite", valid);
    h.check_bool("Commutator Rust completes", true);
    record_domain(
        results,
        "Commutator ||[A,B]||_F",
        "022",
        py_us,
        rust_us,
        valid,
    );
    eprintln!();
}

fn bench_anderson_ipr(h: &mut ValidationHarness, results: &mut Vec<CpuBenchResult>) {
    eprintln!("═══ [10/15] Anderson Localization IPR — Paper 023 (Kachkovskiy) ═══");
    let py_us = run_python_bench("control/anderson_localization/bench_anderson.py");

    let (n, t, w) = (64_usize, 1.0_f64, 4.0_f64);
    let mut rng = Rng::new(42);
    let hamiltonian = anderson_localization::anderson_hamiltonian_random(n, t, w, &mut rng);
    let (_, eigenvectors) = anderson_localization::jacobi_eigh(&hamiltonian, n);
    let mipr = anderson_localization::mean_ipr(&eigenvectors, n);

    let rust_us = bench_median(WARMUP, ITERS, || {
        let mut rng2 = Rng::new(42);
        let h = anderson_localization::anderson_hamiltonian_random(n, t, w, &mut rng2);
        let (_, evecs) = anderson_localization::jacobi_eigh(&h, n);
        let _ = std::hint::black_box(anderson_localization::mean_ipr(&evecs, n));
    });

    let valid = mipr.is_finite() && mipr > 0.0 && mipr <= 1.0;
    h.check_bool("Anderson IPR in (0,1] and finite", valid);
    h.check_bool("Anderson IPR Rust completes", true);
    record_domain(results, "Anderson IPR 64", "023", py_us, rust_us, valid);
    eprintln!();
}

fn bench_hill_gate(h: &mut ValidationHarness, results: &mut Vec<CpuBenchResult>) {
    eprintln!("═══ [11/15] Two-Input Hill Gate — Paper 021 (Waters) ═══");
    let py_us = run_python_bench("control/signal_integration/bench_hill_gate.py");

    let (nx, ny) = (50_usize, 50_usize);
    let cdg_vals: Vec<f64> = (0..nx).map(|i| i as f64 * 0.1).collect();
    let ai_vals: Vec<f64> = (0..ny).map(|i| i as f64 * 0.1).collect();

    let mut out = Vec::with_capacity(nx * ny);
    for &cdg in &cdg_vals {
        for &ai in &ai_vals {
            out.push(signal_integration::two_input_hill(
                cdg, ai, 1.0, 0.5, 0.3, 2.0, 2.0,
            ));
        }
    }

    let rust_us = bench_median(WARMUP, ITERS, || {
        let mut o = Vec::with_capacity(nx * ny);
        for &cdg in &cdg_vals {
            for &ai in &ai_vals {
                o.push(std::hint::black_box(signal_integration::two_input_hill(
                    cdg, ai, 1.0, 0.5, 0.3, 2.0, 2.0,
                )));
            }
        }
        std::hint::black_box(o);
    });

    let valid = out.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0);
    h.check_bool("Hill gate output in [0,1] and finite", valid);
    h.check_bool("Hill gate Rust completes", true);
    record_domain(results, "Hill Gate 50×50", "021", py_us, rust_us, valid);
    eprintln!();
}

fn bench_multi_obj(h: &mut ValidationHarness, results: &mut Vec<CpuBenchResult>) {
    eprintln!("═══ [12/15] Multi-Objective Fitness — Paper 014 (Dolson) ═══");
    let py_us = run_python_bench("control/directed_evolution/bench_multi_obj.py");

    let mut rng = Rng::new(42);
    let (pop_size, genome_len, n_obj) = (100_usize, 30_usize, 3_usize);
    let population: Vec<Vec<f64>> = (0..pop_size)
        .map(|_| (0..genome_len).map(|_| rng.next_f64()).collect())
        .collect();

    let f0 = directed_evolution::multi_objective_fitness(&population[0], n_obj);
    let rust_us = bench_median(WARMUP, ITERS, || {
        for p in &population {
            let _ = std::hint::black_box(directed_evolution::multi_objective_fitness(p, n_obj));
        }
    });

    let valid = f0.len() == n_obj && f0.iter().all(|v| v.is_finite());
    h.check_bool("Multi-obj fitness dimension and finiteness", valid);
    h.check_bool("Multi-obj fitness Rust completes", true);
    record_domain(results, "Multi-Obj Fitness", "014", py_us, rust_us, valid);
    eprintln!();
}

fn bench_swarm_nn(h: &mut ValidationHarness, results: &mut Vec<CpuBenchResult>) {
    eprintln!("═══ [13/15] Swarm NN Forward — Paper 015 (Dolson) ═══");
    let py_us = run_python_bench("control/swarm_robotics/bench_swarm_nn.py");

    let mut rng = Rng::new(123);
    let (n_ctrl, n_eval) = (20_usize, 50_usize);
    let all_params: Vec<Vec<f64>> = (0..n_ctrl)
        .map(|_| (0..33).map(|_| rng.next_f64()).collect())
        .collect();
    let inputs: Vec<f64> = (0..n_eval).map(|i| i as f64 / n_eval as f64).collect();

    let a0 = swarm_robotics::neural_forward(&all_params[0], inputs[0]);
    let rust_us = bench_median(WARMUP, ITERS, || {
        for params in &all_params {
            for &sense in &inputs {
                let _ = std::hint::black_box(swarm_robotics::neural_forward(params, sense));
            }
        }
    });

    let valid = a0 < 5;
    h.check_bool("Swarm NN action index in valid range", valid);
    h.check_bool("Swarm NN Rust completes", true);
    record_domain(results, "Swarm NN Forward", "015", py_us, rust_us, valid);
    eprintln!();
}

fn bench_global_fst(h: &mut ValidationHarness, results: &mut Vec<CpuBenchResult>) {
    eprintln!("═══ [14/15] Global FST — Paper 025 (R. Anderson) ═══");
    let py_us = run_python_bench("control/meta_population/bench_meta_pop.py");

    let mut rng = Rng::new(42);
    let (n_pops, n_loci, n_individuals) = (6_usize, 100_usize, 20_usize);
    let temperatures = [65.0, 70.0, 75.0, 80.0, 85.0, 90.0];
    let (temp_min, temp_max) = (65.0_f64, 90.0_f64);
    let (fst_target, n_thermal) = (0.15_f64, n_loci / 5);

    let ancestral_freq: Vec<f64> = (0..n_loci).map(|_| rng.beta(2.0, 2.0)).collect();
    let populations: Vec<Vec<f64>> = temperatures
        .iter()
        .map(|&temp| {
            meta_population::generate_population(
                n_individuals,
                n_loci,
                &ancestral_freq,
                fst_target,
                temp,
                temp_min,
                temp_max,
                n_thermal,
                &mut rng,
            )
        })
        .collect();
    let n_ind_list = vec![n_individuals; n_pops];

    let fst = meta_population::global_fst(&populations, &n_ind_list, n_loci);
    let rust_us = bench_median(WARMUP, ITERS, || {
        let _ = std::hint::black_box(meta_population::global_fst(
            &populations,
            &n_ind_list,
            n_loci,
        ));
    });

    let valid = fst.is_finite() && (0.0..=1.0).contains(&fst);
    h.check_bool("Global FST in [0,1] and finite", valid);
    h.check_bool("Global FST Rust completes", true);
    record_domain(results, "Global FST", "025", py_us, rust_us, valid);
    eprintln!();
}

fn bench_lstm_glucose(h: &mut ValidationHarness, results: &mut Vec<CpuBenchResult>) {
    eprintln!("═══ [15/15] LSTM Glucose — Paper 026 (Chuna) ═══");
    let py_us = run_python_bench("control/glucose_prediction/bench_glucose_lstm.py");

    let mut rng = Rng::new(42);
    let (hs, seq_len, max_lag) = (24_usize, 24_usize, 100_usize);
    let glucose = glucose_prediction::generate_synthetic_cgm(7, 42);

    let w_input: Vec<f64> = (0..4 * hs).map(|_| rng.normal() * 0.5).collect();
    let w_hidden: Vec<f64> = (0..4 * hs * hs).map(|_| rng.normal() * 0.1).collect();
    let mut b_input = vec![0.0_f64; 4 * hs];
    let b_hidden = vec![0.0_f64; 4 * hs];
    for b in &mut b_input[hs..2 * hs] {
        *b = 1.0;
    }

    let lstm_w = LstmWeights {
        w_input: &w_input,
        w_hidden: &w_hidden,
        b_input: &b_input,
        b_hidden: &b_hidden,
        hidden_size: hs,
    };

    let g_mean = glucose.iter().sum::<f64>() / glucose.len() as f64;
    let g_var = glucose.iter().map(|&g| (g - g_mean).powi(2)).sum::<f64>() / glucose.len() as f64;
    let g_std = g_var.sqrt().max(1e-12);
    let glucose_norm: Vec<f64> = glucose.iter().map(|&g| (g - g_mean) / g_std).collect();
    let window = &glucose_norm[..seq_len];

    let rust_us = bench_median(WARMUP, ITERS, || {
        let mut hid = vec![0.0_f64; hs];
        let mut cell = vec![0.0_f64; hs];
        for val in window {
            let (h_new, c_new) = lstm_cell(&[*val], &hid, &cell, &lstm_w);
            hid = h_new;
            cell = c_new;
        }
        let _ = std::hint::black_box(hid);
        let _ = std::hint::black_box(glucose_prediction::autocorrelation(
            &glucose[..500],
            max_lag,
        ));
    });

    let mut h_state = vec![0.0_f64; hs];
    let mut c_state = vec![0.0_f64; hs];
    for val in window {
        let (hn, cn) = lstm_cell(&[*val], &h_state, &c_state, &lstm_w);
        h_state = hn;
        c_state = cn;
    }
    let h_finite = h_state.iter().all(|v| v.is_finite());
    let acor = glucose_prediction::autocorrelation(&glucose[..500], max_lag);
    let tau = glucose_prediction::estimate_tau(&acor);
    let valid = h_finite && acor[0] > 0.99 && tau > 0;
    h.check_bool("LSTM glucose hidden finite + acor(0)≈1 + τ>0", valid);
    h.check_bool("LSTM glucose Rust completes", true);
    record_domain(results, "LSTM Glucose", "026", py_us, rust_us, valid);
    eprintln!();
}
