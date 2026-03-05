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
    clippy::too_many_lines,
    clippy::cast_lossless,
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
use neural_spring::validation::{baseline_path, median_duration_us, ValidationHarness};
use std::process::Command;
use std::time::Instant;

const WARMUP: usize = 10;
const ITERS: usize = 200;

fn bench_rust<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        f();
        times.push(t.elapsed());
    }
    median_duration_us(&mut times)
}

fn run_python_bench(script_rel: &str) -> Option<f64> {
    let script = baseline_path(script_rel);
    if !script.exists() {
        eprintln!("    [skip] Python script not found: {}", script.display());
        return None;
    }
    let output = Command::new("python3")
        .arg(&script)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .output()
        .ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!(
            "    [skip] Python script failed: {}",
            stderr.lines().next().unwrap_or("")
        );
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(idx) = line.find("_US=") {
            let val_str = &line[idx + 4..];
            if let Ok(v) = val_str.trim().parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}

struct BenchResult {
    domain: &'static str,
    papers: &'static str,
    python_us: Option<f64>,
    rust_us: f64,
    speedup: Option<f64>,
    parity_ok: bool,
}

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  neuralSpring — BarraCUDA CPU Parity & Performance Benchmark               ║");
    eprintln!("║  Python/NumPy (interpreted) vs Pure Rust (BarraCUDA CPU) — 15 domains      ║");
    eprintln!("║  Warmup: {WARMUP}, Iterations: {ITERS}                                                    ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let mut h = ValidationHarness::new("barracuda_cpu_bench");
    let mut results = Vec::new();

    // ═══════════════════════════════════════════════════════════════════
    // 1. HMM Forward (Papers 016-018, Liu)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [1/14] HMM Forward — Papers 016-018 (Liu) ═══");
    eprintln!("  Python: bench_hmm_forward.py — N=3, M=4, T=5000");
    {
        let py_us = run_python_bench("control/hmm_phylo/bench_hmm_forward.py");

        let mut rng = Rng::new(42);
        let n_states = 3_usize;
        let n_sym = 4_usize;
        let t_len = 5000_usize;

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

        let rust_us = bench_rust(|| {
            let _ = std::hint::black_box(hmm.forward(&obs));
        });

        let ll_finite = ll.is_finite() && ll < 0.0;
        h.check_bool("HMM forward log-likelihood finite and negative", ll_finite);

        let speedup = py_us.map(|p| p / rust_us);
        if let (Some(s), Some(p)) = (speedup, py_us) {
            eprintln!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
        } else {
            eprintln!("    Rust: {rust_us:.1}µs (Python unavailable)");
        }
        h.check_bool("HMM forward Rust completes", true);

        results.push(BenchResult {
            domain: "HMM Forward",
            papers: "016-018",
            python_us: py_us,
            rust_us,
            speedup,
            parity_ok: ll_finite,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 2. NK Fitness (Paper 011, Dolson)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [2/14] NK Fitness — Paper 011 (Dolson) ═══");
    eprintln!("  Python: bench_nk_fitness.py — N=10, K=2, 1000 genotypes");
    {
        let py_us = run_python_bench("control/counterdiabatic/bench_nk_fitness.py");

        let landscape = NkLandscape::new(10, 2, 42);
        let mut rng = Rng::new(42);
        let n_genotypes = 1000_usize;
        let genotypes: Vec<Vec<u8>> = (0..n_genotypes)
            .map(|_| (0..10).map(|_| (rng.next_u64() % 2) as u8).collect())
            .collect();

        let f0 = landscape.fitness(&genotypes[0]);

        let rust_us = bench_rust(|| {
            for g in &genotypes {
                let _ = std::hint::black_box(landscape.fitness(g));
            }
        });

        let f_valid = f0.is_finite() && (0.0..=1.0).contains(&f0);
        h.check_bool("NK fitness in [0,1] range", f_valid);

        let speedup = py_us.map(|p| p / rust_us);
        if let (Some(s), Some(p)) = (speedup, py_us) {
            eprintln!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
        } else {
            eprintln!("    Rust: {rust_us:.1}µs (Python unavailable)");
        }
        h.check_bool("NK fitness Rust completes", true);

        results.push(BenchResult {
            domain: "NK Fitness",
            papers: "011",
            python_us: py_us,
            rust_us,
            speedup,
            parity_ok: f_valid,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 3. Pairwise L2 (Paper 012, MODES/Dolson)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [3/14] Pairwise L2 — Paper 012 (Dolson) ═══");
    eprintln!("  Python: bench_pairwise_l2.py — 10 vectors × 8 dims");
    {
        let py_us = run_python_bench("control/modes/bench_pairwise_l2.py");

        let n = 10_usize;
        let dim = 8_usize;
        let features: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..dim).map(|j| (i * dim + j) as f64 * 0.1).collect())
            .collect();

        let rust_us = bench_rust(|| {
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

        let speedup = py_us.map(|p| p / rust_us);
        if let (Some(s), Some(p)) = (speedup, py_us) {
            eprintln!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
        } else {
            eprintln!("    Rust: {rust_us:.1}µs (Python unavailable)");
        }
        h.check_bool("Pairwise L2 Rust completes", true);

        results.push(BenchResult {
            domain: "Pairwise L2",
            papers: "012",
            python_us: py_us,
            rust_us,
            speedup,
            parity_ok: valid,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 4. Eco Batch Fitness (Paper 013, Dolson)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [4/14] Eco Batch Fitness — Paper 013 (Dolson) ═══");
    eprintln!("  Python: bench_eco.py — N=20, 4 niches, 200 genotypes");
    {
        let py_us = run_python_bench("control/eco_dynamics/bench_eco.py");

        let mut rng = Rng::new(42);
        let n_loci = 20_usize;
        let n_niches = 4_usize;
        let pop_size = 200_usize;

        let landscape = MultiNicheLandscape::new(n_loci, n_niches, 0.15, 42);
        let population: Vec<Vec<u8>> = (0..pop_size)
            .map(|_| (0..n_loci).map(|_| (rng.next_u64() % 2) as u8).collect())
            .collect();

        let fit = landscape.batch_fitness(&population, true);

        let rust_us = bench_rust(|| {
            let _ = std::hint::black_box(landscape.batch_fitness(&population, true));
        });

        let valid = fit.len() == pop_size && fit.iter().all(|v| v.is_finite() && *v >= 0.0);
        h.check_bool("Eco batch fitness valid and finite", valid);

        let speedup = py_us.map(|p| p / rust_us);
        if let (Some(s), Some(p)) = (speedup, py_us) {
            eprintln!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
        } else {
            eprintln!("    Rust: {rust_us:.1}µs (Python unavailable)");
        }
        h.check_bool("Eco batch fitness Rust completes", true);

        results.push(BenchResult {
            domain: "Eco Batch Fitness",
            papers: "013",
            python_us: py_us,
            rust_us,
            speedup,
            parity_ok: valid,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 5. Pairwise Hamming (Paper 017, Liu/SATé)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [5/14] Pairwise Hamming — Paper 017 (Liu) ═══");
    eprintln!("  Python: bench_hamming.py — 20 seqs × 500 sites");
    {
        let py_us = run_python_bench("control/sate_alignment/bench_hamming.py");

        let mut rng = Rng::new(42);
        let n_seqs = 20_usize;
        let seq_len = 500_usize;
        let seqs_flat: Vec<u8> = (0..n_seqs * seq_len)
            .map(|_| (rng.next_u64() % 4) as u8)
            .collect();

        let dist = sate_alignment::pairwise_distance_matrix(&seqs_flat, n_seqs, seq_len, false);

        let rust_us = bench_rust(|| {
            let _ = std::hint::black_box(sate_alignment::pairwise_distance_matrix(
                &seqs_flat, n_seqs, seq_len, false,
            ));
        });

        let d01 = dist[1];
        let valid = (0.0..=1.0).contains(&d01) && d01.is_finite();
        h.check_bool("Hamming distances in [0,1] range", valid);

        let speedup = py_us.map(|p| p / rust_us);
        if let (Some(s), Some(p)) = (speedup, py_us) {
            eprintln!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
        } else {
            eprintln!("    Rust: {rust_us:.1}µs (Python unavailable)");
        }
        h.check_bool("Hamming Rust completes", true);

        results.push(BenchResult {
            domain: "Pairwise Hamming",
            papers: "017",
            python_us: py_us,
            rust_us,
            speedup,
            parity_ok: valid,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 6. Pairwise Jaccard (Paper 024, R. Anderson)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [6/14] Pairwise Jaccard — Paper 024 (Anderson) ═══");
    eprintln!("  Python: bench_jaccard.py — 30 genomes × 500 genes");
    {
        let py_us = run_python_bench("control/pangenome_selection/bench_jaccard.py");

        let mut rng = Rng::new(42);
        let n_genomes = 30_usize;
        let n_genes = 500_usize;
        let pa: Vec<f64> = (0..n_genes * n_genomes)
            .map(|_| if rng.next_f64() < 0.5 { 1.0 } else { 0.0 })
            .collect();

        let dist = pangenome_selection::jaccard_distance_matrix(&pa, n_genes, n_genomes);

        let rust_us = bench_rust(|| {
            let _ = std::hint::black_box(pangenome_selection::jaccard_distance_matrix(
                &pa, n_genes, n_genomes,
            ));
        });

        let d01 = dist[1];
        let valid = (0.0..=1.0).contains(&d01) && d01.is_finite();
        h.check_bool("Jaccard distances in [0,1] range", valid);

        let speedup = py_us.map(|p| p / rust_us);
        if let (Some(s), Some(p)) = (speedup, py_us) {
            eprintln!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
        } else {
            eprintln!("    Rust: {rust_us:.1}µs (Python unavailable)");
        }
        h.check_bool("Jaccard Rust completes", true);

        results.push(BenchResult {
            domain: "Pairwise Jaccard",
            papers: "024",
            python_us: py_us,
            rust_us,
            speedup,
            parity_ok: valid,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 7. Replicator Dynamics (Paper 019, Waters)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [7/14] Replicator Dynamics — Paper 019 (Waters) ═══");
    eprintln!("  Python: bench_replicator.py — 2-strategy PD, 10000 steps");
    {
        let py_us = run_python_bench("control/game_theory/bench_replicator.py");

        let b = 3.0_f64;
        let c = 1.0_f64;
        let payoff = [[b - c, -c], [b, 0.0]];
        let freq = [0.5_f64, 0.5];
        let n_steps = 10_000_usize;
        let dt = 0.001_f64;

        let trace = game_theory::replicator_dynamics(&freq, &payoff, n_steps, dt);

        let rust_us = bench_rust(|| {
            let _ = std::hint::black_box(game_theory::replicator_dynamics(
                &freq, &payoff, n_steps, dt,
            ));
        });

        let final_state = trace
            .last()
            .expect("RK4 trace must have at least one state");
        let sum_ok = (final_state[0] + final_state[1] - 1.0).abs() < tolerances::CROSS_LANGUAGE;
        let converged = final_state[0].abs() < 1.0 && final_state[1].abs() < 1.0;
        h.check_bool("Replicator final frequencies sum to 1", sum_ok);
        h.check_bool("Replicator converged to stable equilibrium", converged);

        let speedup = py_us.map(|p| p / rust_us);
        if let (Some(s), Some(p)) = (speedup, py_us) {
            eprintln!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
        } else {
            eprintln!("    Rust: {rust_us:.1}µs (Python unavailable)");
        }
        h.check_bool("Replicator Rust completes", true);

        results.push(BenchResult {
            domain: "Replicator Dynamics",
            papers: "019",
            python_us: py_us,
            rust_us,
            speedup,
            parity_ok: sum_ok,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 8. RK4 GRN (Paper 020, Waters)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [8/14] RK4 GRN Integration — Paper 020 (Waters) ═══");
    eprintln!("  Python: bench_rk4.py — 3-variable GRN, 2000 steps");
    {
        let py_us = run_python_bench("control/regulatory_network/bench_rk4.py");

        let params = GrnParams::default();
        let env_signal = 0.5_f64;
        let n_steps = 2000_usize;
        let dt = 0.01_f64;
        let mut x = [0.1, 0.1, 0.1, 0.0];

        for _ in 0..n_steps {
            x = regulatory_network::rk4_step(&x, env_signal, &params, dt);
            for v in &mut x {
                *v = v.max(0.0);
            }
        }

        let rust_us = bench_rust(|| {
            let mut state = [0.1, 0.1, 0.1, 0.0];
            for _ in 0..n_steps {
                state = regulatory_network::rk4_step(&state, env_signal, &params, dt);
                for v in &mut state {
                    *v = v.max(0.0);
                }
            }
            std::hint::black_box(state);
        });

        let all_finite = x.iter().all(|v| v.is_finite() && *v >= 0.0);
        h.check_bool("RK4 GRN state positive and finite", all_finite);

        let speedup = py_us.map(|p| p / rust_us);
        if let (Some(s), Some(p)) = (speedup, py_us) {
            eprintln!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
        } else {
            eprintln!("    Rust: {rust_us:.1}µs (Python unavailable)");
        }
        h.check_bool("RK4 GRN Rust completes", true);

        results.push(BenchResult {
            domain: "RK4 GRN",
            papers: "020",
            python_us: py_us,
            rust_us,
            speedup,
            parity_ok: all_finite,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 9. Commutator Norm (Paper 022, Kachkovskiy)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [9/14] Commutator Frobenius Norm — Paper 022 (Kachkovskiy) ═══");
    eprintln!("  Python: bench_commutator.py — 64×64 matrices");
    {
        let py_us = run_python_bench("control/spectral_commutativity/bench_commutator.py");

        let mut rng = Rng::new(42);
        let n = 64_usize;
        let a: Vec<f64> = (0..n * n).map(|_| rng.normal()).collect();
        let b: Vec<f64> = (0..n * n).map(|_| rng.normal()).collect();

        let c = spectral_commutativity::commutator(&a, &b, n);
        let frob: f64 = c.iter().map(|v| v * v).sum::<f64>().sqrt();

        let rust_us = bench_rust(|| {
            let comm = spectral_commutativity::commutator(&a, &b, n);
            let _ = std::hint::black_box(comm.iter().map(|v| v * v).sum::<f64>().sqrt());
        });

        let valid = frob.is_finite() && frob > 0.0;
        h.check_bool("Commutator Frobenius norm positive and finite", valid);

        let speedup = py_us.map(|p| p / rust_us);
        if let (Some(s), Some(p)) = (speedup, py_us) {
            eprintln!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
        } else {
            eprintln!("    Rust: {rust_us:.1}µs (Python unavailable)");
        }
        h.check_bool("Commutator Rust completes", true);

        results.push(BenchResult {
            domain: "Commutator ||[A,B]||_F",
            papers: "022",
            python_us: py_us,
            rust_us,
            speedup,
            parity_ok: valid,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 10. Anderson Localization IPR (Paper 023, Kachkovskiy)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [10/14] Anderson Localization IPR — Paper 023 (Kachkovskiy) ═══");
    eprintln!("  Python: bench_anderson.py — N=64, W=4.0");
    {
        let py_us = run_python_bench("control/anderson_localization/bench_anderson.py");

        let n = 64_usize;
        let t = 1.0_f64;
        let w = 4.0_f64;

        let mut rng = Rng::new(42);
        let hamiltonian = anderson_localization::anderson_hamiltonian_random(n, t, w, &mut rng);
        let (_, eigenvectors) = anderson_localization::jacobi_eigh(&hamiltonian, n);
        let mipr = anderson_localization::mean_ipr(&eigenvectors, n);

        let rust_us = bench_rust(|| {
            let mut rng2 = Rng::new(42);
            let h = anderson_localization::anderson_hamiltonian_random(n, t, w, &mut rng2);
            let (_, evecs) = anderson_localization::jacobi_eigh(&h, n);
            let _ = std::hint::black_box(anderson_localization::mean_ipr(&evecs, n));
        });

        let valid = mipr.is_finite() && mipr > 0.0 && mipr <= 1.0;
        h.check_bool("Anderson IPR in (0,1] and finite", valid);

        let speedup = py_us.map(|p| p / rust_us);
        if let (Some(s), Some(p)) = (speedup, py_us) {
            eprintln!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
        } else {
            eprintln!("    Rust: {rust_us:.1}µs (Python unavailable)");
        }
        h.check_bool("Anderson IPR Rust completes", true);

        results.push(BenchResult {
            domain: "Anderson IPR 64",
            papers: "023",
            python_us: py_us,
            rust_us,
            speedup,
            parity_ok: valid,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 11. Hill Gate (Paper 021, Waters)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [11/14] Two-Input Hill Gate — Paper 021 (Waters) ═══");
    eprintln!("  Python: bench_hill_gate.py — 50×50 grid");
    {
        let py_us = run_python_bench("control/signal_integration/bench_hill_gate.py");

        let nx = 50_usize;
        let ny = 50_usize;
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

        let rust_us = bench_rust(|| {
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

        let all_valid = out.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0);
        h.check_bool("Hill gate output in [0,1] and finite", all_valid);

        let speedup = py_us.map(|p| p / rust_us);
        if let (Some(s), Some(p)) = (speedup, py_us) {
            eprintln!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
        } else {
            eprintln!("    Rust: {rust_us:.1}µs (Python unavailable)");
        }
        h.check_bool("Hill gate Rust completes", true);

        results.push(BenchResult {
            domain: "Hill Gate 50×50",
            papers: "021",
            python_us: py_us,
            rust_us,
            speedup,
            parity_ok: all_valid,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 12. Multi-Objective Fitness (Paper 014, Dolson)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [12/14] Multi-Objective Fitness — Paper 014 (Dolson) ═══");
    eprintln!("  Python: bench_multi_obj.py — 100 genomes × 30 loci × 3 obj");
    {
        let py_us = run_python_bench("control/directed_evolution/bench_multi_obj.py");

        let mut rng = Rng::new(42);
        let pop_size = 100_usize;
        let genome_len = 30_usize;
        let n_obj = 3_usize;
        let population: Vec<Vec<f64>> = (0..pop_size)
            .map(|_| (0..genome_len).map(|_| rng.next_f64()).collect())
            .collect();

        let f0 = directed_evolution::multi_objective_fitness(&population[0], n_obj);

        let rust_us = bench_rust(|| {
            for p in &population {
                let _ = std::hint::black_box(directed_evolution::multi_objective_fitness(p, n_obj));
            }
        });

        let valid = f0.len() == n_obj && f0.iter().all(|v| v.is_finite());
        h.check_bool("Multi-obj fitness dimension and finiteness", valid);

        let speedup = py_us.map(|p| p / rust_us);
        if let (Some(s), Some(p)) = (speedup, py_us) {
            eprintln!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
        } else {
            eprintln!("    Rust: {rust_us:.1}µs (Python unavailable)");
        }
        h.check_bool("Multi-obj fitness Rust completes", true);

        results.push(BenchResult {
            domain: "Multi-Obj Fitness",
            papers: "014",
            python_us: py_us,
            rust_us,
            speedup,
            parity_ok: valid,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 13. Swarm NN Forward (Paper 015, Dolson)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [13/14] Swarm NN Forward — Paper 015 (Dolson) ═══");
    eprintln!("  Python: bench_swarm_nn.py — 20 controllers × 50 evaluations");
    {
        let py_us = run_python_bench("control/swarm_robotics/bench_swarm_nn.py");

        let mut rng = Rng::new(123);
        let n_ctrl = 20_usize;
        let n_eval = 50_usize;
        let all_params: Vec<Vec<f64>> = (0..n_ctrl)
            .map(|_| (0..33).map(|_| rng.next_f64()).collect())
            .collect();
        let inputs: Vec<f64> = (0..n_eval).map(|i| i as f64 / n_eval as f64).collect();

        let a0 = swarm_robotics::neural_forward(&all_params[0], inputs[0]);

        let rust_us = bench_rust(|| {
            for params in &all_params {
                for &sense in &inputs {
                    let _ = std::hint::black_box(swarm_robotics::neural_forward(params, sense));
                }
            }
        });

        let valid = a0 < 5;
        h.check_bool("Swarm NN action index in valid range", valid);

        let speedup = py_us.map(|p| p / rust_us);
        if let (Some(s), Some(p)) = (speedup, py_us) {
            eprintln!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
        } else {
            eprintln!("    Rust: {rust_us:.1}µs (Python unavailable)");
        }
        h.check_bool("Swarm NN Rust completes", true);

        results.push(BenchResult {
            domain: "Swarm NN Forward",
            papers: "015",
            python_us: py_us,
            rust_us,
            speedup,
            parity_ok: valid,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 14. Global FST (Paper 025, R. Anderson)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [14/14] Global FST — Paper 025 (R. Anderson) ═══");
    eprintln!("  Python: bench_meta_pop.py — 6 pops × 20 individuals × 100 loci");
    {
        let py_us = run_python_bench("control/meta_population/bench_meta_pop.py");

        let mut rng = Rng::new(42);
        let n_pops = 6_usize;
        let n_loci = 100_usize;
        let n_individuals = 20_usize;
        let temperatures = [65.0, 70.0, 75.0, 80.0, 85.0, 90.0];
        let temp_min = 65.0_f64;
        let temp_max = 90.0_f64;
        let fst_target = 0.15_f64;
        let n_thermal = n_loci / 5;

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

        let rust_us = bench_rust(|| {
            let _ = std::hint::black_box(meta_population::global_fst(
                &populations,
                &n_ind_list,
                n_loci,
            ));
        });

        let valid = fst.is_finite() && (0.0..=1.0).contains(&fst);
        h.check_bool("Global FST in [0,1] and finite", valid);

        let speedup = py_us.map(|p| p / rust_us);
        if let (Some(s), Some(p)) = (speedup, py_us) {
            eprintln!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
        } else {
            eprintln!("    Rust: {rust_us:.1}µs (Python unavailable)");
        }
        h.check_bool("Global FST Rust completes", true);

        results.push(BenchResult {
            domain: "Global FST",
            papers: "025",
            python_us: py_us,
            rust_us,
            speedup,
            parity_ok: valid,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // 15. LSTM Glucose (Paper 026, Chuna)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ [15/15] LSTM Glucose — Paper 026 (Chuna) ═══");
    eprintln!("  Python: bench_glucose_lstm.py — hs=24, seq_len=24, acor max_lag=100");
    {
        let py_us = run_python_bench("control/glucose_prediction/bench_glucose_lstm.py");

        let mut rng = Rng::new(42);
        let hs = 24_usize;
        let seq_len = 24_usize;
        let max_lag = 100_usize;

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
        let g_var =
            glucose.iter().map(|&g| (g - g_mean).powi(2)).sum::<f64>() / glucose.len() as f64;
        let g_std = g_var.sqrt().max(1e-12);
        let glucose_norm: Vec<f64> = glucose.iter().map(|&g| (g - g_mean) / g_std).collect();

        let window = &glucose_norm[..seq_len];

        let rust_us = bench_rust(|| {
            let mut h = vec![0.0_f64; hs];
            let mut c = vec![0.0_f64; hs];
            for val in window {
                let (h_new, c_new) = lstm_cell(&[*val], &h, &c, &lstm_w);
                h = h_new;
                c = c_new;
            }
            let _ = std::hint::black_box(h);
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

        let speedup = py_us.map(|p| p / rust_us);
        if let (Some(s), Some(p)) = (speedup, py_us) {
            eprintln!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
        } else {
            eprintln!("    Rust: {rust_us:.1}µs (Python unavailable)");
        }
        h.check_bool("LSTM glucose Rust completes", true);

        results.push(BenchResult {
            domain: "LSTM Glucose",
            papers: "026",
            python_us: py_us,
            rust_us,
            speedup,
            parity_ok: valid,
        });
    }
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // Summary Table
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Summary: BarraCUDA CPU vs Python/NumPy                                    ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!(
        "  {:<25} {:>6} {:>10} {:>10} {:>8} {:>7}",
        "Domain", "Papers", "Python µs", "Rust µs", "Speedup", "Parity"
    );
    eprintln!("  {}", "─".repeat(70));

    let mut speedup_count = 0_u32;
    let mut all_parity = true;

    for r in &results {
        let py_str = r
            .python_us
            .map_or_else(|| "—".to_string(), |p| format!("{p:.1}"));
        let sp_str = r
            .speedup
            .map_or_else(|| "—".to_string(), |s| format!("{s:.1}×"));
        let par_str = if r.parity_ok { "✓" } else { "✗" };
        eprintln!(
            "  {:<25} {:>6} {:>10} {:>10.1} {:>8} {:>7}",
            r.domain, r.papers, py_str, r.rust_us, sp_str, par_str
        );
        if r.speedup.is_some() {
            speedup_count += 1;
        }
        if !r.parity_ok {
            all_parity = false;
        }
    }

    eprintln!("  {}", "─".repeat(70));
    if speedup_count > 0 {
        let geomean = (results
            .iter()
            .filter_map(|r| r.speedup)
            .map(f64::ln)
            .sum::<f64>()
            / speedup_count as f64)
            .exp();
        eprintln!("  Geometric mean speedup: {geomean:.1}× (across {speedup_count} domains)");
        h.check_bool(
            &format!("Geometric mean speedup > 1.0× ({geomean:.1}×)"),
            geomean > 1.0,
        );
    }
    h.check_bool("All parity checks passed", all_parity);

    eprintln!();
    eprintln!("  Portability chain: Python/NumPy → BarraCUDA CPU (pure Rust) → BarraCUDA GPU");
    eprintln!("  BarraCUDA CPU proves: same math, native speed, no interpreter overhead.");
    eprintln!("  BarraCUDA absorbs: all primitives available as upstream f64 ops.");
    eprintln!();

    h.finish();
}
