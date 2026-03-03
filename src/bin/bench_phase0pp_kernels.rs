// SPDX-License-Identifier: AGPL-3.0-or-later

//! Phase 0++ kernel benchmarks: Rust (pure math) vs Python (`NumPy`).
//!
//! Runs each core computational kernel at the same problem size as its
//! corresponding `control/<module>/bench_*.py` script, then compares
//! median latencies to demonstrate `BarraCUDA` CPU parity and speedup.
//!
//! ```text
//! cargo run --release --bin bench_phase0pp_kernels
//! ```
//!
//! Machine-readable output for each kernel:
//! ```text
//! BENCH_<NAME>_RUST_US=<median_us>
//! ```
//!
//! To get the full comparison (Python + Rust), run:
//! ```text
//! cargo run --release --bin bench_phase0pp_kernels -- --with-python
//! ```

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "validation binary"
)]

use std::io::{BufRead, BufReader};
use std::process::Command;
use std::time::{Duration, Instant};

use neural_spring::counterdiabatic::NkLandscape;
use neural_spring::directed_evolution::multi_objective_fitness;
use neural_spring::game_theory;
use neural_spring::hmm::Hmm;
use neural_spring::modes::l2_distance;
use neural_spring::pangenome_selection;
use neural_spring::regulatory_network::{self, GrnParams};
use neural_spring::rng::Rng;
use neural_spring::sate_alignment;
use neural_spring::signal_integration::two_input_hill;
use neural_spring::spectral_commutativity;
use neural_spring::swarm_robotics::neural_forward;
use neural_spring::validation::median_duration_us;

const WARMUP: usize = 10;
const ITERATIONS: usize = 200;

fn main() {
    let with_python = std::env::args().any(|a| a == "--with-python");

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  neuralSpring — Phase 0++ Kernel Benchmarks                 ║");
    eprintln!("║  Pure Rust math vs Python/NumPy (single-thread)             ║");
    eprintln!("║  Warmup: {WARMUP}, Iterations: {ITERATIONS}                             ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();

    let mut results: Vec<BenchResult> = vec![
        bench_hmm_forward(),
        bench_replicator(),
        bench_commutator(),
        bench_nk_fitness(),
        bench_hamming(),
        bench_jaccard(),
        bench_rk4_grn(),
        bench_pairwise_l2(),
        bench_multi_obj_fitness(),
        bench_hill_gate(),
        bench_swarm_nn_forward(),
    ];

    if with_python {
        run_python_benchmarks(&mut results);
    }

    print_summary(&results);
}

// ── Result type ───────────────────────────────────────────────────────

struct BenchResult {
    name: String,
    kernel_tag: String,
    papers: String,
    rust_us: f64,
    python_us: Option<f64>,
    python_script: String,
}

// ── Benchmark: HMM Forward (Papers 016-018) ──────────────────────────

fn bench_hmm_forward() -> BenchResult {
    let mut rng = Rng::new(42);
    let n_states = 3;
    let n_obs_sym = 4;
    let seq_len = 5000;

    let transition = make_stochastic_matrix(n_states, n_states, &mut rng);
    let emission = make_stochastic_matrix(n_states, n_obs_sym, &mut rng);
    let initial = make_stochastic_row(n_states, &mut rng);

    let hmm = Hmm::new(transition, emission, initial);
    let obs: Vec<usize> = (0..seq_len).map(|_| rng.usize(n_obs_sym)).collect();

    let mut timings = bench_kernel(|| {
        let _ = hmm.forward(&obs);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_HMM_FORWARD_3x5000_RUST_US={us:.1}");

    BenchResult {
        name: "HMM forward (3×5000)".into(),
        kernel_tag: "HMM_FORWARD_3x5000".into(),
        papers: "016-018".into(),
        rust_us: us,
        python_us: None,
        python_script: "control/hmm_phylo/bench_hmm_forward.py".into(),
    }
}

// ── Benchmark: Replicator Dynamics (Paper 019) ───────────────────────

fn bench_replicator() -> BenchResult {
    let pd = game_theory::prisoners_dilemma_payoff(3.0, 1.0);

    let mut timings = bench_kernel(|| {
        let _ = game_theory::replicator_dynamics(&[0.5, 0.5], &pd, 10_000, 0.001);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_REPLICATOR_10000_RUST_US={us:.1}");

    BenchResult {
        name: "Replicator dynamics (10k steps)".into(),
        kernel_tag: "REPLICATOR_10000".into(),
        papers: "019".into(),
        rust_us: us,
        python_us: None,
        python_script: "control/game_theory/bench_replicator.py".into(),
    }
}

// ── Benchmark: Commutator Frobenius Norm (Paper 022) ─────────────────

fn bench_commutator() -> BenchResult {
    let mut rng = Rng::new(42);
    let dim = 64;
    let a = spectral_commutativity::random_matrix(dim, &mut rng);
    let b = spectral_commutativity::random_matrix(dim, &mut rng);

    let mut timings = bench_kernel(|| {
        let _ = spectral_commutativity::commutativity_ratio(&a, &b, dim);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_COMMUTATOR_64x64_RUST_US={us:.1}");

    BenchResult {
        name: "Commutator ‖[A,B]‖_F (64×64)".into(),
        kernel_tag: "COMMUTATOR_64x64".into(),
        papers: "022".into(),
        rust_us: us,
        python_us: None,
        python_script: "control/spectral_commutativity/bench_commutator.py".into(),
    }
}

// ── Benchmark: NK Fitness (Paper 011) ────────────────────────────────

fn bench_nk_fitness() -> BenchResult {
    let landscape = NkLandscape::new(10, 2, 42);
    let mut rng = Rng::new(42);
    let pop_size = 1000;
    let genotypes: Vec<Vec<u8>> = (0..pop_size)
        .map(|_| (0..10).map(|_| u8::from(rng.uniform() < 0.5)).collect())
        .collect();

    let mut timings = bench_kernel(|| {
        let mut total = 0.0_f64;
        for g in &genotypes {
            total += landscape.fitness(g);
        }
        std::hint::black_box(total);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_NK_FITNESS_10x2_1000_RUST_US={us:.1}");

    BenchResult {
        name: "NK fitness (N=10,K=2, 1000 genotypes)".into(),
        kernel_tag: "NK_FITNESS_10x2_1000".into(),
        papers: "011".into(),
        rust_us: us,
        python_us: None,
        python_script: "control/counterdiabatic/bench_nk_fitness.py".into(),
    }
}

// ── Benchmark: Pairwise Hamming (Paper 017) ──────────────────────────

fn bench_hamming() -> BenchResult {
    let mut rng = Rng::new(42);
    let n_seqs = 20;
    let seq_len = 500;
    let seqs: Vec<u8> = (0..n_seqs * seq_len).map(|_| rng.usize(4) as u8).collect();

    let mut timings = bench_kernel(|| {
        let _ = sate_alignment::pairwise_distance_matrix(&seqs, n_seqs, seq_len, false);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_HAMMING_20x500_RUST_US={us:.1}");

    BenchResult {
        name: "Pairwise Hamming (20×500)".into(),
        kernel_tag: "HAMMING_20x500".into(),
        papers: "017".into(),
        rust_us: us,
        python_us: None,
        python_script: "control/sate_alignment/bench_hamming.py".into(),
    }
}

// ── Benchmark: Jaccard Distance (Paper 024) ──────────────────────────

fn bench_jaccard() -> BenchResult {
    let mut rng = Rng::new(42);
    let n_genomes = 30;
    let n_genes = 500;
    let pa: Vec<f64> = (0..n_genes * n_genomes)
        .map(|_| if rng.uniform() < 0.5 { 1.0 } else { 0.0 })
        .collect();

    let mut timings = bench_kernel(|| {
        let _ = pangenome_selection::jaccard_distance_matrix(&pa, n_genes, n_genomes);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_JACCARD_30x500_RUST_US={us:.1}");

    BenchResult {
        name: "Jaccard distance (30×500)".into(),
        kernel_tag: "JACCARD_30x500".into(),
        papers: "024".into(),
        rust_us: us,
        python_us: None,
        python_script: "control/pangenome_selection/bench_jaccard.py".into(),
    }
}

// ── Benchmark: RK4 GRN Integration (Papers 020-021) ─────────────────

fn bench_rk4_grn() -> BenchResult {
    let p = GrnParams::default();
    let x0 = [0.5_f64, 0.1, 0.5, 0.1];

    let mut timings = bench_kernel(|| {
        let _ = regulatory_network::integrate_grn(&x0, 0.5, &p, 2000, 0.01);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_RK4_GRN_2000_RUST_US={us:.1}");

    BenchResult {
        name: "RK4 GRN ODE (2000 steps)".into(),
        kernel_tag: "RK4_GRN_2000".into(),
        papers: "020-021".into(),
        rust_us: us,
        python_us: None,
        python_script: "control/regulatory_network/bench_rk4.py".into(),
    }
}

// ── Benchmark: Pairwise L2 Distance (Paper 012 - MODES) ────────────────

fn bench_pairwise_l2() -> BenchResult {
    let n = 10_usize;
    let dim = 8_usize;
    let features: Vec<f64> = (0..n * dim).map(|i| (i as f64) * 0.1).collect();

    let mut timings = bench_kernel(|| {
        let mut dists = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                dists.push(l2_distance(
                    &features[i * dim..(i + 1) * dim],
                    &features[j * dim..(j + 1) * dim],
                ));
            }
        }
        std::hint::black_box(dists);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_PAIRWISE_L2_10x8_RUST_US={us:.1}");

    BenchResult {
        name: "Pairwise L2 distance (10×8)".into(),
        kernel_tag: "PAIRWISE_L2_10x8".into(),
        papers: "012".into(),
        rust_us: us,
        python_us: None,
        python_script: "control/modes/bench_pairwise_l2.py".into(),
    }
}

// ── Benchmark: Multi-objective Fitness (Paper 014 - Directed Evolution) ─

fn bench_multi_obj_fitness() -> BenchResult {
    let pop_size = 100_usize;
    let genome_len = 30_usize;
    let n_objectives = 3_usize;
    let mut rng = Rng::new(42);
    let population: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();

    let mut timings = bench_kernel(|| {
        let mut all = Vec::with_capacity(pop_size * n_objectives);
        for i in 0..pop_size {
            all.extend(multi_objective_fitness(
                &population[i * genome_len..(i + 1) * genome_len],
                n_objectives,
            ));
        }
        std::hint::black_box(all);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_MULTI_OBJ_FITNESS_100x30x3_RUST_US={us:.1}");

    BenchResult {
        name: "Multi-objective fitness (100×30×3)".into(),
        kernel_tag: "MULTI_OBJ_FITNESS_100x30x3".into(),
        papers: "014".into(),
        rust_us: us,
        python_us: None,
        python_script: "control/directed_evolution/bench_multi_obj.py".into(),
    }
}

// ── Benchmark: Two-input Hill Function (Paper 021 - Signal Integration) ─

fn bench_hill_gate() -> BenchResult {
    let nx = 50_usize;
    let ny = 50_usize;
    let cdg_vals: Vec<f64> = (0..nx).map(|i| i as f64 * 0.1).collect();
    let ai_vals: Vec<f64> = (0..ny).map(|i| i as f64 * 0.1).collect();

    let mut timings = bench_kernel(|| {
        let mut out = Vec::with_capacity(nx * ny);
        for &cdg in &cdg_vals {
            for &ai in &ai_vals {
                out.push(two_input_hill(cdg, ai, 1.0, 0.5, 0.3, 2.0, 2.0));
            }
        }
        std::hint::black_box(out);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_HILL_GATE_50x50_RUST_US={us:.1}");

    BenchResult {
        name: "Two-input Hill grid (50×50)".into(),
        kernel_tag: "HILL_GATE_50x50".into(),
        papers: "021".into(),
        rust_us: us,
        python_us: None,
        python_script: "control/signal_integration/bench_hill_gate.py".into(),
    }
}

// ── Benchmark: Swarm NN Forward Pass (Paper 015 - Swarm Robotics) ────────

fn bench_swarm_nn_forward() -> BenchResult {
    let n_ctrl = 20_usize;
    let n_eval = 50_usize;
    let mut rng = Rng::new(123);
    let all_params: Vec<Vec<f64>> = (0..n_ctrl)
        .map(|_| (0..33).map(|_| rng.uniform()).collect())
        .collect();
    let inputs: Vec<f64> = (0..n_eval).map(|i| (i as f64) / n_eval as f64).collect();

    let mut timings = bench_kernel(|| {
        let mut actions = Vec::with_capacity(n_ctrl * n_eval);
        for ctrl_params in &all_params {
            for &sense in &inputs {
                actions.push(neural_forward(ctrl_params, sense));
            }
        }
        std::hint::black_box(actions);
    });
    let us = median_duration_us(&mut timings);

    println!("BENCH_SWARM_NN_20x50_RUST_US={us:.1}");

    BenchResult {
        name: "Swarm NN forward (20×50)".into(),
        kernel_tag: "SWARM_NN_20x50".into(),
        papers: "015".into(),
        rust_us: us,
        python_us: None,
        python_script: "control/swarm_robotics/bench_swarm_nn.py".into(),
    }
}

// ── Python execution ──────────────────────────────────────────────────

fn run_python_benchmarks(results: &mut [BenchResult]) {
    eprintln!("\n─── Python benchmarks (single-thread NumPy) ───");
    for r in results.iter_mut() {
        let tag = &r.kernel_tag;
        let script = &r.python_script;
        eprintln!("  Running {script} ...");

        let python = std::env::var("NEURALSPRING_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let output = Command::new(&python)
            .arg(script)
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .env("OMP_NUM_THREADS", "1")
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let reader = BufReader::new(&*o.stdout);
                for line in reader.lines().map_while(Result::ok) {
                    let prefix = format!("{tag}_US=");
                    if let Some(val) = line.strip_prefix(&prefix) {
                        if let Ok(us) = val.parse::<f64>() {
                            r.python_us = Some(us);
                            println!("BENCH_{tag}_PYTHON_US={us:.1}");
                        }
                    }
                }
                if r.python_us.is_none() {
                    eprintln!("    WARN: no machine-readable line found for {tag}");
                }
            }
            Ok(o) => {
                eprintln!("    FAIL: exit {}", o.status);
                let stderr = String::from_utf8_lossy(&o.stderr);
                for line in stderr.lines().take(5) {
                    eprintln!("      {line}");
                }
            }
            Err(e) => eprintln!("    SKIP: {e}"),
        }
    }
}

// ── Summary ───────────────────────────────────────────────────────────

fn print_summary(results: &[BenchResult]) {
    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  BENCHMARK RESULTS — Phase 0++ Pure Math Kernels                            ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!(
        "{:<38} {:>6} {:>12} {:>12} {:>10}",
        "Kernel", "Paper", "Rust µs", "Python µs", "Speedup"
    );
    eprintln!("{}", "─".repeat(80));

    for r in results {
        let py_str = r
            .python_us
            .map_or_else(|| "—".to_string(), |v| format!("{v:.1}"));
        let speedup = r
            .python_us
            .map_or_else(|| "—".to_string(), |py| format!("{:.1}×", py / r.rust_us));

        eprintln!(
            "{:<38} {:>6} {:>12.1} {:>12} {:>10}",
            r.name, r.papers, r.rust_us, py_str, speedup
        );
    }

    let total_rust: f64 = results.iter().map(|r| r.rust_us).sum();
    let total_py: Option<f64> = {
        let vals: Vec<f64> = results.iter().filter_map(|r| r.python_us).collect();
        if vals.len() == results.len() {
            Some(vals.iter().sum())
        } else {
            None
        }
    };
    let overall_speedup =
        total_py.map_or_else(|| "—".to_string(), |py| format!("{:.1}×", py / total_rust));

    eprintln!("{}", "─".repeat(80));
    eprintln!(
        "{:<38} {:>6} {:>12.1} {:>12} {:>10}",
        "TOTAL",
        "",
        total_rust,
        total_py.map_or_else(|| "—".to_string(), |v| format!("{v:.1}")),
        overall_speedup
    );
    eprintln!();

    if total_py.is_some() {
        eprintln!("Rust pure math is {overall_speedup} faster than single-thread NumPy.");
        eprintln!("Next: GPU WGSL shaders via metalForge/ for additional acceleration.");
    } else {
        eprintln!("Run with --with-python to get Python comparison timings.");
    }
}

// ── Harness ───────────────────────────────────────────────────────────

fn bench_kernel<F: FnMut()>(mut f: F) -> Vec<Duration> {
    for _ in 0..WARMUP {
        f();
    }
    let mut timings = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        f();
        timings.push(start.elapsed());
    }
    timings
}

// ── Matrix builders ───────────────────────────────────────────────────

fn make_stochastic_matrix(rows: usize, cols: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|_| {
            let raw: Vec<f64> = (0..cols).map(|_| rng.uniform() + 1e-6).collect();
            let sum: f64 = raw.iter().sum();
            raw.iter().map(|&v| v / sum).collect()
        })
        .collect()
}

fn make_stochastic_row(n: usize, rng: &mut Rng) -> Vec<f64> {
    let raw: Vec<f64> = (0..n).map(|_| rng.uniform() + 1e-6).collect();
    let sum: f64 = raw.iter().sum();
    raw.iter().map(|&v| v / sum).collect()
}
