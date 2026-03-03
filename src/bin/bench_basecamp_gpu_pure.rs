// SPDX-License-Identifier: AGPL-3.0-or-later

//! Benchmark baseCamp GPU vs CPU paths for all 5 sub-theses.
//!
//! Measures wall-clock time for each sub-thesis core computation on
//! both GPU (via `Dispatcher::from_gpu`) and CPU (via `Dispatcher::cpu_only`),
//! reporting the speedup factor.

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::primitives::PROBABILITY_FLOOR;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use std::time::{Duration, Instant};

const WARMUP: usize = 3;
const ITERS: usize = 20;

fn median(times: &mut [Duration]) -> Duration {
    times.sort();
    times[times.len() / 2]
}

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(e) => {
            eprintln!("No GPU: {e}");
            std::process::exit(0);
        }
    };

    let gpu_disp = Dispatcher::from_gpu(gpu);
    let cpu_disp = Dispatcher::cpu_only();

    eprintln!("\n=== baseCamp GPU vs CPU Benchmark ===\n");
    eprintln!(
        "{:<45} {:>10} {:>10} {:>8}",
        "Operation", "CPU µs", "GPU µs", "Speedup"
    );
    eprintln!("{}", "-".repeat(78));

    bench_sub01(&gpu_disp, &cpu_disp);
    bench_sub02(&gpu_disp, &cpu_disp);
    bench_sub03(&gpu_disp, &cpu_disp);
    bench_sub04(&gpu_disp, &cpu_disp);
    bench_sub05(&gpu_disp, &cpu_disp);

    eprintln!();
}

fn report(name: &str, cpu_times: &mut [Duration], gpu_times: &mut [Duration]) {
    let cpu_us = median(cpu_times).as_micros();
    let gpu_us = median(gpu_times).as_micros();
    let speedup = if gpu_us > 0 {
        cpu_us as f64 / gpu_us as f64
    } else {
        f64::INFINITY
    };
    eprintln!("{name:<45} {cpu_us:>10} {gpu_us:>10} {speedup:>7.2}×");
}

fn bench_sub01(gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(101);
    let m = 32;
    let n = 32;
    let weights: Vec<f64> = (0..m * n).map(|_| rng.normal()).collect();

    let mut cpu_times = Vec::with_capacity(ITERS);
    let mut gpu_times = Vec::with_capacity(ITERS);

    for i in 0..(WARMUP + ITERS) {
        let t = Instant::now();
        let _ = cpu.weight_spectral_analysis(&weights, m, n);
        if i >= WARMUP {
            cpu_times.push(t.elapsed());
        }
    }
    for i in 0..(WARMUP + ITERS) {
        let t = Instant::now();
        let _ = gpu.weight_spectral_analysis(&weights, m, n);
        if i >= WARMUP {
            gpu_times.push(t.elapsed());
        }
    }
    report(
        "Sub-01: weight spectral (32×32)",
        &mut cpu_times,
        &mut gpu_times,
    );
}

fn bench_sub02(gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(202);
    let n = 16;
    let attention: Vec<f64> = (0..n * n).map(|_| rng.uniform()).collect();

    let mut cpu_times = Vec::with_capacity(ITERS);
    let mut gpu_times = Vec::with_capacity(ITERS);

    for i in 0..(WARMUP + ITERS) {
        let t = Instant::now();
        let _ = neural_spring::information_flow::attention_spectral_analysis(&attention, n);
        if i >= WARMUP {
            cpu_times.push(t.elapsed());
        }
    }
    for i in 0..(WARMUP + ITERS) {
        let t = Instant::now();
        let _ = gpu.attention_spectral_analysis(&attention, n);
        if i >= WARMUP {
            gpu_times.push(t.elapsed());
        }
    }
    report(
        "Sub-02: attention spectral (16×16)",
        &mut cpu_times,
        &mut gpu_times,
    );

    let input: Vec<f64> = (0..16).map(|_| rng.normal()).collect();
    let w1: Vec<f64> = (0..16 * 8).map(|_| rng.normal() * 0.5).collect();
    let w2: Vec<f64> = (0..8 * 4).map(|_| rng.normal() * 0.5).collect();

    let mut cpu_t2 = Vec::with_capacity(ITERS);
    let mut gpu_t2 = Vec::with_capacity(ITERS);

    for i in 0..(WARMUP + ITERS) {
        let t = Instant::now();
        let _ = cpu.mlp_signal_propagation(&input, &[&w1, &w2], &[8, 4]);
        if i >= WARMUP {
            cpu_t2.push(t.elapsed());
        }
    }
    for i in 0..(WARMUP + ITERS) {
        let t = Instant::now();
        let _ = gpu.mlp_signal_propagation(&input, &[&w1, &w2], &[8, 4]);
        if i >= WARMUP {
            gpu_t2.push(t.elapsed());
        }
    }
    report(
        "Sub-02: signal propagation (16→8→4)",
        &mut cpu_t2,
        &mut gpu_t2,
    );
}

fn bench_sub03(gpu: &Dispatcher, cpu: &Dispatcher) {
    fn quadratic(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }
    let params = vec![0.5, -0.3, 0.8, 0.1, 0.6, -0.2, 0.4, 0.9];

    let mut cpu_times = Vec::with_capacity(ITERS);
    let mut gpu_times = Vec::with_capacity(ITERS);

    for i in 0..(WARMUP + ITERS) {
        let t = Instant::now();
        let _ = cpu.landscape_analysis(&quadratic, &params, tolerances::HESSIAN_FD_STEP, 0.1);
        if i >= WARMUP {
            cpu_times.push(t.elapsed());
        }
    }
    for i in 0..(WARMUP + ITERS) {
        let t = Instant::now();
        let _ = gpu.landscape_analysis(&quadratic, &params, tolerances::HESSIAN_FD_STEP, 0.1);
        if i >= WARMUP {
            gpu_times.push(t.elapsed());
        }
    }
    report(
        "Sub-03: landscape analysis (8-dim)",
        &mut cpu_times,
        &mut gpu_times,
    );
}

fn bench_sub04(gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(404);
    let n1 = 16;
    let n2 = 8;
    let n3 = 4;
    let t1 = make_stochastic(n1, n2, &mut rng);
    let t2 = make_stochastic(n2, n3, &mut rng);
    let input: Vec<f64> = {
        let raw: Vec<f64> = (0..n1)
            .map(|_| rng.uniform().max(PROBABILITY_FLOOR))
            .collect();
        let s: f64 = raw.iter().sum();
        raw.iter().map(|&v| v / s).collect()
    };

    let mut cpu_times = Vec::with_capacity(ITERS);
    let mut gpu_times = Vec::with_capacity(ITERS);

    for i in 0..(WARMUP + ITERS) {
        let t = Instant::now();
        let _ = cpu.belief_propagation(&input, &[&t1, &t2], &[n2, n3]);
        if i >= WARMUP {
            cpu_times.push(t.elapsed());
        }
    }
    for i in 0..(WARMUP + ITERS) {
        let t = Instant::now();
        let _ = gpu.belief_propagation(&input, &[&t1, &t2], &[n2, n3]);
        if i >= WARMUP {
            gpu_times.push(t.elapsed());
        }
    }
    report(
        "Sub-04: belief propagation (16→8→4)",
        &mut cpu_times,
        &mut gpu_times,
    );
}

fn bench_sub05(gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(505);
    let n_agents = 32;
    let dim = 3;
    let comm_range = 5.0;
    let positions: Vec<f64> = (0..n_agents * dim).map(|_| rng.uniform() * 10.0).collect();

    let mut cpu_times = Vec::with_capacity(ITERS);
    let mut gpu_times = Vec::with_capacity(ITERS);

    for i in 0..(WARMUP + ITERS) {
        let t = Instant::now();
        let _ = cpu.agent_interaction_graph(&positions, n_agents, dim, comm_range);
        if i >= WARMUP {
            cpu_times.push(t.elapsed());
        }
    }
    for i in 0..(WARMUP + ITERS) {
        let t = Instant::now();
        let _ = gpu.agent_interaction_graph(&positions, n_agents, dim, comm_range);
        if i >= WARMUP {
            gpu_times.push(t.elapsed());
        }
    }
    report(
        "Sub-05: agent interaction (32 agents, 3D)",
        &mut cpu_times,
        &mut gpu_times,
    );
}

fn make_stochastic(rows: usize, cols: usize, rng: &mut Rng) -> Vec<f64> {
    let mut mat = vec![0.0; rows * cols];
    for i in 0..rows {
        let mut row_sum = 0.0;
        for j in 0..cols {
            let v = rng.uniform().max(PROBABILITY_FLOOR);
            mat[i * cols + j] = v;
            row_sum += v;
        }
        for j in 0..cols {
            mat[i * cols + j] /= row_sum;
        }
    }
    mat
}
