// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-spring throughput micro-benchmarks (upstream vs CPU reference).

use crate::helpers::gen_f64_vec;
use neural_spring::gpu_dispatch::Dispatcher;
use std::time::Instant;

pub const fn benchmark_s72_throughput(_dispatcher: &Dispatcher, _cpu: &Dispatcher) {}

#[expect(clippy::too_many_lines, reason = "validation binary")]
pub fn benchmark_throughput(dispatcher: &Dispatcher, cpu: &Dispatcher) {
    println!("\n=== Cross-Spring Throughput Benchmark ===");
    println!("(upstream dispatch includes GPU routing + size-based thresholds)\n");

    let sizes: [u32; 4] = [64, 256, 1024, 4096];
    for sz in sizes {
        let data = gen_f64_vec(sz as usize, 0.001);

        let start = Instant::now();
        for _ in 0..100 {
            std::hint::black_box(dispatcher.mean(&data));
        }
        let upstream_us = start.elapsed().as_secs_f64() * 1e4;

        let start = Instant::now();
        for _ in 0..100 {
            std::hint::black_box(cpu.mean(&data));
        }
        let cpu_us = start.elapsed().as_secs_f64() * 1e4;

        let ratio = cpu_us / upstream_us;
        println!(
            "  mean(n={sz:>5}): upstream {upstream_us:>8.1}µs  cpu {cpu_us:>8.1}µs  ratio {ratio:.2}x"
        );
    }

    let mat_sizes: [usize; 4] = [16, 32, 64, 128];
    for n in mat_sizes {
        let a = gen_f64_vec(n * n, 0.001);
        let b: Vec<f64> = (0..n * n).map(|i| (n * n - i) as f64 * 0.001).collect();

        let start = Instant::now();
        for _ in 0..10 {
            std::hint::black_box(dispatcher.mat_mul(&a, &b, n));
        }
        let upstream_us = start.elapsed().as_secs_f64() * 1e5;

        let start = Instant::now();
        for _ in 0..10 {
            std::hint::black_box(cpu.mat_mul(&a, &b, n));
        }
        let cpu_us = start.elapsed().as_secs_f64() * 1e5;

        let ratio = cpu_us / upstream_us;
        println!(
            "  matmul({n:>3}x{n:>3}): upstream {upstream_us:>8.1}µs  cpu {cpu_us:>8.1}µs  ratio {ratio:.2}x"
        );
    }

    println!();
    println!("--- S59 Rewired Ops Throughput ---\n");

    for sz in [64_i32, 256, 1024, 4096] {
        let data: Vec<f64> = (-50..(-50 + sz)).map(|i| f64::from(i) * 0.01).collect();

        let start = Instant::now();
        for _ in 0..100 {
            std::hint::black_box(dispatcher.gelu(&data));
        }
        let upstream_us = start.elapsed().as_secs_f64() * 1e4;

        let start = Instant::now();
        for _ in 0..100 {
            std::hint::black_box(cpu.gelu(&data));
        }
        let cpu_us = start.elapsed().as_secs_f64() * 1e4;

        let ratio = cpu_us / upstream_us;
        println!(
            "  gelu(n={sz:>5}): upstream {upstream_us:>8.1}µs  cpu {cpu_us:>8.1}µs  ratio {ratio:.2}x"
        );
    }

    for n_states in [3, 8, 16, 32_usize] {
        let alpha: Vec<f64> = (0..n_states).map(|i| (i + 1) as f64).collect();
        let sum: f64 = alpha.iter().sum();
        let alpha: Vec<f64> = alpha.iter().map(|x| x / sum).collect();
        let transition: Vec<f64> = (0..n_states * n_states)
            .map(|i| ((i % n_states) + 1) as f64 / (n_states * (n_states + 1) / 2) as f64)
            .collect();
        let emission: Vec<f64> = vec![1.0 / n_states as f64; n_states];

        let start = Instant::now();
        for _ in 0..100 {
            std::hint::black_box(dispatcher.hmm_forward_step(
                &alpha,
                &transition,
                &emission,
                n_states,
            ));
        }
        let upstream_us = start.elapsed().as_secs_f64() * 1e4;

        let start = Instant::now();
        for _ in 0..100 {
            std::hint::black_box(cpu.hmm_forward_step(&alpha, &transition, &emission, n_states));
        }
        let cpu_us = start.elapsed().as_secs_f64() * 1e4;

        let ratio = cpu_us / upstream_us;
        println!(
            "  hmm_fwd(s={n_states:>3}): upstream {upstream_us:>8.1}µs  cpu {cpu_us:>8.1}µs  ratio {ratio:.2}x"
        );
    }

    println!();
    println!("--- S72 Rewired Ops Throughput ---\n");

    for &(rows, cols) in &[(4, 64), (16, 128), (64, 256)] {
        let matrix: Vec<f64> = (0..rows * cols)
            .map(|i| (i as f64 - 128.0) * 0.01)
            .collect();

        let start = Instant::now();
        for _ in 0..100 {
            std::hint::black_box(dispatcher.softmax_row_wise(&matrix, rows, cols));
        }
        let upstream_us = start.elapsed().as_secs_f64() * 1e4;

        let start = Instant::now();
        for _ in 0..100 {
            std::hint::black_box(cpu.softmax_row_wise(&matrix, rows, cols));
        }
        let cpu_us = start.elapsed().as_secs_f64() * 1e4;

        let ratio = cpu_us / upstream_us;
        println!(
            "  softmax_row({rows:>3}x{cols:>3}): upstream {upstream_us:>8.1}\u{00b5}s  cpu {cpu_us:>8.1}\u{00b5}s  ratio {ratio:.2}x"
        );
    }

    for n_states in [3, 8, 16, 32_usize] {
        let n_obs = 2;
        let initial: Vec<f64> = {
            let raw: Vec<f64> = (0..n_states).map(|i| (i + 1) as f64).collect();
            let s: f64 = raw.iter().sum();
            raw.iter().map(|x| x / s).collect()
        };
        let transition: Vec<f64> = (0..n_states * n_states)
            .map(|i| ((i % n_states) + 1) as f64 / (n_states * (n_states + 1) / 2) as f64)
            .collect();
        let emission: Vec<f64> = (0..n_states * n_obs)
            .map(|i| ((i % n_obs) + 1) as f64 / (n_obs * (n_obs + 1) / 2) as f64)
            .collect();
        let obs: Vec<usize> = (0..10).map(|i| i % n_obs).collect();

        let start = Instant::now();
        for _ in 0..20 {
            std::hint::black_box(dispatcher.hmm_viterbi_chain(
                &initial,
                &transition,
                &emission,
                &obs,
                n_states,
                n_obs,
            ));
        }
        let upstream_us = start.elapsed().as_secs_f64() * 5e4;

        let start = Instant::now();
        for _ in 0..20 {
            std::hint::black_box(cpu.hmm_viterbi_chain(
                &initial,
                &transition,
                &emission,
                &obs,
                n_states,
                n_obs,
            ));
        }
        let cpu_us = start.elapsed().as_secs_f64() * 5e4;

        let ratio = cpu_us / upstream_us;
        println!(
            "  viterbi(s={n_states:>3}): upstream {upstream_us:>8.1}\u{00b5}s  cpu {cpu_us:>8.1}\u{00b5}s  ratio {ratio:.2}x"
        );
    }
}
