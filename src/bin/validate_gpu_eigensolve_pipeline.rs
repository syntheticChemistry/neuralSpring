// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exp 103: GPU-accelerated eigensolve pipeline via `BatchedEighGpu` on RTX 4070.
//!
//! Compares GPU eigensolve (through `Dispatcher`) against CPU `eigh_householder_qr`
//! for a sweep of matrix sizes, measuring wall time and precision agreement.

#![expect(clippy::expect_used, reason = "binary entry point")]

use neural_spring::gpu_dispatch::Dispatcher;
use std::time::Instant;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let dispatcher = rt.block_on(Dispatcher::new());

    println!("═══ Exp 103: GPU-Accelerated Eigensolve Pipeline ═══");
    println!("Backend: {}", dispatcher.backend());
    println!("Adapter: {}", dispatcher.adapter_name());
    if let Some(profile) = dispatcher.driver_profile() {
        println!("FP64 strategy: {:?}", profile.fp64_strategy());
    }
    println!();

    let sizes = [8, 16, 32, 64];
    let mut all_pass = true;

    for &n in &sizes {
        let mut matrix = vec![0.0_f64; n * n];
        let mut rng = neural_spring::rng::Rng::new(42);
        for r in 0..n {
            for c in r..n {
                let v = rng.uniform().mul_add(2.0, -1.0);
                matrix[r * n + c] = v;
                matrix[c * n + r] = v;
            }
        }

        let cpu_start = Instant::now();
        let cpu_result = neural_spring::eigh::eigh_householder_qr(&matrix, n);
        let cpu_us = cpu_start.elapsed().as_secs_f64() * 1_000_000.0;

        let gpu_start = Instant::now();
        let (gpu_evals, _) = dispatcher.eigh(&matrix, n);
        let gpu_us = gpu_start.elapsed().as_secs_f64() * 1_000_000.0;

        let mut cpu_sorted = cpu_result.eigenvalues.clone();
        let mut gpu_sorted = gpu_evals.clone();
        cpu_sorted.sort_by(f64::total_cmp);
        gpu_sorted.sort_by(f64::total_cmp);

        let max_diff = cpu_sorted
            .iter()
            .zip(gpu_sorted.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        let pass = max_diff < 1e-6;
        if !pass {
            all_pass = false;
        }

        let speedup = cpu_us / gpu_us.max(0.001);
        println!(
            "  N={n:3}: CPU={cpu_us:8.1}µs  GPU={gpu_us:8.1}µs  speedup={speedup:5.2}×  max_diff={max_diff:.2e}  {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    println!();
    if all_pass {
        println!("✓ All eigensolve precision checks passed");
    } else {
        println!("✗ Some eigensolve checks failed");
        std::process::exit(1);
    }
}
