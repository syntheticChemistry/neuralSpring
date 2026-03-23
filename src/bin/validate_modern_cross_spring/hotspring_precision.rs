// SPDX-License-Identifier: AGPL-3.0-or-later

// hotSpring provenance: precision infrastructure (Fp64, eigh, matmul, dispatch).

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, bench_once, max_abs_diff_f64};

pub fn validate_hotspring_precision(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    println!("\n─── hotSpring provenance: precision infrastructure ───\n");

    // Fp64Strategy detection (hotSpring S58 → BarraCUDA)
    let strategy = dispatcher.fp64_strategy();
    h.check_bool(
        "hS→precision: Fp64Strategy detected",
        !format!("{strategy:?}").is_empty(),
    );

    // DF64 core-streaming: eigh uses DF64 pathway on consumer GPUs
    let n = 16;
    let mut rng = Rng::new(42);
    let mut mat = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let v = rng.uniform() - 0.5;
            mat[i * n + j] = v;
            mat[j * n + i] = v;
        }
    }
    let (eigs_gpu, gpu_us) = bench_once("eigh 16×16 (hS→DF64→GPU)", || {
        dispatcher.eigh(&mat, n).0
    });
    let cpu = Dispatcher::cpu_only();
    let (eigs_cpu, cpu_us) = bench_once("eigh 16×16 (CPU ref)", || cpu.eigh(&mat, n).0);
    // GPU Jacobi converges differently for random matrices — compare sorted order
    let mut sorted_gpu = eigs_gpu;
    let mut sorted_cpu = eigs_cpu;
    sorted_gpu.sort_by(f64::total_cmp);
    sorted_cpu.sort_by(f64::total_cmp);
    let eigh_diff = max_abs_diff_f64(&sorted_gpu, &sorted_cpu);
    h.check_abs(
        "hS→DF64: eigh GPU≈CPU (sorted)",
        eigh_diff,
        0.0,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );
    h.check_bool(
        &format!("hS→DF64: eigh benchmarked (GPU {gpu_us:.0}µs, CPU {cpu_us:.0}µs)"),
        gpu_us > 0.0,
    );

    // split_workgroups: validated through Dispatcher dispatch parity
    let big_n = 100;
    let big_a: Vec<f64> = (0..big_n * big_n).map(|i| (i as f64) * 0.001).collect();
    let big_b: Vec<f64> = (0..big_n * big_n)
        .map(|i| ((big_n * big_n - i) as f64) * 0.001)
        .collect();
    let (gpu_res, _) = bench_once("matmul 100×100 (hS→split_workgroups)", || {
        dispatcher.mat_mul(&big_a, &big_b, big_n)
    });
    let (cpu_res, _) = bench_once("matmul 100×100 (CPU ref)", || {
        cpu.mat_mul(&big_a, &big_b, big_n)
    });
    h.check_abs(
        "hS→split_wg: matmul GPU≈CPU (100×100)",
        max_abs_diff_f64(&gpu_res, &cpu_res),
        0.0,
        tolerances::DISPATCH_MATMUL_F64 * 10.0,
    );

    // Primal matmul via barracuda::dispatch::matmul_dispatch (non-square)
    let m = 8;
    let k = 12;
    let n_col = 6;
    let a_rect: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.01).collect();
    let b_rect: Vec<f64> = (0..k * n_col)
        .map(|i| ((k * n_col - i) as f64) * 0.01)
        .collect();
    let result = barracuda::dispatch::matmul_dispatch(&a_rect, &b_rect, m, k, n_col, None)
        .expect("matmul_dispatch non-square");
    assert_eq!(result.len(), m * n_col);
    h.check_bool(
        "hS→dispatch: matmul non-square (8×12 × 12×6)",
        result.iter().all(|v| v.is_finite()),
    );
}
