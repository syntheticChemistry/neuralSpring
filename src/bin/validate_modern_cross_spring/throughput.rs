// SPDX-License-Identifier: AGPL-3.0-or-later

// Cross-spring throughput benchmark table (GPU vs CPU dispatcher).

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::rng::Rng;
use neural_spring::validation::{ValidationHarness, bench_once};

fn bench_pair(label: &str, gpu_fn: impl FnOnce(), cpu_fn: impl FnOnce()) -> (f64, f64) {
    let ((), gpu_t) = bench_once(&format!("{label} GPU"), gpu_fn);
    let ((), cpu_t) = bench_once(&format!("{label} CPU"), cpu_fn);
    (gpu_t, cpu_t)
}

fn print_bench_row(label: &str, prov: &str, gpu_us: f64, cpu_us: f64) {
    let ratio = if cpu_us > 0.0 {
        gpu_us / cpu_us
    } else {
        f64::NAN
    };
    println!("  │ {label:<15} │ {prov:<10} │ {gpu_us:>8.1} │ {cpu_us:>8.1} │ {ratio:>7.2}× │");
}

pub fn benchmark_cross_spring_throughput(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    println!("\n─── Cross-spring throughput benchmark ───\n");

    let cpu = Dispatcher::cpu_only();
    let n = 256;
    let mut rng = Rng::new(99);
    let data: Vec<f64> = (0..n * n).map(|_| rng.uniform()).collect();
    let flat: Vec<f64> = data[..1024].to_vec();

    let n_eig = 32;
    let mut sym: Vec<f64> = (0..n_eig * n_eig).map(|_| rng.uniform() - 0.5).collect();
    for i in 0..n_eig {
        for j in (i + 1)..n_eig {
            sym[j * n_eig + i] = sym[i * n_eig + j];
        }
    }

    let mut total_gpu = 0.0_f64;
    let mut total_cpu = 0.0_f64;
    let mut count = 0_usize;

    println!("  ┌─────────────────┬────────────┬──────────┬──────────┬──────────┐");
    println!("  │ Operation       │ Provenance │  GPU µs  │  CPU µs  │ Ratio    │");
    println!("  ├─────────────────┼────────────┼──────────┼──────────┼──────────┤");

    macro_rules! bench_row {
        ($label:expr, $prov:expr, $gpu:expr, $cpu:expr) => {{
            let (g, c) = bench_pair(
                $label,
                || {
                    let _ = $gpu;
                },
                || {
                    let _ = $cpu;
                },
            );
            print_bench_row($label, $prov, g, c);
            total_gpu += g;
            total_cpu += c;
            count += 1;
        }};
    }

    bench_row!(
        "matmul 256²",
        "nS→TS",
        dispatcher.mat_mul(&data, &data, n),
        cpu.mat_mul(&data, &data, n)
    );
    bench_row!(
        "softmax 1K",
        "nS→TS",
        dispatcher.softmax(&flat),
        cpu.softmax(&flat)
    );
    bench_row!(
        "variance 65K",
        "hS+nS→TS",
        dispatcher.variance(&data),
        cpu.variance(&data)
    );
    bench_row!("GELU 1K", "nS→TS", dispatcher.gelu(&flat), cpu.gelu(&flat));
    bench_row!(
        "eigh 32²",
        "hS→TS",
        dispatcher.eigh(&sym, n_eig),
        cpu.eigh(&sym, n_eig)
    );
    bench_row!(
        "frobenius 65K",
        "nS→TS",
        dispatcher.frobenius_norm(&data),
        cpu.frobenius_norm(&data)
    );

    println!("  └─────────────────┴────────────┴──────────┴──────────┴──────────┘");

    h.check_bool(
        &format!("bench: {count}/6 ops timed (total GPU {total_gpu:.0}µs, CPU {total_cpu:.0}µs)"),
        count == 6,
    );
}
