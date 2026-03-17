// SPDX-License-Identifier: AGPL-3.0-or-later

//! baseCamp CPU / GPU parity benchmark.
//!
//! Demonstrates that baseCamp science modules produce identical results via:
//! 1. **Pure Rust CPU** — local math + `barracuda::stats` / `barracuda::special`
//! 2. **`BarraCUDA` GPU** — typed f64 ops (variance, correlation, entropy)
//!
//! Shows `BarraCUDA` CPU is pure math (no interpreter), and GPU provides
//! hardware-portable execution with streaming dispatch.

#![expect(
    clippy::cast_precision_loss,
    clippy::similar_names,
    reason = "validation binary"
)]

use neural_spring::eigh::eigh_householder_qr;
use neural_spring::gpu::Gpu;
use neural_spring::information_flow;
use neural_spring::loss_landscape;
use neural_spring::neural_pgm;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::weight_spectral;
use std::sync::Arc;
use std::time::Instant;

const WARMUP: usize = 3;
const ITERS: usize = 20;

type BenchResult<T> = Result<T, Box<dyn std::error::Error>>;

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        println!("FAIL: {e}");
        std::process::exit(1);
    }
}

async fn run() -> BenchResult<()> {
    let gpu = Gpu::new().await.ok();
    let dev = gpu.as_ref().map(|g| {
        println!(
            "GPU: {} ({:?}, {:?})",
            g.adapter_name, g.device_type, g.backend
        );
        Arc::clone(g.wgpu_device())
    });

    let mut rng = Rng::new(42);

    println!("\n=== baseCamp CPU/GPU Parity Benchmark ===\n");
    println!("--- baseCamp Module Benchmarks (pure Rust CPU) ---\n");

    // ── Sub-thesis 01: Weight Spectral ──────────────────────────────

    let ws_n = 16;
    let ws_w: Vec<f64> = (0..ws_n * ws_n).map(|_| rng.normal()).collect();

    let (cpu_us, result) = bench("Sub-01 weight_spectral (16×16 eigh+IPR+LSR)", || {
        weight_spectral::weight_spectral_analysis(&ws_w, ws_n, ws_n)
    });
    println!(
        "  Sub-01 weight_spectral : {cpu_us:8.1} µs  IPR={:.6}, LSR={:.4}",
        result.mean_ipr, result.level_spacing_ratio
    );

    // ── Sub-thesis 02: Information Flow ─────────────────────────────

    let layers = 20;
    let decaying_var: Vec<f64> = (0..layers).map(|i| (-0.3 * f64::from(i)).exp()).collect();
    let gate_vals: Vec<f64> = (0..200).map(|_| rng.uniform()).collect();
    let attn: Vec<f64> = (0..8 * 8).map(|_| rng.uniform()).collect();

    let (cpu_us, (xi, w_dis, attn_ipr)) = bench("Sub-02 information_flow", || {
        let xi = information_flow::depth_scale(&decaying_var);
        let w = information_flow::gate_disorder_parameter(&gate_vals);
        let a = information_flow::attention_spectral_analysis(&attn, 8);
        (xi, w, a.mean_ipr)
    });
    println!(
        "  Sub-02 info_flow       : {cpu_us:8.1} µs  ξ={xi:.3}, W={w_dis:.4}, attn_IPR={attn_ipr:.4}"
    );

    // ── Sub-thesis 03: Loss Landscape ───────────────────────────────

    let ll_dim = 8;
    let ll_params: Vec<f64> = (0..ll_dim).map(|_| rng.normal()).collect();
    let quadratic = |x: &[f64]| -> f64 { x.iter().map(|&v| v * v).sum() };

    let (cpu_us, r) = bench("Sub-03 loss_landscape (8-dim Hessian)", || {
        loss_landscape::landscape_analysis(&quadratic, &ll_params, tolerances::HESSIAN_FD_STEP, 0.1)
    });
    println!(
        "  Sub-03 loss_landscape  : {cpu_us:8.1} µs  saddle={}, sharpness={:.3}",
        r.saddle_index, r.sharpness
    );

    // ── Sub-thesis 04: Neural PGM ───────────────────────────────────

    let pgm_in_raw: Vec<f64> = (0..8).map(|_| rng.uniform()).collect();
    let pgm_sum: f64 = pgm_in_raw.iter().sum();
    let pgm_in: Vec<f64> = pgm_in_raw.iter().map(|&v| v / pgm_sum).collect();
    let pgm_w1: Vec<f64> = (0..64).map(|_| rng.normal()).collect();
    let pgm_t1 = neural_pgm::weight_to_transition(&pgm_w1, 8, 8);
    let pgm_w2: Vec<f64> = (0..32).map(|_| rng.normal()).collect();
    let pgm_t2 = neural_pgm::weight_to_transition(&pgm_w2, 8, 4);

    let (cpu_us, dists) = bench("Sub-04 neural_pgm (2-layer BP)", || {
        neural_pgm::belief_propagation_chain(
            &pgm_in,
            &[pgm_t1.as_slice(), pgm_t2.as_slice()],
            &[8, 4],
        )
    });
    let final_sum: f64 = dists.last().map_or(0.0, |d| d.iter().sum());
    println!(
        "  Sub-04 neural_pgm      : {cpu_us:8.1} µs  layers={}, Σ={final_sum:.6}",
        dists.len()
    );

    // ── Sub-thesis 05: Agent Coordination ───────────────────────────

    let agents =
        neural_spring::agent_coordination::generate_lattice_agents(4, 2, 0.1, &mut Rng::new(42));
    let n_agents = agents.len();
    let adj = neural_spring::agent_coordination::interaction_graph(&agents, 2.0);
    let lap = neural_spring::agent_coordination::graph_laplacian(&adj, n_agents);

    let (cpu_us, decomp) = bench("Sub-05 agent_coordination (16-node Laplacian)", || {
        eigh_householder_qr(&lap, n_agents)
    });
    let lambda_2 = decomp.eigenvalues.get(1).copied().unwrap_or(0.0);
    println!("  Sub-05 agent_coord     : {cpu_us:8.1} µs  λ₂={lambda_2:.6}");

    // ── BarraCUDA CPU parity: stats + special ────────────────────────

    println!("\n--- BarraCUDA CPU Parity (pure Rust, no interpreter) ---\n");

    let data: Vec<f64> = (0..1000).map(|_| rng.normal()).collect();
    let data2: Vec<f64> = (0..1000).map(|_| rng.normal()).collect();

    let bc_pearson =
        barracuda::stats::correlation::pearson_correlation(&data, &data2).unwrap_or(0.0);
    println!("  pearson(bC CPU)       : {bc_pearson:.12}");

    let obs = vec![10.0, 20.0, 30.0, 40.0];
    let exp_v = vec![25.0, 25.0, 25.0, 25.0];
    let bc_chi2 = barracuda::special::chi_squared_statistic(&obs, &exp_v).unwrap_or(0.0);
    println!("  chi²(bC CPU)          : {bc_chi2:.12}");

    let bc_gamma = barracuda::special::gamma(5.0).unwrap_or(f64::NAN);
    println!("  Γ(5) = 4! = 24        : {bc_gamma:.12}  (bC special::gamma)");

    let bc_erf = barracuda::special::erf(1.0);
    println!("  erf(1)                : {bc_erf:.12}  (bC special::erf)");

    let bc_bessel = barracuda::special::bessel_j0(0.0);
    println!("  J₀(0) = 1             : {bc_bessel:.12}  (bC special::bessel_j0)");

    // ── BarraCUDA GPU streaming parity ───────────────────────────────

    if let Some(ref d) = dev {
        println!("\n--- BarraCUDA GPU Streaming Parity ---\n");

        bench_gpu_streaming_parity(d, &data, &data2, bc_pearson)?;
    }

    println!("\n=== baseCamp parity complete: pure Rust → GPU portable ===");
    Ok(())
}

fn bench_gpu_streaming_parity(
    d: &Arc<barracuda::device::WgpuDevice>,
    data: &[f64],
    data2: &[f64],
    bc_pearson: f64,
) -> BenchResult<()> {
    let var_op = barracuda::ops::variance_f64_wgsl::VarianceF64::new(d.clone())
        .map_err(|e| format!("VarianceF64 init: {e}"))?;
    let gpu_var = var_op
        .variance(data)
        .map_err(|e| format!("GPU variance: {e}"))?;
    let local_var: f64 = {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n
    };
    let var_diff = (local_var - gpu_var).abs();
    println!("  variance CPU(pop)     : {local_var:.10}");
    println!("  variance GPU(Welford) : {gpu_var:.10}  (diff: {var_diff:.2e})");

    let corr_op = barracuda::ops::correlation_f64_wgsl::CorrelationF64::new(d.clone())
        .map_err(|e| format!("CorrelationF64 init: {e}"))?;
    let gpu_pearson = corr_op.correlation(data, data2).unwrap_or(0.0);
    let pearson_diff = (bc_pearson - gpu_pearson).abs();
    println!("  pearson CPU(bC)       : {bc_pearson:.10}");
    println!("  pearson GPU(f64)      : {gpu_pearson:.10}  (diff: {pearson_diff:.2e})");

    let entropy_op = barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64::new(d.clone())
        .map_err(|e| format!("FusedMapReduceF64 init: {e}"))?;
    let probs: Vec<f64> = {
        let raw: Vec<f64> = data.iter().map(|&x| x.abs().max(1e-12)).collect();
        let sum: f64 = raw.iter().sum();
        raw.iter().map(|&r| r / sum).collect()
    };
    let cpu_entropy = neural_spring::primitives::shannon_entropy(&probs);
    let gpu_entropy = entropy_op
        .shannon_entropy(&probs)
        .map_err(|e| format!("GPU entropy: {e}"))?;
    let entropy_diff = (cpu_entropy - gpu_entropy).abs();
    println!("  entropy CPU           : {cpu_entropy:.10}");
    println!("  entropy GPU(fused)    : {gpu_entropy:.10}  (diff: {entropy_diff:.2e})");
    Ok(())
}

fn bench<F, R>(_name: &str, f: F) -> (f64, R)
where
    F: Fn() -> R,
{
    for _ in 0..WARMUP {
        let _ = std::hint::black_box(f());
    }
    let start = Instant::now();
    let mut result = f();
    for _ in 1..ITERS {
        result = f();
    }
    (start.elapsed().as_micros() as f64 / ITERS as f64, result)
}
