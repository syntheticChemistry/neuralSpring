// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation: `metalForge` mixed-hardware dispatch parity.
//!
//! Proves that the `Dispatcher::mixed_dispatch()` wiring correctly routes
//! workloads across GPU, CPU, and (simulated) NPU paths, producing
//! identical results regardless of substrate. Tests the full stack:
//!
//! 1. `metalForge::dispatch` → substrate heuristics
//! 2. `metalForge::mixed` → cross-device cost model
//! 3. `metalForge::pcie_bridge` → transfer cost estimation
//! 4. `Dispatcher::mixed_dispatch()` → end-to-end routing
//!
//! This validator is the "mixed-hardware portability proof" for `ToadStool`.

#![expect(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "validation binary"
)]

use neural_spring::gpu_dispatch::{Dispatcher, MixedWorkload};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring_forge::mixed::MixedSubstrate;
use neural_spring_forge::pcie_bridge::PcieBridge;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("mixed_hardware");
    let mut rng = Rng::new(42);
    let dispatcher = Dispatcher::new().await;

    // ═══════════════════════════════════════════════════════════════════
    // 1. Small workload → CPU substrate (no GPU dispatch overhead)
    // ═══════════════════════════════════════════════════════════════════

    let small: Vec<f64> = (0..32).map(|_| rng.normal()).collect();
    let (small_var, small_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "small_variance",
            compute_us: 10.0, // 10 µs compute — well below GPU overhead
            data_bytes: 256,  // 256 bytes
            npu_available: false,
            needs_realtime: false, // no NPU, not realtime
        },
        |dev| {
            barracuda::ops::variance_reduce_f64::VarianceReduceF64::population_variance(
                dev.clone(),
                &small,
            )
            .map_err(|e| format!("{e}"))
        },
        || {
            let n = small.len() as f64;
            let m = small.iter().sum::<f64>() / n;
            small.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n
        },
    );

    h.check_bool(
        "small workload routed to CPU",
        small_sub == MixedSubstrate::CpuOnly,
    );
    h.check_bool("small variance is finite", small_var.is_finite());

    // ═══════════════════════════════════════════════════════════════════
    // 2. Large workload → GPU substrate (compute dominates transfer)
    // ═══════════════════════════════════════════════════════════════════

    let large: Vec<f64> = (0..4096).map(|_| rng.normal()).collect();
    let large_bytes = (large.len() * 8) as u64;

    let cpu_var = {
        let n = large.len() as f64;
        let m = large.iter().sum::<f64>() / n;
        large.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n
    };

    let (gpu_var, large_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "large_variance",
            compute_us: 50_000.0, // 50 ms compute — exceeds GPU overhead + transfer
            data_bytes: large_bytes,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| {
            barracuda::ops::variance_reduce_f64::VarianceReduceF64::population_variance(
                dev.clone(),
                &large,
            )
            .map_err(|e| format!("{e}"))
        },
        || cpu_var,
    );

    if dispatcher.has_gpu() {
        h.check_bool(
            "large workload routed to GPU",
            large_sub == MixedSubstrate::GpuOnly,
        );
        h.check_abs(
            "large variance CPU vs GPU parity",
            gpu_var,
            cpu_var,
            tolerances::GPU_VARIANCE_F64,
        );
    } else {
        h.check_bool(
            "large workload fallback to CPU (no GPU)",
            large_sub == MixedSubstrate::CpuOnly,
        );
        h.check_bool("large variance CPU result finite", gpu_var.is_finite());
    }

    // ═══════════════════════════════════════════════════════════════════
    // 3. Realtime inference → NPU substrate (simulated)
    // ═══════════════════════════════════════════════════════════════════

    let rt_data: Vec<f64> = (0..64).map(|_| rng.normal()).collect();
    let (_, npu_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "realtime_inference",
            compute_us: 5_000.0,
            data_bytes: 512,
            npu_available: true,
            needs_realtime: true, // NPU available + realtime required
        },
        |dev| {
            barracuda::ops::variance_reduce_f64::VarianceReduceF64::population_variance(
                dev.clone(),
                &rt_data,
            )
            .map_err(|e| format!("{e}"))
        },
        || {
            let n = rt_data.len() as f64;
            let m = rt_data.iter().sum::<f64>() / n;
            rt_data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n
        },
    );

    h.check_bool(
        "realtime + NPU → GpuToNpu substrate",
        npu_sub == MixedSubstrate::GpuToNpu,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 4. PCIe bridge transfer cost validation
    // ═══════════════════════════════════════════════════════════════════

    let bridge_gpu_npu = PcieBridge::new("RTX 4070", "AKD1000");
    h.check_bool(
        "PCIe GPU↔NPU bridge: no P2P (conservative)",
        !bridge_gpu_npu.can_p2p(),
    );

    let cost_staged = bridge_gpu_npu.transfer_cost(4_194_304); // 4 MB
    h.check_bool(
        "PCIe 4MB staged transfer cost > 0",
        cost_staged.estimated_us() > 0.0,
    );

    let cost_p2p_sim = neural_spring_forge::mixed::gpu_npu_cost(4_194_304, true);
    let cost_staged_sim = neural_spring_forge::mixed::gpu_npu_cost(4_194_304, false);
    h.check_bool(
        "P2P transfer faster than CPU-staged",
        cost_p2p_sim.estimated_us() < cost_staged_sim.estimated_us(),
    );

    // ═══════════════════════════════════════════════════════════════════
    // 5. Mixed dispatch parity: correlation through both paths
    // ═══════════════════════════════════════════════════════════════════

    let corr_x: Vec<f64> = (0..256).map(|_| rng.normal()).collect();
    let corr_y: Vec<f64> = (0..256).map(|_| rng.normal()).collect();
    let cpu_pearson =
        barracuda::stats::correlation::pearson_correlation(&corr_x, &corr_y).unwrap_or(0.0);

    let (mixed_pearson, corr_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "correlation",
            compute_us: 100_000.0,
            data_bytes: (corr_x.len() * 16) as u64,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| {
            let op = barracuda::ops::correlation_f64_wgsl::CorrelationF64::new(dev.clone())
                .map_err(|e| format!("{e}"))?;
            op.correlation(&corr_x, &corr_y).map_err(|e| format!("{e}"))
        },
        || cpu_pearson,
    );

    if dispatcher.has_gpu() {
        h.check_abs(
            "mixed dispatch: Pearson CPU vs GPU parity",
            mixed_pearson,
            cpu_pearson,
            tolerances::GPU_PEARSON_F64,
        );
        h.check_bool(
            "mixed dispatch: correlation routed to GPU",
            corr_sub == MixedSubstrate::GpuOnly,
        );
    } else {
        h.check_bool(
            "mixed dispatch: Pearson CPU-only finite",
            mixed_pearson.is_finite(),
        );
        h.check_bool(
            "mixed dispatch: correlation routed to CPU",
            corr_sub == MixedSubstrate::CpuOnly,
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // 6. Transfer cost model: GPU→CPU bandwidth bounds
    // ═══════════════════════════════════════════════════════════════════

    let cost_small_xfer = neural_spring_forge::mixed::gpu_cpu_cost(1024);
    let cost_large_xfer = neural_spring_forge::mixed::gpu_cpu_cost(1_073_741_824);
    h.check_bool(
        "transfer cost scales with data size",
        cost_large_xfer.estimated_us() > cost_small_xfer.estimated_us(),
    );
    h.check_bool(
        "1GB GPU→CPU transfer < 40ms (PCIe 4 x16)",
        cost_large_xfer.estimated_us() < 40_000.0,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 7. Dispatch boundary verification: crossover points
    // ═══════════════════════════════════════════════════════════════════

    let overhead = neural_spring_forge::dispatch::GPU_DISPATCH_OVERHEAD_US as f64;
    let small_compute = overhead * 0.5;
    let large_compute = overhead * 10.0;

    let (_, sub_below) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "boundary_below",
            compute_us: small_compute,
            data_bytes: 1024,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| {
            barracuda::ops::variance_reduce_f64::VarianceReduceF64::population_variance(
                dev.clone(),
                &[1.0, 2.0, 3.0],
            )
            .map_err(|e| format!("{e}"))
        },
        || 0.667,
    );
    h.check_bool(
        "below crossover → CPU",
        sub_below == MixedSubstrate::CpuOnly,
    );

    let (_, sub_above) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "boundary_above",
            compute_us: large_compute,
            data_bytes: 1024,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| {
            barracuda::ops::variance_reduce_f64::VarianceReduceF64::population_variance(
                dev.clone(),
                &[1.0, 2.0, 3.0],
            )
            .map_err(|e| format!("{e}"))
        },
        || 0.667,
    );
    if dispatcher.has_gpu() {
        h.check_bool(
            "above crossover → GPU",
            sub_above == MixedSubstrate::GpuOnly,
        );
    } else {
        h.check_bool(
            "above crossover → CPU (no GPU available)",
            sub_above == MixedSubstrate::CpuOnly,
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // nS-06: Immunological Anderson — mixed-hardware dispatch
    // ═══════════════════════════════════════════════════════════════════
    // NUCLEUS atomic pattern for AD classification pipeline:
    //   Tower: spectral eigensolve (disorder analysis)
    //   Node:  AD classification (regime detection)
    //   Nest:  provenance (result storage + audit trail)

    // Tower atomic: eigensolve for tissue lattice spectral analysis
    let tower_eig_result = dispatcher.eigh(
        &[
            2.0, -1.0, 0.0, -1.0, //
            -1.0, 2.0, -1.0, 0.0, //
            0.0, -1.0, 2.0, -1.0, //
            -1.0, 0.0, -1.0, 2.0, //
        ],
        4,
    );
    h.check_bool(
        "nS06 tower: eigensolve produces 4 eigenvalues",
        tower_eig_result.0.len() == 4,
    );
    h.check_bool(
        "nS06 tower: smallest eigenvalue ≈ 0 (connected lattice)",
        tower_eig_result.0[0].abs() < 0.5,
    );

    // Node atomic: KL divergence for cytokine distribution shift
    let node_kl = dispatcher.kl_divergence(
        &[0.60, 0.15, 0.10, 0.08, 0.05, 0.02],
        &[0.25, 0.20, 0.18, 0.15, 0.12, 0.10],
    );
    h.check_bool("nS06 node: KL divergence finite", node_kl.is_finite());
    h.check_bool("nS06 node: KL divergence positive", node_kl > 0.0);

    // Nest atomic: Shannon entropy for cell population disorder
    let nest_entropy = dispatcher.shannon_entropy(&[0.25, 0.20, 0.18, 0.15, 0.12, 0.10]);
    h.check_bool("nS06 nest: entropy finite", nest_entropy.is_finite());
    h.check_bool("nS06 nest: entropy positive", nest_entropy > 0.0);

    // PCIe bridge cost for NPU export of AD classifier weights
    let ad_weights_bytes = 64 * 3 * 8; // 64 reservoir × 3 heads × 8 bytes
    let npu_bridge = PcieBridge::new("RTX 4070", "AKD1000");
    let export_cost = npu_bridge.transfer_cost(ad_weights_bytes);
    h.check_bool(
        "nS06 PCIe: AD classifier NPU export cost finite",
        export_cost.estimated_us() > 0.0,
    );

    h.finish();
}
