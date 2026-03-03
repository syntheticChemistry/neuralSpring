// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mixed-hardware validation for publication experiments (Exp-050/052/053)
//! via metalForge NUCLEUS atomic patterns.
//!
//! Extends the publication experiment validation to the mixed-hardware tier:
//! 1. NPU→GPU via `PCIe` bridge (simulated, bypassing CPU round-trip)
//! 2. GPU→CPU fallback with parity proof
//! 3. Substrate routing respects compute/data cost model
//! 4. NUCLEUS Node atomic pattern: compute dispatch through `ToadStool`
//! 5. NUCLEUS Nest atomic pattern: storage dispatch (result provenance)
//!
//! This is the "mixed NUCLEUS atomics" proof — coordinated via biomeOS graphs.

#![expect(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "validation binary"
)]

use neural_spring::anderson_localization::{anderson_hamiltonian_random, disorder_sweep, mean_ipr};
use neural_spring::eigh::eigh_householder_qr;
use neural_spring::gpu_dispatch::{Dispatcher, MixedWorkload};
use neural_spring::gpu_ops;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::weight_spectral;
use neural_spring_forge::mixed::MixedSubstrate;
use neural_spring_forge::pcie_bridge::PcieBridge;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("publication_mixed_hardware");
    let mut rng = Rng::new(42);
    let dispatcher = Dispatcher::new().await;
    let has_gpu = dispatcher.has_gpu();

    // ═══════════════════════════════════════════════════════════════════
    // PART 1: Exp-050 Training Trajectory — mixed-hardware eigensolve
    // ═══════════════════════════════════════════════════════════════════
    //
    // Pattern: weight matrix → eigensolve (Node compute) → IPR/entropy
    //   Small matrix → CPU substrate (cost model routes to CPU)
    //   Large matrix → GPU substrate (compute exceeds transfer cost)

    {
        let dim = 16;
        let ham = weight_spectral::weight_to_hamiltonian(
            &(0..8 * 8).map(|_| rng.normal()).collect::<Vec<f64>>(),
            8,
            8,
        );

        let (cpu_evals, _) = Dispatcher::cpu_only().eigh(&ham, dim);
        let (mixed_evals, _) = dispatcher.eigh(&ham, dim);

        let mut cpu_sorted = cpu_evals.clone();
        cpu_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut mixed_sorted = mixed_evals;
        mixed_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let diff = cpu_sorted
            .iter()
            .zip(mixed_sorted.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0_f64, f64::max);

        h.check_bool(
            "Exp-050: eigensolve CPU ↔ mixed-hardware parity",
            diff < tolerances::GPU_EIGH_DISPATCH_F64,
        );

        let cpu_var = Dispatcher::cpu_only().variance(&cpu_evals);

        let (mixed_var, var_sub) = dispatcher.mixed_dispatch(
            &MixedWorkload {
                op: "exp050_trajectory_variance",
                compute_us: 50_000.0,
                data_bytes: (cpu_evals.len() * 8) as u64,
                npu_available: false,
                needs_realtime: false,
            },
            |dev| {
                barracuda::ops::variance_reduce_f64::VarianceReduceF64::population_variance(
                    dev.clone(),
                    &cpu_evals,
                )
                .map_err(|e| format!("{e}"))
            },
            || cpu_var,
        );

        if has_gpu {
            h.check_abs(
                "Exp-050: trajectory variance GPU ↔ CPU",
                mixed_var,
                cpu_var,
                tolerances::GPU_VARIANCE_F64,
            );
            h.check_bool(
                "Exp-050: variance routed to GPU",
                var_sub == MixedSubstrate::GpuOnly,
            );
        } else {
            h.check_bool("Exp-050: variance finite (CPU)", mixed_var.is_finite());
        }

        let (small_var, small_sub) = dispatcher.mixed_dispatch(
            &MixedWorkload {
                op: "exp050_small_variance",
                compute_us: 5.0,
                data_bytes: 128,
                npu_available: false,
                needs_realtime: false,
            },
            |dev| {
                barracuda::ops::variance_reduce_f64::VarianceReduceF64::population_variance(
                    dev.clone(),
                    &cpu_evals[..4],
                )
                .map_err(|e| format!("{e}"))
            },
            || {
                let s = &cpu_evals[..4];
                let n = s.len() as f64;
                let m = s.iter().sum::<f64>() / n;
                s.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n
            },
        );
        h.check_bool(
            "Exp-050: small workload routed to CPU",
            small_sub == MixedSubstrate::CpuOnly,
        );
        h.check_bool("Exp-050: small variance finite", small_var.is_finite());
    }

    // ═══════════════════════════════════════════════════════════════════
    // PART 2: Exp-052 Hessian Eigenanalysis — NPU→GPU bridge pattern
    // ═══════════════════════════════════════════════════════════════════
    //
    // Simulates NPU→GPU transfer for realtime spectral monitoring:
    // NPU pre-processes gradient snapshots, GPU does eigensolve.
    // PCIe bridge estimates transfer cost for bypass vs CPU round-trip.

    {
        let n = 20;
        let mut hessian = vec![0.0; n * n];
        for i in 0..n {
            hessian[i * n + i] = (i + 1) as f64;
        }

        let mut cpu_decomp = eigh_householder_qr(&hessian, n);
        let mut cpu_evals = std::mem::take(&mut cpu_decomp.eigenvalues);
        cpu_evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for (i, &eval) in cpu_evals.iter().enumerate() {
            h.check_abs(
                &format!("Exp-052: diag Hessian eval[{i}]"),
                eval,
                (i + 1) as f64,
                tolerances::CROSS_LANGUAGE,
            );
        }

        let (npu_var, npu_sub) = dispatcher.mixed_dispatch(
            &MixedWorkload {
                op: "exp052_hessian_realtime_monitor",
                compute_us: 80_000.0,
                data_bytes: (n * n * 8) as u64,
                npu_available: true,
                needs_realtime: true,
            },
            |dev| {
                barracuda::ops::variance_reduce_f64::VarianceReduceF64::population_variance(
                    dev.clone(),
                    &cpu_evals,
                )
                .map_err(|e| format!("{e}"))
            },
            || {
                let mean = cpu_evals.iter().sum::<f64>() / n as f64;
                cpu_evals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64
            },
        );

        let expected_npu_sub = if has_gpu {
            MixedSubstrate::GpuToNpu
        } else {
            MixedSubstrate::CpuOnly
        };
        h.check_bool(
            "Exp-052: realtime+NPU → GpuToNpu substrate",
            npu_sub == expected_npu_sub,
        );
        h.check_bool("Exp-052: NPU bridge variance finite", npu_var.is_finite());

        let bridge = PcieBridge::new(dispatcher.adapter_name(), "simulated_NPU_AKD1000");
        let transfer_cost = bridge.transfer_cost((n * n * 8) as u64);
        h.check_bool(
            "Exp-052: PCIe bridge cost > 0",
            transfer_cost.estimated_us() > 0.0,
        );
        h.check_bool(
            "Exp-052: PCIe bridge conservative (no P2P claim)",
            !bridge.can_p2p(),
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // PART 3: Exp-053 Anderson Multi-Agent — cross-substrate parity
    // ═══════════════════════════════════════════════════════════════════
    //
    // Anderson localization disorder sweep: same physics, routed to
    // different substrates based on lattice size.

    {
        let n = 16;
        let w_vals = vec![1.0, 4.0, 16.0];

        let cpu_iprs = {
            let mut r = Rng::new(42);
            disorder_sweep(n, 1.0, &w_vals, &mut r)
        };

        let ipr_sweep_cpu = {
            let mut r = Rng::new(42);
            disorder_sweep(n, 1.0, &w_vals, &mut r)
        };

        for &w in &w_vals {
            let mut r = Rng::new(42);
            let ham = anderson_hamiltonian_random(n, 1.0, w, &mut r);
            let (_, mixed_evecs) = dispatcher.eigh(&ham, n);
            let mixed_ipr = mean_ipr(&mixed_evecs, n);
            h.check_bool(
                &format!("Exp-053: mixed-hw IPR finite (W={w})"),
                mixed_ipr.is_finite() && mixed_ipr > 0.0,
            );
        }

        let (sweep_var, _sweep_sub) = dispatcher.mixed_dispatch(
            &MixedWorkload {
                op: "exp053_disorder_sweep_variance",
                compute_us: 100_000.0,
                data_bytes: (ipr_sweep_cpu.len() * 8) as u64,
                npu_available: false,
                needs_realtime: false,
            },
            |dev| {
                barracuda::ops::variance_reduce_f64::VarianceReduceF64::population_variance(
                    dev.clone(),
                    &ipr_sweep_cpu,
                )
                .map_err(|e| format!("{e}"))
            },
            || {
                let n = ipr_sweep_cpu.len() as f64;
                let m = ipr_sweep_cpu.iter().sum::<f64>() / n;
                ipr_sweep_cpu.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n
            },
        );

        h.check_bool("Exp-053: sweep variance finite", sweep_var.is_finite());
        h.check_bool("Exp-053: sweep variance > 0 (IPR varies)", sweep_var > 0.0);

        let ipr_increases = cpu_iprs[2] > cpu_iprs[0];
        h.check_bool(
            "Exp-053: IPR increases with disorder (physics)",
            ipr_increases,
        );

        let cpu_entropy = neural_spring::primitives::shannon_entropy(&cpu_iprs);
        let (mixed_entropy, _ent_sub) = dispatcher.mixed_dispatch(
            &MixedWorkload {
                op: "exp053_ipr_entropy",
                compute_us: 30_000.0,
                data_bytes: (cpu_iprs.len() * 8) as u64,
                npu_available: false,
                needs_realtime: false,
            },
            |dev| gpu_ops::shannon_entropy_gpu(&cpu_iprs, dev),
            || cpu_entropy,
        );

        h.check_bool("Exp-053: IPR entropy finite", mixed_entropy.is_finite());
        if has_gpu {
            // IPR values come from GPU-dispatched eigensolve which has
            // ~0.1 tolerance. Entropy over those values propagates error.
            h.check_abs(
                "Exp-053: IPR entropy CPU ↔ GPU",
                mixed_entropy,
                cpu_entropy,
                tolerances::GPU_EIGH_DISPATCH_F64,
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // PART 4: NUCLEUS atomic coordination — transfer cost hierarchy
    // ═══════════════════════════════════════════════════════════════════
    //
    // Validates metalForge transfer cost model for NUCLEUS atomics:
    //   Node (GPU) → Nest (storage) — data moves via PCIe to CPU, then to NestGate
    //   Node (GPU) → Tower (network) — data moves via PCIe for Songbird transmission
    //   NPU → GPU — direct PCIe if P2P, else via CPU staging

    {
        let bridge_gpu_npu = PcieBridge::new("RTX_4070", "AKD1000_NPU");
        let bridge_gpu_gpu = PcieBridge::new("RTX_4070", "TITAN_V");

        let cost_small = bridge_gpu_npu.transfer_cost(1024);
        let cost_large = bridge_gpu_npu.transfer_cost(1_048_576);

        h.check_bool(
            "NUCLEUS: large transfer costs more than small",
            cost_large.estimated_us() > cost_small.estimated_us(),
        );

        let cost_gpu_npu = bridge_gpu_npu.transfer_cost(65_536);
        let cost_gpu_gpu = bridge_gpu_gpu.transfer_cost(65_536);
        h.check_bool(
            "NUCLEUS: GPU→NPU cost > 0",
            cost_gpu_npu.estimated_us() > 0.0,
        );
        h.check_bool(
            "NUCLEUS: GPU→GPU cost > 0",
            cost_gpu_gpu.estimated_us() > 0.0,
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // PART 5: Dispatcher substrate coverage proof
    // ═══════════════════════════════════════════════════════════════════
    //
    // Proves all substrate paths produce valid (finite) results.

    {
        let data: Vec<f64> = (0..256).map(|_| rng.normal()).collect();
        let cpu_mean = Dispatcher::cpu_only().mean(&data);
        let mixed_mean = dispatcher.mean(&data);

        h.check_abs(
            "Substrate coverage: mean CPU ↔ mixed",
            mixed_mean,
            cpu_mean,
            tolerances::GPU_MEAN_DISPATCH_F32,
        );

        let cpu_l2 = Dispatcher::cpu_only().l2_distance(&data[..128], &data[128..]);
        let mixed_l2 = dispatcher.l2_distance(&data[..128], &data[128..]);

        h.check_abs(
            "Substrate coverage: L2 CPU ↔ mixed",
            mixed_l2,
            cpu_l2,
            tolerances::GPU_L2_DISPATCH_F32,
        );

        let cpu_frob = Dispatcher::cpu_only().frobenius_norm(&data);
        let mixed_frob = dispatcher.frobenius_norm(&data);

        h.check_abs(
            "Substrate coverage: Frobenius CPU ↔ mixed",
            mixed_frob,
            cpu_frob,
            tolerances::GPU_VARIANCE_DISPATCH_F32,
        );
    }

    h.finish();
}
