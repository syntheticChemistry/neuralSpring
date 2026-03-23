// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation: metalForge cross-system dispatch — GPU → NPU → CPU.
//!
//! Proves the full metalForge stack end-to-end:
//!
//! 1. Hardware discovery (`probe_gpus`, `probe_cpu`, `discover`)
//! 2. Substrate capability classification
//! 3. Domain-specific dispatch heuristics (all 8 workload types)
//! 4. Multi-substrate computation parity (CPU ↔ GPU via `mixed_dispatch`)
//! 5. Transfer cost model (bandwidth tiers, multi-hop, P2P vs staged)
//! 6. Cross-system dispatch chain benchmarking
//!
//! This is the "cross-systems usage" proof for metalForge — validates
//! that the cost model correctly routes workloads across GPU, NPU, and
//! CPU substrates with numerical parity.
//!
//! ## Provenance
//!
//! Validation class: System.
//! Analytical reference: hardware discovery, substrate capability classification, cost model.
//! Components: metalForge inventory, dispatch heuristics, mixed (transfer cost), `pcie_bridge`, Dispatcher.

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use neural_spring::gpu_dispatch::{Dispatcher, MixedWorkload};
use neural_spring::primitives;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring_forge::dispatch::{self, Substrate};
use neural_spring_forge::inventory;
use neural_spring_forge::mixed::{
    self, BandwidthTier, MixedSubstrate, chained_transfer_cost, compare_transfer_paths,
    transfer_cost_for_tier,
};
use neural_spring_forge::pcie_bridge::PcieBridge;
use neural_spring_forge::substrate::{Capability, SubstrateKind};

fn validate_hardware_discovery(h: &mut ValidationHarness) {
    let substrates = inventory::discover();

    let cpu_count = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Cpu)
        .count();
    h.check_bool("discovery: exactly 1 CPU", cpu_count == 1);

    let Some(cpu) = substrates.iter().find(|s| s.kind == SubstrateKind::Cpu) else {
        h.check_bool("CPU: must exist in inventory", false);
        return;
    };
    h.check_bool("CPU: has f64", cpu.has(&Capability::F64Compute));
    h.check_bool("CPU: has f32", cpu.has(&Capability::F32Compute));
    h.check_bool("CPU: has CpuCompute", cpu.has(&Capability::CpuCompute));
    h.check_bool("CPU: name non-empty", !cpu.identity.name.is_empty());

    let gpu_count = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Gpu)
        .count();
    h.check_bool("discovery: found GPU(s)", gpu_count > 0);

    for gpu in substrates.iter().filter(|s| s.kind == SubstrateKind::Gpu) {
        h.check_bool(
            &format!("GPU '{}': has shader dispatch", gpu.identity.name),
            gpu.has(&Capability::ShaderDispatch),
        );
        h.check_bool(
            &format!("GPU '{}': has f32", gpu.identity.name),
            gpu.has(&Capability::F32Compute),
        );
    }
}

fn validate_domain_heuristics(h: &mut ValidationHarness) {
    h.check_bool(
        "pairwise: small → CPU",
        dispatch::pairwise_substrate(20, 500) == Substrate::Cpu,
    );
    h.check_bool(
        "pairwise: large → GPU",
        dispatch::pairwise_substrate(200, 1000) == Substrate::Gpu,
    );

    h.check_bool(
        "fitness: small → CPU",
        dispatch::batch_fitness_substrate(100, 10) == Substrate::Cpu,
    );
    h.check_bool(
        "fitness: large → GPU",
        dispatch::batch_fitness_substrate(50_000, 64) == Substrate::Gpu,
    );

    h.check_bool(
        "ODE: small → CPU",
        dispatch::ode_substrate(10, 100) == Substrate::Cpu,
    );
    h.check_bool(
        "ODE: large → GPU",
        dispatch::ode_substrate(1000, 2000) == Substrate::Gpu,
    );

    h.check_bool(
        "HMM: small → CPU",
        dispatch::hmm_substrate(3, 100) == Substrate::Cpu,
    );
    h.check_bool(
        "HMM: large → GPU",
        dispatch::hmm_substrate(3, 5000) == Substrate::Gpu,
    );

    h.check_bool(
        "spatial: small → CPU",
        dispatch::spatial_substrate(100) == Substrate::Cpu,
    );
    h.check_bool(
        "spatial: large → GPU",
        dispatch::spatial_substrate(10_000) == Substrate::Gpu,
    );

    h.check_bool(
        "IPR: small → CPU",
        dispatch::batch_ipr_substrate(10, 100) == Substrate::Cpu,
    );
    h.check_bool(
        "IPR: large → GPU",
        dispatch::batch_ipr_substrate(1000, 100) == Substrate::Gpu,
    );

    h.check_bool(
        "logsumexp: small → CPU",
        dispatch::logsumexp_substrate(10, 100) == Substrate::Cpu,
    );
    h.check_bool(
        "logsumexp: large → GPU",
        dispatch::logsumexp_substrate(100, 500) == Substrate::Gpu,
    );

    h.check_bool(
        "stochastic: small → CPU",
        dispatch::stochastic_substrate(5, 10, 100) == Substrate::Cpu,
    );
    h.check_bool(
        "stochastic: large → GPU",
        dispatch::stochastic_substrate(100, 50, 200) == Substrate::Gpu,
    );
}

fn validate_multi_substrate_parity(
    h: &mut ValidationHarness,
    rng: &mut Rng,
    dispatcher: &Dispatcher,
) {
    let data: Vec<f64> = (0..1024).map(|_| rng.normal()).collect();
    let data_bytes = (data.len() * 8) as u64;

    let cpu_var = {
        let n = data.len() as f64;
        let m = data.iter().sum::<f64>() / n;
        data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n
    };

    let (mixed_var, var_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "cross_system_variance",
            compute_us: 50_000.0,
            data_bytes,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| {
            let var_op = barracuda::ops::variance_f64_wgsl::VarianceF64::new(dev.clone())
                .map_err(|e| format!("{e}"))?;
            var_op.variance(&data).map_err(|e| format!("{e}"))
        },
        || cpu_var,
    );

    if dispatcher.has_gpu() {
        h.check_abs(
            "parity: variance CPU ↔ GPU",
            mixed_var,
            cpu_var,
            tolerances::GPU_VARIANCE_F64,
        );
        h.check_bool("parity: variance → GPU", var_sub == MixedSubstrate::GpuOnly);
    } else {
        h.check_bool("parity: variance CPU-only finite", mixed_var.is_finite());
    }

    let x: Vec<f64> = (0..512).map(|_| rng.normal()).collect();
    let y: Vec<f64> = (0..512).map(|_| rng.normal()).collect();
    let cpu_pearson = barracuda::stats::correlation::pearson_correlation(&x, &y).unwrap_or(0.0);

    let (mixed_pearson, pear_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "cross_system_pearson",
            compute_us: 80_000.0,
            data_bytes: (x.len() * 16) as u64,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| {
            let op = barracuda::ops::correlation_f64_wgsl::CorrelationF64::new(dev.clone())
                .map_err(|e| format!("{e}"))?;
            op.correlation(&x, &y).map_err(|e| format!("{e}"))
        },
        || cpu_pearson,
    );

    if dispatcher.has_gpu() {
        h.check_abs(
            "parity: Pearson CPU ↔ GPU",
            mixed_pearson,
            cpu_pearson,
            tolerances::GPU_PEARSON_F64,
        );
        h.check_bool("parity: Pearson → GPU", pear_sub == MixedSubstrate::GpuOnly);
    } else {
        h.check_bool("parity: Pearson CPU-only finite", mixed_pearson.is_finite());
    }

    let raw: Vec<f64> = (0..256)
        .map(|_| rng.uniform().abs() + primitives::POSITIVE_DATA_GUARD)
        .collect();
    let sum: f64 = raw.iter().sum();
    let e_data: Vec<f64> = raw.iter().map(|&x| x / sum).collect();
    let cpu_entropy = neural_spring::primitives::shannon_entropy(&e_data);

    let (mixed_entropy, ent_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "cross_system_entropy",
            compute_us: 60_000.0,
            data_bytes: (e_data.len() * 8) as u64,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| neural_spring::gpu_ops::shannon_entropy_gpu(&e_data, dev),
        || cpu_entropy,
    );

    if dispatcher.has_gpu() {
        h.check_abs(
            "parity: entropy CPU ↔ GPU",
            mixed_entropy,
            cpu_entropy,
            tolerances::GPU_ENTROPY_F64,
        );
        h.check_bool("parity: entropy → GPU", ent_sub == MixedSubstrate::GpuOnly);
    } else {
        h.check_bool("parity: entropy CPU-only finite", mixed_entropy.is_finite());
    }
}

fn validate_transfer_cost_hierarchy(h: &mut ValidationHarness) {
    let bytes: u64 = 16_777_216; // 16 MB

    let cost_shared = transfer_cost_for_tier(bytes, BandwidthTier::SharedMemory);
    let cost_pcie5 = transfer_cost_for_tier(bytes, BandwidthTier::Pcie5X16);
    let cost_pcie4x16 = transfer_cost_for_tier(bytes, BandwidthTier::Pcie4X16);
    let cost_pcie4x4 = transfer_cost_for_tier(bytes, BandwidthTier::Pcie4X4);

    h.check_bool(
        "hierarchy: shared < PCIe5 < PCIe4x16 < PCIe4x4",
        cost_shared.estimated_us() < cost_pcie5.estimated_us()
            && cost_pcie5.estimated_us() < cost_pcie4x16.estimated_us()
            && cost_pcie4x16.estimated_us() < cost_pcie4x4.estimated_us(),
    );

    let chained_gpu_cpu_npu =
        chained_transfer_cost(bytes, BandwidthTier::Pcie4X16, BandwidthTier::Pcie4X4);
    let direct_gpu_npu = transfer_cost_for_tier(bytes, BandwidthTier::Pcie4X4);
    h.check_bool(
        "multi-hop: GPU→CPU→NPU slower than GPU→NPU direct",
        chained_gpu_cpu_npu.estimated_us() > direct_gpu_npu.estimated_us(),
    );

    let (p2p, staged, p2p_faster) = compare_transfer_paths(
        bytes,
        BandwidthTier::Pcie4X4,
        BandwidthTier::Pcie4X16,
        BandwidthTier::Pcie4X4,
    );
    h.check_bool("P2P beats staged for 16MB", p2p_faster);
    h.check_bool(
        "P2P cost < staged cost",
        p2p.estimated_us() < staged.estimated_us(),
    );

    let bridge = PcieBridge::new("RTX 4070", "AKD1000");
    h.check_bool("bridge: conservative no-P2P", !bridge.can_p2p());
    let bridge_cost = bridge.transfer_cost(bytes);
    h.check_bool(
        "bridge: 16MB cost realistic (100–5000µs)",
        bridge_cost.estimated_us() > tolerances::BRIDGE_COST_MIN_US
            && bridge_cost.estimated_us() < tolerances::BRIDGE_COST_MAX_US,
    );
}

fn validate_npu_routing(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    let sub_rt_npu = mixed::mixed_substrate(50_000.0, 1_048_576, true, true, true);
    h.check_bool(
        "NPU routing: realtime + GPU + NPU → GpuToNpu",
        sub_rt_npu == MixedSubstrate::GpuToNpu,
    );

    let sub_no_gpu_rt = mixed::mixed_substrate(50_000.0, 1_048_576, false, true, true);
    h.check_bool(
        "NPU routing: no GPU + NPU + realtime → NpuOnly",
        sub_no_gpu_rt == MixedSubstrate::NpuOnly,
    );

    let sub_npu_no_rt = mixed::mixed_substrate(50_000.0, 1_048_576, true, true, false);
    h.check_bool(
        "NPU routing: not realtime → GPU (ignores NPU)",
        sub_npu_no_rt == MixedSubstrate::GpuOnly,
    );

    let small_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let (_, npu_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "npu_probe",
            compute_us: 5_000.0,
            data_bytes: 512,
            npu_available: true,
            needs_realtime: true,
        },
        |dev| {
            let var_op = barracuda::ops::variance_f64_wgsl::VarianceF64::new(dev.clone())
                .map_err(|e| format!("{e}"))?;
            var_op.variance(&small_data).map_err(|e| format!("{e}"))
        },
        || 1.25,
    );
    h.check_bool(
        "NPU live: GpuToNpu selected (falls back to GPU/CPU)",
        npu_sub == MixedSubstrate::GpuToNpu
            || npu_sub == MixedSubstrate::GpuOnly
            || npu_sub == MixedSubstrate::CpuOnly,
    );
}

fn validate_crossover_sweep(h: &mut ValidationHarness) {
    let overhead = dispatch::GPU_DISPATCH_OVERHEAD_US as f64;
    let data_bytes: u64 = 65_536; // 64 KB
    let xfer = mixed::gpu_cpu_cost(data_bytes).estimated_us();
    let threshold = xfer + overhead;

    let mut last_substrate = MixedSubstrate::CpuOnly;
    let mut crossover_found = false;

    for factor_10x in 0..30 {
        let compute_us = 10.0_f64 * 1.5_f64.powi(factor_10x);
        let sub = mixed::mixed_substrate(compute_us, data_bytes, true, false, false);

        if last_substrate == MixedSubstrate::CpuOnly && sub == MixedSubstrate::GpuOnly {
            crossover_found = true;
            let ratio = compute_us / threshold;
            h.check_bool(
                &format!(
                    "crossover at {compute_us:.0}µs (threshold {threshold:.0}µs, ratio {ratio:.2})"
                ),
                ratio > tolerances::DISPATCH_COST_RATIO_MIN
                    && ratio < tolerances::DISPATCH_COST_RATIO_MAX,
            );
            break;
        }
        last_substrate = sub;
    }
    h.check_bool("crossover: CPU→GPU transition found", crossover_found);
}

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("cross_system_dispatch");
    let mut rng = Rng::new(42);
    let dispatcher = Dispatcher::new().await;

    validate_hardware_discovery(&mut h);
    validate_domain_heuristics(&mut h);
    validate_multi_substrate_parity(&mut h, &mut rng, &dispatcher);
    validate_transfer_cost_hierarchy(&mut h);
    validate_npu_routing(&mut h, &dispatcher);
    validate_crossover_sweep(&mut h);

    h.finish();
}
