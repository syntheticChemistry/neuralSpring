// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation: `metalForge` `PCIe` bridge + mixed-hardware dispatch.
//!
//! Exercises the full `metalForge` stack:
//!
//! 1. `PCIe` bandwidth tier cost model
//! 2. Direct P2P vs CPU-staged transfer comparison
//! 3. Chained multi-hop transfer estimation
//! 4. Substrate selection heuristics across all workload sizes
//! 5. Mixed dispatch routing through `Dispatcher`
//!
//! This is the "cross-systems portability proof" — validates that
//! the cost model correctly routes workloads to the optimal substrate.

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use neural_spring::gpu_dispatch::{Dispatcher, MixedWorkload};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring_forge::dispatch;
use neural_spring_forge::mixed::{
    self, compare_transfer_paths, transfer_cost_for_tier, BandwidthTier, MixedSubstrate,
};
use neural_spring_forge::pcie_bridge::PcieBridge;

fn validate_bandwidth_tiers(harness: &mut ValidationHarness) {
    let cost_pcie4x16 = transfer_cost_for_tier(1_048_576, BandwidthTier::Pcie4X16);
    let cost_pcie4x4 = transfer_cost_for_tier(1_048_576, BandwidthTier::Pcie4X4);
    let cost_pcie5 = transfer_cost_for_tier(1_048_576, BandwidthTier::Pcie5X16);
    let cost_shared = transfer_cost_for_tier(1_048_576, BandwidthTier::SharedMemory);

    harness.check_bool(
        "tier: x16 faster than x4 for 1MB",
        cost_pcie4x16.estimated_us() < cost_pcie4x4.estimated_us(),
    );
    harness.check_bool(
        "tier: PCIe5 faster than PCIe4 x16",
        cost_pcie5.estimated_us() < cost_pcie4x16.estimated_us(),
    );
    harness.check_bool(
        "tier: shared memory fastest",
        cost_shared.estimated_us() < cost_pcie5.estimated_us(),
    );
    harness.check_bool(
        "tier: all costs positive",
        cost_pcie4x16.estimated_us() > 0.0
            && cost_pcie4x4.estimated_us() > 0.0
            && cost_pcie5.estimated_us() > 0.0
            && cost_shared.estimated_us() > 0.0,
    );
}

fn validate_p2p_and_chaining(harness: &mut ValidationHarness) {
    let (p2p, staged, p2p_faster) = compare_transfer_paths(
        4_194_304,
        BandwidthTier::Pcie4X4,
        BandwidthTier::Pcie4X16,
        BandwidthTier::Pcie4X4,
    );
    harness.check_bool("p2p: direct wins over staged for 4MB", p2p_faster);
    harness.check_bool(
        "p2p: cost model consistent",
        p2p.estimated_us() < staged.estimated_us(),
    );

    let (_, _, p2p_faster_small) = compare_transfer_paths(
        1024,
        BandwidthTier::Pcie4X4,
        BandwidthTier::Pcie4X16,
        BandwidthTier::Pcie4X4,
    );
    harness.check_bool(
        "p2p: still wins for small transfers (latency-dominated)",
        p2p_faster_small,
    );

    let chained =
        mixed::chained_transfer_cost(1_048_576, BandwidthTier::Pcie4X16, BandwidthTier::Pcie4X4);
    let direct = transfer_cost_for_tier(1_048_576, BandwidthTier::Pcie4X4);

    harness.check_bool(
        "chained: 2-hop slower than direct",
        chained.estimated_us() > direct.estimated_us(),
    );
    let ratio = chained.estimated_us() / direct.estimated_us();
    harness.check_bool(
        "chained: overhead reasonable (< 3x direct)",
        ratio < tolerances::BRIDGE_CHAIN_OVERHEAD_MAX,
    );
}

fn validate_substrate_selection(harness: &mut ValidationHarness) {
    let sub_tiny = mixed::mixed_substrate(10.0, 256, true, false, false);
    harness.check_bool("substrate: tiny → CPU", sub_tiny == MixedSubstrate::CpuOnly);

    let sub_large = mixed::mixed_substrate(100_000.0, 1_048_576, true, false, false);
    harness.check_bool(
        "substrate: large → GPU",
        sub_large == MixedSubstrate::GpuOnly,
    );

    let sub_npu = mixed::mixed_substrate(50_000.0, 1_048_576, true, true, true);
    harness.check_bool(
        "substrate: realtime+NPU → GpuToNpu",
        sub_npu == MixedSubstrate::GpuToNpu,
    );

    let sub_no_gpu = mixed::mixed_substrate(100_000.0, 1_048_576, false, false, false);
    harness.check_bool(
        "substrate: no GPU → CPU",
        sub_no_gpu == MixedSubstrate::CpuOnly,
    );

    let sub_npu_only = mixed::mixed_substrate(100_000.0, 1_048_576, false, true, true);
    harness.check_bool(
        "substrate: NPU only → NpuOnly",
        sub_npu_only == MixedSubstrate::NpuOnly,
    );

    let overhead = dispatch::GPU_DISPATCH_OVERHEAD_US as f64;
    let data_bytes: u64 = 8192;
    let xfer_cost = mixed::gpu_cpu_cost(data_bytes).estimated_us();
    let threshold = xfer_cost + overhead;

    let sub_below = mixed::mixed_substrate(threshold * 0.9, data_bytes, true, false, false);
    let sub_above = mixed::mixed_substrate(threshold * 1.1, data_bytes, true, false, false);
    harness.check_bool(
        "crossover: below threshold → CPU",
        sub_below == MixedSubstrate::CpuOnly,
    );
    harness.check_bool(
        "crossover: above threshold → GPU",
        sub_above == MixedSubstrate::GpuOnly,
    );
}

fn validate_bridge_api(harness: &mut ValidationHarness) {
    let bridge = PcieBridge::new("RTX 4070", "AKD1000");
    harness.check_bool("bridge: conservative no-P2P", !bridge.can_p2p());

    let bridge_cost = bridge.transfer_cost(4_194_304);
    harness.check_bool(
        "bridge: 4MB transfer cost positive",
        bridge_cost.estimated_us() > 0.0,
    );
    harness.check_bool(
        "bridge: uses CPU-staged latency (no P2P)",
        bridge_cost.latency_us > tolerances::BRIDGE_PROBE_MIN_US,
    );
}

fn validate_live_dispatch(harness: &mut ValidationHarness, rng: &mut Rng, dispatcher: &Dispatcher) {
    let small: Vec<f64> = (0..32).map(|_| rng.normal()).collect();
    let (small_var, small_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "small_variance",
            compute_us: 10.0,
            data_bytes: 256,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| {
            barracuda::ops::variance_reduce_f64::VarianceReduceF64::population_variance(
                dev.clone(),
                &small,
            )
            .map_err(|e| format!("{e}"))
        },
        || {
            let count = small.len() as f64;
            let mean = small.iter().sum::<f64>() / count;
            small.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / count
        },
    );
    harness.check_bool(
        "live: small → CPU substrate",
        small_sub == MixedSubstrate::CpuOnly,
    );
    harness.check_bool("live: small variance finite", small_var.is_finite());

    let large: Vec<f64> = (0..4096).map(|_| rng.normal()).collect();
    let cpu_ref = {
        let count = large.len() as f64;
        let mean = large.iter().sum::<f64>() / count;
        large.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / count
    };
    let (large_var, large_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "large_variance",
            compute_us: 50_000.0,
            data_bytes: (large.len() * 8) as u64,
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
        || cpu_ref,
    );
    if dispatcher.has_gpu() {
        harness.check_bool(
            "live: large → GPU substrate",
            large_sub == MixedSubstrate::GpuOnly,
        );
        harness.check_abs(
            "live: large variance GPU parity",
            large_var,
            cpu_ref,
            tolerances::GPU_VARIANCE_F64,
        );
    } else {
        harness.check_bool(
            "live: large → CPU fallback",
            large_sub == MixedSubstrate::CpuOnly,
        );
        harness.check_bool("live: large variance finite", large_var.is_finite());
    }
}

#[tokio::main]
async fn main() {
    let mut harness = ValidationHarness::new("metalforge_pcie");
    let mut rng = Rng::new(42);
    let dispatcher = Dispatcher::new().await;

    validate_bandwidth_tiers(&mut harness);
    validate_p2p_and_chaining(&mut harness);
    validate_substrate_selection(&mut harness);
    validate_bridge_api(&mut harness);
    validate_live_dispatch(&mut harness, &mut rng, &dispatcher);

    harness.finish();
}
