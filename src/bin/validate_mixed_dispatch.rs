// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation: `metalForge` mixed-hardware dispatch infrastructure.
//!
//! Validates `MixedSubstrate` selection, `TransferCost` estimations,
//! and `PcieBridge` API contracts for GPU-NPU-CPU dispatch routing.
//!
//! ## Validated components
//!
//! - `mixed::mixed_substrate` — workload-aware dispatch selection
//! - `mixed::gpu_cpu_cost` / `gpu_npu_cost` — transfer cost model
//! - `pcie_bridge::PcieBridge` — device pair abstraction
//! - `pcie_bridge::detect_p2p` — P2P capability detection

use neural_spring::validation::ValidationHarness;
use neural_spring_forge::mixed::{
    gpu_cpu_cost, gpu_npu_cost, mixed_substrate, MixedSubstrate, PCIE4_X16_BANDWIDTH_GBPS,
    PCIE4_X4_BANDWIDTH_GBPS,
};
use neural_spring_forge::pcie_bridge::{detect_p2p, PcieBridge};

fn main() {
    let mut h = ValidationHarness::new("mixed_dispatch");

    validate_transfer_costs(&mut h);
    validate_substrate_selection(&mut h);
    validate_pcie_bridge(&mut h);
    validate_cost_ordering(&mut h);
    validate_bandwidth_constants(&mut h);

    h.finish();
}

fn validate_transfer_costs(h: &mut ValidationHarness) {
    // 1 MB GPU→CPU via PCIe 4.0 x16: should be ~35 µs
    let cost_1mb = gpu_cpu_cost(1_048_576);
    let us = cost_1mb.estimated_us();
    h.check_bool(
        &format!("GPU→CPU 1MB: {us:.1} µs in [30, 50] range"),
        (30.0..=50.0).contains(&us),
    );

    // 1 MB GPU→NPU via P2P: should be ~135 µs (limited by x4 bandwidth)
    let cost_npu_p2p = gpu_npu_cost(1_048_576, true);
    let us_p2p = cost_npu_p2p.estimated_us();
    h.check_bool(
        &format!("GPU→NPU P2P 1MB: {us_p2p:.1} µs in [100, 200]"),
        (100.0..=200.0).contains(&us_p2p),
    );

    // GPU→NPU staged should be slower than P2P
    let cost_npu_staged = gpu_npu_cost(1_048_576, false);
    h.check_bool(
        "GPU→NPU staged slower than P2P",
        cost_npu_staged.estimated_us() > cost_npu_p2p.estimated_us(),
    );

    // Zero bytes should be approximately latency-only
    let cost_0 = gpu_cpu_cost(0);
    h.check_upper(
        &format!(
            "GPU→CPU 0 bytes ≈ latency ({:.1} µs)",
            cost_0.estimated_us()
        ),
        cost_0.estimated_us(),
        10.0,
    );
}

fn validate_substrate_selection(h: &mut ValidationHarness) {
    // Small workload → CPU
    h.check_bool(
        "small compute (100 µs) → CPU",
        mixed_substrate(100.0, 1024, true, false, false) == MixedSubstrate::CpuOnly,
    );

    // Large workload → GPU
    h.check_bool(
        "large compute (100 ms) → GPU",
        mixed_substrate(100_000.0, 1_048_576, true, false, false) == MixedSubstrate::GpuOnly,
    );

    // Real-time inference with NPU → GpuToNpu
    h.check_bool(
        "realtime + NPU → GpuToNpu",
        mixed_substrate(50_000.0, 1_048_576, true, true, true) == MixedSubstrate::GpuToNpu,
    );

    // No GPU → CPU
    h.check_bool(
        "no GPU → CPU",
        mixed_substrate(50_000.0, 1_048_576, false, false, false) == MixedSubstrate::CpuOnly,
    );

    // No GPU but NPU + realtime → NpuOnly
    h.check_bool(
        "no GPU + NPU + realtime → NpuOnly",
        mixed_substrate(50_000.0, 1_048_576, false, true, true) == MixedSubstrate::NpuOnly,
    );
}

fn validate_pcie_bridge(h: &mut ValidationHarness) {
    let bridge = PcieBridge::new("RTX 4070", "AKD1000");
    h.check_bool("bridge: default no P2P", !bridge.can_p2p());

    let cost = bridge.transfer_cost(1_048_576);
    h.check_bool(
        &format!("bridge transfer cost > 0 ({:.1} µs)", cost.estimated_us()),
        cost.estimated_us() > 0.0,
    );

    h.check_bool(
        "detect_p2p: conservative false without PCI topology",
        !detect_p2p("GPU", "NPU"),
    );
}

fn validate_cost_ordering(h: &mut ValidationHarness) {
    // x16 should be faster than x4 for same data
    let x16_cost = gpu_cpu_cost(10_000_000); // 10 MB on x16
    let x4_cost = gpu_npu_cost(10_000_000, true); // 10 MB on x4
    h.check_bool(
        "PCIe x16 faster than x4 for 10 MB",
        x16_cost.estimated_us() < x4_cost.estimated_us(),
    );

    // Larger data → longer transfer
    let small = gpu_cpu_cost(1000);
    let large = gpu_cpu_cost(10_000_000);
    h.check_bool(
        "10 MB transfer slower than 1 KB",
        large.estimated_us() > small.estimated_us(),
    );
}

fn validate_bandwidth_constants(h: &mut ValidationHarness) {
    h.check_bool(
        &format!("PCIe 4.0 x16 bandwidth = {PCIE4_X16_BANDWIDTH_GBPS} GB/s"),
        (PCIE4_X16_BANDWIDTH_GBPS - 31.5).abs() < f64::EPSILON,
    );
    h.check_bool(
        &format!("PCIe 4.0 x4 bandwidth = {PCIE4_X4_BANDWIDTH_GBPS} GB/s"),
        (PCIE4_X4_BANDWIDTH_GBPS - 7.9).abs() < f64::EPSILON,
    );
}
