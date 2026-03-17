// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mixed-hardware dispatch validation — exercises the `metalForge` substrate
//! routing, `PCIe` bridge, and NUCLEUS atomic coordination patterns.
//!
//! Validates the complete mixed-hardware stack:
//! 1. Substrate discovery (GPU, CPU, NPU)
//! 2. `PCIe` transfer cost model (bandwidth tiers, latency)
//! 3. Mixed-dispatch routing (GPU→CPU, GPU→NPU, chained transfers)
//! 4. `PcieBridge` P2P detection and cost estimation
//! 5. Dispatcher mixed-dispatch integration
//! 6. NUCLEUS atomic patterns (tower, node, nest)
//!
//! This proves the `metalForge` infrastructure is ready for `ToadStool`
//! absorption and multi-device workloads.
//!
//! ## Provenance
//!
//! | Baseline | Source |
//! |----------|--------|
//! | Eigenvalues 2.381966, 4.618034 | Analytical: eigenvalues of \[\[3,1\],\[1,4\]\] via characteristic polynomial |

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "validation binary"
)]

use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::{Dispatcher, MixedWorkload};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring_forge::mixed::{
    self, BandwidthTier, MixedSubstrate, PCIE_DMA_LATENCY_US, PCIE4_X4_BANDWIDTH_GBPS,
    PCIE4_X16_BANDWIDTH_GBPS,
};
use neural_spring_forge::pcie_bridge::PcieBridge;
use neural_spring_forge::substrate::{Capability, Identity, Properties, Substrate, SubstrateKind};

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("validate_mixed_hardware_dispatch");

    validate_substrate_discovery(&mut h);
    validate_npu_substrate_kind(&mut h);
    validate_pcie_transfer_costs(&mut h);
    validate_bandwidth_tiers(&mut h);
    validate_chained_transfer(&mut h);
    validate_p2p_comparison(&mut h);
    validate_pcie_bridge(&mut h);
    validate_mixed_routing_cpu_only(&mut h);
    validate_mixed_routing_gpu_small(&mut h);
    validate_mixed_routing_gpu_large(&mut h);
    validate_mixed_routing_npu_realtime(&mut h);
    let disp = auto_dispatcher().await;
    validate_dispatcher_mixed_dispatch(&mut h, &disp);
    validate_nucleus_tower_pattern(&mut h, &disp);
    validate_nucleus_node_pattern(&mut h, &disp);
    validate_nucleus_nest_pattern(&mut h, &disp);

    h.finish();
}

fn validate_substrate_discovery(h: &mut ValidationHarness) {
    let gpu = Substrate {
        kind: SubstrateKind::Gpu,
        identity: Identity::named("RTX 4070"),
        properties: Properties {
            memory_bytes: Some(12 * 1024 * 1024 * 1024),
            has_f64: true,
            ..Properties::default()
        },
        capabilities: vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::ShaderDispatch,
            Capability::ScalarReduce,
            Capability::Eigensolve,
            Capability::FusedMapReduce,
            Capability::TimestampQuery,
        ],
    };

    h.check_bool("GPU substrate has f64", gpu.has(&Capability::F64Compute));
    h.check_bool(
        "GPU substrate has shader dispatch",
        gpu.has(&Capability::ShaderDispatch),
    );
    h.check_bool(
        "GPU substrate capability count ≥ 5",
        gpu.capabilities.len() >= 5,
    );

    let summary = gpu.capability_summary();
    h.check_bool("capability summary contains f64", summary.contains("f64"));
    h.check_bool(
        "capability summary contains shader",
        summary.contains("shader"),
    );
}

fn validate_npu_substrate_kind(h: &mut ValidationHarness) {
    let npu = Substrate {
        kind: SubstrateKind::Npu,
        identity: Identity::named("AKD1000"),
        properties: Properties::default(),
        capabilities: vec![Capability::NpuInference, Capability::NpuBatch],
    };

    h.check_bool("NPU kind is Npu", npu.kind == SubstrateKind::Npu);
    h.check_bool(
        "NPU has inference capability",
        npu.has(&Capability::NpuInference),
    );
    h.check_bool("NPU has batch capability", npu.has(&Capability::NpuBatch));
    h.check_bool("NPU does NOT have f64", !npu.has(&Capability::F64Compute));

    let display = format!("{npu}");
    h.check_bool("NPU display contains NPU", display.contains("NPU"));
    h.check_bool("NPU display contains AKD1000", display.contains("AKD1000"));
}

fn validate_pcie_transfer_costs(h: &mut ValidationHarness) {
    let cost_1mb = mixed::gpu_cpu_cost(1_048_576);
    let us = cost_1mb.estimated_us();
    h.check_bool("1MB GPU→CPU cost > 30µs", us > 30.0);
    h.check_bool("1MB GPU→CPU cost < 50µs", us < 50.0);
    h.check_abs(
        "GPU→CPU bandwidth",
        cost_1mb.bandwidth_gbps,
        PCIE4_X16_BANDWIDTH_GBPS,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "GPU→CPU latency",
        cost_1mb.latency_us,
        PCIE_DMA_LATENCY_US,
        tolerances::EXACT_F64,
    );
}

fn validate_bandwidth_tiers(h: &mut ValidationHarness) {
    h.check_abs(
        "PCIe 4.0 x16 bandwidth",
        BandwidthTier::Pcie4X16.bandwidth_gbps(),
        31.5,
        tolerances::GPU_VARIANCE_DISPATCH_F32,
    );
    h.check_abs(
        "PCIe 4.0 x4 bandwidth",
        BandwidthTier::Pcie4X4.bandwidth_gbps(),
        7.9,
        tolerances::GPU_VARIANCE_DISPATCH_F32,
    );
    h.check_abs(
        "PCIe 5.0 x16 bandwidth",
        BandwidthTier::Pcie5X16.bandwidth_gbps(),
        63.0,
        tolerances::GPU_VARIANCE_DISPATCH_F32,
    );
    h.check_bool(
        "shared memory bandwidth > 100 GB/s",
        BandwidthTier::SharedMemory.bandwidth_gbps() > 100.0,
    );
    h.check_abs(
        "shared memory latency ~0.1µs",
        BandwidthTier::SharedMemory.latency_us(),
        0.1,
        tolerances::GPU_NORMAL_DISTANCE_SYMMETRIC_F32,
    );
}

fn validate_chained_transfer(h: &mut ValidationHarness) {
    let direct = mixed::transfer_cost_for_tier(1_048_576, BandwidthTier::Pcie4X4);
    let chained =
        mixed::chained_transfer_cost(1_048_576, BandwidthTier::Pcie4X16, BandwidthTier::Pcie4X4);

    h.check_bool(
        "chained (GPU→CPU→NPU) slower than direct",
        chained.estimated_us() > direct.estimated_us(),
    );

    h.check_abs(
        "chained bandwidth bottleneck is x4",
        chained.bandwidth_gbps,
        PCIE4_X4_BANDWIDTH_GBPS,
        tolerances::GPU_VARIANCE_DISPATCH_F32,
    );
}

fn validate_p2p_comparison(h: &mut ValidationHarness) {
    let (p2p, staged, p2p_faster) = mixed::compare_transfer_paths(
        4_194_304,
        BandwidthTier::Pcie4X4,
        BandwidthTier::Pcie4X16,
        BandwidthTier::Pcie4X4,
    );
    h.check_bool("P2P faster than staged for 4MB", p2p_faster);
    h.check_bool(
        "P2P cost < staged cost",
        p2p.estimated_us() < staged.estimated_us(),
    );
}

fn validate_pcie_bridge(h: &mut ValidationHarness) {
    let bridge = PcieBridge::new("RTX 4070", "AKD1000");
    h.check_bool(
        "PCIe bridge conservative (no P2P without proof)",
        !bridge.can_p2p(),
    );

    let cost = bridge.transfer_cost(1_048_576);
    h.check_bool("bridge transfer cost positive", cost.estimated_us() > 0.0);
    h.check_abs(
        "bridge uses x4 bandwidth (NPU link)",
        cost.bandwidth_gbps,
        PCIE4_X4_BANDWIDTH_GBPS,
        tolerances::GPU_VARIANCE_DISPATCH_F32,
    );
}

fn validate_mixed_routing_cpu_only(h: &mut ValidationHarness) {
    let sub = mixed::mixed_substrate(100.0, 1024, false, false, false);
    h.check_bool("no GPU/NPU → CpuOnly", sub == MixedSubstrate::CpuOnly);
}

fn validate_mixed_routing_gpu_small(h: &mut ValidationHarness) {
    let sub = mixed::mixed_substrate(100.0, 1024, true, false, false);
    h.check_bool(
        "small GPU workload → CpuOnly (dispatch overhead dominates)",
        sub == MixedSubstrate::CpuOnly,
    );
}

fn validate_mixed_routing_gpu_large(h: &mut ValidationHarness) {
    let sub = mixed::mixed_substrate(100_000.0, 1_048_576, true, false, false);
    h.check_bool(
        "large GPU workload → GpuOnly",
        sub == MixedSubstrate::GpuOnly,
    );
}

fn validate_mixed_routing_npu_realtime(h: &mut ValidationHarness) {
    let sub = mixed::mixed_substrate(50_000.0, 1_048_576, true, true, true);
    h.check_bool("realtime + NPU → GpuToNpu", sub == MixedSubstrate::GpuToNpu);

    let sub_npu_only = mixed::mixed_substrate(50_000.0, 1_048_576, false, true, true);
    h.check_bool(
        "no GPU + NPU + realtime → NpuOnly",
        sub_npu_only == MixedSubstrate::NpuOnly,
    );
}

async fn auto_dispatcher() -> Dispatcher {
    Gpu::new()
        .await
        .map_or_else(|_| Dispatcher::cpu_only(), Dispatcher::from_gpu)
}

fn validate_dispatcher_mixed_dispatch(h: &mut ValidationHarness, disp: &Dispatcher) {
    let workload = MixedWorkload {
        op: "test_mean",
        compute_us: 100_000.0,
        data_bytes: 1_048_576,
        npu_available: false,
        needs_realtime: false,
    };

    let data = [2.0, 4.0, 6.0, 8.0, 10.0];
    let (result, substrate) = disp.mixed_dispatch(
        &workload,
        |dev| {
            let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let t = barracuda::tensor::Tensor::from_data(&f32_data, vec![5], dev.clone())
                .map_err(|e| format!("{e}"))?;
            let mean = t.mean().map_err(|e| format!("{e}"))?;
            let v = mean.to_vec().map_err(|e| format!("{e}"))?;
            Ok(f64::from(v[0]))
        },
        || data.iter().sum::<f64>() / data.len() as f64,
    );

    h.check_abs(
        "mixed dispatch mean",
        result,
        6.0,
        tolerances::TENSOR_EXACT_F32,
    );
    h.check_bool(
        "mixed dispatch chose GPU or CPU",
        substrate == MixedSubstrate::GpuOnly || substrate == MixedSubstrate::CpuOnly,
    );
}

fn validate_nucleus_tower_pattern(h: &mut ValidationHarness, disp: &Dispatcher) {
    let a = vec![4.0, 1.0, 1.0, 3.0];
    let (eigenvalues, _) = disp.eigh(&a, 2);
    let mut sorted = eigenvalues;
    sorted.sort_by(f64::total_cmp);

    h.check_abs(
        "tower: eigensolve λ_min",
        sorted[0],
        2.381_966,
        tolerances::EIGENSOLVER_SMALL_MATRIX,
    );
    h.check_abs(
        "tower: eigensolve λ_max",
        sorted[1],
        4.618_034,
        tolerances::EIGENSOLVER_SMALL_MATRIX,
    );

    let entropy = disp.shannon_entropy(&[0.3, 0.7]);
    h.check_bool("tower: entropy positive", entropy > 0.0);
    h.check_bool("tower: entropy < ln(2)", entropy < 2.0_f64.ln() + 0.01);
}

fn validate_nucleus_node_pattern(h: &mut ValidationHarness, disp: &Dispatcher) {
    let pop = vec![2.0, 0.0, 1.0, 1.0, 0.0, 2.0];
    let freqs = disp.allele_frequencies(&pop, 3, 2);
    h.check_bool("node: allele_freq len=2", freqs.len() == 2);
    h.check_abs(
        "node: allele_freq[0]",
        freqs[0],
        0.5,
        tolerances::TENSOR_EXACT_F32,
    );

    let pi = disp.nucleotide_diversity(&pop, 3, 2);
    h.check_bool("node: diversity finite", pi.is_finite());
    h.check_bool("node: diversity ≥ 0", pi >= 0.0);
}

fn validate_nucleus_nest_pattern(h: &mut ValidationHarness, disp: &Dispatcher) {
    let coords = vec![(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)];
    let geo = disp.geographic_distances(&coords);
    h.check_bool("nest: geo distance matrix 3×3", geo.len() == 9);

    h.check_abs(
        "nest: self-dist[0,0]",
        geo[0],
        0.0,
        tolerances::TENSOR_EXACT_F32,
    );

    let dist_01 = geo[1];
    h.check_abs(
        "nest: dist(0,1)",
        dist_01,
        5.0,
        tolerances::TENSOR_EXACT_F32,
    );

    let dist_02 = geo[2];
    h.check_abs(
        "nest: dist(0,2)",
        dist_02,
        10.0,
        tolerances::TENSOR_EXACT_F32,
    );

    h.check_bool(
        "nest: triangle inequality",
        geo[1] + geo[5] >= geo[2] - tolerances::TENSOR_EXACT_F32,
    );
}
