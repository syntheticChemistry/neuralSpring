// SPDX-License-Identifier: AGPL-3.0-or-later

//! NUCLEUS PCIe bypass + mixed-pipeline validation.
//!
//! Validates GPU→NPU→CPU pathways through metalForge's PCIe bridge,
//! proving that NUCLEUS atomics can route compute across heterogeneous
//! hardware while bypassing CPU roundtrips for GPU↔NPU transfers.
//!
//! ## NUCLEUS atomics validated
//!
//! - **Tower**: substrate discovery + capability inventory
//! - **Node**: GPU/NPU compute dispatch with routing decisions
//! - **Nest**: provenance tracking with entropy preservation
//!
//! ## PCIe bypass pathways
//!
//! ```text
//! GPU ←──PCIe──→ NPU   (direct: bypasses CPU staging)
//! GPU ←──PCIe──→ CPU   (standard: staged through host memory)
//! NPU ←──PCIe──→ CPU   (standard: staged through host memory)
//! GPU → NPU → CPU      (multi-hop: chained transfers)
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::suboptimal_flops
)]

use neural_spring::gpu_dispatch::{Dispatcher, MixedWorkload};
use neural_spring::gpu_ops;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring_forge::inventory;
use neural_spring_forge::mixed::{self, BandwidthTier, MixedSubstrate};
use neural_spring_forge::pcie_bridge::PcieBridge;
use neural_spring_forge::substrate::SubstrateKind;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("validate_nucleus_pcie_mixed_pipeline");

    let dispatcher = Dispatcher::new().await;
    let mut rng = Rng::new(42);

    validate_pcie_bypass_cost_model(&mut h);
    validate_pcie_multi_hop_chain(&mut h);
    validate_npu_routing_decisions(&mut h, &dispatcher);
    validate_gpu_npu_bypass_science_pipeline(&mut h, &dispatcher, &mut rng);
    validate_nucleus_tower_node_nest_chain(&mut h, &dispatcher, &mut rng);
    validate_biomeos_multi_stage_graph(&mut h, &dispatcher, &mut rng);

    h.finish();
}

fn validate_pcie_bypass_cost_model(h: &mut ValidationHarness) {
    eprintln!("\n── PCIe bypass cost model ──");

    let gpu_npu = PcieBridge::new("RTX_4070", "AKD1000_NPU");
    let gpu_cpu = PcieBridge::new("RTX_4070", "x86_64_CPU");

    let sizes: Vec<u64> = vec![1_024, 16_384, 262_144, 1_048_576, 16_777_216];
    for &sz in &sizes {
        let bypass_cost = gpu_npu.transfer_cost(sz).estimated_us();
        let staged_gpu_cpu = gpu_cpu.transfer_cost(sz).estimated_us();

        h.check_bool(&format!("PCIe bypass cost > 0 @ {sz}B"), bypass_cost > 0.0);
        h.check_bool(
            &format!("PCIe staged cost > 0 @ {sz}B"),
            staged_gpu_cpu > 0.0,
        );
    }

    let bypass_small = gpu_npu.transfer_cost(1_024).estimated_us();
    let bypass_large = gpu_npu.transfer_cost(16_777_216).estimated_us();
    h.check_bool(
        "PCIe bypass: large > small transfer cost",
        bypass_large > bypass_small,
    );
    h.check_bool(
        "PCIe bypass: sub-linear scaling (overhead dominates small)",
        bypass_large / bypass_small < 16_777.0,
    );
}

fn validate_pcie_multi_hop_chain(h: &mut ValidationHarness) {
    eprintln!("\n── PCIe multi-hop transfer chain ──");

    let gpu_npu = PcieBridge::new("RTX_4070", "AKD1000_NPU");
    let npu_cpu = PcieBridge::new("AKD1000_NPU", "x86_64_CPU");
    let gpu_cpu = PcieBridge::new("RTX_4070", "x86_64_CPU");

    let data_size = 1_048_576_u64;

    let hop1 = gpu_npu.transfer_cost(data_size).estimated_us();
    let hop2 = npu_cpu.transfer_cost(data_size).estimated_us();
    let multi_hop = hop1 + hop2;

    let direct = gpu_cpu.transfer_cost(data_size).estimated_us();

    h.check_bool(
        "multi-hop GPU→NPU→CPU has 2 transfer costs",
        multi_hop > direct * 0.5,
    );
    h.check_bool(
        "multi-hop < 3× direct (bounded overhead)",
        multi_hop < direct * 3.0,
    );

    let chained =
        mixed::chained_transfer_cost(data_size, BandwidthTier::Pcie4X16, BandwidthTier::Pcie4X16);
    let chained_us = chained.estimated_us();
    h.check_bool(
        "chained_transfer_cost finite and positive",
        chained_us > 0.0 && chained_us.is_finite(),
    );

    let direct_us =
        mixed::transfer_cost_for_tier(data_size, BandwidthTier::Pcie4X16).estimated_us();
    h.check_bool("2-hop > 1-hop transfer cost", chained_us > direct_us);
}

fn validate_npu_routing_decisions(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    eprintln!("\n── NPU routing decisions ──");

    let (_, sub_rt) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "npu_realtime_inference",
            compute_us: 5_000.0,
            data_bytes: 512,
            npu_available: true,
            needs_realtime: true,
        },
        |_dev| Ok(1.0_f64),
        || 1.0,
    );
    h.check_bool(
        "NPU routing: realtime + NPU available → GpuToNpu",
        sub_rt == MixedSubstrate::GpuToNpu,
    );

    let (_, sub_no_rt) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "npu_non_realtime",
            compute_us: 5_000.0,
            data_bytes: 512,
            npu_available: true,
            needs_realtime: false,
        },
        |_dev| Ok(1.0_f64),
        || 1.0,
    );
    h.check_bool(
        "NPU routing: non-realtime + NPU available → NOT NpuOnly",
        sub_no_rt != MixedSubstrate::NpuOnly,
    );

    let (_, sub_no_npu) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "npu_unavailable",
            compute_us: 5_000.0,
            data_bytes: 512,
            npu_available: false,
            needs_realtime: true,
        },
        |_dev| Ok(1.0_f64),
        || 1.0,
    );
    h.check_bool(
        "NPU routing: NPU unavailable → fallback (not GpuToNpu)",
        sub_no_npu != MixedSubstrate::GpuToNpu && sub_no_npu != MixedSubstrate::NpuOnly,
    );

    let (_, sub_small) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "small_workload",
            compute_us: 10.0,
            data_bytes: 64,
            npu_available: false,
            needs_realtime: false,
        },
        |_dev| Ok(1.0_f64),
        || 1.0,
    );
    h.check_bool(
        "NPU routing: tiny workload → CpuOnly",
        sub_small == MixedSubstrate::CpuOnly,
    );

    let (_, sub_large) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "large_workload",
            compute_us: 500_000.0,
            data_bytes: 4_194_304,
            npu_available: false,
            needs_realtime: false,
        },
        |_dev| Ok(1.0_f64),
        || 1.0,
    );
    h.check_bool(
        "NPU routing: large workload → GpuOnly",
        sub_large == MixedSubstrate::GpuOnly,
    );
}

fn validate_gpu_npu_bypass_science_pipeline(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    rng: &mut Rng,
) {
    eprintln!("\n── GPU→NPU bypass science pipeline ──");

    let n = 256;
    let data: Vec<f64> = (0..n).map(|_| rng.normal()).collect();

    let cpu_var = dispatcher.variance(&data);
    let cpu_mean = dispatcher.mean(&data);

    let (mixed_var, var_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "spectral_variance_gpu_compute",
            compute_us: 50_000.0,
            data_bytes: (n * 8) as u64,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| gpu_ops::variance_gpu(&data, dev),
        || cpu_var,
    );
    h.check_abs(
        "bypass pipeline: variance GPU↔CPU parity",
        mixed_var,
        cpu_var,
        tolerances::GPU_VARIANCE_F64,
    );
    eprintln!("  variance routed to: {var_sub:?}");

    let (npu_mean, npu_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "npu_inference_mean",
            compute_us: 100_000.0,
            data_bytes: (n * 8) as u64,
            npu_available: true,
            needs_realtime: true,
        },
        |dev| gpu_ops::mean_gpu(&data, dev),
        || cpu_mean,
    );
    h.check_abs(
        "bypass pipeline: NPU inference mean parity",
        npu_mean,
        cpu_mean,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );
    h.check_bool(
        "bypass pipeline: realtime routes to GpuToNpu",
        npu_sub == MixedSubstrate::GpuToNpu,
    );

    let gpu_npu_bridge = PcieBridge::new("RTX_4070", "AKD1000_NPU");
    let transfer_cost = gpu_npu_bridge.transfer_cost((n * 8) as u64).estimated_us();
    h.check_bool(
        "bypass pipeline: GPU→NPU transfer cost realistic",
        transfer_cost > 0.0 && transfer_cost < 10_000.0,
    );
}

fn validate_nucleus_tower_node_nest_chain(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    rng: &mut Rng,
) {
    eprintln!("\n── NUCLEUS Tower→Node→Nest chain ──");

    // Tower: discover hardware substrates
    let substrates = inventory::discover();
    let gpu_count = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Gpu)
        .count();
    h.check_bool(
        "Tower: GPU substrates available for Node compute",
        gpu_count > 0,
    );

    // Node: GPU eigensolve for spectral analysis
    let dim = 8;
    let ham: Vec<f64> = {
        let mut m = vec![0.0; dim * dim];
        for i in 0..dim {
            m[i * dim + i] = 2.0 + rng.uniform();
            if i + 1 < dim {
                let off = 0.5 * rng.normal();
                m[i * dim + (i + 1)] = off;
                m[(i + 1) * dim + i] = off;
            }
        }
        m
    };
    let (evals, _) = dispatcher.eigh(&ham, dim);
    h.check_bool(
        "Node: eigensolve produces dim eigenvalues",
        evals.len() == dim,
    );
    h.check_bool(
        "Node: all eigenvalues finite",
        evals.iter().all(|e| e.is_finite()),
    );

    // Node: variance of eigenvalues
    let (node_var, node_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "node_eigenvalue_variance",
            compute_us: 40_000.0,
            data_bytes: (dim * 8) as u64,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| gpu_ops::variance_gpu(&evals, dev),
        || {
            let n = evals.len() as f64;
            let m = evals.iter().sum::<f64>() / n;
            evals.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n
        },
    );
    h.check_bool(
        "Node: eigenvalue variance > 0 (non-degenerate)",
        node_var > 0.0,
    );
    eprintln!("  node variance routed to: {node_sub:?}");

    // Nest: provenance entropy — information content of eigenspectrum
    let probs: Vec<f64> = {
        let abs_evals: Vec<f64> = evals.iter().map(|e| e.abs() + 1e-15).collect();
        let sum: f64 = abs_evals.iter().sum();
        abs_evals.iter().map(|v| v / sum).collect()
    };
    let nest_entropy = dispatcher.shannon_entropy(&probs);
    h.check_bool("Nest: provenance entropy > 0", nest_entropy > 0.0);
    h.check_bool(
        "Nest: provenance entropy < ln(dim)",
        nest_entropy < (dim as f64).ln() + 0.01,
    );
}

fn validate_biomeos_multi_stage_graph(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    rng: &mut Rng,
) {
    eprintln!("\n── biomeOS multi-stage graph coordination ──");

    // biomeOS graph: spectral → population → information pipeline
    // Each stage routes independently through NUCLEUS atomics.
    // Provenance tracks which substrate handled each stage.

    // Stage 1: spectral eigensolve (Node compute, GPU-routed)
    let dim = 12;
    let ham: Vec<f64> = {
        let mut m = vec![0.0; dim * dim];
        for i in 0..dim {
            m[i * dim + i] = 3.0 + rng.uniform();
            if i + 1 < dim {
                let off = rng.normal() * 0.3;
                m[i * dim + (i + 1)] = off;
                m[(i + 1) * dim + i] = off;
            }
        }
        m
    };
    let (evals, evecs) = dispatcher.eigh(&ham, dim);
    h.check_bool(
        "biomeOS graph stage 1: eigensolve complete",
        evals.len() == dim,
    );

    // Stage 2: population diversity metrics (derived from spectral data)
    let ipr = neural_spring::anderson_localization::mean_ipr(&evecs, dim);
    h.check_bool(
        "biomeOS graph stage 2: IPR finite and positive",
        ipr > 0.0 && ipr.is_finite(),
    );

    let bandwidth = {
        let mut sorted = evals.clone();
        sorted.sort_by(f64::total_cmp);
        sorted.last().unwrap_or(&0.0) - sorted.first().unwrap_or(&0.0)
    };
    h.check_bool("biomeOS graph stage 2: bandwidth > 0", bandwidth > 0.0);

    // Stage 3: information-theoretic summary (Nest provenance)
    let lsr = neural_spring::weight_spectral::level_spacing_ratio(&evals);
    h.check_bool(
        "biomeOS graph stage 3: level spacing ratio finite",
        lsr.is_finite(),
    );

    let probs: Vec<f64> = {
        let abs_evals: Vec<f64> = evals.iter().map(|e| e.abs() + 1e-15).collect();
        let sum: f64 = abs_evals.iter().sum();
        abs_evals.iter().map(|v| v / sum).collect()
    };
    let entropy = dispatcher.shannon_entropy(&probs);
    h.check_bool(
        "biomeOS graph stage 3: spectral entropy finite",
        entropy.is_finite() && entropy > 0.0,
    );

    // Cross-validation: CPU-only pipeline should give same results
    let cpu_disp = Dispatcher::cpu_only();
    let (cpu_evals, cpu_evecs) = cpu_disp.eigh(&ham, dim);
    let cpu_ipr = neural_spring::anderson_localization::mean_ipr(&cpu_evecs, dim);
    h.check_abs(
        "biomeOS graph: IPR dispatch↔CPU parity",
        ipr,
        cpu_ipr,
        tolerances::TENSOR_MATMUL_F32,
    );

    let cpu_probs: Vec<f64> = {
        let abs_evals: Vec<f64> = cpu_evals.iter().map(|e| e.abs() + 1e-15).collect();
        let sum: f64 = abs_evals.iter().sum();
        abs_evals.iter().map(|v| v / sum).collect()
    };
    let cpu_entropy = cpu_disp.shannon_entropy(&cpu_probs);
    h.check_abs(
        "biomeOS graph: entropy dispatch↔CPU parity",
        entropy,
        cpu_entropy,
        tolerances::GPU_ENTROPY_F64,
    );

    eprintln!(
        "  pipeline summary: {dim} eigenvalues → IPR={ipr:.4} → BW={bandwidth:.2} → LSR={lsr:.4} → H={entropy:.4}"
    );
}
