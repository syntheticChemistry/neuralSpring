// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation: NUCLEUS atomic compute dispatch for spectral science.
//!
//! Validates the full NUCLEUS atomic coordination pattern:
//!
//! - **Tower** (BearDog+Songbird): authentication scope + capability discovery
//! - **Node** (Tower+ToadStool): GPU compute dispatch for eigensolve/IPR/variance
//! - **Nest** (Tower+NestGate): result provenance + storage routing
//!
//! Each section proves that the spectral science pipeline can run through
//! the NUCLEUS dispatch infrastructure with CPU↔GPU parity. This validator
//! exercises the `Dispatcher` API the same way biomeOS graphs orchestrate
//! across atomics — capability-based routing, not hardcoded paths.
//!
//! Targets for `ToadStool` absorption:
//! - `dispatcher.eigh` → `barracuda::linalg::batched_eigh_gpu`
//! - `dispatcher.disorder_sweep` → `barracuda::spectral::disorder_sweep_gpu`
//! - `dispatcher.mixed_dispatch` → `barracuda::unified_hardware::route`

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]

use neural_spring::anderson_localization::{anderson_hamiltonian_random, disorder_sweep, mean_ipr};
use neural_spring::eigh::eigh_householder_qr;
use neural_spring::gpu_dispatch::{Dispatcher, MixedWorkload};
use neural_spring::gpu_ops;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::weight_spectral;
use neural_spring_forge::inventory;
use neural_spring_forge::mixed::MixedSubstrate;
use neural_spring_forge::pcie_bridge::PcieBridge;
use neural_spring_forge::substrate::SubstrateKind;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("nucleus_compute_dispatch");
    let mut rng = Rng::new(42);
    let dispatcher = Dispatcher::new().await;
    let has_gpu = dispatcher.has_gpu();

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: Tower atomic — capability discovery + substrate inventory
    // ═══════════════════════════════════════════════════════════════════
    //
    // In the NUCLEUS, BearDog authenticates, Songbird discovers.
    // Here we validate the substrate inventory that Songbird would use
    // to locate GPU/NPU/CPU compute targets.

    validate_tower_discovery(&mut h);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 2: Node atomic — ToadStool GPU compute dispatch
    // ═══════════════════════════════════════════════════════════════════
    //
    // Node = Tower + ToadStool. ToadStool provides GPU compute:
    //   eigh → eigensolve, disorder_sweep → batch Anderson, variance → stats.
    // The dispatcher routes to GPU when available, CPU otherwise.

    validate_node_eigensolve(&mut h, &dispatcher, &mut rng, has_gpu);
    validate_node_anderson(&mut h, &dispatcher, &mut rng, has_gpu);
    validate_node_hessian(&mut h, &dispatcher, &mut rng, has_gpu);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 3: Nest atomic — storage routing + result provenance
    // ═══════════════════════════════════════════════════════════════════
    //
    // Nest = Tower + NestGate. Validates that compute results can be
    // serialized and transferred with correct provenance metadata.
    // PCIe bridge cost model validates transfer overhead.

    validate_nest_provenance(&mut h, &dispatcher, &mut rng, has_gpu);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 4: Mixed atomics — Node↔Nest coordination
    // ═══════════════════════════════════════════════════════════════════
    //
    // Full NUCLEUS pipeline: compute on Node → transfer → store on Nest.
    // Validates mixed_dispatch routing, substrate selection, and parity.

    validate_mixed_atomic_coordination(&mut h, &dispatcher, &mut rng, has_gpu);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 5: biomeOS graph — spectral→popgen→entropy pipeline
    // ═══════════════════════════════════════════════════════════════════
    //
    // biomeOS orchestrates multi-stage pipelines via capability graphs.
    // Validates that chained atomics produce consistent results regardless
    // of per-stage substrate routing decisions.

    validate_biome_graph_coordination(&mut h, &dispatcher, &mut rng);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 6: PCIe bridge — NPU↔GPU bypass validation
    // ═══════════════════════════════════════════════════════════════════

    validate_pcie_bypass(&mut h);

    h.finish();
}

fn validate_tower_discovery(h: &mut ValidationHarness) {
    let substrates = inventory::discover();

    let has_cpu = substrates.iter().any(|s| s.kind == SubstrateKind::Cpu);
    h.check_bool("Tower: CPU substrate discovered", has_cpu);

    let gpu_count = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Gpu)
        .count();
    h.check_bool("Tower: GPU(s) discovered", gpu_count > 0);

    let total = substrates.len();
    h.check_bool("Tower: at least 2 substrates (CPU+GPU)", total >= 2);

    for s in &substrates {
        h.check_bool(
            &format!("Tower: substrate '{}' has name", s.identity.name),
            !s.identity.name.is_empty(),
        );
    }
}

fn validate_node_eigensolve(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    rng: &mut Rng,
    has_gpu: bool,
) {
    let dim = 24;
    let weights: Vec<f64> = (0..12 * 12).map(|_| rng.normal()).collect();
    let ham = weight_spectral::weight_to_hamiltonian(&weights, 12, 12);

    let cpu_decomp = eigh_householder_qr(&ham, dim);
    let (gpu_evals, gpu_evecs) = dispatcher.eigh(&ham, dim);

    let mut cpu_indices: Vec<usize> = (0..cpu_decomp.eigenvalues.len()).collect();
    cpu_indices.sort_by(|&a, &b| {
        cpu_decomp.eigenvalues[a]
            .partial_cmp(&cpu_decomp.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let cpu_sorted: Vec<f64> = cpu_indices
        .iter()
        .map(|&i| cpu_decomp.eigenvalues[i])
        .collect();
    let mut gpu_indices: Vec<usize> = (0..gpu_evals.len()).collect();
    gpu_indices.sort_by(|&a, &b| {
        gpu_evals[a]
            .partial_cmp(&gpu_evals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let gpu_sorted: Vec<f64> = gpu_indices.iter().map(|&i| gpu_evals[i]).collect();

    let eval_diff = cpu_sorted
        .iter()
        .zip(gpu_sorted.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        "Node eigh: CPU ↔ dispatch parity",
        eval_diff < tolerances::GPU_EIGH_DISPATCH_F64,
    );

    let cpu_ipr = mean_ipr(&cpu_decomp.eigenvectors, dim);
    let dispatch_ipr = mean_ipr(&gpu_evecs, dim);
    h.check_bool("Node eigh: IPR finite", dispatch_ipr.is_finite());
    h.check_bool("Node eigh: IPR > 0", dispatch_ipr > 0.0);

    if has_gpu {
        h.check_abs(
            "Node eigh: IPR CPU ↔ GPU",
            dispatch_ipr,
            cpu_ipr,
            tolerances::GPU_EIGH_DISPATCH_F64,
        );
    }

    let cpu_var = {
        let n = cpu_decomp.eigenvalues.len() as f64;
        let m = cpu_decomp.eigenvalues.iter().sum::<f64>() / n;
        cpu_decomp
            .eigenvalues
            .iter()
            .map(|&x| (x - m).powi(2))
            .sum::<f64>()
            / n
    };
    let dispatch_var = dispatcher.variance(&gpu_evals);

    h.check_abs(
        "Node eigh: eigenvalue variance CPU ↔ dispatch",
        dispatch_var,
        cpu_var,
        tolerances::GPU_VARIANCE_DISPATCH_F32,
    );
}

fn validate_node_anderson(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    _rng: &mut Rng,
    _has_gpu: bool,
) {
    let n = 20;
    let w_values = vec![0.5, 2.0, 8.0, 16.0];

    let cpu_iprs = {
        let mut r = Rng::new(99);
        disorder_sweep(n, 1.0, &w_values, &mut r)
    };

    h.check_bool(
        "Node Anderson: IPR count matches W count",
        cpu_iprs.len() == w_values.len(),
    );

    for (i, &w) in w_values.iter().enumerate() {
        let mut r = Rng::new(99);
        let ham = anderson_hamiltonian_random(n, 1.0, w, &mut r);
        let (_, dispatch_evecs) = dispatcher.eigh(&ham, n);
        let dispatch_ipr = mean_ipr(&dispatch_evecs, n);

        h.check_bool(
            &format!("Node Anderson[{i}]: IPR finite (W={w})"),
            dispatch_ipr.is_finite() && dispatch_ipr > 0.0,
        );
    }

    h.check_bool(
        "Node Anderson: IPR increases with disorder",
        cpu_iprs.last().unwrap_or(&0.0) > cpu_iprs.first().unwrap_or(&1.0),
    );

    let (sweep_var, _sweep_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "nucleus_anderson_sweep_var",
            compute_us: 80_000.0,
            data_bytes: (cpu_iprs.len() * 8) as u64,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| {
            barracuda::ops::variance_reduce_f64::VarianceReduceF64::population_variance(
                dev.clone(),
                &cpu_iprs,
            )
            .map_err(|e| format!("{e}"))
        },
        || {
            let len = cpu_iprs.len() as f64;
            let m = cpu_iprs.iter().sum::<f64>() / len;
            cpu_iprs.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / len
        },
    );

    h.check_bool("Node Anderson: sweep variance > 0", sweep_var > 0.0);
}

fn validate_node_hessian(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    rng: &mut Rng,
    _has_gpu: bool,
) {
    let n = 16;
    let mut hessian = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let v = rng.normal();
            hessian[i * n + j] = v;
            hessian[j * n + i] = v;
        }
    }

    let (cpu_evals, _) = Dispatcher::cpu_only().eigh(&hessian, n);
    let (dispatch_evals, _) = dispatcher.eigh(&hessian, n);

    let mut cpu_sorted = cpu_evals;
    cpu_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut dispatch_sorted = dispatch_evals;
    dispatch_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let eval_diff = cpu_sorted
        .iter()
        .zip(dispatch_sorted.iter())
        .map(|(c, d)| (c - d).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        "Node Hessian: eigenvalue CPU ↔ dispatch parity",
        eval_diff < tolerances::GPU_EIGH_DISPATCH_F64,
    );

    let negative_count = cpu_sorted
        .iter()
        .filter(|&&v| v < -tolerances::CROSS_LANGUAGE)
        .count();
    let positive_count = cpu_sorted
        .iter()
        .filter(|&&v| v > tolerances::CROSS_LANGUAGE)
        .count();
    h.check_bool(
        "Node Hessian: has both positive and negative eigenvalues",
        negative_count > 0 && positive_count > 0,
    );

    let spectral_range = cpu_sorted.last().unwrap_or(&0.0) - cpu_sorted.first().unwrap_or(&0.0);
    h.check_bool("Node Hessian: spectral range > 0", spectral_range > 0.0);
}

fn validate_nest_provenance(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    rng: &mut Rng,
    has_gpu: bool,
) {
    let data: Vec<f64> = (0..512).map(|_| rng.normal()).collect();
    let data_bytes = (data.len() * 8) as u64;

    let cpu_mean = Dispatcher::cpu_only().mean(&data);
    let cpu_var = Dispatcher::cpu_only().variance(&data);

    let dispatch_mean = dispatcher.mean(&data);
    let dispatch_var = dispatcher.variance(&data);

    h.check_abs(
        "Nest provenance: mean CPU ↔ dispatch",
        dispatch_mean,
        cpu_mean,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );
    h.check_abs(
        "Nest provenance: variance CPU ↔ dispatch",
        dispatch_var,
        cpu_var,
        tolerances::GPU_VARIANCE_DISPATCH_F32,
    );

    let cpu_frob = Dispatcher::cpu_only().frobenius_norm(&data);
    let dispatch_frob = dispatcher.frobenius_norm(&data);
    h.check_abs(
        "Nest provenance: Frobenius CPU ↔ dispatch",
        dispatch_frob,
        cpu_frob,
        tolerances::GPU_VARIANCE_DISPATCH_F32,
    );

    let (mixed_mean, mean_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "nest_result_mean",
            compute_us: 40_000.0,
            data_bytes,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| gpu_ops::mean_gpu(&data, dev),
        || cpu_mean,
    );

    h.check_abs(
        "Nest provenance: mixed mean CPU ↔ dispatch",
        mixed_mean,
        cpu_mean,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );
    if has_gpu {
        h.check_bool(
            "Nest provenance: result → GPU substrate",
            mean_sub == MixedSubstrate::GpuOnly,
        );
    }
}

fn validate_mixed_atomic_coordination(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    rng: &mut Rng,
    has_gpu: bool,
) {
    let dim = 16;
    let weights: Vec<f64> = (0..8 * 8).map(|_| rng.normal()).collect();
    let ham = weight_spectral::weight_to_hamiltonian(&weights, 8, 8);

    let (evals, evecs) = dispatcher.eigh(&ham, dim);

    let ipr = mean_ipr(&evecs, dim);

    let cpu_var = {
        let n = evals.len() as f64;
        let m = evals.iter().sum::<f64>() / n;
        evals.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n
    };

    let (node_var, _node_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "nucleus_pipeline_eigenvalue_variance",
            compute_us: 60_000.0,
            data_bytes: (evals.len() * 8) as u64,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| {
            barracuda::ops::variance_reduce_f64::VarianceReduceF64::population_variance(
                dev.clone(),
                &evals,
            )
            .map_err(|e| format!("{e}"))
        },
        || cpu_var,
    );

    h.check_bool("Atomic coordination: variance finite", node_var.is_finite());
    h.check_bool("Atomic coordination: IPR > 0", ipr > 0.0);

    if has_gpu {
        h.check_abs(
            "Atomic coordination: variance GPU ↔ CPU",
            node_var,
            cpu_var,
            tolerances::GPU_VARIANCE_F64,
        );
    }

    let probs: Vec<f64> = {
        let raw: Vec<f64> = (0..64).map(|_| rng.uniform().abs() + 1e-10).collect();
        let sum: f64 = raw.iter().sum();
        raw.iter().map(|v| v / sum).collect()
    };
    let cpu_entropy = neural_spring::primitives::shannon_entropy(&probs);

    let (mixed_entropy, _) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "nucleus_pipeline_entropy",
            compute_us: 30_000.0,
            data_bytes: (probs.len() * 8) as u64,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| gpu_ops::shannon_entropy_gpu(&probs, dev),
        || cpu_entropy,
    );

    if has_gpu {
        h.check_abs(
            "Atomic coordination: entropy GPU ↔ CPU",
            mixed_entropy,
            cpu_entropy,
            tolerances::GPU_ENTROPY_F64,
        );
    } else {
        h.check_bool(
            "Atomic coordination: entropy finite",
            mixed_entropy.is_finite(),
        );
    }

    let cpu_l2 = Dispatcher::cpu_only().l2_distance(&weights[..32], &weights[32..64]);
    let dispatch_l2 = dispatcher.l2_distance(&weights[..32], &weights[32..64]);
    h.check_abs(
        "Atomic coordination: L2 CPU ↔ dispatch",
        dispatch_l2,
        cpu_l2,
        tolerances::GPU_L2_DISPATCH_F32,
    );
}

fn validate_biome_graph_coordination(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    rng: &mut Rng,
) {
    // biomeOS graph pattern: spectral → popgen → entropy pipeline
    // Tower discovers substrates, Node runs each stage, Nest collects provenance.
    // This validates the full pipeline produces consistent results regardless
    // of which substrate each stage routes to.

    let n_loci = 4;
    let n_ind_a = 10;
    let n_ind_b = 10;
    let pop_a: Vec<f64> = (0..n_ind_a * n_loci).map(|_| rng.uniform()).collect();
    let pop_b: Vec<f64> = (0..n_ind_b * n_loci).map(|_| rng.uniform()).collect();

    // Stage 1 (Node): allele frequencies via dispatcher
    let af_a = dispatcher.allele_frequencies(&pop_a, n_ind_a, n_loci);
    let af_b = dispatcher.allele_frequencies(&pop_b, n_ind_b, n_loci);
    h.check_bool(
        "biomeOS pipeline: AF dim matches n_loci",
        af_a.len() == n_loci && af_b.len() == n_loci,
    );

    // Stage 2 (Node): nucleotide diversity
    let pi_a = dispatcher.nucleotide_diversity(&pop_a, n_ind_a, n_loci);
    let pi_b = dispatcher.nucleotide_diversity(&pop_b, n_ind_b, n_loci);
    h.check_bool(
        "biomeOS pipeline: π finite and non-negative",
        pi_a.is_finite() && pi_a >= 0.0 && pi_b.is_finite() && pi_b >= 0.0,
    );

    // Stage 3 (Node): FST between populations
    let fst = dispatcher.pairwise_fst(&pop_a, n_ind_a, &pop_b, n_ind_b, n_loci);
    h.check_bool("biomeOS pipeline: FST finite", fst.is_finite());

    // Stage 4 (Nest): entropy of allele frequency distribution
    let af_normed: Vec<f64> = {
        let sum: f64 = af_a.iter().map(|v| v.abs()).sum::<f64>() + 1e-15;
        af_a.iter().map(|v| v.abs() / sum).collect()
    };
    let ent = dispatcher.shannon_entropy(&af_normed);
    h.check_bool("biomeOS pipeline: AF entropy finite", ent.is_finite());

    // Cross-check: CPU-only pipeline gives same result
    let cpu_disp = Dispatcher::cpu_only();
    let cpu_af_a = cpu_disp.allele_frequencies(&pop_a, n_ind_a, n_loci);
    let af_diff = af_a
        .iter()
        .zip(cpu_af_a.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "biomeOS pipeline: AF dispatch↔CPU parity",
        af_diff,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_pcie_bypass(h: &mut ValidationHarness) {
    let gpu_npu = PcieBridge::new("RTX_4070", "AKD1000_NPU");
    let gpu_cpu = PcieBridge::new("RTX_4070", "x86_64_CPU");
    let npu_cpu = PcieBridge::new("AKD1000_NPU", "x86_64_CPU");

    let sizes = [1024_u64, 65_536, 1_048_576, 16_777_216];
    for &sz in &sizes {
        let cost_gpu_npu = gpu_npu.transfer_cost(sz).estimated_us();
        let cost_gpu_cpu = gpu_cpu.transfer_cost(sz).estimated_us();
        let cost_npu_cpu = npu_cpu.transfer_cost(sz).estimated_us();

        h.check_bool(
            &format!("PCIe bypass: all costs > 0 @ {sz} bytes"),
            cost_gpu_npu > 0.0 && cost_gpu_cpu > 0.0 && cost_npu_cpu > 0.0,
        );
    }

    let small_cost = gpu_npu.transfer_cost(1024).estimated_us();
    let large_cost = gpu_npu.transfer_cost(16_777_216).estimated_us();
    h.check_bool(
        "PCIe bypass: transfer scales with data size",
        large_cost > small_cost,
    );

    h.check_bool(
        "PCIe bypass: conservative P2P (GPU→NPU)",
        !gpu_npu.can_p2p(),
    );
    h.check_bool(
        "PCIe bypass: conservative P2P (GPU→CPU)",
        !gpu_cpu.can_p2p(),
    );
}
