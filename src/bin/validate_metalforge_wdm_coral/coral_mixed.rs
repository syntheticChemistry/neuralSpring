// SPDX-License-Identifier: AGPL-3.0-or-later

//! coralForge Evoformer dispatch, NUCLEUS nest provenance, mixed routing,
//! PCIe bypass cost validation, and CPU reference helpers.

use neural_spring::gpu_dispatch::{Dispatcher, MixedWorkload};
use neural_spring::primitives;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring_forge::mixed::MixedSubstrate;
use neural_spring_forge::pcie_bridge::PcieBridge;

// ═══════════════════════════════════════════════════════════════════
// Node: coralForge Evoformer attention via mixed dispatch
// ═══════════════════════════════════════════════════════════════════

pub fn validate_node_coral_attention(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
    let n_res = 6;
    let head_dim = 4;

    let q: Vec<f64> = (0..n_res * head_dim).map(|_| rng.normal() * 0.3).collect();
    let k_t: Vec<f64> = (0..head_dim * n_res).map(|_| rng.normal() * 0.3).collect();

    let wl = MixedWorkload {
        op: "evoformer_attention_matmul",
        compute_us: 300_000.0,
        data_bytes: ((n_res * head_dim * 2) * 8) as u64,
        npu_available: false,
        needs_realtime: false,
    };

    let (scores, substrate) = disp.mixed_dispatch(
        &wl,
        |dev| {
            barracuda::dispatch::matmul_dispatch(&q, &k_t, n_res, head_dim, n_res, Some(dev))
                .map_err(|e| format!("{e}"))
        },
        || rect_matmul_cpu(&q, &k_t, n_res, head_dim, n_res),
    );

    h.check_bool(
        "Node coral attn: correct shape",
        scores.len() == n_res * n_res,
    );
    h.check_bool(
        "Node coral attn: all finite",
        scores.iter().all(|v| v.is_finite()),
    );
    h.check_bool(
        "Node coral attn: GPU routing (large compute)",
        substrate == MixedSubstrate::GpuOnly,
    );

    let scale = (head_dim as f64).sqrt();
    let scaled: Vec<f64> = scores.iter().map(|&v| v / scale).collect();
    let probs = disp.softmax(&scaled[..n_res]);
    let sum: f64 = probs.iter().sum();
    h.check_abs(
        "Node coral attn: softmax sums to 1",
        sum,
        1.0,
        tolerances::TENSOR_EXACT_F32,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Node: coralForge triangle multiply via mixed dispatch
// ═══════════════════════════════════════════════════════════════════

pub fn validate_node_coral_trimul(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
    let n = 4;
    let c = 2;

    let proj_a: Vec<f64> = (0..n * n * c).map(|_| rng.normal() * 0.3).collect();
    let proj_b: Vec<f64> = (0..n * n * c).map(|_| rng.normal() * 0.3).collect();

    let wl = MixedWorkload {
        op: "evoformer_triangle_multiply",
        compute_us: 400_000.0,
        data_bytes: ((n * n * c * 2) * 8) as u64,
        npu_available: false,
        needs_realtime: false,
    };

    let (result, substrate) = disp.mixed_dispatch(
        &wl,
        |dev| {
            let mut out = vec![0.0_f64; n * n * c];
            for i in 0..n {
                for j in 0..n {
                    for ch in 0..c {
                        let a_col: Vec<f64> =
                            (0..n).map(|k| proj_a[(i * n + k) * c + ch]).collect();
                        let b_col: Vec<f64> =
                            (0..n).map(|k| proj_b[(j * n + k) * c + ch]).collect();
                        let dot = barracuda::dispatch::matmul_dispatch(
                            &a_col,
                            &b_col,
                            1,
                            n,
                            1,
                            Some(dev),
                        )
                        .map_err(|e| format!("{e}"))?;
                        out[(i * n + j) * c + ch] = dot[0];
                    }
                }
            }
            Ok(out)
        },
        || {
            let mut out = vec![0.0_f64; n * n * c];
            for i in 0..n {
                for j in 0..n {
                    for ch in 0..c {
                        let a_col: Vec<f64> =
                            (0..n).map(|k| proj_a[(i * n + k) * c + ch]).collect();
                        let b_col: Vec<f64> =
                            (0..n).map(|k| proj_b[(j * n + k) * c + ch]).collect();
                        out[(i * n + j) * c + ch] =
                            a_col.iter().zip(b_col.iter()).map(|(a, b)| a * b).sum();
                    }
                }
            }
            out
        },
    );

    h.check_bool(
        "Node coral trimul: correct shape",
        result.len() == n * n * c,
    );
    h.check_bool(
        "Node coral trimul: all finite",
        result.iter().all(|v| v.is_finite()),
    );
    h.check_bool(
        "Node coral trimul: GPU routing",
        substrate == MixedSubstrate::GpuOnly,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Node: coralForge confidence head dispatch
// ═══════════════════════════════════════════════════════════════════

pub fn validate_node_coral_confidence(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
    let n_res = 8;
    let d = 4;

    let repr: Vec<f64> = (0..n_res * d).map(|_| rng.normal() * 0.5).collect();
    let w: Vec<f64> = (0..d).map(|_| rng.normal() * 0.3).collect();
    let bias = rng.normal() * 0.1;

    let wl = MixedWorkload {
        op: "alphafold3_pldt_head",
        compute_us: 100_000.0,
        data_bytes: ((n_res * d) * 8) as u64,
        npu_available: false,
        needs_realtime: false,
    };

    let (logits, substrate) = disp.mixed_dispatch(
        &wl,
        |dev| {
            barracuda::dispatch::matmul_dispatch(&repr, &w, n_res, d, 1, Some(dev))
                .map_err(|e| format!("{e}"))
        },
        || rect_matmul_cpu(&repr, &w, n_res, d, 1),
    );

    let pldt: Vec<f64> = logits
        .iter()
        .map(|&l| primitives::sigmoid(l + bias))
        .collect();

    h.check_bool(
        "Node coral confidence: pLDDT in [0,1]",
        pldt.iter().all(|&v| (0.0..=1.0).contains(&v)),
    );
    h.check_bool(
        "Node coral confidence: GPU routing",
        substrate == MixedSubstrate::GpuOnly,
    );

    let mean_conf = disp.mean(&pldt);
    h.check_bool(
        "Node coral confidence: mean in valid range",
        (0.0..=1.0).contains(&mean_conf),
    );
}

// ═══════════════════════════════════════════════════════════════════
// Nest: WDM result provenance + entropy tracking
// ═══════════════════════════════════════════════════════════════════

pub fn validate_nest_wdm_provenance(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
    let predictions: Vec<f64> = (0..64).map(|_| rng.normal().abs()).collect();

    let cpu_mean = Dispatcher::cpu_only().mean(&predictions);
    let gpu_mean = disp.mean(&predictions);

    h.check_abs(
        "Nest WDM: mean GPU↔CPU provenance parity",
        gpu_mean,
        cpu_mean,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );

    let probs: Vec<f64> = {
        let raw: Vec<f64> = predictions
            .iter()
            .map(|v| v.abs() + primitives::POSITIVE_DATA_GUARD)
            .collect();
        let sum: f64 = raw.iter().sum();
        raw.iter().map(|&r| r / sum).collect()
    };

    let gpu_entropy = disp.shannon_entropy(&probs);
    let cpu_entropy = Dispatcher::cpu_only().shannon_entropy(&probs);

    h.check_abs(
        "Nest WDM: entropy GPU↔CPU provenance parity",
        gpu_entropy,
        cpu_entropy,
        tolerances::GPU_ENTROPY_F64,
    );
    h.check_bool(
        "Nest WDM: entropy > 0 (informative predictions)",
        gpu_entropy > 0.0,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Mixed routing: WDM workload size thresholds
// ═══════════════════════════════════════════════════════════════════

pub fn validate_mixed_wdm_routing(h: &mut ValidationHarness, disp: &Dispatcher) {
    let data = [1.0, 2.0, 3.0, 4.0];

    let small_wl = MixedWorkload {
        op: "wdm_small_inference",
        compute_us: 50.0,
        data_bytes: 256,
        npu_available: false,
        needs_realtime: false,
    };
    let (_, sub_small) = disp.mixed_dispatch(
        &small_wl,
        |dev| {
            let f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let t = barracuda::tensor::Tensor::from_data(&f32, vec![4], dev.clone())
                .map_err(|e| format!("{e}"))?;
            let v = t
                .mean()
                .map_err(|e| format!("{e}"))?
                .to_vec()
                .map_err(|e| format!("{e}"))?;
            Ok(f64::from(v[0]))
        },
        || data.iter().sum::<f64>() / data.len() as f64,
    );
    h.check_bool(
        "Mixed WDM: small → CPU",
        sub_small == MixedSubstrate::CpuOnly,
    );

    let large_wl = MixedWorkload {
        op: "wdm_batch_inference",
        compute_us: 500_000.0,
        data_bytes: 8_388_608,
        npu_available: false,
        needs_realtime: false,
    };
    let (_, sub_large) = disp.mixed_dispatch(
        &large_wl,
        |dev| {
            let f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let t = barracuda::tensor::Tensor::from_data(&f32, vec![4], dev.clone())
                .map_err(|e| format!("{e}"))?;
            let v = t
                .mean()
                .map_err(|e| format!("{e}"))?
                .to_vec()
                .map_err(|e| format!("{e}"))?;
            Ok(f64::from(v[0]))
        },
        || data.iter().sum::<f64>() / data.len() as f64,
    );
    h.check_bool(
        "Mixed WDM: large batch → GPU",
        sub_large == MixedSubstrate::GpuOnly,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Mixed routing: coralForge realtime folding → NPU
// ═══════════════════════════════════════════════════════════════════

pub fn validate_mixed_coral_routing(h: &mut ValidationHarness, disp: &Dispatcher) {
    let data = [0.5, 1.0, 1.5, 2.0];

    let realtime_wl = MixedWorkload {
        op: "evoformer_realtime_folding",
        compute_us: 300_000.0,
        data_bytes: 4_194_304,
        npu_available: true,
        needs_realtime: true,
    };
    let (_, sub_rt) = disp.mixed_dispatch(
        &realtime_wl,
        |dev| {
            let f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let t = barracuda::tensor::Tensor::from_data(&f32, vec![4], dev.clone())
                .map_err(|e| format!("{e}"))?;
            let v = t
                .mean()
                .map_err(|e| format!("{e}"))?
                .to_vec()
                .map_err(|e| format!("{e}"))?;
            Ok(f64::from(v[0]))
        },
        || data.iter().sum::<f64>() / data.len() as f64,
    );
    h.check_bool(
        "Mixed coral: realtime folding → GpuToNpu",
        sub_rt == MixedSubstrate::GpuToNpu,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Mixed: heterogeneous pipeline — WDM on GPU, then confidence on CPU
// ═══════════════════════════════════════════════════════════════════

pub fn validate_mixed_heterogeneous_pipeline(
    h: &mut ValidationHarness,
    disp: &Dispatcher,
    rng: &mut Rng,
) {
    let batch = 4;
    let d = 3;
    let input: Vec<f64> = (0..batch * d).map(|_| rng.normal()).collect();
    let w: Vec<f64> = (0..d * d).map(|_| rng.normal() * 0.3).collect();

    let wl_compute = MixedWorkload {
        op: "wdm_mlp_compute_phase",
        compute_us: 200_000.0,
        data_bytes: ((batch * d + d * d) * 8) as u64,
        npu_available: false,
        needs_realtime: false,
    };

    let (compute_result, sub_compute) = disp.mixed_dispatch(
        &wl_compute,
        |dev| {
            barracuda::dispatch::matmul_dispatch(&input, &w, batch, d, d, Some(dev))
                .map_err(|e| format!("{e}"))
        },
        || rect_matmul_cpu(&input, &w, batch, d, d),
    );

    h.check_bool(
        "Heterogeneous: GPU compute phase succeeded",
        compute_result.len() == batch * d,
    );
    h.check_bool(
        "Heterogeneous: compute → GPU",
        sub_compute == MixedSubstrate::GpuOnly,
    );

    let wl_postprocess = MixedWorkload {
        op: "confidence_postprocess",
        compute_us: 10.0,
        data_bytes: (batch * d * 8) as u64,
        npu_available: false,
        needs_realtime: false,
    };

    let (post_result, sub_post) = disp.mixed_dispatch(
        &wl_postprocess,
        |dev| {
            let f32: Vec<f32> = compute_result.iter().map(|&v| v as f32).collect();
            let t = barracuda::tensor::Tensor::from_data(&f32, vec![batch * d], dev.clone())
                .map_err(|e| format!("{e}"))?;
            let v = t
                .mean()
                .map_err(|e| format!("{e}"))?
                .to_vec()
                .map_err(|e| format!("{e}"))?;
            Ok(f64::from(v[0]))
        },
        || compute_result.iter().sum::<f64>() / compute_result.len() as f64,
    );

    h.check_bool("Heterogeneous: postprocess finite", post_result.is_finite());
    h.check_bool(
        "Heterogeneous: postprocess → CPU (small)",
        sub_post == MixedSubstrate::CpuOnly,
    );
}

// ═══════════════════════════════════════════════════════════════════
// PCIe bypass: GPU→NPU transfer cost for folding workloads
// ═══════════════════════════════════════════════════════════════════

pub fn validate_pcie_folding_bypass(h: &mut ValidationHarness) {
    let gpu_npu = PcieBridge::new("RTX_4070", "AKD1000_NPU");
    let gpu_cpu = PcieBridge::new("RTX_4070", "x86_64_CPU");

    let wdm_sizes: [u64; 3] = [4_096, 262_144, 4_194_304];
    for &sz in &wdm_sizes {
        let direct = gpu_npu.transfer_cost(sz).estimated_us();
        let staged = gpu_cpu.transfer_cost(sz).estimated_us();

        h.check_bool(
            &format!("PCIe WDM: transfer cost > 0 @ {sz}B"),
            direct > 0.0 && staged > 0.0,
        );
    }

    let small = gpu_npu.transfer_cost(4_096).estimated_us();
    let large = gpu_npu.transfer_cost(4_194_304).estimated_us();
    h.check_bool(
        "PCIe folding: large transfer costs more than small",
        large > small,
    );

    let direct_cost = gpu_npu.transfer_cost(1_048_576).estimated_us();
    let staged_cost = gpu_cpu.transfer_cost(1_048_576).estimated_us()
        + PcieBridge::new("x86_64_CPU", "AKD1000_NPU")
            .transfer_cost(1_048_576)
            .estimated_us();

    h.check_bool(
        "PCIe bypass: GPU→NPU direct cheaper than GPU→CPU→NPU staged",
        direct_cost < staged_cost,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

pub fn rect_matmul_cpu(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k).fold(0.0, |acc, p| a[i * k + p].mul_add(b[p * n + j], acc));
        }
    }
    c
}
