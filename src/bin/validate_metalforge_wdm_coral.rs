// SPDX-License-Identifier: AGPL-3.0-or-later

//! metalForge mixed-hardware validation for WDM + coralForge domain workloads.
//!
//! Proves that WDM surrogate inference and coralForge Evoformer operations
//! route correctly through the metalForge mixed-hardware substrate model,
//! with NUCLEUS atomic coordination (tower discovery, node compute, nest
//! provenance) and PCIe bypass cost modelling for GPU→NPU→CPU paths.
//!
//! ## NUCLEUS atomics exercised
//!
//! - **Tower**: substrate discovery, capability enumeration
//! - **Node**: GPU compute dispatch for WDM MLP + coralForge attention
//! - **Nest**: result provenance transfer, entropy tracking
//!
//! ## metalForge routing scenarios
//!
//! - Small WDM inference → CPU (dispatch overhead dominates)
//! - Large WDM batch → GPU (compute dominates)
//! - Realtime folding → GPU→NPU (latency-critical)
//! - Mixed pipeline: WDM on GPU + confidence on CPU (heterogeneous)
//!
//! ## PCIe bypass
//!
//! - GPU→NPU direct transfer cost vs GPU→CPU→NPU staged
//! - Bandwidth tier detection from adapter name

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::suboptimal_flops,
    clippy::suspicious_operation_groupings
)]

use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::{Dispatcher, MixedWorkload};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{exit_no_gpu, ValidationHarness};
use neural_spring_forge::inventory;
use neural_spring_forge::mixed::MixedSubstrate;
use neural_spring_forge::pcie_bridge::PcieBridge;
use neural_spring_forge::substrate::SubstrateKind;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("metalforge_wdm_coral");

    let Ok(gpu) = Gpu::new().await else {
        exit_no_gpu();
    };
    let disp = Dispatcher::from_gpu(gpu);

    let mut rng = Rng::new(42);

    // Tower: substrate discovery for WDM/coralForge workloads
    validate_tower_wdm_coral(&mut h);

    // Node: WDM surrogate compute dispatch
    validate_node_wdm_transport(&mut h, &disp, &mut rng);
    validate_node_wdm_eos(&mut h, &disp, &mut rng);
    validate_node_wdm_sqw_lstm(&mut h, &disp, &mut rng);
    validate_node_wdm_transfer_mlp(&mut h, &disp, &mut rng);
    validate_node_wdm_esn_spectral(&mut h, &disp, &mut rng);

    // Node: coralForge Evoformer compute dispatch
    validate_node_coral_attention(&mut h, &disp, &mut rng);
    validate_node_coral_trimul(&mut h, &disp, &mut rng);
    validate_node_coral_confidence(&mut h, &disp, &mut rng);

    // Nest: provenance and entropy tracking
    validate_nest_wdm_provenance(&mut h, &disp, &mut rng);

    // Mixed routing: heterogeneous domain pipelines
    validate_mixed_wdm_routing(&mut h, &disp);
    validate_mixed_coral_routing(&mut h, &disp);
    validate_mixed_heterogeneous_pipeline(&mut h, &disp, &mut rng);

    // PCIe bypass: GPU→NPU direct transfer for folding workloads
    validate_pcie_folding_bypass(&mut h);

    h.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Tower: substrate discovery for WDM/coralForge domain workloads
// ═══════════════════════════════════════════════════════════════════

fn validate_tower_wdm_coral(h: &mut ValidationHarness) {
    let substrates = inventory::discover();

    let has_gpu = substrates.iter().any(|s| s.kind == SubstrateKind::Gpu);
    let has_cpu = substrates.iter().any(|s| s.kind == SubstrateKind::Cpu);

    h.check_bool("Tower: GPU discovered for WDM/coral", has_gpu);
    h.check_bool("Tower: CPU discovered for fallback", has_cpu);
    h.check_bool("Tower: multi-substrate available", substrates.len() >= 2);

    for s in &substrates {
        h.check_bool(
            &format!("Tower: '{}' capabilities non-empty", s.identity.name),
            !s.capabilities.is_empty() || s.kind == SubstrateKind::Cpu,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// Node: WDM transport MLP inference via mixed dispatch
// ═══════════════════════════════════════════════════════════════════

fn validate_node_wdm_transport(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
    let batch = 8;
    let d_in = 2;
    let d_h = 16;

    let input: Vec<f64> = (0..batch * d_in).map(|_| rng.normal()).collect();
    let w: Vec<f64> = (0..d_in * d_h).map(|_| rng.normal() * 0.3).collect();

    let wl = MixedWorkload {
        op: "wdm_transport_mlp_layer1",
        compute_us: 200_000.0,
        data_bytes: ((batch * d_in + d_in * d_h) * 8) as u64,
        npu_available: false,
        needs_realtime: false,
    };

    let (result, substrate) = disp.mixed_dispatch(
        &wl,
        |dev| {
            barracuda::dispatch::matmul_dispatch(&input, &w, batch, d_in, d_h, Some(dev))
                .map_err(|e| format!("{e}"))
        },
        || rect_matmul_cpu(&input, &w, batch, d_in, d_h),
    );

    h.check_bool(
        "Node WDM transport: result correct shape",
        result.len() == batch * d_h,
    );
    h.check_bool(
        "Node WDM transport: all finite",
        result.iter().all(|v| v.is_finite()),
    );
    h.check_bool(
        "Node WDM transport: routed to GPU (large compute)",
        substrate == MixedSubstrate::GpuOnly,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Node: WDM EOS via mixed dispatch
// ═══════════════════════════════════════════════════════════════════

fn validate_node_wdm_eos(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
    let n = 4;
    let input: Vec<f64> = (0..n).map(|_| rng.normal()).collect();

    let wl_small = MixedWorkload {
        op: "wdm_eos_single_point",
        compute_us: 50.0,
        data_bytes: (n * 8) as u64,
        npu_available: false,
        needs_realtime: false,
    };

    let cpu_result = barracuda::stats::mean(&input);
    let (mixed_result, substrate) = disp.mixed_dispatch(
        &wl_small,
        |dev| {
            let f32_data: Vec<f32> = input.iter().map(|&v| v as f32).collect();
            let t = barracuda::tensor::Tensor::from_data(&f32_data, vec![n], dev.clone())
                .map_err(|e| format!("{e}"))?;
            let mean = t.mean().map_err(|e| format!("{e}"))?;
            let v = mean.to_vec().map_err(|e| format!("{e}"))?;
            Ok(f64::from(v[0]))
        },
        || cpu_result,
    );

    h.check_abs(
        "Node WDM EOS: small → correct result",
        mixed_result,
        cpu_result,
        0.01,
    );
    h.check_bool(
        "Node WDM EOS: small → CPU (overhead dominates)",
        substrate == MixedSubstrate::CpuOnly,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Node: WDM ESN spectral analysis via eigensolve dispatch
// ═══════════════════════════════════════════════════════════════════

fn validate_node_wdm_esn_spectral(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
    let n = 12;
    let raw: Vec<f64> = (0..n * n).map(|_| rng.normal() * 0.3).collect();
    let sym: Vec<f64> = {
        let mut s = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let v = 0.5 * (raw[i * n + j] + raw[j * n + i]);
                s[i * n + j] = v;
                s[j * n + i] = v;
            }
        }
        s
    };

    let (evals, _) = disp.eigh(&sym, n);

    h.check_bool(
        "Node ESN spectral: correct eigenvalue count",
        evals.len() == n,
    );
    h.check_bool(
        "Node ESN spectral: all finite",
        evals.iter().all(|v| v.is_finite()),
    );

    let spectral_radius = evals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    h.check_bool("Node ESN spectral: radius > 0", spectral_radius > 0.0);

    let eval_var = disp.variance(&evals);
    h.check_bool("Node ESN spectral: eigenvalue variance > 0", eval_var > 0.0);
}

// ═══════════════════════════════════════════════════════════════════
// Node: WDM SQW LSTM gate computation via mixed dispatch (nW-03)
// ═══════════════════════════════════════════════════════════════════

fn validate_node_wdm_sqw_lstm(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
    let hs = 4;
    let input_size = 1;

    let w_i: Vec<f64> = (0..4 * hs * input_size)
        .map(|_| rng.normal() * 0.2)
        .collect();
    let w_h: Vec<f64> = (0..4 * hs * hs).map(|_| rng.normal() * 0.1).collect();
    let b: Vec<f64> = (0..4 * hs).map(|_| rng.normal() * 0.05).collect();

    let x_input = vec![0.7_f64; input_size];
    let h_prev = vec![0.0_f64; hs];

    // LSTM gate: pre-activation = w_i × x + w_h × h_prev + b
    let wl = MixedWorkload {
        op: "wdm_sqw_lstm_gate",
        compute_us: 200_000.0,
        data_bytes: ((4 * hs * input_size + 4 * hs * hs + hs) * 8) as u64,
        npu_available: false,
        needs_realtime: false,
    };

    let cpu_input_proj = rect_matmul_cpu(&w_i, &x_input, 4 * hs, input_size, 1);
    let cpu_hidden_proj = rect_matmul_cpu(&w_h, &h_prev, 4 * hs, hs, 1);
    let cpu_gates: Vec<f64> = cpu_input_proj
        .iter()
        .zip(cpu_hidden_proj.iter())
        .zip(b.iter())
        .map(|((i, h), b)| i + h + b)
        .collect();

    let (mixed_input_proj, substrate) = disp.mixed_dispatch(
        &wl,
        |dev| {
            barracuda::dispatch::matmul_dispatch(&w_i, &x_input, 4 * hs, input_size, 1, Some(dev))
                .map_err(|e| format!("{e}"))
        },
        || cpu_input_proj.clone(),
    );

    h.check_bool(
        "Node WDM SQW: LSTM gate shape",
        mixed_input_proj.len() == 4 * hs,
    );
    h.check_bool(
        "Node WDM SQW: GPU routing (LSTM compute)",
        substrate == MixedSubstrate::GpuOnly,
    );

    let max_diff = cpu_gates
        .iter()
        .zip(
            mixed_input_proj
                .iter()
                .zip(cpu_hidden_proj.iter())
                .zip(b.iter())
                .map(|((i, h), b)| i + h + b),
        )
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "Node WDM SQW: gate pre-activations parity",
        max_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Node: WDM Transfer classical→WDM MLP via mixed dispatch (nW-04)
// ═══════════════════════════════════════════════════════════════════

fn validate_node_wdm_transfer_mlp(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
    let classical_dim = 3;
    let hid = 6;
    let wdm_dim = 2;

    let w1: Vec<f64> = (0..hid * classical_dim)
        .map(|_| rng.normal() * 0.3)
        .collect();
    let b1: Vec<f64> = (0..hid).map(|_| rng.normal() * 0.1).collect();
    let w2: Vec<f64> = (0..wdm_dim * hid).map(|_| rng.normal() * 0.3).collect();
    let b2: Vec<f64> = (0..wdm_dim).map(|_| rng.normal() * 0.1).collect();
    let x: Vec<f64> = (0..classical_dim).map(|_| rng.normal()).collect();

    let wl = MixedWorkload {
        op: "wdm_transfer_mlp_forward",
        compute_us: 200_000.0,
        data_bytes: ((hid * classical_dim + wdm_dim * hid) * 8) as u64,
        npu_available: false,
        needs_realtime: false,
    };

    // CPU forward
    let cpu_hidden = rect_matmul_cpu(&w1, &x, hid, classical_dim, 1);
    let cpu_act: Vec<f64> = cpu_hidden
        .iter()
        .zip(b1.iter())
        .map(|(h, b)| (h + b).max(0.0))
        .collect();
    let cpu_out: Vec<f64> = {
        let o = rect_matmul_cpu(&w2, &cpu_act, wdm_dim, hid, 1);
        o.iter().zip(b2.iter()).map(|(o, b)| o + b).collect()
    };

    let (mixed_hidden, substrate) = disp.mixed_dispatch(
        &wl,
        |dev| {
            barracuda::dispatch::matmul_dispatch(&w1, &x, hid, classical_dim, 1, Some(dev))
                .map_err(|e| format!("{e}"))
        },
        || cpu_hidden.clone(),
    );

    let mixed_act: Vec<f64> = mixed_hidden
        .iter()
        .zip(b1.iter())
        .map(|(h, b)| (h + b).max(0.0))
        .collect();
    let mixed_out: Vec<f64> = {
        let o = rect_matmul_cpu(&w2, &mixed_act, wdm_dim, hid, 1);
        o.iter().zip(b2.iter()).map(|(o, b)| o + b).collect()
    };

    let out_diff = cpu_out
        .iter()
        .zip(mixed_out.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        "Node WDM Transfer: MLP output parity",
        out_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );
    h.check_bool(
        "Node WDM Transfer: GPU routing (large compute)",
        substrate == MixedSubstrate::GpuOnly,
    );
    h.check_bool(
        "Node WDM Transfer: all outputs finite",
        mixed_out.iter().all(|v| v.is_finite()),
    );
}

// ═══════════════════════════════════════════════════════════════════
// Node: coralForge Evoformer attention via mixed dispatch
// ═══════════════════════════════════════════════════════════════════

fn validate_node_coral_attention(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
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

fn validate_node_coral_trimul(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
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

fn validate_node_coral_confidence(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
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

    let pldt: Vec<f64> = logits.iter().map(|&l| sigmoid(l + bias)).collect();

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

fn validate_nest_wdm_provenance(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
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
        let raw: Vec<f64> = predictions.iter().map(|v| v.abs() + 1e-10).collect();
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

fn validate_mixed_wdm_routing(h: &mut ValidationHarness, disp: &Dispatcher) {
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

fn validate_mixed_coral_routing(h: &mut ValidationHarness, disp: &Dispatcher) {
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

fn validate_mixed_heterogeneous_pipeline(
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

fn validate_pcie_folding_bypass(h: &mut ValidationHarness) {
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

fn rect_matmul_cpu(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k).fold(0.0, |acc, p| a[i * k + p].mul_add(b[p * n + j], acc));
        }
    }
    c
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
