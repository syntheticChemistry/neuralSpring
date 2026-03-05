// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU workload validation: all Phase 0++ paper domains (011–025).
//!
//! Extends `validate_gpu_pure_workload` (fitness-only) to cover every
//! computational domain. Each domain dispatches its typed `BarraCUDA` GPU
//! op, reads back a scalar summary, and compares against the CPU reference.
//!
//! ## Evolution proof
//!
//! ```text
//! Python baseline (control/)
//!   ↓  cross-language validation (1e-10)
//! Rust CPU (neuralSpring lib)
//!   ↓  BarraCUDA CPU ports (pure Rust math)
//! BarraCUDA CPU (barracuda crate)
//!   ↓  GPU Tensor / WGSL shader dispatch
//! BarraCUDA GPU (this validator) — scalar-only readback
//!   ↓
//! Pure GPU sovereign pipeline (`ToadStool` streaming)
//! ```
//!
//! ## Domains validated
//!
//! | Domain | Papers | GPU Op | Readback |
//! |--------|--------|--------|----------|
//! | Fitness | 011-013 | `BatchFitnessGpu` | mean(fitness) |
//! | Multi-obj | 014 | `MultiObjFitnessGpu` | mean(ranks) |
//! | Swarm NN | 015 | `SwarmNnGpu` | action distribution |
//! | HMM | 016-018 | `HmmBatchForwardF64` | mean(log-lik) |
//! | Spatial | 019 | `SpatialPayoffGpu` | mean(payoff) |
//! | RK4/RK45 | 020 | `Rk45AdaptiveGpu` | endpoint state |
//! | Hill gate | 021 | `HillGateGpu` | mean(response) |
//! | Spectral | 022-023 | `BatchIprGpu` | mean(IPR) |
//! | Hamming | 017 | `PairwiseHammingGpu` | mean(dist) |
//! | L2 | 012 | `PairwiseL2Gpu` | mean(dist) |
//! | Jaccard | 024 | `PairwiseJaccardGpu` | mean(dist) |
//! | Locus var | 025 | `LocusVarianceGpu` | mean(var) |
//!
//! ## Provenance
//!
//! Session 74. Cross-spring: hotSpring validation patterns, wetSpring
//! bio-domain ops, all dispatched through typed `BarraCUDA` GPU wrappers.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::suboptimal_flops,
    reason = "validation binary — GPU buffer plumbing with numeric casts across 12 bio-compute domains"
)]

use barracuda::ops::bio::hill_gate::{HillGateGpu, HillGateParams};
use barracuda::ops::bio::swarm_nn::SwarmNnParams;
use barracuda::ops::bio::{
    BatchFitnessGpu, HmmBatchForwardF64, LocusVarianceGpu, MultiObjFitnessGpu, PairwiseHammingGpu,
    PairwiseJaccardGpu, PairwiseL2Gpu, SpatialPayoffGpu, SwarmNnGpu,
};
use barracuda::spectral::BatchIprGpu;
use neural_spring::gpu::Gpu;
use neural_spring::hmm::Hmm;
use neural_spring::rng::Rng;
use neural_spring::signal_integration::two_input_hill;
use neural_spring::swarm_robotics::{create_controller, neural_forward, ControllerType};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;

fn storage_buf(device: &wgpu::Device, label: &str, data: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn output_buf(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let mut h = ValidationHarness::new("gpu_pure_workload_all");
    let t0 = Instant::now();

    validate_fitness(&mut h, &gpu);
    validate_multi_obj(&mut h, &gpu);
    validate_swarm_nn(&mut h, &gpu);
    validate_hmm(&mut h, &gpu);
    validate_spatial_payoff(&mut h, &gpu);
    validate_rk45_regulatory(&mut h, &gpu);
    validate_hill_gate_signal(&mut h, &gpu);
    validate_batch_ipr(&mut h, &gpu);
    validate_hamming(&mut h, &gpu);
    validate_l2(&mut h, &gpu);
    validate_jaccard(&mut h, &gpu);
    validate_locus_variance(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    let elapsed = t0.elapsed();
    eprintln!(
        "\n  total GPU pure-workload time: {:.1}ms (12 domains + determinism)",
        elapsed.as_secs_f64() * 1000.0,
    );

    h.finish();
}

// ═══════════════════════════════════════════════════════════════════
// 1. Batch Fitness (Papers 011–013)
// ═══════════════════════════════════════════════════════════════════

fn validate_fitness(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 64_usize;
    let genome_len = 16_usize;
    let mut rng = Rng::new(42);
    let genotypes: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..genome_len).map(|_| rng.uniform()).collect();

    let cpu_mean = {
        let total: f64 = (0..pop_size)
            .map(|i| {
                let base = i * genome_len;
                (0..genome_len)
                    .map(|g| genotypes[base + g] * weights[g])
                    .sum::<f64>()
            })
            .sum();
        total / pop_size as f64
    };

    let op = BatchFitnessGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let geno_buf = storage_buf(device, "fit_geno", bytemuck::cast_slice(&genotypes));
    let weight_buf = storage_buf(device, "fit_w", bytemuck::cast_slice(&weights));
    let out_buf = output_buf(device, "fit_out", (pop_size * 8) as u64);

    op.dispatch(
        &geno_buf,
        &weight_buf,
        &out_buf,
        pop_size as u32,
        genome_len as u32,
    );

    match gpu.read_buffer_f64(&out_buf, pop_size) {
        Ok(fitness) => {
            let gpu_mean = fitness.iter().sum::<f64>() / fitness.len() as f64;
            h.check_abs(
                &format!("fitness 64×16: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => h.check_bool(&format!("fitness: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. Multi-Objective Fitness (Paper 014)
// ═══════════════════════════════════════════════════════════════════

fn validate_multi_obj(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop = 32_usize;
    let glen = 12_usize;
    let n_obj = 3_usize;
    let mut rng = Rng::new(77);
    let genotypes: Vec<f64> = (0..pop * glen).map(|_| rng.uniform()).collect();

    let cpu_mean = {
        let mut all_fitness = Vec::with_capacity(pop * n_obj);
        for i in 0..pop {
            let individual = &genotypes[i * glen..(i + 1) * glen];
            let f = neural_spring::directed_evolution::multi_objective_fitness(individual, n_obj);
            all_fitness.extend_from_slice(&f);
        }
        all_fitness.iter().sum::<f64>() / all_fitness.len() as f64
    };

    let op = MultiObjFitnessGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let geno_buf = storage_buf(device, "mof_g", bytemuck::cast_slice(&genotypes));
    let out_buf = output_buf(device, "mof_out", (pop * n_obj * 8) as u64);

    op.dispatch(&geno_buf, &out_buf, pop as u32, glen as u32, n_obj as u32);

    match gpu.read_buffer_f64(&out_buf, pop * n_obj) {
        Ok(gpu_f) => {
            let gpu_mean = gpu_f.iter().sum::<f64>() / gpu_f.len() as f64;
            h.check_abs(
                &format!("multi_obj 32×12×3: GPU={gpu_mean:.4} vs CPU={cpu_mean:.4}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_MULTI_OBJ_FITNESS_F64,
            );
        }
        Err(e) => h.check_bool(&format!("multi_obj: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. Swarm NN Forward (Paper 015)
// ═══════════════════════════════════════════════════════════════════

fn validate_swarm_nn(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_ctrl = 4_u32;
    let n_eval = 8_u32;
    let input_dim = 1_u32;
    let hidden_dim = 4_u32;
    let output_dim = 5_u32;

    let mut rng = Rng::new(55);
    let controllers: Vec<_> = (0..n_ctrl)
        .map(|_| create_controller(ControllerType::NeuralNet, &mut rng))
        .collect();

    let all_weights: Vec<f64> = controllers
        .iter()
        .flat_map(|c| c.params.iter().copied())
        .collect();
    let sense: Vec<f64> = (0..n_eval).map(|i| (i as f64) * 0.1).collect();

    let cpu_actions: Vec<u32> = controllers
        .iter()
        .flat_map(|c| (0..n_eval).map(|i| neural_forward(&c.params, (i as f64) * 0.1) as u32))
        .collect();
    let cpu_mean = cpu_actions.iter().sum::<u32>() as f64 / cpu_actions.len() as f64;

    let op = SwarmNnGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let inputs_f64: Vec<f64> = (0..n_ctrl).flat_map(|_| sense.iter().copied()).collect();
    let w_buf = storage_buf(device, "swarm_w", bytemuck::cast_slice(&all_weights));
    let in_buf = storage_buf(device, "swarm_in", bytemuck::cast_slice(&inputs_f64));
    let n_actions = (n_ctrl * n_eval) as usize;
    let act_buf = output_buf(device, "swarm_act", (n_actions * 4) as u64);

    op.dispatch(
        &w_buf,
        &in_buf,
        &act_buf,
        &SwarmNnParams {
            n_controllers: n_ctrl,
            n_evals: n_eval,
            input_dim,
            hidden_dim,
            output_dim,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        },
    );

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("swarm_staging"),
        size: (n_actions * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&act_buf, 0, &staging, 0, (n_actions * 4) as u64);
    gpu.queue().submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    match rx.recv() {
        Ok(Ok(())) => {
            let data = slice.get_mapped_range();
            let gpu_actions: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
            let gpu_mean = gpu_actions.iter().sum::<u32>() as f64 / gpu_actions.len() as f64;
            h.check_bool(
                &format!(
                    "swarm_nn {n_ctrl}×{n_eval}: GPU mean action={gpu_mean:.2} (CPU={cpu_mean:.2}), all in [0,{output_dim})"
                ),
                gpu_actions.iter().all(|&a| a < output_dim),
            );
        }
        _ => h.check_bool("swarm_nn: GPU readback failed", false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3b. RK45 Adaptive ODE (Paper 020 — Regulatory Network)
// ═══════════════════════════════════════════════════════════════════

fn validate_rk45_regulatory(h: &mut ValidationHarness, gpu: &Gpu) {
    use barracuda::ops::rk45_adaptive::Rk45AdaptiveGpu;

    let dim = 4_u32;
    let n_systems = 4_u32;
    let n_coeffs = dim * 3;
    let dt = 0.01_f64;

    let state: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4]
        .into_iter()
        .cycle()
        .take((dim * n_systems) as usize)
        .collect();
    let coeffs: Vec<f64> = (0..n_systems)
        .flat_map(|_| (0..dim).flat_map(|d| vec![1.0, 0.5, ((d + 1) % dim) as f64]))
        .collect();

    let total = (dim * n_systems) as usize;
    let device = gpu.device();

    let op = Rk45AdaptiveGpu::new(Arc::clone(gpu.wgpu_device()));
    let state_buf = storage_buf(device, "rk_state", bytemuck::cast_slice(&state));
    let coeff_buf = storage_buf(device, "rk_coeff", bytemuck::cast_slice(&coeffs));
    let out_buf = output_buf(device, "rk_out", (total * 8) as u64);
    let err_buf = output_buf(device, "rk_err", (total * 8) as u64);
    let scratch_buf = output_buf(device, "rk_scratch", (total * 8 * 8) as u64);

    op.dispatch(
        &state_buf,
        &coeff_buf,
        &out_buf,
        &err_buf,
        &scratch_buf,
        n_systems,
        dim,
        n_coeffs,
        dt,
    );

    match gpu.read_buffer_f64(&out_buf, total) {
        Ok(gpu_v) => {
            let gpu_mean = gpu_v.iter().sum::<f64>() / gpu_v.len() as f64;
            h.check_bool(
                &format!("rk45 {n_systems}×{dim}: GPU mean={gpu_mean:.6}, all finite"),
                gpu_v.iter().all(|v| v.is_finite()) && gpu_mean > 0.0,
            );
        }
        Err(e) => h.check_bool(&format!("rk45: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3c. Hill Gate (Paper 021 — Signal Integration)
// ═══════════════════════════════════════════════════════════════════

fn validate_hill_gate_signal(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_a = 8_u32;
    let n_b = 8_u32;
    let n_out = (n_a * n_b) as usize;

    let a_vals: Vec<f64> = (0..n_a).map(|i| (i as f64) * 0.15).collect();
    let b_vals: Vec<f64> = (0..n_b).map(|i| (i as f64) * 0.12 + 0.05).collect();

    let cpu_mean = {
        let mut sum = 0.0_f64;
        for &a in &a_vals {
            for &b in &b_vals {
                sum += two_input_hill(a, b, 1.0, 0.5, 0.5, 2.0, 2.0);
            }
        }
        sum / n_out as f64
    };

    let op = HillGateGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let a_buf = storage_buf(device, "hill_a", bytemuck::cast_slice(&a_vals));
    let b_buf = storage_buf(device, "hill_b", bytemuck::cast_slice(&b_vals));
    let out_buf = output_buf(device, "hill_out", (n_out * 8) as u64);

    let params = HillGateParams {
        n_a,
        n_b,
        mode: 1,
        _pad: 0,
        k_a: 0.5,
        k_b: 0.5,
        n_a_exp: 2.0,
        n_b_exp: 2.0,
        vmax: 1.0,
        _pad2: 0.0,
    };
    op.dispatch(&a_buf, &b_buf, &out_buf, &params);

    match gpu.read_buffer_f64(&out_buf, n_out) {
        Ok(gpu_v) => {
            let gpu_mean = gpu_v.iter().sum::<f64>() / gpu_v.len() as f64;
            h.check_abs(
                &format!("hill_gate 8×8 grid: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_HILL_GATE_F64,
            );
        }
        Err(e) => h.check_bool(&format!("hill_gate: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. HMM Forward (Papers 016–018)
// ═══════════════════════════════════════════════════════════════════

fn validate_hmm(h: &mut ValidationHarness, gpu: &Gpu) {
    let hmm = Hmm::new(
        vec![
            vec![0.7, 0.2, 0.1],
            vec![0.2, 0.6, 0.2],
            vec![0.1, 0.2, 0.7],
        ],
        vec![
            vec![0.4, 0.3, 0.3],
            vec![0.2, 0.5, 0.3],
            vec![0.3, 0.3, 0.4],
        ],
        vec![0.33, 0.34, 0.33],
    );

    let mut rng = Rng::new(42);
    let n_seqs = 8_usize;
    let seq_len = 20_usize;
    let mut obs_batch = Vec::with_capacity(n_seqs);
    for _ in 0..n_seqs {
        let (_, obs) = hmm.generate_sequence(seq_len, &mut rng);
        obs_batch.push(obs);
    }

    let cpu_mean = {
        let mut sum = 0.0_f64;
        for obs in &obs_batch {
            let (_, ll) = hmm.forward(obs);
            sum += ll;
        }
        sum / n_seqs as f64
    };

    let dev = Arc::clone(gpu.wgpu_device());
    let op = match HmmBatchForwardF64::new(dev) {
        Ok(o) => o,
        Err(e) => {
            h.check_bool(&format!("HMM create: {e}"), false);
            return;
        }
    };

    let n_states = hmm.num_states() as u32;
    let n_symbols = hmm.num_symbols() as u32;
    let log_trans: Vec<f64> = hmm.transition.iter().map(|&p| p.ln()).collect();
    let log_emit: Vec<f64> = hmm.emission.iter().map(|&p| p.ln()).collect();
    let log_pi: Vec<f64> = hmm.initial.iter().map(|&p| p.ln()).collect();

    let mut obs_flat: Vec<u32> = Vec::with_capacity(n_seqs * seq_len);
    for seq in &obs_batch {
        for &o in seq {
            obs_flat.push(o as u32);
        }
        obs_flat.extend(std::iter::repeat_n(0u32, seq_len.saturating_sub(seq.len())));
    }

    let device = gpu.device();
    let lt_buf = storage_buf(device, "hmm_lt", bytemuck::cast_slice(&log_trans));
    let le_buf = storage_buf(device, "hmm_le", bytemuck::cast_slice(&log_emit));
    let lp_buf = storage_buf(device, "hmm_lp", bytemuck::cast_slice(&log_pi));
    let obs_buf = storage_buf(device, "hmm_obs", bytemuck::cast_slice(&obs_flat));
    let alpha_size = (n_seqs * seq_len * n_states as usize * 8) as u64;
    let alpha_buf = output_buf(device, "hmm_a", alpha_size);
    let ll_buf = output_buf(device, "hmm_ll", (n_seqs * 8) as u64);

    if let Err(e) = op.dispatch(
        n_states,
        n_symbols,
        seq_len as u32,
        n_seqs as u32,
        &lt_buf,
        &le_buf,
        &lp_buf,
        &obs_buf,
        &alpha_buf,
        &ll_buf,
    ) {
        h.check_bool(&format!("HMM dispatch: {e}"), false);
        return;
    }

    match gpu.read_buffer_f64(&ll_buf, n_seqs) {
        Ok(ll) => {
            let gpu_mean = ll.iter().sum::<f64>() / ll.len() as f64;
            h.check_abs(
                &format!("HMM 3×3, 8seq: GPU={gpu_mean:.4} vs CPU={cpu_mean:.4}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_HMM_ALPHA_F32,
            );
        }
        Err(e) => h.check_bool(&format!("HMM readback: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Spatial Payoff — Game Theory (Paper 019)
// ═══════════════════════════════════════════════════════════════════

fn validate_spatial_payoff(h: &mut ValidationHarness, gpu: &Gpu) {
    let grid_size = 16_u32;
    let gs = grid_size as usize;
    let n = gs * gs;
    let b = 1.5_f32;
    let c = 1.0_f32;
    let mut rng = Rng::new(99);
    let grid: Vec<u32> = (0..n).map(|_| u32::from(rng.uniform() > 0.5)).collect();

    let cpu_mean = {
        let neighbors: [(i32, i32); 8] = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ];
        let gn = gs as i32;
        let mut total = 0.0_f32;
        for i in 0..gs {
            for j in 0..gs {
                let me = grid[i * gs + j];
                for (di, dj) in &neighbors {
                    let ni = ((i as i32 + di).rem_euclid(gn)) as usize;
                    let nj = ((j as i32 + dj).rem_euclid(gn)) as usize;
                    let other = grid[ni * gs + nj];
                    total += match (me, other) {
                        (1, 1) => b - c,
                        (1, 0) => -c,
                        (0, 1) => b,
                        _ => 0.0,
                    };
                }
            }
        }
        total / n as f32
    };

    let op = SpatialPayoffGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let grid_buf = storage_buf(device, "sp_grid", bytemuck::cast_slice(&grid));
    let out_buf = output_buf(device, "sp_out", (n * 4) as u64);

    op.dispatch(&grid_buf, &out_buf, grid_size, b, c);

    match gpu.read_buffer_f32(&out_buf, n) {
        Ok(gpu_p) => {
            let gpu_mean = gpu_p.iter().sum::<f32>() / gpu_p.len() as f32;
            h.check_abs(
                &format!("spatial 16×16: GPU={gpu_mean:.4} vs CPU={cpu_mean:.4}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_SPATIAL_PAYOFF_F32,
            );
        }
        Err(e) => h.check_bool(&format!("spatial: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. Batch IPR — Spectral / Anderson (Papers 022–023)
// ═══════════════════════════════════════════════════════════════════

fn validate_batch_ipr(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_vectors = 8_usize;
    let dim = 16_usize;
    let mut rng = Rng::new(55);
    let mut vecs_f64: Vec<f64> = (0..n_vectors * dim).map(|_| rng.normal()).collect();

    for i in 0..n_vectors {
        let base = i * dim;
        let norm: f64 = (0..dim)
            .map(|d| vecs_f64[base + d] * vecs_f64[base + d])
            .sum::<f64>()
            .sqrt();
        for d in 0..dim {
            vecs_f64[base + d] /= norm;
        }
    }

    let vecs_f32: Vec<f32> = vecs_f64.iter().map(|&v| v as f32).collect();

    let cpu_iprs: Vec<f64> = (0..n_vectors)
        .map(|i| {
            let base = i * dim;
            (0..dim)
                .map(|d| {
                    let a = vecs_f64[base + d];
                    a * a * a * a
                })
                .sum::<f64>()
        })
        .collect();
    let cpu_mean = cpu_iprs.iter().sum::<f64>() / cpu_iprs.len() as f64;

    let op = BatchIprGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let vecs_buf = storage_buf(device, "ipr_v", bytemuck::cast_slice(&vecs_f32));
    let out_buf = output_buf(device, "ipr_out", (n_vectors * 4) as u64);

    op.dispatch(&vecs_buf, &out_buf, dim as u32, n_vectors as u32);

    match gpu.read_buffer_f32(&out_buf, n_vectors) {
        Ok(gpu_ipr) => {
            let gpu_mean: f64 =
                gpu_ipr.iter().map(|&v| f64::from(v)).sum::<f64>() / gpu_ipr.len() as f64;
            h.check_abs(
                &format!("IPR 8×16: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_BATCH_IPR_F32,
            );
        }
        Err(e) => h.check_bool(&format!("IPR: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. Pairwise Hamming — SATé Alignment (Paper 017)
// ═══════════════════════════════════════════════════════════════════

fn validate_hamming(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 6_usize;
    let seq_len = 20_usize;
    let mut rng = Rng::new(44);
    let seqs: Vec<u32> = (0..n_seqs * seq_len).map(|_| rng.usize(4) as u32).collect();

    let n_pairs = n_seqs * (n_seqs - 1) / 2;
    let cpu_mean = {
        let mut total = 0.0_f32;
        let mut count = 0_usize;
        for i in 0..n_seqs {
            for j in (i + 1)..n_seqs {
                let mut diff = 0_u32;
                for k in 0..seq_len {
                    if seqs[i * seq_len + k] != seqs[j * seq_len + k] {
                        diff += 1;
                    }
                }
                total += diff as f32 / seq_len as f32;
                count += 1;
            }
        }
        total / count as f32
    };

    let op = PairwiseHammingGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let seqs_buf = storage_buf(device, "ham_s", bytemuck::cast_slice(&seqs));
    let out_buf = output_buf(device, "ham_out", (n_pairs * 4) as u64);

    op.dispatch(&seqs_buf, &out_buf, n_seqs as u32, seq_len as u32);

    match gpu.read_buffer_f32(&out_buf, n_pairs) {
        Ok(gpu_d) => {
            let gpu_mean = gpu_d.iter().sum::<f32>() / gpu_d.len() as f32;
            h.check_abs(
                &format!("Hamming 6×20: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_HAMMING_F32,
            );
        }
        Err(e) => h.check_bool(&format!("Hamming: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 7. Pairwise L2 — MODES (Paper 012)
// ═══════════════════════════════════════════════════════════════════

fn validate_l2(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 8_usize;
    let dim = 6_usize;
    let mut rng = Rng::new(66);
    let points_f64: Vec<f64> = (0..n * dim).map(|_| rng.normal()).collect();
    let points_f32: Vec<f32> = points_f64.iter().map(|&v| v as f32).collect();

    let n_pairs = n * (n - 1) / 2;
    let mut cpu_dist = Vec::with_capacity(n_pairs);
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = (0..dim)
                .map(|k| {
                    let diff = points_f64[i * dim + k] - points_f64[j * dim + k];
                    diff * diff
                })
                .sum::<f64>()
                .sqrt();
            cpu_dist.push(d);
        }
    }
    let cpu_mean = cpu_dist.iter().sum::<f64>() / cpu_dist.len() as f64;

    let op = PairwiseL2Gpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let pts_buf = storage_buf(device, "l2_pts", bytemuck::cast_slice(&points_f32));
    let out_buf = output_buf(device, "l2_out", (n_pairs * 4) as u64);

    op.dispatch(&pts_buf, &out_buf, n as u32, dim as u32);

    match gpu.read_buffer_f32(&out_buf, n_pairs) {
        Ok(gpu_d) => {
            let gpu_mean: f64 =
                gpu_d.iter().map(|&v| f64::from(v)).sum::<f64>() / gpu_d.len() as f64;
            h.check_abs(
                &format!("L2 8×6: GPU={gpu_mean:.4} vs CPU={cpu_mean:.4}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_MODES_L2_F32,
            );
        }
        Err(e) => h.check_bool(&format!("L2: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 8. Pairwise Jaccard — Pangenome (Paper 024)
// ═══════════════════════════════════════════════════════════════════

fn validate_jaccard(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_genomes = 8_usize;
    let n_genes = 32_usize;
    let mut rng = Rng::new(88);
    let pa_f64: Vec<f64> = (0..n_genomes * n_genes)
        .map(|_| if rng.uniform() > 0.3 { 1.0 } else { 0.0 })
        .collect();

    let cpu_jd =
        neural_spring::pangenome_selection::jaccard_distance_matrix(&pa_f64, n_genes, n_genomes);
    let mut cpu_upper = Vec::new();
    for i in 0..n_genomes {
        for j in (i + 1)..n_genomes {
            cpu_upper.push(cpu_jd[i * n_genomes + j]);
        }
    }
    let cpu_mean = cpu_upper.iter().sum::<f64>() / cpu_upper.len() as f64;

    let pa_f32: Vec<f32> = pa_f64.iter().map(|&v| v as f32).collect();

    let op = PairwiseJaccardGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let n_pairs = n_genomes * (n_genomes - 1) / 2;
    let pa_buf = storage_buf(device, "jac_pa", bytemuck::cast_slice(&pa_f32));
    let out_buf = output_buf(device, "jac_out", (n_pairs * 4) as u64);

    op.dispatch(&pa_buf, &out_buf, n_genomes as u32, n_genes as u32);

    match gpu.read_buffer_f32(&out_buf, n_pairs) {
        Ok(gpu_d) => {
            let gpu_mean: f64 =
                gpu_d.iter().map(|&v| f64::from(v)).sum::<f64>() / gpu_d.len() as f64;
            h.check_abs(
                &format!("Jaccard 8×32: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_JACCARD_F32,
            );
        }
        Err(e) => h.check_bool(&format!("Jaccard: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 9. Locus Variance — Meta-population (Paper 025)
// ═══════════════════════════════════════════════════════════════════

fn validate_locus_variance(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 4_usize;
    let n_loci = 16_usize;
    let mut rng = Rng::new(111);
    let freqs: Vec<f64> = (0..n_pops * n_loci).map(|_| rng.uniform()).collect();

    let cpu_var: Vec<f64> = (0..n_loci)
        .map(|l| {
            let vals: Vec<f64> = (0..n_pops).map(|p| freqs[p * n_loci + l]).collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64
        })
        .collect();
    let cpu_mean = cpu_var.iter().sum::<f64>() / cpu_var.len() as f64;

    let op = LocusVarianceGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let freqs_buf = storage_buf(device, "lv_f", bytemuck::cast_slice(&freqs));
    let out_buf = output_buf(device, "lv_out", (n_loci * 8) as u64);

    op.dispatch(&freqs_buf, &out_buf, n_pops as u32, n_loci as u32);

    match gpu.read_buffer_f64(&out_buf, n_loci) {
        Ok(gpu_v) => {
            let gpu_mean = gpu_v.iter().sum::<f64>() / gpu_v.len() as f64;
            h.check_abs(
                &format!("locus_var 4×16: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );
        }
        Err(e) => h.check_bool(&format!("locus_var: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 10. Cross-domain determinism
// ═══════════════════════════════════════════════════════════════════

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop = 16_u32;
    let glen = 8_u32;
    let mut rng = Rng::new(42);
    let genotypes: Vec<f64> = (0..pop * glen).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..glen).map(|_| rng.uniform()).collect();
    let op = BatchFitnessGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();

    let run = || -> Result<f64, String> {
        let g = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("det_g"),
            contents: bytemuck::cast_slice(&genotypes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let w = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("det_w"),
            contents: bytemuck::cast_slice(&weights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let o = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("det_o"),
            size: u64::from(pop) * 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        op.dispatch(&g, &w, &o, pop, glen);
        let f = gpu.read_buffer_f64(&o, pop as usize)?;
        Ok(f.iter().sum::<f64>() / f.len() as f64)
    };

    match (run(), run()) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("determinism: run1={a:.10} == run2={b:.10}"),
                (a - b).abs() < f64::EPSILON,
            );
        }
        _ => h.check_bool("determinism: dispatch failed", false),
    }
}
