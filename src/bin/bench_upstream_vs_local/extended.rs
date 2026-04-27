// SPDX-License-Identifier: AGPL-3.0-or-later

//! Extended upstream-vs-local benchmarks: multi-objective fitness,
//! pairwise L2, and swarm NN forward pass.

use barracuda::ops::bio::swarm_nn::SwarmNnParams;
use barracuda::ops::bio::{MultiObjFitnessGpu, PairwiseL2Gpu, SwarmNnGpu};
use bytemuck::{Pod, Zeroable};
use neural_spring::bench::BenchResult;
use neural_spring::bench::{
    self, BindingKind, DispatchParams, alloc_f32, bind_entry as be, buf_desc, create_pipeline,
};
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const SR: BindingKind = BindingKind::StorageRead;
const SW: BindingKind = BindingKind::StorageWrite;
const UNI: BindingKind = BindingKind::Uniform;

use super::{ITERATIONS, WARMUP};

// ─── Multi-Objective Fitness (Directed Evolution 014) ────────────────

const MULTI_OBJ_WGSL: &str = neural_spring_forge::shaders::MULTI_OBJ_FITNESS;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MultiObjParams {
    pop_size: u32,
    genome_len: u32,
    n_objectives: u32,
    _pad: u32,
}

pub fn bench_multi_obj(gpu: &Gpu) -> BenchResult {
    let pop_size = 5_000_u32;
    let genome_len = 40_u32;
    let n_objectives = 4_u32;
    let n_fitness = (pop_size * n_objectives) as usize;
    let mut rng = Rng::new(42);
    let genotypes: Vec<f32> = (0..pop_size * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_multi_obj"),
        source: wgpu::ShaderSource::Wgsl(MULTI_OBJ_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "multi_obj_fitness", &[SR, SW, UNI]);
    let gen_buf =
        device.create_buffer_init(&buf_desc("gen", &genotypes, wgpu::BufferUsages::STORAGE));
    let fit_buf = alloc_f32(device, n_fitness);
    let params_buf = device.create_buffer_init(&buf_desc(
        "p",
        &[MultiObjParams {
            pop_size,
            genome_len,
            n_objectives,
            _pad: 0,
        }],
        wgpu::BufferUsages::UNIFORM,
    ));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[be(0, &gen_buf), be(1, &fit_buf), be(2, &params_buf)],
    });
    let wg = (pop_size * n_objectives).div_ceil(256);
    let local_us = bench::time_dispatch(
        &DispatchParams {
            device,
            queue,
            gpu,
            pipeline: &pipeline,
            bg: &bg,
            workgroups: wg,
            readback_buf: &fit_buf,
            readback_count: n_fitness,
        },
        WARMUP,
        ITERATIONS,
    );

    let op = MultiObjFitnessGpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = bench::time_upstream(WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, n_fitness);
        op.dispatch(&gen_buf, &out, pop_size, genome_len, n_objectives);
        gpu.read_buffer_f32(&out, n_fitness).ok();
    });

    BenchResult {
        name: format!("MultiObjFitness {pop_size}×{n_objectives}"),
        origin: "neuralSpring 014 (DirEvo)",
        local_us,
        upstream_us,
    }
}

// ─── Pairwise L2 (MODES 012 — novelty metric) ──────────────────────

const L2_WGSL: &str = neural_spring_forge::shaders::PAIRWISE_L2;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct L2Params {
    n: u32,
    dim: u32,
}

pub fn bench_pairwise_l2(gpu: &Gpu) -> BenchResult {
    let n = 200_u32;
    let dim = 50_u32;
    let n_pairs = (n * (n - 1) / 2) as usize;
    let mut rng = Rng::new(42);
    let features: Vec<f32> = (0..n * dim).map(|_| rng.uniform() as f32).collect();

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_l2"),
        source: wgpu::ShaderSource::Wgsl(L2_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "pairwise_l2", &[SR, SW, UNI]);
    let feat_buf =
        device.create_buffer_init(&buf_desc("feat", &features, wgpu::BufferUsages::STORAGE));
    let dist_buf = alloc_f32(device, n_pairs);
    let params_buf = device.create_buffer_init(&buf_desc(
        "p",
        &[L2Params { n, dim }],
        wgpu::BufferUsages::UNIFORM,
    ));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[be(0, &feat_buf), be(1, &dist_buf), be(2, &params_buf)],
    });
    let wg = (n_pairs as u32).div_ceil(256);
    let local_us = bench::time_dispatch(
        &DispatchParams {
            device,
            queue,
            gpu,
            pipeline: &pipeline,
            bg: &bg,
            workgroups: wg,
            readback_buf: &dist_buf,
            readback_count: n_pairs,
        },
        WARMUP,
        ITERATIONS,
    );

    let op = PairwiseL2Gpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = bench::time_upstream(WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, n_pairs);
        let _ = op.dispatch(&feat_buf, &out, n, dim);
        gpu.read_buffer_f32(&out, n_pairs).ok();
    });

    BenchResult {
        name: format!("PairwiseL2 {n}×{dim}"),
        origin: "neuralSpring 012 (MODES)",
        local_us,
        upstream_us,
    }
}

// ─── Swarm NN Forward (Swarm Robotics 015) ──────────────────────────

const SWARM_NN_WGSL: &str = neural_spring_forge::shaders::SWARM_NN_FORWARD;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SwarmConfig {
    n_controllers: u32,
    n_evals: u32,
}

pub fn bench_swarm_nn(gpu: &Gpu) -> BenchResult {
    let n_controllers = 500_u32;
    let n_evals = 20_u32;
    let n_total = (n_controllers * n_evals) as usize;
    let weights_per_ctrl = 33_u32;
    let mut rng = Rng::new(42);
    let weights: Vec<f32> = (0..n_controllers * weights_per_ctrl)
        .map(|_| (rng.uniform() as f32).mul_add(2.0, -1.0))
        .collect();
    let inputs: Vec<f32> = (0..n_controllers * n_evals)
        .map(|_| rng.uniform() as f32)
        .collect();

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local_swarm_nn"),
        source: wgpu::ShaderSource::Wgsl(SWARM_NN_WGSL.into()),
    });
    let (pipeline, bgl) = create_pipeline(device, &shader, "swarm_nn_forward", &[SR, SR, SW, UNI]);
    let wt_buf =
        device.create_buffer_init(&buf_desc("weights", &weights, wgpu::BufferUsages::STORAGE));
    let inp_buf =
        device.create_buffer_init(&buf_desc("inputs", &inputs, wgpu::BufferUsages::STORAGE));
    let act_buf = alloc_f32(device, n_total);
    let cfg_buf = device.create_buffer_init(&buf_desc(
        "cfg",
        &[SwarmConfig {
            n_controllers,
            n_evals,
        }],
        wgpu::BufferUsages::UNIFORM,
    ));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            be(0, &wt_buf),
            be(1, &inp_buf),
            be(2, &act_buf),
            be(3, &cfg_buf),
        ],
    });
    let wg = (n_controllers * n_evals).div_ceil(256);
    let local_us = bench::time_dispatch(
        &DispatchParams {
            device,
            queue,
            gpu,
            pipeline: &pipeline,
            bg: &bg,
            workgroups: wg,
            readback_buf: &act_buf,
            readback_count: n_total,
        },
        WARMUP,
        ITERATIONS,
    );

    let op = SwarmNnGpu::new(Arc::clone(gpu.wgpu_device()));
    let upstream_us = bench::time_upstream(WARMUP, ITERATIONS, || {
        let out = alloc_f32(device, n_total);
        op.dispatch(
            &wt_buf,
            &inp_buf,
            &out,
            &SwarmNnParams {
                n_controllers,
                n_evals,
                input_dim: 1,
                hidden_dim: 4,
                output_dim: 5,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        );
        gpu.read_buffer_f32(&out, n_total).ok();
    });

    BenchResult {
        name: format!("SwarmNN {n_controllers}×{n_evals}"),
        origin: "neuralSpring 015 (Swarm)",
        local_us,
        upstream_us,
    }
}
