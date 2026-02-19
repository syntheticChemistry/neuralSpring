// SPDX-License-Identifier: AGPL-3.0-only

//! Fused GPU pipeline infrastructure — single-encoder dispatch.
//!
//! Barracuda's per-op dispatch creates a new `CommandEncoder` and calls
//! `queue.submit()` for every tensor operation.  At ~200 us per submit
//! (bind group creation + driver overhead), a 9-op MLP wastes 1.8 ms
//! in dispatch alone while the actual f32 compute takes ~5 us.
//!
//! This module provides helpers to pre-compile shaders, pre-allocate
//! buffers, and record multiple compute passes into a **single**
//! `CommandEncoder` with one `queue.submit()`.
//!
//! **`ToadStool` handoff**: Once `TensorSession` supports `MatMul`,
//! `ReLU`, `LayerNorm`, `Softmax`, etc., this module can be retired.

use barracuda::device::capabilities::DeviceCapabilities;
use barracuda::device::WgpuDevice;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub type Dev = Arc<WgpuDevice>;

// ═══════════════════════════════════════════════════════════════════
// WGSL shader sources (inlined from barracuda for fused dispatch)
// ═══════════════════════════════════════════════════════════════════

pub const MATMUL_WGSL: &str = include_str!(
    "../../../phase1/toadstool/crates/barracuda/src/shaders/math/matmul.wgsl"
);
pub const MATMUL_TILED_WGSL: &str = include_str!(
    "../../../phase1/toadstool/crates/barracuda/src/shaders/math/matmul_tiled.wgsl"
);
pub const MATMUL_CPU_TILED_WGSL: &str = include_str!("matmul_cpu_tiled.wgsl");
pub const MATMUL_GPU_EVOLVED_WGSL: &str = include_str!("matmul_gpu_evolved.wgsl");
pub const ADD_WGSL: &str = include_str!(
    "../../../phase1/toadstool/crates/barracuda/src/shaders/math/elementwise_add.wgsl"
);
pub const RELU_WGSL: &str = include_str!(
    "../../../phase1/toadstool/crates/barracuda/src/shaders/activation/relu.wgsl"
);
pub const GELU_WGSL: &str = include_str!(
    "../../../phase1/toadstool/crates/barracuda/src/shaders/activation/gelu.wgsl"
);
pub const SOFTMAX_WGSL: &str = include_str!(
    "../../../phase1/toadstool/crates/barracuda/src/shaders/activation/softmax_simple.wgsl"
);
pub const LAYER_NORM_WGSL: &str = include_str!(
    "../../../phase1/toadstool/crates/barracuda/src/shaders/norm/layer_norm.wgsl"
);

// ═══════════════════════════════════════════════════════════════════
// Params structs (matching WGSL uniform layouts)
// ═══════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
#[allow(clippy::pub_underscore_fields)]
pub struct MatMulParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LayerNormParams {
    pub size: u32,
    pub feature_size: u32,
    pub epsilon: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GeluParams {
    pub size: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct HeadShapeParams {
    pub seq_len: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub d_head: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct AttentionParams {
    pub batch_size: u32,
    pub num_heads: u32,
    pub seq_len: u32,
    pub head_dim: u32,
}

// ═══════════════════════════════════════════════════════════════════
// ShaderCache — pre-compiled pipelines
// ═══════════════════════════════════════════════════════════════════

/// Pre-compiled compute pipelines for all fused ops.
pub struct ShaderCache {
    pub matmul: wgpu::ComputePipeline,
    pub matmul_tiled: wgpu::ComputePipeline,
    pub matmul_cpu_tiled: wgpu::ComputePipeline,
    pub matmul_gpu_evolved: wgpu::ComputePipeline,
    pub add: wgpu::ComputePipeline,
    pub relu: wgpu::ComputePipeline,
    pub gelu: wgpu::ComputePipeline,
    pub softmax: wgpu::ComputePipeline,
    pub layer_norm: wgpu::ComputePipeline,
    pub head_split: wgpu::ComputePipeline,
    pub head_concat: wgpu::ComputePipeline,
    pub attention: wgpu::ComputePipeline,
}

impl ShaderCache {
    /// Compile all shaders once.
    #[must_use]
    pub fn new(device: &Dev) -> Self {
        let d = device.device();
        let compile = |src: &str, label: &str| -> wgpu::ComputePipeline {
            let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: "main",
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
        };

        Self {
            matmul: compile(MATMUL_WGSL, "fused_matmul"),
            matmul_tiled: compile(MATMUL_TILED_WGSL, "fused_matmul_tiled"),
            matmul_cpu_tiled: compile(MATMUL_CPU_TILED_WGSL, "fused_matmul_cpu_tiled"),
            matmul_gpu_evolved: compile(MATMUL_GPU_EVOLVED_WGSL, "fused_matmul_gpu_evolved"),
            add: compile(ADD_WGSL, "fused_add"),
            relu: compile(RELU_WGSL, "fused_relu"),
            gelu: compile(GELU_WGSL, "fused_gelu"),
            softmax: compile(SOFTMAX_WGSL, "fused_softmax"),
            layer_norm: compile(LAYER_NORM_WGSL, "fused_layer_norm"),
            head_split: compile(HEAD_SPLIT_WGSL, "fused_head_split"),
            head_concat: compile(HEAD_CONCAT_WGSL, "fused_head_concat"),
            attention: compile(BATCHED_ATTENTION_WGSL, "fused_attention"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Buffer helpers
// ═══════════════════════════════════════════════════════════════════

/// Create a storage buffer with initial f32 data.
#[must_use]
pub fn buf_init(device: &Dev, data: &[f32], label: &str) -> wgpu::Buffer {
    device.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

/// Create an empty storage buffer for `count` f32 values.
#[must_use]
pub fn buf_empty(device: &Dev, count: usize, label: &str) -> wgpu::Buffer {
    #[allow(clippy::cast_possible_truncation)]
    let size = (count * 4) as u64;
    device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Create a uniform buffer from a `Pod` struct.
#[must_use]
pub fn buf_uniform<T: Pod>(device: &Dev, data: &T, label: &str) -> wgpu::Buffer {
    device.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Read `count` f32 values from a GPU buffer.
#[must_use]
#[allow(clippy::expect_used, clippy::missing_panics_doc)]
pub fn readback(device: &Dev, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
    #[allow(clippy::cast_possible_truncation)]
    let byte_size = (count * 4) as u64;
    let staging = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback_staging"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_size);
    device.queue().submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    device.device().poll(wgpu::Maintain::Wait);
    rx.recv().expect("map_async channel").expect("map_async failed");

    let mapped = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    result
}

// ═══════════════════════════════════════════════════════════════════
// Bind-group helpers (auto-derive layout from pipeline)
// ═══════════════════════════════════════════════════════════════════

/// Create a bind group matching a pipeline's auto-layout at group 0.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn bind_group(
    device: &Dev,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&wgpu::Buffer],
    label: &str,
) -> wgpu::BindGroup {
    let layout = pipeline.get_bind_group_layout(0);
    let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
        .iter()
        .enumerate()
        .map(|(i, buf)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buf.as_entire_binding(),
        })
        .collect();
    device.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &layout,
        entries: &entries,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Pass-recording helpers
// ═══════════════════════════════════════════════════════════════════

/// Record a compute pass with the given pipeline and bind group.
pub fn record_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    workgroups: (u32, u32, u32),
    label: &str,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bg, &[]);
    pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
}

/// Dispatch dimensions for naive matmul: `row=global_id.x, col=global_id.y`.
#[must_use]
pub const fn matmul_dispatch(m: u32, n: u32) -> (u32, u32, u32) {
    (m.div_ceil(16), n.div_ceil(16), 1)
}

/// Dispatch dimensions for tiled matmul (16x16): `row=global_id.y, col=global_id.x`.
#[must_use]
pub const fn matmul_tiled_dispatch(m: u32, n: u32) -> (u32, u32, u32) {
    (n.div_ceil(16), m.div_ceil(16), 1)
}

/// Dispatch dimensions for CPU-optimized matmul (32x32 tiles, 8x4 workgroup).
#[must_use]
pub const fn matmul_cpu_tiled_dispatch(m: u32, n: u32) -> (u32, u32, u32) {
    (n.div_ceil(32), m.div_ceil(32), 1)
}

/// Dispatch dimensions for GPU-evolved matmul (32x32 output tiles, 16x16 workgroup,
/// each thread computes 2x2).
#[must_use]
pub const fn matmul_gpu_evolved_dispatch(m: u32, n: u32) -> (u32, u32, u32) {
    (n.div_ceil(32), m.div_ceil(32), 1)
}

/// Detect whether the device is running on a CPU software adapter.
#[must_use]
pub fn is_cpu_backend(device: &Dev) -> bool {
    device.adapter_info().device_type == wgpu::DeviceType::Cpu
}

/// Dispatch function type for matmul: `(M, N) -> (x, y, z)` workgroups.
pub type MatMulDispatchFn = fn(u32, u32) -> (u32, u32, u32);

/// Device-aware matmul configuration derived from `BarraCUDA` `DeviceCapabilities`.
///
/// Instead of hardcoding thresholds, query the device for vendor-specific
/// optimal tile sizes and cache the result for the lifetime of the pipeline.
/// This enables `ToadStool` to tune per-vendor without `neuralSpring` changes.
#[derive(Debug, Clone)]
pub struct MatmulConfig {
    pub is_cpu: bool,
    pub vendor: u32,
    pub vendor_name: &'static str,
    /// Recommended tile size from `BarraCUDA` for this device class.
    pub recommended_tile_size: u32,
}

impl MatmulConfig {
    /// Build from `BarraCUDA` `DeviceCapabilities` — runtime discovery, no hardcoding.
    #[must_use]
    pub fn from_device(device: &Dev) -> Self {
        let caps = DeviceCapabilities::from_device(device);
        Self {
            is_cpu: caps.device_type == wgpu::DeviceType::Cpu,
            vendor: caps.vendor,
            vendor_name: caps.vendor_name(),
            recommended_tile_size: caps.optimal_matmul_tile_size(),
        }
    }
}

/// Select the best matmul pipeline and dispatch function for given dimensions.
///
/// Uses `MatmulConfig` (from `BarraCUDA` `DeviceCapabilities`) for device-aware
/// routing — four tiers:
///
/// - `M,N < threshold`: naive matmul (no shared memory, safe for tiny matrices)
/// - CPU: BLAS-evolved cpu-tiled (32x32, vec4, 8x4 micro-kernel, 4x k-unroll)
/// - GPU, small: 16x16 shared-memory tiles (better occupancy for small workloads)
/// - GPU, large: double-buffered gpu-evolved (32x32 output tiles, 2x2 micro-kernel,
///   vec4 B-tile, 4x k-unroll, load/compute overlap)
///
/// The GPU tier boundary (256) balances occupancy vs memory-latency hiding.
/// Below it, more workgroups keep the SM scheduler fed. Above it, double-buffering
/// and the larger register-blocked micro-kernel dominate.
#[must_use]
pub fn select_matmul<'a>(
    cache: &'a ShaderCache,
    m: u32,
    _k: u32,
    n: u32,
    config: &MatmulConfig,
) -> (&'a wgpu::ComputePipeline, MatMulDispatchFn) {
    let min_tiled = config.recommended_tile_size.max(16);
    if m < min_tiled || n < min_tiled {
        (&cache.matmul, matmul_dispatch as MatMulDispatchFn)
    } else if config.is_cpu {
        (&cache.matmul_cpu_tiled, matmul_cpu_tiled_dispatch as MatMulDispatchFn)
    } else if m >= 256 || n >= 256 {
        (&cache.matmul_gpu_evolved, matmul_gpu_evolved_dispatch as MatMulDispatchFn)
    } else {
        (&cache.matmul_tiled, matmul_tiled_dispatch as MatMulDispatchFn)
    }
}

/// Dispatch dimensions for elementwise ops on `count` elements.
#[must_use]
pub const fn elementwise_dispatch(count: u32) -> (u32, u32, u32) {
    (count.div_ceil(256), 1, 1)
}

// ═══════════════════════════════════════════════════════════════════
// Inline WGSL for head-split, head-concat, and attention passes
// ═══════════════════════════════════════════════════════════════════

pub const HEAD_SPLIT_WGSL: &str = r"
struct Params {
    seq_len: u32,
    d_model: u32,
    n_heads: u32,
    d_head: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.n_heads * params.seq_len * params.d_head;
    if (idx >= total) { return; }

    let h = idx / (params.seq_len * params.d_head);
    let s = (idx / params.d_head) % params.seq_len;
    let d = idx % params.d_head;

    let src = s * params.d_model + h * params.d_head + d;
    output[idx] = input[src];
}
";

pub const HEAD_CONCAT_WGSL: &str = r"
struct Params {
    seq_len: u32,
    d_model: u32,
    n_heads: u32,
    d_head: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.d_model;
    if (idx >= total) { return; }

    let s = idx / params.d_model;
    let flat_d = idx % params.d_model;
    let h = flat_d / params.d_head;
    let d = flat_d % params.d_head;

    let src = h * params.seq_len * params.d_head + s * params.d_head + d;
    output[idx] = input[src];
}
";

/// Batched fused attention: Q @ K^T / sqrt(d), softmax, @ V.
///
/// Layout: Q, K, V are `[n_heads, seq_len, d_head]`.
/// Output is `[n_heads, seq_len, d_head]`.
///
/// Each thread computes one output element `(head, seq_pos, dim)`.
/// Scores are recomputed (flash-style) to avoid O(seq^2) temp memory.
pub const BATCHED_ATTENTION_WGSL: &str = r"
struct Params {
    n_heads: u32,
    seq_len: u32,
    d_head: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d_idx = gid.x;
    let seq_idx = gid.y;
    let head = gid.z;

    if (seq_idx >= params.seq_len || d_idx >= params.d_head || head >= params.n_heads) {
        return;
    }

    let head_offset = head * params.seq_len * params.d_head;

    // Pass 1: compute max score for numerical stability
    var max_score: f32 = -1e10;
    for (var k = 0u; k < params.seq_len; k++) {
        var score: f32 = 0.0;
        for (var d = 0u; d < params.d_head; d++) {
            score += query[head_offset + seq_idx * params.d_head + d]
                   * key[head_offset + k * params.d_head + d];
        }
        max_score = max(max_score, score * params.scale);
    }

    // Pass 2: compute exp sum
    var sum_exp: f32 = 0.0;
    for (var k = 0u; k < params.seq_len; k++) {
        var score: f32 = 0.0;
        for (var d = 0u; d < params.d_head; d++) {
            score += query[head_offset + seq_idx * params.d_head + d]
                   * key[head_offset + k * params.d_head + d];
        }
        sum_exp += exp(score * params.scale - max_score);
    }

    // Pass 3: weighted sum of values
    var weighted_sum: f32 = 0.0;
    for (var k = 0u; k < params.seq_len; k++) {
        var score: f32 = 0.0;
        for (var d = 0u; d < params.d_head; d++) {
            score += query[head_offset + seq_idx * params.d_head + d]
                   * key[head_offset + k * params.d_head + d];
        }
        let w = exp(score * params.scale - max_score) / sum_exp;
        weighted_sum += w * value[head_offset + k * params.d_head + d_idx];
    }

    output[head_offset + seq_idx * params.d_head + d_idx] = weighted_sum;
}
";

/// Create a fresh `CommandEncoder`.
#[must_use]
pub fn new_encoder(device: &Dev, label: &str) -> wgpu::CommandEncoder {
    device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(label),
    })
}

/// Submit an encoder and wait for completion (blocking).
pub fn submit_and_wait(device: &Dev, encoder: wgpu::CommandEncoder) {
    device.queue().submit(Some(encoder.finish()));
    device.device().poll(wgpu::Maintain::Wait);
}

/// Dispatch for batched attention: `(d_head/16, seq/16, n_heads)`.
#[must_use]
pub const fn attention_dispatch(d_head: u32, seq_len: u32, n_heads: u32) -> (u32, u32, u32) {
    (d_head.div_ceil(16), seq_len.div_ceil(16), n_heads)
}
