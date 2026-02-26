// SPDX-License-Identifier: AGPL-3.0-or-later
//
// layer_norm_f64.wgsl — f64-emulated Layer Normalization
//
// LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
//
// Uses ToadStool df64 (double-float) emulation for f64 precision on consumer GPUs.
// Each workgroup processes one row (sequence position) of the input.
//
// Cross-spring: benefits baseCamp (Sub-02 attention), WDM surrogates,
// sovereign folding (Evoformer), all transformer architectures.
//
// Absorption target: barracuda::ops::layer_norm_f64
// Requires: df64_core.wgsl (auto-injected by compile_shader_df64)

struct Params {
    seq_len: u32,
    hidden_dim: u32,
    eps_hi: f32,
    eps_lo: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> shared_sum_hi: array<f32, 256>;
var<workgroup> shared_sum_lo: array<f32, 256>;

@compute @workgroup_size(256)
fn layer_norm(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let dim = params.hidden_dim;

    if row >= params.seq_len { return; }

    let base = row * dim;

    // Phase 1: Compute mean via df64 reduction
    var acc = df64_zero();
    var i = tid;
    while i < dim {
        acc = df64_add(acc, df64_from_f32(input[base + i]));
        i += 256u;
    }
    shared_sum_hi[tid] = acc.hi;
    shared_sum_lo[tid] = acc.lo;
    workgroupBarrier();

    var stride = 128u;
    while stride > 0u {
        if tid < stride {
            let r = df64_add(
                Df64(shared_sum_hi[tid], shared_sum_lo[tid]),
                Df64(shared_sum_hi[tid + stride], shared_sum_lo[tid + stride]),
            );
            shared_sum_hi[tid] = r.hi;
            shared_sum_lo[tid] = r.lo;
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    let mean = shared_sum_hi[0] / f32(dim);
    workgroupBarrier();

    // Phase 2: Compute variance via df64
    acc = df64_zero();
    i = tid;
    while i < dim {
        let diff = input[base + i] - mean;
        let sq = two_prod(diff, diff);
        acc = df64_add(acc, sq);
        i += 256u;
    }
    shared_sum_hi[tid] = acc.hi;
    shared_sum_lo[tid] = acc.lo;
    workgroupBarrier();

    stride = 128u;
    while stride > 0u {
        if tid < stride {
            let r = df64_add(
                Df64(shared_sum_hi[tid], shared_sum_lo[tid]),
                Df64(shared_sum_hi[tid + stride], shared_sum_lo[tid + stride]),
            );
            shared_sum_hi[tid] = r.hi;
            shared_sum_lo[tid] = r.lo;
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    let variance = shared_sum_hi[0] / f32(dim);
    let inv_std = 1.0 / sqrt(variance + params.eps_hi);
    workgroupBarrier();

    // Phase 3: Normalize + scale + shift
    i = tid;
    while i < dim {
        let normalized = (input[base + i] - mean) * inv_std;
        output[base + i] = gamma[i] * normalized + beta[i];
        i += 256u;
    }
}
