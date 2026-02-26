// SPDX-License-Identifier: AGPL-3.0-or-later
//
// attention_apply_f64.wgsl — Weighted sum of values with df64 accumulation
//
// Pass 3 of 3-pass f64 SDPA.
// output[b,h,q,d] = sum_j (weights[b,h,q,j] * V[b,h,j,d])
//
// df64 accumulation prevents drift in long kv_seq_len reductions.
//
// Absorption target: barracuda::ops::attention_apply_f64
// Requires: df64_core.wgsl (auto-injected by compile_shader_df64)

struct AttentionParams {
    batch_size: u32,
    num_heads:  u32,
    q_seq_len:  u32,
    kv_seq_len: u32,
    head_dim:   u32,
    _pad0:      u32,
    _pad1:      u32,
    _pad2:      u32,
}

@group(0) @binding(0) var<storage, read>       weights: array<f32>;
@group(0) @binding(1) var<storage, read>       value:   array<f32>;
@group(0) @binding(2) var<storage, read_write> output:  array<f32>;
@group(0) @binding(3) var<uniform>             params:  AttentionParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bh    = gid.z;
    let q_pos = gid.y;
    let d     = gid.x;

    if q_pos >= params.q_seq_len || d >= params.head_dim { return; }

    let b = bh / params.num_heads;
    let h = bh % params.num_heads;
    if b >= params.batch_size { return; }

    let H   = params.num_heads;
    let Sq  = params.q_seq_len;
    let Skv = params.kv_seq_len;
    let D   = params.head_dim;

    let w_base = b * H * Sq * Skv + h * Sq * Skv + q_pos * Skv;
    let v_base = b * H * Skv * D + h * Skv * D;

    var acc = df64_zero();
    for (var j = 0u; j < Skv; j++) {
        let prod = two_prod(weights[w_base + j], value[v_base + j * D + d]);
        acc = df64_add(acc, prod);
    }

    let out_idx = b * H * Sq * D + h * Sq * D + q_pos * D + d;
    output[out_idx] = acc.hi;
}
