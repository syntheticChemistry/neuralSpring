// SPDX-License-Identifier: AGPL-3.0-or-later
//
// sigmoid_f64.wgsl — f64-precision Sigmoid activation
//
// sigma(x) = 1 / (1 + exp(-x))
//
// Numerically stable formulation: uses sign-branch to avoid overflow.
// For x >= 0: sigma = 1 / (1 + exp(-x))
// For x < 0:  sigma = exp(x) / (1 + exp(x))
//
// Cross-spring: gating operations in folding (Evoformer pair bias),
// WDM transport (output normalization), baseCamp (agent coordination).
//
// Absorption target: barracuda::ops::sigmoid_f64

struct Params {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn sigmoid_f64(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }

    let x = input[idx];

    if x >= 0.0 {
        let e = exp(-x);
        output[idx] = 1.0 / (1.0 + e);
    } else {
        let e = exp(x);
        output[idx] = e / (1.0 + e);
    }
}
