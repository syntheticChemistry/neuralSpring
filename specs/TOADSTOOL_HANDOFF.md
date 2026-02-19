# ToadStool Handoff — neuralSpring Local Evolutions

This document catalogues BarraCUDA / ToadStool shortcomings that
`neuralSpring` has evolved around locally, following the `hotSpring`
pattern.  Each section names the issue, the local fix, and a suggested
one-line upstream change for the ToadStool team to absorb.

---

## 1. `layer_norm_wgsl` GPU→CPU→GPU Round-Trip

**File**: `barracuda/src/ops/layer_norm_wgsl.rs` lines 179-182

```rust
// After dispatching the WGSL shader:
let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;
Ok(Tensor::new(output_data, shape.to_vec(), device.clone()))
```

**Problem**: After the compute shader writes results to `output_buffer`, the
data is read back to CPU (`read_buffer`) and then uploaded again via
`Tensor::new()`.  Any pipeline that chains `layer_norm → matmul → softmax`
pays a full GPU→CPU→GPU round-trip per normalization.

**Local fix**: `neuralSpring/src/evolved/layer_norm.rs` — dispatches the
same WGSL shader but returns the raw `wgpu::Buffer` directly, skipping
the readback.  Result stays GPU-resident.

**Validation**: `validate_barracuda_tensor` passes identical numerical
checks for both stock `layer_norm_wgsl` and `evolved::layer_norm` on
GPU (RTX 4070) and CPU (llvmpipe).  46/46 checks pass on both backends.

**Suggested upstream change**:
Replace the `read_buffer` + `Tensor::new()` with `Tensor::from_buffer()`:
```rust
Ok(Tensor::from_buffer(output_buffer, shape.to_vec(), device.clone()))
```
This requires making `Tensor::from_buffer` `pub` (currently `pub(crate)`).

---

## 2. `log_softmax_wgsl` GPU→CPU→GPU Round-Trip

**File**: `barracuda/src/ops/log_softmax_wgsl.rs` lines 175-178

```rust
let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;
Ok(Tensor::new(output_data, shape.to_vec(), device.clone()))
```

**Problem**: Same pattern as `layer_norm_wgsl` — unnecessary round-trip.

**Local fix**: `neuralSpring/src/evolved/log_softmax.rs` — same approach,
keeps result in GPU buffer.

**Validation**: 46/46 checks pass.  Evolved log-softmax matches analytical
expected values to within `TENSOR_TRANSCENDENTAL_F32` (1e-3) tolerance.

**Suggested upstream change**: Same as layer_norm — use `from_buffer`.

---

## 3. `Tensor::from_buffer` is `pub(crate)`

**File**: `barracuda/src/tensor.rs`

**Problem**: External crates cannot construct a `Tensor` from an existing
`wgpu::Buffer`.  This forces the round-trip pattern seen in ops #1 and #2,
and prevents external crates from building GPU-resident pipelines.

**Suggested upstream change**:
```rust
// Change:
pub(crate) fn from_buffer(/* ... */) -> Self
// To:
pub fn from_buffer(/* ... */) -> Self
```

This single change would retire both local evolutions above.

---

## 4. Per-Op Command Submission (No Batching)

**Problem**: Each Tensor operation creates its own `CommandEncoder`,
dispatches one compute pass, and submits it individually.  A chained
sequence like `relu → layer_norm → matmul → softmax` submits 4 separate
command buffers.  On GPU this means 4 submission round-trips; on CPU
(llvmpipe) the overhead is per-submit fence synchronization.

BarraCUDA's `TensorContext` provides `begin_batch()` / `end_batch()` for
batched dispatch, but Tensor ops do not use it.

**Local workaround**: neuralSpring's evolved ops accept raw buffers and
can be composed into a single `CommandEncoder` with multiple compute
passes before a single `queue.submit()`.

**Suggested upstream change**: Wire Tensor ops through `TensorContext`
batching, or add a `Tensor::chain()` API that accumulates compute passes
into one submission.

---

## 5. `WgpuDevice` `science_limits()` Incompatible With CPU Software

**Problem**: `WgpuDevice::new_cpu()` requests `science_limits()` (512 MB
`max_storage_buffer_binding_size`) which llvmpipe cannot provide (caps at
128 MB).  This means `new_cpu()` always fails on standard CPU software
rasterizers.

**Local fix**: `neuralSpring/src/gpu.rs` `create_relaxed()` manually
creates a `wgpu::Device` with `Limits::downlevel_defaults()` and wraps
it via `WgpuDevice::from_existing()`.

**Suggested upstream change**: Add a `WgpuDevice::new_cpu_relaxed()` or
make `new_cpu()` fall back to the adapter's own limits when
`science_limits()` cannot be satisfied.

---

## Benchmark Data

### ML Inference — 3-Way Comparison

Full analysis: `specs/BENCHMARK_ANALYSIS.md`

**MLP (4→64→64→10)**:

| Backend | Median | Throughput |
|---------|--------|------------|
| Python/NumPy | 23 µs | 42,965 inf/s |
| BarraCUDA CPU (llvmpipe) | 4.7 ms | 211 inf/s |
| BarraCUDA GPU (RTX 4070) | 4.0 ms | 247 inf/s |

**Transformer encoder block (d=32, h=4, seq=8)**:

| Backend | Median | Throughput |
|---------|--------|------------|
| Python/NumPy | 77 µs | 13,044 blk/s |
| BarraCUDA CPU (llvmpipe) | 11.0 ms | 91 blk/s |
| BarraCUDA GPU (RTX 4070) | 13.8 ms | 73 blk/s |

**Root cause**: Per-op `queue.submit()` + bind group creation (~200 µs
per op) dominates compute (~5 µs total). GPU ≈ CPU because tensors are
too small to amortize launch latency. `TensorSession` batching would
reduce MLP to ~250 µs (single submit).

Max abs diff vs Python: MLP 1.5e-8, Transformer 1.1e-6.

### Tensor Ops

Collected with `bench_barracuda_tensor` (release build, 20 iterations,
3 warmup):

### GPU — NVIDIA RTX 4070 (Vulkan)

| Op | Median | Min | Max |
|----|--------|-----|-----|
| relu | 1.7ms | 1.2ms | 2.3ms |
| gelu_wgsl | 85µs | 75µs | 117µs |
| sigmoid | 68µs | 62µs | 93µs |
| softmax | 4.1ms | 3.8ms | 5.2ms |
| layer_norm_wgsl (stock) | 1.7ms | 1.5ms | 2.2ms |
| matmul | 3.4ms | 2.6ms | 5.1ms |
| add | 7µs | 6µs | 221µs |
| mse_loss | 191µs | 98µs | 567µs |
| **evolved::layer_norm (no RT)** | **329µs** | **253µs** | **1.1ms** |
| **evolved::log_softmax (no RT)** | **317µs** | **143µs** | **715µs** |

### CPU — llvmpipe (Vulkan)

| Op | Median | Min | Max |
|----|--------|-----|-----|
| relu | 887µs | 668µs | 1.0ms |
| gelu_wgsl | 723µs | 309µs | 981µs |
| sigmoid | 651µs | 541µs | 827µs |
| softmax | 144.8ms | 124.8ms | 172.7ms |
| layer_norm_wgsl (stock) | 902µs | 844µs | 1.0ms |
| matmul | 810µs | 684µs | 1.1ms |
| add | 56µs | 52µs | 84µs |
| mse_loss | 60.7ms | 502µs | 123.5ms |
| **evolved::layer_norm (no RT)** | **897µs** | **573µs** | **1.3ms** |
| **evolved::log_softmax (no RT)** | **995µs** | **553µs** | **1.0ms** |

---

## 6. `leaky_relu_wgsl` Params Mismatch (Rust 4B vs WGSL 8B)

**File**: `barracuda/src/ops/leaky_relu_wgsl.rs` lines 46-48

```rust
struct Params {
    size: u32,
}
```

**WGSL**: `barracuda/src/shaders/activation/leaky_relu.wgsl` lines 9-12

```wgsl
struct Params {
    size: u32,
    negative_slope: f32,
}
```

**Problem**: The Rust `Params` struct is 4 bytes (only `size`), but the
WGSL shader expects 8 bytes (`size` + `negative_slope`).  This causes a
wgpu validation panic: "Buffer is bound with size 4 where the shader
expects 8".  The `negative_slope` parameter is never set from Rust.

**Suggested upstream change**: Add `negative_slope: f32` to the Rust
`Params` struct and expose it as a parameter on `leaky_relu_wgsl()`.

---

## 7. `elu_wgsl` Params Mismatch (Same Pattern)

**File**: `barracuda/src/ops/elu_wgsl.rs` lines 46-48

**WGSL**: `barracuda/src/shaders/activation/elu.wgsl` lines 9-12

**Problem**: Same as `leaky_relu_wgsl` — Rust has `{ size: u32 }` but
WGSL expects `{ size: u32, alpha: f32 }`.  Causes wgpu panic.

**Suggested upstream change**: Add `alpha: f32` to Rust `Params`.

---

## 8. MHA Projection Dispatch Bug (z-dimension)

**File**: `barracuda/src/ops/mha/projections.rs`

**Shader**: `barracuda/src/shaders/attention/mha_projection.wgsl` and
`barracuda/src/shaders/tensor/mha_output.wgsl`

Both shaders use `@workgroup_size(16, 16, 1)` — the z-dimension has
size **1** per workgroup.  But the Rust dispatch code divides by 16:

```rust
// projections.rs:project_with_head_split()
let workgroups_z = params.seq_len.div_ceil(16);  // BUG: should be div_ceil(1)

// projections.rs:concat_and_project()
let workgroups_z = params.d_model.div_ceil(16);   // BUG: should be div_ceil(1)
```

**Problem**: With `seq_len=8` and `div_ceil(16)=1`, only **1** workgroup
is dispatched in z, covering `global_id.z=0` only.  Sequence positions
1–7 are never computed; their output stays at zero.  Similarly for the
output projection: `d_model=32` / 16 = 2 workgroups, covering only
`global_id.z` values 0–1 instead of 0–31.

**Impact**: `multi_head_attention()` produces mostly-zero outputs for any
non-trivial seq_len / d_model.  The MLP pipeline is unaffected because
it does not use MHA.

**Local fix**: `neuralSpring/src/evolved/mha.rs` — decomposes MHA into:
1. `matmul` for Q/K/V projections (verified correct)
2. CPU-side head-split / concat (unavoidable until barracuda adds `permute`)
3. `attention()` for SDPA (dispatch is correct: z = `batch * heads`)
4. `matmul` for output projection

**Validation**: `validate_barracuda_ml_inference` transformer checks now
pass 6/6, with max abs diff ~1.1e-6 vs Python.

**Suggested upstream change**: Fix the dispatch in `projections.rs`:
```rust
// project_with_head_split:
let workgroups_z = params.seq_len;   // workgroup_size.z = 1

// concat_and_project:
let workgroups_z = params.d_model;   // workgroup_size.z = 1
```

---

## 9. Softmax Incorrect on Pooled Buffers

**File**: `barracuda/src/shaders/activation/softmax_simple.wgsl`

```wgsl
let N = arrayLength(&input);   // physical buffer length, not logical tensor size
```

**Problem**: When the `add` operation returns a pooled output buffer
(via `TensorContext::acquire_pooled_output`), the buffer may be **larger**
than the tensor's logical size.  The softmax shader normalizes over
`arrayLength(&input)` which returns the physical buffer size.

For example: an MLP's final layer produces shape `[1, 10]` = 10 elements.
If the pool returns a 64-element buffer (reused from a previous hidden
layer), softmax normalizes over 64 elements.  The extra 54 elements are
uninitialized (typically zero), contributing `exp(0 - max_logit)` to the
denominator.  The resulting probabilities no longer sum to 1.

**Impact**: Any softmax on a tensor whose buffer came from the pool
(i.e., was produced by `add`, `matmul`, or other ops using `TensorContext`)
will produce incorrect results whenever the pool returns an oversized
buffer.  Fresh tensors from `Tensor::from_data()` are unaffected.

**Local fix**: In ML inference pipelines, re-upload logits before softmax:
```rust
let logit_data = logits.to_vec()?;
Tensor::from_data(&logit_data, logits.shape().to_vec(), device.clone())?
    .softmax()?
```
This forces an exact-size buffer.

**Suggested upstream change**: Pass the logical tensor size via a uniform
buffer to the softmax shader, or ensure `acquire_pooled_output(size)`
returns an **exact-size** buffer (not larger).

---

---

## 10. Fused Pipeline — Eliminating Dispatch Overhead

**Location**: `src/evolved/fused_pipeline.rs`, `fused_mlp.rs`, `fused_transformer.rs`

**Problem**: BarraCUDA's per-op dispatch creates a new `CommandEncoder`,
bind groups, and `queue.submit()` for each tensor operation. At ~200 µs
per submit, a 9-op MLP wastes 1.8 ms in dispatch alone while actual
compute takes ~5 µs. Python/NumPy does the same MLP in 23 µs.

**Local fix**: Pre-compile all shaders, pre-allocate all intermediate
buffers, and pre-create all bind groups **once**. Record all compute
passes into a **single** `CommandEncoder` and submit once. This collapses
N submissions into 1.

**New GPU shaders** (inline WGSL in `fused_pipeline.rs`):
- `HEAD_SPLIT_WGSL`: `[seq, d_model]` → `[n_heads, seq, d_head]` index remapping
- `HEAD_CONCAT_WGSL`: reverse of head_split
- `BATCHED_ATTENTION_WGSL`: fused Q·K^T/√d → softmax → ·V for all heads

These eliminate the CPU round-trips in the evolved MHA workaround.

**Results** (RTX 4070, Vulkan):

| Pipeline | MLP | Transformer | Speedup vs Per-Op |
|----------|-----|-------------|-------------------|
| Per-op | 4.0 ms | 13.3 ms | 1.0× |
| **Fused** | **92 µs** | **174 µs** | **43.6× / 76.6×** |

**Suggested upstream**: Integrate fused dispatch patterns into
`TensorSession`. Extend session ops to include `MatMul`, `ReLU`, `GELU`,
`LayerNorm`, `Softmax`, `Attention`. This would make the local fused
pipeline unnecessary.

---

## Resolution Status

| # | Shortcoming | Local Fix | Upstream Absorbed |
|---|-------------|-----------|-------------------|
| 1 | `layer_norm` round-trip | `evolved::layer_norm` | Pending |
| 2 | `log_softmax` round-trip | `evolved::log_softmax` | Pending |
| 3 | `from_buffer` `pub(crate)` | Raw buffer management | Pending |
| 4 | Per-op submission | Manual encoder batching | Pending |
| 5 | `science_limits()` CPU | `create_relaxed()` | Pending |
| 6 | `leaky_relu` Params mismatch | Skip (cannot workaround) | Pending |
| 7 | `elu` Params mismatch | Skip (cannot workaround) | Pending |
| 8 | MHA projection dispatch | `evolved::mha` | Pending |
| 9 | Softmax on pooled buffers | Re-upload before softmax | Pending |
| 10 | Per-op dispatch overhead | `evolved::fused_pipeline` | Pending |

Once ToadStool absorbs these fixes, the local evolutions in
`neuralSpring/src/evolved/` can be retired.
