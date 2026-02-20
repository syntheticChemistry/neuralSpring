# ToadStool Handoff — neuralSpring Local Evolutions

This document catalogues BarraCUDA / ToadStool shortcomings that
`neuralSpring` has evolved around locally, following the `hotSpring`
pattern.  Each section names the issue, the local fix, and a suggested
one-line upstream change for the ToadStool team to absorb.

**Last reviewed:** ToadStool commit `82f953c8` (Feb 19, 2026) — HEAD as of Feb 20.
**Canonical handoff:** `wateringHole/handoffs/NEURALSPRING_TOADSTOOL_HANDOFF_FEB20_2026.md`

---

## Resolution Status

ToadStool has been highly active (80+ commits since Feb 15) with hotSpring
absorption (NAK eigensolve, `StatefulPipeline`, `ReduceScalarPipeline`,
`CellListGpu`), deep debt sessions (wgpu v22, sleep elimination, zero-copy),
and sovereign compute (Phases 0–3). **None of the 11 neuralSpring shortcomings
have been addressed.** All ToadStool work focused on hotSpring feedback, deep
debt, and infrastructure — not neuralSpring evolution items.

| # | Shortcoming | Severity | Local Fix | Upstream Status |
|---|-------------|----------|-----------|-----------------|
| 1 | Per-op submission (S-01) | **Critical** | Fused pipeline | **Not absorbed** |
| 2 | Naive matmul (S-02) | **Critical** | 4-tier shader router | **Not absorbed** |
| 3 | MHA z-dispatch bug (S-03) | **High** | `evolved::mha` | **Not absorbed** |
| 4 | Softmax pooled buffers (S-04) | **Medium** | Re-upload logits | **Not absorbed** |
| 5 | `leaky_relu` Params mismatch (S-05) | **Low** | Cannot workaround | **Not absorbed** |
| 6 | `elu` Params mismatch (S-06) | **Low** | Cannot workaround | **Not absorbed** |
| 7 | `from_buffer` `pub(crate)` (S-07) | **High** | Raw buffer mgmt | **Not absorbed** |
| 8 | `layer_norm` round-trip (S-08) | **Medium** | `evolved::layer_norm` | **Not absorbed** |
| 9 | `log_softmax` round-trip (S-09) | **Medium** | `evolved::log_softmax` | **Not absorbed** |
| 10 | `science_limits()` CPU (S-10) | **Medium** | `create_relaxed()` | **Not absorbed** |
| 11 | `TensorSession` limited (S-11) | **High** | Fused pipeline | **Not absorbed** |

### What we absorbed from ToadStool

| ToadStool Evolution | neuralSpring Action |
|---|---|
| `WORKGROUP_SIZE_1D` / `WORKGROUP_SIZE_2D` constants | Imported into dispatch functions (replaces hardcoded 256/16) |
| `GpuDriverProfile` | Captured in `MatmulConfig` for future per-driver specialization |
| wgpu v22 API | Already matched since initial port |
| `probe::seed_cache_from_heuristics()` | Called automatically by `WgpuDevice::from_existing()` |
| WGSL shader stability | All 8 `include_str!` shaders verified unchanged — no drift |

### New ToadStool capabilities available for leverage

| Capability | API | neuralSpring Use Case |
|------------|-----|----------------------|
| `StatefulPipeline` | `staging::StatefulPipeline::run_iterations()` | EA loops, ODE integration, HMM chains |
| `ReduceScalarPipeline` | `pipeline::ReduceScalarPipeline::sum_f64()` | Fitness aggregation, log-likelihood |
| `KernelRouter` | `device::KernelRouter::route()` | 4-tier matmul selection |
| NAK eigensolve | `batched_eigh_nak_optimized_f64.wgsl` | Anderson localization eigensolver |

Once ToadStool absorbs the 11 fixes above, the local evolutions in
`neuralSpring/src/evolved/` (~2075 lines) can be retired entirely.

---

## 1. Per-Op Command Submission (S-01)

**Impact:** 46–78× penalty.

Each Tensor operation creates its own `CommandEncoder`, dispatches one
compute pass, and submits it individually.  A chained sequence like
`relu → layer_norm → matmul → softmax` submits 4 separate command
buffers.

BarraCUDA's `TensorContext` provides `begin_batch()` / `end_batch()` for
batched dispatch, but Tensor ops do not use it.

**Local workaround:** `src/evolved/fused_pipeline.rs` + `fused_mlp.rs` +
`fused_transformer.rs` — pre-compile shaders, pre-allocate buffers,
record all passes in one `CommandEncoder`, submit once.

**Result:** MLP 92 µs (43.6×), Transformer 174 µs (76.6×) vs per-op.

**Suggested upstream:** Extend `TensorSession` ops from
`{Add, Mul, Fma, Scale}` to include
`{MatMul, ReLU, GELU, LayerNorm, Softmax, Attention}`.

**ToadStool note:** `StatefulPipeline::KernelDispatch` is the right
abstraction for this — it already records multiple dispatches into one
encoder submit. Wire ML ops through the same pattern.

---

## 2. Naive Matmul — Zero Cache Reuse (S-02)

**Impact:** CPU 3× slower than Python. GPU misses memory-latency hiding.

`matmul.wgsl` reads K elements from global memory per output element.
NumPy calls OpenBLAS with hand-tuned cache-tiled GEMM.

**Local fix:** 4-tier `DeviceCapabilities`-driven shader router.
Shaders: `matmul_cpu_tiled.wgsl` (263 lines), `matmul_gpu_evolved.wgsl`
(302 lines). Both double-buffered, vec4 B-tile, k-loop unrolled.

**Result:** CPU 1.1× faster than Python at MLP large (3.1M FLOPs).
GPU 104× faster at TF medium (103M FLOPs).

**Suggested upstream:**
1. Copy shaders to `barracuda/src/shaders/math/`
2. Wire into `KernelRouter` using `DeviceCapabilities::device_type`

---

## 3. `Tensor::from_buffer` is `pub(crate)` (S-07)

**File:** `barracuda/src/tensor.rs` line 90

External crates cannot construct a `Tensor` from an existing
`wgpu::Buffer`.  Forces the round-trip pattern in S-08 and S-09.

**Suggested fix (one character):**
```rust
pub fn from_buffer(/* ... */) -> Self
```

This retires both S-08 and S-09 local evolutions.

---

## 4. `layer_norm_wgsl` GPU→CPU→GPU Round-Trip (S-08)

**File:** `barracuda/src/ops/layer_norm_wgsl.rs` lines 179–182

```rust
let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;
Ok(Tensor::new(output_data, shape.to_vec(), device.clone()))
```

**Local fix:** `src/evolved/layer_norm.rs` — dispatches same WGSL shader,
returns raw `wgpu::Buffer`. Stock: 1.7 ms GPU. Evolved: 329 µs GPU.

**Suggested fix:** Replace with `Tensor::from_buffer()` (requires S-07).

---

## 5. `log_softmax_wgsl` GPU→CPU→GPU Round-Trip (S-09)

Same pattern as S-08. Same fix. Local fix: `src/evolved/log_softmax.rs`.

---

## 6. MHA Projection Dispatch Bug (S-03)

**File:** `barracuda/src/ops/mha/projections.rs` lines 165–167

Shaders use `@workgroup_size(16, 16, 1)` — z-size is 1. But dispatch
divides by 16:

```rust
let workgroups_z = params.seq_len.div_ceil(16);  // BUG: should be seq_len
```

With `seq_len=8`, only 1 z-workgroup dispatched. Positions 1–7 zeroed.

**Fix:**
```rust
let workgroups_z = params.seq_len;   // project_with_head_split
let workgroups_z = params.d_model;   // concat_and_project
```

**Local fix:** `src/evolved/mha.rs` decomposes MHA into separate matmuls.

---

## 7. Softmax Incorrect on Pooled Buffers (S-04)

**File:** `barracuda/src/shaders/activation/softmax_simple.wgsl`

```wgsl
let N = arrayLength(&input);   // physical buffer, not logical tensor
```

Oversized pooled buffers corrupt softmax denominator.

**Fix:** Pass `logical_size` via uniform buffer.

**Local fix:** Re-upload logits before softmax to force exact-size buffer.

---

## 8. `leaky_relu_wgsl` Params Mismatch (S-05)

Rust `Params` is 4 bytes. WGSL expects 8 bytes. Causes wgpu panic.

**Fix:** Add `negative_slope: f32` to Rust `Params` struct.

---

## 9. `elu_wgsl` Params Mismatch (S-06)

Same as S-05. Add `alpha: f32` to Rust `Params`.

---

## 10. `WgpuDevice::new_cpu()` Requires `science_limits()` (S-10)

`science_limits()` requests 512 MB. llvmpipe caps at 128 MB. `new_cpu()`
always fails on standard CPU software rasterizers.

**Fix:** Add `new_cpu_relaxed()` or fallback to adapter limits.

**Local fix:** `src/gpu.rs` `create_relaxed()`.

---

## 11. `TensorSession` Limited to `{Add, Mul, Fma, Scale}` (S-11)

ML inference requires `{MatMul, ReLU, GELU, LayerNorm, Softmax, Attention}`.

**Fix:** Extend `SessionOp` enum and wire through session batching.

---

## Benchmark Data

### ML Inference — 3-Way Comparison

Full analysis: `specs/BENCHMARK_ANALYSIS.md`

**MLP (4→64→64→10):**

| Backend | Median | Throughput |
|---------|--------|------------|
| Python/NumPy | 23 µs | 42,965 inf/s |
| BarraCUDA CPU (llvmpipe) | 4.7 ms | 211 inf/s |
| BarraCUDA GPU (RTX 4070) | 4.0 ms | 247 inf/s |

**Transformer encoder block (d=32, h=4, seq=8):**

| Backend | Median | Throughput |
|---------|--------|------------|
| Python/NumPy | 77 µs | 13,044 blk/s |
| BarraCUDA CPU (llvmpipe) | 11.0 ms | 91 blk/s |
| BarraCUDA GPU (RTX 4070) | 13.8 ms | 73 blk/s |

**Root cause:** Per-op `queue.submit()`. GPU ≈ CPU because tensors are
too small to amortize launch latency.

### Fused Pipeline (Single-Encoder)

| Pipeline | MLP | Transformer | Speedup vs Per-Op |
|----------|-----|-------------|-------------------|
| Per-op | 4.5 ms | 12.8 ms | 1.0× |
| **Fused** | **97 µs** | **164 µs** | **46× / 78×** |

### 3-Way Scaling (Fused + Evolved Shaders)

| Scale | Py(1t) | CPU | GPU | CPU/Py | GPU/Py |
|-------|--------|-----|-----|--------|--------|
| MLP large (3.1M) | 3.0 ms | **2.7 ms** | **178 µs** | **1.1× faster** | 16.8× |
| TF medium (103M) | 59 ms | **15.1 ms** | **566 µs** | **3.9× faster** | 104× |
| TF xlarge (6.6B) | 232 ms | 1.42 s | **17.8 ms** | — | **13.1× faster** |
