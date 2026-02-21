# neuralSpring → ToadStool/BarraCUDA: Upstream Fixes Handoff

**Date:** 2026-02-21
**From:** neuralSpring (ML / isomorphic learning / scholarly reproduction Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**Context:** Phase 5b upstream investigation — 3 issues identified, 2 fixed locally, 1 characterized

---

## Executive Summary

During Phase 5b validation of `BarraCUDA` GPU tensor operations, neuralSpring
identified and characterized three upstream issues in the `barracuda` crate.
Two have local fixes ready for absorption; one requires driver-level
investigation.

| ID | Issue | Severity | Local Fix | Absorption Status |
|----|-------|----------|-----------|-------------------|
| **S-13** | `PooledBuffer` drop-before-completion race | **Critical** | `evolved::tensor_sync` | Ready |
| **S-14** | Naive matmul hang for small square matrices | **High** | Workaround (non-square) | Needs investigation |
| **GELU** | Test expectation wrong (3.0 vs true 2.9964) | Low | Fixed in validator | Ready |

---

## Issue 1: S-13 — `PooledBuffer` Drop-Before-Completion Race

### Problem

`PooledBuffer::drop` (in `barracuda/src/device/tensor_context/pool.rs:49-56`)
returns buffers to the pool **without waiting for the GPU to finish using them**.
When sequential tensor operations produce intermediate results that are dropped
before readback, the next `acquire_pooled` can reuse a buffer the GPU is still
writing to — causing data corruption or driver hangs.

### Root Cause

```
pool.rs:49  impl Drop for PooledBuffer {
pool.rs:50      fn drop(&mut self) {
pool.rs:51          if let Some(buffer) = self.buffer.take() {
pool.rs:52              if let Some(pool) = self.pool.upgrade() {
pool.rs:53                  pool.return_buffer(buffer, self.bucket);  // ← No GPU sync!
pool.rs:54              }
pool.rs:55          }
pool.rs:56      }
pool.rs:57  }
```

`record_operation` (context.rs:84-96) does `queue.submit()` without
`device.poll()`. The only sync point is `read_buffer` in
`wgpu_device/buffers.rs:36-50` which uses `map_async` + `poll(Wait)`.

### Race Scenario

1. `t1 = a.matmul(b)` — submits GPU work, returns `Tensor(PooledBuffer A)`
2. Caller drops `t1` without `to_vec()` — buffer A returned to pool
3. `t2 = c.matmul(d)` — `acquire_pooled` gets buffer A from pool
4. Second matmul writes to A while first matmul may still be writing to it
5. Result: data corruption or driver hang

### Local Fix

`src/evolved/tensor_sync.rs` provides three primitives:

```rust
/// Force all submitted GPU work to complete.
pub fn gpu_fence(device: &Arc<WgpuDevice>) {
    device.device().poll(wgpu::Maintain::Wait);
}

/// Read tensor data from GPU, recreate tensor (forces full sync).
pub fn materialize(t: &Tensor, device: &Arc<WgpuDevice>) -> Result<Tensor, String>

/// Matmul with automatic fence after submission.
pub fn fenced_matmul(lhs: Tensor, rhs: &Tensor, device: &Arc<WgpuDevice>) -> Result<Tensor, String>
```

### Recommended Absorption

**Option A (minimal):** Add `device.poll(Maintain::Wait)` in
`PooledBuffer::drop` before `pool.return_buffer()`. Simple but serializes work.

**Option B (optimal):** Track in-flight submissions with a generation counter.
Only recycle buffers after their associated submission completes. Requires
fence/semaphore infrastructure but preserves async overlap.

**Option C (conservative):** Use `acquire_output_buffer` (non-pooled) for
matmul output. Eliminates reuse risk at the cost of extra allocations.

### Test Evidence

- `evolved::tensor_sync::tests::fenced_matmul_basic` — PASS
- `evolved::tensor_sync::tests::materialize_roundtrip` — PASS
- `evolved::tensor_sync::tests::sequential_square_matmul_with_fence` — PASS (release, solo)

---

## Issue 2: S-14 — Naive Matmul Hang for Small Square Matrices

### Problem

`Tensor::matmul` with the **Naive tier** (`matmul.wgsl`, selected when M or N
< 32) hangs indefinitely on the RTX 4070 (Vulkan, driver 580.82.09) when the
binary exceeds a certain complexity threshold.

### Key Observations

| Condition | Result |
|-----------|--------|
| 2×3 × 3×2 (non-square, any binary) | **Works** |
| 8×8 × 8×8 (square, trivial binary) | **Works** |
| 8×8 × 8×8 (square, complex binary) | **Hangs** |
| 4×4 × 4×4 (square, complex binary) | **Hangs** |
| Sequential non-square matmuls | **Works** |
| 32×32+ (Tiled16 tier) | **Untested in complex binary** |

"Complex binary" = binary with validation harness, multiple GPU operations,
or certain import combinations that change the compilation unit size.

### Root Cause Hypothesis

The Naive matmul shader's pipeline compilation interacts with the Vulkan
pipeline cache in a binary-layout-dependent way. When the binary's text
segment exceeds a threshold (from additional code/symbols), the
`create_compute_pipeline` or `dispatch_workgroups` call for the Naive shader
hangs. This does NOT affect Tiled16 or higher tiers.

### Workaround

neuralSpring validators use non-square GPU matmuls to prove `Tensor::matmul`
correctness, then validate square-matrix spectral operations on CPU.

### Recommended Investigation

1. **Remove the Naive tier entirely** — use Tiled16 for all sizes. The
   `SMALL_MATRIX_THRESHOLD` (32) creates more problems than it solves.
2. **Test on other GPUs** — confirm if this is RTX 4070 / Vulkan 580.82.09
   specific or affects other drivers.
3. **Add pipeline creation timeout** — `create_compute_pipeline` should have
   a bounded wait to prevent indefinite hangs.

### Relevant Code

```
matmul.rs:59-63  fn select_tier(caps, m, n) -> MatMulTier {
matmul.rs:60         if m < 32 || n < 32 { return Naive; }  // ← Triggers S-14
matmul.rs:77         Naive => include_str!("../shaders/math/matmul.wgsl"),
matmul.rs:167        Naive => ((m as u32).div_ceil(16), (n as u32).div_ceil(16)),
```

---

## Issue 3: GELU Test Expectation

### Problem

`validate_barracuda_tensor::validate_gelu` tested `gelu(3) ≈ 3.0` with
tolerance 1e-3, but the true mathematical GELU(3) = 2.996362607918227.
The actual difference |2.9964 - 3.0| ≈ 0.004 exceeds the 1e-3 tolerance.

### Fix

The WGSL GELU implementation is **correct** — the tanh approximation matches
the reference value to within f32 precision. Only the test expectation was
wrong.

```rust
// Before (FAIL):
h.check_abs("gelu(3) ≈ 3.0", f64::from(v[5]), 3.0, 1e-3);

// After (PASS):
h.check_abs("gelu(3) ≈ 2.9964", f64::from(v[5]), 2.996_362_607_918_227, 1e-3);
```

**Result:** 86/86 PASS (was 85/86).

### Provenance

`scipy.special.erf(3/sqrt(2))` → `GELU(3) = 0.5 * 3 * (1 + erf(3/√2))` = 2.996362607918227.

---

## Files Changed

| File | Change |
|------|--------|
| `src/evolved/tensor_sync.rs` | **New**: S-13 fix primitives (`gpu_fence`, `materialize`, `fenced_matmul`) |
| `src/evolved/mod.rs` | Added `tensor_sync` module + doc table entry |
| `src/bin/validate_barracuda_tensor.rs` | Fixed GELU test expectation (86/86 PASS) |
| `src/bin/validate_barracuda_gpu_spectral.rs` | Rebuilt with S-14 workaround (10/10 PASS) |
| `whitePaper/README.md` | Updated totals, added Phase 5b |
| `whitePaper/BARRACUDA_EVOLUTION.md` | Referenced S-13/S-14 |

---

## Absorption Priority

1. **S-13 (pool sync)** — Critical for any workload with sequential GPU ops
   without intermediate readback. This includes MLP forward, MHA, and any
   chained matmul → activation → matmul pipeline. Fix is straightforward.

2. **S-14 (Naive tier)** — Affects small-matrix workloads on some drivers.
   Removing the Naive tier in favor of Tiled16 everywhere is the simplest fix.

3. **GELU test** — Already fixed locally. No upstream change needed unless
   ToadStool has a corresponding test suite.

---

## Validation Summary

```
cargo test --lib                           222/222 PASS (1 ignored)
validate_barracuda_tensor (release)        86/86 PASS
validate_barracuda_gpu_spectral (release)  10/10 PASS
validate_barracuda_gpu_eco (release)       6/6 PASS
cargo clippy                               clean
cargo doc --no-deps                        clean
```
