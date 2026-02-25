# neuralSpring → ToadStool Handoff V27: S62 Sync & Full Absorption

**Date**: February 25, 2026
**From**: neuralSpring (ML validation & evolutionary computation)
**To**: ToadStool / BarraCUDA team
**Phase**: Session 62 — S-03b fully resolved, 21/21 shaders absorbed
**ToadStool HEAD**: `02207c4a`
**License**: AGPL-3.0-or-later

---

## Executive Summary

neuralSpring has synced with `ToadStool` S60–S62 (`9404fdb4` → `02207c4a`).
The critical S-03b MHA projection hang is **fully resolved upstream**: `ToadStool`
`0c998992` decomposed the fused MHA projection into `Tensor::matmul` +
`head_split.wgsl` / `head_concat.wgsl` — exactly the approach neuralSpring
evolved locally. All **21/21** WGSL shaders from neuralSpring are now absorbed
upstream. The `evolved/mha.rs` module is now a thin wrapper delegating to
`barracuda::ops::mha::MultiHeadAttention`.

## What Changed (S62 Sync)

### S-03b: MHA Projection Hang — RESOLVED

| Item | Before (S59) | After (S62) |
|------|-------------|-------------|
| MHA projections | Fused matmul+reshape → GPU hang at B=4,S=128,H=8,d=512 | Decomposed: matmul + head_split/concat shaders |
| Local shaders | `head_split.wgsl`, `head_concat.wgsl` (2 local) | **Absorbed upstream** `0c998992` |
| `evolved/mha.rs` | CPU head-split/concat workaround | Thin wrapper → `MultiHeadAttention::new().execute()` |
| Shader count | 19/21 absorbed | **21/21 absorbed** |

### New Upstream Features Available

| Feature | Module | neuralSpring Relevance |
|---------|--------|----------------------|
| `BandwidthTier` | `unified_hardware::transfer` | Transfer cost estimation (replaces local PCIe heuristics) |
| `NvkLargeBufferLimit` | `device::driver_profile` | TITAN V 1.2 GB allocation guard |
| `ComputeDispatch` builder | `device::compute_pipeline` | Eliminates 80-line GPU dispatch boilerplate |
| `Conv2dGpu` | `ops::nn::conv2d_gpu` | Full NCHW Conv2D (stride/pad/dilation/groups) |
| `SpMM f64` | `ops::sparse_gemm_f64` | Sparse × dense matrix multiply |
| `TransE f64` | `ops::transe_score_f64` | Knowledge graph triple scoring |
| `PeakDetectF64` | `ops::peak_detect_f64` | 1D peak detection with prominence |
| `cpu-math` feature gate | `Cargo.toml` | CPU-only math without GPU dependency |
| `BarracudaError::gpu_ctx()` | `error.rs` | Convenience for GPU error wrapping |
| `unified_hardware` decomp | `unified_hardware/` | Split into cpu_executor, discovery, scheduler, traits, transfer, types |

### `unified_hardware` Refactored

The flat `unified_hardware.rs` (783 lines) was decomposed into a proper module:

| Submodule | Purpose |
|-----------|---------|
| `cpu_executor.rs` | Always-available CPU fallback with SIMD detection |
| `discovery.rs` | Runtime hardware discovery (GPU, NPU, TPU) |
| `scheduler.rs` | Operation-to-hardware matching |
| `traits.rs` | `ComputeExecutor` and `TensorStorage` traits |
| `transfer.rs` | PCIe/NVLink/SharedMemory bandwidth tiers |
| `types.rs` | `HardwareType`, capability descriptors |

neuralSpring's `metalForge/forge/` crate compiles cleanly against this.

## Current State

| Metric | Value |
|--------|-------|
| Python controls | 206/206 PASS |
| Rust lib tests | 500 PASS |
| validate_all | 145/146 PASS (1 pre-existing logsumexp) |
| Cross-spring evolution | 22/22 PASS |
| Functions rewired to upstream | 16 + MHA |
| WGSL shaders absorbed | **21/21** |
| GPU dispatch ops | 38 (9 upstream, 29 local) |
| Named tolerances | 101+ |
| Clippy warnings | 0 |
| ToadStool HEAD | `02207c4a` |

---

## Updated Absorption Targets

### Resolved This Session

| Item | Resolution |
|------|-----------|
| S-03b MHA projection hang | Upstream decomposed into matmul + head_split/head_concat |
| `head_split.wgsl` | Absorbed upstream `0c998992` |
| `head_concat.wgsl` | Absorbed upstream `0c998992` |

### Still Open

| Priority | Item | Action |
|----------|------|--------|
| 1 | **LogSumExp buffer size** | Fix binding layout: 8 bytes for f64, not 4 |
| 2 | **`Tensor::mean()`** | Merge Session 44 fix (wrong entry point + double-divide) |
| 3 | **HMM backward/Viterbi dispatch** | Add `hmm_backward_dispatch` / `hmm_viterbi_dispatch` |
| 4 | **Tridiagonal eigensolver** | NAK-optimized Sturm bisection for Papers 022–023 |

### Opportunities from New S62 Features

| Opportunity | How |
|------------|-----|
| Simplify GPU dispatch boilerplate | Use `ComputeDispatch` builder in validation binaries |
| Leverage `BandwidthTier` | Replace local PCIe heuristics in metalForge |
| NVK allocation guard | Wire `check_allocation_safe()` into TITAN V workloads |
| `cpu-math` feature | Enable lightweight barracuda imports for CPU-only paths |

---

## Lessons Learned (S62 Sync)

### 1. The Write → Absorb → Lean Cycle Completes

neuralSpring's MHA workaround (decompose projections into matmul + head reshapes)
was exactly the fix `ToadStool` adopted upstream. The Spring ecosystem works as
designed: Springs evolve what they need, `ToadStool` absorbs the best approach,
all Springs benefit.

### 2. Feature Gating Is Clean

The `gpu` feature gate in barracuda Cargo.toml (default-on) means neuralSpring's
existing code compiles unchanged. The clean separation of CPU-only modules
(`error`, `linalg`, `numerical`, `special`, `tolerances`, `validation`, `stats`)
from GPU modules enables future lightweight imports.

### 3. `unified_hardware` Decomposition Doesn't Break Consumers

The refactoring from a flat 783-line file into a proper module with public
re-exports preserved all import paths. neuralSpring's `metalForge/forge/`
compiled without changes.

---

## Modified Files (This Session)

| File | Change |
|------|--------|
| `src/evolved/mha.rs` | Rewired to delegate to upstream `MultiHeadAttention` |
| `src/evolved/mod.rs` | Updated absorption status: 21/21, S-03b resolved |
| All root docs | Updated ToadStool HEAD, session range, shader counts |
| `specs/TOADSTOOL_HANDOFF.md` | Added S62 sync entry |
| `wateringHole/` | V26 archived, V27 created |

---

*neuralSpring → ToadStool V27: Session 62 sync. S-03b FULLY RESOLVED.
21/21 WGSL shaders absorbed. evolved/mha.rs delegates to upstream.
ToadStool HEAD `02207c4a`. 500 lib tests, 145/146 validators,
0 clippy warnings, 101+ tolerances. Write → Absorb → Lean complete
for all shaders.*
