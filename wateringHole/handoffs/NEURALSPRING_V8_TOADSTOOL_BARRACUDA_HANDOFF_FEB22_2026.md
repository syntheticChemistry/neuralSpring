# neuralSpring v8 → ToadStool / BarraCUDA Team Handoff

**Date:** February 22, 2026 (post-Session 39 sync)
**From:** neuralSpring (ML validation & evolutionary computation biome)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**Supersedes:** `archive/NEURALSPRING_V7_TOADSTOOL_BARRACUDA_HANDOFF_FEB22_2026.md`
**ToadStool HEAD:** `d45fdfb3` (Session 39)

---

## Executive Summary

Session 39 absorbed 5 of neuralSpring's 8 local WGSL shaders as generalized
upstream variants. Combined with the 8 already absorbed at `77f70b2e`,
**13 of 17 shaders (76%) are now upstream**. S-13 (PooledBuffer race) is
**FIXED**. Precision fixes (TS-001, TS-003, TS-004) flow automatically.

| Category | Before (V7) | After (V8) |
|----------|-------------|------------|
| Shaders upstream | 8/16 (50%) | 13/17 (76%) |
| S-13 PooledBuffer race | Open | **FIXED** |
| Conv2D/Pool WGSL | Not available | Available (not wired) |
| Trig/pow precision | Baseline | **Improved** (TS-001, TS-003) |

Phase 5b validation unchanged: **ALL GREEN** across all 7 tiers.

| Tier | Coverage | Status |
|------|----------|--------|
| Python control (Py) | 25/25 (100%) | **ALL PASS** |
| Rust CPU (Rs) | 25/25 (100%) | **ALL PASS** |
| BarraCUDA CPU (bC) | 24/25 (96%) | **ALL GREEN** |
| BarraCUDA GPU Tensor (gT) | 23/25 (92%) | **ALL GREEN** |
| metalForge WGSL (mF) | 14/25 (56%) | **ALL PASS** |
| GPU Pipeline (gP) | 7/25 (28%) | **ALL PASS** |
| Cross-dispatch (xD) | 15/15 (100%) | **ALL GREEN** |

**Grand total: 1560+ validation checks** (206 Python + 1354+ Rust/GPU).
115 validation binaries, 31 modules, 17 WGSL shaders (4 still local-only).

---

## Part 1: What Changed Since V7

### 1.1 Five Shaders Absorbed Upstream (Session 39)

ToadStool absorbed neuralSpring's local shaders as generalized variants.
The upstream versions improve on our originals:

| Shader | Upstream Improvement |
|--------|---------------------|
| `pairwise_l2.wgsl` | O(1) closed-form pair index decode vs O(N) linear search |
| `multi_obj_fitness.wgsl` | Bessel correction (n-1 divisor) for unbiased variance |
| `hill_gate.wgsl` | Mode 0 (paired) / mode 1 (grid) generalization |
| `swarm_nn_forward.wgsl` | Generic MLP dimensions via `SwarmParams`, clamped sigmoid for stability |
| `mean_reduce.wgsl` | Effectively identical (barracuda credits neuralSpring) |

**Status**: Local copies retained for validation compatibility. Validators use
local binding layouts via `include_str!`. Future migration to upstream APIs
when Rust wrappers are available.

### 1.2 Bug Fixes Now Flowing to neuralSpring

| Fix | Impact | How |
|-----|--------|-----|
| **S-13** PooledBuffer race | Eliminates buffer reuse before GPU completion | Deferred return + device poll |
| **TS-003** trig precision | Better sin/cos/asin accuracy | 7-term Taylor + Cody-Waite |
| **TS-001** pow_f64 | Handles 2^k for |k| up to 1023 | Extended exp/log polynomials |
| **TS-004** FusedMapReduceF64 | Both passes in single command encoder | Buffer conflict eliminated |

All flow automatically via neuralSpring's path dependency on barracuda.

### 1.3 New Capabilities Available

| Capability | Files | Use Case | Wired? |
|-----------|-------|----------|--------|
| Conv2D WGSL | `ops/nn/conv2d.wgsl` | LeNet-5 conv layers | No (shader exists, executor not wired) |
| MaxPool2D WGSL | `ops/nn/maxpool2d.wgsl` | LeNet-5 pooling | No |
| AvgPool2D WGSL | `ops/nn/avgpool2d.wgsl` | Alternative pooling | No |
| CPU Conv/Pool | `cpu_conv_pool.rs` | CPU fallback for Conv/Pool | Yes (CpuExecutor) |
| ESN export/import | `esn_v2.rs` | GPU-train → NPU-deploy pipeline | No |

---

## Part 2: What neuralSpring Still Has Locally

### 2.1 Four Local-Only Shaders

| Shader | Domain | Binary | Checks | Why Still Local |
|--------|--------|--------|--------|-----------------|
| `head_split.wgsl` | MHA | `validate_mha_gpu` | 5/5 | Upstream uses different params (`HeadSplitParams` vs `Params`) |
| `head_concat.wgsl` | MHA | `validate_mha_gpu` | 5/5 | Same — paired with head_split |
| `xoshiro128ss.wgsl` | PRNG | `validate_gpu_prng` | 5/5 | Upstream uses one-shot model; local has persistent state |
| `swarm_nn_scores.wgsl` | Swarm | `validate_gpu_pipeline_swarm` | PASS | No upstream equivalent |

### 2.2 Evolved Modules Still Active

| Module | LOC | Issue | Status |
|--------|-----|-------|--------|
| `mha.rs` | 182 | S-03b: native MHA projection shaders hang | CPU workaround via local shaders |
| `hmm_forward_gpu.rs` | 270 | `HmmBatchForwardF64` now validated (11/11 PASS, 2.47e-10 diff) | **Upstream wired** — local retained for f32 fallback |

---

## Part 3: Bug Reports and Fixes (Updated)

### 3.1 S-16 — FIXED (transpose dispatch)

Same as V7. One-line fix: `const TILE: u32 = 16`.

### 3.2 S-15 — Root-Caused (matmul hang on small-magnitude data)

Same as V7. Workaround: `rng.uniform() * 0.5 + 0.5`. WGPU/Vulkan driver bug.

**Updated recommendation**: ToadStool should consider adding a magnitude guard
in `Tensor::matmul` that warns/errors when min(|input|) < 0.1 on Vulkan backend.

### 3.3 S-14 — Workaround Confirmed (naive matmul hang)

Same as V7. A×B^T pattern avoids Naive tier selection.

### 3.4 S-13 — FIXED (Session 39)

**Previously**: PooledBuffer dropped before GPU work completed → buffer reuse race.
**Fix**: `drop()` calls `pool.defer_return(buffer, bucket)`. `drain_pending()`
runs `device.poll(MaintainBase::Poll)` before returning buffers to pool.
**Status**: Fixed upstream. neuralSpring benefits automatically.

### 3.5 S-03b — Partial Fix (MHA projection shaders)

Same as V7. Local `head_split.wgsl`/`head_concat.wgsl` decompose MHA.
10/10 PASS at production sizes.

---

## Part 4: What ToadStool Has That neuralSpring Should Use More

### High Impact

| API | Current Use | Opportunity |
|-----|------------|-------------|
| `HmmBatchForwardF64` | **VALIDATED** (11/11 PASS, 2.47e-10 diff) | ✅ Done — `validate_barracuda_hmm_f64` |
| `spectral::{anderson_*, hofstadter_*, lanczos}` | Not used | Replace local model construction code |
| `WGSL_BATCHED_EIGH_NAK_OPTIMIZED` | Not used | GPU-native Anderson eigensolve |
| `TensorSession` ML ops | Available but not wired | Replace `evolved::mha` CPU workaround |
| `cpu_conv_pool::{conv2d, max_pool2d}` | Not used | LeNet-5 full conv+pool validation |

### Medium Impact

| API | Current Use | Opportunity |
|-----|------------|-------------|
| `ReduceScalarPipeline::sum_f64` | Not used (local `mean_reduce`) | Fitness aggregation, log-likelihood |
| `BatchedRK4F64` | Not used | CPU-threaded ODE parameter sweeps |
| `ops::bio::{FelsensteinGpu, GillespieGpu}` | Not used | Future paper extensions |
| `ops::linalg::{InverseF64, LinSolveF64}` | Not used | GPU dense linear algebra |

---

## Part 5: Concrete Next Steps

### For ToadStool (by priority)

| Priority | Action | Impact |
|----------|--------|--------|
| P0 | **Wire Conv2D/MaxPool2D WGSL** to GpuExecutor | Enables full LeNet-5 GPU validation |
| P0 | **Fix S-15** in matmul dispatch (magnitude guard) | Unblocks all negative-data workloads |
| P1 | **Expose `hill_gate`/`multi_obj_fitness`/etc.** as Rust wrapper APIs | neuralSpring can migrate off local copies |
| P1 | **Fix S-03b** (decompose native MHA projection shaders) | neuralSpring retires `evolved::mha` |
| P2 | **Fix S-14** (remove Naive matmul tier) | Simplifies matmul path |
| P2 | **Unify `head_split`/`head_concat` params** | neuralSpring can use upstream shaders |

### For neuralSpring (by priority)

| Priority | Action | Impact |
|----------|--------|--------|
| ~~P1~~ | ~~Wire `HmmBatchForwardF64` for Paper 016~~ | ✅ **DONE** — `validate_barracuda_hmm_f64` (11/11 PASS) |
| ~~P1~~ | ~~Wire upstream bio op wrappers~~ | ✅ **DONE** — `validate_barracuda_bio_ops` (12/12 PASS) |
| ~~P1~~ | ~~Benchmark upstream vs local dispatch~~ | ✅ **DONE** — `bench_upstream_vs_local` (0.92–1.16× overhead) |
| P1 | Wire `cpu_conv_pool` for LeNet-5 conv/pool layers | Blocked: `pub(crate)` — needs ToadStool to expose |
| P2 | Wire `WGSL_BATCHED_EIGH_NAK_OPTIMIZED` for Paper 023 | GPU-native Anderson eigensolve |
| P2 | Migrate validators to upstream shader binding layouts | Retire 5 local shader copies |
| P3 | Wire `ReduceScalarPipeline::sum_f64` for pipeline validators | Replace local `mean_reduce.wgsl` |

---

## Appendix A: Shader Absorption Summary

| Category | Count | Shaders |
|----------|-------|---------|
| Identical upstream (77f70b2e) | 8 | hmm_forward_log, batch_fitness_eval, rk4_parallel, pairwise_jaccard, pairwise_hamming, locus_variance, spatial_payoff, batch_ipr |
| Generalized upstream (d45fdfb3) | 5 | pairwise_l2, multi_obj_fitness, hill_gate, swarm_nn_forward, mean_reduce |
| Still local-only | 4 | head_split, head_concat, xoshiro128ss, swarm_nn_scores |
| **Total** | **17** | **13 upstream (76%)** |

## Appendix B: Codebase Health

| Metric | Value |
|--------|-------|
| Rust lib tests | 256 unit + 9 doc-tests |
| Line coverage | 94.9% |
| Clippy | 0 warnings (pedantic + nursery) |
| `unsafe` | Forbidden (`#![forbid(unsafe_code)]`) |
| Python baselines | 206/206 PASS |
| Total validation checks | 1560+ |
| Validation binaries | 115 |
| WGSL shaders | 17 (13 upstream, 4 local) |
| Modules | 31 + 3 evolved |

## Appendix C: Active Handoff Documents

| Document | Status |
|----------|--------|
| `NEURALSPRING_V8_TOADSTOOL_BARRACUDA_HANDOFF_FEB22_2026.md` | **Current** |
| `archive/NEURALSPRING_V7_*_FEB22_2026.md` | Superseded |
| `archive/NEURALSPRING_V6_*_FEB22_2026.md` (2 files) | Superseded |
| `archive/NEURALSPRING_V5_*_FEB22_2026.md` (2 files) | Superseded |

---

*neuralSpring v8 handoff — Session 39 sync. 13/17 shaders upstream, S-13 fixed,
ALL GREEN. Following hotSpring's unidirectional handoff pattern.*
