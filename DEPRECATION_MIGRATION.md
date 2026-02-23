# neuralSpring — Deprecation & Migration Guide

**Date**: February 23, 2026 (post-Sessions 44–46)
**ToadStool HEAD**: `6ee71f07` + 2 local fixes pending absorption (mean_reduce, chi²)
**Status**: Migration complete — deprecated modules fossilized, S-03b locally resolved, gpu_dispatch active

All 12 neuralSpring shortcomings (S-01 through S-12) are absorbed by
ToadStool at `77f70b2e`. Deprecated workaround modules have been removed
from the active codebase and fossilized in `metalForge/fossils/evolved_s01_s11/`.
S-12 (eigensolver) resolved via Householder+QR — `src/eigh.rs` delegates
to upstream. Three new shortcomings (S-14, S-15, S-16) discovered during
Phase 5b+ full-stack validation. Two upstream bugs fixed in Session 44
(`Tensor::mean()` entry point, chi-squared expected values).
See `wateringHole/handoffs/NEURALSPRING_V14_SESSION46_HANDOFF_FEB23_2026.md`.

---

## Fossilized (~2,864 LOC evolved + ~1,127 LOC bench)

See `metalForge/fossils/FOSSIL_RECORD.md` for the full inventory.

### Evolved Modules → `metalForge/fossils/evolved_s01_s11/`

| Module | LOC | Shortcoming | BarraCUDA Replacement |
|--------|-----|-------------|----------------------|
| `fused_pipeline.rs` | 680 | S-01 | `TensorSession` |
| `fused_mlp.rs` | 356 | S-01/S-11 | `TensorSession::{matmul, relu, gelu, run}` |
| `fused_transformer.rs` | 725 | S-01/S-11 | `TensorSession::{head_split, attention, layer_norm}` |
| `layer_norm.rs` | 268 | S-08 | `Tensor::layer_norm_wgsl()` |
| `log_softmax.rs` | 259 | S-09 | `Tensor::log_softmax_wgsl()` |
| `matmul_cpu_tiled.wgsl` | 270 | S-02 | `ops::matmul` CpuTiled32 |
| `matmul_gpu_evolved.wgsl` | 306 | S-02 | `ops::matmul` GpuEvolved32 |

### Bench Binaries → `metalForge/fossils/bench/`

| Binary | Why fossilized |
|--------|----------------|
| `bench_fused_inference.rs` | Deep coupling to fused pipeline |
| `bench_scaling.rs` | Deep coupling to fused pipeline |

---

## Rewired (Active — Using Native APIs)

| Binary | What changed | Date |
|--------|-------------|------|
| `bench_barracuda_tensor` | Evolved `layer_norm`/`log_softmax` → native `Tensor::layer_norm_wgsl()`/`log_softmax_wgsl()` | Feb 20 |
| `validate_barracuda_tensor` | Same rewiring (earlier) | Feb 20 |
| `gpu.rs` | CPU path → `WgpuDevice::new_cpu_relaxed()` | Feb 20 |
| 7 GPU binaries | Duplicated device init (~800 LOC) → unified `Gpu::new()` | Feb 21 |
| 10 validation binaries | Hardcoded tolerances → centralized tolerances/ module constants | Feb 21 |

---

## Still Active in `src/evolved/` (2 modules + exports)

| Module | LOC | Why active | Path to absorption |
|--------|-----|-----------|-------------------|
| `mod.rs` | ~50 | WGSL shader exports (`batch_fitness_eval`, `rk4_parallel`, `mean_reduce`) | Absorb into `barracuda::ops` |
| `mha.rs` | 182 | Evolved MHA with GPU head_split/head_concat shaders (S-03b locally resolved) | ToadStool native MHA when projection shaders stabilize |
| `hmm_forward_gpu.rs` | 270 | No BarraCUDA equivalent | Candidate for `ops::hmm` |

## Newly Fossilized (Session 40)

| Module | LOC | Reason | Location |
|--------|-----|--------|----------|
| `tensor_sync.rs` | 179 | S-13 **FIXED** upstream at `5437c170`. Zero callers remain | `metalForge/fossils/evolved_s13/` |

### S-03b: Locally Resolved via WGSL Head Split/Concat Shaders

The z-dispatch fix (S-03) was absorbed by ToadStool. The native
`Tensor::multi_head_attention` projection shaders hang on RTX 4070 / Vulkan,
but S-03b is locally resolved via dedicated `head_split.wgsl` and
`head_concat.wgsl` shaders validated by `validate_mha_gpu` (10/10 PASS at
production sizes up to B=4, S=128, H=8, d=512).

**Binaries using evolved MHA**:
- `validate_barracuda_ml_inference`
- `validate_mha_gpu` (GPU head_split/head_concat validation)
- `bench_transformer_block`

---

## Deep Evolution (Sessions 1-2 Audit)

| Change | Scope | Impact |
|--------|-------|--------|
| `primitives.rs` module | 8 library modules | Shannon (6 variants), Hill (3), sigmoid (2), RK4 (2) centralized; magic numbers (`1e-15`, `1e-300`, `1e-20`) promoted to named constants |
| Flat row-major HMM | `hmm.rs` + 5 binaries | `Vec<Vec<f64>>` → flat `Vec<f64>` for transition, emission, alpha, posterior — GPU buffer-ready |
| Flat row-major spectral | `spectral_commutativity.rs` + 3 binaries | All matrix ops → flat `Vec<f64>` with explicit `n` dimension |
| `require!` macro | `validation.rs` + 8 binaries | All `.expect()` → graceful `require!(h, result, label)` — no panic on GPU failure |
| Zero-copy genotypes | `eco_dynamics.rs` | `Vec<u8>` → `&[u8]`, `HashSet<Vec<u8>>` → `HashSet<&[u8]>` |
| SPDX headers | 40 Python/shell files | All files have `AGPL-3.0-or-later` identifier |
| Runtime discovery | `surrogate_validation.py` | Cross-primal `airSpring` path → env var + sibling probe |

---

## Migration Complete

| Priority | Action | Status |
|----------|--------|--------|
| ~~Done~~ | Rewire `validate_barracuda_tensor` | **Complete** |
| ~~Done~~ | Rewire `gpu.rs` to `new_cpu_relaxed()` | **Complete** |
| ~~Done~~ | Rewire `bench_barracuda_tensor` to native ops | **Complete** |
| ~~P1~~ | Migrate MHA to native — **S-03b locally resolved** (head_split/head_concat WGSL) | Kept evolved::mha + GPU shaders |
| ~~P2~~ | Migrate fused benchmarks | **Fossilized** |
| ~~P3~~ | Remove WGSL shaders | **Fossilized** |
| ~~P4~~ | Remove evolved modules | **Fossilized** (except mha + hmm) |

---

## Phase 5a: New BarraCUDA Shortcomings Discovered

GPU `Tensor` validation across 7 domains uncovered 3 new bugs:

| # | Shortcoming | Severity | Status |
|---|-------------|----------|--------|
| S-14 | Naive matmul hang (small square matrices, complex binaries) | Medium | Characterized, workaround (non-square shapes) |
| S-15 | Matmul hang with negative or sparse f32 input data | Critical | Characterized, workaround (positive-only data) |
| S-16 | 2D transpose dispatch uses `optimal_workgroup_size` (256) instead of tile size (16) | High | Root cause confirmed, one-line fix identified |

See `wateringHole/handoffs/archive/NEURALSPRING_V12_SESSION43_HANDOFF_FEB22_2026.md`
for full diagnosis, reproduction steps, and recommended fixes.

---

*Migration guide — neuralSpring rewired to modern ToadStool/BarraCUDA.*
