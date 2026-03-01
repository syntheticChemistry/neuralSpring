# neuralSpring — Deprecation & Migration Guide

**Date**: March 1, 2026 (Sessions 44–100)
**ToadStool HEAD**: `1dd7e338` (S70+++: cross-spring absorption, DF64 ML shaders, SimpleMlp, matmul_ref, architecture safety)
**Status**: Migration complete — deprecated modules fossilized, S-03b resolved upstream, gpu_dispatch active (47 ops, ~97% GPU, 7 domain files). coralForge unified. 218 binaries, 200/200 validate_all, 746 lib tests. Zero unsafe, zero production mocks, zero cross-primal logic. S100: 4 unused deps removed, zero clippy pedantic+nursery warnings, capability-based primal discovery.

All 12 neuralSpring shortcomings (S-01 through S-12) are absorbed by
ToadStool at `77f70b2e`. Deprecated workaround modules have been removed
from the active codebase and fossilized in `metalForge/fossils/evolved_s01_s11/`.
S-12 (eigensolver) resolved via Householder+QR — `src/eigh.rs` delegates
to upstream. Three shortcomings (S-14, S-15, S-16) discovered during
Phase 5b+ full-stack validation — **all resolved upstream** at ToadStool `a4996b34` (S39).
S-17 (pow f64 crash) also **resolved upstream** at `c82c23d1` (S58).
See `wateringHole/handoffs/archive/NEURALSPRING_V19_SESSION51_HANDOFF_FEB24_2026.md`.

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

## Still Active in `src/evolved/` (1 module + re-exports)

| Module | LOC | Why active | Path to absorption |
|--------|-----|-----------|-------------------|
| `mod.rs` | ~87 | WGSL shader re-exports for local validation binaries | Retire when validators use upstream bindings |
| `mha.rs` | 182 | Thin wrapper delegating to `barracuda::ops::mha::MultiHeadAttention` | Retire when callers use upstream 3D API directly |

S-03b fully resolved upstream at `ToadStool` `0c998992` (S60–S61).
`hmm_forward_gpu.rs` retired to `metalForge/fossils/evolved_hmm_forward_gpu/` — `HmmBatchForwardF64` (wetSpring) is now primary.

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
| S-14 | Naive matmul hang (small square matrices, complex binaries) | Medium | **RESOLVED** upstream (`a4996b34` S39: Naive tier removed) |
| S-15 | Matmul hang when elements ≤ 0.1 magnitude (WGPU/Vulkan driver bug) | Critical | **RESOLVED** upstream (`a4996b34` S39) |
| S-16 | 2D transpose dispatch uses `optimal_workgroup_size` (256) instead of tile size (16) | High | **RESOLVED** upstream (`a4996b34` S39) |
| S-17 | `pow(f64,f64)` crashes NVVM/NAK on Ada Lovelace + Volta | High | **RESOLVED** upstream (`c82c23d1` S58) |

See `wateringHole/handoffs/archive/NEURALSPRING_V12_SESSION43_HANDOFF_FEB22_2026.md`
for full diagnosis, reproduction steps, and recommended fixes.

---

## Session 49 — Deep Audit & Code Quality (February 23, 2026)

| Change | Scope | Impact |
|--------|-------|--------|
| `gpu_ops.rs` → `gpu_ops/` | 6 submodules (linalg, activation, reduction, bio, population, eigensolver) | All files < 1000 LOC |
| Tolerance centralization | 42 `NamedTolerance` entries in `tolerances/` registry | Zero standalone inline magic numbers in validation binaries |
| `clippy::doc_markdown` resolved | 31 files (8 library + 23 binaries) | Allow removed, doc comments fixed |
| `#![allow]` tightened | `validate_gpu_phase_b.rs` (9→4), `anderson_localization.rs`, `swarm_robotics.rs`, 4 binaries | Underlying code fixed, redundant suppression removed |
| Test coverage push | 264→459 lib tests, 83%→92.9% line coverage | 110 new tests across 12 modules |
| `.expect()` → graceful exits | All non-test production code | Zero `.expect()` / `.unwrap()` / `todo!()` in production |

## Session 80 — Comprehensive Debt Audit (February 26, 2026)

| Change | Scope | Impact |
|--------|-------|--------|
| Inline `1e-30` guards promoted | `gpu_ops/reduction.rs`, `gpu_ops/population.rs`, `wdm_surrogate.rs` | All 4 sites → `tolerances::LOG_ZERO_GUARD` |
| Tolerance derivation annotations | `tolerances/mod.rs` | `LOG_ZERO_GUARD`, `SWARM_FITNESS_COMPARISON`, `KAPPUS_WEGNER_REL` documented |
| Validation binary modernization | `validate_barracuda_wdm_eos.rs` | 16 `unwrap()` → `Result<Vec<f32>, String>` via `gpu_mlp_forward` |
| Shared validation helpers | `validation.rs` | `validate_tensor_unary` + `validate_tensor_reduction` extracted |
| Large binary refactoring | `validate_barracuda_tensor.rs` | 966 → 911 lines via shared helpers |
| Coverage expansion | `wdm_surrogate.rs`, `basecamp.rs` (tests_cpu.rs) | 14 + 12 new tests; 604 total lib tests, 93.5% coverage |
| WDM EOS provenance | `provenance.rs` | Added `WDM_EOS_PROVENANCE` record |
| CI evolution | `baselines.yml`, `rust.yml` | Artifact upload + cross-validation job |

## Session 81 — Deep Debt Evolution (February 26, 2026)

| Change | Scope | Impact |
|--------|-------|--------|
| 25 new named tolerances | `tolerances/{mod,gpu,registry}.rs` | 107+ → **129+ named constants** |
| Magic number sweep | 21 validation binaries | ~50 inline literals replaced |
| `spectral_entropy` rewire | `weight_spectral.rs` | → `barracuda::stats::shannon_from_frequencies` (39th function rewire) |
| Cross-platform probe | `metalForge/forge/src/probe.rs` | `#[cfg(target_os = "linux")]` gating |
| PyTorch seeding | 7 Python training scripts | Full deterministic seeding |
| Clippy fixes | `validation.rs`, `wdm_surrogate.rs`, `tolerances/gpu.rs` | All resolved |

## Session 82 — Titan V Pure Rust Pipeline Validation (February 26, 2026)

| Change | Scope | Impact |
|--------|-------|--------|
| `batched_eigh_nak_optimized_f64.wgsl` fix | `fma(f64)` → `a * b + c` | WGSL spec compliance; Sovereign Compiler re-fuses at IR |
| Explicit f64 float literals | `select()` + division contexts | Prevents abstract-float-to-f32 coercion |
| Full Titan V sweep | 33 binaries, 384/384 checks | All domains validated on NVK GV100 |
| RTX 4070 regression test | All validators | Zero regressions |

## Session 83 — ToadStool S68 Universal Precision Sync (February 26, 2026)

| Change | Scope | Impact |
|--------|-------|--------|
| 3 shader constants privatized | `WGSL_PAIRWISE_{JACCARD,HAMMING}`, `WGSL_SPATIAL_PAYOFF` | Switched to local shader copies |
| `WGSL_LOCUS_VARIANCE` removed | `forge::shaders` import | Switched to `WGSL_LOCUS_VARIANCE_F64` |
| `rk4_parallel.wgsl` → `rk4_parallel_f64.wgsl` | RK4 validator + forge | Local f32 copy (f64 requires Sovereign polyfill) |
| `WGSL_SWARM_NN_SCORES` privatized | `validate_gpu_pipeline_swarm` | Rewired to forge constant |
| `WGSL_LOGSUMEXP_REDUCE` renamed | `validate_gpu_logsumexp` | Rewired to forge constant |
| 14 ToadStool HEAD refs updated | All active docs | `17932267` → `1dd7e338` |
| variance_ddof gap closed | BARRACUDA_USAGE gap #3 | `variance_ddof(data, ddof)` at ToadStool S66 |

*Migration guide — neuralSpring rewired to modern ToadStool/BarraCUDA.*
