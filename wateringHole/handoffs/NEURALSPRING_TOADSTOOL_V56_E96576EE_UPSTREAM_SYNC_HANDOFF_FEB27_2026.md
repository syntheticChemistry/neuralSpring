# neuralSpring → ToadStool/BarraCUDA Handoff: V56 Upstream Sync + Evolution Review

**Date**: February 27, 2026
**From**: neuralSpring (Session 88+)
**To**: ToadStool/BarraCUDA team
**ToadStool pin**: `e96576ee` (3 commits past `f0feb226`: CPU feature-gate fix, root docs cleanup, GPU device-lost resilience)
**neuralSpring**: 172 binaries, 668 lib + 43 forge tests, **171/172 validate\_all**
**Supersedes**: V55 (Phase 4 WGSL + streaming + NUCLEUS, now in archive/)
**License**: AGPL-3.0-or-later

---

## Executive Summary

This handoff documents neuralSpring's sync to ToadStool `e96576ee` and a
comprehensive review of the ToadStool evolution from S50–S68+. The primary
code change is rewiring `compile_shader_f64_hybrid` to upstream's
`compile_shader_df64`. All validation passes (171/172).

The ToadStool evolution since S50 represents a massive maturation:
- **703 WGSL shaders** — all f64 canonical, zero f32-only
- **Universal precision pipeline** — F16/F32/F64/DF64 via single entry point
- **46 cross-spring absorption items** completed
- **4,176+ tests upstream** — zero warnings, zero debt

---

## Part 1: What neuralSpring Rewired

### `compile_shader_f64_hybrid` → `compile_shader_df64`

| Aspect | Before (V55) | After (V56) |
|--------|-------------|-------------|
| DF64 preamble | Manual: `barracuda::ops::lattice::su3::{WGSL_DF64_CORE, WGSL_DF64_TRANSCENDENTALS}` | Upstream: `WgpuDevice::compile_shader_df64()` (proper `include_str!`) |
| Pipeline | `format!` concat → `compile_shader_f64` | `compile_shader_df64` (ILP optimizer + Sovereign compiler) |
| Provenance | Reaching into lattice QCD module for DF64 constants | Correct: device-level API |

This eliminates neuralSpring's only direct dependency on barracuda's lattice QCD
module for non-lattice work.

### Pin Update: `f0feb226` → `e96576ee`

| Commit | What | Impact |
|--------|------|--------|
| `89356efa` | CPU feature-gate fix: gate shader refs in numerical/stats | Prevents compile errors without `gpu` feature |
| `92679172` | Root docs cleaned, stale scripts archived | Cleaner upstream tree |
| `e96576ee` | GPU device-lost resilience for standalone testing | More robust testing |

---

## Part 2: ToadStool Evolution Review (S50–S68+)

### Precision Architecture (S66–S68)

ToadStool evolved from mixed f32/f64 shaders to a **universal precision architecture**:

```
f64 canonical source
    ↓ compile_shader_universal(source, precision)
    ├── F32:  downcast_f64_to_f32()
    ├── F64:  compile_shader_f64() (driver patching + Sovereign)
    ├── DF64: downcast_f64_to_df64() → compile_shader_df64()
    └── F16:  downcast_f64_to_f16() (range clamping)
```

- **S68**: Dual-layer universal precision — 296 f32-only shaders eliminated
- **S67**: `compile_shader_universal`, `Precision::Df64`, template system
- **S66**: `compile_shader_df64()`, 6 DF64 math shaders, `anyhow` → typed errors

### Cross-Spring Absorptions (S50–S66)

| Session | Items | Highlights |
|---------|-------|------------|
| S52 | 18 items | CG infrastructure, domain dispatch, all-domains absorption |
| S51 | 5 items | CG shaders, ESN NPU, generic ODE, CPU solver |
| S54 | 5 items | baseCamp primitives, 5 WGSL shaders |
| S56 | Final | All cross-spring absorptions complete — 46 items total |

### Previously-Missing APIs (Now Available)

| API | Location | neuralSpring Status |
|-----|----------|-------------------|
| `LogSumExp` | `barracuda::ops::logsumexp` | Wired + validated (5/5 PASS) |
| `PairwiseDistance` | `barracuda::ops::pairwise_distance` | Available; PairwiseL2Gpu used for bio |
| `BatchedEighGpu` | `barracuda::ops::linalg::batched_eigh_gpu` | Wired in `gpu_ops/eigensolver.rs` |

### Code Quality Evolution

| Metric | S50 | S68+ |
|--------|-----|------|
| Tests | ~3,500 | 4,176+ |
| f32-only shaders | 296 | 0 |
| Warnings | >0 | 0 |
| Absorption items pending | 46 | 0 |

---

## Part 3: What Remains Local (neuralSpring Sovereign)

### Sovereign Folding Shaders (15 df64 WGSL)

These are AlphaFold2/Evoformer primitives that are domain-specific to
neuralSpring. They use `compile_shader_df64` (now upstream) but the shader
sources are neuralSpring's own:

- layer\_norm, GELU, sigmoid, softmax
- SDPA scores/apply, IPA scores
- triangle mul outgoing/incoming, triangle attention
- MSA row/col attention, outer product mean
- backbone, torsion

These are not absorption candidates — they're neuralSpring's science shaders.

### Phase 4 Shaders (4 validation WGSL)

- `hmm_backward_log.wgsl` — validated, target: `barracuda::ops::bio::hmm_backward`
- `hmm_viterbi.wgsl` — validated, target: `barracuda::ops::bio::hmm_viterbi`
- `matrix_correlation.wgsl` — validated, target: `barracuda::stats::matrix_correlation_gpu`
- `linear_regression.wgsl` — validated, target: `barracuda::stats::linear_regression_gpu`

These are ready for upstream absorption when ToadStool is ready.

### CPU Fallbacks (2 scalar functions)

- `hmm_backward_step` — local scalar CPU; GPU path uses Tensor matmul
- `hmm_viterbi_step` — local scalar CPU; GPU path uses Tensor argmax

These provide the CPU fallback when GPU is unavailable. Not redundant.

---

## Part 4: Validation Matrix

| Suite | Result |
|-------|--------|
| `cargo test --lib` | **668/668 PASS** |
| `cargo test -p neural-spring-forge --lib` | **43/43 PASS** |
| `cargo fmt --check` | PASS |
| `validate_all` | **171/172 PASS** (1 pre-existing WDM damping) |
| Total checks | **2970+** |

### Key validator results post-rewire

| Validator | Checks | Status |
|-----------|--------|--------|
| `sovereign_folding_gpu` | 21/21 | Uses `compile_shader_df64` (rewired) |
| `sovereign_folding_gpu_pipeline` | 16/16 | Uses `compile_shader_df64` (rewired) |
| `gpu_shader_phase4` | 22/22 | Direct WGSL dispatch (unaffected) |
| `streaming_spectral_pipeline` | 28/28 | BatchIprGpu + Dispatcher (unaffected) |
| `toadstool_spectral_absorption` | 294/294 | Full absorption readiness |
| `cross_spring_evolution` | 52/52 | All cross-spring provenance |

---

## Part 5: Cross-Spring Alignment

| Spring | Version | ToadStool Pin | Key State |
|--------|---------|---------------|-----------|
| wetSpring | V61 | `e96576ee` | 79 barracuda primitives, nanopore field genomics |
| hotSpring | V0614 | `e96576ee` | df64 strategy origin, NVK patterns, 22 papers |
| neuralSpring | V56 | `e96576ee` | 42 WGSL shaders, compile\_shader\_df64 rewired |

---

*End of V56 handoff. Previous handoffs archived in `wateringHole/handoffs/archive/`.*
