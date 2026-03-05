# neuralSpring → ToadStool/BarraCUDA V86 Handoff

**Date**: March 5, 2026
**From**: neuralSpring (Session 128)
**To**: ToadStool/BarraCUDA team
**License**: AGPL-3.0-or-later
**Supersedes**: V85 (S127)
**ToadStool pin**: S94b HEAD
**BarraCUDA**: HEAD (6 commits past v0.3.3: fused reductions, TensorContext, subgroup, DF64 precision tier, NAK fix, chi_squared gating)

## Executive Summary

- **`VarianceReduceF64` → `VarianceF64` rewire** across 14 files (~25 call sites) — aligns all validators/benches with the modern fused Welford API
- **ToadStool absorption tracker updated**: neuralSpring V75/S113 → V85/S127
- **BarraCUDA HEAD verified**: 0 breaking changes, 0 new regressions, all 6 post-v0.3.3 commits compatible
- **coralNAK confirmed**: local at `ecoPrimals/coralNAK/`, Phase 2 complete (sovereign Rust NVIDIA shader compiler)
- **Quality gates**: 0 clippy, 0 doc warnings, 871/883 lib tests (12 upstream GPU SIGSEGV)

## Part 1: VarianceReduceF64 → VarianceF64 Rewire

| Old API (static) | New API (instance) |
|-------------------|--------------------|
| `VarianceReduceF64::population_variance(dev, &data)?` | `var_op.variance(&data)?` |
| `VarianceReduceF64::variance(dev, &data)?` | `var_op.sample_variance(&data)?` |
| `VarianceReduceF64::mean(dev, &data)?` | `var_op.mean_variance(&data, 0)?[0]` |
| `VarianceReduceF64::std(dev, &data)?` | `var_op.sample_std_dev(&data)?` |
| `VarianceReduceF64::population_std(dev, &data)?` | `var_op.std_dev(&data)?` |

**Benefits**:
- Fused single-pass Welford (fewer GPU dispatches)
- TensorContext integration (pooled buffers, pipeline cache)
- Fp64Strategy-aware shader selection (native f64 vs DF64 per hardware)
- Instance reuse eliminates per-call pipeline compilation

**Files rewired**:

| File | Calls |
|------|-------|
| `validate_barracuda_tensor_f64.rs` | 5 (mean, pop_var, var, pop_std, std) |
| `validate_compute_dispatch.rs` | 2 |
| `validate_cross_system_dispatch.rs` | 2 |
| `validate_mixed_hardware.rs` | 5 |
| `validate_metalforge_pcie.rs` | 2 |
| `validate_nucleus_compute_dispatch.rs` | 2 |
| `validate_publication_gpu_pipeline.rs` | 1 |
| `validate_publication_mixed_hardware.rs` | 4 |
| `validate_toadstool_spectral_absorption.rs` | 3 |
| `bench_rewire_evolution.rs` | 2 |
| `bench_basecamp_parity.rs` | 1 |
| `validate_toadstool_s79_rewire.rs` | comments only |
| `meta_population/mod.rs` | doc comments |
| `tolerances/gpu.rs` | doc comment |

## Part 2: Upstream Compatibility Verification

BarraCUDA HEAD (6 commits past v0.3.3):

| Commit | Change | neuralSpring Impact |
|--------|--------|---------------------|
| `5533658` | Full codebase quality pass | None — docs/lint only |
| `0b6ebef` | Fused reduction shaders + TensorContext | Already consumed (S126) |
| `66e2774` | Subgroup detection + 10 ops to TensorContext | Compatible — no API breaks |
| `7797d90` | DF64 precision tier for 15 ops | Compatible — Fp64Strategy already used |
| `4629bdd` | DF64 naga rewriter NAK fix | Beneficial — fixes TITAN V compound assignments |
| `15d3774` | chi_squared feature gating | Compatible — no API breaks |

## Part 3: ToadStool Action Items (Updated)

### New from S128

1. **Fused LSTM cell WGSL shader**: neuralSpring uses per-step `Tensor::matmul` + CPU sigmoid/tanh. A fused shader eliminates host round-trips — natural ToadStool streaming candidate.
2. **Autocorrelation GPU op**: CPU-only in neuralSpring. Useful for time-series regime detection.
3. **R² score GPU op**: CPU-only in neuralSpring. Useful for model evaluation on GPU.

### Resolved from V85

4. Flash attention: available in barraCuda (`FlashAttention`). neuralSpring can wire when needed.
5. LayerNorm+GELU: available in barraCuda (`layernorm_fused_f64`). neuralSpring can wire when needed.

### Ongoing

6. **GPU SIGSEGV**: 12 tests fail (BarraCUDA Tensor + ComputeDispatch on llvmpipe). Not blocking.
7. **L-BFGS**: Available but not wired in neuralSpring.
8. **TensorSession batching**: Available but not wired — high potential for multi-op inference speedup.

## Part 4: Quality Gates (S128)

| Gate | Result |
|------|--------|
| `cargo fmt` | clean |
| `cargo clippy` | 0 warnings |
| `cargo doc` | 0 warnings |
| `cargo test --lib` | 871/883 (12 upstream) |
| Named tolerances | 140+ |
| `#[allow(` | 0 |
| Files >1000 LOC | 0 |

## Part 5: coralNAK Status

Sovereign Rust NVIDIA shader compiler at `ecoPrimals/coralNAK/`:
- Phase 2 complete: NAK sources wired, 183 tests, 0 compile errors
- Phase 3 next: naga SPIR-V frontend
- Phase 4 next: f64 DFMA software lowering
- Future barraCuda path: WGSL → naga → coral-nak → native binary (bypasses SPIR-V → driver compiler)

*Supersedes: V85 (S127)*
