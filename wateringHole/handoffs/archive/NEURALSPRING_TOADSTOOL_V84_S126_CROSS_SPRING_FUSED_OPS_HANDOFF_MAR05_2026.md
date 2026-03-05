<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring → ToadStool/barraCuda Handoff V84 — Cross-Spring Fused Op Absorption

**Date**: March 5, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/barraCuda team
**License**: AGPL-3.0-or-later
**Covers**: Session 126 — fused op absorption, cross-spring provenance tracking, validation + benchmark
**Supersedes**: V83 (S125 wgpu 28 migration + BarraCUDA v0.3.3 sync)
**barraCuda**: v0.3.3 standalone (`../barraCuda/crates/barracuda`)
**ToadStool HEAD**: `9d359814` (S94b)
**wgpu**: 28

---

## Executive Summary

- **Fused op absorption**: `variance_gpu` upgraded to `VarianceF64` (fused Welford). Three new wrappers: `mean_variance_gpu`, `correlation_full_gpu`, `correlation_matrix_gpu`.
- **Cross-spring provenance**: Each fused op documents its origin Spring(s) in doc-comments and validation output.
- **New binaries**: `validate_toadstool_s94b_wgpu28` + `bench_cross_spring_evolution` (13 ops from 5 Springs).
- **Quality gates**: fmt ✓ · clippy 0 (pedantic+nursery) · 871/883 lib (12 GPU SIGSEGV upstream) · doc ✓ · 240 binaries · 218/218 validate_all.

---

## Part 1: Fused Op Absorption

### New/Upgraded Functions in `gpu_ops/reduction.rs`

| Function | BarraCUDA API | Origin Spring | What Changed |
|----------|-------------|--------------|-------------|
| `variance_gpu` | `VarianceF64::variance()` | hotSpring | Upgraded from `VarianceReduceF64` to fused Welford WGSL |
| `mean_variance_gpu` | `VarianceF64::mean_variance()` | hotSpring | **New** — single dispatch returns `[mean, variance]` |
| `correlation_full_gpu` | `CorrelationF64::correlation_full()` | wetSpring+hotSpring | **New** — returns `CorrelationResult` (means+variances+r) |
| `correlation_matrix_gpu` | `stats_f64::matrix_correlation()` | airSpring+groundSpring | **New** — n×p data → p×p Pearson matrix, single WGSL dispatch |

### Performance Impact

- `mean_variance_gpu`: Eliminates separate mean + variance dispatches (2 → 1 GPU launch)
- `correlation_full_gpu`: Returns 5 statistics from one kernel (was: `pearson_correlation_gpu` returned only r)
- `correlation_matrix_gpu`: Single GPU dispatch for full correlation matrix (was: O(p²) pairwise calls)

---

## Part 2: Cross-Spring Shader Evolution Map

```text
hotSpring  (precision physics) ──────────────────────────────────────►
  │ DF64 core, Welford variance, logsumexp, eigensolve
  │ math_f64 polyfills, compound assignment naga rewrite
  └──────────────────────────────────────► BarraCUDA v0.3.3

wetSpring  (bioinformatics) ─────────────────────────────────────────►
  │ FusedMapReduceF64 (Shannon/Simpson), log_f64 fix
  │ diversity_fusion, Bray-Curtis, Gillespie, ODE biosystems
  │ bio shaders: ANI, SNP, dN/dS, pangenome, HMM, DADA2
  └──────────────────────────────────────► BarraCUDA v0.3.3

neuralSpring (ML/neuroevolution) ────────────────────────────────────►
  │ compile_shader_universal, fused chi-squared, fused KL divergence
  │ swarm_nn, batch_fitness, pairwise_*, locus_variance
  │ hill_gate, HMM backward/Viterbi, multi_obj_fitness
  └──────────────────────────────────────► BarraCUDA v0.3.3

airSpring  (atmospheric) ────────────────────────────────────────────►
  │ batched_elementwise_f64, Richards PDE, seasonal pipeline
  │ sensor correlation, ET₀ pipeline
  └──────────────────────────────────────► BarraCUDA v0.3.3

groundSpring (hydrology) ────────────────────────────────────────────►
  │ batched_multinomial, MC propagation, RAWR weighted mean
  │ LbfgsGpu, jackknife, evolution stats (Kimura)
  └──────────────────────────────────────► BarraCUDA v0.3.3
```

---

## Part 3: Benchmark Results (bench_cross_spring_evolution)

13 ops benchmarked from 5 Springs:

| Spring | Ops | Benchmark Scenarios |
|--------|-----|-------------------|
| hotSpring | 4 | Welford mean+variance (50k), variance (50k), LogSumExp (10k), BatchedEigh (20×16) |
| wetSpring | 4 | Shannon entropy (10k), correlation_full (50k), Pearson (50k), DiversityFusion (32×200) |
| neuralSpring | 3 | chi-squared (1k), KL divergence (1k), pairwise L2 (100×32) |
| airSpring+groundSpring | 1 | correlation matrix (200×10 → 10×10) |

---

## Part 4: New Lib Tests

| Test | Expected | Exercises |
|------|----------|-----------|
| `gpu_mean_variance_fused` | mean≈5.0, var≈4.0 | `VarianceF64::mean_variance()` |
| `gpu_correlation_full_fused` | r≈1.0, mean_x≈3.0, mean_y≈6.0 | `CorrelationF64::correlation_full()` |
| `gpu_correlation_matrix_known` | diag=1.0, off-diag=±1.0 | `stats_f64::matrix_correlation()` |

---

## Quality Gates (S126)

| Gate | Result |
|------|--------|
| `cargo fmt` | Clean |
| `cargo clippy` (pedantic+nursery) | 0 warnings |
| `cargo test --lib` | 871/883 PASS (12 GPU SIGSEGV — upstream) |
| `cargo doc` | 0 warnings |
| Validation binaries | 240 |
| `validate_all` | 218/218 |
| New lib tests | +3 (880 → 883) |
| New binaries | +2 (238 → 240) |

---

## Counts

| Metric | Value |
|--------|-------|
| Library tests | 883 (871 PASS, 12 GPU upstream SIGSEGV) |
| Validation/bench binaries | 240 |
| `validate_all` | 218/218 |
| ToadStool HEAD | `9d359814` (S94b) |
| BarraCUDA version | v0.3.3 |
| wgpu version | 28 |
| Fused ops absorbed | 4 (mean_variance, correlation_full, correlation_matrix, variance upgrade) |
| Cross-spring benchmark ops | 13 from 5 Springs |

---

*V84 — neuralSpring Session 126 (March 5, 2026)*
