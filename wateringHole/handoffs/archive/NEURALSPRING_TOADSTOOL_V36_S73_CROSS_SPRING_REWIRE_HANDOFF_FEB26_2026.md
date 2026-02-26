# neuralSpring → ToadStool/BarraCUDA Handoff V36 — Cross-Spring Rewiring

**Session 73 | February 26, 2026**
**Previous**: V35 (Session 72 — Full ToadStool sync, all shortcomings resolved)

## Executive Summary

Session 73 completes the **cross-spring rewiring** of neuralSpring production code
to use modern ToadStool/BarraCUDA Tensor APIs. Four new upstream rewires validate
and benchmark the cross-spring shader evolution pipeline.

**Status**: ALL 17 shortcomings RESOLVED upstream. 21 functions now rewired to upstream
(was 17). 39/39 cross-spring evolution validator PASS. 580/580 lib tests PASS.

---

## S73 Rewiring Details

### 1. Viterbi `argmax_dim(0)` — neuralSpring → ToadStool S60

**Before**: `hmm_viterbi_step_gpu` computed scores on GPU via `Tensor::max_dim(0)`,
then read back the full score matrix to CPU and ran a manual argmax loop per state.

**After**: Uses `Tensor::argmax_dim(0)` + `to_vec_u32()` to extract psi indices
directly from the Tensor API. No score matrix readback needed for argmax.

**Location**: `src/gpu_ops/bio.rs` lines 323–383

**Lineage**: neuralSpring V20 requested `argmax_dim(axis)` for GPU Viterbi paths.
ToadStool implemented in `tensor_axis_ops.rs` (S60, commit `0c998992`).
The CPU path in `hmm.rs` and `cpu_fallback.rs` still uses `max_by` / manual loop
(appropriate for small N).

### 2. `Dispatcher::softmax_row_wise` — neuralSpring V20 → ToadStool S60

**Before**: Row-wise softmax required manual per-row dispatch or the
`ScaledDotProductAttention` operator. `neural_pgm::weight_to_transition` did
manual per-row softmax on CPU.

**After**: New `Dispatcher::softmax_row_wise(matrix, n_rows, n_cols)` method
uses `Tensor::softmax_dim(1)` when GPU is available, falling back to
`weight_to_transition` on CPU.

**Location**: `src/gpu_dispatch/dispatch_ops.rs` lines 83–99

**Precision**: f32 Tensor path gives ~1e-7 agreement with f64 CPU reference.
Tolerance `DISPATCH_F32_ROUNDTRIP` (1e-6) covers this.

### 3. `fst_single_locus` — wetSpring S53 → BarraCUDA bio

**Before**: neuralSpring only computed multi-locus Weir-Cockerham θ (FST).
Single-locus F-statistics (θ, f, F) were not available.

**After**: New `meta_population::fst_single_locus` wraps upstream
`barracuda::ops::bio::fst_variance_decomposition`. Returns `(fst, f_is, f_it)`.
Wright's identity `(1-F_IT) = (1-F_IS)(1-F_ST)` validated to machine precision.

**Location**: `src/meta_population.rs` + `src/gpu_dispatch/dispatch_ops.rs`

### 4. `pairwise_fst_full` — enriched multi-locus FST

**Before**: `pairwise_fst` returned only θ (ratio-of-sums estimator).

**After**: `pairwise_fst_full` calls upstream `fst_variance_decomposition` per-locus,
then averages (mean-of-ratios estimator). Returns `(fst, f_is, f_it)`.

**Note**: Mean-of-ratios and ratio-of-sums are different estimators — the θ values
differ by ~1% on typical data. Both are valid Weir-Cockerham estimators.

---

## Cross-Spring Evolution Lineage

```text
hotSpring → BarraCUDA precision layer:
  • df64_core.wgsl (double-float f32-pair emulation)
  • pow_f64 polyfill (S-17 RESOLVED — patch_transcendentals_in_code covers pow)
  • Fp64Strategy (Native/Hybrid detection)
  • GpuDriverProfile (hardware-adaptive dispatch)
  • Taylor-series sin/cos (7-term + Cody-Waite)
  • Lanczos eigensolver (lattice QCD heritage)

wetSpring → BarraCUDA bio+spectral layer:
  • HMM forward/backward (phylogenetics)
  • 5 ODE bio systems
  • NMF, Anderson localization, ridge regression
  • fst_variance_decomposition [S73 rewire — now used by neuralSpring]

neuralSpring → BarraCUDA validation+ops layer:
  • ValidationHarness, exit_no_gpu, require! macro
  • batch_fitness, pairwise_l2/hamming/jaccard, spatial_payoff
  • eigh, batch_ipr, swarm_nn, KernelRouter
  • ESD, marchenko_pastur, effective_rank, gelu/hmm_forward dispatch

S73 cross-spring rewiring (this session):
  • argmax_dim(axis) → Viterbi psi extraction (was CPU loop)
  • softmax_dim(axis) → Dispatcher::softmax_row_wise (was manual per-row)
  • fst_variance_decomposition → fst_single_locus + pairwise_fst_full
  • Total: 21 functions rewired to upstream (was 17)
```

---

## Benchmark Highlights

### Correctness: 39/39 cross-spring evolution validator PASS

### Performance observations

| Operation | GPU path | CPU path | Notes |
|-----------|----------|----------|-------|
| `softmax_row_wise(4×64)` | ~4ms | ~1µs | Device init overhead dominates; GPU wins at >1K rows |
| `softmax_row_wise(64×256)` | ~4ms | ~65µs | Still device-bound; batching needed |
| `viterbi(s=3, T=10)` | ~76ms | ~0.6µs | Per-step round-trips; GPU wins for large N |
| `viterbi(s=32, T=10)` | ~83ms | ~10µs | GPU cost is dispatch overhead, not compute |
| `fst_single_locus` | N/A (CPU) | sub-µs | Scalar reduction; no GPU benefit |

**Key insight**: For small workloads (N<64 states, <256 matrix elements), the GPU
dispatch overhead exceeds the computation. The `Dispatcher` size-based thresholds
correctly route these to CPU. The rewiring proves API correctness; performance
benefit comes at scale (which neuralSpring validation workloads don't exercise).

---

## Tolerances Added

| Constant | Value | Justification |
|----------|-------|---------------|
| `DISPATCH_F32_ROUNDTRIP` | 1e-6 | f64→f32 Tensor→f64: f32 mantissa gives ~7 digits |
| `DISPATCH_VITERBI_F32` | 1e-5 | Viterbi accumulates T steps of f32 max-reduction |

Total named tolerances: **107+** (was 105+).

---

## Absorption Gap (unchanged from V35)

ToadStool references neuralSpring handoffs V16/V18. Handoffs V19–V36 have not
been consumed. V36 provides the complete delta including:
- Tolerance standardization (150+ named constants)
- Smart refactoring (gpu_dispatch 862→304 lines)
- 4 new upstream rewires (argmax_dim, softmax_dim, fst_variance_decomposition)
- Complete cross-spring lineage documentation

---

## Full Metrics (S73)

| Metric | Value |
|--------|-------|
| Library tests | 580/580 PASS |
| Integration tests | 9/9 PASS |
| Cross-spring validator | 39/39 PASS |
| Coverage | 94.53% |
| Named tolerances | 107+ |
| Upstream rewires | 21 |
| Shortcomings resolved | 17/17 |
| Clippy warnings | 0 |
| SPDX compliance | 100% |
| Files ≤1000 lines | 100% |

---

*AGPL-3.0-or-later | neuralSpring → ToadStool V36 | Session 73 | February 26, 2026*
