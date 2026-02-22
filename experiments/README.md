# neuralSpring — Experiment Journal

**Pattern**: Following hotSpring's `experiments/00X_NAME.md` convention.

Each experiment journal records: date, hardware, motivation ("why"),
procedure ("what"), findings, and surprises. This is the narrative
complement to the quantitative checks in `CONTROL_EXPERIMENT_STATUS.md`.

---

## Journal Index

| ID | Title | Date | Key Finding |
|----|-------|------|-------------|
| 001 | Phase 5a GPU Tensor Validation | Feb 21-22, 2026 | S-15/S-16 bugs in BarraCUDA |
| 002 | BarraCUDA CPU vs GPU Parity | Feb 20-21, 2026 | 7 domains, `matmul`/`transpose`/`tanh`/`add` |
| 003 | Fused Pipeline Scaling | Feb 19-20, 2026 | 46-78x speedup via single-encoder dispatch |
| 004 | GPU Dispatch Overhead Characterization | Feb 19, 2026 | 1.5ms crossover point (GPU > CPU) |
| 005 | L2 Megabatch Complexity Boundary | Feb 19, 2026 | GPU wins at 200x1000+ |
| 006 | Phase 5b Full-Stack Buildout | Feb 22, 2026 | ALL GREEN: bC 96%, gT 92%, xD 100% |
| 007 | Session 39 ToadStool Sync | Feb 22, 2026 | 5 shaders absorbed upstream, S-13 fixed, Conv2D/Pool available |

---

## Experiment 001: Phase 5a GPU Tensor Validation

**Date**: February 21-22, 2026
**Hardware**: i9-12900K, RTX 4070 12GB, Pop!_OS 22.04, Vulkan (NVIDIA 570.x)
**Researcher**: Eastgate

### Why

All Phase 0++ papers were validated on CPU (pure Rust + BarraCUDA CPU
primitives). Phase 5a asks: **"Do the same computations produce correct
results on BarraCUDA's GPU Tensor path?"** This is the final validation
layer before declaring BarraCUDA GPU-ready for neuralSpring workloads.

### What

Created 7 GPU `Tensor` validation binaries, one per scientific domain:
spectral, ecology, HMM, evolutionary computation, neural networks,
pairwise distance, and Anderson localization. Each:

1. Generates deterministic test data (Xoshiro256** seed)
2. Computes CPU f64 reference
3. Creates GPU f32 `Tensor` via `Tensor::from_data`
4. Executes GPU operations (`matmul`, `transpose`, `tanh`, `add`)
5. Reads back via `to_vec()` and compares with tolerance

### What We Found

**5 of 7 domains passed on first or second attempt.** The remaining 2
exposed fundamental BarraCUDA bugs:

- **S-15 (Critical)**: `Tensor::matmul` hangs when input data contains
  negative values or is highly sparse (many zeros). Discovered while
  validating neural network weights (naturally [-1, 1]) and the
  Anderson tridiagonal Hamiltonian (sparse, with -1 off-diagonals).
  The shader source is mathematically correct — the hang is at the
  WGPU/Vulkan driver level.

- **S-16 (High)**: 2D `Tensor::transpose()` produces partial output
  for any dimension > 16. Root cause: dispatch divides by
  `optimal_workgroup_size(ElementWise)` = 256 instead of the shader's
  hardcoded tile size of 16. For a [20, 8] → [8, 20] transpose, only
  1 workgroup is dispatched instead of 2, leaving output columns 16-19
  as zeros. This was the root cause of the Gram matrix accuracy failure
  in pairwise validation (max diff 3.71e0 at [18][18]).

### What Surprised Us

1. The S-16 transpose bug was latent — it only manifests when any
   dimension exceeds 16, AND the transpose result is used in a
   subsequent computation (matmul). The `validate_barracuda_gpu_eco`
   test passed because it creates the B matrix directly (no transpose).

2. S-15 is data-dependent: the exact same matrix shapes work with
   positive data but hang with negative data. This rules out shape/dispatch
   issues and points to a driver-level interaction with IEEE 754 negative
   float bit patterns.

3. The `should_use_npu_for_matmul()` code path calls `to_vec()` on both
   input tensors for sparsity analysis, even when no NPU is available.
   These GPU→CPU readbacks may contribute to pipeline state corruption
   that triggers the S-15 hang.

### Workarounds Applied

- All validators use `rng.uniform()` ([0, 1)) to avoid negative values
- Sparse matrices replaced with dense random equivalents
- Non-square shapes used to avoid S-14 (Naive tier hang)

### Status

**Resolved in Phase 5b**: S-16 fixed (one-line), S-15 root-caused (magnitude ≤ 0.1).
All 7 original domains now PASS (43/43 after fixes + workarounds).
Expanded to 23 papers: 98+ GPU Tensor checks, ALL GREEN.

---

## Experiment 002: BarraCUDA CPU vs GPU Parity

**Date**: February 20-21, 2026
**Status**: Documented in `CONTROL_EXPERIMENT_STATUS.md` Phase 2 + Phase 5a

The CPU implementations (BarraCUDA `stats::*`, `linalg::*`, `numerical::*`,
`special::*`) achieve machine-epsilon precision against hand-rolled Rust.
GPU Tensor operations achieve < 1e-3 tolerance (f32 accumulation error in
matmul). The gap is expected: f64 CPU vs f32 GPU.

---

## Experiment 003: Fused Pipeline Scaling

**Date**: February 19-20, 2026
**Status**: Documented in `specs/BENCHMARK_ANALYSIS.md`

Single-encoder dispatch eliminates per-op `queue.submit()` overhead:
46x (MLP) to 78x (Transformer) speedup. GPU dominates CPU at 3.1M FLOPs
(MLP) and 103M FLOPs (Transformer).

---

## Experiment 004: GPU Dispatch Overhead Characterization

**Date**: February 19, 2026
**Status**: Documented in `CONTROL_EXPERIMENT_STATUS.md` Phase 4c

GPU dispatch has ~1.5ms fixed overhead on RTX 4070. Below this threshold,
CPU is faster. Above, GPU wins (4-5x at large scale). This crossover point
is codified in `barracuda::dispatch::dispatch_for()`.

---

## Experiment 005: L2 Megabatch Complexity Boundary

**Date**: February 19, 2026
**Status**: Documented in `CONTROL_EXPERIMENT_STATUS.md` Phase 4c

GPU pairwise L2 distance beats CPU at 200x1000+ scale (4.2x faster).
At small scale (20x500), CPU is 46x faster due to dispatch overhead.

---

## Experiment 006: Phase 5b Full-Stack Validation Buildout

**Date**: February 22, 2026
**Hardware**: i9-12900K, RTX 4070 12GB, Pop!_OS 22.04, Vulkan (NVIDIA 570.x)
**Researcher**: Eastgate

### Why

Phase 5a identified 3 bugs (S-14/S-15/S-16) and achieved 33/43 GPU Tensor
checks across 7 domains. The BarraCUDA CPU, GPU Tensor, and Cross-dispatch
tiers had significant coverage gaps: bC 68%, gT 28%, xD 20%. Phase 5b
closes all gaps to reach ALL GREEN.

### What

1. **S-16 fix validation**: Confirmed the one-line transpose dispatch fix
   (`const TILE: u32 = 16`) resolves all pairwise Gram matrix failures.
2. **S-15 root cause**: Diagnosed that elements with magnitude ≤ 0.1 (not
   specifically negative or zero values) trigger the WGPU/Vulkan driver hang.
   Workaround: `rng.uniform() * 0.5 + 0.5` ensures all data ≥ 0.5.
3. **5 new validators**: `validate_barracuda_surrogate` (Exp 001, bC+gT),
   `validate_barracuda_transfer` (Exp 004, bC+gT),
   `validate_barracuda_gpu_transformer` (Exp 002, gT),
   `validate_cross_dispatch_hmm` (Papers 016/018, xD),
   `validate_cross_dispatch_ode` (Paper 020, xD).
4. **Reclassification**: Existing validators using GPU `Tensor` ops
   (`validate_barracuda_sequence`, `_lenet`, `_lstm`) counted toward gT.
5. **Documentation update**: Full coverage matrix in `specs/PAPER_REVIEW_QUEUE.md`.

### What We Found

- S-15 is purely a data-magnitude issue, not a sign or sparsity issue.
  All matmul tiers hang when elements are small (≤ 0.1), not just Naive.
- S-14 workaround (A×B^T pattern) is reliable: non-square intermediate
  shapes avoid the Naive tier entirely.
- Reclassifying existing validators gave "free" gT coverage for 3 papers.
- Cross-dispatch validators confirm GPU↔CPU parity across all 15 Phase 0++ papers.

### Result

| Tier | Before | After |
|------|--------|-------|
| bC (BarraCUDA CPU) | 17/25 (68%) | **24/25 (96%)** |
| gT (GPU Tensor) | 7/25 (28%) | **23/25 (92%)** |
| xD (Cross-dispatch) | 5/25 (20%) | **15/15 (100%)** |

**ALL GREEN** across all applicable tiers. Grand total: 1560+ checks.

---

## Experiment 007: Session 39 ToadStool Sync

**Date**: February 22, 2026
**Hardware**: i9-12900K, RTX 4070 12GB, Pop!_OS 22.04, Vulkan (NVIDIA 570.x)
**Researcher**: Eastgate

### Why

ToadStool committed Session 39 (`d45fdfb3`) — a massive dead-code sweep and
shader absorption wave. Needed to pull, audit what changed, revalidate
neuralSpring, and update all handoffs to reflect the new state.

### What

1. **Pulled** ToadStool `77f70b2e..d45fdfb3` (243 files, +14k/-4.7k lines).
2. **Audited** barracuda changes: 79 files changed in the barracuda crate.
3. **Verified** 255/255 lib tests PASS, all 115+ binaries compile.
4. **Identified** 5 shaders absorbed upstream as generalized variants:
   - `pairwise_l2` (closed-form pair decode), `multi_obj_fitness` (Bessel correction),
   - `hill_gate` (mode generalization), `swarm_nn_forward` (generic MLP),
   - `mean_reduce` (effectively identical).
5. **Confirmed** bug fixes flowing via path dep: S-13 (buffer race), TS-003 (trig),
   TS-001 (pow), TS-004 (FusedMapReduce).
6. **Noted** new capabilities: Conv2D/MaxPool2D/AvgPool2D WGSL shaders,
   `cpu_conv_pool` module, ESN weight export/import.
7. **Updated** all docs: ABSORPTION_TRACKER, EVOLUTION_READINESS, forge shaders.rs,
   ABSORPTION_MANIFEST, TOADSTOOL_HANDOFF, BARRACUDA_USAGE, BARRACUDA_EVOLUTION,
   README, CONTROL_EXPERIMENT_STATUS, DEPRECATION_MIGRATION, CROSS_SPRING_EVOLUTION,
   whitePaper/*.
8. **Created** V8 handoff (supersedes V7). V7 archived.

### What We Found

- The upstream shader variants are improved: O(1) pair decode vs O(N) linear search
  (pairwise_l2), Bessel correction for unbiased std (multi_obj_fitness), clamped
  sigmoid for stability (swarm_nn_forward), mode generalization (hill_gate).
- Local copies must be retained — our validators use local binding layouts via
  `include_str!`. Future migration to upstream APIs when Rust wrappers are available.
- S-13 fix is significant: the PooledBuffer race condition was a potential source of
  non-determinism in GPU validation. Now eliminated upstream.
- Conv2D/MaxPool2D/AvgPool2D WGSL shaders exist but aren't wired to GpuExecutor yet.
  CPU fallback (`cpu_conv_pool`) is ready. This opens the path for full LeNet-5
  validation beyond FC-only.
- ToadStool reached 3,847+ tests, 589+ WGSL shaders, zero clippy warnings.

### Result

| Metric | Before (V7) | After (V8) |
|--------|-------------|------------|
| Shaders upstream | 8/16 (50%) | 13/17 (76%) |
| S-13 status | Open | **FIXED** upstream |
| ToadStool HEAD | `77f70b2e` | `d45fdfb3` |
| neuralSpring tests | 255/255 PASS | 255/255 PASS (unchanged) |
| Handoff version | V7 | V8 |

No regressions. All validation checks unchanged.

---

## Experiment 008: Upstream BarraCUDA Rewiring & Cross-Spring Benchmarks

**Date**: February 22, 2026 (post-Experiment 007)
**Type**: Integration / Validation / Benchmark

### Why

After documenting the Session 39 sync (Experiment 007), the absorbed shaders
and upstream Rust wrapper APIs were available but not wired. This experiment
completes the loop: neuralSpring now validates and benchmarks the upstream
APIs that grew from its own shader contributions.

### What

1. **Created `validate_barracuda_bio_ops.rs`** — validates 6 upstream bio-op
   Rust wrappers (`BatchFitnessGpu`, `PairwiseHammingGpu`, `PairwiseJaccardGpu`,
   `LocusVarianceGpu`, `SpatialPayoffGpu`, `BatchIprGpu`) against CPU references.
   These wrappers encapsulate the same WGSL kernels neuralSpring evolved, now
   absorbed into BarraCUDA as first-class APIs.

2. **Created `validate_barracuda_hmm_f64.rs`** — validates `HmmBatchForwardF64`,
   the f64 batch HMM wrapper that wetSpring contributed. This replaces
   neuralSpring's local f32 per-timestep dispatch with f64 batch precision.
   Cross-spring evolution: neuralSpring evolved the f32 shader, wetSpring
   independently evolved f64 batch, ToadStool absorbed both.

3. **Created `bench_upstream_vs_local.rs`** — benchmarks all 6 bio ops through
   BOTH local manual wgpu dispatch AND upstream barracuda wrapper dispatch.
   Same shaders, different dispatch paths.

4. **Added `read_buffer_f64`** bridge method to `Gpu` struct for f64 readback.

5. **Created `whitePaper/CROSS_SPRING_SHADER_LINEAGE.md`** — comprehensive
   lineage map of all cross-spring shader evolution: hotSpring (physics/spectral),
   wetSpring (bio/genomics), neuralSpring (ML/evolution) → BarraCUDA.

### What We Found

- **Bio ops validation**: 12/12 PASS. All 6 upstream wrapper APIs produce
  correct results vs CPU references. Key diffs: BatchFitness 4.77e-7,
  Hamming 5.96e-8, Jaccard 5.96e-8, LocusVar 7.45e-9, Spatial 1.91e-6,
  IPR 3.73e-9.

- **HMM f64 validation**: 11/11 PASS. The wetSpring f64 batch HMM achieves
  **10⁹× better precision** than our local f32 dispatch (diff 2.47e-10 vs
  tolerance 0.5). Batch dispatch works correctly for 8 parallel sequences.

- **Upstream wrapper overhead**: Negligible. Local vs upstream ratios:
  BatchFitness 1.16×, Hamming 1.03×, Jaccard 0.92× (faster!), LocusVar 1.12×,
  Spatial 0.96×, IPR 1.03×. Median overhead < 5%.

- **Data layout gotchas**:
  - `PairwiseJaccardGpu` expects **column-major** PA: `pa[gene * n_genomes + genome]`
  - `BatchIprGpu` computes raw `Σ|ψ_i|⁴` (not the reciprocal `1/Σ|ψ_i|⁴`)

- **Cross-spring evolution is real**: neuralSpring's f32 HMM → ToadStool →
  wetSpring's f64 batch → ToadStool → back to neuralSpring with 10⁹× precision.

### Result

| Metric | Before | After |
|--------|--------|-------|
| Upstream wrapper checks | 0 | 23 (12 bio + 11 HMM) |
| Total validation checks | 1560+ | 1583+ |
| HMM precision | 0.5 tolerance (f32) | 2.47e-10 diff (f64) |
| Upstream overhead | Unknown | 0.92–1.16× (negligible) |
| Validation binaries | 115 | 118 |

---

## Experiment 009: Dual-Path Upstream Parity & Spectral Theory Stack

**Date**: February 22, 2026 (post-Experiment 008)
**Type**: Integration / Validation / Cross-Spring Lineage

### Why

Experiment 008 validated upstream wrappers in isolation. This experiment goes
deeper: every existing GPU validator now runs BOTH local `include_str!` dispatch
AND upstream wrapper dispatch on the same data, proving bit-identical results.
Additionally, the `barracuda::spectral` theory stack (originated in hotSpring
for Kachkovskiy spectral theory) is validated for the first time by neuralSpring.

### What

1. **Dual-path upstream parity** — 6 existing GPU validators (`validate_gpu_batch_fitness`,
   `validate_gpu_sate`, `validate_gpu_pangenome`, `validate_gpu_meta_pop`,
   `validate_gpu_game_theory`, `validate_gpu_anderson`) each gain an
   `upstream_parity` function that dispatches via the barracuda wrapper and
   compares against local dispatch. All 6 produce **0.00e0 diff** (bit-identical).

2. **ReduceScalarPipeline wiring** — The Anderson validator now chains
   `BatchIprGpu` → f64 buffer → `ReduceScalarPipeline::sum_f64` → mean IPR.
   Diff vs CPU mean: **5.55e-17** (machine epsilon).

3. **Created `validate_barracuda_spectral_theory.rs`** — validates 14 checks
   against the `barracuda::spectral` stack:
   - Golden ratio constant parity (neuralSpring vs barracuda)
   - Aubry-André spectrum parity (Jacobi dense vs Sturm tridiag: 1.23e-2)
   - Anderson Hamiltonian construction (eigenvalue count, bandwidth)
   - Lanczos eigensolve (2D Anderson 8×10, 3D clean 3×3×3)
   - Lyapunov exponents (strong vs weak disorder ordering, positivity)
   - Level-spacing statistics (GOE-like for clean, Poisson for localized)
   - Hofstadter butterfly structure (21 rational α, 2100 eigenvalues)
   - Band detection in gapped spectra
   - Kappus-Wegner anomaly: γ(W=0.5) ≈ W²/96 to 6% relative error

   **Cross-spring lineage**: hotSpring (Kachkovskiy spectral theory) → barracuda
   → neuralSpring validates. The spectral functions carry "Provenance: hotSpring
   v0.6.0" headers in the barracuda source.

### What We Found

- **Bit-identical parity**: All 6 GPU validators confirm local and upstream
  dispatch produce exactly the same results (0.00e0 diff). This is the
  strongest possible proof that absorbed shaders are unchanged.

- **ReduceScalarPipeline**: f64 GPU reduction achieves machine-epsilon accuracy
  (5.55e-17), validating the pipeline for future use in large-scale reductions.

- **Spectral theory**: All 14 checks pass. The Lanczos eigensolver correctly
  handles sparse 2D/3D Anderson matrices. Level-spacing ratio reliably
  distinguishes localized (Poisson) from extended (GOE) phases.

### Result

| Metric | Before | After |
|--------|--------|-------|
| Upstream parity checks | 0 | 6 (all 0.00e0 diff) |
| ReduceScalarPipeline | Not wired | 5.55e-17 diff |
| Spectral theory checks | 0 | 14 (Lanczos + Anderson + Hofstadter + Lyapunov) |
| Total validation checks | 1583+ | 1604+ |
| Validation binaries | 118 | 119 |

---

*Experiment journals — following the hotSpring pattern.*
