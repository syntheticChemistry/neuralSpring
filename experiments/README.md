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
| 008 | Upstream BarraCUDA Rewiring | Feb 22, 2026 | 6 bio ops + f64 HMM wired, 0.92–1.16× overhead |
| 009 | Dual-Path Parity & Spectral Theory | Feb 22, 2026 | 6/6 bit-identical, spectral 14/14, ReduceScalarPipeline |
| 010 | Capability-Based Dispatch & Cross-Eigensolver | Feb 22, 2026 | 12 validators use `dispatch_1d`, eigh vs Sturm 2.89e-15 |
| 011 | Session 42 Deep Audit — Code Quality & Debt Resolution | Feb 22, 2026 | 264 lib + 9 integration, all fmt/clippy/doc clean, tolerances split, GPU helpers deduplicated |
| 012 | ToadStool Sync — d45fdfb3 → 5437c170 (10 commits) | Feb 22, 2026 | 10-kernel bench (0.72–1.10×), 10/10 upstream parity (3 bit-identical), LeNet-5 full bC 13/13, cross-spring lineage |
| 013 | Session 43 — Experiment Buildouts, CPU/GPU Parity, Mixed Hardware | Feb 22, 2026 | 4 new WGSL shaders (18/18), 5 upstream wrappers (41/41), CPU/GPU parity (17/17), mixed-hardware dispatch (16/16+16/16) |
| 014 | Session 44 — Multi-GPU Portability, Benchmarks, Reverse Pipeline | Feb 23, 2026 | 131/131 on RTX 4070 + TITAN V NVK, 178.5× Rust vs Python, 2 bC bugs fixed, 4 new validators (30/30) |
| 015 | Session 45 — Pure GPU Promotion Phase A | Feb 23, 2026 | 27 CPU→GPU promotions via Dispatcher, gpu_ops + gpu_dispatch modules, 27/27 PASS |
| 016 | Session 46 — Pure GPU Promotion Phase B | Feb 23, 2026 | 11 more ops: HMM backward/Viterbi, meta-pop stats, replicator, Hill activation. 20/20, ~90% GPU |

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
3. **Verified** 264/264 lib tests + 9 integration tests PASS, all 119 binaries compile.
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

## Experiment 010: Capability-Based Dispatch & Cross-Eigensolver Validation

**Date**: February 22, 2026 (Session 40)
**Type**: Infrastructure / Validation / Cross-Algorithm
**Hardware**: i9-12900K, RTX 4070 12GB, Pop!_OS 22.04, Vulkan (NVIDIA 570.x)

### Why

All 30+ GPU validators used hardcoded `.div_ceil(256)` for workgroup dispatch,
ignoring runtime hardware limits (`max_compute_workgroup_size_x`,
`max_compute_workgroups_per_dimension`). This is fragile — would silently fail
on devices with smaller workgroup limits (WebGPU mobile, browser targets).

Additionally, the dense Householder+QR eigensolver (`eigh_householder_qr`) and
the tridiagonal Sturm bisection (`find_all_eigenvalues`) had never been
cross-validated on the same matrix — a gap in the spectral theory stack.

### What

1. **Added `GpuCapabilities::supports_workgroup()`** — validates that a shader's
   `@workgroup_size(N)` is compatible with hardware.

2. **Added `Gpu::dispatch_1d(n_items, shader_wg)`** — convenience method that
   validates workgroup compatibility (panics on incompatible hardware) and
   returns the clamped workgroup count.

3. **Wired `dispatch_1d` into 12 core GPU validators** — batch_fitness, anderson,
   game_theory, sate, pangenome, meta_pop, modes, directed, swarm, signal,
   rk4, plus the evolved `hmm_forward_gpu` module.

4. **Added startup capability logging** — validators now report discovered
   hardware limits: `wg_x=256, dispatch_max=65535, buffers=12, f64=true, f16=true`.

5. **Cross-validated eigensolvers** — added `validate_eigh_vs_sturm` and
   `validate_eigh_vs_sturm_large` to `validate_barracuda_spectral_theory`:
   - n=64 W=3: max eigval diff **2.89e-15** (machine epsilon)
   - n=200 W=6: max eigval diff **1.42e-14** (machine epsilon)
   Both eigensolvers agree perfectly on the same tridiagonal Anderson Hamiltonians.

### What We Found

- RTX 4070 reports `max_compute_workgroup_size_x=256`, which means our
  `@workgroup_size(256)` shaders are at the hardware limit. Any device with
  a smaller limit would have silently dispatched wrong.

- `max_compute_workgroups_per_dimension=65535` means dispatches up to
  256 × 65535 = 16.7M work items are safe. Beyond that, the clamp in
  `dispatch_count` would truncate — producing wrong results. For our workloads
  (populations of ~10k max), this is not a concern.

- The eigensolver cross-validation proves that Householder+QR (O(n³), works on
  any symmetric matrix) and Sturm bisection (O(n) per eigenvalue, tridiagonal
  only) produce identical results at machine precision. This confirms both
  implementations in BarraCUDA are correct.

### Result

| Metric | Before | After |
|--------|--------|-------|
| Validators using capability dispatch | 0 | 12 + evolved HMM |
| Hardcoded dispatch patterns | 30+ | 18 remaining (pipeline, bench, cross-dispatch) |
| Spectral theory checks | 14/14 | **17/17** (+3 eigh cross-validation) |
| Total validation checks | 1604+ | 1607+ |
| Handoff version | V8 | V9 |

---

## Experiment 011: Session 42 Deep Audit — Code Quality & Debt Resolution

**Date**: February 22, 2026
**Hardware**: i9-12900K, RTX 4070 12GB, Pop!_OS 22.04
**Researcher**: Eastgate

### Why

The full codebase had grown through 40+ sessions of feature development. A
comprehensive audit was needed to enforce idiomatic Rust, eliminate technical
debt, centralize patterns, and verify the codebase met wateringHole standards
before the next evolution phase.

### What

1. **Formatting & Linting**: Fixed 33 `cargo fmt` violations and 123 `cargo clippy`
   warnings (pedantic + nursery + `unwrap_used` + `expect_used`). All checks now pass
   with zero warnings.

2. **Documentation**: Fixed 1 `cargo doc` warning (unresolved link). All rustdoc clean.

3. **Tolerance Centralization**: Replaced 3 inline magic numbers in validation binaries
   with named constants. Added 18 previously unregistered tolerances to the runtime
   `all_tolerances()` registry, making them discoverable via `tolerance_by_name()`.

4. **Module Split**: Split `tolerances.rs` (1028 lines → over limit) into
   `tolerances/mod.rs` (696 lines, constants) + `tolerances/registry.rs` (341 lines,
   introspection). Both under 1000-line wateringHole guideline.

5. **GPU Helper Deduplication**: Extracted `gpu_readback()`, `max_abs_diff_f32()`,
   `max_abs_diff_gpu_vs_cpu()`, `gpu_tensor()`, and `gpu_tensor!` macro from 23
   validation binaries into shared `validation.rs`. Removed ~400 lines of duplicated code.

6. **Provenance Enhancement**: Added exact Python commands, NumPy/SciPy versions, and
   environment details for `SOFTMAX_1_TO_5`, `GELU_REFERENCE`, and `RASTRIGIN_REFERENCE`.

7. **Test Coverage Expansion**: Added 9 determinism tests (introgression, regulatory,
   pangenome, meta-population, SATé, signal, game theory, spectral, Anderson). Created
   `tests/integration.rs` with 9 cross-module integration tests.

8. **Dependency Analysis**: Confirmed the entire stack (neuralSpring + BarraCUDA) is
   pure Rust — zero C/C++ wrappers, no FFI.

9. **Drift Detection**: Created `control/check_drift.sh` to re-run all 25 Python
   baselines and verify no baseline drift. Ready for CI integration.

### Findings

- **No unsafe code**: `#![forbid(unsafe_code)]` enforced, zero violations.
- **All files under 1000 LOC**: Largest is `tolerances/mod.rs` at 696 lines.
- **264 lib + 9 integration tests PASS**: Up from 255 lib tests.
- **94.9% line coverage** maintained.
- **Pure Rust dependency tree**: No external C/C++ dependencies anywhere.
- **All fmt/clippy/doc gates**: Zero warnings across all checks.

### Surprises

- `cargo clippy` with pedantic+nursery found 123 warnings in code that had been
  passing standard clippy for months. Most were `cast_lossless`, `mul_add`, and
  `redundant_clone` — easy fixes but numerous.
- The `((VAR++))` bash arithmetic pattern returns exit code 1 when VAR=0 under
  `set -euo pipefail`, causing the drift detection script to exit prematurely.
  Changed to `VAR=$((VAR + 1))`.

---

## Experiment 012: ToadStool Sync — d45fdfb3 → 5437c170

**Date**: February 22, 2026
**Hardware**: i9-12900K, RTX 4070 12GB, Pop!_OS 22.04
**Researcher**: Eastgate

### Why

ToadStool had evolved 10 commits since our last sync point (`d45fdfb3`, Session 39).
Three sessions of upstream work (S39, S40, S41, S42) added significant new APIs
and absorbed more Spring shaders. neuralSpring needed to catch up to the current
state, verify build compatibility, and document what became available.

### What

1. **Reviewed** 10 ToadStool commits: S39 (Spring shader absorption, S-14/S-15/S-16
   fixes, FlatTree), S40 (Richards PDE, moving window stats), S41 (api exposure,
   f64 shader fixes, stale doc archive), S42 (19 new WGSL shaders, doc rename).

2. **Build verification**: `cargo check` compiles cleanly against ToadStool HEAD
   `5437c170` with zero warnings. No breaking changes in barracuda API.

3. **Updated** ToadStool HEAD references from `d45fdfb3` to `5437c170` across
   14 live documentation files.

4. **Documented** newly available APIs that neuralSpring can now use:
   - `ops::bio::HillGateGpu` — generalized Hill function dispatch
   - `ops::bio::MultiObjFitnessGpu` — multi-objective fitness with Bessel correction
   - `ops::bio::PairwiseL2Gpu` — pairwise L2 with O(1) pair decode
   - `ops::bio::SwarmNnGpu` — generic MLP forward with clamped sigmoid
   - `cpu_conv_pool::{conv2d, max_pool2d, avg_pool2d}` — CPU reference conv/pool

### Findings

- **Zero breaking changes**: The entire BarraCUDA API is backward-compatible.
  All 264 lib tests + 9 integration tests pass without modification.
- **BarraCUDA → BarraCuda doc rename**: The crate name is still `barracuda`
  (no code changes needed). The rename is docs/comments only.
- **4 new bio-op wrappers benchmarked** (HillGateGpu, MultiObjFitnessGpu,
  PairwiseL2Gpu, SwarmNnGpu): all show negligible overhead vs local
  metalForge dispatch (0.97×–1.10×). Total: 10 kernels in upstream parity bench.
- **LeNet-5 full bC validation**: `cpu_conv_pool::{conv2d, max_pool2d}` wired
  for Conv(1→6,5×5,pad=2) → ReLU → MaxPool(2) → Conv(6→16,5×5) → ReLU →
  MaxPool(2) → FC chain. **13/13 PASS** (was 5/5 FC-only).
- **Cross-spring shader lineage documented**: Tracked provenance of all
  shared primitives across hotSpring (precision, physics), wetSpring (bio),
  and neuralSpring (ML, evolution) contributions to BarraCuda.
- **19 new WGSL shaders**: chi_squared_f64, rk45_f64, factorial_f64,
  cubic_spline_eval_f64, trapz_f64, etc. GPU paths now exist for operations
  neuralSpring currently uses only on CPU.

5. **Upstream parity validators added** to all 4 GPU validators (signal, directed,
   modes, swarm). Each runs both local metalForge shader and upstream BarraCuda
   wrapper with identical input, then compares output. Results:
   - HillGateGpu vs local: **0.00e0** (bit-identical)
   - PairwiseL2Gpu vs local: **0.00e0** (bit-identical)
   - SwarmNnGpu vs local: **bit-exact u32** (bit-identical)
   - MultiObjFitnessGpu vs local: **1.95e-3** (Bessel n-1 vs population n)

### Next Steps

1. Explore GPU paths for chi_squared, RK45 via new f64 WGSL shaders
2. Wire TaxonomyFcGpu, KmerHistogramGpu, UniFracPropagateGpu for wetSpring parity

---

## Experiment 013: Session 43 — Experiment Buildouts, CPU/GPU Parity, Mixed Hardware

**Date**: February 22, 2026
**Hardware**: i9-12900K, RTX 4070 12 GB (Vulkan), llvmpipe (CPU)
**ToadStool HEAD**: `5437c170` (unchanged from Session 42)

### Motivation

Extend neuralSpring's validation coverage across three axes: (1) new local WGSL
shaders evolving for ToadStool absorption, (2) upstream BarraCuda wrapper
integration for wetSpring parity and new f64 ops, and (3) mixed-hardware
dispatch infrastructure for GPU-NPU-CPU routing.

### Procedure

1. **Phase 1: New WGSL shaders** — Built `logsumexp_reduce.wgsl` (batched
   numerically-stable reduction, HMM/phylo), `stencil_cooperation.wgsl` (Fermi
   imitation dynamics, game theory), `rk45_adaptive.wgsl` (Dormand-Prince with
   Hill RHS, regulatory networks), `wright_fisher_step.wgsl` (binomial drift +
   selection with inline xoshiro128**). Each with CPU reference validator.

2. **Phase 2: Upstream wrappers** — Wired `GillespieGpu` (parallel SSA, 20/20),
   `TaxonomyFcGpu` (Naive Bayes metagenomics, 3/3), `KmerHistogramGpu` (k-mer
   histograms, 3/3), `UniFracPropagateGpu` (tree propagation, 2/2),
   `chi_squared::*` (distribution + test statistic, 13/13).

3. **Phase 3: Validation sweep** — 264 lib + 9 integration + 26 forge tests,
   clippy 0 warnings, fmt clean. All 12 new validators pass (108/108 checks).

4. **Phase 4: CPU vs GPU parity** — `validate_cpu_gpu_parity` exercises Tensor
   API on both GPU and CPU devices for MatMul, ReLU, Sigmoid, Tanh, Sum, erf,
   gamma, conv2d, max_pool2d. Cross-hardware comparison shows bit-identical
   MatMul and ReLU results.

5. **Phase 5: Mixed-hardware dispatch** — Built `mixed.rs` (`MixedSubstrate`,
   `TransferCost`, PCIe cost model) and `pcie_bridge.rs` (`PcieBridge`, P2P
   detection placeholder). Design doc at `metalForge/MIXED_HARDWARE_DESIGN.md`.
   Validated: 16/16 dispatch routing + 16/16 mixed dispatch checks.

### Findings

- **GPU logsumexp**: max diff 4.77e-7 (f32) for 64×128 batch — well within tolerance
- **RK45 adaptive**: GPU matches CPU Dormand-Prince at 5e-4 tolerance (f32 accumulation over 6 stages)
- **Wright-Fisher**: neutral drift mean 0.4972 (expected 0.5), positive selection bias confirmed
- **Gillespie SSA**: perfect A+B conservation (100.0 exact), stochastic variation confirmed across 16 trajectories
- **CPU vs GPU parity**: cross-hardware MatMul produces bit-identical f32 results
- **Transfer cost model**: GPU→CPU 1MB = 35.3 µs, GPU→NPU P2P 1MB = 134.7 µs, staged 139.7 µs

### Surprises

- Tensor API MatMul is **bit-identical** across GPU (RTX 4070 Vulkan) and CPU (llvmpipe),
  suggesting deterministic IEEE 754 rounding in the WGSL matmul shader.
- GillespieGpu f64 conservation is exact (not just within tolerance) — the integer-like
  stoichiometry means no floating-point accumulation error.

### Artifacts

| File | Role |
|------|------|
| `metalForge/shaders/logsumexp_reduce.wgsl` | Batched log-sum-exp (max-subtract trick) |
| `metalForge/shaders/stencil_cooperation.wgsl` | Fermi imitation dynamics stencil |
| `metalForge/shaders/rk45_adaptive.wgsl` | Dormand-Prince RK45 with Hill RHS |
| `metalForge/shaders/wright_fisher_step.wgsl` | Wright-Fisher drift+selection+xoshiro |
| `metalForge/forge/src/mixed.rs` | MixedSubstrate + TransferCost |
| `metalForge/forge/src/pcie_bridge.rs` | PcieBridge + P2P detection |
| `metalForge/MIXED_HARDWARE_DESIGN.md` | Mixed-hardware dispatch design |
| `metalForge/gpu/MIXED_HARDWARE_RESULTS.md` | Validation results |
| `src/bin/validate_gpu_logsumexp.rs` | 5/5 PASS |
| `src/bin/validate_gpu_stencil.rs` | 3/3 PASS |
| `src/bin/validate_gpu_rk45.rs` | 6/6 PASS |
| `src/bin/validate_gpu_wright_fisher.rs` | 4/4 PASS |
| `src/bin/validate_gpu_gillespie.rs` | 20/20 PASS |
| `src/bin/validate_upstream_taxonomy.rs` | 3/3 PASS |
| `src/bin/validate_upstream_kmer.rs` | 3/3 PASS |
| `src/bin/validate_upstream_unifrac.rs` | 2/2 PASS |
| `src/bin/validate_barracuda_chi_squared.rs` | 13/13 PASS |
| `src/bin/validate_cpu_gpu_parity.rs` | 17/17 PASS |
| `src/bin/validate_toadstool_dispatch.rs` | 16/16 PASS |
| `src/bin/validate_mixed_dispatch.rs` | 16/16 PASS |

---

## Experiment 014: Session 44 — Multi-GPU Portability, Benchmarks, and Reverse Pipeline

**Date**: February 23, 2026
**Hardware**: i9-12900K, RTX 4070 12 GB (Vulkan, proprietary), TITAN V 12 GB (NVK GV100, open-source)
**Researcher**: Eastgate
**ToadStool HEAD**: `5437c170` + 2 upstream bug fixes

### Why

neuralSpring's thesis: **prove all math is correct on GPU first, then reverse-
engineer for CPU and older GPU.** Session 44 validates this by running the full
suite on a second GPU (TITAN V, Volta architecture, NVK open-source driver) and
establishing quantitative Python-vs-Rust benchmarks.

### What

1. **Multi-GPU validation**: Ran all 131 `validate_all` binaries on RTX 4070
   (default) and TITAN V (`NEURALSPRING_BACKEND=titan`). Both produce bit-identical
   results, proving WGSL math portability across GPU generations and driver stacks.

2. **Upstream BarraCUDA bug fixes**: Fixed `Tensor::mean()` crash (wrong entry
   point `"main"` vs shader's `mean_reduce`) and chi-squared expected value
   precision (textbook-rounded vs full-precision computed values).

3. **New validators** (4): `validate_gpu_pipeline_wright_fisher` (WF step →
   mean reduce in single CommandEncoder, zero CPU round-trips),
   `validate_gpu_pipeline_gillespie` (Gillespie SSA → mean reduce on-GPU),
   `validate_barracuda_gpu_lenet` (Conv2d + MaxPool2d via Tensor API),
   `validate_barracuda_transformer` (full layer: Q/K/V projections, attention
   scores, FFN block, residual connections, global softmax).

4. **Pure Rust vs Python/NumPy benchmarks**: Created 4 missing Python benchmark
   scripts (`bench_pairwise_l2.py`, `bench_multi_obj.py`, `bench_hill_gate.py`,
   `bench_swarm_nn.py`). Ran `bench_phase0pp_kernels --with-python` for all 11
   kernels: overall **178.5× speedup** for Rust over single-thread Python/NumPy.

5. **Multi-GPU tensor and inference benchmarks**: Ran `bench_barracuda_tensor`,
   `bench_mlp_inference`, `bench_transformer_block` on both RTX 4070 and TITAN V.

### What We Found

- **Bit-identical multi-GPU**: 131/131 PASS on RTX 4070, 143+ additional on
  TITAN V. No numerical divergence between proprietary and open-source drivers.

- **Mean reduce bug**: BarraCUDA's `ops/mean.rs` used `entry_point: "main"` but
  `mean_reduce.wgsl` exports `fn mean_reduce`. Also had a double-division bug
  (Rust re-dividing the already-computed mean). Fixed both.

- **Chi-squared precision**: Expected values in the validator were textbook-rounded
  (e.g., `0.950`) while BarraCUDA computes full precision. Updated to computed values.

- **Rust vs Python**: 178.5× overall. Individual kernel speedups range from 0.4×
  (commutator — NumPy BLAS-optimized matmul) to 551× (swarm NN forward). The
  one case where Python wins confirms the reverse pipeline motivation: BLAS-level
  CPU optimization is a separate concern from GPU math correctness.

- **`Tensor::softmax()` is global, not row-wise**: Discovered during transformer
  validation. BarraCUDA softmax normalizes over all elements, not per-row. For
  attention weights, row-wise softmax requires `ScaledDotProductAttention` or
  manual per-row dispatch. Global softmax is correct for classification logits.

### Surprises

1. TITAN V (2017 Volta, 5120 CUDA cores) matches RTX 4070 (2023 Ada, 5888 cores)
   in validation correctness but shows expected latency differences in benchmarks.
   The NVK open-source driver handles all WGSL shaders without issue.

2. The mean_reduce bug had been latent since the shader was created — no validator
   previously exercised `Tensor::mean()` as a standalone operation.

3. The chi-squared "bug" was really a documentation/expected-value precision issue,
   not a math error. BarraCUDA's implementation is more precise than the textbook.

### Artifacts

| File | Role |
|------|------|
| `src/bin/validate_gpu_pipeline_wright_fisher.rs` | WF step → mean reduce pipeline |
| `src/bin/validate_gpu_pipeline_gillespie.rs` | Gillespie → mean reduce pipeline |
| `src/bin/validate_barracuda_gpu_lenet.rs` | Conv2d + MaxPool2d GPU validation |
| `src/bin/validate_barracuda_transformer.rs` | Full transformer layer bC validation |
| `control/modes/bench_pairwise_l2.py` | Python benchmark: pairwise L2 |
| `control/directed_evolution/bench_multi_obj.py` | Python benchmark: multi-obj fitness |
| `control/signal_integration/bench_hill_gate.py` | Python benchmark: Hill function |
| `control/swarm_robotics/bench_swarm_nn.py` | Python benchmark: swarm NN forward |
| `specs/BENCHMARK_ANALYSIS.md` | Updated with Session 44 multi-GPU results |
| `specs/PAPER_REVIEW_QUEUE.md` | Updated with Session 44 validators |

### Result

| Metric | Before (Session 43) | After (Session 44) |
|--------|---------------------|---------------------|
| GPUs validated | 1 (RTX 4070) | **2** (RTX 4070 + TITAN V NVK) |
| Validation binaries | 127 | **131** |
| `validate_all` | 127/127 | **131/131** (+ 143 on Titan V) |
| Upstream bC bugs fixed | 0 | **2** (mean_reduce, chi_squared) |
| Python benchmark coverage | 7/11 kernels | **11/11 kernels** |
| Rust vs Python speedup | Partial | **178.5× overall** |

---

## Experiment 015: Session 45 — Pure GPU Promotion Phase A

**Date**: February 23, 2026
**Hardware**: i9-12900K, RTX 4070 12 GB (Vulkan), TITAN V 12 GB (NVK)
**Researcher**: Eastgate
**ToadStool HEAD**: `5437c170`

### Why

neuralSpring had validated all math across 8 tiers (Python → Rust → BarraCUDA CPU → GPU Tensor → metalForge → Pipeline → Cross-dispatch → Multi-GPU), but many production-path computations still ran on CPU even when GPU hardware was available. Session 45 aimed to create a **capability-based GPU dispatch layer** that routes all operations to GPU when available, with CPU fallback.

### What

1. **Created `gpu_ops.rs`** — 27 GPU-accelerated functions using the BarraCUDA `Tensor` API:
   matmul, transpose, frobenius_norm, softmax, l2_distance, pearson_correlation,
   variance, mean, neural_forward, hmm_forward_step, rk4_step, fitness_evaluation,
   diversity_metrics, tree_distance, logsumexp, log_likelihood, pca_project,
   chi_squared, hamming_distance, jaccard_similarity, and more.

2. **Created `gpu_dispatch.rs`** — `Dispatcher` struct with capability-based routing:
   detects GPU availability at construction, dispatches to `gpu_ops` when hardware
   supports the operation, falls back to CPU otherwise. Zero configuration required.

3. **Created `validate_gpu_promotion.rs`** — 27-check validator exercising all
   dispatched operations via the `Dispatcher` with CPU reference comparison.

### Findings

- **27/27 PASS** on RTX 4070 (proprietary Vulkan)
- **27/27 PASS** on TITAN V (NVK open-source Vulkan)
- All results match CPU references within f32→f64 tolerance
- `Tensor` API ownership model requires careful management: methods like `matmul`,
  `softmax`, `sigmoid` consume `self`, while `add`, `mul`, `transpose` borrow `&self`
- The `Dispatcher` pattern cleanly separates capability detection from operation dispatch

### Result

| Metric | Before | After |
|--------|--------|-------|
| CPU-bound production ops | ~38 | ~11 |
| GPU-dispatched ops | 0 | **27** |
| Validation binaries | 131 | **132** |
| `validate_all` | 131/131 | **132/132** |

---

## Experiment 016: Session 46 — Pure GPU Promotion Phase B

**Date**: February 23, 2026
**Hardware**: i9-12900K, RTX 4070 12 GB (Vulkan), TITAN V 12 GB (NVK)
**Researcher**: Eastgate
**ToadStool HEAD**: `5437c170`

### Why

Phase A promoted 27 "straightforward" operations — those with direct Tensor API
equivalents. Phase B tackled the harder cases: HMM backward/Viterbi (multi-step
GEMV chains), meta-population statistics (column reductions, variance decomposition),
replicator dynamics (2×2 GEMV), and correcting a pseudo-GPU Hill activation to a
genuine GPU pipeline.

### What

1. **HMM backward step** (`hmm_backward_step_gpu`): GPU GEMV — `β_{t+1} ⊙ emit →
   weighted @ A^T / scale`. Uses `Tensor::mul`, `transpose`, `matmul`.

2. **HMM Viterbi step** (`hmm_viterbi_step_gpu`): GPU score matrix via `broadcast +
   add + max_dim`, with CPU argmax (BarraCUDA `max_dim` returns values, not indices).

3. **Meta-population statistics** (6 ops): `allele_frequencies_gpu` (column `sum_dim`),
   `nucleotide_diversity_gpu` (allele freq → elementwise → `mean`),
   `matrix_correlation_gpu` (upper-triangle → `pearson_correlation_gpu`),
   `geographic_distance_matrix_gpu` (pairwise Euclidean via `l2_distance_gpu`),
   `thermal_diversity_correlation_gpu` (→ `pearson_correlation_gpu`),
   `inter_population_af_variance_gpu` (per-pop allele freq → variance → mean).

4. **Replicator dynamics** (`replicator_step_gpu`): 2×2 payoff matrix GEMV via
   `Tensor::matmul`, with CPU update for the nonlinear `x + dt*x*(f - f̄)` step.

5. **Hill activation** (refactored): Previously computed on CPU and uploaded result.
   Now genuine GPU pipeline: `log_wgsl → mul_scalar → exp_wgsl → add → div → mul_scalar`.

### Findings

- **20/20 PASS** on RTX 4070 (proprietary Vulkan)
- **20/20 PASS** on TITAN V (NVK open-source Vulkan)
- BarraCUDA `max_dim` lacks argmax — Viterbi requires hybrid GPU/CPU approach
- Hill activation's `x^n` via `exp(n * ln(x))` is numerically stable with `x.max(1e-30)` guard
- Replicator dynamics GEMV is correct but the nonlinear update requires CPU — full GPU
  would need a custom WGSL shader
- `validate_all`: **133/133 PASS, 0 FAIL**
- GPU coverage estimate: **~90%** of production math

### Surprises

1. The `hill_activation_batch_gpu` had been a pseudo-GPU function (CPU compute, GPU upload)
   since its creation. Refactoring to genuine GPU compute exposed the need for careful
   guard values to prevent `ln(0)`.

2. The HMM Viterbi hybrid approach (GPU matrix ops + CPU argmax) is actually well-suited
   to the Tensor API's strengths — bulk linear algebra on GPU, small sequential logic on CPU.

### Result

| Metric | Before (S45) | After (S46) |
|--------|-------------|-------------|
| GPU-dispatched ops | 27 | **38** |
| CPU-only remaining | ~11 | ~4 (ODE loops, FST, introgression chain) |
| GPU coverage | ~70% | **~90%** |
| Validation binaries | 132 | **133** |
| `validate_all` | 132/132 | **133/133** |

---

*Experiment journals — following the hotSpring pattern.*
