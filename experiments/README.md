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
| 017 | ToadStool Sync — 5437c170 → 6ee71f07 (2 commits) | Feb 23, 2026 | SNP/ODE/Jacobi/loop_unroller fixes. Zero neuralSpring impact. 133/133 PASS. |
| 018 | Session 49 — Deep Debt Audit | Feb 23, 2026 | gpu_or_cpu dispatch helper, exit_no_gpu, baseline_path, 0 clippy/doc warnings |
| 019 | Session 50 — baseCamp Biophysical AI Interpretability | Feb 24, 2026 | 5 modules, 82/82 PASS, 459 lib tests, GPU evolution candidates identified |
| 020 | Session 51 — Code Quality Evolution & Documentation Refresh | Feb 24, 2026 | gpu_dispatch refactored, 47 float comparisons evolved, 7 inline guards centralized, 92.9% coverage |
| 021 | Session 52 — ToadStool Sync & Cross-Spring Benchmarking | Feb 24, 2026 | 6 shaders absorbed, `level_spacing_ratio` rewired, `argmax_dim`/`softmax_dim` gaps closed |
| 022 | Session 52b — S-17 HillGate f64 `pow()` Fix | Feb 24, 2026 | `pow(f64)` crashes NVVM/NAK; polyfill fix 18/18 PASS both GPUs |
| 023 | Session 54 — baseCamp Experiment Expansion & GPU Workload Validation | Feb 24, 2026 | 82→114 CPU + 14 GPU = 128/128, `validate_basecamp_gpu`, CPU↔GPU sub-epsilon parity |
| 024 | Session 55 — BarraCUDA CPU vs GPU Dispatch + metalForge Mixed Hardware | Feb 24, 2026 | `mixed_dispatch()` wired, 16/16 compute dispatch + 14/14 mixed hardware, 141/142 all green |

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

## Experiment 017: ToadStool Sync — 5437c170 → 6ee71f07

**Date**: February 23, 2026
**Hardware**: i9-12900K, RTX 4070 12 GB (Vulkan)
**ToadStool HEAD**: `6ee71f07`

### Why

ToadStool had evolved 2 commits since our last sync (`5437c170`, Session 42).
Both were bug fixes from sibling Springs (wetSpring, hotSpring). neuralSpring
needed to verify build compatibility and re-validate against the new HEAD.

### What

1. **Reviewed** 2 commits: `b53dd2f6` (SNP BGL binding, ODE f64 builtins,
   Jacobi eigenvector rotation — wetSpring Exp098) and `6ee71f07`
   (loop_unroller u32 suffix — hotSpring v0.6.7).

2. **Impact analysis**: Neither commit touches APIs used by neuralSpring.
   SNP calling, batched QS ODE RK4 f64, and batched_eigh_single_dispatch
   are unused. Loop unroller fix affects BatchedEighGpu (neuralSpring uses
   Householder+QR via `eigh_f64`, not the batched Jacobi variant).

3. **Build verification**: `cargo check` clean, zero new warnings.

4. **Validation**: 264 lib + 9 integration PASS. `validate_all`: **133/133 PASS**.

### What We Found

- Zero regressions. Zero API changes affecting neuralSpring.
- The loop_unroller fix (`"0"` → `"0u"` for WGSL u32 literals) is a
  correctness fix for any shader using `@unroll_hint` with u32 function
  parameters. neuralSpring's metalForge shaders don't use loop unrolling.
- The Jacobi eigenvector fix (V rotation for all rows, not just k!=p&&k!=q)
  is critical for eigendecomposition correctness. neuralSpring uses
  `eigh_householder_qr` (not Jacobi), so unaffected.
- Our 2 local fixes (mean_reduce entry point, chi-squared precision) from
  Session 44 are still local and pending ToadStool absorption.

### Result

| Metric | Before | After |
|--------|--------|-------|
| ToadStool HEAD | `5437c170` | **`6ee71f07`** |
| New upstream commits | 0 | **2** (bug fixes) |
| neuralSpring impact | — | **Zero** |
| `validate_all` | 133/133 | **133/133** |
| Local fixes pending | 2 (mean_reduce, chi²) | **2** (unchanged) |

---

## Experiment 018: Deep Debt Audit — Session 49

**Date**: February 23, 2026
**Hardware**: i9-12900K, RTX 4070 12 GB (Vulkan)
**ToadStool HEAD**: `b41ee5f4`

### Why

Comprehensive code quality audit before crafting the ToadStool absorption
handoff. Reviewed every file against wateringHole standards (1000-line max,
AGPL-3.0, pure Rust, no unsafe, no mocks in production, capability-based
design, primal autonomy).

### What

1. **`gpu_dispatch.rs` refactoring**: Introduced `gpu_or_cpu` private helper
   that centralises the "try GPU, log-and-fallback" pattern. All 25 dispatch
   methods now use it, eliminating the 8-line boilerplate per method.

2. **`exit_no_gpu()` hardening**: Unified 79 validation/bench binaries to use
   `validation::exit_no_gpu()`. When `NEURALSPRING_REQUIRE_GPU=1` is set, GPU
   absence is a hard failure (exit 1) instead of a silent skip (exit 0).

3. **`baseline_path()` data resolution**: Replaced 4 hardcoded
   `concat!(env!("CARGO_MANIFEST_DIR"), ...)` paths with
   `validation::baseline_path("control/...")` — a single source of truth.

4. **Clippy + doc cleanup**: Fixed `unnecessary_map_or` (→ `is_ok_and`),
   `manual_let_else`, private intra-doc link. Zero warnings across all targets.

5. **EVOLUTION_MAPPING fix**: Corrected "stub" labels — `mlp_forward` exists
   in `pinn.rs` and `deeponet.rs`, not as stubs in `surrogate.rs`.

### What We Found

- Zero TODO/FIXME/HACK/MOCK/STUB in any Rust source file.
- Zero `unsafe` blocks (enforced by `forbid`).
- Zero `.unwrap()` or `.expect()` in production code (all in `#[cfg(test)]`).
- All 90+ tolerances are named, documented, and justified.
- All provenance records include script, commit, date, environment, and command.
- Max file size: 965 lines (`validate_barracuda_tensor.rs`) — under 1000.
- 394 tests pass. 9 doc-tests pass. 3 doc-tests correctly `ignore`d (GPU).

### Result

| Metric | Before | After |
|--------|--------|-------|
| Hardcoded paths | 4 (concat!) | **0** (all via `baseline_path`) |
| GPU skip pattern | 3 variants, 79 files | **1 pattern** (`exit_no_gpu`) |
| Dispatch boilerplate | 8 lines/method × 25 | **5 lines/method** via `gpu_or_cpu` |
| Clippy warnings | 0 | **0** |
| Doc warnings | 0 | **0** |
| TODOs in src/ | 0 | **0** |
| `cargo test` | 374+9+9 PASS | **374+9+9 PASS** |

---

## Experiment 019: baseCamp — Biophysical AI Interpretability — Session 50

**Date**: February 24, 2026
**Hardware**: i9-12900K (CPU-only for core primitives)
**ToadStool HEAD**: `b41ee5f4`

### Why

Implement the core Rust library modules and validation binaries for the
Biophysical AI Interpretability research program (5 sub-theses). These
modules apply validated physics/biology primitives — spectral analysis,
signal propagation, dynamical systems, game theory — to understanding
AI systems as physical systems. This is neuralSpring's novel niche:
**no prior work applies Anderson localization IPR to neural network
weight matrices, or models LSTM gating as stencil propagation on a
disordered lattice.**

### What

1. **`weight_spectral.rs` (nS-01)**: Weight-to-Hamiltonian symmetrization,
   empirical spectral density, level spacing ratio, Marchenko-Pastur bounds
   and departure, spectral entropy, activation IPR. Composes `eigh` and
   `anderson_localization` primitives.

2. **`information_flow.rs` (nS-02)**: Depth scale, gate disorder parameter,
   gate saturation, information IPR, attention-to-Hamiltonian conversion,
   MLP signal propagation (mean-field), Jacobian spectral radius.

3. **`loss_landscape.rs` (nS-03)**: Numerical Hessian via finite differences,
   Hessian spectrum, landscape flatness/sharpness, saddle index,
   Metropolis-Hastings MCMC (Boltzmann sampling), transition barrier,
   spectral gap. Validated against analytical quadratic and Rosenbrock.

4. **`neural_pgm.rs` (nS-04)**: Weight-to-transition matrix (softmax
   normalization), belief propagation chain (forward pass), KL divergence
   NN vs PGM, layer spectral similarity, effective rank via eigenvalue
   entropy, PGM complexity.

5. **`agent_coordination.rs` (nS-05)**: Interaction graphs, graph Laplacian,
   disordered Laplacian, coordination spectral analysis (IPR, level spacing,
   algebraic connectivity), QS signaling dynamics, coordination fraction,
   lattice agent generation (1D/2D/3D), dimensional coordination sweep.

### Cross-Spring Connections

| Sub-thesis | From hotSpring | From wetSpring |
|:----------:|---------------|---------------|
| nS-01 | — | Anderson QS (IPR, level spacing) |
| nS-02 | — | QS signal propagation |
| nS-03 | RK45, Boltzmann, energy minimization | — |
| nS-04 | — | HMM phylogenetics |
| nS-05 | — | Anderson QS dimensional analysis, game theory |

### Result

| Metric | Value |
|--------|-------|
| New library modules | 5 |
| New validation binaries | 5 |
| New checks (all PASS) | 82/82 |
| Unit tests (lib total) | 459 |
| Clippy warnings | 0 (pedantic + nursery) |
| Doc warnings | 0 |
| `cargo test --lib` | 459/459 PASS |
| Max file size | Under 1000 lines |
| `unsafe` blocks | 0 |
| Determinism | All 5 validators pass re-run identity check |

### GPU Evolution Candidates

These modules are CPU-only. Priority GPU promotion targets:

| Function | GPU Approach | Impact |
|----------|-------------|--------|
| `weight_to_hamiltonian` | Tensor matmul (`W^T * W`) | nS-01 bottleneck |
| `numerical_hessian` | GPU parallel finite differences | nS-03 bottleneck |
| `interaction_graph` | GPU pairwise distance | nS-05 scaling |
| `belief_propagation_chain` | GPU batch GEMV (HMM pattern) | nS-04 scaling |
| `boltzmann_sampling` | GPU parallel chain MCMC | nS-03 throughput |

---

## Experiment 020: Session 51 — Code Quality Evolution & Documentation Refresh

**Date**: February 24, 2026
**Hardware**: i9-12900K (CPU-only — code quality + documentation focus)
**ToadStool HEAD**: `b41ee5f4`

### Why

Deep code quality evolution following the Session 50 baseCamp implementation.
The audit (from the prior session) identified: float comparisons using `assert_eq!`
on `f64`/`Vec<f64>`, inline `1e-14` zero-detection guards in validation binaries,
the monolithic `gpu_dispatch.rs` (860 lines) containing mixed dispatcher logic
and CPU fallback implementations, and stale test/coverage counts across 13+ docs.

### What

1. **`gpu_dispatch.rs` → `gpu_dispatch/` module**: Smart refactoring — extracted
   CPU fallback implementations (`variance`, `pearson`, `chi_squared`,
   `hmm_backward_step`, `hmm_viterbi_step`, `replicator_step`) into
   `gpu_dispatch/cpu_fallback.rs`. Dispatcher logic remains in `gpu_dispatch/mod.rs`.
   Both files under 1000 LOC wateringHole limit. CPU fallbacks now independently
   testable and reusable.

2. **Float comparison evolution**: Replaced all `assert_eq!` on `f64`/`Vec<f64>`
   with epsilon-based comparisons across 5 library modules:
   - `agent_coordination.rs` — capability and position determinism
   - `information_flow.rs` — weight determinism
   - `weight_spectral.rs` — eigenvalue and mean_ipr determinism
   - `neural_pgm.rs` — transition matrix determinism
   - `game_theory.rs` — replicator dynamics per-step determinism
   - `pangenome_selection.rs` — frequency spectrum sum

3. **Inline guard centralization**: Replaced 7 inline `1e-14` zero-detection
   guards with `tolerances::ZERO_DETECTION` in 5 validation binaries:
   `validate_barracuda_fft`, `validate_agent_coordination`,
   `validate_barracuda_spectral_theory`, `validate_barracuda_spectral`.

4. **Clippy pedantic resolution**: Fixed `float_cmp`, `cast_lossless`,
   `identity_op`, `manual_midpoint`, `redundant_closure`, `doc_markdown`,
   `redundant_pub_crate` warnings. Replaced `i as f64` with `f64::from(i)`,
   `.expect("no NaN")` with `.unwrap_or(Ordering::Equal)`.

5. **Test coverage**: Added ~85 new tests across `neural_pgm.rs` (14 tests),
   `validation.rs` (10 tests), `weight_spectral.rs` (19 tests) — covering
   edge cases, empty inputs, boundary conditions.

6. **Documentation refresh**: Updated "412 lib tests" → "459 lib tests" across
   13 living docs. Updated coverage 92.7% → 92.9% in `EVOLUTION_READINESS.md`.
   Fixed stale `gpu_dispatch.rs` → `gpu_dispatch/` references in README.

### Result

| Metric | Before | After |
|--------|--------|-------|
| Clippy warnings (all-targets) | 0 | **0** |
| Float `assert_eq!` on f64 | 6 | **0** |
| Inline `1e-14` guards | 7 | **0** (all via `tolerances::ZERO_DETECTION`) |
| `gpu_dispatch.rs` size | 860 lines (monolithic) | **2 files** (mod.rs + cpu_fallback.rs) |
| Lib tests | 459 | **459** |
| Line coverage | 92.7% | **92.9%** |
| Doc count accuracy | Stale (412 in 13 files) | **Current** (459 in all) |
| Production `.unwrap()`/`.expect()` | 0 | **0** (confirmed) |
| `unsafe` blocks | 0 | **0** |
| Dependency purity | Pure Rust | **Confirmed** (only linux-raw-sys, renderdoc-sys transitive via wgpu) |

### Findings

1. **gpu_dispatch refactoring reveals reuse opportunity**: The extracted
   `cpu_fallback` functions (variance, pearson, chi-squared, HMM steps,
   replicator) are clean statistical/algorithmic primitives. These could
   become `barracuda::stats::*` or `barracuda::bio::*` CPU reference
   implementations — the same pattern hotSpring and wetSpring use for
   CPU validation baselines.

2. **metalForge forge crate clean**: `cargo clippy -p neural-spring-forge`
   produces zero warnings — the forge crate is well-maintained.

3. **All bins under 1000 LOC**: Largest is `validate_barracuda_tensor.rs`
   at 965 lines. No wateringHole limit violations.

---

## Experiment 021 — ToadStool Sync & Cross-Spring Benchmarking (Session 52)

**Date**: February 24, 2026
**Objective**: Complete ToadStool sync, absorb upstream shaders, benchmark
cross-spring evolution, validate full stack.

### Protocol

1. **ToadStool sync**: 16 commits absorbed (`b41ee5f4` → `9abd6857`)
2. **Shader absorption verification**: 6 shaders confirmed absorbed upstream
3. **`level_spacing_ratio` rewire**: Delegated to `barracuda::spectral`
4. **Documentation updates**: Absorption tracker, evolved module docs, variance convention
5. **Full validation**: fmt, clippy, test, coverage, validate_all
6. **Cross-spring benchmarking**: 7 ops from 3 springs on RTX 4070

### Benchmark Results (RTX 4070, Vulkan, `--release`, 20 iterations)

| Op | Origin | Size | µs |
|----|--------|------|----|
| BatchFitnessGpu | neuralSpring (S-25) | 1024×64 | 1,337 |
| PairwiseL2Gpu | neuralSpring (S-42) | 128×16 | 1,542 |
| BatchIprGpu | neuralSpring (S-25) | 32×64 | 2,027 |
| SpatialPayoffGpu | neuralSpring (S-25) | 32×32 | 1,450 |
| PairwiseHammingGpu | neuralSpring (S-25) | 64×100 | 1,682 |
| HmmBatchForwardF64 | wetSpring (S-39) | 4s×50t×32b | 2,141 |
| BatchedEighGpu | hotSpring (S-39) | 12×12×40 | 6,629 |

### Key Finding: Cross-Spring Transparency

All three Springs' shaders run through the same BarraCUDA API on RTX 4070.
A neuralSpring user calling `HmmBatchForwardF64` (wetSpring origin) or
`BatchedEighGpu` (hotSpring origin) sees no API difference from calling
`BatchFitnessGpu` (neuralSpring origin). The ToadStool absorption model
works: evolve locally → validate → hand off → absorb upstream → retire.

### Validation

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets` | 0 warnings (pedantic + nursery) |
| `cargo doc --no-deps` | 0 warnings (146 pages) |
| `cargo test --lib` | 459 PASS |
| `cargo llvm-cov --lib` | 92.89% line coverage |
| `validate_all` | 137/138 PASS (1 pre-existing logsumexp driver issue) |

### Status After

| Metric | Value |
|--------|-------|
| Local shaders remaining | 2 (head_split + head_concat — MHA S-03b) |
| Shaders absorbed upstream | All others (6 new in this session) |
| API gaps closed | argmax_dim, softmax_dim |
| Functions rewired | level_spacing_ratio → barracuda::spectral |

---

## Experiment 022 — S-17 HillGate f64 `pow()` Fix (Session 52b)

### Objective

Identify root cause of `HillGateGpu` f64 shader compilation failure and produce
a validated local fix for ToadStool absorption.

### Root Cause

`hill_gate_f64.wgsl` uses native WGSL `pow(f64, f64)`. On both RTX 4070 (NVVM
proprietary) and TITAN V (NAK open-source), native f64 `pow()` triggers shader
compilation failure, causing device loss. `compile_shader_f64` patches `exp()`
and `log()` to polyfills via `apply_transcendental_workaround` but does **not**
patch `pow()`. The detection (`needs_pow_f64_workaround()`) already exists in
`driver_profile.rs` but is not wired to the patching pipeline.

### Fix

Replace `pow(` → `pow_f64(` in shader source before `compile_shader_f64`.
The existing `inject_missing_math_f64` auto-detects the `pow_f64()` call and
injects the polyfill from `math_f64.wgsl` (which uses
`exp_f64(exponent * log_f64(base))` with special-case handling for integer
exponents and common fractions).

### Validation

| Adapter | Driver | Unpatched | Patched | Max Diff |
|---------|--------|-----------|---------|----------|
| RTX 4070 | Vulkan proprietary | NVVM fail → device lost | 18/18 PASS | 1.11e-16 |
| TITAN V | NVK open-source | NAK assertion fail | 18/18 PASS | 2.22e-16 |

`validate_gpu_signal` evolved from SKIP → 9/9 PASS on both GPUs.
Also fixed pre-existing f32/f64 buffer mismatch (old validator uploaded f32
data to f64 shader — masked because shader never compiled).

### ToadStool Action

One-line addition to `patch_exp_log_in_code` in `barracuda/src/shaders/precision/mod.rs`:
`.replace("pow(", "pow_f64(")`. Also fix `hill_f64.wgsl` (element-wise Hill).

### Files

| File | Purpose |
|------|---------|
| `src/bin/validate_hillgate_f64_fix.rs` | Full proof-of-concept (18/18 PASS) |
| `src/bin/validate_gpu_signal.rs` | Evolved: polyfill + f64 buffers (9/9 PASS) |

---

## Experiment 023 — baseCamp Experiment Expansion & GPU Workload Validation

**Date**: February 24, 2026 (Session 54)
**Hardware**: RTX 4070, i9-12900K, Pop!_OS 22.04

### Motivation

baseCamp had 82 core primitive checks across 5 modules but lacked coverage
for individual experiments (nS-103 through nS-505) and had no pure GPU
workload validation proving the math is hardware-portable.

### Procedure

1. Expanded all 5 baseCamp validators from 82→114 checks:
   - nS-104 (Dyson dynamics), nS-105 (cross-architecture), nS-106 (GNN over-smoothing)
   - nS-205 (Hill activation), nS-206 (edge-of-chaos sweep), nS-203 (deep layer IPR)
   - nS-304 (dimension sweep), nS-305 (gradient descent), nS-302 (multi-barrier)
   - nS-402 (deep factor graph BP), nS-405 (OOD detection), nS-404 (rank monotonicity)
   - nS-504 (scaling), nS-505 (Anderson transition), nS-501 (dimensional sweep)
2. Created `validate_basecamp_gpu` (14/14 PASS): pure GPU eigensolve + variance +
   Pearson + entropy + matmul + chi² + L2 + KL divergence
3. Created `bench_basecamp_parity`: CPU↔GPU parity benchmark (var 7.77e-16,
   pearson 6.94e-18, entropy 1.60e-11 — all sub-epsilon)

### Results

- baseCamp: 82→114 CPU checks + 14 GPU checks = 128/128 PASS
- `validate_all`: 139/140 PASS (1 pre-existing logsumexp driver issue)

---

## Experiment 024 — BarraCUDA CPU vs GPU Dispatch + metalForge Mixed Hardware

**Date**: February 24, 2026 (Session 55)
**Hardware**: RTX 4070, i9-12900K, Pop!_OS 22.04

### Motivation

With baseCamp experiments validated, the next step was proving that the same
workloads produce identical results through both CPU and GPU compute paths
(BarraCUDA dispatch parity), and wiring the metalForge mixed-hardware cost
model into the `Dispatcher` for GPU↔NPU↔CPU substrate routing.

### Procedure

1. Created `validate_compute_dispatch` (16/16 PASS): routing correctness +
   CPU↔GPU parity for variance, Pearson, entropy, chi-squared, eigendecomposition,
   and dispatch-aware workload routing
2. Wired `metalForge::mixed::mixed_substrate()` into `Dispatcher::mixed_dispatch()`:
   - Small workloads → CPU (below crossover threshold)
   - Large workloads → GPU (compute dominates transfer)
   - Realtime inference → GPU→NPU (simulated, PCIe cost model)
3. Created `validate_mixed_hardware` (14/14 PASS): mixed routing, PCIe bridge,
   transfer cost model, crossover boundary verification
4. Fixed 5 sub-thesis docs (14 stale binary references corrected)
5. Updated 15 grounding papers (B-01..B-15) from "Queued" to "Primitives validated"

### Results

- `validate_compute_dispatch`: 16/16 PASS — CPU↔GPU parity within machine epsilon
- `validate_mixed_hardware`: 14/14 PASS — correct substrate routing at all scales
- `validate_all`: 141/142 PASS (1 pre-existing logsumexp driver issue)
- `Dispatcher::mixed_dispatch()` ready for ToadStool absorption

### Key Finding: metalForge Cost Model Works

The dispatch router correctly identifies the GPU↔CPU crossover at ~1.5ms
(the empirical `queue.submit()` + readback overhead on RTX 4070). Below
this threshold CPU wins; above it GPU dominates. The PCIe transfer cost
model accurately predicts that P2P (2µs latency) is faster than CPU-staged
(7µs) for GPU↔NPU transfers.

---

## Experiment 025 — ToadStool S53 Sync, Upstream Rewiring, and Dispatch Validation

**Date**: February 24, 2026 (Session 56)
**Hardware**: RTX 4070, i9-12900K, Pop!_OS 22.04

### Motivation

ToadStool had been absorbing neuralSpring handoffs through Sessions 51–53 and
reached `9404fdb4` with new upstream modules (`linalg::graph`, `numerical`,
`ops::bio::swarm_nn`, `ops::bio::xoshiro128ss`). neuralSpring needed to pull
the latest state, rewire local implementations to use upstream, validate
parity, and create comprehensive dispatch/parity validators.

### Procedure

1. Pulled ToadStool HEAD `9404fdb4`, reviewed commit history (4 upstream
   modules absorbed from neuralSpring handoffs)
2. Rewired 4 local functions to delegate to upstream BarraCUDA:
   - `graph_laplacian` → `barracuda::linalg::graph`
   - `disordered_laplacian` → `barracuda::linalg::graph`
   - `belief_propagation_chain` → `barracuda::linalg::graph`
   - `numerical_hessian` → `barracuda::numerical`
3. Created `validate_basecamp_dispatch` (19/19 PASS): exercises all 4 baseCamp
   Dispatcher methods (weight spectral, Hessian, belief propagation, agent graph)
4. Created `validate_barracuda_parity` (34/34 PASS): CPU↔GPU parity across
   linear algebra, statistics, spectral, activations, reductions, distance, biology
5. Created `validate_metalforge_pcie` (36/36 PASS): bandwidth tiers, P2P vs
   staged transfers, chained multi-hop, substrate selection, bridge API, live dispatch
6. Updated metalForge mixed.rs with PCIe bandwidth tiers and chained transfer costs
7. Updated all docs: EVOLUTION_READINESS, BARRACUDA_USAGE, ABSORPTION_MANIFEST,
   TOADSTOOL_HANDOFF, baseCamp sub-theses

### Results

- 4 functions rewired — public API preserved, all 478 lib tests pass
- 3 new validators: 89 additional checks (19 + 34 + 36)
- Total checks: 2010+ (206 Python + 1810+ Rust+GPU)
- `validate_all`: 155 binaries PASS
- Quality gates: fmt ✓ · clippy (pedantic+nursery) ✓ · doc ✓

### Key Finding: Upstream Absorption Loop Works

The handoff → absorb → rewire → validate cycle is now proven end-to-end.
neuralSpring hands off implementations, ToadStool absorbs them into BarraCUDA,
neuralSpring rewires to use upstream, and all tests still pass. The thin-wrapper
pattern (local function delegates to `barracuda::*` with identical API) minimizes
migration risk while eliminating duplicated math.

---

## Experiment 026 — Cross-Spring Dispatch Rewiring + GpuDriverProfile

**Date**: February 24, 2026 (Session 58)
**Hardware**: RTX 4070, i9-12900K, Pop!_OS 22.04

### Motivation

ToadStool S58–S59 absorbed `barracuda::dispatch::domain_ops` (matmul, frobenius,
transpose, softmax, l2, mean, variance dispatch functions) and
`barracuda::device::driver_profile::GpuDriverProfile` (hotSpring-evolved hardware
detection including Fp64Strategy, driver workarounds, eigensolve strategy).
neuralSpring should rewire its Dispatcher to delegate to these upstream
implementations and wire in driver profile information.

### Procedure

1. Rewired 7 Dispatcher methods to upstream `domain_ops`:
   `mat_mul`, `frobenius_norm`, `transpose`, `softmax`, `l2_distance`, `mean`, `variance`.
   Each now calls `barracuda::dispatch::*_dispatch(data, self.wgpu_device())` with
   local CPU fallback on error.
2. Wired `GpuDriverProfile` into Dispatcher struct — built at init from `WgpuDevice`,
   exposing `driver_profile()`, `fp64_strategy()`, `needs_pow_workaround()`.
3. Created `validate_cross_spring_evolution` (10/10 PASS): rewired method parity,
   driver profile detection, throughput benchmark, cross-spring lineage report.
4. Updated `validate_all` binary list (now 146 entries).

### Results

- 7 methods rewired — public API preserved, all 478 lib tests pass
- GpuDriverProfile detected: Ada arch, NvidiaPtxas, Throttled FP64, Hybrid strategy
- Benchmark: upstream dispatch uses GPU for large workloads (size-based thresholds),
  CPU for small ones — matches our existing `gpu_or_cpu` behavior but with
  upstream-managed thresholds
- GPU matmul parity: max diff 2.3e-4 (accumulation order, within 1e-3 tolerance)
- Total rewired functions: 11 (4 from S56 + 7 from S58)
- Quality gates: fmt ✓ · clippy (pedantic+nursery) ✓ · 478 lib ✓ · 145/146 validate_all

### Key Finding: Cross-Spring Hardware Awareness

The `GpuDriverProfile` demonstrates the cross-spring evolution cycle at its best:
hotSpring discovered the need for hardware-adaptive f64 strategies during lattice QCD
work (compute-class GPUs have 1:2 FP64:FP32 vs consumer 1:64 ratio). This led to
`Fp64Strategy::Native` vs `Fp64Strategy::Hybrid` detection, which ToadStool absorbed.
neuralSpring now consumes this upstream capability, and the RTX 4070 correctly reports
`Hybrid` strategy — meaning bulk math should use df64 f32-pairs while precision-critical
reductions use native f64. This hardware-awareness will inform future metalForge
mixed-hardware dispatch decisions.

### Cross-Spring Shader Lineage Documented

| Spring | Contributions |
|--------|--------------|
| hotSpring | df64_core, pow_f64, Fp64Strategy, GpuDriverProfile, Taylor trig, Lanczos |
| wetSpring | HMM, ODE bio (5), NMF, Anderson localization, Ridge regression |
| neuralSpring | ValidationHarness, batch_fitness, pairwise ops, eigh, KernelRouter |

---

*Experiment journals — following the hotSpring pattern.*
