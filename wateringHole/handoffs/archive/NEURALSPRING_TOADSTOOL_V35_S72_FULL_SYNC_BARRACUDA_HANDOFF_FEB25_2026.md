# neuralSpring → ToadStool/BarraCUDA Handoff V35

**Session 72 — Full ToadStool Sync, All 17 Shortcomings Resolved, New API Leverage**
**Date**: February 25, 2026
**From**: neuralSpring
**To**: ToadStool / BarraCUDA core team
**License**: AGPL-3.0-or-later
**ToadStool HEAD**: `02207c4a` (47 commits reviewed: `77f70b2e`..`02207c4a`)
**Supersedes**: V34 (Session 71 — Tolerance Evolution)

---

## Executive Summary

Session 72 performed a comprehensive review of all 47 ToadStool commits since
our last deep-tracked commit (`77f70b2e`). Key findings:

1. **ALL 17 shortcomings RESOLVED upstream** — S-01 through S-17, including
   S-14/S-15/S-16 (matmul/transpose bugs fixed at `a4996b34`) and S-17
   (pow polyfill absorbed at `c82c23d1`). neuralSpring docs fully updated.

2. **Previously blocked Tensor APIs now available** — `argmax_dim(axis)` and
   `softmax_dim(axis)` remove the two longest-standing neuralSpring requests.
   These enable full Viterbi GPU path and proper row-wise attention softmax.

3. **New upstream APIs identified for leverage** — `fst_variance_decomposition`,
   `Conv2dGpu`, `PeakDetectF64`, `MovingWindowStats`, `SparseGemmF64`,
   `TranseScoreF64`, `ridge_regression`, `NMF`.

4. **ToadStool absorption gap identified** — ToadStool docs reference
   neuralSpring handoff V16/V18 (baseCamp, Feb 2026). V19–V35 not yet consumed.
   This handoff provides the complete delta.

5. **Validator workarounds retained** — S-14/S-15 data patterns (positive-only,
   A×B^T) kept in 18+ validators as defense-in-depth. Not harmful, still correct.

---

## Part 1: ToadStool Commit Review (47 Commits)

### Commits Audited

| ToadStool Session | Commit | Key Changes |
|-------------------|--------|-------------|
| S39 | `d45fdfb3`..`a4996b34` | Dead code sweep, **S-14/S-15/S-16 FIXED**, Spring shaders absorbed, `FlatTree`, `sparse_eigh`, `execute_to_buffer` |
| S40 | `28b5f1a7` | Richards PDE solver, `MovingWindowStats`, dependency audit |
| S41 | `a2326909`..`aad63fb9` | 6 f64 shader compile fixes, API exposure for Springs, doc cleanup |
| S42 | `446de03c`..`5437c170` | **19 new WGSL shaders**, BarraCUDA → BarraCuda rename |
| S45–S46 | `c8076a2d`..`fe573095` | Deep debt, typed errors, lattice QCD, MD transport, bio ODE, MHA fix |
| S49 | `9bd71391` | Shader-first architecture, doc cleanup |
| S50 | `e8c6f582` | Deep audit remediation |
| S51 | `6f3382d0` | CG shaders, ESN NPU, generic ODE, CPU solver |
| S52 | `8eac60d7`..`8d43d4df` | **18 cross-spring items absorbed**, +103 tests, smart refactor (all under 1000 lines), CG infra, domain dispatch |
| S53 | `d0832483`..`9abd6857` | Coverage push (+193 tests), 4176 tests total, `Box<dyn Error>` elimination |
| S54 | `8c9a4c48` | **baseCamp primitives absorbed** (5 WGSL shaders), GPU fixes |
| S55 | `ff631c9f` | Deep debt, refactor, hardcoding, stubs, unsafe audit |
| S56 | `c475d5c0` | **Final absorptions** — all cross-spring items complete |
| S57 | `ece5a403`..`f78cf3b0` | Coverage push (+47 tests), archive cleanup, doc updates |
| S58 | `c82c23d1` | **S-17 FIXED** (pow polyfill), df64, `Fp64Strategy`, ODE bio, NMF |
| S59 | `9404fdb4` | Anderson correlated, ridge regression, validation harness |
| S60–S61 | `0c998992` | **MHA decomposition** (S-03b RESOLVED), Conv2D GPU, NVK guard, SpMM, TransE |
| S62 | `2dc76044` | `BandwidthTier`, `PeakDetectF64`, pool padding |
| HEAD | `02207c4a` | **DF64 expansion** + architectural evolution, deep debt reduction |

### Net Effect

| Metric | Pre-review | Post-review |
|--------|-----------|-------------|
| Shortcomings open | 4 (S-14, S-15, S-17, some S-16 refs) | **0** |
| Blocked API requests | 3 (`argmax_dim`, `softmax_dim`, `WGSL_MEAN_REDUCE`) | 1 (`WGSL_MEAN_REDUCE` still private) |
| New upstream APIs available | — | 9 (see Part 2) |
| ToadStool test count | ~4,000 | 14,200+ |

---

## Part 2: New Upstream APIs Available for neuralSpring

### Already Available (Leverage Opportunities)

| API | Module | neuralSpring Use Case | Priority |
|-----|--------|----------------------|----------|
| `argmax_dim(axis)` | `tensor::Tensor` | Viterbi argmax — currently CPU-only | **P1** |
| `softmax_dim(axis)` | `tensor::Tensor` | Row-wise attention softmax | **P1** |
| `fst_variance_decomposition` | `ops::bio::fst_variance` | FST was CPU-only in neuralSpring | **P2** |
| `Conv2dGpu` | `ops::nn` | Full NCHW Conv2D — LeNet validation | **P2** |
| `PeakDetectF64` | `ops::peak_detect_f64` | Spectral peak detection | P3 |
| `MovingWindowStats` | `ops::moving_window_stats` | Sliding window mean/var/min/max | P3 |
| `SparseGemmF64` | `ops::sparse_gemm_f64` | Sparse matrix multiply | P3 |
| `ridge_regression` | `linalg::ridge` | Ridge regression | P3 |
| `NMF` | `linalg::nmf` | Non-negative matrix factorization | P3 |

### Still Blocked

| API | Status |
|-----|--------|
| `WGSL_MEAN_REDUCE` public constant | Not exported — still private in `mean.rs` |

---

## Part 3: Shortcoming Resolution Summary

### All 17 Shortcomings — RESOLVED

| # | Shortcoming | Resolution | ToadStool Commit |
|---|-------------|-----------|-----------------|
| S-01 | Per-op submission | `TensorSession` single-encoder batch | `fbedd222` |
| S-02 | Naive matmul | 4-tier `KernelRouter` | `82f953c8` |
| S-03 | MHA z-dispatch | `workgroups_z = params.seq_len` | `82f953c8` |
| S-03b | MHA projection hang | Decomposed into matmul + head_split/head_concat | `0c998992` |
| S-04 | Softmax pooled | `params.size` uniform | `82f953c8` |
| S-05 | leaky_relu Params | `{size, negative_slope}` | `82f953c8` |
| S-06 | elu Params | `{size, alpha}` | `82f953c8` |
| S-07 | from_buffer pub(crate) | `pub fn from_buffer()` | `81a6fd4b` |
| S-08 | layer_norm round-trip | `from_pooled_buffer` | `81a6fd4b` |
| S-09 | log_softmax round-trip | `from_pooled_buffer` | `81a6fd4b` |
| S-10 | science_limits CPU | `new_cpu_relaxed()` | `81a6fd4b` |
| S-11 | TensorSession limited | ML ops in SessionOp | `fbedd222` |
| S-12 | eigh_f64 accuracy | Householder+QR | `77f70b2e` |
| S-13 | PooledBuffer race | Deferred return + device poll | `5437c170` |
| S-14 | Naive matmul hang | Naive tier removed | `a4996b34` |
| S-15 | Matmul magnitude hang | Driver-level fix | `a4996b34` |
| S-16 | Transpose dispatch | `const TILE: u32 = 16` | `a4996b34` |
| S-17 | pow(f64) crash | `patch_transcendentals_in_code` covers pow | `c82c23d1` |

---

## Part 4: Absorption Gap — What ToadStool Hasn't Consumed

ToadStool docs reference neuralSpring handoff **V16/V18** (baseCamp, Feb 2026).
Handoffs **V19–V35** contain the following uncovered items:

### High Priority (Recommended for Next Absorption)

| Item | Handoff | Description |
|------|---------|-------------|
| Tolerance registry pattern | V31–V34 | `tolerance_registry!` macro (105+ constants, compile-time validation, runtime introspection) |
| GPU test serialization | V31 | `OnceLock<Mutex<()>>` pattern for 580 tests with zero flakiness |
| Streaming JSON loading | V33 | `BufReader` + `serde_json::from_reader` for large reference datasets |

### Medium Priority

| Item | Handoff | Description |
|------|---------|-------------|
| Cross-spring evolution benchmarks | V32 | 22/22 PASS — Variance 3.49×, Entropy 2.56×, Pearson 1.33× |
| Smart refactoring pattern | V34 | Semantic extraction > arbitrary splitting (tolerance macro 891→257, gpu_dispatch 862→304) |
| Ad-hoc tolerance elimination | V34 | 150+ bare numerics → named constants pattern |

### Informational (No Action Required)

| Item | Handoff | Description |
|------|---------|-------------|
| 94.53% coverage ceiling | V33 | GPU error branches are architectural limit |
| Pure Rust dependency verification | V34 | All crates ecoBin compliant |
| 580 lib tests, 0 warnings | V34 | Quality gates all green |

---

## Part 5: What neuralSpring Validated for ToadStool

### BarraCUDA API Stability

**Zero breaking changes** across 47 commits (S39–S62). Every ToadStool sync
compiled cleanly in neuralSpring. The `barracuda` crate API is remarkably stable.

### Cross-Spring Value Delivered

| Spring | Contribution | Validated By |
|--------|-------------|-------------|
| **hotSpring** | df64_core, pow_f64 polyfill, Welford variance, GpuDriverProfile | 3.49× variance speedup |
| **wetSpring** | HMM f64, fused map-reduce, log_f64 fix, dN/dS, pangenome classify | 2.56× entropy speedup |
| **neuralSpring** | eigh Householder+QR, batch fitness, tolerance registry, MHA validation | Trillion-fold eigensolver accuracy |

### Validator Pattern Learnings

| Pattern | Impact |
|---------|--------|
| Defense-in-depth data generation | Validators remain correct even if bugs reappear |
| `patch_pow_to_polyfill` retained | Safety net for WGSL loaded outside barracuda pipeline |
| Named tolerances in ALL test assertions | Global tolerance policy changes now possible |

---

## Part 6: Full Metrics (Session 72)

| Gate | Result |
|------|--------|
| `cargo fmt --check` | **PASS** |
| `cargo clippy --all-targets -D warnings -W pedantic -W nursery` | **0 warnings** |
| `cargo test --lib` | **580/580 PASS** |
| `cargo test --test integration` | **9/9 PASS** |
| `cargo doc --no-deps` | **0 warnings** |
| Shortcomings resolved | **17/17** (was 13/17 before S72 review) |
| Blocked API requests | **1** (`WGSL_MEAN_REDUCE`) — was 3 |
| New upstream APIs available | **9** |
| ToadStool commits reviewed | **47** |
| Docs updated | **15+ files** |

---

## Supersedes

- V34: Session 71 — Tolerance Evolution, BarraCUDA Absorption Recommendations
  (`wateringHole/handoffs/archive/`)

---

*neuralSpring → ToadStool handoff V35 — AGPL-3.0-or-later*
