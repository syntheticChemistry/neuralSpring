# neuralSpring → ToadStool/BarraCUDA Handoff: V49 Cross-Spring Evolution + Learnings

**Date:** February 26, 2026
**From:** neuralSpring Session 85
**To:** ToadStool/BarraCUDA team
**ToadStool pin:** S68 (`f0feb226`)
**neuralSpring:** 604 lib + 43 forge + 9 integration tests, 166 binaries, 150/150 GPU PASS

---

## Executive Summary

- Five-spring provenance map complete: ~700 WGSL shaders traced to origin Spring
- 28/28 cross-spring benchmark PASS with full provenance annotations
- **Hamming 20.85× regression** identified — investigation target (see Part 3)
- All 39 function rewires + 6 shader sources verified against modern S68 APIs
- Comprehensive doc sweep: all stale counts fixed across 20+ documents
- Recommendations for ToadStool evolution (see Part 5)

---

## Part 1: Five-Spring Provenance Map

neuralSpring now tracks the complete provenance of ~700 WGSL shaders across
all five Springs:

| Spring | ~Shaders | Domain | Key Contributions |
|--------|----------|--------|-------------------|
| hotSpring | ~100 | Nuclear physics | DF64 core, lattice QCD, HFB, Lanczos, ESN, precision infra |
| wetSpring | ~80 | Bioinformatics | HMM, DADA2, diversity, ODE, NMF, Bray-Curtis, SNP |
| neuralSpring | ~34 | ML/neuroevolution | Batch fitness, pairwise ops, IPR, session API, validation harness |
| airSpring | ~15 | Precision agriculture | Regression, hydrology, moving_window, kriging |
| groundSpring | ~5 | Hydrogeology | RAWR bootstrap, batched multinomial |
| ToadStool | ~466 | Core infrastructure | Math, linalg, nn, activations, sovereign compiler |

Full map: `whitePaper/CROSS_SPRING_SHADER_LINEAGE.md`

---

## Part 2: S68 API Benchmark Results (RTX 4070, Release)

### New S66–S68 Stats APIs (N=10,000)

| API | Origin | Time (µs) |
|-----|--------|-----------|
| `fit_linear` | airSpring | 44.1 |
| `fit_quadratic` | airSpring | 75.0 |
| `fit_exponential` | airSpring | 126.1 |
| `fit_all` | airSpring | 388.2 |
| `spearman_correlation` | wetSpring+hotSpring | 449.1 |
| `rawr_mean` | groundSpring | 619.9 |

### GPU Dispatch Provenance (N=50,000)

| Op | Origin | Time (µs) |
|----|--------|-----------|
| Variance (Welford) | hotSpring | 10,426 |
| Pearson (Correlation) | wetSpring+hotSpring | 9,230 |
| Shannon (FusedMapReduce) | wetSpring | 4,504 |
| MatMul 200×200 | neuralSpring | 2,482 |

### Rewire Evolution — f32→f64 Speedups

| Op | f32 Tensor (µs) | f64 Upstream (µs) | Speedup | Origin |
|----|-----------------|-------------------|---------|--------|
| Variance | 10,515 | 3,162 | **3.33×** | hotSpring Welford |
| Pearson | 4,343 | 4,232 | **1.03×** | wetSpring+hotSpring |
| Entropy | 8,022 | 3,912 | **2.05×** | wetSpring fused |

---

## Part 3: Hamming 20.85× Regression — Investigation Target

**Severity**: High
**Location**: `bench_upstream_vs_local` — `PairwiseHammingGpu` 200×500

| Path | Time (µs) |
|------|-----------|
| Local metalForge (f32) | 2,401 |
| Upstream BarraCUDA (f64) | 50,060 |
| **Ratio** | **20.85× slower** |

**Cause**: Upstream `PairwiseHammingGpu` uses f64 path even for small sizes
where f32 would suffice. The local metalForge shader uses f32 directly.

**Recommended action**: Consider size-based f32/f64 routing in `PairwiseHammingGpu`,
or expose a public f32 constant for consumers that don't need f64 precision on
integer Hamming distance computation (which is exact in f32).

All other 9 kernels are within 0.33–1.26× — negligible overhead.

---

## Part 4: API Friction Points

### S68 Universal Precision — LazyLock Privatization

ToadStool S68 changed several `pub const WGSL_*: &str` to private
`static WGSL_*: LazyLock<String>`. This broke neuralSpring's re-exports.
We worked around by using local shader copies, but downstream consumers
doing `pub use barracuda::ops::bio::*::WGSL_*` will also break.

**Recommendation**: Consider exposing `pub fn wgsl_source() -> &str` methods
on typed ops for consumers that need shader source access (benchmarking,
validation, custom pipeline assembly).

### Variance Convention

- `barracuda::stats::variance` uses sample variance (÷(N-1))
- `barracuda::dispatch::variance_dispatch` uses population variance (÷N)
- `barracuda::stats::variance_ddof(data, ddof)` resolves this (S66)

neuralSpring uses `variance_ddof(data, 0)` for GPU parity. No action needed,
but documenting the convention difference for other consumers.

### SimpleMLP Gap

`barracuda::nn::SimpleMLP` with JSON weight loading + forward pass would
simplify WDM surrogate validation. Currently uses custom path.

---

## Part 5: Recommendations for ToadStool Evolution

### From neuralSpring's Experience

1. **GPU test serialization**: neuralSpring uses a crate-level `test_gpu_lock`
   mutex + shared `Gpu` instance to avoid wgpu device contention in parallel
   tests. Candidate for `barracuda::testing::GpuTestHarness`.

2. **Crossover heuristics**: ~186µs structural floor per `queue.submit()`.
   CPU→GPU crossover at ~1,946µs compute. Below that, CPU wins. These
   heuristics could be centralized in `barracuda::dispatch::config`.

3. **Domain dispatch substrates**: `metalForge/forge/src/dispatch.rs` has
   proven heuristics: `logsumexp_substrate(batch, width > 20k → GPU)`,
   `stochastic_substrate(n > 100k → GPU)`. Candidate for upstream absorption.

4. **Baseline path convention**: `env!("CARGO_MANIFEST_DIR")`-relative paths
   for control data. Candidate for `barracuda::testing::baseline_path()`.

5. **Public f32 shader constants**: For ops where f64 precision is unnecessary
   (integer Hamming distance, Jaccard on binary vectors), exposing f32 shader
   variants would avoid the universal-precision overhead on small workloads.

### Cross-Spring Absorption Targets

| neuralSpring Contribution | Status | Recommendation |
|---------------------------|--------|----------------|
| `ValidationHarness` | Absorbed S64 | Keep both; neuralSpring's has `check_abs_result()` |
| `exit_no_gpu` / `require_gpu` | Absorbed S64 | Candidate for `barracuda::testing::require_gpu()` |
| `GpuTestHarness` pattern | Not yet | New: GPU test serialization with shared device |
| Domain dispatch heuristics | Not yet | `dispatch::config` — crossover points |
| `SimpleMLP` JSON loading | Not yet | High-priority for WDM surrogates |
| `WGSL_MEAN_REDUCE` public | Not yet | Expose for validator use |

---

## Part 6: Validation

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets -- -D warnings` | 0 warnings |
| `cargo test --lib` | **604/604 PASS** |
| `cargo test -p neural-spring-forge --lib` | **43/43 PASS** |
| `validate_all` | **150/150 PASS** |
| `bench_cross_spring_evolution` | **28/28 PASS** |
| `validate_cross_spring_evolution` | **52/52 PASS** |
| `bench_upstream_vs_local` | **10/10 kernels** |

---

## Part 7: Verification Commands

```bash
cd /home/eastgate/Development/ecoPrimals/neuralSpring
cargo test --lib                           # 604/604 PASS
cargo test -p neural-spring-forge --lib    # 43/43 PASS
cargo clippy --all-targets -- -D warnings  # 0 warnings
cargo run --release --bin validate_all     # 150/150 PASS
cargo run --release --bin bench_cross_spring_evolution  # 28/28 PASS
cargo run --release --bin bench_upstream_vs_local       # 10/10 kernels
```

---

## Part 8: Updated neuralSpring Documents

| Document | What Changed |
|----------|-------------|
| `specs/BARRACUDA_USAGE.md` | S84–85 session, Hamming regression, five-spring benchmark |
| `specs/TOADSTOOL_HANDOFF.md` | 39 rewires (was 30) |
| `specs/CROSS_SPRING_EVOLUTION.md` | 604 tests, 28/28 bench, 39 rewires |
| `specs/EVOLUTION_MAPPING.md` | 604 tests, 166 binaries |
| `whitePaper/STUDY.md` | 604 tests, 166 binaries |
| `whitePaper/README.md` | V48, 129+ tolerances, 28/28 bench |
| `whitePaper/CROSS_SPRING_SHADER_LINEAGE.md` | Five-spring model, S84 benchmarks |
| `whitePaper/baseCamp/*.md` | Session ranges extended to S85 |
| `whitePaper/baseCamp/waters.md` | Fixed `quorum_sensing.rs` → `signal_integration.rs` |
| `EVOLUTION_READINESS.md` | 604 tests, S84 five-spring benchmark |
| `CONTROL_EXPERIMENT_STATUS.md` | S84–85 sessions added |
| `CHANGELOG.md` | 0.4.1 (S84) + 0.4.2 (S85) releases |
| `experiments/README.md` | Exp052 (S84) + Exp053 (S85) |

---

## Action Items for ToadStool

1. **Investigate Hamming 20.85× regression** on `PairwiseHammingGpu` 200×500
2. **Consider public f32 shader constants** for integer-distance ops
3. **Consider `wgsl_source()` methods** on typed ops for downstream validation
4. **`barracuda::nn::SimpleMLP`** with JSON weight loading for surrogate models
5. **`barracuda::testing::GpuTestHarness`** — shared device + mutex pattern
6. **Document variance convention** (`stats::variance` ÷(N-1) vs `dispatch` ÷N)

---

*neuralSpring V49 handoff — February 26, 2026, Session 85. AGPL-3.0-or-later.*
