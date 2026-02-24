# neuralSpring V19 — Session 51: Code Quality Evolution + ToadStool Sync + Handoff

**Date**: February 24, 2026
**ToadStool HEAD**: `9abd6857` (Sessions 50–53: 16 commits since `b41ee5f4`)
**neuralSpring Session**: 51 (Code Quality Evolution + ToadStool Sync)
**Previous**: V18 (Session 50 — baseCamp Biophysical AI Interpretability)

---

## Part 1: What Changed in Session 51

Session 51 is a deep code quality evolution pass. No new modules or validation
binaries; all changes harden the existing 36 modules and 138 binaries.

### 1.1 Structural Refactoring

**`gpu_dispatch.rs` (860 LOC) → `gpu_dispatch/` module directory:**

| File | LOC | Contents |
|------|-----|----------|
| `gpu_dispatch/mod.rs` | ~730 | `Dispatcher` struct, `gpu_or_cpu` pattern, 25 dispatch methods |
| `gpu_dispatch/cpu_fallback.rs` | ~130 | 6 CPU reference implementations (variance, pearson, chi_squared, hmm_backward_step, hmm_viterbi_step, replicator_step) |

The CPU fallbacks are now independently testable and could be absorbed into
`barracuda::stats` / `barracuda::bio` as CPU reference implementations.

### 1.2 Clippy Pedantic Evolution

All warnings resolved at `pedantic + nursery` level:

| Lint | Count | Fix Pattern |
|------|-------|------------|
| `float_cmp` | 6 | `assert_eq!(a, b)` → `(a - b).abs() < f64::EPSILON` |
| `cast_lossless` | 2 | `i as f64` → `f64::from(i)` |
| `identity_op` | 1 | `arr[i * 8 + 0]` → `arr[i * 8]` |
| `manual_midpoint` | 1 | `(lo + hi) / 2.0` → `f64::midpoint(lo, hi)` |
| `redundant_closure` | 1 | `.map(\|i\| f64::from(i))` → `.map(f64::from)` |
| `doc_markdown` | 2 | Math notation in doc comments wrapped in backticks |
| `redundant_pub_crate` | 6 | `pub(crate) fn` → `pub fn` in private submodule |

### 1.3 Hardcoding Centralization

Replaced 7 inline `1e-14` zero-detection guards with `tolerances::ZERO_DETECTION`
in 5 validation binaries:

- `validate_barracuda_fft.rs` (4 instances)
- `validate_agent_coordination.rs` (1 instance)
- `validate_barracuda_spectral_theory.rs` (1 instance)
- `validate_barracuda_spectral.rs` (1 instance)

Algorithm convergence parameters (Nelder-Mead tolerance, bisect epsilon) remain
correctly inline — they're hyperparameters, not validation tolerances.

### 1.4 ToadStool Sync (`b41ee5f4` → `9abd6857`)

Pulled 16 new commits (Sessions 50–53). Key absorptions by ToadStool:

| ToadStool Change | Impact on neuralSpring |
|-----------------|----------------------|
| `domain_ops.rs` — 9 dispatch wrappers (matmul, variance, mean, hmm_forward, etc.) | **H-003 ABSORBED**: Pattern mirrors our `gpu_dispatch`. No rewire needed — our module is more comprehensive (25+ methods). |
| `Tensor::argmax_dim(axis)` | **M-001 ABSORBED**: Closes our API gap for Viterbi. Available for future use. |
| `Tensor::softmax_dim(axis)` | **M-001 ABSORBED**: Closes our API gap for row-wise attention. Available for future use. |
| `barracuda::spectral::level_spacing_ratio` | **REWIRED**: `weight_spectral::level_spacing_ratio` now delegates to upstream. ~15 LOC removed. |
| `barracuda::tolerances` | ToadStool created own tolerance module mirroring our pattern. |
| `fst_variance_decomposition` | Weir-Cockerham FST. Complementary to our pairwise FST matrix — no duplication. |
| `barracuda::provenance` | 12 `ProvenanceTag` constants for cross-spring tracking. |
| Smart refactors (15 files) | All barracuda .rs files now under 1000 lines. |
| 4,176 tests, 0 clippy | ToadStool quality matches Spring standards. |

**Build + test result**: 459 lib tests PASS, 0 clippy warnings, 92.9% coverage — fully compatible.

### 1.5 Test Coverage

| Metric | Before (S50) | After (S51) |
|--------|-------------|-------------|
| Lib tests | 459 | **459** (unchanged — coverage tests added in prior session) |
| Line coverage | 92.7% | **92.9%** |
| Production `.unwrap()`/`.expect()` | 0 | **0** (audit confirmed) |
| Files > 1000 LOC | 0 | **0** (largest: 965) |

### 1.5 Documentation Refresh

Updated "412 lib tests" → "459 lib tests" across 13 living docs.
Updated coverage 92.7% → 92.9% in `EVOLUTION_READINESS.md`.
Fixed `gpu_dispatch.rs` → `gpu_dispatch/` in README directory tree.
Added Experiment 020 to `experiments/README.md`.

---

## Part 2: BarraCUDA Primitives — Current Usage Summary

### Validated Primitives (Session 51 verified)

| Category | Count | Key APIs |
|----------|-------|----------|
| Typed GPU ops | 12 | BatchFitnessGpu, PairwiseHammingGpu, PairwiseJaccardGpu, PairwiseL2Gpu, LocusVarianceGpu, SpatialPayoffGpu, MultiObjFitnessGpu, BatchIprGpu, SwarmNnGpu, WrightFisherGpu, StencilCooperationGpu, HillGateGpu |
| Tensor API | 30+ | matmul, transpose, add, sub, mul, sigmoid, tanh, gelu, softmax, conv2d, maxpool2d, mean, sum, etc. |
| CPU primitives | 18 | variance, pearson, eigh, solve, cholesky, lu, svd, rk45, chi_squared, etc. |
| Shaders consumed | 13 upstream + 8 local | All validated |
| `eigh_f64` consumers | 9 modules | 4 core + 5 baseCamp |

### New Absorption Candidates from gpu_dispatch/cpu_fallback.rs

These CPU fallback functions are now cleanly separated and could be absorbed
as `barracuda::stats::*` or `barracuda::bio::*` CPU reference implementations:

| Function | Signature | Potential Location |
|----------|-----------|-------------------|
| `variance(data: &[f64]) -> f64` | Population variance | `barracuda::stats::variance` (exists — validate parity) |
| `pearson(x: &[f64], y: &[f64]) -> f64` | Pearson correlation | `barracuda::stats::pearson_correlation` (exists — validate parity) |
| `chi_squared(observed: &[f64], expected: &[f64]) -> f64` | Chi-squared statistic | `barracuda::stats::chi_squared` (candidate) |
| `hmm_backward_step(...)` | Single HMM backward step | `barracuda::bio::hmm_backward_step` (candidate) |
| `hmm_viterbi_step(...)` | Single HMM Viterbi step | `barracuda::bio::hmm_viterbi_step` (candidate) |
| `replicator_step(...)` | Replicator dynamics step | `barracuda::bio::replicator_step` (candidate) |

### API Gaps (Updated — 2 Closed by ToadStool S52)

| Gap | Impact | Status |
|-----|--------|--------|
| ~~`argmax_dim()`~~ | Viterbi needs indices | **CLOSED** — `Tensor::argmax_dim(axis)` now available |
| `pow_scalar(n)` | Hill activation `x^n` | `exp(n * ln(x))` pipeline |
| ~~`softmax_dim(axis)`~~ | Row-wise attention | **CLOSED** — `Tensor::softmax_dim(axis)` now available |
| `div(other)` elementwise | Ratio computation | Reciprocal + `mul` |
| Native `ops::mha` | Retire `evolved::mha` | Projection shaders hang on RTX 4070 |

---

## Part 3: Cross-Spring Dependencies

### From hotSpring

| Primitive | Usage |
|-----------|-------|
| Anderson localization (IPR, level spacing) | baseCamp nS-01 through nS-05 |
| Boltzmann sampling | baseCamp nS-03 |
| RK45 ODE integration | Regulatory, signal, game theory |
| hotSpring validation pattern | `ValidationHarness`, centralized tolerances, exit 0/1 |

### From wetSpring

| Primitive | Usage |
|-----------|-------|
| HMM phylogenetics | baseCamp nS-04 |
| QS cooperation dynamics | baseCamp nS-05 |
| `HmmBatchForwardF64` | Primary HMM GPU path (evolved version retired) |

### To Other Springs

neuralSpring contributes the following patterns that other Springs may adopt:

| Pattern | Description |
|---------|------------|
| `gpu_or_cpu` dispatch | Capability-based GPU/CPU routing via closure pattern |
| `exit_no_gpu()` | Unified GPU unavailability handling for CI |
| `baseline_path()` | CARGO_MANIFEST_DIR-relative baseline resolution |
| `tolerances::ZERO_DETECTION` | Centralized zero-detection constant (1e-14) |

---

## Part 4: What ToadStool/BarraCUDA Team Should Know

### 4.1 Absorption Opportunities

**Priority 1 — General-Purpose Primitives:**

| Primitive | From | Generalized Form | Target |
|-----------|------|-------------------|--------|
| `graph_laplacian(adjacency)` | `agent_coordination.rs` | `D - A` | `ops::linalg` |
| `effective_rank(eigenvalues)` | `neural_pgm.rs` | Entropy-based rank | `ops::linalg` |
| `empirical_spectral_density(eigenvalues, bins)` | `weight_spectral.rs` | Histogram | `ops::stats` |
| `numerical_hessian(f, x, h)` | `loss_landscape.rs` | Central FD Hessian | `ops::numerical` |
| `level_spacing_ratio(eigenvalues)` | `weight_spectral.rs` | GOE/Poisson stat | `ops::stats` |

**Priority 2 — Testing Patterns:**

| Pattern | Description | Why Absorb |
|---------|------------|------------|
| `gpu_or_cpu` dispatch | `gpu_or_cpu(name, gpu_fn, cpu_fn)` closure | All Springs use this pattern |
| `exit_no_gpu()` | `REQUIRE_GPU=1` → exit 1, else graceful skip | CI standardization |
| `baseline_path(rel)` | `CARGO_MANIFEST_DIR`-relative data paths | All Springs need this |
| `require!` macro | `.expect()` replacement for validation binaries | Reusable across Springs |

### 4.2 GPU Shader Candidates from baseCamp

| Function | GPU Approach | Priority |
|----------|-------------|----------|
| `weight_to_hamiltonian` | Tensor matmul (`W^T * W`) | High (nS-01 bottleneck) |
| `numerical_hessian` | GPU parallel finite differences | High (nS-03 bottleneck) |
| `belief_propagation_chain` | GPU batch GEMV (HMM pattern) | Medium |
| `interaction_graph` | GPU pairwise distance | Medium |
| `boltzmann_sampling` | GPU parallel chain MCMC | Low |

### 4.3 Known Issues (Unchanged)

| # | Issue | Status |
|---|-------|--------|
| S-14 | Naive matmul hang (small square, N < 32) | Workaround (A×B^T) |
| S-15 | Matmul hang when elements ≤ 0.1 magnitude | Root-caused (driver bug), workaround (data ≥ 0.5) |
| S-16 | Transpose dispatch `optimal_workgroup_size` vs tile | **FIXED** |

### 4.4 Dependency Health

All dependencies are pure Rust. The only `-sys` crates in the tree are:
- `linux-raw-sys` — kernel constant definitions (via wgpu/rustix)
- `renderdoc-sys` — optional wgpu debug integration

No C compilation, no `openssl-sys`, no `ring`. Clean cross-compilation path
for ecoBin targets.

### 4.5 What Worked Well

1. **Typed op migration** (Session 48): All 12 typed ops use f64, validated on
   RTX 4070 + TITAN V. The f32→f64 alignment is complete.
2. **`gpu_or_cpu` pattern**: Proves capability-based dispatch works. ToadStool
   should consider absorbing as `barracuda::dispatch()`.
3. **Centralized tolerances**: 90+ named constants. Zero inline magic numbers
   in validation binaries.
4. **hotSpring validation pattern**: `ValidationHarness` + exit 0/1 + centralized
   tolerances is battle-tested across 138 binaries.

---

## Part 5: Documentation Updates

| Document | Change |
|----------|--------|
| `README.md` | 459 tests, `gpu_dispatch/` module, coverage 92.9% |
| `EVOLUTION_READINESS.md` | 459 tests, 92.9% coverage |
| `CONTROL_EXPERIMENT_STATUS.md` | 459 tests, 138 binaries, 36 modules |
| `DEPRECATION_MIGRATION.md` | Coverage 92.9%, test count corrected |
| `specs/BARRACUDA_USAGE.md` | Session 51 section added |
| `specs/EVOLUTION_MAPPING.md` | 459 tests |
| `specs/TOADSTOOL_HANDOFF.md` | 459 tests |
| `specs/README.md` | 459 tests |
| `experiments/README.md` | Experiment 020 added, journal index updated |
| `metalForge/ABSORPTION_MANIFEST.md` | 459 tests |
| `whitePaper/STUDY.md` | 459 tests |
| `whitePaper/METHODOLOGY.md` | 459 tests |
| `whitePaper/README.md` | 459 tests |
| `whitePaper/baseCamp/extensions.md` | 459 tests |

---

## Cumulative neuralSpring Status

| Metric | Value |
|--------|-------|
| Library modules | 36 + 2 evolved + gpu_ops/ + gpu_dispatch/ |
| Validation binaries | 138 + validate_all + 6 bench |
| Lib tests | 459 |
| Integration tests | 9 |
| Doc tests | 9 |
| Forge tests | 26 |
| Line coverage | 92.9% |
| Clippy (pedantic + nursery) | 0 warnings |
| Doc warnings | 0 |
| Production `.unwrap()`/`.expect()` | 0 |
| `unsafe` blocks | 0 |
| Python baselines | 206/206 PASS |
| BarraCUDA checks | 272/272 PASS |
| bC coverage | 24/25 (96%) |
| gT coverage | 23/25 (92%) |
| mF coverage | 15/15 (100% applicable) |
| gP coverage | 15/15 (100% applicable) |
| xD coverage | 15/15 (100%) |
| Open data | 25/25 papers + 5 baseCamp |
| License | AGPL-3.0-or-later |

---

## Session 52 Addendum — ToadStool Sync & Cross-Spring Benchmarking

### ToadStool Sync (16 commits, `b41ee5f4` → `9abd6857`)

6 shaders confirmed absorbed upstream:

| Shader | Upstream API |
|--------|-------------|
| `xoshiro128ss.wgsl` | `barracuda::ops::prng_xoshiro` |
| `logsumexp_reduce.wgsl` | `barracuda::ops::LogsumexpWgsl` |
| `stencil_cooperation.wgsl` | `barracuda::StencilCooperationGpu` |
| `wright_fisher_step.wgsl` | `barracuda::WrightFisherGpu` |
| `rk45_adaptive.wgsl` | `barracuda::ops::rk45_adaptive` |
| `swarm_nn_scores.wgsl` | `barracuda::SwarmNnGpu` |

API gaps closed: `argmax_dim`, `softmax_dim`. `level_spacing_ratio` rewired to upstream.
Only `head_split` + `head_concat` remain local (MHA S-03b workaround).

### Cross-Spring Benchmark (RTX 4070, Vulkan, `--release`)

| Op | Origin | µs |
|----|--------|----|
| BatchFitnessGpu 1024×64 | neuralSpring | 1,337 |
| PairwiseL2Gpu 128×16 | neuralSpring | 1,542 |
| SpatialPayoffGpu 32×32 | neuralSpring | 1,450 |
| PairwiseHammingGpu 64×100 | neuralSpring | 1,682 |
| BatchIprGpu 32×64 | neuralSpring | 2,027 |
| HmmBatchForwardF64 4s×50t×32b | wetSpring | 2,141 |
| BatchedEighGpu 12×12×40 | hotSpring | 6,629 |

### Validation (Session 52)

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets` | 0 warnings |
| `cargo doc --no-deps` | 0 warnings (146 pages) |
| `cargo test --lib` | 459 PASS |
| `cargo llvm-cov --lib` | 92.89% line coverage |
| `validate_all` | 137/138 PASS (1 pre-existing logsumexp driver) |

---

## Session 52b Addendum — S-17 HillGate f64 `pow()` Fix

### Root Cause

`hill_gate_f64.wgsl` uses native WGSL `pow(f64, f64)` which fails on:
- **RTX 4070** (Ada Lovelace, proprietary NVVM): compilation failure → device lost
- **TITAN V** (NVK, open-source NAK): assertion `alu.def.bit_size() == 32`

`compile_shader_f64` patches `exp()`/`log()` to polyfills but misses `pow()`.

### Fix

Replace `pow(` → `pow_f64(` in shader source. `inject_missing_math_f64`
auto-injects the `pow_f64` polyfill (uses `exp_f64(n * log_f64(base))`).

### Validation

| Adapter | Max Diff | Result |
|---------|----------|--------|
| RTX 4070 (Vulkan, proprietary) | 1.11e-16 | 18/18 PASS |
| TITAN V (NVK, open-source) | 2.22e-16 | 18/18 PASS |

`validate_gpu_signal` upgraded from SKIP → 9/9 PASS on both GPUs.
Also fixed pre-existing f32/f64 buffer mismatch in original validator.

### ToadStool Action

One-line fix in `barracuda/src/shaders/precision/mod.rs`:

```rust
fn patch_exp_log_in_code(code: &str) -> String {
    code.replace("exp(", "exp_f64(")
        .replace("log(", "log_f64(")
        .replace("pow(", "pow_f64(")  // S-17
}
```

Also fix `hill_f64.wgsl` (element-wise Hill) — same native `pow(f64)` pattern.

---

*neuralSpring V19 handoff — Sessions 51–52b. S-17 HillGate f64 fix + code quality evolution + ToadStool sync + cross-spring benchmarking. Only 2 local shaders remain. 459 lib tests, 92.89% coverage, 137/138 validators PASS. gpu_signal upgraded from SKIP → 9/9 PASS. All 25 papers on open data. Zero debt.*
