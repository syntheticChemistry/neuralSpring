# neuralSpring V17 — Session 49: Deep Debt Audit + ToadStool Absorption Handoff

**Date**: February 23, 2026
**ToadStool HEAD**: `b41ee5f4`
**neuralSpring Session**: 49 (deep debt audit + documentation refresh)
**Previous**: V16 (Session 48 — mass typed-op rewiring)

---

## Part 1: What Changed in Session 49

Session 49 is a code quality hardening pass — no new validation checks, no new
binaries. All existing checks confirmed passing.

### 1.1 `gpu_dispatch.rs` — DRY Dispatch Refactoring

Created a private `gpu_or_cpu` helper centralising the "try GPU, log-and-fallback"
pattern used by all 25 dispatch methods:

```rust
fn gpu_or_cpu<T>(
    &self,
    op: &str,
    gpu_fn: impl FnOnce(&Arc<WgpuDevice>) -> Result<T, String>,
    cpu_fn: impl FnOnce() -> T,
) -> T
```

Each dispatch method is now 5 lines (closure pair) instead of 8 (if-let-match-fallback).
The `Dispatcher` remains the single capability-based routing point.

**ToadStool relevance**: This pattern proves `WgpuDevice`-based dispatch works.
Consider absorbing a `barracuda::dispatch::gpu_or_cpu` primitive.

### 1.2 `exit_no_gpu()` — CI-Fidelity Hardening

Unified 79 validation/bench binaries to use `validation::exit_no_gpu()`:

```rust
pub fn exit_no_gpu() -> ! {
    if gpu_required() { exit(1); }   // NEURALSPRING_REQUIRE_GPU=1
    exit(0);                          // graceful skip
}
```

Before: 3 syntactic patterns across 79 files (`let...else`, `match`, inline
`eprintln! + exit(0)`). After: one call, environment-driven policy.

**ToadStool relevance**: All Springs need this pattern. Candidate for
`barracuda::testing::require_gpu()`.

### 1.3 `baseline_path()` — Data Resolution

Replaced 4 hardcoded `concat!(env!("CARGO_MANIFEST_DIR"), ...)` paths with:

```rust
pub fn baseline_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}
```

**ToadStool relevance**: Testing utility candidate for `barracuda::testing`.

### 1.4 Documentation Refresh

- `EVOLUTION_MAPPING.md`: Corrected "stub" labels (mlp_forward exists in
  `pinn.rs`/`deeponet.rs`, not as stubs).
- Created `whitePaper/baseCamp/` (5 per-faculty briefings, following wetSpring pattern).
- Experiment 018 journal entry documenting all debt audit changes.

---

## Part 2: Validation Results

No new checks — all existing checks confirmed passing.

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets -- -D warnings` | PASS (0 warnings) |
| `cargo doc --no-deps` | PASS (0 warnings) |
| `cargo test` | PASS (374 lib + 9 integration + 9 doc-tests) |
| Max file size | 965 lines (under 1000 wateringHole limit) |
| `unsafe` blocks | 0 (`forbid` enforced) |
| TODO/FIXME/MOCK/STUB | 0 in src/ |
| Hardcoded paths | 0 (all via `baseline_path`) |
| `.unwrap()` in non-test | 0 |

---

## Part 3: ToadStool Absorption Recommendations

### 3.1 Primitives neuralSpring Offers for Absorption

| Primitive | Source | Consumer Count | Recommended ToadStool Module |
|-----------|--------|----------------|------------------------------|
| `gpu_or_cpu` dispatch | `gpu_dispatch.rs` | 25 methods | `barracuda::dispatch::gpu_or_cpu` |
| `exit_no_gpu()` / `gpu_required()` | `validation.rs` | 79 binaries | `barracuda::testing::require_gpu` |
| `baseline_path()` | `validation.rs` | 4 binaries | `barracuda::testing::baseline_path` |
| `require!` macro | `validation.rs` | 133 binaries | `barracuda::testing::require` |
| `ValidationHarness` | `validation.rs` | 133 binaries | `barracuda::testing::ValidationHarness` |
| Named tolerances registry | `tolerances/registry.rs` | 90+ constants | `barracuda::testing::tolerances` |
| Shannon entropy | `primitives.rs` | 8 tests | `barracuda::stats::entropy` |
| Hill activation | `primitives.rs` | 3 modules | `barracuda::numerical::hill` |
| LOG_GUARD / DIVISION_GUARD | `primitives.rs` | 5 modules | `barracuda::numerical::constants` |

### 3.2 WGSL Shaders Still Local (Absorption Ready)

| Shader | Lines | Domain | Recommended ToadStool Path |
|--------|-------|--------|---------------------------|
| `logsumexp_reduce.wgsl` | 45 | HMM/phylo batched logsumexp | `barracuda::ops::reduce` |
| `stencil_cooperation.wgsl` | 52 | Fermi imitation game theory | `barracuda::ops::stencil` |
| `rk45_adaptive.wgsl` | 68 | Dormand-Prince RK45 ODE | `barracuda::ops::ode` |
| `wright_fisher_step.wgsl` | 55 | Drift+selection+xoshiro | `barracuda::ops::popgen` |
| `head_split.wgsl` | 30 | MHA data movement | `barracuda::ops::mha` |
| `head_concat.wgsl` | 30 | MHA data movement | `barracuda::ops::mha` |
| `xoshiro128ss.wgsl` | 35 | GPU PRNG | `barracuda::ops::prng` |
| `swarm_nn_scores.wgsl` | 40 | Swarm robotics scoring | New: `barracuda::ops::bio::swarm` |

### 3.3 API Gaps for ToadStool to Address

| Gap | Impact | Workaround | Priority |
|-----|--------|------------|----------|
| No `argmax_dim()` | Viterbi needs indices | CPU argmax after `max_dim` readback | P0 |
| No `pow_scalar(n)` | Hill activation `x^n` | `exp(n * ln(x))` | P1 |
| No `softmax_dim(axis)` | Row-wise attention softmax | `ScaledDotProductAttention` | P1 |
| No `div(other)` | Elementwise ratio | Uploaded reciprocal + `mul` | P1 |
| `Tensor::mean()` bug | Entry point + double-divide | Local fix, pending upstream merge | P0 |
| HillGateGpu f64 RTX 4070 | Driver limitation | f32 path; f64 graceful skip | P2 |
| S-15 matmul magnitude hang | Elements ≤ 0.1 magnitude | Data ≥ 0.5 | P1 (driver investigation) |

### 3.4 Ownership Model Documentation Request

neuralSpring discovered that Tensor methods have inconsistent ownership:

- **Consuming** (`self`): `matmul`, `softmax`, `sigmoid`, `gelu_wgsl`, `log_wgsl`, `exp_wgsl`, `sqrt_wgsl`, `broadcast`
- **Borrowing** (`&self`): `transpose`, `add`, `sub`, `mul`, `sum`, `mean`, `max`, `norm`, `mul_scalar`, `add_scalar`, `div_scalar`, `sum_dim`, `mean_dim`, `max_dim`, `min_dim`, `reshape`, `to_vec`

This needs documentation in BarraCUDA's Tensor API docs so all Springs can
plan ownership chains correctly.

---

## Part 4: Cross-Spring Patterns for ToadStool

### 4.1 The Write → Absorb → Lean Cycle

neuralSpring is now in the **Lean** phase for most primitives:

| Phase | Primitives | Status |
|-------|-----------|--------|
| **Lean** (using upstream directly) | matmul, transpose, softmax, sigmoid, tanh, gelu, conv2d, maxpool2d, relu, elu, leaky_relu, layer_norm, log_softmax, FFT, eigh, solve, cholesky, lu, svd, rk45, chi_squared, variance, pearson_correlation, 12 typed bio ops | 90%+ |
| **Absorb** (local, ready for upstream) | 8 WGSL shaders, ValidationHarness, tolerances registry, `gpu_or_cpu` dispatch | 8% |
| **Write** (local, needs design work) | mixed-hardware dispatch, NPU reservoir path | 2% |

### 4.2 What hotSpring and wetSpring Should Know

- **`gpu_or_cpu` pattern**: Reusable across all Springs. Same dispatch model
  works for physics (hotSpring) and biology (wetSpring).
- **`exit_no_gpu` CI hardening**: Set `NEURALSPRING_REQUIRE_GPU=1` in CI to
  catch silent GPU-skip regressions. Proposed as cross-Spring standard.
- **Tolerance naming convention**: All tolerances use `MODULE_METRIC_TYPE`
  naming (e.g., `INTROGRESSION_FRACTION_ABS`, `ANDERSON_IPR_REL`). Same
  convention recommended for hotSpring/wetSpring.
- **baseCamp briefings**: Per-faculty briefings link papers → modules →
  validation tiers → ToadStool primitives. Useful pattern for thesis writing.

### 4.3 metalForge Mixed-Hardware Status

| Component | Purpose | Maturity |
|-----------|---------|----------|
| `mixed.rs` | MixedSubstrate enum + TransferCost model | Design validated, not production |
| `pcie_bridge.rs` | PcieBridge + P2P detection | Infrastructure ready |
| Heuristics | `logsumexp_substrate`, `stochastic_substrate` | Validated in dispatch.rs |
| Multi-GPU validation | RTX 4070 + TITAN V NVK bit-identical | 133/133 PASS |

Ready for metalForge evolution when ToadStool provides unified multi-device
allocation.

---

## Part 5: Cumulative Status

| Metric | Value |
|--------|-------|
| Papers reproduced | 25/25 (ALL COMPLETE) |
| Python checks | 206/206 PASS |
| Rust+GPU checks | 1600+ PASS |
| Grand total | 1800+ checks |
| Library tests | 374 lib + 9 integration + 9 doc-tests |
| Validation binaries | 133 |
| WGSL shaders | 21 (13 upstream absorbed, 8 local) |
| Typed BarraCUDA ops | 12 (all f64 aligned) |
| Tensor API methods | 30+ (all validated) |
| CPU primitives | 18 (all validated) |
| GPU promotion | 38 ops (~90% of production math) |
| Multi-GPU | Bit-identical (RTX 4070 + TITAN V NVK) |
| Code debt | 0 (zero TODO/FIXME/MOCK/STUB/hardcoded paths/unsafe) |
| Clippy | 0 warnings (pedantic + nursery) |
| Doc | 0 warnings |
| Line coverage | 92.7% |
| Max file | 965 lines |

## Part 6: Handoff Chain

| Version | Session | Focus |
|---------|---------|-------|
| V1–V5 | 39 | Initial evolution, upstream fixes, forge, absorption |
| V6–V8 | 40 | Barracuda GPU, evolution sync |
| V9–V11 | 41–42 | ToadStool BarraCUDA sync, deep audit |
| V12 | 43 | Session 43 buildouts |
| V13 | 44 | Multi-GPU + benchmarks |
| V14 | 46 | Pure GPU Phase B |
| V15 | 47 | Typed op migration |
| V16 | 48 | Mass typed-op rewiring (28 binaries) |
| **V17** | **49** | **Deep debt audit + documentation + ToadStool absorption handoff** |

---

## Part 7: Files Changed This Session

### Modified
- `src/validation.rs` — added `gpu_required()`, `exit_no_gpu()`, `baseline_path()`
- `src/gpu_dispatch.rs` — added `gpu_or_cpu` helper, refactored 25 methods
- `src/bin/validate_barracuda_ml_inference.rs` — baseline_path + exit_no_gpu
- `src/bin/bench_transformer_block.rs` — baseline_path
- `src/bin/bench_mlp_inference.rs` — baseline_path
- `src/bin/validate_barracuda_tensor_f64.rs` — let-else fix
- 76 additional `src/bin/*.rs` — exit_no_gpu standardisation
- `specs/EVOLUTION_MAPPING.md` — corrected stub labels
- `specs/TOADSTOOL_HANDOFF.md` — updated canonical handoff pointer
- `specs/BARRACUDA_USAGE.md` — Session 49 section
- `CONTROL_EXPERIMENT_STATUS.md` — Session 49 quality gates
- `README.md` — updated footer
- `whitePaper/README.md` — added baseCamp reference, updated footer
- `experiments/README.md` — Experiment 018 journal entry

### Created
- `whitePaper/baseCamp/README.md` — per-faculty briefings overview
- `whitePaper/baseCamp/dolson.md` — evolutionary computation (011–015)
- `whitePaper/baseCamp/liu.md` — phylogenetics / HMM (016–018)
- `whitePaper/baseCamp/waters.md` — microbial cooperation (019–021)
- `whitePaper/baseCamp/kachkovskiy.md` — spectral theory (022–023)
- `whitePaper/baseCamp/anderson.md` — population genetics (024–025)
- `wateringHole/handoffs/NEURALSPRING_V17_SESSION49_HANDOFF_FEB23_2026.md` (this file)

### Archived
- `wateringHole/handoffs/NEURALSPRING_V16_SESSION48_HANDOFF_FEB23_2026.md` → archive/

---

*neuralSpring V17 — Session 49. Deep debt audit, zero debt, zero warnings, documentation refresh, ToadStool absorption handoff. 1800+ checks, 25 papers, ~90% GPU.*
