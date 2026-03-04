<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring → ToadStool/barraCuda Handoff V80 — Deep Debt Audit + CI Hardening + BarraCUDA Evolution Review

**Date**: March 3, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/barraCuda team
**License**: AGPL-3.0-or-later
**Covers**: Session 120 — comprehensive audit against wateringHole standards, CI hardening, idiomatic evolution
**Supersedes**: V79 (S119 deep lint evolution + shared validation helpers)
**barraCuda**: v0.3.1 standalone (`../barraCuda/crates/barracuda`)

---

## Executive Summary

- **Complete codebase audit**: Every gate from wateringHole standards verified — lint, fmt, clippy (pedantic+nursery), doc, test coverage, file size, license, provenance, safety, zero-copy, data sources.
- **Zero `#[allow(` remaining**: The last 6 test-module `#[allow(` converted to `#[expect(` or removed (unfulfilled). 2 unnecessary suppressions in `tests_cpu.rs`/`tests_gpu.rs` eliminated. The entire codebase now uses precise `#[expect(` with documented reasons.
- **CI hardened to all-features**: `.github/workflows/rust.yml` clippy step upgraded to `--all-targets --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery`. Feature-gated `rpc_service` (under `primal`) is now linted in CI. Makefile and justfile `lint-rust` aligned with `--all-features` and `RUSTDOCFLAGS="-D warnings"`.
- **Production suboptimal_flops fixed**: `anderson_localization.rs` — `0.5 + f64::from(i) * 1.5` → `f64::from(i).mul_add(1.5, 0.5)` for fused multiply-add precision.
- **18 test warnings resolved**: Targeted `#[expect(` with reason strings across 6 test modules: `msa.rs`, `pairformer.rs`, `triangle.rs`, `geography.rs`, `backbone.rs`, `gpu.rs`, `tests_cpu_basecamp.rs`, `tests_ops.rs`.
- **Audit confirms**: 869/869 lib tests, 212/212 validate_all, 0 clippy warnings, 0 doc warnings, 0 fmt issues, 0 unsafe, 0 files >1000 lines, 337/337 SPDX headers, 41 provenance records, 139+ named tolerances, all quality gates pass.

---

## Part 1: BarraCUDA Usage Inventory & Evolution Review

### Current API Surface (v0.3.1)

| Metric | Value |
|--------|-------|
| Files with barracuda imports | ~117 |
| Submodules exercised | 25+ (device, tensor, ops::bio, dispatch, stats, linalg, numerical, special, spectral, nautilus, staging, pipeline, unified_math, unified_hardware, tolerances, ...) |
| Upstream rewires (local → delegate) | 44 |
| WGSL shaders absorbed upstream | 21/21 |
| Local shaders remaining | 4 (`xoshiro128ss`, `swarm_nn_scores`, `head_split`, `head_concat`) |
| metalForge WGSL shaders | 42 |
| CPU→GPU dispatch ops | 47 (~97%, 7 domain files) |
| Named tolerances consumed | 139+ |
| Shortcomings resolved | 17/17 (S-01 through S-17) |

### What neuralSpring Exercises That Other Springs Don't

neuralSpring is uniquely positioned in the ecosystem as the **ML/neuroevolution validation layer**. It exercises barraCuda APIs that other springs touch lightly or not at all:

| API Area | neuralSpring Usage | Other Springs |
|----------|-------------------|---------------|
| `ops::bio::hmm_*` | Forward/backward/Viterbi (3 papers) | wetSpring (16S classification) |
| `ops::bio::fst_*` | Population genetics (2 papers) | — |
| `nautilus` | ESN reservoir, evolutionary computation (4 papers) | hotSpring (brain), airSpring (brain) |
| `pipeline::StatefulPipeline` | HMM chains, ODE integrators | — |
| `tensor::Tensor` matmul paths | MLP, Transformer, LSTM, Pairformer, TriangleAttention | hotSpring (smaller models) |
| coralForge df64 | MSA attention, pair representation, structure module | — |
| `staging::StagingBuffer` | Large tensor readback (protein structures) | — |

### 4 Local Shaders — Absorption Recommendation

| Shader | Purpose | Absorption Path |
|--------|---------|-----------------|
| `xoshiro128ss.wgsl` | GPU-resident PRNG for stochastic algorithms | → `barracuda::random::xoshiro128ss` (general utility) |
| `swarm_nn_scores.wgsl` | Fused swarm NN forward + fitness scoring | Specialized — keep local or generalize as `batched_mlp_score` |
| `head_split.wgsl` | Multi-head attention head splitting | → `barracuda::ops::attention::head_split` (used by all attention) |
| `head_concat.wgsl` | Multi-head attention head concatenation | → `barracuda::ops::attention::head_concat` (used by all attention) |

`head_split`/`head_concat` are the strongest absorption candidates — every transformer/attention model needs them, and they're currently duplicated between neuralSpring and any other spring doing attention.

### Shared Validation Helpers — Absorption Candidates (from V79)

| Helper | Signature | Cross-Spring Value |
|--------|-----------|-------------------|
| `max_abs_diff_f64` | `(a: &[f64], b: &[f64]) -> f64` | Universal — every spring needs this |
| `bench_once` | `<F: FnOnce() -> T, T>(label: &str, f: F) -> (T, f64)` | Common in validators |
| `bench_median` | `<F: FnMut()>(warmup: usize, iters: usize, f: F) -> f64` | Standard benchmarking |
| `median_duration_us` | `(times: &mut [Duration]) -> f64` | Statistics utility |

### Evolution Blockers (unchanged from V79)

| Blocker | Detail | Owner |
|---------|--------|-------|
| `StatefulPipeline` batching | HMM chain, ODE loops — reduce CPU round-trips | barraCuda P2 |
| Flash Attention | coralForge MSA/pair attention at scale | barraCuda P2 |
| `UnidirectionalPipeline` streaming | Streaming fitness eval for EA | barraCuda P2 |
| NestGate data acquisition | NCBI/PDB/HuggingFace pipeline for coralForge | NestGate |

---

## Part 2: CI Hardening Details

### Before S120

```yaml
# rust.yml
- name: Clippy (pedantic)
  run: cargo clippy -- -D warnings
```

This missed: `--all-features` (so `primal` feature-gated `rpc_service` was never linted), `--all-targets` (so bin-only code paths could drift), and pedantic/nursery warning levels.

### After S120

```yaml
- name: Clippy (pedantic + nursery, all features)
  run: cargo clippy --all-targets --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery
```

Makefile and justfile aligned identically, with `RUSTDOCFLAGS="-D warnings"` for doc checks.

### Impact for barraCuda

If barraCuda's CI doesn't already run `--all-features`, this is worth adopting. Feature-gated code accumulates lint debt silently when not checked in CI.

---

## Part 3: Idiomatic Rust Patterns Confirmed

### `#[expect(` Over `#[allow(`

neuralSpring has completed the full migration. Pattern summary for adoption:

- **Production code**: `#[expect(clippy::lint, reason = "why")]` — compiler warns when suppression becomes unnecessary
- **Test code where lint doesn't fire**: `#[allow(clippy::expect_used)]` (not `#[expect(`) — `expect_used`/`unwrap_used` don't fire in `#[cfg(test)]` contexts
- **Cross-compilation imports**: `#[allow(clippy::wildcard_imports)]` for `use super::*` in test modules that compile in both lib and test contexts

### `mul_add` for FMA

`f64::from(i).mul_add(1.5, 0.5)` instead of `0.5 + f64::from(i) * 1.5`. The compiler may fuse this into a hardware FMA instruction, improving both precision and performance. `clippy::suboptimal_flops` catches these.

### Zero `unwrap()`/`expect()` in Library Code

All 869 lib tests use `unwrap()`/`expect()`, but zero production library code does. This is enforced by `clippy::unwrap_used` and `clippy::expect_used` at the crate level.

---

## Part 4: Quality State (S120)

| Gate | Value |
|------|-------|
| `cargo fmt --check` | Clean |
| `cargo check --all-targets --all-features` | Clean |
| `cargo clippy` (pedantic+nursery, all-features) | **0 warnings** |
| `cargo doc --no-deps` | Clean (0 warnings) |
| `cargo test --lib` | **869/869 PASS** |
| `validate_all` | **212/212 PASS** |
| `#[allow(` in entire codebase | **0** |
| `#[expect(` with reasons | **All** |
| unsafe code | **0** |
| Files > 1000 lines | **0** (max 953) |
| SPDX-License-Identifier headers | **337/337** |
| TODO/FIXME markers | **0** |
| Production mocks | **0** |
| Hardcoded paths | **0** |
| Provenance records | **41** (full Python trace) |
| Named tolerances | **139+** (centralized, documented derivations) |

---

## Part 5: Evolution Readiness Summary

### Tier A — Ready for GPU shader promotion (no blockers)

All 15 Phase 0++ papers + 5 WDM surrogates + 6 baseCamp sub-theses. 47/48 dispatch ops are GPU-resident via `gpu_dispatch::Dispatcher`. The single remaining CPU-only path is `hmm_viterbi` (argmax chain, needs `StatefulPipeline`).

### Tier B — Needs upstream API evolution

| Module | Blocker | barraCuda Ticket |
|--------|---------|-----------------|
| `coral_forge/msa.rs` | Flash Attention for O(N²) MSA at production scale | P2 |
| `coral_forge/pairformer.rs` | Fused TriMul + attention pipeline | P2 |
| HMM Viterbi chain | `StatefulPipeline` for GPU-resident argmax chain | P2 |
| ODE integrators | `StatefulPipeline` for RK4/Euler loops | P2 |

### Tier C — New science, new shaders needed

| Module | Need |
|--------|------|
| coralForge structure module | IPA (Invariant Point Attention) — requires SE(3)-equivariant ops |
| Protein diffusion | Denoising diffusion on SE(3) — new shader class |
| Field genomics (NestGate) | Nanopore signal → basecall GPU pipeline |

---

## Action Items for ToadStool/barraCuda

1. **Absorb `head_split`/`head_concat` shaders** — universal attention primitives, eliminates cross-spring duplication
2. **Absorb shared validation helpers** — `max_abs_diff_f64`, `bench_once`, `bench_median`, `median_duration_us` into `barracuda::validation`
3. **Adopt `--all-features` in barraCuda CI** if not already present — prevents feature-gated lint debt
4. **Consider `#[expect(` migration** — neuralSpring's experience: ~477 unfulfilled suppressions discovered during migration (widespread over-suppression)
5. **Pipeline batching (P2)** — `StatefulPipeline` and `UnidirectionalPipeline` unblock Tier B modules

---

*V80 — neuralSpring Session 120 (March 3, 2026)*
