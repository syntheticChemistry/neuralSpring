# neuralSpring V119 → barraCuda / toadStool Deep Debt Execution Handoff

**Date**: March 18, 2026
**From**: neuralSpring Session 168, V119
**To**: barraCuda / toadStool teams
**License**: AGPL-3.0-or-later
**Covers**: V118–V119, Session 168
**Supersedes**: V118 Deep Audit + Ecosystem Evolution Handoff

## Executive Summary

Session 168 executed on all findings from the comprehensive S167 audit.
Zero clippy warnings workspace-wide (was 66), provenance mapping fixed
(was non-functional for 49+ experiments), largest file refactored below
500 LOC, `TensorSession`/`StatefulPipeline` wired to `Dispatcher`,
8 new property tests, CI hardened workspace-wide.

**Key metrics:** 1312 tests (1164 lib + 73 playGround + 75 forge),
267 binaries, 68 modules, 225 named tolerances, 14 proptest invariants,
zero clippy (pedantic+nursery), zero unsafe, zero C deps, zero mocks in
production, zero `#[allow()]` workspace-wide.

---

## Part 1 — What Changed (Session 168)

### 1.1 Provenance Fix: `expected_source()` (Critical)

`BaselineProvenance::expected_source()` was matching on `self.label` —
short identifiers like `"surrogate"` — but the actual label values are
full descriptive strings like `"Surrogate Benchmark Functions (Rastrigin,
Rosenbrock, Ackley)"`. The match arms never matched; the method always
returned `""`.

**Fix**: Switched to `self.script` (the Python baseline path), which is
stable and concise. All 49+ experiments now correctly map to their
expected data source (JSON file, `"inline analytical"`, etc.).

| Before | After |
|--------|-------|
| 9 match arms on label strings | 49+ match arms on script paths |
| Always returned `""` | Returns correct source for all experiments |

### 1.2 Smart Refactor: `ipc_client.rs` (885 → 448 LOC)

Extracted biomeOS 5-tier socket resolution and capability-based primal
discovery into `playGround/src/discovery.rs` (439 LOC). The split follows
natural responsibility boundaries:

- `ipc_client.rs` (448 LOC): JSON-RPC protocol, error types, call mechanics
- `discovery.rs` (439 LOC): socket resolution, capability discovery, primal discovery

Backward-compatible re-exports maintained.

### 1.3 TensorSession / StatefulPipeline Wiring

`Dispatcher` now exposes factory methods for barraCuda's fused pipeline APIs:

```rust
// Fused multi-op (matmul → GELU → softmax → layer_norm in one encoder)
pub fn tensor_session(&self) -> Option<TensorSession>

// Iterative GPU kernels (ODE loops, eigensolvers, training)
pub fn stateful_pipeline(&self, config: StatefulConfig) -> Option<StatefulPipeline>
```

Both return `None` when no GPU is available, enabling graceful fallback.

### 1.4 Clippy Zero: 66 Warnings → 0

| Lint | Count | Resolution |
|------|-------|------------|
| `unwrap_used` / `expect_used` in tests | 48 | `#[expect(reason)]` on test modules |
| `float_cmp` in tests | 6 | `#[expect(reason)]` where IEEE 754 exact |
| `approx_constant` | 1 | `3.14` → `7.89` in fuzz test |
| `suboptimal_flops` | 4 | `mul_add()` |
| `cast_possible_truncation` | 1 | `u32::try_from()` |
| `identical_match_arms` | 1 | Combined provenance match arms |
| `items_after_statements` | 1 | `#[expect(reason)]` with justification |
| Doc warning (missing backticks) | 1 | Added backticks |

### 1.5 Property Test Expansion (6 → 14 invariants)

| Test | Module | Invariant |
|------|--------|-----------|
| `r_squared_perfect_prediction_is_one` | metrics | R² = 1.0 when prediction = observation |
| `rmse_always_nonnegative` | metrics | RMSE ≥ 0 for arbitrary pairs |
| `mae_always_nonnegative` | metrics | MAE ≥ 0 for arbitrary pairs |
| `rmse_geq_mae` | metrics | Cauchy-Schwarz: RMSE ≥ MAE |
| `frobenius_norm_always_nonnegative` | spectral | ‖A‖_F ≥ 0 |
| `frobenius_norm_zero_for_zero_matrix` | spectral | ‖0‖_F = 0 |
| `transpose_involution` | spectral | (A^T)^T = A |
| `distance_to_normal_nonneg` | spectral | d(A, normal) ≥ 0 |

### 1.6 Absorption Manifest Updates

- `head_split.wgsl` and `head_concat.wgsl` moved to "Lean phase" —
  upstream `barracuda::ops::mha` equivalents confirmed working
- `logsumexp_reduce.wgsl` removed from "Planned" — already absorbed upstream

### 1.7 CI Workspace-Wide

All CI steps evolved from single-crate to workspace coverage:

| Step | Before | After |
|------|--------|-------|
| Build | `cargo build` | `cargo build --workspace --all-targets` |
| Test | `cargo test` | `cargo test --workspace --lib` + `--test '*'` |
| Clippy | `cargo clippy` | `cargo clippy --workspace --all-targets --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery` |
| Fmt | `cargo fmt --check` | `cargo fmt --all -- --check` |
| Doc | `cargo doc --no-deps` | `cargo doc --workspace --no-deps` |

---

## Part 2 — What barraCuda Should Absorb

### 2.1 coralForge df64 WGSL Shaders (15 shaders)

These are the last major local shader set. All use `compile_shader_df64`
and achieve ~14-digit precision on consumer GPUs:

| Shader | Domain | barraCuda Absorption Point |
|--------|--------|---------------------------|
| `gelu_f64.wgsl` | Activation | `barracuda::ops::activations` |
| `sigmoid_f64.wgsl` | Activation | `barracuda::ops::activations` |
| `layer_norm_f64.wgsl` | Normalization | `barracuda::ops::norm` |
| `softmax_f64.wgsl` | Attention | `barracuda::ops::attention` |
| `sdpa_scores_f64.wgsl` | Attention | `barracuda::ops::mha` |
| `attention_apply_f64.wgsl` | Attention | `barracuda::ops::mha` |
| `msa_col_attention_scores_f64.wgsl` | Protein (Evoformer) | `barracuda::ops::mha` or new `bio::msa` |
| `msa_row_attention_scores_f64.wgsl` | Protein (Evoformer) | `barracuda::ops::mha` or new `bio::msa` |
| `outer_product_mean_f64.wgsl` | Protein (Evoformer) | `barracuda::ops::bio` |
| `triangle_attention_f64.wgsl` | Protein (Evoformer) | `barracuda::ops::bio` |
| `triangle_mul_incoming_f64.wgsl` | Protein (Evoformer) | `barracuda::ops::bio` |
| `triangle_mul_outgoing_f64.wgsl` | Protein (Evoformer) | `barracuda::ops::bio` |
| `backbone_update_f64.wgsl` | Protein (Structure) | `barracuda::ops::bio` |
| `torsion_angles_f64.wgsl` | Protein (Structure) | `barracuda::ops::bio` |
| `ipa_scores_f64.wgsl` | Protein (Structure) | `barracuda::ops::bio` |

**Priority**: `gelu_f64`, `layer_norm_f64`, `softmax_f64` are general-purpose
and would benefit all springs. Protein shaders are neuralSpring-specific but
validated and ready.

### 2.2 WGSL Binding Layout Alignment (Unchanged from V118)

Local `mean_reduce.wgsl` uses `@binding(2) var<uniform>` (single-workgroup).
Upstream `WGSL_MEAN_REDUCE` uses shared-memory tree reduction. Aligning the
binding layout would let 7 validated pipeline binaries migrate to upstream.

### 2.3 L-BFGS Wiring Points (Unchanged from V118)

`barracuda::optimize::lbfgs` (CPU) and `LbfgsGpu` exist. neuralSpring wiring:
- `pinn.rs`: PDE residual minimization (Burgers equation collocation)
- `loss_landscape.rs`: gradient-based descent for transition-state analysis

### 2.4 StatefulPipeline Documentation Request

neuralSpring has wired `StatefulPipeline` into `Dispatcher` but usage patterns
need upstream guidance:
- Trait bounds on `KernelDispatch` implementations
- State lifecycle (when buffers are recycled vs. persisted)
- Error recovery semantics (partial pipeline failure)

Candidates for `StatefulPipeline` in neuralSpring:
- HMM forward/backward/Viterbi chains (currently per-step dispatch)
- ODE integration loops (rk4/rk45 multi-step)
- Iterative eigensolvers

### 2.5 barraCuda Contract Constants

neuralSpring now validates barraCuda's tolerance contract with named constants
in `validate_toadstool_s93_barracuda_extraction.rs`:

```rust
mod upstream_expected {
    pub const HYDRO_CROP_COEFFICIENT_ABS_TOL: f64 = 1e-6;
    pub const PHYSICS_ANDERSON_EIGENVALUE_ABS_TOL: f64 = 1e-10;
    pub const BIO_DIVERSITY_SHANNON_ABS_TOL: f64 = 1e-8;
    pub const BIO_DIVERSITY_SIMPSON_ABS_TOL: f64 = 1e-10;
}
```

If barraCuda changes these values, neuralSpring's contract test will catch it.
Consider publishing tolerance values in `barracuda::tolerances` module docs.

---

## Part 3 — What toadStool Should Absorb

### 3.1 Discovery Module Pattern

neuralSpring's `playGround/src/discovery.rs` implements the biomeOS 5-tier
socket resolution pattern in a clean, focused module (439 LOC):

1. `$BIOMEOS_SOCKET_DIR` environment variable
2. `$XDG_RUNTIME_DIR/biomeOS/`
3. `$HOME/.biomeOS/sockets/`
4. `/run/user/$UID/biomeOS/`
5. `$TMPDIR/biomeOS/`

Plus `discover_by_capability()` for runtime capability-based primal discovery.
This pattern should be canonical across all springs/primals.

### 3.2 Dispatcher GPU Pipeline Factories

The `tensor_session()` / `stateful_pipeline()` factory pattern on `Dispatcher`
provides a clean API for callers who need fused pipelines. toadStool's
`ResourceOrchestrator` could adopt this pattern to expose session-scoped
GPU contexts to springs.

---

## Part 4 — Learnings Relevant to Ecosystem Evolution

### 4.1 `#[expect(reason)]` for Test Modules

Module-level `#[expect(clippy::unwrap_used, reason = "...")]` is the clean
pattern for test code that intentionally panics on failure. Applying it at
module scope (not per-function) keeps test code uncluttered while satisfying
pedantic+nursery lints. All springs should adopt this.

### 4.2 Provenance Matching: Use Stable Fields

Matching on human-readable labels is fragile — labels drift as descriptions
evolve. Script paths are stable identifiers that survive label rewording.
Provenance systems should key on immutable fields (file paths, commit hashes,
timestamps), not on display strings.

### 4.3 Smart Refactoring > Split Refactoring

The `ipc_client.rs` refactor succeeded because it followed natural responsibility
boundaries (protocol vs. discovery), not arbitrary line-count splits. The resulting
modules are each cohesive and independently testable. File size limits should be
a signal to look for natural seams, not a trigger for mechanical splitting.

### 4.4 Population vs Sample Variance — Document, Don't Unify

neuralSpring's CPU fallback uses population variance (÷N) to match Python
baselines. barraCuda uses sample variance (÷(N−1)). Both are correct for
their contexts. The difference is documented in `specs/BARRACUDA_USAGE.md`.
This is a recurring pattern — when two correct conventions coexist, document
the choice rather than forcing unification.

---

## Part 5 — Quality Metrics

| Category | Count |
|----------|-------|
| Library unit tests | 1164 (+8) |
| playGround unit tests | 73 |
| Forge unit tests | 75 |
| Integration tests | 9 |
| Property tests | 36 (+8) |
| IPC fuzz tests | 5 |
| Named tolerances | 225 |
| Validation binaries | 238 |
| Benchmark binaries | 18 |
| **Total test artifacts** | **1312 lib + 267 binaries** |

Zero clippy (pedantic+nursery, workspace-wide including tests), zero fmt diffs,
zero doc warnings, zero unsafe, zero C deps, zero `#[allow()]`, zero mocks in
production, all files ≤500 LOC. MSRV 1.87, Edition 2024. ecoBin CI verified.

## Files Changed (S168)

| File | Change |
|------|--------|
| `src/provenance/mod.rs` | `expected_source()` match on `self.script` (49+ mappings) |
| `src/coral_forge/activation.rs` | Inline tolerances → `tolerances::*` |
| `src/primitives.rs` | `mul_add()` in proptest, `#[expect(float_cmp)]` |
| `src/property_tests.rs` | 8 new proptests (metrics + spectral), `try_from()` |
| `src/safe_cast.rs` | `#[expect(unwrap_used)]` on tests |
| `src/gpu_dispatch/mod.rs` | `tensor_session()`, `stateful_pipeline()`, doc fix |
| `src/bin/validate_toadstool_s93_barracuda_extraction.rs` | `upstream_expected` module |
| `playGround/src/ipc_client.rs` | 885 → 448 LOC (discovery extracted) |
| `playGround/src/discovery.rs` | **New** — 439 LOC, 5-tier socket + capability discovery |
| `playGround/src/lib.rs` | `pub mod discovery;` |
| `playGround/src/biomeos_client.rs` | `#[expect(expect_used)]` on tests |
| `playGround/src/inference/weights.rs` | `#[expect(float_cmp)]` on tests |
| `playGround/src/mcp_tools.rs` | `#[expect(expect_used)]` on tests |
| `playGround/src/model_config.rs` | `#[expect(unwrap_used)]` on tests |
| `playGround/tests/integration.rs` | `#[expect(unwrap_used, expect_used)]` |
| `metalForge/ABSORPTION_MANIFEST.md` | head_split/head_concat lean, logsumexp removed |
| `.github/workflows/rust.yml` | Workspace-wide CI |
| Root docs | S168 entries |

---

*AGPL-3.0-or-later — neuralSpring → barraCuda / toadStool*
