<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring V152 — Tier 4 IPC-First + LTEE B1 + Foundation Seeding

**Session**: S201b | **Date**: May 11, 2026 | **Handoff**: V152

---

## Summary

neuralSpring now qualifies for the Pillar 5 Tier 4 IPC-first exit gate.
`default = []` in Cargo.toml — barracuda is no longer linked by default.
LTEE B1 baseline started. Foundation Threads 5+7 seeded.

---

## Changes

### Tier 4 IPC-First Defaults

- `Cargo.toml` `default = ["barracuda"]` → `default = []`
- 48 files feature-gated with `#[cfg(feature = "barracuda")]`:
  - 11 WGSL shader re-exports (one per science module)
  - `bench`, `nucleus_pipeline` — gated entirely
  - Per-function gates across 26 modules (primitives, metrics, eigh, etc.)
- 241 GPU-dependent `[[bin]]` stanzas have `required-features = ["barracuda"]`
- CPU fallback implementations for 12 functions:
  - `primitives::{sigmoid, gelu, relu, relu_vec, hill_activation, hill_repression, shannon_entropy, shannon_equitability, shannon_entropy_from_counts, pearson_r}`
  - `metrics::{r_squared, rmse, mae, nse}`
- UniBin builds without barracuda: `cargo build --no-default-features --features guidestone --bin neuralspring_unibin`
- 693 tests pass IPC-first (no barracuda), 1,300 pass with `--features barracuda`
- 19 certification tests pass

### Deep Debt Cleanup

- **Deprecated `ipc_dispatch` removed** — 400-line monolithic module deleted. Graduated to per-primal `ipc/` tree in S200. Zero callers remained.
- **Typed `IpcError` hierarchy** — `error::IpcError` with `NotDiscovered`, `Transport`, `Protocol` variants. IPC facade and all 6 submodules migrated from `Result<_, String>` to `Result<_, IpcError>`. `From` impls for backward compatibility.
- **Dead code gate fixed** — `scaffold::fieldmap` properly `#[cfg]`-gated instead of `allow(dead_code)`.
- **Playground warnings eliminated** — zero workspace warnings (was 18).

### LTEE B1 Baseline

- `control/ltee_mutation_accumulation/ltee_mutation_accumulation.py` — 8/8 PASS
- Barrick 2009 Ara-1 mutation accumulation time series
- Linear rate: 3.59e-3 mutations/generation
- Power-law exponent: 0.82 (sublinear accumulation)
- Expected values JSON for lithoSpore module 2
- 12-paper LTEE queue added to `specs/PAPER_REVIEW_QUEUE.md`

### Foundation Seeding

- **Thread 5** (ML Surrogates): NEW — 15 sources, 12 targets
  - LSTM, ESN, WDM surrogates, evolutionary dynamics, LTEE B1
- **Thread 7** (Anderson): EXPANDED — 6 neuralSpring targets added (total 24)
  - baseCamp nS-01..06 + Evoformer spectral validation

Foundation now at 7/10 threads (was 5/10).

---

## Quality Gates

| Metric | Value |
|--------|-------|
| Lib tests (barracuda) | 1,300 PASS |
| Lib tests (IPC-first) | 693 PASS |
| Certification tests | 19 PASS (L0-L5) |
| Forge tests | 73 PASS |
| playGround tests | 80 PASS |
| Total workspace | 1,453 |
| Workspace warnings | Zero |
| `forbid(unsafe_code)` | Enforced |
| Files > 800L | None |
| `default` features | `[]` (IPC-first) |
| `required-features` bins | 241 (GPU-gated) |

---

## Tier 4 Exit Gate Status

| Spring | `default` | Qualifies |
|--------|-----------|-----------|
| groundSpring | `[]` | Yes |
| healthSpring | `[]` | Yes |
| ludoSpring | `[]` (exemplar) | Yes |
| **neuralSpring** | **`[]`** | **Yes** |
| wetSpring | `["barracuda-lib"]` | No |
| airSpring | `["local", "testutil"]` | No |
| hotSpring | `["barracuda-local"]` | No |

4/7 springs qualify — Pillar 5 target (4+) met.

---

## Upstream Gaps (for primalSpring)

1. **LTEE Phase 2 coordination** — 36 paper-spring assignments queued, B1
   started in neuralSpring. groundSpring B2 is ecosystem critical path.
2. **Foundation Thread 5+7 review** — neuralSpring seeded both. Review
   source/target quality for sweetGrass braid integration.

*neuralSpring V152 | Session S201b | AGPL-3.0-or-later*
