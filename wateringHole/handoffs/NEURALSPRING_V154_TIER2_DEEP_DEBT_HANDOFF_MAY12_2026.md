<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring V154 — Tier 2 Wiring + Deep Debt Audit Handoff

**Session**: S202c | **Date**: May 12, 2026 | **Handoff**: V154
**Prior**: V153 (S202–S202b, May 12, 2026 — River Delta seeding + NestGate IPC + Gap 11 drift)

---

## Summary

Session S202c responds to the "Ecosystem Wave Sync (May 12, 2026)" audit.
Tier 2 convergence is wired (toadStool S250 unblocked it), a comprehensive
deep debt audit found zero actionable debt, and downstream product gaps in
projectNUCLEUS and foundation were closed.

---

## 1. Tier 2 Convergence — toadStool Wiring

### What shipped

`toadstool.validate` and `toadstool.list_workloads` are now wired in
neuralSpring's IPC surface, completing the Tier 2 pre-flight path:

| File | Change |
|------|--------|
| `src/ipc/toadstool.rs` | Expanded from 1 method to 3: `compute_dispatch`, `validate`, `list_workloads`. `ValidateResult` struct parses all 6 response fields. |
| `src/capabilities.rs` | +2 constants: `TOADSTOOL_VALIDATE`, `TOADSTOOL_LIST_WORKLOADS` (36 total) |
| `src/ipc/mod.rs` | `CAPABILITY_HINTS` 17→19 entries. `IpcMathClient` gains `validate_workload()` and `list_workloads()` facade methods. |

### Tier 2 status for neuralSpring

| Requirement | Status |
|-------------|--------|
| `--format json` on validation binaries | **DONE** (S202, `ValidationHarness`) |
| `toadstool.validate` IPC client | **DONE** (S202c) |
| `toadstool.list_workloads` IPC client | **DONE** (S202c) |
| `barracuda.precision.route` IPC client | **BLOCKED** — not implemented in barraCuda |

### What springs need from upstream

`barracuda.precision.route` is specified in `LIVE_SCIENCE_API.md` but is
**not implemented** in barraCuda's `REGISTERED_METHODS` or handler dispatch.
neuralSpring will wire it when upstream ships.

---

## 2. Deep Debt Audit — Comprehensive Sweep

A full codebase audit across all `.rs` files (521+), all dependencies, and all
documentation found **zero actionable debt**:

| Metric | Result |
|--------|--------|
| Files >800 lines | **0** (max: `tolerances/mod.rs` at 776L) |
| `unsafe` blocks | **0** (`unsafe_code = "forbid"` workspace-wide) |
| Production mocks | **0** (all `FAKE_SOCKET`/`mock_*` in `#[cfg(test)]` only) |
| Production `panic!` | **0** (all 10 occurrences are test assertions) |
| `TODO`/`FIXME`/`HACK` | **0** |
| `todo!()`/`unimplemented!()` | **0** |
| C dependencies | **0** (all pure Rust except wgpu/barracuda GPU domain) |
| Hardcoded primal names | **0** in production (one documented local constant in forge) |
| Paper queue | **27/27 complete**, queue closed |
| Duplicate dep versions | **0** (all in `[workspace.dependencies]`) |

### Benchmark coverage

- **16 Rust bench binaries** with CPU, GPU, dispatch tier, cross-spring, and industry GPU parity coverage
- **20 Python bench scripts** in `control/` with paired Rust counterparts
- **Kokkos parity**: documented in `specs/BENCHMARK_ANALYSIS.md` with estimated baselines
- **Polybench/SPEC**: documented as gap (not started)
- **SYCL**: N/A for current architecture
- **Galaxy**: N/A (workflow engine, not kernel benchmark)

---

## 3. Downstream Product Gaps Closed

### projectNUCLEUS

**New**: `workloads/neuralspring/neuralspring-ltee-b1-mutation.toml` — workload
TOML for `validate_ltee_b1_mutation_accumulation --format json`. Enables Tier 2
lithoSpore ingestion of B1 ML surrogate predictions.

neuralSpring now has **3 NUCLEUS workloads**: certification, ML validation, B1 LTEE.

### foundation

**Updated**: `lineage/THREAD_INDEX.toml` Thread 5 entry now includes
`ml_expression = "expressions/ML_SURROGATES.md"`, `ml_data_sources`, and
`ml_data_targets` fields — wiring the neuralSpring ML surrogate slice alongside
the main LTEE evolutionary dynamics expression.

### lithoSpore

**Status**: B1 ML surrogate data is ready for ingestion. B3/B4/B6 domain
outputs remain QUEUED — these require domain-specific trained models, not
infrastructure work.

---

## 4. Quality Gates

| Metric | Value |
|--------|-------|
| Workspace tests (IPC-first) | **892** (728 lib + 11 integration + 73 forge + 80 playGround) |
| Certification tests | **19** (guideStone L0–L5 ALL PASS) |
| Clippy warnings | **0** (pedantic+nursery+cast deny) |
| Python baselines | **397/397 PASS** |
| Capabilities registered | **36** (34 prior + `toadstool.validate` + `toadstool.list_workloads`) |
| CAPABILITY_HINTS | **19** entries |
| IPC modules | **7** (barracuda, toadstool, beardog, squirrel, coralreef, skunkbat, nestgate) |
| Deploy graphs | **4** |
| Named tolerances | **233+** |

---

## 5. NUCLEUS Composition Patterns

### How neuralSpring composes for deployment

```
biomeOS orchestrator
  ├── neuralSpring (neuralspring binary)
  │     ├── IPC → barraCuda (stats.*, tensor.*)
  │     ├── IPC → toadStool (compute.dispatch, toadstool.validate, toadstool.list_workloads)
  │     ├── IPC → BearDog (crypto.hash)
  │     ├── IPC → Squirrel (inference.*)
  │     ├── IPC → coralReef (shader.compile.*)
  │     ├── IPC → skunkBat (security.audit_log)
  │     └── IPC → NestGate (content.put/get/exists)
  └── neuralAPI gateway → JSON-RPC to neuralspring binary
```

### CapabilityRouter pattern (reference for other springs)

neuralSpring's `CapabilityRouter` maps capabilities to discovered primal
sockets at runtime. Springs never hardcode socket paths or primal names in
their IPC calls — they declare *what* they need (a capability), and discovery
resolves *where* to find it. Pattern in `src/ipc/mod.rs`:

1. `CAPABILITY_HINTS` maps capability strings to expected primal names
2. `CapabilityRouter::from_hints()` resolves sockets via `discover_primal_socket`
3. `IpcMathClient` facade methods call through `router.require(capability)?`
4. Missing primals produce typed `IpcError::NotDiscovered` — no panics

### Tier 2 validation flow (reference for other springs)

```
ValidationHarness::new("validator_name")
  → .check("name", expected, actual, tolerance)
  → .finish()  // detects --format json or NEURALSPRING_JSON=1
       ├── JsonSink   → structured JSON for projectNUCLEUS/lithoSpore
       └── StdoutSink → human-readable CLI output
```

---

## 6. Upstream Gaps — For Primal Teams

### barraCuda

- `barracuda.precision.route` is specified in `LIVE_SCIENCE_API.md` but not
  implemented. neuralSpring (and likely other springs) will wire it once shipped.

### primalSpring

- `primalSpring/docs/PRIMAL_GAPS.md` Layer 3 table still shows neuralSpring
  "Gap 11 (18 RPC methods)" as open. This was **resolved in S201b** (12 RPC,
  4 composable, 5 CPU fallback). Flagged three times now (V153, S202b, S202c).
  **Request correction.**

---

## 7. For Sister Springs

### Tier 2 wiring pattern (if you have `--format json`)

Add to your `src/ipc/toadstool.rs` (or equivalent):

```rust
pub fn validate(socket: &Path, workload_path: &str, dry_run: bool, timeout: Duration)
    -> Result<ValidateResult, IpcError>
```

Add `TOADSTOOL_VALIDATE` and `TOADSTOOL_LIST_WORKLOADS` to your capabilities
constants. Wire through your `CAPABILITY_HINTS` → `CapabilityRouter`.

### Deep debt checklist (what we verified)

Use this as a template for your own sweep:
- `grep -r 'unsafe' src/` — should be zero (or documented exceptions)
- `grep -r 'todo!\|unimplemented!' src/` — should be zero
- `grep -r 'TODO\|FIXME\|HACK' src/` — should be zero
- `grep -r '#\[allow' src/` — should be zero (or documented)
- `wc -l src/**/*.rs | sort -rn | head -20` — no files >800L
- `grep -r 'panic!' src/` — verify all are in `#[cfg(test)]`
- Confirm all deps in `[workspace.dependencies]`

---

*neuralSpring V154 | Session S202c | May 12, 2026*
