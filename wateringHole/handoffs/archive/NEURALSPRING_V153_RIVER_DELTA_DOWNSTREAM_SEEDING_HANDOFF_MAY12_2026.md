<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring V153 — River Delta Downstream Seeding Handoff

**Session**: S202–S202b | **Date**: May 12, 2026 | **Handoff**: V153
**Prior**: V152 (S201b, May 11, 2026 — Tier 4 IPC-first + LTEE B1 + foundation seeding)

---

## Summary

Session S202 responds to the primalSpring "River Delta Springs — Downstream
Seeding Sprint (May 12, 2026)" audit blurb. All three neuralSpring action
items from the blurb are complete:

1. **`--format json` for Tier 2 projectNUCLEUS ingestion** — done at the
   `ValidationHarness` level, so every validation binary gets structured JSON
   output without per-binary changes
2. **Foundation Thread 5 expression** — `ML_SURROGATES.md` authored, thread
   elevated from "mapped" to "active" in `THREAD_INDEX.toml`
3. **Deep debt sweep** — IPC evolution, dependency consolidation, test coverage
   expansion, doc reconciliation

---

## What Changed (S202)

### Code

| Change | Impact |
|--------|--------|
| `ValidationHarness::finish()` now auto-detects `--format json` / `--format=json` / `NEURALSPRING_JSON=1` | All validation binaries emit structured JSON (suite, checks, tolerances) for Tier 2 projectNUCLEUS |
| `CapabilityRouter` replaces hardcoded 6-primal `IpcMathClient` struct | Capability-addressed routing via `CAPABILITY_HINTS` (14 mappings). Public API unchanged |
| `tarpc`, `toml`, `pollster` hoisted to `[workspace.dependencies]` | Single version bump point; metalForge forge aligned |
| `metrics.rs` CPU fallback: `r_squared` guards `ss_tot ≈ 0` | No more NaN on constant `y_true` edge case |
| `metrics::tests` ungated from `barracuda`-only to `#[cfg(test)]` | +10 tests, CPU path now covered |
| `tests/integration.rs`: `gelu` test gated `#[cfg(feature = "barracuda")]` | `cargo test --workspace` passes IPC-first |
| `src/ipc/nestgate.rs`: NestGate IPC module | `content_put`, `content_get`, `content_exists` — content-addressed storage |
| `src/capabilities.rs`: 3 NestGate constants | `CONTENT_PUT`, `CONTENT_GET`, `CONTENT_EXISTS` |
| `src/ipc/mod.rs`: NestGate in `CAPABILITY_HINTS` | 17 hints (was 14); `PrimalSlot::Nestgate`; facade methods |
| `control/ltee_mutation_accumulation/README.md` | lithoSpore handoff: pipeline, artifacts, PRNG notes |

### Foundation (gardens/)

| Change | Where |
|--------|-------|
| `ML_SURROGATES.md` authored | `gardens/foundation/expressions/` |
| `THREAD_INDEX.toml` thread 5 wired | expression, sources, targets populated; status → "active" |
| `expressions/README.md` updated | All 5 active expressions listed |

### projectNUCLEUS (gardens/)

| Change | Where |
|--------|-------|
| `PRIMALSPRING_JSON=1` in `[execution.env]` | `workloads/neuralspring/neuralspring-ml-validation.toml` |
| `PRIMALSPRING_JSON=1` in `[execution.env]` | `workloads/neuralspring/neuralspring-certification.toml` |

### Documentation

All living docs updated to S202/V153: README, EVOLUTION_READINESS,
CONTROL_EXPERIMENT_STATUS, PRIMAL_GAPS, FOUNDATION_SEEDING, CHANGELOG,
whitePaper/README, baseCamp/README, experiments/README, sporeprint/validation-summary.

---

## Quality Gates

- **868 workspace tests** (704 lib + 11 integration + 73 forge + 80 playGround) — 0 failures
- **19 certification tests** (guidestone L5) — all PASS
- **0 clippy warnings** (pedantic + nursery + cast deny)
- **0 unsafe code** — `unsafe_code = "forbid"` workspace-wide
- **0 TODO/FIXME/HACK** in production code
- **0 mocks in production** — all isolated to `#[cfg(test)]`
- Edition 2024, MSRV 1.87

---

## Upstream Primal Gaps (for primal teams)

### Open gaps requiring upstream action

| Gap | Primal | Status | What neuralSpring needs |
|-----|--------|--------|------------------------|
| Gap 5 | NestGate | **wip** | `content.put/get/exists` IPC wired (S202b); weight_loader integration pending |
| Gap 6 | BearDog | **wip** | BTSP session establishment for composed-mode IPC |
| Gap 9 | barraCuda | **open** | `plasma_dispersion` unconditionally imports `domain-lattice`; feature-gate belongs upstream |
| Gap 10 | barraCuda / coralReef | **tracking** | 25 shader absorption candidates (WGSL → upstream `ops/`/`stats/`) |

### Closed gaps (S201b–S202)

| Gap | Resolution |
|-----|------------|
| Gap 11 (18 IPC surface gaps) | 12 via barraCuda RPC expansion, 4 composable, 5 CPU fallback |
| Gap 1 (inference) | Wired; handlers return `SERVICE_UNAVAILABLE` when Squirrel absent |
| Gap 2 (barraCuda direct import) | `default = []`; IPC-first validated |
| Gaps 3, 4, 7, 8 | All resolved (S178–S180) |

---

## Patterns for Downstream Absorption

### For other springs

1. **`ValidationHarness` + `JsonSink` + `NdjsonSink`** — reusable validation
   framework with structured output. Copy `src/validation/mod.rs` + `sink.rs`.
   Every binary that calls `h.finish()` gets `--format json` for free.

2. **`CapabilityRouter` + `CAPABILITY_HINTS`** — capability-addressed IPC
   routing. Spring declares what it needs, not who provides it. Template:
   `src/ipc/mod.rs`.

3. **IPC-first build pattern** — `default = []`, CPU fallbacks with
   `#[cfg(not(feature = "barracuda"))]`, `required-features` on GPU bins.
   Enables CI without GPU, audit without primal trees.

4. **LTEE B1 pipeline** — Python baseline → `expected_values.json` →
   Rust `ValidationHarness` binary. Template for any peer-reviewed
   reproduction: `control/ltee_mutation_accumulation/` + `validate_ltee_b1_*`.

### For projectNUCLEUS

- NUCLEUS workload TOMLs now set `PRIMALSPRING_JSON=1` in `[execution.env]`
  for structured output. This is the pattern for Tier 2 ingestion.
- `neuralspring_unibin validate` uses `primalspring::ValidationResult::finish()`
  which honors `PRIMALSPRING_JSON`. Standalone validators use `ValidationHarness`
  which honors `--format json` / `NEURALSPRING_JSON=1`.

### For foundation

- Thread 5 expression (`ML_SURROGATES.md`) covers 4 pillars, 6 ML architectures,
  12 validated targets. New surrogate models (Transformer, GNN) will add targets
  with `validated = false` for foundation tracking.

---

## Composition & Deployment

### NUCLEUS participation

neuralSpring registers as a **nucleated primal** via biomeOS:
- `nucleus.register` at startup
- `capability.register` for all 34 capabilities
- `nucleus.heartbeat` background loop
- `nucleus.deregister` on shutdown

### Deploy graphs (4)

All at V151/S200b. Declare sequential coordination with capability-based
node addressing:
- `neuralspring_deploy.toml` — full spring niche (5 phases, 34 capabilities)
- `neuralspring_inference_pipeline.toml` — inference chain with sweetGrass provenance
- `neuralspring_spectral_analysis.toml` — science pipeline
- `neuralspring_math_pipeline.toml` — minimal node_atomic for parity checks

### Primal roster

| Primal | Integration | Feature |
|--------|-------------|---------|
| barraCuda | Direct import + IPC | `barracuda` feature (optional) |
| toadStool | IPC only | runtime discovery |
| BearDog | IPC (Tower probe) | runtime discovery |
| Songbird | Discovery/liveness | runtime discovery |
| coralReef | IPC + forge bridge | runtime discovery |
| Squirrel | IPC (inference) | runtime discovery |
| skunkBat | IPC (audit log) | runtime discovery |
| nestGate | IPC (`content.put/get/exists`) + deploy graph | **Gap 5 wip**: IPC wired, weight_loader pending |
| loamSpine | Deploy graph germination | provenance trio |
| sweetGrass | Deploy graph provenance | provenance trio |
| rhizoCrypt | Deploy graph DAG | provenance trio |
| petalTongue | Visualization push | optional push client |

---

## Known Debt

1. **Test count reconciliation**: IPC-first = 867; barracuda-enabled = ~1,310+.
   Living docs now report IPC-first counts. Legacy docs may still reference
   barracuda-included totals.

2. **Deploy graph version drift**: Graphs say V151/S200b. Consider bumping
   to V153 if graph content changes.

3. **Kokkos GPU parity**: Only estimated baselines, not matched-hardware runs.
   Requires hardware access for proper benchmarking.

4. **LTEE queue**: 11 papers queued (B2–B9, E2–E5). B1 complete. This is
   planned roadmap, not debt.

5. **`tools/composition_template.sh`**: Fixed — was referencing missing
   `ttt_composition.sh`, now points to `neural_composition.sh`.

6. **Upstream registry drift**: `primalSpring/docs/PRIMAL_GAPS.md` Layer 3
   table still lists neuralSpring as "Gap 11 (18 RPC methods)" open.
   **Gap 11 was resolved in S201b** (12 RPC + 4 composable + 5 CPU fallback).
   Request primalSpring update their Layer 3 table.

---

*neuralSpring V153 | Sessions S202–S202b | May 12, 2026*
