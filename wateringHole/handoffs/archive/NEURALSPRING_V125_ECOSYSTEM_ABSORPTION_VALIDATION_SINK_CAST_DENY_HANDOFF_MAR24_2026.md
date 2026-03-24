<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring V125 — Ecosystem Absorption: ValidationSink, Cast Deny, Provenance Integrity

**Session**: S175 (March 24, 2026)
**Previous**: V124 (S174 — deep audit, zero debt, provenance alignment)
**Status**: Complete

---

## What Changed

### 1. Cast Lint Hardening (P0)

`cast_possible_truncation` and `cast_sign_loss` promoted from `warn` to `deny`
in `[workspace.lints.clippy]`. All 1,353 cast sites across 177 files are covered
by existing `#[expect(clippy::cast_*)]` attributes. Zero breakage on promotion.

This aligns with hotSpring V134's stricter cast policy.

### 2. ValidationSink Pattern (P0)

Absorbed from ecosystem convergence of wetSpring V134, airSpring V010,
and groundSpring V121. New module `src/validation/sink.rs`:

- **Trait**: `ValidationSink` with `on_check(&Check)` + `on_finish(name, passed, total, success)`
- **Implementations**:
  - `StdoutSink` — human-readable `[PASS]/[FAIL]` (default, matches original behavior)
  - `JsonSink<W>` — single JSON blob with all checks (CI artifact collection)
  - `NdjsonSink<W>` — one JSON line per check (streaming pipelines)
  - `CollectingSink` — programmatic inspection in tests
  - `SilentSink` — no-op for benchmarks
- **Harness integration**: `ValidationHarness::emit_to_sink()` and `emit_json()`;
  `finish()` now delegates to `StdoutSink` internally
- **12 new tests** covering all sink implementations

### 3. Provenance Integrity Tests (P0)

Four new tests in `src/provenance/mod.rs`:

- `provenance_scripts_exist_on_disk` — all 49 registered Python scripts present
- `provenance_scripts_have_provenance_header` — each has `# Provenance: see src/provenance/`
- `provenance_scripts_have_spdx_header` — each has SPDX license identifier
- `provenance_scripts_content_stability` — hash + size sanity for baseline drift detection

### 4. Deploy Graph Updated

`graphs/neuralspring_deploy.toml` bumped from V105/S155 to V124/S174.

### 5. Leverage Guide Refreshed

`NEURALSPRING_LEVERAGE_GUIDE.md` at wateringHole updated to V124/S174 with
current metrics: 1,400+ tests, 232+ tolerances, 261 binaries, cast deny,
ValidationSink.

### 6. Full Audit Results

| Check | Result |
|-------|--------|
| sled dependencies | None |
| unsafe code | None (`#![forbid(unsafe_code)]`) |
| Production mocks | None (all in `#[cfg(test)]`) |
| Files > 1000 LOC | None (largest: 879 LOC bench) |
| `#[allow()]` in prod | None |
| Cast lints | Denied |
| Clippy warnings | Zero |
| Fmt issues | Zero |

---

## Metrics

| Metric | V124 | V125 |
|--------|------|------|
| Lib tests | 1,199 | 1,211 |
| Total tests | ~1,385 | ~1,400 |
| Cast lints | warn | **deny** |
| Validation sinks | 0 | **5** |
| Provenance integrity tests | 0 | **4** |
| Tolerance constants | 232+ | 232+ |
| Validation binaries | 261 | 261 |

---

## Cross-Spring Notes

- **For all springs**: The `ValidationSink` trait is now ecosystem-converged across
  wetSpring, airSpring, groundSpring, and neuralSpring. New springs should adopt
  this pattern. See `src/validation/sink.rs` for the reference implementation.
- **For barraCuda**: No new absorption requests beyond V124 handoff.
- **For primalSpring**: Deploy graph at `graphs/neuralspring_deploy.toml` is current.
