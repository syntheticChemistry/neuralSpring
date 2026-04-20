# neuralSpring V135 Handoff — guideStone Level 3 (Bare ALL PASS)

**Date**: April 19, 2026
**Session**: S184
**From**: neuralSpring
**For**: primalSpring, barraCuda, toadStool, Squirrel, all spring teams
**primalSpring**: v0.9.16 (Phase 44)

---

## Summary

`neuralspring_guidestone` v0.2.0 reaches **Level 3** — bare ALL PASS.
29/29 checks pass without any primals running. All 5 guideStone properties
are certified at the bare level. Property 3 (Self-Verifying) upgraded from
PARTIAL to CERTIFIED via BLAKE3 CHECKSUMS.

---

## What Changed (V134 → V135)

### 1. BLAKE3 CHECKSUMS — Property 3 Certified

- New `validation/CHECKSUMS` manifest: 15 validation-critical files
- Files: guideStone binary, tolerances (5), provenance (3), validation (2),
  RNG, capability registry, Python baseline tolerances, Cargo.toml
- Verified via `primalspring::checksums::verify_manifest()` in Phase 1
- New `examples/gen_checksums.rs` generates manifest (feature-gated)

### 2. Structured Output

- `v.section("Phase 1: Bare Properties")` replaces raw `println!`
- `ValidationResult::print_banner()` for consistent banner formatting
- Supports `PRIMALSPRING_JSON=1` for machine-readable output

### 3. FAMILY_ID Support

- Reads `FAMILY_ID` env for family-isolated socket discovery
- Per v0.9.16 depot pattern: `{capability}-{family}.sock` resolution
- Graceful default when unset (no family isolation)

### 4. Protocol Tolerance

- `is_protocol_error()` classifies HTTP-on-UDS as SKIP, not FAIL
- Applied to BearDog (BTSP required) and Songbird (HTTP-on-UDS) checks
- Aligns with primalSpring v0.9.16 protocol tolerance pattern

### 5. Version Bump and Exit Logic

- `GUIDESTONE_VERSION` bumped to `"0.2.0"`
- Banner shows "Level 3" (current readiness, not target)
- Bare exit: `exit_code() == 0 → exit(2)` (aligned with primalSpring pattern)

---

## Bare Check Results (29/29 PASS)

```
Phase 1: Bare Properties
  P1:deterministic_rng                 PASS  (seed=42, bitwise equality)
  P1:deterministic_rng_triple          PASS  (three identical runs)
  P2:provenance_registry_populated     PASS  (49 records, min 40)
  P2:provenance_all_labelled           PASS
  P2:provenance_all_scripted           PASS
  P2:provenance_all_committed          PASS
  p3:checksums_manifest                PASS  (15 entries)
  p3:checksum:* (15 files)             PASS  (all BLAKE3 hashes match)
  P4:ecobin_compliant                  PASS
  P4:pure_rust_forbid_unsafe           PASS
  P4:no_network_required               PASS
  P5:tolerance_count                   PASS  (228+, min 200)
  P5:tolerances_all_finite             PASS
  P5:tolerances_all_named              PASS
  P5:tolerances_all_categorized        PASS

Phase 2: Discovery + Liveness
  (4 SKIP — no NUCLEUS deployed)

Exit code: 2 (bare only)
```

---

## For primalSpring

1. **Readiness update**: neuralSpring guideStone is now Level 3 in
   `downstream_manifest.toml` (`guidestone_readiness = 3`)
2. **CHECKSUMS pattern adopted**: Following `primalspring_guidestone` exactly
3. **Next step**: Level 4 requires deploying a live NUCLEUS from plasmidBin
   and running `neuralspring_guidestone` externally against it

## For barraCuda

1. **Gap 11 still open**: 18 library-only methods (eigh, Pearson, chi-squared,
   ESN, NN, etc.) have no JSON-RPC equivalents. Confirmed again against v0.9.16
2. **IPC parity validated**: `stats.mean`, `tensor.matmul`, `tensor.create`
   wired via `validate_parity` / `validate_parity_vec` — ready for live testing

## For toadStool

1. **compute.dispatch**: Wired via `CompositionContext::call("compute", ...)`
2. **Discovery**: Uses `discover_by_capability("compute")` with family fallback

## For Squirrel

1. **inference.complete / inference.embed**: Wired in Phase 3 domain parity
2. **Discovery**: `discover_by_capability("ai")` with family fallback

## For All Springs

1. **CHECKSUMS pattern**: `examples/gen_checksums.rs` + `validation/CHECKSUMS`
   is a reusable pattern. Any spring can adopt it for P3 certification
2. **Protocol tolerance**: HTTP-on-UDS primals (Songbird, petalTongue) are
   classified as SKIP via `is_protocol_error()` — adopt this in your harnesses
3. **FAMILY_ID**: Export before launching NUCLEUS to get family-isolated sockets

---

## Readiness Matrix

| Level | Description | Status |
|-------|-------------|--------|
| 1 | Validation exists | DONE |
| 2 | Properties documented | DONE |
| 3 | Bare guideStone works (29/29 PASS) | **DONE** |
| 4 | NUCLEUS guideStone works | PENDING |
| 5 | Certified (cross-substrate parity) | PENDING |

---

## Files Changed

| File | Change |
|------|--------|
| `examples/gen_checksums.rs` | NEW — BLAKE3 manifest generator |
| `validation/CHECKSUMS` | NEW — 15-file BLAKE3 manifest |
| `src/bin/neuralspring_guidestone.rs` | P3 CHECKSUMS, v.section(), FAMILY_ID, protocol tolerance, v0.2.0 |
| `Cargo.toml` | `[[example]]` entry for gen_checksums |
| `docs/GUIDESTONE_PROPERTIES.md` | P3 CERTIFIED, Level 3 evidence, readiness matrix |
| `docs/PRIMAL_GAPS.md` | Gap 13 Level 3, v0.9.16 notes, Gap 11 v0.9.16 update |
| `graphs/neuralspring_deploy.toml` | V135/S184 |

---

## Gap Status

| Gap | Description | Status |
|-----|-------------|--------|
| 11 | barraCuda 18 IPC surface gaps | **open** (confirmed v0.9.16) |
| 13 | guideStone evolution | Level 3 (was Level 2) |

---

## Known Issues from v0.9.16

| Issue | Workaround |
|-------|------------|
| BearDog requires `BEARDOG_FAMILY_SEED` | Export before launch |
| Songbird/petalTongue HTTP on UDS | `is_protocol_error()` → SKIP |
| BearDog resets without BTSP | Expected — liveness skipped |

---

## Primal Use & Evolution Patterns

### Primal Dependency Profile

neuralSpring uses primals in two distinct modes:

| Primal | Library (Level 2) | IPC (Level 5+) | Status |
|--------|-------------------|-----------------|--------|
| **barraCuda** | ~400 `barracuda::` imports across `src/`, `metalForge/`, `playGround/` (device, tensor, ops, stats, nn, spectral, nautilus, dispatch, staging) | `IpcMathClient`: 9 methods (`stats.mean`, `tensor.matmul`, `tensor.create`, `compute.dispatch`, `crypto.hash`, `inference.*`) | Library = production; IPC = guideStone + proto-nucleate validators |
| **primalSpring** | `primalspring::composition`, `checksums`, `tolerances`, `validation` (optional, `guidestone` feature only) | N/A (library API only) | guideStone binary + gen_checksums example |
| **toadStool** | None (zero library imports) | `compute.dispatch.submit` via `ToadStoolClient` in playGround; `CompositionContext::call("compute", ...)` in guideStone | IPC only |
| **BearDog** | None | `crypto.hash` via `hash_bytes()` in guideStone; health probes in `tower.rs` | IPC only |
| **Songbird** | None | Discovery mesh probes in `tower.rs`; `resolve_capability()` in guideStone | IPC only |
| **Squirrel** | None | `inference.complete`/`inference.embed` via `try_squirrel_route()` in handlers; guideStone Phase 3 | IPC only |
| **coralReef** | `coralreef_bridge.rs` (forge, optional feature) | `shader.compile.wgsl` via `CoralReefClient` in playGround | Both, but forge bridge is thin |
| **petalTongue** | `PetalTonguePushClient` (`src/visualization/`) | Push visualization data over UDS | Library-thin IPC |
| **NestGate** | None | Not in proto-nucleate; weight storage planned for spring-deploy | Deferred |

### IPC Migration Summary (barraCuda)

barraCuda is the primary math dependency. For the Rust proof (Level 2), all math
is called via direct library imports (`barracuda::ops::*`, `barracuda::stats::*`,
etc.). For the primal proof (Level 5+), these calls must migrate to IPC.

**Already wired via IPC**: `stats.mean`, `tensor.matmul`, `tensor.create`,
`stats.std_dev`, `stats.weighted_mean`, `compute.dispatch`, `crypto.hash`,
`inference.complete`, `inference.embed` (9 methods via `IpcMathClient` + guideStone)

**Blocked on Gap 11** (18 methods with no JSON-RPC equivalent): `eigh_householder_qr`,
`pearson_correlation`, `chi_squared_statistic`, `empirical_spectral_density`,
`solve_f64_cpu`, `esn_v2::*`, `nn::SimpleMlp`, `nautilus::*`, etc.

### NUCLEUS Deployment Patterns

neuralSpring's guideStone follows the layered certification model:

```
primalspring_guidestone (Level 4, 67/67)
    ↓ certifies base composition
neuralspring_guidestone (Level 3, 29/29 bare)
    ↓ validates domain science on top
```

Deploy from `plasmidBin/` ecobins → launch NUCLEUS (12 primals) → run
`primalspring_guidestone` → exit 0 → run `neuralspring_guidestone` → validate
7 `PROTO_NUCLEATE_VALIDATION_CAPABILITIES` via IPC → exit 0 = primal proof.

### Ecosystem Learnings for Upstream Teams

1. **Dual IPC stacks**: neuralSpring maintains both `IpcMathClient` (custom, 9
   methods) and `primalspring::composition::CompositionContext` (upstream API).
   The custom client was built before the upstream API existed. Springs should
   use `CompositionContext` directly — it handles discovery, fallback, health
   checks, and protocol tolerance. The custom client is retained only as a
   "validation window" for development-time parity comparison.

2. **`composed` feature is unused**: The `composed = ["primal"]` feature in
   `Cargo.toml` was created for future IPC-only paths but no `#[cfg(feature =
   "composed")]` exists in source. It can be removed or left as a no-op alias.

3. **Protocol tolerance is essential**: Songbird and petalTongue speak HTTP on
   UDS sockets. Without `is_protocol_error()` classification, health checks
   for these primals produce confusing failures. All springs should adopt this.

4. **BLAKE3 checksums pattern**: `examples/gen_checksums.rs` + `validation/CHECKSUMS`
   is reusable. Any spring can adopt `primalspring::checksums` for P3 certification
   in under 50 lines of code.

5. **guideStone inheritance works**: Our guideStone binary uses only
   `primalspring::composition` and `primalspring::checksums` — no internal primalSpring
   APIs. The layered model (base certification → domain certification) is clean
   and springs should follow it rather than reimplementing composition validation.

---

**License**: AGPL-3.0-or-later
