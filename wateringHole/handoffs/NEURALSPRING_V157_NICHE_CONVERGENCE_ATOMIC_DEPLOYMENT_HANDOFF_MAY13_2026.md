# neuralSpring V157 — Niche Convergence → Atomic Deployment Handoff

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Session:** S205 (May 13, 2026)
**From:** neuralSpring (syntheticChemistry)
**To:** primalSpring (L2 coordination)
**Prior:** V156 (S204b deep debt resolution), V155 (S203 Tier 2 convergence)

---

## Summary

Responding to the "Niche Convergence → Atomic Deployment" directive. All
neuralSpring-specific items addressed:

1. **NestGate weight persistence wired** — `store_to_nestgate`,
   `load_safetensors_from_nestgate`, `load_safetensors_layer_from_nestgate`
   added to `weight_loader.rs`. Model artifacts are now round-trippable
   through NestGate content-addressed storage (BLAKE3 hash-as-key, base64
   encode/decode). 3 new tests.

2. **Squirrel inference pipeline completed** — `IpcMathClient::inference_models()`
   facade added (was absent despite `inference_complete` and `inference_embed`
   being present). `has_squirrel()` discovery convenience method added.
   Full inference composition scenario (`s_inference_composition`) exercises
   `complete`, `embed`, `models`, and round-trip.

3. **Wire name hygiene verified** — `crypto.hash` (BearDog) and
   `security.audit_log` (skunkBat) confirmed correct across all 30+ call
   sites. No `defense.audit` or `message` vs `data` mismatches found.

4. **plasmidBin deployment readiness** — `harvest_ecobin.sh` verified
   (musl static, no banned C deps, size report, staging). Manifest niche
   correctly lists `[beardog, songbird, toadstool, barracuda, coralreef,
   biomeos, squirrel]`.

5. **Foundation thread coverage** — Threads 4 (enviro genomics), 9 (gaming),
   10 (provenance) confirmed with expressions. neuralSpring's domain threads
   (3 immuno, 5 LTEE, 7 Anderson) fully covered.

---

## Code Changes (S205)

| File | Change |
|------|--------|
| `src/weight_loader.rs` | +3 NestGate functions, +3 tests, base64 encode/decode |
| `src/ipc/mod.rs` | +`inference_models()`, +`has_squirrel()` |
| `src/error.rs` | +`IpcError::Other(String)` variant |
| `Cargo.toml` | +`base64 = "0.22"` (workspace + dep) |

---

## Quality Gates

| Metric | Value |
|--------|-------|
| Workspace tests (IPC-first) | **910** (was 907) |
| Clippy warnings (new) | **0** |
| `unsafe` blocks | **0** (`#![forbid(unsafe_code)]`) |
| TODO/FIXME/HACK | **0** |
| Files >800 LOC | **0** |
| Production mocks | **0** |
| `#[allow()]` attributes | **0** |
| Capability constants | **37** |
| IPC modules | **7** |
| CAPABILITY_HINTS | **20** |

---

## Gap Status

| Gap | Status | Notes |
|-----|--------|-------|
| Gap 1 (inference.*) | **wip** | NestGate weight loading now wired. Remaining: Squirrel provider registration, WGSL tokenization pipeline |
| Gap 2 (barraCuda IPC) | **implemented** | Tier 4 IPC-first, barracuda optional |
| Gap 5 (coralReef shader) | **wip** | Awaiting coralReef shader.compile.wgsl live |
| Gap 11 (RPC count) | **CLOSED** | primalSpring confirmed correction |
| NestGate weights | **WIRED** | S205: `store_to_nestgate`, `load_safetensors_from_nestgate` |
| Squirrel composition | **EXERCISED** | Full pipeline: complete + embed + models + round-trip |
| Wire hygiene | **VERIFIED** | crypto.hash, security.audit_log — all correct |
| plasmidBin | **READY** | harvest.sh, manifest, checksums verified |

---

## Upstream Notes for primalSpring

1. **Gap 11 text still stale** — `primalSpring/docs/PRIMAL_GAPS.md` lines
   247 and 419 still reference "Gap 11 (18 RPC methods)". The audit says
   this is corrected but the code hasn't been updated.

2. **neuralSpring test count** — IPC-first (default) is 910. Full
   barracuda-enabled is ~1,453. Manifest shows 1,403 (outdated). Consider
   which count to standardize on.

3. **Ecosystem posture acknowledged** — holding on full NUCLEUS compositions
   until Tower + Node + Nest atomics all prove live. Deepening niche
   (NestGate weight persistence, Squirrel inference).

---

## Directive Compliance

| Directive Item | Status |
|----------------|--------|
| NestGate weight storage (`content.put`/`content.get`) | **DONE** |
| Tier 2 depth / GPU parity | Leveraging 397/397 baselines, 15 CPU benchmarks. GPU parity awaits live barraCuda |
| Squirrel auto-discovery exercise | **DONE** |
| plasmidBin deployment readiness | **VERIFIED** |
| Foundation thread coverage (4, 9, 10) | **CONFIRMED** (not neuralSpring's domain) |
| Wire name hygiene | **VERIFIED** |
| Hold on full NUCLEUS | **ACKNOWLEDGED** |
