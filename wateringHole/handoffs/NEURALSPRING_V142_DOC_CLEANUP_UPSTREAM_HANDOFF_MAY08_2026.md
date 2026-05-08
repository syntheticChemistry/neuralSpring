# neuralSpring V142 — Doc Cleanup, Upstream Primal Handoffs, and Downstream Absorption

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Date:** May 8, 2026
**Session:** S192
**From:** neuralSpring
**To:** All primal teams, all spring teams, sporeGarden (projectNUCLEUS, foundation)
**Prior:** V141 (full sweep, same day), V140 (parity response, same day)

---

## Summary

Final session of the S190-S192 triple: parity response → full sweep → doc cleanup.
All docs synchronized to S192. Upstream primal handoffs crafted for all teams with
neuralSpring-relevant debt. Downstream absorption review for projectNUCLEUS and
foundation. Archive sweep clean (zero debris, zero stale TODOs).

## Part 1: What We Delivered (S190–S192)

| Deliverable | Session | Status |
|-------------|---------|--------|
| barraCuda `optional = true` (IPC-first) | S190 | Done |
| Registry cross-sync test (389 methods) | S190 | Done |
| exp094 NUCLEUS composition parity | S190 | Done |
| 3 new deploy graphs (4 total) | S190 | Done |
| PRIMAL_GAPS 1-4 IMPLEMENTED | S190 | Done |
| projectNUCLEUS + foundation review | S191 | Done |
| 45 new inline unit tests | S191 | Done |
| Liu faculty notebooks (016-018, 26/26) | S191 | Done |
| Benchmark gap roadmap | S191 | Done |
| Tier 4 IPC validator audit | S191 | Done |
| Root/baseCamp/experiments/wateringHole refresh | S192 | Done |
| Upstream primal handoff (this document) | S192 | Done |

## Part 2: Upstream Primal Handoffs

These are evolution items that **require upstream primal team action** for
neuralSpring to advance to guideStone Level 4/5 and full NUCLEUS composition.

---

### barraCuda — 18 IPC Surface Gaps (Priority: HIGH)

neuralSpring's `PRIMAL_GAPS.md` Gap 11 documents 18 `barraCuda` JSON-RPC methods
that neuralSpring needs but barraCuda does not yet expose via IPC:

**Category A — Tensor Lifecycle (5 methods):**
- `tensor.create` — create tensor on device
- `tensor.read` — readback to host
- `tensor.zeros` / `tensor.ones` — factory methods
- `tensor.from_vec` — create from host data

**Category B — Core Math (8 methods):**
- `tensor.matmul` — GEMM/GEMV
- `tensor.add` / `tensor.sub` / `tensor.mul` — elementwise ops
- `tensor.transpose` — shape manipulation
- `stats.mean` — statistical reduction (exp094 validates parity)
- `stats.variance` / `stats.std` — statistical reductions

**Category C — ML Ops (5 methods):**
- `nn.softmax` / `nn.gelu` / `nn.relu` — activation functions
- `nn.layer_norm` — normalization
- `nn.attention` — scaled dot-product attention

**Validated:** neuralSpring's exp094 probes `stats.mean` parity over IPC.
All 18 methods have local `barracuda` crate equivalents that pass validation.

**What we need:** JSON-RPC surface expansion on barraCuda's primal binary.
The existing `IpcMathClient` in neuralSpring's playGround demonstrates the
consumption pattern. primalSpring's capability registry lists these methods.

---

### Squirrel — Provider Registration (Priority: MEDIUM)

neuralSpring's inference routing (`try_squirrel_route` in `handlers.rs`) can
dispatch `inference.complete`, `inference.embed`, `inference.models` to Squirrel
via IPC. **But:** Squirrel does not yet support `inference.register_provider`,
which would allow neuralSpring to register as a provider of domain-specific
inference (spectral analysis, HMM, etc.).

**What we need:** `inference.register_provider` method on Squirrel's JSON-RPC
surface. Parameters: `{ provider_id, capabilities: [...], endpoint }`.

---

### coralReef — Wire Contract Alignment (Priority: MEDIUM)

neuralSpring's metalForge/forge has a `coralreef` feature gate with an IPC
bridge to coralReef. exp094 validates shader capability availability. But
direct `compile_shader_universal` routing through coralReef IPC is not yet
wired for production.

**What we need:** coralReef to stabilize `shader.compile.*` JSON-RPC methods
with MethodGate authorization. neuralSpring will then route `metalForge` shader
compilation through coralReef when available (honest skip when not).

---

### toadStool — Compute Dispatch Surface (Priority: MEDIUM)

neuralSpring uses `compute.dispatch.submit` for workload dispatch. exp094
validates compute dispatch health. But the full `ToadStoolClient` IPC client
for composed mode needs toadStool to stabilize its dispatch surface.

**What we need:** `compute.dispatch.submit` response schema finalization.
Hardware discovery delegation API (`compute.hardware.discover`).

---

### NestGate — Weight Tensor Storage (Priority: LOW)

neuralSpring inference requires model weight loading. NestGate provides
content-addressed storage via rhizoCrypt/loamSpine. But weight tensor storage
(large binary blobs, chunked reads) is not yet part of NestGate's API.

**What we need:** `storage.put_tensor` / `storage.get_tensor` methods that
handle large binary payloads efficiently (streaming, chunked transport).

---

### BearDog/Songbird — BTSP Session Wire (Priority: LOW)

neuralSpring probes BearDog and Songbird for discovery at startup. BTSP Phase 3
is active but neuralSpring does not yet establish its own BTSP session.

**What we need:** neuralSpring's Tower Atomic startup to negotiate a BearDog
BTSP session (`identity.session.create`) and use it for all subsequent RPC.
This is a Level 4 guideStone requirement.

---

### barraCuda — Feature Gate Bug (Priority: LOW)

Gap 9: `special::plasma_dispersion` is not feature-gated in barraCuda. When
`barracuda` is a default dependency, this isn't an issue. With `optional = true`,
any code touching `special::plasma_dispersion` will fail if `barracuda` is
disabled.

**What we need:** barraCuda to feature-gate `special::plasma_dispersion` behind
a `plasma` or `special` feature, or make it unconditionally available.

---

## Part 3: Spring Teams — What to Absorb

### For primalSpring

- **Registry cross-sync pattern**: neuralSpring's `registry_methods_in_primalspring_canonical()` test validates our capability list against your 389-method registry. Other springs should replicate this pattern. We use `include_str!("../../primalSpring/config/capability_registry.toml")` for zero-network validation.
- **exp094 replication**: Our `experiments/exp094_neuralspring_composition_parity` crate follows your `exp094`/`exp095` template. `CompositionContext::from_live_discovery_with_fallback()` is the pattern.
- **Paper notebooks → sporePrint**: 8 notebooks (72/72 checks) are ready for `primals.eco/lab/notebooks/`. Liu faculty (016-018) just completed.

### For other springs

- **barraCuda `optional = true` pattern**: `Cargo.toml` with `barracuda = { ..., optional = true }`, `[features] default = ["barracuda"]`, `barracuda = ["dep:barracuda", "dep:neural-spring-forge", "dep:wgpu"]`. Then `#[cfg(feature = "barracuda")]` on GPU modules. This enables IPC-first sovereign deployment without breaking existing default builds.
- **Benchmark analysis**: `specs/BENCHMARK_ANALYSIS.md` has a benchmark gap roadmap. Kokkos needs matched-hardware runs. Polybench/GPU linear algebra subset maps to barraCuda but stencils are out of scope.

## Part 4: Downstream Absorption (projectNUCLEUS + foundation)

### projectNUCLEUS

neuralSpring's 4 deploy graphs are ready for projectNUCLEUS consumption:
- `neuralspring_deploy.toml` — base primal deployment
- `neuralspring_inference_pipeline.toml` — Squirrel-mediated inference chain
- `neuralspring_spectral_analysis.toml` — spectral/Anderson analysis pipeline
- `composition/neuralspring_math_pipeline.toml` — barraCuda/toadStool math

**Absorption target:** projectNUCLEUS should include neuralSpring's deploy graphs
in its `graphs/` directory, or reference them from the `downstream_manifest.toml`.

**neuralAPI from biomeOS:** neuralSpring's inference capabilities (`inference.complete`,
`inference.embed`, `inference.models`) are designed for biomeOS neural API routing.
When biomeOS exposes a neural API endpoint, neuralSpring can serve as the compute
backend via JSON-RPC over UDS.

### foundation

neuralSpring contributes to foundation threads:
- **Thread 5 (Evolutionary Biology)**: Papers 011-015 (Dolson), 024-025 (Anderson/Campbell)
- **Thread 7 (Anderson Mathematics)**: Papers 022-023 (Kachkovskiy), Paper 001 (Anderson localization)

8 publishable notebooks ready. Data source manifests in foundation should reference
neuralSpring's `experiments/results/paper-baselines.json` for validation metadata.

## Part 5: Codebase Health

| Metric | Value |
|--------|-------|
| Workspace tests | 1,432 (1,279 lib + 73 forge + 80 playGround) |
| Clippy | 0 (pedantic + nursery + cast deny) |
| Unsafe | 0 (`#![forbid(unsafe_code)]`) |
| TODO/FIXME | 0 |
| Mocks in production | 0 |
| `#[allow()]` | 0 (all `#[expect(reason)]`) |
| `.unwrap()` in lib | 0 |
| Files >800L | 0 |
| Paper notebooks | 8 (72/72 checks, 2 faculties) |
| Deploy graphs | 4 |
| Experiment crates | 1 (exp094) |
| guideStone | Level 3 (29/29 bare checks) |
| Named tolerances | 233 |
| Edition | 2024, MSRV 1.87 |
| barraCuda | v0.3.13 (optional, default-enabled) |

## Part 6: Archive Sweep Results

| Check | Result |
|-------|--------|
| Stale TODOs in `.rs` | 0 (1 historical note in doc comment, not actionable) |
| Stale TODOs in `.py` | 0 |
| `.bak` / `.old` / `.tmp` / `.orig` | 0 |
| Orphan scripts | 0 |
| Archivable assessment files | 2 candidates in `specs/coral_forge_assessment/` (download + eval scripts — retain for provenance) |
| Dead code modules | 0 |
| `metalForge/fossils/` | Already archival by design — no action needed |

---

*License: AGPL-3.0-or-later*
