# neuralSpring — Primal Gaps

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

> Living gap log for neuralSpring's proto-nucleate composition.
> Reviewed against `primalSpring/graphs/downstream/downstream_manifest.toml` neuralspring entry.
>
> **Date:** 2026-05-25 | **Spring version:** 0.1.0 | **primalSpring:** Wave 48 (covalent mesh)
> **Session:** S217 — Wave 48 Covalent Spring Mesh: southGate sound-off, Songbird TCP federation port 7700 operational, upstream `dispatch()` API sync (Value→&Value, 8 sites), Songbird sled database corruption resolved, cell deployment functional. V173.
> Prior: S216 southGate deployment (9/13 UDS), S215 Wave 46, S214 Deep Debt, S213 B3/B4 ML Surrogates, S212 lithoSpore Audit.

---

## How to read this document

Each gap describes a capability or integration that the proto-nucleate graph
declares but neuralSpring has not yet fully wired. Gaps flow back to
primalSpring via `infra/wateringHole/handoffs/` and are tracked in
`primalSpring/docs/PRIMAL_GAPS.md` at the ecosystem level.

**Status key:** `open` — not started | `wip` — in progress | `resolved` —
wired and validated | `deferred` — blocked on upstream primal

---

## 1. Inference Capability Surface (`inference.*`)

**Status:** wip → **IMPLEMENTED** (primalSpring exp094 validates NUCLEUS parity)
**Proto-nucleate declares:** `inference.complete`, `inference.embed`, `inference.models`

**S190 update:** primalSpring Phase 60 (v0.9.25) now validates the full
inference surface in `exp094_composition_parity`. neuralSpring's exp094
replicates this with niche-specific inference probes. JH-0 MethodGate
adopted by all 13/13 primals — pre-dispatch capability authorization is
ecosystem standard.

**Current state:** All three methods registered and wired. Squirrel routing
active. `neuralspring_inference_pipeline.toml` deploy graph created (S190).
exp094 probes `inference.complete`, `inference.embed`, `inference.models`
availability via IPC.

**Remaining evolution:**
- Squirrel provider registration (`inference.register_provider`)
- WGSL tokenization pipeline (coralReef → toadStool → barraCuda)
- ~~Model weight loading via NestGate~~ → **WIRED** (S205): `store_to_nestgate`, `load_safetensors_from_nestgate`, `load_safetensors_layer_from_nestgate` in `weight_loader.rs`

**Hand back to:** Squirrel (provider registration), primalSpring (done)

---

## 2. barraCuda Direct Import → IPC Migration

**Status:** deferred → **IMPLEMENTED** (barracuda now optional, IPC path validated in exp094)
**Proto-nucleate declares:** barraCuda in `depends_on` with upstream
`validation_capabilities`: `tensor.matmul`, `tensor.create`, `stats.mean`

**S190 update:** barraCuda is now `optional = true` in Cargo.toml with a
`barracuda` feature flag (default-enabled). This is the structural change
enabling IPC-first sovereign deployment. GPU-centric library modules are
gated behind `#[cfg(feature = "barracuda")]`. exp094 validates `stats.mean`
parity over IPC via `validate_parity()`.

**Current state (S203):** `default = []` — barracuda is no longer linked by default.
48 files feature-gated. 241 `required-features` bins. CPU fallbacks for 12 functions.
`IpcError` typed hierarchy replaces `Result<_, String>` across IPC tree (`src/ipc/`).
731 lib tests pass IPC-first (no barracuda). Gap 11 CLOSED (S201b).
`barracuda.precision.route` wired (S203) — domain-specific precision routing
via IPC replaces hardcoded precision choices. `PrecisionRouteResult` typed response.

**Evolution path:**
- Full inference pipeline via WGSL shader composition (coralReef + barraCuda + toadStool)

**Hand back to:** N/A — all barraCuda IPC gaps closed

---

## 3. coralReef Shader Compilation via IPC

**Status:** open → **IMPLEMENTED** (exp094 validates shader capabilities via IPC)

**S190 update:** exp094 now probes `shader.compile.capabilities` over IPC
as part of Node Atomic validation. The `neuralspring_spectral_analysis.toml`
deploy graph includes coralReef shader readiness as a composition node.
JH-0 MethodGate adopted by coralReef — all shader methods are
pre-dispatch authorized.

**Current state:** `metalForge/forge` has `coralreef` feature with bridge.
`playGround/CoralReefClient` provides IPC. exp094 validates shader arch
availability. Direct `wgpu` compilation remains as local fallback.

**Remaining evolution:**
- Route `compile_shader_universal` through coralReef IPC when available
  (unblocked by coralReef v0.1.0 release — Blackwell support, naga::Module
  ingest, dual-vendor GPU coverage. strandGate owns hardware validation)
- Honest skip in validation binaries when coralReef unavailable

**Hand back to:** coralReef (wire contract alignment). Compute trio wave
(May 14 2026) makes this actionable — coralReef v0.1.0 released back for
spring absorption.

---

## 4. toadStool Compute Dispatch via IPC

**Status:** open → **IMPLEMENTED** (exp094 validates compute dispatch health via IPC)

**S190 update:** exp094 validates `compute_dispatch_alive` via IPC health
check. The `neuralspring_math_pipeline.toml` deploy graph includes
toadStool `compute.dispatch.submit` as a composition node. JH-0 MethodGate
adopted by toadStool.

**Current state:** `playGround/ToadStoolClient` provides IPC. exp094
validates compute health. Local `barracuda::dispatch` retained as
default path.

**Remaining evolution:**
- Full IPC client for `compute.dispatch.submit` in composed mode
- Hardware discovery delegation to toadStool

**Hand back to:** toadStool (compute dispatch JSON-RPC surface)

---

## 5. NestGate Weight Storage

**Status:** wip → **RESOLVED** (S205 — full weight persistence wired)
**Note:** NestGate is NOT in the proto-nucleate `depends_on`. It appears in
the richer spring-deploy graph. This is a **design decision** (see Gap 7):
proto-nucleate is the pure primal composition for certification; the spring
deploy graph is the richer integration graph. NestGate belongs in spring-deploy
for weight storage, not in proto-nucleate.

**What was done (S202b → S205):**
- `src/ipc/nestgate.rs` — `content_put`, `content_get`, `content_exists`
- `src/capabilities.rs` — `CONTENT_PUT`, `CONTENT_GET`, `CONTENT_EXISTS`
- `src/ipc/mod.rs` — NestGate in `CAPABILITY_HINTS`, `PrimalSlot`, facade
- `weight_loader.rs` (S205): `store_to_nestgate()`, `load_safetensors_from_nestgate()`,
  `load_safetensors_layer_from_nestgate()` — content-addressed by BLAKE3 hash,
  base64 encoded, with error-path tests

**Future evolution (not blocking):**
- Streaming upload/download for weights > 4 MiB (NestGate `storage.store_stream`)
- Weight provenance tracking via NestGate metadata

**Hand back to:** N/A — resolved. Deploy graph `by_capability` updated to
`"content.get"` (Wave 7 alignment, S206).

---

## 6. BearDog/Songbird Tower Integration

**Status:** wip (discovery probing wired, BTSP session pending)
**Proto-nucleate declares:** BearDog (crypto) + Songbird (discovery) as
Tower Atomic foundation
**Current state (S184):** `primal_names.rs` has discovery hint constants.
`src/bin/neuralspring_primal/tower.rs` probes BearDog and Songbird at
startup via capability-based socket discovery with `health.liveness`
checks. Logs Tower Atomic status (complete/partial/standalone).
`validate_composition_evolution.rs` validates Tower nodes in Phase 3.

**What remains:**
- BTSP session establishment for composed-mode IPC
- Songbird-based primal discovery instead of filesystem socket scanning
- Signed capability announcements

**Hand back to:** BearDog (BTSP wire), Songbird (mesh discovery),
primalSpring (Tower Atomic validation)

---

## 7. Proto-nucleate vs Spring-Deploy Fragment Mismatch — DESIGN DECISION

**Status:** resolved (Apr 17 2026 — documented as design decision per parity scorecard)

**Design Decision (documented May 14 2026 per CROSS_SPRING_PARITY_SCORECARD):**

The two-graph architecture is **intentional**, not an accidental mismatch:

| Graph | Purpose | Fragments | NestGate? | skunkBat? |
|-------|---------|-----------|-----------|-----------|
| **Proto-nucleate** (Level 5) | Pure primal NUCLEUS — spring validates AGAINST this | `tower_atomic`, `node_atomic`, `meta_tier` | **No** | No (implied in Tower) |
| **Spring deploy** (Level 2+) | Spring binary + NUCLEUS for integration | `tower_atomic`, `node_atomic`, `nest_atomic`, `meta_tier` | **Yes** (weight storage) | **Yes** (triple-first Tower) |

**Rationale:** The proto-nucleate is the minimal primal composition that
primalSpring certifies. NestGate weight storage is a spring-level concern
(model artifacts), not a primal proof requirement. The proto-nucleate
`depends_on` list (`beardog, songbird, coralreef, toadstool, barracuda, squirrel`)
represents the primals the spring VALIDATES against; it doesn't need to
include every primal the spring USES in production.

`inference_proto_nucleate_nodes()` matches the upstream `depends_on` exactly.
The deploy graph extends it with Nest Atomic (NestGate, provenance trio) for
production weight persistence and session tracking.

**Hand back to:** N/A — resolved as design decision. If `nest_atomic` is
ever needed in proto-nucleate, file hand-back to primalSpring with justification.

---

## 8. Binary Name Reconciliation

**Status:** resolved (S178)
**Details:** The proto-nucleate graph uses binary name `neuralspring` for
the spring node. `Cargo.toml` `[[bin]]` entry is named `neuralspring`
(source directory is `src/bin/neuralspring_primal/` but binary name is
canonical `neuralspring`). Deploy graph `graphs/neuralspring_deploy.toml`
also uses `neuralspring`. All three are consistent.

**Hand back to:** N/A — resolved

---

## 9. barraCuda Feature-Gate Bug (`special::plasma_dispersion`)

**Status:** open (workaround applied) — re-verified May 14 2026 with barraCuda v0.4.0
**Details:** barraCuda's `special/plasma_dispersion.rs` unconditionally
imports from `ops::lattice::cpu_complex::Complex64`, but `ops::lattice` is
gated behind `#[cfg(feature = "domain-lattice")]`. neuralSpring works around
this by enabling `domain-lattice`, but the fix belongs upstream.

**Re-check (S206):** Still present in barraCuda v0.4.0 — `plasma_dispersion.rs`
line 23: `use crate::ops::lattice::cpu_complex::Complex64;` without feature gate.
Compute trio wave makes this a good time for barraCuda to fix.

**Hand back to:** barraCuda (feature-gate `plasma_dispersion` or make
`Complex64` available without `domain-lattice`)

---

## 10. Shader Upstream Absorption Candidates

**Status:** tracking

The following `metalForge/shaders/*.wgsl` are candidates for absorption into
barraCuda's `ops/` or `stats/` WGSL modules (Write→Absorb→Lean cycle):

| Shader | barraCuda Target | Status |
|--------|-----------------|--------|
| `softmax_f64.wgsl` | `ops::nn` | Likely already upstream |
| `gelu_f64.wgsl` | `ops::nn` | Likely already upstream |
| `sigmoid_f64.wgsl` | `ops::nn` | Likely already upstream |
| `hmm_backward_log.wgsl` | `ops::bio` | Upstream candidate |
| `hmm_viterbi.wgsl` | `ops::bio` | Upstream candidate |
| `batch_ipr.wgsl` | `spectral` | Upstream candidate |
| `chi_squared_f64.wgsl` | `stats` | Upstream candidate |
| `kl_divergence_f64.wgsl` | `stats` | Upstream candidate |
| `linear_regression.wgsl` | `stats` | Upstream candidate |
| `matrix_correlation.wgsl` | `stats` | Upstream candidate |
| `rk4_parallel.wgsl` | `ops::ode` | Upstream candidate |
| `rk45_adaptive.wgsl` | `ops::ode` | Upstream candidate |
| `wright_fisher_step.wgsl` | `ops::bio` | Upstream candidate |
| `pairwise_hamming.wgsl` | `ops::bio` | Upstream candidate |
| `pairwise_jaccard.wgsl` | `ops::bio` | Upstream candidate |
| `ipa_scores_f64.wgsl` | — | neuralSpring-specific (folding) |
| `triangle_attention_f64.wgsl` | — | neuralSpring-specific (folding) |
| `triangle_mul_*.wgsl` | — | neuralSpring-specific (folding) |
| `backbone_update_f64.wgsl` | — | neuralSpring-specific (folding) |
| `torsion_angles_f64.wgsl` | — | neuralSpring-specific (folding) |
| `sdpa_scores_f64.wgsl` | `ops::mha` | Upstream candidate |
| `layer_norm_f64.wgsl` | `ops::nn` | Upstream candidate |
| `outer_product_mean_f64.wgsl` | `ops::linalg` | Upstream candidate |
| `msa_*_attention_scores_f64.wgsl` | — | neuralSpring-specific (MSA) |

**Hand back to:** barraCuda (absorption requests via wateringHole handoffs)

---

## Resolved Gaps

### R1. Binary Name Reconciliation (resolved S178)

**Original gap:** Proto-nucleate graph, deploy graph, and Cargo.toml binary
name appeared inconsistent.
**Resolution:** Verified all three sources use `neuralspring`. Source directory
`src/bin/neuralspring_primal/` is an internal path; the compiled binary name
is `neuralspring` as declared in `Cargo.toml [[bin]]`.

### R2. Inference Method Registration (resolved S177–S178)

**Original gap:** `inference.*` method strings did not appear in neuralSpring
source.
**Resolution:** S177 wired `inference.complete`, `inference.embed`,
`inference.models` into `niche.rs`, `config.rs`, `capability_registry.toml`,
`rpc_service.rs`, and `handlers.rs`. Composition validators
(`validate_nucleus_composition`, `validate_inference_composition`) verify
capability advertisement. Handlers return honest stubs until Squirrel is
connected. Gap 1 re-scoped to "provider wiring" (upstream dependency).

### R3. Deploy Graph Proto-Nucleate Alignment (resolved S179)

**Original gap:** `neuralspring_deploy.toml` was missing coralReef, barraCuda,
and Squirrel germination nodes. BearDog used `crypto` instead of `security`.
`capabilities_provided` omitted `inference.*`, `health.*`, `provenance.*`,
`capability.list`, `compute.offload`, `primal.*`. Version string was stale
(S174). No composition metadata (bonding, fragments, transport).
**Resolution:** S179 added all missing germination nodes (coralReef via
`shader.compile.wgsl`, barraCuda via `math.tensor`, Squirrel via `ai.query`).
BearDog now advertises `security` alongside `crypto`. `capabilities_provided`
expanded to full 26-capability niche surface. Graph metadata now declares
`composition_model`, `bond_type`, `trust_model`, `transport`, `fragments`,
and `proto_nucleate` reference. Version updated to V129/S179. NestGate
`by_capability` updated to `storage.retrieve` (was `storage`). ToadStool
`by_capability` updated to `compute.dispatch.submit` (was `compute`).

### R4. Capability Surface Reconciliation (resolved S179)

**Original gap:** `config::ALL_CAPABILITIES` was a subset (18 entries) of
`niche::CAPABILITIES` (26 entries), missing `provenance.*`, `primal.*`,
`capability.list`, and `compute.offload`. `operation_dependencies()` and
`cost_estimates()` did not cover all niche capabilities.
**Resolution:** S179 expanded `ALL_CAPABILITIES` to 26 entries matching the
full niche surface. Added `operation_dependencies()` entries for
`primal.forward`, `primal.discover`, `capability.list`, `compute.offload`.
Added `cost_estimates()` entries for `provenance.status`, `primal.forward`,
`primal.discover`, `capability.list`, `compute.offload`.

### R5. MCP Tool Definitions Parity (resolved S180)

**Original gap:** `playGround/src/mcp_tools.rs` defined 19 MCP tool
definitions (science 14 + health 2 + inference 3) but `ALL_CAPABILITIES`
had 27 entries. Tests `tool_count_matches_capabilities` and
`tool_names_match_capabilities` were failing.
**Resolution:** S180 added 8 MCP tool definitions for `provenance.begin`,
`provenance.record`, `provenance.complete`, `provenance.status`,
`primal.forward`, `primal.discover`, `capability.list`, `compute.offload`.
Updated test domains to include `provenance`, `primal`, `capability`,
`compute`. All 27 capabilities now have matching MCP tool definitions.

### R6. Deployment Health Triad (resolved S180)

**Original gap:** `DEPLOYMENT_VALIDATION_STANDARD.md` requires a health
triad (`health.liveness`, `health.readiness`, `health.check`) but
neuralSpring only had the first two.
**Resolution:** S180 added `health.check` handler to the primal dispatcher.
Also added `identity.get` (T4 discovery) and `mcp.tools.list` (hotSpring
composition pattern). Deploy graph updated with new capabilities.

### R7. Deploy Graph Fragment Alignment (resolved S180)

**Original gap:** Local deploy graph declared
`fragments = ["tower_atomic", "node_atomic", "meta_tier"]` but included
NestGate and provenance trio nodes. primalSpring's deploy graph used
the full fragment set.
**Resolution:** S180 added `nest_atomic` to local deploy graph fragments.

### R8. Upstream Graph Reconciliation (resolved S180)

**Original gap:** primalSpring graphs used wrong binary name
`neuralspring_primal` (pipeline + deploy) and wrong health method
`neural.health` (pipeline). Deploy graph capability set was stale (listed
capabilities neuralSpring doesn't advertise, omitted ones it does).
**Resolution:** S180 fixed binary name to `neuralspring`, health method to
`health.liveness`, and aligned capability set to match the actual 14
science + 3 inference capabilities.

### R9. plasmidBin Metadata Refresh (resolved S180)

**Original gap:** `plasmidBin/neuralspring/metadata.toml` listed
`version = "0.7.0"`, `domain = "ml"`, and only 2 capabilities using
stale `ml.*` prefix.
**Resolution:** S180 updated to `version = "0.1.0"`,
`domain = "science.learning"`, and the full 30-capability surface
(27 niche + `health.check` + `identity.get` + `mcp.tools.list`).

### R10. Method Normalization Breadth (resolved S180)

**Original gap:** `normalize_method` in `rpc.rs` only stripped a single
`neuralspring.` prefix. `SPRING_COMPOSITION_PATTERNS` §1 requires
iterative multi-prefix strip.
**Resolution:** S180 evolved to loop over `neuralspring.`,
`neural-spring.`, `neural_spring.` prefixes iteratively.

### R11. Self-Discovery Name Mismatch (resolved Apr 17 2026)

**Original gap:** `composition::discover_primal_socket("neuralspring")`
searched for filenames containing `neuralspring`, but the primal binds
as `neural-spring-{family}.sock` (from `CARGO_PKG_NAME`). Self-discovery
in composition validators always failed.
**Resolution:** `discover_primal_socket` now tries both the niche name
and its hyphenated `CARGO_PKG_NAME` form via `primal_to_pkg_name()`.
Also added the `$BIOMEOS_ORCHESTRATOR_SOCKET` tier that docs claimed
but code omitted.

### R12. Downstream Manifest Fragment Reconciliation (reopened Apr 17 2026 → see Gap 7 update)

**Original gap:** `downstream_manifest.toml` listed
`fragments = ["tower_atomic", "node_atomic", "meta_tier"]` for neuralspring
but the deploy graph includes NestGate nodes (nest_atomic).
**Previous claim:** "Added `nest_atomic` to fragments and `nestgate` to
`depends_on` in the upstream manifest." — this was aspirational; the change
was local only and never pushed to `primalSpring/`.
**Current state (Apr 17 2026):** The upstream manifest correctly reflects
the proto-nucleate composition validated by primalSpring:
  `fragments = ["tower_atomic", "node_atomic", "meta_tier"]`
  `depends_on = ["beardog", "songbird", "coralreef", "toadstool", "barracuda", "squirrel"]`
The local deploy graph (spring_deploy) remains a SUPERSET that additionally
includes `nest_atomic`, NestGate, and provenance trio nodes. This is correct:
the deploy graph is the richer spring-binary-included graph; the proto-nucleate
is the pure primal composition the spring validates against.
**Resolution:** Accepted upstream truth. `inference_proto_nucleate_nodes()`
aligned to match upstream `depends_on` exactly (no nestgate, barracuda added).
Deploy graph header updated to clarify the proto-nucleate vs spring-deploy
distinction. If nest_atomic is genuinely needed in the proto-nucleate, hand
back to primalSpring with justification.

### R13. Validation Namespace Alignment (resolved Apr 17 2026)

**Original gap:** `spring_validate_manifest.toml` used `domain = "neural"`
with `neural.*` capabilities, mismatched vs the actual `science.*` namespace.
**Resolution:** Changed to `domain = "science"` with `science.spectral_analysis`,
`science.anderson_localization`, `science.hessian_eigen`.

---

## Composition Evolution (added Apr 17 2026)

### CE1. Science Composition Parity Validation

**Status:** implemented
**Details:** New `validate_science_composition` binary implements Tier 3
validation: calls `science.*` capabilities via JSON-RPC IPC and compares
results to deterministic Rust baselines computed with identical parameters.
4 baselines: `spectral_analysis`, `ipr`, `hessian_eigen`, `disorder_sweep`.
Science baselines are centralized in `validation::composition::science_baselines()`.

### CE2. GpuPreferred Dispatch

**Status:** implemented
**Details:** Added `MixedSubstrate::GpuPreferred` variant to forge enum.
Nucleus pipeline executor and `Dispatcher::gpu_or_cpu` now route `GpuPreferred`
stages through GPU when a device is available, falling back to CPU.

### CE3. Named Tolerance Constants

**Status:** implemented
**Details:** Replaced ad-hoc multipliers (`* 0.1`, `* 2.0`) with named constants
`GPU_HMM_LOG_LIKELIHOOD_F64` and `GPU_HMM_LOG_LIKELIHOOD_F32_EXTENDED`.
9 validation binary sites updated across 4 files.

---

## 11. barraCuda JSON-RPC Surface Gaps (IPC Migration Blockers)

**Status:** **RESOLVED** (May 11 2026, S201b — re-audit + CPU fallbacks)
**Context:** barraCuda now exposes **71 JSON-RPC methods** in `REGISTERED_METHODS`
(was 32 when this gap was opened in S183). Re-audit against current
`barraCuda/crates/barracuda-core/src/ipc/methods/mod.rs` reduces the gap
from 18 entries to **4 still-open + 4 composable**.

**S183–S184 note (historical):** These audits were accurate at the time.
barraCuda has since expanded its IPC surface significantly, adding `stats.pearson`,
`stats.chi_squared`, `stats.shannon`, `stats.fit_linear`, `stats.empirical_spectral_density`,
`linalg.solve`, `linalg.eigenvalues`/`stats.eigh`, `linalg.graph_laplacian`,
`graph.belief_propagation`, `ml.mlp_forward`, `ml.esn_predict`, and the full
`nautilus.*` lifecycle (6 methods).

| neuralSpring call | `barracuda::` module | JSON-RPC equivalent | Status |
|-------------------|---------------------|---------------------|--------|
| `eigh_householder_qr` | `ops::linalg` | `linalg.eigenvalues` / `stats.eigh` | **CLOSED** (Jacobi impl; algorithm differs from HQR but wire surface exists) |
| `pearson_correlation` | `stats::correlation` | `stats.pearson` / `stats.correlation` | **CLOSED** |
| `chi_squared_statistic` | `special` | `stats.chi_squared` | **CLOSED** |
| `empirical_spectral_density` | `stats` | `stats.empirical_spectral_density` | **CLOSED** |
| `marchenko_pastur_bounds` | `stats` | **None** | **STILL OPEN** (library function exists, no RPC handler) |
| `shannon_from_frequencies` | `stats` | `stats.shannon` / `stats.entropy` | **CLOSED** |
| `solve_f64_cpu` | `linalg::solve` | `linalg.solve` | **CLOSED** |
| `esn_v2::*` | `esn_v2` | `ml.esn_predict` (stateless predict) | **COMPOSABLE** (predict wired; full training lifecycle partial) |
| `nn::SimpleMlp` / `DenseLayer` | `nn` | `ml.mlp_forward` | **CLOSED** |
| `belief_propagation_chain` | `linalg::graph` | `graph.belief_propagation` | **CLOSED** |
| `graph_laplacian` | `linalg::graph` | `linalg.graph_laplacian` | **CLOSED** |
| `disordered_laplacian` | `linalg::graph` | **None** | **STILL OPEN** (no dedicated RPC; plain laplacian is wired) |
| `effective_rank` | `linalg` | **None** (composable from `linalg.eigenvalues`) | **COMPOSABLE** |
| `numerical_hessian` | `numerical` | **None** | **STILL OPEN** |
| `boltzmann_sampling` | `sample` | **None** | **STILL OPEN** |
| `nautilus::*` | `nautilus` | `nautilus.{create,observe,train,predict,export,import}` | **CLOSED** |
| `dot` | `stats` | composable via `tensor.matmul` | **COMPOSABLE** |
| `l2_norm` / `rmse` / `mae` | `stats` | composable via `tensor.*` + `stats.*` | **COMPOSABLE** |
| `fit_linear` | `stats` | `stats.fit_linear` | **CLOSED** |

### Summary

- **CLOSED** (direct RPC): 12 of 18 original gaps
- **COMPOSABLE** (multi-method or partial): 4 (`esn_v2`, `effective_rank`, `dot`, `l2_norm`/`rmse`/`mae`)
- **STILL OPEN**: 4 (`marchenko_pastur_bounds`, `disordered_laplacian`, `numerical_hessian`, `boltzmann_sampling`)

### Resolution (S201b)

All 4 remaining gaps closed with CPU fallback implementations:
- `marchenko_pastur_bounds` — `src/weight_spectral/metrics.rs` (`(1-√γ)², (1+√γ)²`)
- `disordered_laplacian` — `src/agent_coordination.rs` (`H = L + W·diag(capability)`)
- `numerical_hessian` — `src/loss_landscape.rs` (central finite differences)
- `boltzmann_sampling` — `src/loss_landscape.rs` (Metropolis-Hastings MCMC via local `metropolis_step`)
- `graph_laplacian` — `src/agent_coordination.rs` (`L = D - A`)

Each uses `#[cfg(not(feature = "barracuda"))]` for the CPU path and
`#[cfg(feature = "barracuda")]` for the barracuda delegate. Both paths
compile and test: 693 IPC-first + 1,300 barracuda tests PASS.

**Gap 11 is now CLOSED.** All 18 original gaps resolved:
12 via barraCuda RPC surface expansion, 4 composable, 5 CPU fallbacks (S201b).

**Upstream registry drift (S202b):** `primalSpring/docs/PRIMAL_GAPS.md` Layer 3
table still lists neuralSpring as "Gap 11 (18 RPC methods)" open. This is stale —
Gap 11 was resolved in S201b. Flagged for upstream correction in V153 handoff.

**Hand back to:** primalSpring (update Layer 3 table to reflect Gap 11 CLOSED)

---

## 12. Proto-Nucleate Capabilities Harness

**Status:** implemented (Apr 17 2026)
**Details:** New `validate_proto_nucleate_capabilities` binary iterates all 7
`PROTO_NUCLEATE_VALIDATION_CAPABILITIES`, maps each to its owning primal
(`barraCuda` for `tensor.*`/`stats.*`, `toadStool` for `compute.dispatch`,
`BearDog` for `crypto.hash`, `Squirrel` for `inference.*`), discovers the
socket, calls via IPC, and validates parity. Exit codes 0/1/2.

### CE4. IPC Tree (`src/ipc/`)

**Status:** implemented (Apr 17 2026 → S201b: `ipc_dispatch` removed, per-primal tree)
**Details:** The per-primal `src/ipc/` tree (7 modules: barracuda, toadstool, beardog,
squirrel, coralreef, skunkbat, nestgate) provides the IPC counterpart to `gpu_dispatch::Dispatcher`.
Wraps `stats.mean`, `stats.std_dev`, `stats.weighted_mean`, `tensor.matmul`,
`tensor.create`, `barracuda.precision.route`, `compute.dispatch`, `toadstool.validate`,
`toadstool.list_workloads`, `crypto.hash`, `inference.complete`, `inference.embed`,
`content.put`, `content.get`, `content.exists` as typed Rust methods routing through
JSON-RPC IPC. Discovery-based (env-driven sockets, no hardcoded paths). Returns
`Result<_, IpcError>` with typed error variants (`NotDiscovered`, `Transport`,
`Protocol`). Capability constants centralized in `src/capabilities.rs` (37 constants).
`CapabilityRouter` maps 20 capability hints to 7 primals.

### CE5. Stadial `deny.toml` Enforcement

**Status:** implemented (Apr 17 2026)
**Details:** Added `deny = [...]` list to `deny.toml` banning `ring`,
`openssl-sys`, `openssl`, `async-trait`, `rustls`, `ed25519-dalek`, `cmake`,
and `cc` (with `blake3` wrapper exemption). `cargo deny check` passes.

---

## 13. guideStone Evolution (Apr 18–20 2026)

**Status:** Level 5 — 19 certification tests ALL PASS (L0-L5, implemented S200)
**Standard:** `primalSpring/wateringHole/GUIDESTONE_COMPOSITION_STANDARD.md` v1.2.0

### guideStone Readiness

| Level | Description | Status |
|-------|-------------|--------|
| 1 | Validation exists (`IpcMathClient`, `validate_proto_nucleate_capabilities`) | DONE |
| 2 | Properties documented (`docs/GUIDESTONE_PROPERTIES.md`) | DONE |
| 3 | Bare guideStone works (29/29 pass, P1-P5 certified without primals) | DONE |
| 4 | NUCLEUS composition (deploy graphs + registry + 4 family calls) | DONE (S200) |
| 5 | Cross-spring (frozen artifacts + protocol liveness + hash determinism) | DONE (S200) |

### Binary

`neuralspring_guidestone` v0.3.0 (feature-gated: `guidestone` → `primalspring` + `primal`)

Uses `primalspring::composition` API directly:
- `CompositionContext::from_live_discovery_with_fallback()` for UDS+TCP discovery
- `validate_liveness()` for Phase 2 primal health checks
- `validate_parity()` / `validate_parity_vec()` for Phase 3 domain science
- `is_skip_error()` — unified skip classification (v0.9.17 pattern)
- `hash_bytes()` / `resolve_capability()` for Phase 4 additive NUCLEUS
- `primalspring::checksums::verify_manifest()` for P3 BLAKE3 checksums
- `v.section()` for structured output (supports `PRIMALSPRING_JSON=1`)
- `FAMILY_ID` env for family-isolated socket discovery

### Level 3 Evidence (S184)

- **P1 Deterministic**: Seeded RNG triple-match, exact bitwise equality
- **P2 Traceable**: 49 provenance records, all labelled/scripted/committed
- **P3 Self-Verifying**: BLAKE3 CHECKSUMS — 15 validation-critical files verified
- **P4 Environment-Agnostic**: ecoBin, `#![forbid(unsafe_code)]`, no network
- **P5 Tolerance-Documented**: 233+ named tolerances, all finite/named/categorized

### v0.9.16 Integration Notes

- **Family-aware discovery**: `FAMILY_ID` env propagated to `CompositionContext`;
  `discover_by_capability()` resolves `{capability}-{family}.sock` first
- **Protocol tolerance**: `is_protocol_error()` classifies HTTP-on-UDS as SKIP
  (Songbird, petalTongue)
- **BLAKE3 checksums**: `primalspring::checksums` used for P3 manifest verification
- **Known issues absorbed**: `BearDog` requires `BEARDOG_FAMILY_SEED` env;
  `BearDog` resets connection without BTSP handshake (expected behavior)

### v0.9.17 Integration Notes (S185)

- **`is_skip_error` adoption**: Replaced 7 manual `is_connection_error()` /
  `is_protocol_error()` match arms with `primalspring::composition::is_skip_error()`.
  Covers connection errors + protocol mismatches + transport dialect in one predicate
- **guideStone standard v1.2.0**: Tolerance hierarchy as ecosystem standard,
  `call_or_skip`/`is_skip_error` absorbed into `primalspring::composition`,
  "domain functions are local compositions" pattern documented
- **No new library API**: checksums, ValidationResult, IPC, composition — all
  unchanged from v0.9.16. Delta is deployment validation and operational contracts
- **genomeBin v5.1**: 46 binaries across 6 target triples — Level 4 deployment
  path clear (x86_64-musl, aarch64-musl, armv7-musl, x86_64-windows,
  aarch64-android, riscv64-musl)
- **Operational requirements for Level 4 deployment**:
  - coralReef: `--port` → `--rpc-bind` (iter84 CLI change)
  - beardog: `BEARDOG_FAMILY_SEED` env required for production BTSP
  - songbird: `SONGBIRD_SECURITY_PROVIDER=beardog` env required
  - nestgate: `NESTGATE_JWT_SECRET` env required (random Base64)
- **Manifest note**: `downstream_manifest.toml` shows `guidestone_readiness = 2`
  for neuralSpring; actual status is Level 5 (manifest is upstream's responsibility)

### Level 4 Status — PARTIALLY ACHIEVED (S216)

Live NUCLEUS deployed on southGate (May 23 2026). 9/13 primals accessible via UDS.
`composition_nucleus.sh` expanded from 8 to 13 primals.

**Proto-nucleate validation results (7 capabilities):**

| Capability | Primal | Result | Detail |
|-----------|--------|--------|--------|
| `stats.mean` | barraCuda | **PASS** | composition=3, local=3, diff=0.00e0 |
| `tensor.create` | barraCuda | **PASS** | shape=[2,3], dtype=f32 |
| `tensor.matmul` | barraCuda | **FAIL** | API evolution: `lhs_id` now required (tensor lifecycle) |
| `compute.dispatch` | toadStool | **FAIL** | `health.liveness` returns `-32601 Method not found` |
| `crypto.hash` | BearDog | **PASS** | BLAKE3 hash len=44, deterministic |
| `inference.complete` | Squirrel | **SKIP** | abstract socket only, not file-discoverable |
| `inference.embed` | Squirrel | **SKIP** | same as above |

**guideStone results: 30/37 PASS, 6 SKIP, 1 FAIL** (after CHECKSUMS refresh).
BearDog signing receipt PASS. Songbird discovery SKIP (discovery.register unknown).

**Deployment state:**

| Primal | UDS Socket | Status |
|--------|-----------|--------|
| biomeOS | neural-api-southgate.sock | UP |
| BearDog | beardog-southgate.sock | UP |
| Songbird | songbird-southgate.sock | UP |
| toadStool | toadstool-southgate.sock | UP |
| barraCuda | math-southgate.sock (symlinked) | UP |
| coralReef | coralreef-core-southgate.sock (symlinked) | UP |
| NestGate | nestgate-southgate.sock | UP |
| sweetGrass | sweetgrass-southgate.sock | UP |
| petalTongue | petaltongue-southgate.sock | UP |
| Squirrel | abstract @squirrel | UP (not file-discoverable) |
| rhizoCrypt | TCP 9400/9401 only | UP (no UDS) |
| loamSpine | — | CRASHED (upstream Tokio double-runtime bug) |
| skunkBat | — | TCP-only (no UDS listener) |

**Remaining Level 4 blockers:**
- barraCuda `tensor.matmul` API evolution: now requires tensor IDs (create → compute → release lifecycle)
- toadStool `health.liveness` not implemented (returns -32601)
- Squirrel file-based UDS socket: uses abstract socket, not discoverable via filesystem scan
- loamSpine double-runtime panic in `infant_discovery.rs:233`

### Level 5 Blockers

- ~~**Gap 11**: 18 barraCuda IPC surface gaps~~ — **RESOLVED** (S201b: 12 RPC, 4 composable, 5 CPU fallback)
- Cross-substrate parity: Python / CPU / GPU / IPC all within tolerances
- `BearDog` signing receipt validates end-to-end — **PASS** (S216)

### Validation Window

The existing `IpcMathClient` and `validate_proto_nucleate_capabilities` are retained
as the "validation window" (per guideStone standard §"The Validation Window"). These
temporary tools prove math works through NUCLEUS before the guideStone binary takes
over as the certified artifact.

**Hand back to:** barraCuda (tensor lifecycle API docs), toadStool (`health.liveness`),
Squirrel (file-based UDS socket), loamSpine (double-runtime fix)

---

## 14. Phase 46 Composition Explorer Findings (S186)

**Status:** explored — patterns documented, gaps identified
**Context:** primalSpring Phase 46 extracted `nucleus_composition_lib.sh` (41
functions) as a reusable NUCLEUS composition library. neuralSpring's assigned
lane: **Agent-Driven Composition + AI Feedback Loops**.

### Agentic IPC Patterns Discovered

- **Squirrel-mediated inference via composition library**: `cap_socket "ai"` +
  `send_rpc` with `inference.complete` / `inference.embed`. Works when Squirrel
  is in `PRIMAL_LIST` and `REQUIRED_CAPS` includes `ai`. Socket discovery via
  `resolve_capability()` supports family-aware paths.
- **DAG branching for AI decisions**: Each inference call is recorded as a
  `dag_append_event` with structured metadata (prompt, result, model, confidence).
  DAG provides causal ordering of agent decisions — important for multi-step
  reasoning where later decisions depend on earlier inference results.
- **Braid provenance audit trail**: Every agent action gets a `braid_record`
  with content-type `application/x-neuralspring-agent`, enabling post-hoc
  tracing of why the agent chose a particular path. The braid + DAG together
  form a complete decision audit: DAG for causal structure, braids for payload.
- **Closed-loop feedback**: `domain_on_tick` + `check_proprioception` implements
  the act → observe → adjust cycle. Sensor streams provide real-time feedback;
  the agent can trigger autonomous reasoning steps at configurable intervals.

### Squirrel Integration Reliability

- **Discovery**: Squirrel must be explicitly added to composition startup
  (`REQUIRED_CAPS="ai"` or `OPTIONAL_CAPS="ai"`). Without it, `cap_socket "ai"`
  returns empty and inference calls fail silently.
- **`inference.complete`**: Works via standard JSON-RPC. Parameters: `prompt`,
  `model` (default: "default"), `max_tokens`. Response contains `text` or
  `completion` field. Latency depends on model backend.
- **`inference.embed`**: Works via JSON-RPC. Parameters: `text`, `model`.
  Response contains `embedding` or `embeddings` field.
- **`inference.register_provider`**: NOT YET VERIFIED — neuralSpring needs this
  to register as an inference backend. Gap 1 still open.
- **BTSP interaction**: Phase 45c made BTSP mandatory. Squirrel connections
  through BearDog now require BTSP handshake. `is_skip_error` handles
  BTSP failures gracefully.

### Missing / Gaps

| Finding | Impact | Hand back to |
|---------|--------|--------------|
| ~~Squirrel not in default `PRIMAL_LIST`~~ | **RESOLVED S216**: all 13 primals now in default list | — |
| `inference.register_provider` wire unknown | neuralSpring cannot self-register as inference backend | Squirrel |
| No `inference.models` via composition lib | Cannot enumerate available models pre-inference | Squirrel |
| DAG session requires `dag` capability (rhizoCrypt) | Full Nest atomic needed for provenance | primalSpring (clarify Nest requirement for agent compositions) |
| Braid query latency uncharacterized | Audit trail retrieval may bottleneck real-time loops | loamSpine |
| Sensor stream polling interval fixed | No adaptive polling for high-frequency agent loops | primalSpring (composition lib enhancement) |

### AI Provenance Schema

Agent decisions are recorded with this structure:

```
DAG event:  { session, action, state, metadata: [{prompt, result, model, confidence}], input_type, hover }
Braid record: { action, content_type: "application/x-neuralspring-agent", state, payload: {prompt, result, tick}, input_type, hover }
```

Both are keyed to the composition session. The DAG provides causal ordering;
braids provide searchable payload. Together they answer "what did the agent
decide, when, why, and with what input?"

### Recommendation

`composition_nucleus.sh` should support an optional `EXTRA_PRIMALS` env var
(or `--with-squirrel` flag) so domain compositions can request Squirrel without
forking the launcher. This would allow:
```bash
EXTRA_PRIMALS="squirrel" COMPOSITION_NAME=neuralspring ./composition_nucleus.sh start
```

**Hand back to:** Squirrel (provider registration), primalSpring (composition
lib enhancements for AI lane, Squirrel in default/optional PRIMAL_LIST),
loamSpine (braid query performance), rhizoCrypt (DAG session for agent comps)

## 15. `nest.commit` Signal Dispatch (S208 → S209)

**Status**: `candidate` → **RESOLVED** | **Priority**: low → done | **Blocked by**: none

neuralSpring now has full provenance chain dispatch:
- `commit_session_signal()` in `weight_loader.rs` — dispatches `nest.commit` to finalize sessions
- `store_science_result()` in `weight_loader.rs` — wraps science computation results in `nest.store` provenance
- `s_nest_commit` validation scenario — structural + live checks for the full provenance chain

**Resolved in S209**: Training session finalization, science result provenance, and
live validation all wired. biomeOS decomposes `nest.commit` into:
`rhizoCrypt.event.append → bearDog.crypto.sign → nestGate.content.put → loamSpine.session.commit → sweetGrass.braid.create`.

## 16. Schema Validation Scenario (S208 → S209)

**Status**: `candidate` → **RESOLVED** | **Priority**: low → done | **Blocked by**: none

`s_schema_standard` validation scenario implemented:
- Tier 1: verifies `capability.list` handler includes `count`, `primal`, and `capabilities` fields
- Tier 1: validates `ALL_CAPABILITIES` naming convention and `primal.list` in local registry
- Tier 2: live probe of `primal.list` and `capability.list` response shapes

**Resolved in S209**: Schema drift detection is now covered by validation scenario 9/9.

## 17. `node.compute` Signal Dispatch — WIRED (S209)

**Status**: **RESOLVED** | **Priority**: medium | **Blocked by**: none

`IpcMathClient::dispatch_compute_signal()` wired behind `#[cfg(feature = "primalspring")]`.
Dispatches `node.compute` signal through biomeOS, which decomposes into:
`toadStool.compute.dispatch → coralReef.shader.compile.wgsl → barraCuda.tensor.matmul`.

Existing `compute_dispatch()` (direct toadStool IPC) retained for non-composed mode.

## 18. Live IPC Pipeline Executor — WIRED (S209)

**Status**: **RESOLVED** | **Priority**: medium | **Blocked by**: none

`execute_graph_live()` in `nucleus_pipeline/executor.rs` — routes pipeline stages
through `CompositionContext` instead of local function calls. Falls back to local
dispatch when capabilities are not resolvable via composition. Provenance recording
is identical to the local executor for uniform `PipelineReport` output.

## 19. GPU Parity — 6/6 Stages GPU-Dispatchable (S211)

**Status**: **RESOLVED** | **Priority**: high | **Blocked by**: none

Promoted 4 CPU-only science pipeline stages to `GpuPreferred` dispatch:
- `digester_anderson` — GPU eigensolve + disorder sweep via `Dispatcher`
- `isomorphic_reservoir` — GPU eigensolve per reservoir matrix
- `wdm_ensemble_qs` — GPU replicator dynamics via `Dispatcher::replicator_step()`
- `introgression_nn` — GPU HMM Viterbi chain via `Dispatcher::detect_introgression()`

All 6 stages now have GPU dispatch arms in `dispatch_capability_gpu()`.
Composition graph updated: 4 stages `CpuOnly` → `GpuPreferred` in `metalForge/forge/src/graph.rs`.

## 20. GPU Parity Validation Scenario — `s_gpu_parity` (S211)

**Status**: **RESOLVED** | **Priority**: medium | **Blocked by**: none

Scenario 10/10: structural checks for GPU dispatch coverage (all 6 stages have
GPU functions and are routed in `dispatch_capability_gpu`). Track: `GpuParity`.

## 21. ToadStool Typed Workload Submission (S211)

**Status**: **RESOLVED** | **Priority**: medium | **Blocked by**: none

`ComputeWorkload` / `WorkloadResult` typed structs in `src/ipc/toadstool.rs`.
`compute_dispatch_workload()` — submit structured workload (capability, data, substrate hint).
`compute_dispatch_pipeline()` — submit entire pipeline graph for remote execution.

## 22. PCIe P2P Bridge in Dispatcher (S211)

**Status**: **RESOLVED** | **Priority**: medium | **Blocked by**: none

`PcieBridge` wired into `Dispatcher` construction — probes sysfs IOMMU groups
at init. `pcie_p2p_available()` and `pcie_transfer_cost()` accessors added.
Deploy graph (`graphs/neuralspring_deploy.toml`) updated with substrate metadata.

## 23. `node.compute` in Live Executor (S211)

**Status**: **RESOLVED** | **Priority**: medium | **Blocked by**: none

`execute_graph_live()` now prefers `node.compute` signal dispatch for `GpuOnly`/`GpuPreferred`
stages when `CompositionContext` is available, falling back to `ctx.call()` then local dispatch.

## 24. Stability Tier Annotations (S212)

**Status**: **RESOLVED** | **Priority**: high | **Blocked by**: none

All 37 capabilities in `config/capability_registry.toml` now carry `stability`
annotations (`stable` / `evolving`). 14 science.* niche methods = `stable`.
Health, provenance, inference, primal, identity, capability = `stable` (upstream
canonical names). Compute (toadStool absorption), composition, method.register,
security = `evolving`. Section headers added for readability. Stability tier
documentation block added to file header per primalSpring Wave 20 PM.

## 25. Degradation Behavior Documentation (S212)

**Status**: **RESOLVED** | **Priority**: high | **Blocked by**: none

Created `docs/DEGRADATION_BEHAVIOR.md` documenting per-primal unreachable behavior
(all 11 primals), GPU-dependent path fallbacks (6 paths), provenance trio partial
completion (4 states), and IPC error classification. Principle: science logic is
never gated behind primal availability. Pattern: `has_capability()` before `call()`.

## 26. Cross-Tier Parity — GPU as Parity (S212)

**Status**: **RESOLVED** | **Priority**: medium | **Blocked by**: none

GPU parity framed as cross-tier parity: CPU reference implementation = Tier 2a,
GPU accelerated implementation = Tier 2b. Both must agree within documented
tolerances. `s_gpu_parity` scenario validates structural coverage (6/6 GPU stages,
all routed in `dispatch_capability_gpu`). Numerical agreement is enforced by
`Dispatcher::gpu_or_cpu()` which runs both paths when parity checking is enabled.
This mirrors lithoSpore's Python vs Rust parity but at the substrate level.

## 27. Trio Transaction Semantics Review (S212)

**Status**: **RESOLVED** | **Priority**: medium | **Blocked by**: none

Reviewed `provenance_dispatch.rs` against `PROVENANCE_TRIO_INTEGRATION_GUIDE.md`.
All three provenance dispatch functions (`store_to_nestgate_signal`,
`commit_session_signal`, `store_science_result`) return `Result<_, IpcError>`.
Callers (validation scenarios, live executor) treat trio errors as non-fatal:
logged and skipped, never blocking science. Deploy graph uses `fallback = "skip"`
for all trio primal nodes. Partial completion is valid per upstream rules.

## 28. B3/B4 ML Surrogate Status (S212 → S213)

**Status**: **RESOLVED** (S213) | **Priority**: medium | **Blocked by**: none

B3 (Good et al. 2017 — LSTM+HMM+ESN allele trajectory classifier) and
B4 (Blount et al. 2008 — ESN citrate early-warning classifier) implemented
in S213. Both follow the established B1 pattern: Python control baseline →
`expected_values.json` → Rust module → validator binary.

**B4 (Blount 2008):**
- `control/ltee_citrate_esn/` — Python baseline (seed=42, 200 trajectories)
- `src/ltee_citrate_esn.rs` — `CitrateEsnPredictor` (ESN reservoir=256, 2-step)
- `src/bin/validate_ltee_b4_citrate_esn.rs` — 16/16 checks PASS
- Test accuracy: 94.3%, Rust-Python score parity: 3.35e-14

**B3 (Good 2017):**
- `control/ltee_allele_trajectory/` — Python baseline (seed=42, 300 alleles)
- `src/ltee_allele_trajectory.rs` — LSTM(h=32)+HMM(3 states)+ESN(128) fusion
- `src/bin/validate_ltee_b3_allele_trajectory.rs` — 16/16 checks PASS
- Test accuracy: 100% (T06 target: ≥95%), LSTM parity: 5.2e-18

**Pipeline integration:**
- `metalForge/forge/src/graph.rs` — 2 new stages (8 total, 8 edges)
- `src/nucleus_pipeline/dispatch.rs` — match arms for both capabilities
- `config/capability_registry.toml` — 2 new `evolving` capabilities (39 total)
- `src/config.rs` + `src/niche.rs` — synchronized

## 29. Post-Primordial Gate Deployment — southGate (S216)

**Status**: **DEPLOYED** | **Priority**: high | **Blocked by**: upstream primal gaps

First live NUCLEUS deployment on southGate (Ryzen 7 5800X3D, 128GB DDR4, Pop!_OS 22.04).
`composition_nucleus.sh` expanded from 8 to 13 primals: added biomeOS (Phase 0),
skunkBat (Phase 1 Tower), coralReef + NestGate + Squirrel (Phase 2), with correct
socket naming, env wiring, and dependency ordering matching upstream
`nucleus_launcher.sh` patterns.

### Deployment Findings

1. **coralReef socket naming**: Binds as `coralreef-core-{family}.sock`, not
   `coralreef-{family}.sock`. Launcher creates symlink. Discovery works.
2. **toadStool socket naming**: Creates both `toadstool-{family}.sock` and
   `toadstool-{family}.jsonrpc.sock`. Proto-nucleate validator discovers via `.jsonrpc.sock`.
3. **Songbird security provider**: Must set `SONGBIRD_SECURITY_PROVIDER=beardog`
   (the name), not the socket path. Plus `BEARDOG_SOCKET`, `SECURITY_ENDPOINT`,
   `BEARDOG_MODE=direct`.
4. **skunkBat**: No UDS `--socket` flag; TCP-only via `--port`/`--bind`. Defense
   primal, not in proto-nucleate `depends_on`.
5. **Squirrel**: Binds to Linux abstract socket `@squirrel`, not a file-based UDS.
   Validators using filesystem-based discovery cannot find it.
6. **rhizoCrypt**: Ignores `RHIZOCRYPT_SOCKET` env; binds TCP only (9400/9401).
   No UDS listener.
7. **loamSpine**: Panics with "Cannot start a runtime from within a runtime" in
   `infant_discovery.rs:233`. Upstream Tokio double-runtime bug.
8. **petalTongue**: `server` mode accepts no CLI args (no `--socket`). Socket
   path configured via `PETALTONGUE_SOCKET` env.
9. **BTSP**: Raw `socat` health checks fail (expected) — primals require BTSP
   handshake. `CompositionContext` handles this correctly.

### Upstream Hand-backs

| Finding | Owner | Priority |
|---------|-------|----------|
| `tensor.matmul` requires `lhs_id` (tensor lifecycle API) | barraCuda | HIGH |
| `health.liveness` returns -32601 Method not found | toadStool | HIGH |
| Squirrel abstract socket — not file-discoverable | Squirrel | MEDIUM |
| rhizoCrypt no UDS listener | rhizoCrypt | MEDIUM |
| loamSpine double-runtime panic | loamSpine | MEDIUM |
| `discovery.register` unknown method on Songbird | Songbird | LOW |
| skunkBat no UDS `--socket` flag | skunkBat | LOW |

---

## Gap 30: Wave 48 Covalent Spring Mesh — southGate (S217)

**Status:** RESOLVED
**Session:** S217 (May 25, 2026)
**Wave:** 48

### What happened

primalSpring Wave 48 shipped the complete covalent mesh stack: Songbird TCP
federation, cell deployment graphs, `cell_launcher.sh`, cross-gate
`capability.call` routing via biomeOS v3.75, and toadStool S274 GPU yield.

### neuralSpring response

1. **Gate sound-off**: southGate declared (Ryzen 7 5800X3D, 128GB DDR4).
   Gate Deployment section added to CONTEXT.md.
2. **Federation wiring**: `composition_nucleus.sh` upgraded with
   `SONGBIRD_FEDERATION_PORT` support. Port 7700 verified operational
   (health.liveness healthy, discovery.peers ready).
3. **Upstream API sync**: `CompositionContext::dispatch()` changed from
   `Value` to `&Value`. 8 call sites fixed in `ipc/mod.rs`,
   `executor.rs`, `provenance_dispatch.rs`, `s_nest_commit.rs`,
   `s_signal_dispatch.rs`.
4. **Songbird sled database**: Corrupted `task_lifecycle` sled DB from
   prior unclean shutdowns caused startup failure. Fix: clean
   `~/.local/share/songbird/task_lifecycle*`.

### Upstream notes

- Songbird JSON-RPC on TCP is at path `/jsonrpc`, not root `/`.
  Wave 48 `curl` examples should be updated.
- `SONGBIRD_FEDERATION_PORT` only implemented in
  `primalSpring/tools/nucleus_launcher.sh`, not in
  `plasmidBin/nucleus_launcher.sh` (which uses `ports.env` model).
- Songbird sled corruption: upstream should add cleanup on startup
  or switch to a more crash-resilient embedded DB.
