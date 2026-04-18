# neuralSpring — Primal Gaps

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

> Living gap log for neuralSpring's proto-nucleate composition.
> Reviewed against `primalSpring/graphs/downstream/downstream_manifest.toml` neuralspring entry.
>
> **Date:** 2026-04-17 | **Spring version:** 0.1.0 | **primalSpring:** v0.9.15
> **Session:** S181+ — Audit remediation: manifest reconciliation, composition
> parity validation (Tier 3: Rust→IPC), discovery bug fix, GpuPreferred dispatch,
> tolerance hygiene, science baselines for composition round-trip validation.

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

**Status:** wip (Squirrel routing wired, provider registration pending)
**Proto-nucleate declares:** `inference.complete`, `inference.embed`, `inference.models`
**Current state (S181):** All three method strings are registered, wired,
and now route through Squirrel when discovered:
- `src/niche.rs`: `CAPABILITIES` array includes all three `inference.*` methods,
  with `operation_dependencies()` and `cost_estimates()` entries.
- `src/config.rs`: `ALL_CAPABILITIES` includes `inference.*` (unit-tested
  against `niche::CAPABILITIES`).
- `config/capability_registry.toml`: All three declared with descriptions.
- `src/rpc_service.rs`: tarpc service trait defines `inference_complete`,
  `inference_embed`, `inference_models` with typed request/response structs.
- `src/bin/neuralspring_primal/handlers.rs`: JSON-RPC handlers now attempt
  Squirrel discovery via `try_squirrel_route()` — if Squirrel is running,
  inference requests are forwarded. Falls back to stub with
  `"status": "squirrel_unavailable"` when Squirrel is not present.
- Composition validators (`validate_inference_composition`,
  `validate_nucleus_composition`, `validate_composition_evolution`) verify
  capability advertisement and live IPC probing.

**What remains:**
- Squirrel provider registration (`inference.register_provider`) so Squirrel
  discovers neuralSpring as an inference backend
- WGSL tokenization pipeline (forward pass as shader composition through
  coralReef → toadStool → barraCuda)
- Model weight loading via NestGate (`storage.retrieve`)

**Blocked on:**
- Squirrel `inference.register_provider` wire — does this method exist yet?
- coralReef pipeline compilation for multi-step ML graphs
- NestGate `storage.retrieve` for weight tensors

**Hand back to:** Squirrel (provider registration wire), primalSpring
(validate composition once wired)

---

## 2. barraCuda Direct Import → IPC Migration

**Status:** deferred
**Proto-nucleate declares:** barraCuda as IPC node (`math.tensor`,
`math.stats`, `math.activation`, `math.noise` capabilities)
**Current state:** barraCuda is a compile-time `path` dependency. All math
is called via direct Rust imports (`barracuda::ops::*`, `barracuda::nn::*`,
`barracuda::dispatch::*`).

**What is needed:**
- Capability-based IPC client for `math.*` operations
- Gradual migration: keep direct import for validation baselines, add IPC
  path for composition mode
- Feature gate: `#[cfg(feature = "composed")]` for IPC paths vs direct

**Rationale for deferral:** Direct import is correct at the current maturity
stage (validation). IPC migration happens when biomeOS orchestrates the full
graph and neuralSpring runs as a composed node rather than a standalone
validator.

**Hand back to:** barraCuda (expose `math.*` via JSON-RPC), primalSpring
(composition validation cell in `deployment_matrix.toml`)

---

## 3. coralReef Shader Compilation via IPC

**Status:** open
**Proto-nucleate declares:** coralReef node with `shader.compile.wgsl`,
`shader.compile.spirv` capabilities
**Current state:** `metalForge/forge` has an optional `coralreef` feature
with a `coralreef_bridge` module. `playGround` has a `CoralReefClient`.
Library code compiles shaders directly via `wgpu`.

**What is needed:**
- Route `compile_shader_universal` through coralReef IPC when available
- Fallback to local `wgpu` compilation when coralReef is not running
- Honest skip (exit 2) in validation binaries when coralReef unavailable

**Hand back to:** coralReef (ensure `shader.compile.wgsl` accepts the same
source format neuralSpring produces), primalSpring (wire in proto-nucleate)

---

## 4. toadStool Compute Dispatch via IPC

**Status:** open
**Proto-nucleate declares:** toadStool node with `compute.dispatch.submit`,
`compute.execute` capabilities
**Current state:** `playGround` has a `ToadStoolClient`. Library code
dispatches via `barracuda::dispatch` (local).

**What is needed:**
- IPC client for `compute.dispatch.submit` when running in composed mode
- Hardware discovery delegation to toadStool instead of local `wgpu`
  enumeration

**Hand back to:** toadStool (compute dispatch JSON-RPC surface), primalSpring

---

## 5. NestGate Weight Storage

**Status:** open
**Proto-nucleate declares:** NestGate (optional) with `storage.store`,
`storage.retrieve`, `storage.list`
**Current state:** Weight loading uses `safetensors` from local filesystem
(`src/weight_loader.rs`). No NestGate integration.

**What is needed:**
- IPC client for `storage.retrieve` to load model weights from NestGate
- Fallback to local filesystem when NestGate unavailable
- Weight provenance tracking via NestGate metadata

**Hand back to:** NestGate (weight tensor storage API)

---

## 6. BearDog/Songbird Tower Integration

**Status:** wip (discovery probing wired, BTSP session pending)
**Proto-nucleate declares:** BearDog (crypto) + Songbird (discovery) as
Tower Atomic foundation
**Current state (S181):** `primal_names.rs` has discovery hint constants.
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

## 7. Proto-nucleate Fragment Inconsistency

**Status:** resolved (S181 — local deploy graph + upstream reconciled Apr 17 2026)
**Details:** The proto-nucleate entry in `downstream_manifest.toml` declared
`fragments = ["tower_atomic", "node_atomic", "meta_tier"]` but the deploy graph
includes NestGate nodes. Local deploy graph was already fixed (R7/S180).
Upstream `downstream_manifest.toml` now includes `nest_atomic` and `nestgate`
in the neuralspring entry (Apr 17 2026 audit remediation).

**Hand back to:** N/A — resolved

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

**Status:** open (workaround applied)
**Details:** barraCuda's `special/plasma_dispersion.rs` unconditionally
imports from `ops::lattice::cpu_complex::Complex64`, but `ops::lattice` is
gated behind `#[cfg(feature = "domain-lattice")]`. neuralSpring works around
this by enabling `domain-lattice`, but the fix belongs upstream.

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

### R12. Downstream Manifest Fragment Reconciliation (resolved Apr 17 2026)

**Original gap:** `downstream_manifest.toml` listed
`fragments = ["tower_atomic", "node_atomic", "meta_tier"]` for neuralspring
but the deploy graph includes NestGate nodes (nest_atomic).
**Resolution:** Added `nest_atomic` to fragments and `nestgate` to
`depends_on` in the upstream manifest.

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
