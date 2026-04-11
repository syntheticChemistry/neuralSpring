# neuralSpring — Primal Gaps

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

> Living gap log for neuralSpring's proto-nucleate composition.
> Reviewed against `neuralspring_inference_proto_nucleate.toml` v1.1.0.
>
> **Date:** 2026-04-10 | **Spring version:** 0.1.0 | **primalSpring:** v0.9.9

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

**Status:** open
**Proto-nucleate declares:** `inference.complete`, `inference.embed`, `inference.models`
**Current state:** These method strings do not appear in neuralSpring source.
Squirrel integration exists only in `playGround/` (`SquirrelClient`) as an
MCP adapter shell — `ai.query`, `capability.announce`, `tool.execute`.

**What is needed:**
- Wire types for `inference.*` methods in `src/rpc_service.rs` and JSON-RPC
  handler in `neuralspring_primal/handlers.rs`
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

**Status:** open
**Proto-nucleate declares:** BearDog (crypto) + Songbird (discovery) as
Tower Atomic foundation
**Current state:** `primal_names.rs` has discovery hint constants. No
runtime integration with BearDog signing or Songbird mesh.

**What is needed:**
- BTSP session establishment for composed-mode IPC
- Songbird-based primal discovery instead of filesystem socket scanning
- Signed capability announcements

**Hand back to:** BearDog, Songbird, primalSpring (Tower Atomic validation)

---

## 7. Proto-nucleate Fragment Inconsistency

**Status:** open
**Details:** The proto-nucleate graph declares
`fragments = ["tower_atomic", "node_atomic", "meta_tier"]` but includes
NestGate as a node. If NestGate is part of the composition, `nest_atomic`
should be in the fragment list. NestGate is marked `required = false`
(optional) which may explain the omission, but the fragment list should
document this explicitly.

**Hand back to:** primalSpring (graph authoring)

---

## 8. Binary Name Reconciliation

**Status:** open
**Details:** The proto-nucleate graph uses binary name `neuralspring` for
the spring node. The pipeline and deploy graphs use `neuralspring_primal`.
The actual `Cargo.toml` `[[bin]]` entry with `required-features = ["primal"]`
is named `neuralspring`. These should be reconciled — the proto-nucleate is
correct; pipeline/deploy graphs should be updated.

**Hand back to:** primalSpring (graph consistency)

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

_None yet — this is the initial gap inventory._
