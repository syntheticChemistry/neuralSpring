# neuralSpring V131 — Comprehensive Primal Evolution Handoff

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Date:** 2026-04-11
**Session:** S181
**Spring:** neuralSpring 0.1.0
**barraCuda:** v0.3.11
**Handoff scope:** All primal teams and spring teams — composition patterns,
absorption targets, NUCLEUS deployment learnings, neuralAPI readiness

---

## Context: The Four-Layer Validation Stack

neuralSpring has evolved through four validation layers that demonstrate the
ecoPrimals composition evolution pattern. This handoff documents what we
learned and what each primal needs to absorb or evolve.

```
Layer 1 — Python validates Rust (science correctness)
  397/397 Python PASS → 4500+ Rust+GPU PASS
  Pattern: seed-matched, tolerance-gated, provenance-tracked

Layer 2 — Rust validates GPU/WGSL (compute sovereignty)
  barraCuda CPU → GPU Tensor → WGSL → TensorSession → cross-substrate
  Pattern: f64-canonical, dispatch-transparent, multi-GPU bit-identical

Layer 3 — Rust+Python validate primal IPC (composition fidelity)
  30 capabilities → biomeOS → health triad → identity → MCP tools
  4 composition validators → honest skip (exit 2) → deploy graph alignment
  Pattern: capability-based discovery, no identity coupling

Layer 4 — plasmidBin + benchScale validate ecoBin (deployment)
  ecoBin harvest → metadata → fetch + smoke → C1-C7 probes
  Pattern: static binary, UDS IPC, platform-agnostic
```

---

## Per-Primal Handoff

### To barraCuda

**What neuralSpring uses (130+ imports, 46 rewires):**
- `barracuda::dispatch::*` — softmax, matmul, mean, variance, L2, frobenius,
  transpose, eigenvectors, rk4 (9 domain ops)
- `barracuda::ops::bio::*` — hmm_viterbi, fst_variance_decomposition
- `barracuda::ops::lattice::*` (via domain-lattice feature)
- `barracuda::nn::SimpleMlp` — WDM surrogate replacement
- `barracuda::stats::*` — mae, shannon, hill, fit_linear, pearson
- `barracuda::linalg::*` — graph_laplacian, disordered_laplacian, solve_f64_cpu
- `barracuda::numerical::*` — hessian
- `Tensor` API — matmul, softmax_dim, argmax_dim, GPU kernels

**Feature-gate bug (Gap 9):**
`special/plasma_dispersion.rs` imports `ops::lattice::cpu_complex::Complex64`
unconditionally, but `ops::lattice` is behind `#[cfg(feature = "domain-lattice")]`.
neuralSpring works around this by enabling the feature. Fix belongs upstream.

**Shader absorption candidates (17 shaders):**
These WGSL shaders in `metalForge/shaders/` follow the Write→Absorb→Lean cycle
and are candidates for barraCuda upstream absorption:

| Shader | Target module |
|--------|--------------|
| `hmm_backward_log.wgsl` | `ops::bio` |
| `hmm_viterbi.wgsl` | `ops::bio` |
| `wright_fisher_step.wgsl` | `ops::bio` |
| `pairwise_hamming.wgsl` | `ops::bio` |
| `pairwise_jaccard.wgsl` | `ops::bio` |
| `batch_ipr.wgsl` | `spectral` |
| `chi_squared_f64.wgsl` | `stats` |
| `kl_divergence_f64.wgsl` | `stats` |
| `linear_regression.wgsl` | `stats` |
| `matrix_correlation.wgsl` | `stats` |
| `rk4_parallel.wgsl` | `ops::ode` |
| `rk45_adaptive.wgsl` | `ops::ode` |
| `sdpa_scores_f64.wgsl` | `ops::mha` |
| `layer_norm_f64.wgsl` | `ops::nn` |
| `outer_product_mean_f64.wgsl` | `ops::linalg` |
| `softmax_f64.wgsl` | `ops::nn` (likely already upstream) |
| `gelu_f64.wgsl` | `ops::nn` (likely already upstream) |

**Composition evolution pattern for barraCuda:**
Currently compile-time `path` dependency. Future: capability-based IPC
client for `math.*` operations behind `#[cfg(feature = "composed")]`.
Direct import stays for validation baselines.

---

### To Squirrel

**What neuralSpring wires:**
`try_squirrel_route()` in `handlers.rs` discovers Squirrel at runtime via
socket scanning for `inference.route` capability. When found, forwards
`inference.complete`, `inference.embed`, `inference.models` via JSON-RPC
over UDS. Falls back to stub response when Squirrel unavailable.

**What Squirrel needs to provide:**
1. `inference.register_provider` method — neuralSpring needs to register
   itself as an inference backend so Squirrel can route requests to it
2. Capability advertisement: Squirrel should advertise `inference.route`
   (or equivalent) so neuralSpring can discover it
3. MCP tool registration: neuralSpring advertises 30 tools via `mcp.tools.list`
   that Squirrel can consume for AI-assisted composition

**Composition pattern:** Consumer-first — neuralSpring discovers Squirrel
dynamically per-request, no startup dependency. This is the recommended
pattern for optional runtime dependencies.

---

### To BearDog

**What neuralSpring wires:**
`tower.rs` probes BearDog at startup via capability-based socket discovery
(looks for `capability.security.attest`), then sends `health.liveness`
JSON-RPC probe. Logs status. Non-blocking — neuralSpring runs standalone
when BearDog is absent.

**What BearDog needs to provide:**
1. `crypto.btsp_handshake` JSON-RPC surface — neuralSpring is ready to
   establish BTSP sessions but the wire doesn't exist yet
2. Signed capability announcements — for composed-mode trust validation
3. UDS socket at standard path with `capability.security.attest` advertised

**Composition pattern:** Startup probe, degrade gracefully. This is the
recommended Tower Atomic integration pattern.

---

### To Songbird

**What neuralSpring wires:**
`tower.rs` probes Songbird at startup via capability-based socket discovery
(looks for `discovery.mesh.join`), then sends `health.liveness` probe.
Reports Tower completeness (BearDog + Songbird = complete Tower).

**What Songbird needs to provide:**
1. Mesh discovery — neuralSpring currently discovers primals via filesystem
   socket scanning. Songbird mesh would replace this for production.
2. `discovery.mesh.join` method advertised on UDS socket
3. Primal registration forwarding to biomeOS

**Composition pattern:** Same as BearDog — startup probe, degrade gracefully.

---

### To coralReef

**What neuralSpring has:**
- `metalForge/forge` crate with optional `coralreef` feature
- `playGround/src/coralreef_client.rs` — typed IPC client
- 41 WGSL shaders that compile through `wgpu` directly

**What neuralSpring needs from coralReef:**
1. `shader.compile.wgsl` method that accepts the same WGSL source format
   neuralSpring produces (f64-canonical, df64 preamble aware)
2. Feature-gate recommendation for `coralreef` IPC vs direct `wgpu`
3. Pipeline compilation for multi-step ML graphs (inference chains)

**Composition pattern:** Optional IPC — local `wgpu` compilation as fallback.

---

### To toadStool

**What neuralSpring has:**
- `playGround/src/toadstool_client.rs` — typed IPC client (discovery
  fixed in S181: `compute.submit` → `compute.dispatch.submit`)
- All GPU dispatch currently through local `barracuda::dispatch`

**What neuralSpring needs from toadStool:**
1. `compute.dispatch.submit` accepting GPU workload descriptors
2. Hardware discovery delegation (replace local `wgpu` enumeration)
3. Cross-node GPU routing for multi-GPU composition

**Composition pattern:** Optional IPC — local dispatch as fallback.

---

### To NestGate

**What neuralSpring needs:**
1. `storage.retrieve` for model weight tensors (currently loaded from local
   filesystem via `safetensors`)
2. `storage.store` for trained model checkpoints
3. `storage.list` for weight provenance tracking
4. Weight provenance metadata integration

**Composition pattern:** Not yet wired. Needs IPC client in playGround.

---

### To biomeOS

**What neuralSpring wires:**
- Registration at startup via `biomeos::register_with_biomeos()`
- Heartbeat maintenance
- 30 capabilities advertised
- Deploy graph V131 with Tower, Node, Nest, Meta-tier fragments
- `identity.get` returns full T4 primal identity

**No gaps with biomeOS** — integration is complete and validated.

---

### To primalSpring

**Upstream proto-nucleate (`neuralspring_inference_proto_nucleate.toml`):**
1. Fragment list should add `nest_atomic` — NestGate is in the node list
   but fragments only declare `tower_atomic`, `node_atomic`, `meta_tier`
2. Binary name and health method reconciled (S180)
3. Capability set aligned to actual 14 science + 3 health + 3 inference +
   4 provenance + 6 routing = 30 capabilities

**Composition validation patterns discovered:**
1. **Honest skip (exit 2)** — validators that probe live primals return
   exit 0 (pass), 1 (fail), or 2 (primals not running). `validate_all`
   treats exit 2 as skip, not failure.
2. **5-tier socket discovery** — env var → XDG_RUNTIME → /tmp →
   /run/user/$UID → fallback. Used by all IPC clients.
3. **Capability-based discovery** — never discover by primal name, always
   by capability string. Enables composition without identity coupling.
4. **Degrade gracefully** — Tower probes at startup but neuralSpring runs
   independently. Squirrel routing tries per-request but falls back to stub.
5. **Feature gating** — `primal` feature for IPC dependencies, `composed`
   feature for composition-mode paths.

---

## Validation Tier Summary

| Tier | What it validates | Binary | Status |
|------|-------------------|--------|--------|
| 1 | Python → Rust (science correctness) | `validate_all` (242 science bins) | **green** |
| 2 | Rust → NUCLEUS (primal wiring) | `validate_nucleus_composition`, `validate_inference_composition`, `validate_primal_discovery` | **green** (skip-aware) |
| 3 | Composition evolution (full coherence) | `validate_composition_evolution` | **green** (S181) |
| 4 | Deployment (ecoBin + plasmidBin) | `harvest_ecobin.sh` + benchScale | **ready** |

---

## Wiring Status Matrix

| Primal | Discovery | Liveness | Capability probe | Live routing | Status |
|--------|-----------|----------|-----------------|-------------|--------|
| biomeOS | socket | heartbeat | — | register/heartbeat | **wired** |
| BearDog | startup probe | probe | — | — | **discovery only** |
| Songbird | startup probe | probe | — | — | **discovery only** |
| Squirrel | per-request | — | — | inference.* forward | **routed** |
| toadStool | playGround client | — | — | — | **client ready** |
| coralReef | playGround client | — | — | — | **client ready** |
| NestGate | — | — | — | — | **open** |
| barraCuda | direct Rust import | — | — | — | **deferred** |

---

## Learnings for NUCLEUS Deployment via neuralAPI from biomeOS

1. **Composition validators are essential** — they catch capability surface
   drift, deploy graph misalignment, and IPC wiring gaps that unit tests miss
2. **30 capabilities is the right surface** — health triad + identity + MCP
   tools are not optional; they're required for biomeOS graph orchestration
3. **Squirrel routing should be per-request** — startup binding creates
   unnecessary coupling; discover-and-forward is more resilient
4. **Tower is optional at the spring level** — BearDog/Songbird are
   infrastructure; springs degrade gracefully without them
5. **Feature gates separate concerns** — `primal` for any IPC, `composed`
   for full composition mode, default for standalone validation
6. **Deploy graph is the source of truth for biomeOS** — keep it aligned
   with `niche.rs` capabilities and proto-nucleate graph
7. **MCP tool definitions bridge AI and composition** — Squirrel consumes
   `mcp.tools.list` to expose spring capabilities to AI workflows

---

## Files Modified in S181 (for review)

| File | Change |
|------|--------|
| `src/config.rs` | 3 capabilities added to `ALL_CAPABILITIES` |
| `src/niche.rs` | 3 capabilities + deps + costs |
| `src/validation/mod.rs` | `check_abs_or_rel` tolerance mode fix |
| `config/capability_registry.toml` | 3 new capability entries |
| `playGround/src/mcp_tools.rs` | 3 MCP tool defs, test counts |
| `playGround/src/toadstool_client.rs` | discovery capability fix |
| `src/bin/neuralspring_primal/handlers.rs` | Squirrel routing |
| `src/bin/neuralspring_primal/tower.rs` | Tower discovery (new) |
| `src/bin/neuralspring_primal/main.rs` | Tower probe call |
| `src/bin/validate_composition_evolution.rs` | Tier 3 validator (new) |
| `Cargo.toml` | new binary + `composed` feature |
| `graphs/neuralspring_deploy.toml` | V131/S181 |
| `specs/BARRACUDA_REQUIREMENTS.md` | version refs |
| `docs/PRIMAL_GAPS.md` | S181 status |
| `CHANGELOG.md` | S181 entry |
