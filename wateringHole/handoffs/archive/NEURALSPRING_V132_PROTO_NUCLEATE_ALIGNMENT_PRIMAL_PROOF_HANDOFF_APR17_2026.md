# neuralSpring V132 — Proto-Nucleate Alignment & Primal Proof Handoff

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Date**: April 17, 2026
**Spring**: neuralSpring 0.1.0 (S181+)
**barraCuda**: v0.3.12
**primalSpring**: v0.9.15 (upstream source of truth)
**Supersedes**: V131 handoffs (audit remediation, cross-team absorption)
**Direction**: neuralSpring → primalSpring, barraCuda, biomeOS, all springs

---

## 1. What Changed

neuralSpring's composition validation code has been aligned to the upstream
primalSpring-validated proto-nucleate composition. All previous local-only
modifications that claimed upstream reconciliation have been corrected.

### 1.1 Proto-Nucleate Nodes Aligned to Upstream

**Source of truth**: `primalSpring/graphs/downstream/downstream_manifest.toml`
`[[downstream]] spring_name = "neuralspring"`

```toml
domain = "ml_inference"
particle_profile = "proton_heavy"
fragments = ["tower_atomic", "node_atomic", "meta_tier"]
depends_on = ["beardog", "songbird", "coralreef", "toadstool", "barracuda", "squirrel"]
validation_capabilities = [
    "tensor.matmul", "tensor.create", "compute.dispatch",
    "inference.complete", "inference.embed", "stats.mean", "crypto.hash",
]
```

**Changes to `inference_proto_nucleate_nodes()`**:
- **Added**: `barracuda` node (`by_capability = "tensor.matmul"`)
- **Removed**: `nestgate` — NOT in proto-nucleate `depends_on` (spring-deploy only)
- **Added**: `primal_names::BARRACUDA` constant (`"barracuda"`)
- **Added**: `PROTO_NUCLEATE_VALIDATION_CAPABILITIES` constant (7 IPC methods)

### 1.2 Proto-Nucleate vs Spring-Deploy Clarification

| Graph | Purpose | Fragments | NestGate? | Spring Binary? |
|-------|---------|-----------|-----------|----------------|
| Proto-nucleate (`downstream_manifest.toml`) | Pure primal NUCLEUS — spring validates AGAINST this | `tower_atomic`, `node_atomic`, `meta_tier` | No | No |
| Spring deploy (`spring_deploy_manifest.toml`) | Spring + NUCLEUS for integration | `tower_atomic`, `node_atomic`, `nest_atomic`, `nucleus`, `meta_tier` | Yes | Yes |
| Local deploy (`neuralspring_deploy.toml`) | Full niche deployment | `tower_atomic`, `node_atomic`, `nest_atomic`, `meta_tier` | Yes | Yes |

### 1.3 False Claim Corrected (R12)

Previous sessions claimed `nest_atomic` and `nestgate` were pushed to the
upstream `downstream_manifest.toml`. This was aspirational — the change was
local only and never pushed to primalSpring. The upstream manifest correctly
reflects the proto-nucleate composition primalSpring validated.

---

## 2. For primalSpring

### 2.1 Manifest Alignment Confirmed

neuralSpring's composition code now matches the upstream `downstream_manifest.toml`
entry exactly. No changes needed to the upstream manifest.

### 2.2 If NestGate Belongs in Proto-Nucleate

If neuralSpring's primal proof eventually needs `storage.retrieve` for weight
loading via IPC (currently done via local filesystem), file a manifest change:
- Add `nest_atomic` to fragments
- Add `nestgate` to `depends_on`
- Add `storage.store`, `storage.retrieve` to `validation_capabilities`

This should only happen when NestGate's JSON-RPC API for tensor weight
storage is ready and neuralSpring can demonstrate IPC parity.

### 2.3 Spring Validation Manifest — Aligned

`spring_validate_manifest.toml` correctly uses:
- `domain = "science"`
- `capabilities = ["science.spectral_analysis", "science.anderson_localization", "science.hessian_eigen"]`

### 2.4 Niche-to-Package Name Mapping

`primal_to_pkg_name()` maps niche names to `CARGO_PKG_NAME` forms. primalSpring
may want to standardize this in the ecosystem — all springs with compound names
have this ambiguity (e.g. `neuralspring` socket vs `neural-spring-*.sock`).

---

## 3. For barraCuda

### 3.1 Proto-Nucleate Node Registration

barraCuda is now registered as a proto-nucleate node in neuralSpring's
composition validators with `by_capability = "tensor.matmul"`. The upstream
`validation_capabilities` that map to barraCuda:

| Capability | barraCuda Module | Primal Proof Status |
|------------|-----------------|-------------------|
| `tensor.matmul` | `barracuda::tensor::matmul` | Direct import (Tier 1) |
| `tensor.create` | `barracuda::tensor::Tensor::new` | Direct import (Tier 1) |
| `stats.mean` | `barracuda::stats::mean` | Direct import (Tier 1) |
| `compute.dispatch` | `barracuda::dispatch` | Direct import (Tier 1) |

**Next step**: When barraCuda exposes `tensor.matmul`, `tensor.create`,
`stats.mean` via JSON-RPC IPC, neuralSpring can migrate from direct Rust
imports to capability-based IPC calls (Tier 3 primal proof).

### 3.2 Existing Hand-Backs (from V131)

- `special::plasma_dispersion` feature-gate bug (unconditional `domain-lattice` import)
- 2 GPU shaders + 6 generic f64 activations as absorption candidates
- `GpuPreferred` dispatch pattern for upstream adoption

---

## 4. For biomeOS

### 4.1 Deployment Patterns

neuralSpring's deploy graph (`neuralspring_deploy.toml`) is a spring-deploy
graph — it includes the spring binary as a node alongside NUCLEUS primals.

For the primal proof (Level 5), biomeOS deploys the proto-nucleate graph
(pure primals, no spring binary). neuralSpring then validates externally:

```
biomeos deploy --graph proto_nucleate_template.toml --params downstream_manifest::neuralspring
# → NUCLEUS composition starts (pure primals only)
# neuralSpring runs validation harness externally
# → validate_science_composition calls 7 capabilities via IPC
# → compares against deterministic Rust baselines
# → exit 0 (pass) / exit 1 (fail) / exit 2 (primals unavailable)
```

### 4.2 Orchestrator Socket Discovery

`$BIOMEOS_ORCHESTRATOR_SOCKET` is the highest-priority discovery tier in
`discover_primal_socket()`. biomeOS should set this when deploying the
proto-nucleate so the validation harness can find the orchestrator.

---

## 5. For All Springs

### 5.1 Three-Tier Validation Pattern

Every spring should evolve through three validation tiers:

| Tier | What | Validates Against | Exit Codes |
|------|------|------------------|------------|
| 1 | Python → Rust | Python baselines | 0/1 |
| 2 | Rust CPU → GPU | CPU references | 0/1 |
| 3 | Rust → IPC | Proto-nucleate primals | 0/1/2 (honest skip) |

### 5.2 Proto-Nucleate vs Spring-Deploy

The proto-nucleate is a PURE PRIMAL composition — no spring binary.
The spring validates AGAINST it. The spring-deploy is RICHER — it includes
the spring binary + NUCLEUS primals for integration testing.

Do not conflate the two. `inference_proto_nucleate_nodes()` should match
`downstream_manifest.toml` `depends_on` exactly.

### 5.3 Validation Capabilities as Structured Data

neuralSpring now has `PROTO_NUCLEATE_VALIDATION_CAPABILITIES` as a constant
array matching the upstream `validation_capabilities` field. Every spring
should do the same — codify the upstream manifest entry as structured data
in the spring's composition module.

### 5.4 Composition Cycle

```
Read manifest entry → Wire IPC → Validate against baselines
  → Discover gaps → Hand back to primalSpring → Primals evolve → Repeat
```

### 5.5 Known Upstream Blockers

| Blocker | Primal | Impact |
|---------|--------|--------|
| Squirrel `inference.register_provider` | Squirrel | neuralSpring can't register as inference backend |
| barraCuda JSON-RPC IPC surface | barraCuda | Can't do Tier 3 `tensor.matmul` via IPC |
| coralReef pipeline compilation | coralReef | Can't compile multi-step ML graphs via IPC |
| NestGate tensor storage API | NestGate | Can't load model weights via IPC |

---

## 6. Composition Patterns for NUCLEUS Deployment via neuralAPI

neuralSpring provides AI inference for the entire ecosystem. Any spring that
adds Squirrel to its composition immediately gains `inference.*` capabilities.

### 6.1 Capability Surface (30 registered)

**Science**: `science.spectral_analysis`, `science.anderson_localization`,
`science.hessian_eigen`, `science.ipr`, `science.disorder_sweep`,
`science.training_trajectory`, `science.evoformer_block`, `science.structure_module`,
`science.folding_health`, `science.gpu_dispatch`, `science.cross_spring_provenance`,
`science.cross_spring_benchmark`, `science.precision_routing`, `science.agent_coordination`

**Inference**: `inference.complete`, `inference.embed`, `inference.models`

**Health**: `health.liveness`, `health.readiness`, `health.check`

**Provenance**: `provenance.begin`, `provenance.record`, `provenance.complete`, `provenance.status`

**Deployment**: `capability.list`, `identity.get`, `mcp.tools.list`,
`compute.offload`, `primal.forward`, `primal.discover`

### 6.2 Files Modified in This Handoff

| File | Change |
|------|--------|
| `src/validation/composition.rs` | Aligned `inference_proto_nucleate_nodes()` to upstream; added `PROTO_NUCLEATE_VALIDATION_CAPABILITIES` |
| `src/primal_names.rs` | Added `BARRACUDA` discovery hint constant |
| `src/bin/validate_primal_discovery.rs` | Added barracuda to discovery probe targets |
| `graphs/neuralspring_deploy.toml` | Header updated: proto-nucleate vs spring-deploy distinction |
| `docs/PRIMAL_GAPS.md` | R12 corrected; Gap 7 reclassified; Gap 2/5 scoped |

---

## 7. Summary

| Dimension | Status |
|-----------|--------|
| Proto-nucleate nodes | Aligned to upstream `depends_on` exactly |
| Validation capabilities | 7 IPC methods codified as `PROTO_NUCLEATE_VALIDATION_CAPABILITIES` |
| Tests | 1,230 lib + 73 forge + 80 playGround — 0 failures |
| Clippy | Zero warnings (pedantic + nursery) |
| Deploy graph | Clarified proto-nucleate vs spring-deploy distinction |
| PRIMAL_GAPS.md | R12 corrected, all gaps accurately tracked |
| Handoff | V132 — primalSpring audits upstream |
