<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring V159 — Primal Evolution & Composition Patterns Upstream Handoff

**From**: neuralSpring (Session S205c — Doc reconciliation + primal evolution review)
**To**: primalSpring + all primal teams + spring teams
**Date**: May 13, 2026

---

## Summary

neuralSpring has completed its niche convergence arc (V155–V158): Tier 2 fully
wired, NestGate weight persistence, Squirrel inference pipeline, and four
consecutive deep-debt audits at zero across all 7 categories. This handoff
documents primal usage patterns, composition learnings, and evolution
opportunities for upstream teams.

---

## 1. Primal Usage Map (7 IPC modules)

neuralSpring wires 7 primals via `src/ipc/` with typed facades and
`CapabilityRouter` discovery (20 hints, 37 capabilities):

| Primal | Module | Capabilities Used | Role |
|--------|--------|-------------------|------|
| **barraCuda** | `ipc/barracuda.rs` | `stats.mean`, `stats.std_dev`, `stats.weighted_mean`, `tensor.matmul`, `tensor.create`, `barracuda.precision.route` | Math engine — stats, tensors, precision routing |
| **toadStool** | `ipc/toadstool.rs` | `compute.dispatch`, `toadstool.validate`, `toadstool.list_workloads` | Compute orchestration — dispatch, workload pre-flight |
| **BearDog** | `ipc/beardog.rs` | `crypto.hash` | Crypto — BLAKE3 hashing for provenance |
| **Squirrel** | `ipc/squirrel.rs` | `inference.complete`, `inference.embed`, `inference.models` | AI inference — model completion, embedding, model listing |
| **coralReef** | `ipc/coralreef.rs` | `shader.compile.wgsl`, `shader.compile.capabilities` | Shader compilation — WGSL to native GPU binary |
| **skunkBat** | `ipc/skunkbat.rs` | `security.audit_log` | Security audit logging |
| **NestGate** | `ipc/nestgate.rs` | `content.put`, `content.get`, `content.exists` | Content-addressed storage — model weight persistence |

### IPC Facade Pattern

All calls go through `IpcMathClient`, which wraps `CapabilityRouter` for
socket discovery. The router uses `CAPABILITY_HINTS` (20 entries mapping
capability strings to primal names) plus `discover_primal_socket()` for
UDS resolution. This pattern is reusable by other springs.

---

## 2. Composition Patterns Learned

### Pattern A: Capability-Based Discovery (recommended for all springs)

```
CapabilityRouter::from_hints(&CAPABILITY_HINTS)
  → discover_primal_socket(primal_name, family_id)
  → JSON-RPC 2.0 over UDS
```

This replaces hardcoded primal paths with dynamic, family-isolated discovery.
neuralSpring has zero hardcoded socket paths in production code.

### Pattern B: NestGate Content-Addressed Weight Persistence

```
local file → base64 encode → content.put → BLAKE3 hash returned
BLAKE3 hash → content.get → base64 decode → safetensors deserialize
```

Any spring storing model artifacts, checkpoints, or large binary blobs should
use this pattern instead of local filesystem paths. The BLAKE3 hash provides
content addressing and integrity verification.

### Pattern C: Squirrel Inference Pipeline

```
has_squirrel() → inference.models → inference.complete / inference.embed
```

Discovery-first: check if Squirrel is available, query available models,
then invoke. `try_squirrel_route` provides graceful fallback when Squirrel
is unavailable.

### Pattern D: Agent-Driven Composition (neural_composition.sh)

```
nucleus_composition_lib.sh (41 functions)
  → cap_socket "ai" / "inference"
  → inference.complete → DAG branching → braid provenance
  → closed-loop feedback (act → observe → adjust)
```

This demonstrates how springs can use AI inference in composition scripts
for autonomous decision-making within NUCLEUS workflows.

---

## 3. Evolution Opportunities for Primal Teams

### For barraCuda

- **GPU parity benchmarks**: neuralSpring has 15 CPU domain baselines at 38.6x
  geomean. Matched-hardware GPU parity validation (cuBLAS, cuFFT, cuDNN, Flash
  Attention, Kokkos, SciPy, LAMMPS) is `hotSpring`'s niche but would benefit
  from cross-spring validation.
- **Tensor API coverage**: 90 ops validated. New domains (WGSL tokenization via
  coralReef, larger transformer models) will exercise additional API surface.

### For Squirrel

- **Provider registration**: neuralSpring exposes `inference.*` capabilities but
  cannot register as a Squirrel provider until Squirrel supports provider
  registration (upstream feature request).
- **Model artifact flow**: NestGate weight persistence is wired; Squirrel could
  use the same `content.put`/`content.get` pattern for model distribution.

### For NestGate

- **Weight storage exercised**: `store_to_nestgate` and `load_safetensors_from_nestgate`
  are wired and tested (error paths validated). Full round-trip requires NestGate
  running — integration testing with live NestGate would validate the complete flow.

### For toadStool

- **Workload pre-flight**: `toadstool.validate` and `toadstool.list_workloads`
  are wired. Future: richer workload metadata (estimated FLOPS, memory
  requirements) would help springs make better dispatch decisions.

### For coralReef

- **WGSL tokenization pipeline**: neuralSpring has validated WGSL compilation
  via `shader.compile.wgsl`. The tokenization pipeline (parsing WGSL into
  AST for optimization) is blocked on coralReef live deployment.

### For BearDog / Songbird

- **Startup discovery**: Tower Atomic discovery (BearDog + Songbird liveness
  probes) is wired and validated. No evolution gaps found.

---

## 4. Composition Patterns for NUCLEUS Deployment

### neuralAPI from biomeOS

neuralSpring participates in biomeOS deployment graphs as a **Node** + **Meta**
component. The deploy graph (`graphs/neuralspring_deploy.toml`) specifies:

- Tower Atomic startup (BearDog, Songbird)
- Nest node (NestGate for weight storage)
- neuralspring binary (validate, serve, certify subcommands)
- Health triad (`health.check`, `identity.get`, `mcp.tools.list`)

### Atomic Instantiation Pattern

```
plasmidBin ecoBin fetch → neuralspring binary
  → certify (guideStone L0–L5)
  → serve (JSON-RPC surface with 37 capabilities)
  → health.check heartbeat
```

The `harvest_ecobin.sh` script stages the binary for plasmidBin distribution.
genomeBin v5.1 provides 46 binaries across 6 target triples.

---

## 5. What neuralSpring Learned (Relevant to All Springs)

1. **IPC tree pattern scales**: 7 modules, each <200 LOC, with a shared facade.
   The CapabilityRouter + CAPABILITY_HINTS pattern eliminates hardcoded paths.

2. **Deep debt auditing is repeatable**: 4 consecutive zero-debt audits (S199,
   S202c, S204b, S205b) prove the patterns are sustainable.

3. **Base64 encoding for NestGate**: Content-addressed storage requires
   serialization. Base64 + safetensors works for ML weights; other springs
   should choose appropriate serialization for their domain artifacts.

4. **guideStone L5 is achievable**: 19 certification tests across 6 layers
   (bare/discovery/parity/nucleus/composition/cross-spring). The
   `neuralspring_guidestone` binary pattern is reusable.

5. **Edition 2024 + MSRV 1.87**: No regressions. All workspace crates compile
   cleanly. Proptest 24 invariants provide ongoing correctness guarantees.

---

## 6. Current State

| Metric | Value |
|--------|-------|
| Session | S205b |
| Handoff | V159 (this document) |
| Workspace tests | 910 (IPC-first) |
| Capabilities | 37 |
| IPC modules | 7 |
| Deep debt | Zero (4 audits) |
| Python baselines | 397/397 |
| Papers reproduced | 27/27 |
| guideStone | Level 5 (19 certification tests) |
| Clippy | 0 warnings (pedantic+nursery+cast deny) |

---

## 7. Hold Items

- **Full NUCLEUS compositions**: Holding per primalSpring directive.
- **Squirrel provider registration**: Blocked on upstream Squirrel feature.
- **WGSL tokenization pipeline**: Blocked on coralReef live deployment.
- **Matched-hardware GPU benchmarks**: hotSpring's niche — neuralSpring provides baselines.

---

*neuralSpring V159 | Session S205c | AGPL-3.0-or-later*
