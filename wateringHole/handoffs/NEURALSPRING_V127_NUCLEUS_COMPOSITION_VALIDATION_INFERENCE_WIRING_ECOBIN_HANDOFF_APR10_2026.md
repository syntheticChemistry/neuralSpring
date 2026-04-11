# neuralSpring V127 — NUCLEUS Composition Validation, Inference Wiring, ecoBin Harvest

**Date**: April 10, 2026
**Session**: S177
**Spring**: neuralSpring v0.1.0
**barraCuda**: v0.3.11
**License**: AGPL-3.0-or-later

---

## Summary

neuralSpring now validates at three layers:

1. **Python baselines** validate Rust science correctness (397/397)
2. **Rust validation** validates barraCuda GPU fidelity (~1,225 lib tests)
3. **NUCLEUS composition** validates primal composition patterns (3 validators)

This handoff documents the composition validation layer, inference capability
wiring, ecoBin harvest readiness, and gaps discovered for upstream evolution.

---

## What Changed (S176 → S177)

### New: Composition Validation Binaries

| Binary | Checks | Purpose |
|--------|--------|---------|
| `validate_nucleus_composition` | Bonding, capabilities, 7-node discovery | Proto-nucleate graph validation |
| `validate_inference_composition` | inference.* registration, Squirrel chain | Inference capability chain |
| `validate_primal_discovery` | by_capability routing, 5-tier discovery | Capability-based primal discovery |

All three follow the **honest skip** pattern (exit 2 when no primals available)
per primalSpring's `exit_code_skip_aware()`. CI handles exit 2 as success.

### New: Composition Validation Infrastructure

`src/validation/composition.rs` provides:
- `discover_primal_socket()` — 5-tier biomeOS discovery order
- `json_rpc_call()` — JSON-RPC 2.0 over Unix sockets (newline-delimited)
- `probe_liveness()` / `probe_capabilities()` / `call_capability()`
- `inference_proto_nucleate_nodes()` — 7-node graph from proto-nucleate TOML
- `BondType` enum (Metallic/Ionic/Covalent/Weak)
- `exit_code_skip_aware()` — 0=pass, 1=fail, 2=all-skipped

### New: inference.* Capability Wiring

| Layer | File | What |
|-------|------|------|
| Niche | `src/niche.rs` | 3 capabilities + deps + costs + bonding policy |
| Config | `src/config.rs` | `ALL_CAPABILITIES` updated |
| Registry | `config/capability_registry.toml` | 3 TOML entries |
| RPC | `src/rpc_service.rs` | Typed request/response structs |
| Handlers | `src/bin/neuralspring_primal/handlers.rs` | Stub handlers (not_yet_wired) |
| Dispatch | `src/bin/neuralspring_primal/main.rs` | Method routing |
| MCP | `playGround/src/mcp_tools.rs` | 3 tool definitions (19 total) |

### New: NUCLEUS Bonding Policy

```
Bond type:    Metallic (shared trust domain, no per-call auth)
Trust model:  InternalNucleus (all traffic inside NUCLEUS perimeter)
Encryption:   Tower=full BTSP, Node/Nest/Meta=tower_delegated
```

### New: ecoBin Harvest

- `Cargo.toml` release profile: strip, LTO, codegen-units=1
- `scripts/harvest_ecobin.sh`: musl build → static verify → plasmidBin staging
- CI `ecobin-cross-check` job already validates musl + ARM cross-compile

### Fixed: barraCuda Dependency Paths

All `Cargo.toml` files now use `../../primals/barraCuda/crates/barracuda`
(was `../barraCuda/...`). CI checkout paths aligned. `include_str!` macros
in `diagnose_f64_regression.rs` updated.

### Fixed: Precision Enum Exhaustiveness

`compile_shader_universal()` now handles all 12 `Precision` variants:
Binary, Int2, Q4, Q8, Fp8E5M2, Fp8E4M3, Bf16, F16, F32, Df64, Qf128, Df128.

---

## For barraCuda Team

### Upstream Absorption Candidates (41 WGSL shaders in metalForge)

These shaders have been validated in neuralSpring and are candidates for
absorption into `barracuda::shaders::*`. Priority by ecosystem demand:

| Shader | Domain | Consumers | Priority |
|--------|--------|-----------|----------|
| `softmax_f64.wgsl` | nn | all springs | High |
| `layer_norm_f64.wgsl` | nn | all springs | High |
| `gelu_f64.wgsl` | nn | all springs | High |
| `sigmoid_f64.wgsl` | nn | all springs | High |
| `attention_apply_f64.wgsl` | nn/transformer | neuralSpring, coralForge | High |
| `sdpa_scores_f64.wgsl` | nn/transformer | neuralSpring, coralForge | High |
| `batch_ipr.wgsl` | spectral | neuralSpring, hotSpring | Medium |
| `hmm_backward_log.wgsl` | bio | neuralSpring, wetSpring | Medium |
| `hmm_viterbi.wgsl` | bio | neuralSpring, wetSpring | Medium |
| `wright_fisher_step.wgsl` | bio | neuralSpring | Medium |
| `rk4_parallel.wgsl` | ode | hotSpring, neuralSpring | Medium |
| `rk45_adaptive.wgsl` | ode | hotSpring, neuralSpring | Medium |

### Feature-Gate Bug

`barracuda::special::plasma_dispersion` unconditionally imports from
`ops::lattice::cpu_complex`, but `ops::lattice` is gated behind `domain-lattice`.
Workaround: neuralSpring enables `domain-lattice`. Fix: gate the import in
`special/plasma_dispersion.rs` or make it unconditional.

---

## For toadStool Team

### Dispatch Protocol Readiness

neuralSpring's `playGround/src/toadstool_client.rs` exercises:
- `compute.dispatch.submit` / `compute.dispatch.result` / `compute.dispatch.capabilities`
- `science.substrate.discover`
- GPU info, memory, health

The composition validators now probe toadStool via `by_capability: compute.dispatch.submit`.

### What neuralSpring Needs from toadStool

1. Stable `compute.dispatch.submit` protocol (currently working in playGround tests)
2. `science.substrate.discover` response format stabilization
3. Health endpoint alignment (`toadstool.health` vs `health.liveness`)

---

## For Squirrel Team

### Inference Chain Architecture

```
Any spring → Squirrel (ai.query) → neuralSpring (inference.*)
                                  → Ollama fallback (until native WGSL ML)
```

neuralSpring exposes `inference.complete`, `inference.embed`, `inference.models`
via JSON-RPC. Currently stub handlers returning `status: "not_yet_wired"`.

### What neuralSpring Needs from Squirrel

1. Provider registration protocol (neuralSpring announces itself as inference provider)
2. Capability routing (`ai.query` → discover neuralSpring → forward to `inference.*`)
3. Fallback chain (Squirrel → neuralSpring → Ollama if no local WGSL model)

---

## For coralReef Team

### Shader Compile Integration

neuralSpring's `metalForge/forge` has optional `coralreef` feature for
`coralreef_bridge`. Currently uses local `wgpu` compilation path.

### What neuralSpring Needs from coralReef

1. `shader.compile.wgsl` IPC endpoint stability
2. Multi-shader batch compilation (`shader.compile.wgsl_multi`)
3. Compilation cache (avoid recompiling identical shaders)

---

## For biomeOS Team

### Deploy Graph State

`graphs/neuralspring_deploy.toml` covers science niche (Tower → optional
toadStool/NestGate → neuralSpring by `science.spectral_analysis`).

The **inference proto-nucleate** (`neuralspring_inference_proto_nucleate.toml`)
is broader: adds coralReef, barracuda, squirrel as first-class nodes.

### neuralAPI Integration Points

neuralSpring exposes 19 capabilities for biomeOS neuralAPI routing:
- 14 science methods + 2 health + 3 inference + capability.list + compute.offload + provenance.*
- All registered in `config/capability_registry.toml`
- All exposed via `capability.list` JSON-RPC endpoint
- Bonding: Metallic, InternalNucleus, tower_delegated encryption

### What neuralSpring Needs from biomeOS

1. Graph deploy with honest skip for unavailable optional nodes
2. Capability routing via `by_capability` (not by primal name)
3. neuralAPI domain registration for `science.learning`

---

## For Spring Teams

### Composition Validation Patterns (Reusable)

The `validation::composition` module and validator patterns are reusable
across springs. Key patterns:

1. **Proto-nucleate node descriptor**: Declare expected nodes, `by_capability`, required/optional
2. **5-tier discovery**: `$BIOMEOS_ORCHESTRATOR_SOCKET` → `$XDG_RUNTIME_DIR/biomeos/` → `/tmp/biomeos/` → legacy dirs
3. **Honest skip (exit 2)**: No primals → exit 2 (not 0, not 1). CI accepts 2 as non-failure
4. **Bonding validation**: Check bond type + trust model + encryption tiers match proto-nucleate
5. **Capability coverage**: Verify niche advertises all expected capabilities

### Evolution Path Summary

```
Python baseline → Rust validation → barraCuda CPU → barraCuda GPU
  → fused TensorSession → sovereign dispatch (coralReef)
    → primal composition (proto-nucleate: IPC to primals by capability)
      → NUCLEUS deployment (biomeOS deploys the graph)

Composition validation targets:
  Python + Rust → science correctness ✓
  Rust + NUCLEUS → primal composition patterns ✓ (this session)
  NUCLEUS + biomeOS → deployment validation (next: live primals)
```

---

## Metrics

| Metric | S176 | S177 |
|--------|------|------|
| Lib tests | 1,211 | 1,225 |
| Forge tests | 73 | 73 |
| playGround tests | 80 | 80 |
| Integration tests | 12 | 14 |
| Binaries | 261 | 264 |
| `.rs` files | 466 | 518 |
| Capabilities | 16 | 19 |
| MCP tools | 16 | 19 |
| Clippy warnings | 0 | 0 |
| `#[allow()]` | 0 | 0 |
| barraCuda | v0.3.7 | v0.3.11 |
