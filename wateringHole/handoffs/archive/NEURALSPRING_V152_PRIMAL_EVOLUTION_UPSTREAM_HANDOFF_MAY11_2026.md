<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring V152 — Primal Evolution & NUCLEUS Composition Upstream Handoff

**Session**: S201b | **Date**: May 11, 2026 | **Handoff**: V152 (companion)

This document is the upstream handoff for primal teams and spring teams to
absorb neuralSpring's evolution patterns. It covers IPC architecture, capability
discovery, NUCLEUS composition, and deployment via neuralAPI from biomeOS.

---

## 1. Primal Surface Summary (What neuralSpring Consumes)

neuralSpring talks to 6 primals via IPC (JSON-RPC over Unix domain sockets).
Each has a dedicated submodule in `src/ipc/`:

| Primal | Module | Capabilities Used | Purpose |
|--------|--------|-------------------|---------|
| barraCuda | `ipc/barracuda.rs` | `stats.mean`, `stats.std_dev`, `stats.weighted_mean`, `tensor.matmul`, `tensor.create` | GPU math, tensor ops, statistics |
| toadStool | `ipc/toadstool.rs` | `compute.dispatch`, `compute.offload` | Compute orchestration |
| BearDog | `ipc/beardog.rs` | `crypto.hash` | Signing, integrity |
| Squirrel | `ipc/squirrel.rs` | `inference.complete`, `inference.embed`, `inference.models` | AI narration |
| coralReef | `ipc/coralreef.rs` | `shader.compile.wgsl`, `shader.compile.capabilities` | Shader compilation |
| skunkBat | `ipc/skunkbat.rs` | `security.audit_log` | Audit logging |

### Key Architecture Pattern

neuralSpring never imports primal code directly. All primal interaction is via:

1. **Capability-based discovery** — `biomeos::find_socket()` locates primals
2. **JSON-RPC** — method calls use constants from `src/capabilities.rs` (31 constants)
3. **Typed errors** — `IpcError::NotDiscovered | Transport | Protocol`
4. **Graceful degradation** — missing primals produce `IpcError::NotDiscovered`, not panics

### For Primal Teams: Required Capabilities

If a primal changes or removes any of these JSON-RPC method names, neuralSpring
validation breaks. The canonical method names are in `src/capabilities.rs`:

```
stats.mean, stats.std_dev, stats.weighted_mean
tensor.matmul, tensor.create
compute.dispatch, compute.offload
crypto.hash
inference.complete, inference.embed, inference.models
shader.compile.wgsl, shader.compile.capabilities
security.audit_log
```

**Contract**: neuralSpring expects `{"jsonrpc":"2.0","method":"<cap>","params":{...},"id":1}` — standard JSON-RPC 2.0.

---

## 2. IPC-First Architecture (Pattern for Other Springs)

neuralSpring implements "Tier 4 IPC-first" — the recommended pattern for all springs:

### How It Works

- `Cargo.toml` `default = []` — no primal crates linked at compile time
- GPU-dependent code behind `#[cfg(feature = "barracuda")]`
- 241 GPU-only binaries use `required-features = ["barracuda"]` in `Cargo.toml`
- CPU fallback implementations for core math (`sigmoid`, `gelu`, `relu`, `pearson_r`, etc.)
- `cargo check --workspace` and `cargo test` pass without any primals

### Why Springs Should Adopt This

1. **Build isolation** — spring compiles without primals installed
2. **Test isolation** — 693 tests run without GPU hardware
3. **Deployment flexibility** — UniBin deploys on bare CPU, GPU activates via IPC
4. **Audit clarity** — `primalSpring` can audit spring code without primal trees

### Migration Recipe for Other Springs

```
1. Set `default = []` in Cargo.toml
2. Feature-gate primal re-exports: #[cfg(feature = "primal_name")]
3. Add `required-features` to GPU-dependent [[bin]] stanzas
4. Provide CPU fallbacks for core math functions
5. Verify: cargo check --workspace && cargo test (no features)
6. Verify: cargo test --features primal_name (full suite)
```

---

## 3. NUCLEUS Composition Patterns

neuralSpring validates NUCLEUS composition at 6 layers (guideStone L0–L5):

### Layer Architecture

| Layer | What It Validates | Test Count |
|-------|-------------------|------------|
| L0 (bare) | 5 certified properties without primals | 29 checks |
| L1 (discovery) | `CompositionContext` — can we find primals? | liveness probes |
| L2 (parity) | Domain science — does IPC math match CPU? | 7 capabilities |
| L3 (nucleus) | Additive NUCLEUS — signing, mesh discovery | BearDog + Songbird |
| L4 (composition) | Deploy graphs, capability registry, family calls | graph validation |
| L5 (cross-spring) | Frozen artifacts, protocol liveness, hash determinism | cross-substrate |

### Composition via Deploy Graphs

neuralSpring defines NUCLEUS composition via `deploy_graph` fragments:

```toml
# downstream_manifest.toml (excerpt)
bond_type = "Metallic"
trust_model = "InternalNucleus"
fragments = ["tower_atomic", "node_atomic", "meta_tier"]
primals = ["beardog", "songbird", "skunkbat", "coralreef",
           "toadstool", "barracuda", "nestgate", "squirrel"]
```

Each fragment describes which primals compose a deployment tier.
Springs validate against these graphs — they don't define them.

### For Upstream: Composition Gaps

| Gap | Owner | Status | Impact |
|-----|-------|--------|--------|
| Gap 5: NestGate `storage.retrieve` | NestGate team | Open | Model weight storage via IPC |
| Gap 6: BearDog BTSP session | BearDog team | WIP | End-to-end signing chain |
| Gap 9: barraCuda feature-gate bug | barraCuda team | Open | `plasma_dispersion` unconditional import |
| Gap 11: barraCuda JSON-RPC surface | barraCuda team | Open | 18 method gaps in RPC surface |
| Gap 10: Shader upstream absorption | toadStool/coralReef | Tracking | 17 upstream shader candidates |

---

## 4. Deployment via neuralAPI from biomeOS

### The Deployment Stack

```
biomeOS (orchestrator)
  └─ neuralAPI (JSON-RPC gateway — routes to springs)
       └─ neuralSpring UniBin (single binary, subcommands)
            ├─ neuralspring validate    (run certification)
            ├─ neuralspring certify     (guideStone L0-L5)
            └─ neuralspring serve       (serve via IPC)
```

### Socket Discovery

neuralSpring discovers biomeOS sockets via:
1. `$BIOMEOS_SOCKET_DIR` env var (override)
2. `$XDG_RUNTIME_DIR/biomeos/` (default)
3. Family-isolated: `$BIOMEOS_SOCKET_DIR/$FAMILY_ID/` for multi-tenant

Constants in `src/config.rs`:
- `ENV_BIOMEOS_SOCKET_DIR` = `"BIOMEOS_SOCKET_DIR"`
- `DEFAULT_FAMILY_ID` = `"default"`

### For biomeOS Team

neuralSpring is ready for neuralAPI integration:
- UniBin builds with `cargo build --release --bin neuralspring_unibin`
- No-GPU mode: `cargo build --no-default-features --features guidestone --bin neuralspring_unibin`
- Health check: `health.check` / `health.liveness` / `health.readiness` on the surface
- 34 capabilities registered in `capability_registry.toml`

---

## 5. Patterns Learned (For Downstream Absorption)

### What Worked Well

1. **Capability constants module** — `src/capabilities.rs` with 31 `pub const` entries. Eliminates typos in JSON-RPC method names. Every spring should have one.

2. **Primal names with display variants** — `src/primal_names.rs` separates lowercase discovery hints (`"barracuda"`) from display names (`"barraCuda"`). Prevents presentation bugs.

3. **Typed IPC errors** — `IpcError::{NotDiscovered, Transport, Protocol}` replaces `Result<_, String>`. Callers can match on error kind rather than parsing messages.

4. **Workspace dependencies** — `[workspace.dependencies]` in root `Cargo.toml` centralizes versions. Prevents `serde` 1.0.214 in one crate and 1.0.210 in another.

5. **CPU fallback pattern** — `#[cfg(not(feature = "barracuda"))]` provides pure-Rust implementations so the spring works without GPU. Simple, no trait gymnastics.

6. **6-layer certification** — guideStone L0–L5 progressively validates from bare properties to cross-spring parity. Each layer is a separate module that can be tested independently.

7. **Required-features gating** — GPU binaries declare `required-features = ["barracuda"]` in `Cargo.toml`. `cargo check --workspace` passes without GPU; CI can test CPU-only.

### What Other Springs Should Adopt

| Pattern | Files to Reference | Effort |
|---------|-------------------|--------|
| Capability constants | `src/capabilities.rs` | Low — create module, move literals |
| Primal name constants | `src/primal_names.rs` | Low — one module |
| IPC-first defaults | `Cargo.toml` (`default = []`) | Medium — feature-gate audit |
| Typed IPC errors | `src/error.rs` (`IpcError`) | Medium — error hierarchy |
| CPU fallbacks | `src/primitives.rs`, `src/metrics.rs` | Medium — per-function |
| guideStone certification | `src/certification/` (6 modules) | High — incremental layers |
| Workspace deps | `Cargo.toml` (`[workspace.dependencies]`) | Low — centralize versions |

---

## 6. Open Items for primalSpring Audit

1. **Test count reconciliation** — confirm 1,300 / 693 / 1,453 match `cargo test` output on CI
2. **Gap 5/6/9/11** — upstream primal teams should review and absorb or close
3. **LTEE B1** — B1 Python baseline complete; Rust reproduction is next sprint
4. **Foundation Threads 5+7** — seeded; sweetGrass braid integration pending review
5. **Shader absorption (Gap 10)** — 17 upstream candidates for toadStool/coralReef
6. **`ipc_dispatch` removal** — verify no downstream consumers reference the old module

---

*neuralSpring V152 | Session S201b | AGPL-3.0-or-later*
