# neuralSpring V145 — Doc Reconciliation, Upstream Primal Handoff & Archive Sweep

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Session:** S195 | **Date:** May 9, 2026 | **Supersedes:** V144

---

## 1. What Changed (S194-S195)

### S194: Deep Debt Sweep
- Feature-gated `loss_landscape`, `weight_spectral`, `wdm_esn` behind `#[cfg(feature = "barracuda")]` (11 total modules now gated)
- 20 new inline tests: IPC edge cases (9), rpc_service serde (8), config resolve_family_id (3)
- Centralized `BIOMEOS_FAMILY_ID` in `config::resolve_family_id()` — eliminates env var string duplication
- Aligned `temp-env` to `0.3.6` across workspace

### S195: Doc Reconciliation & Upstream Handoff
- All canonical test counts unified to **1,295 lib + 73 forge + 80 playGround = 1,448**
- All session/handoff refs unified to **S195 / V145**
- Deploy graph `neuralspring_deploy.toml` metadata updated from V137/S186 to V145/S195
- Frozen `validation-state.json` refreshed (was S188/1,234 lib)
- `NOTEBOOK_PATTERN.md` completed with Liu faculty batch (8 total notebooks, 72/72 checks)

---

## 2. neuralSpring Current Architecture (for upstream teams)

### 2a. IPC Surface — Per-Primal Modules

neuralSpring's `src/ipc/` tree provides per-primal IPC adapters via `IpcMathClient`:

| Module | Primal | Methods | Used For |
|--------|--------|---------|----------|
| `ipc/barracuda.rs` | barraCuda | `stats.mean`, `stats.std_dev`, `stats.weighted_mean`, `tensor.matmul`, `tensor.create` | Core math + tensor lifecycle |
| `ipc/toadstool.rs` | toadStool | `compute.dispatch` | Heterogeneous compute routing |
| `ipc/beardog.rs` | BearDog | `crypto.hash` | BLAKE3 signing, provenance |
| `ipc/squirrel.rs` | Squirrel | `inference.complete`, `inference.embed`, `inference.models` | LLM inference routing |
| `ipc/coralreef.rs` | coralReef | `shader.compile.wgsl`, `shader.capabilities` | GPU shader compilation |

All calls use `CompositionContext::from_live_discovery_with_fallback()` and JSON-RPC 2.0 over Unix sockets with biomeOS 4-tier socket resolution.

### 2b. Eukaryotic Architecture (post-S193)

- **UniBin**: `neuralspring-unibin` with `certify`, `validate`, `serve`, `status`, `version` subcommands
- **Certification organelle**: `src/certification/` — 4-layer guidestone (bare/discovery/parity/nucleus)
- **Validation scenarios**: `src/validation/scenarios/` — 6 absorbed scenarios with `ScenarioMeta`
- **IPC tree**: `src/ipc/` — 5 per-primal modules (graduated from monolithic `ipc_dispatch.rs`)
- **Fossilized**: `fossilRecord/` — 3 pre-extinction patterns with provenance READMEs

### 2c. Deploy Graphs

| Graph | Scope | Primals |
|-------|-------|---------|
| `neuralspring_deploy.toml` | Full NUCLEUS composition | barraCuda, toadStool, BearDog, Squirrel, coralReef, Songbird |
| `neuralspring_spectral_analysis.toml` | Science domain | barraCuda |
| `neuralspring_inference_pipeline.toml` | Inference chain | Squirrel |
| `neuralspring_math_pipeline.toml` | Math composition | barraCuda, toadStool |

---

## 3. Upstream Primal Team Debt (Hand-Backs)

### 3a. barraCuda — 18 JSON-RPC Surface Gaps

neuralSpring uses `barracuda::` library calls that have **no JSON-RPC equivalent**, blocking full Level 5 IPC-only deployment. Current surface: 32 methods. Required additions:

| Category | Missing Methods | Priority |
|----------|----------------|----------|
| **Linear algebra** | `linalg.eigh`, `linalg.solve`, `linalg.svd` | HIGH — blocks spectral analysis IPC |
| **Statistics** | `stats.pearson`, `stats.chi_squared`, `stats.shannon`, `stats.spectral_density`, `stats.marchenko_pastur` | HIGH — blocks science composition parity |
| **Neural network** | `nn.forward` (SimpleMlp/DenseLayer), `nn.conv`, `nn.pool` | MEDIUM — blocks inference composition |
| **Domain** | `bio.belief_propagation`, `bio.wright_fisher`, `ode.rk4`, `ode.rk45` | MEDIUM — blocks bio domain |
| **Reservoir** | `esn.*` (ESN config/train/predict) | LOW — WDM classifier path |

**Feature-gate bug**: `special::plasma_dispersion` imports `Complex64` from `domain-lattice`-gated module unconditionally. neuralSpring works around with `domain-lattice` feature enabled. Fix belongs upstream.

### 3b. toadStool — Compute Dispatch Surface

- Current: `compute.dispatch` works for basic routing
- Needed: Stable JSON-RPC wire contract for `compute.offload` with workload metadata (shape, dtype, priority)
- neuralSpring's `Dispatcher` routes mixed CPU/GPU workloads; toadStool integration validates this path

### 3c. coralReef — Shader Compilation Wire Contract

- Current: `shader.compile.wgsl` and `shader.capabilities` work
- 17 neuralSpring WGSL shaders are candidates for upstream absorption into barraCuda's `ops/` modules:
  - `hmm_backward_log.wgsl`, `hmm_viterbi.wgsl` → `ops::bio`
  - `batch_ipr.wgsl` → `spectral`
  - `chi_squared_f64.wgsl`, `kl_divergence_f64.wgsl`, `linear_regression.wgsl`, `matrix_correlation.wgsl` → `stats`
  - `rk4_parallel.wgsl`, `rk45_adaptive.wgsl` → `ops::ode`
  - `wright_fisher_step.wgsl`, `pairwise_hamming.wgsl` → `ops::bio`

### 3d. BearDog/Songbird — Tower Integration

- **Current**: Discovery probing wired, `crypto.hash` IPC operational
- **Needed**: BTSP session establishment, Songbird mesh discovery (replace filesystem socket scanning), signed capability announcements
- Tower Atomic startup validates BearDog + Songbird presence via `health.liveness`

### 3e. Squirrel — Inference Provider Registration

- Current: `inference.complete`, `inference.embed`, `inference.models` all operational
- Needed: `inference.register_provider` for dynamic provider routing (ollama, vLLM, native-wgsl)

### 3f. NestGate — Weight Storage (spring-deploy only)

- Not in proto-nucleate `depends_on`; belongs to full spring-deploy graph
- Current: Local `safetensors` loading via `weight_loader.rs`
- Needed: `storage.retrieve` IPC client for NestGate-mediated weight loading

---

## 4. Downstream Spring Team Absorption Patterns

### 4a. Patterns Available for Other Springs to Absorb

| Pattern | Source | Description |
|---------|--------|-------------|
| **IPC tree graduation** | `src/ipc/` | Monolithic IPC dispatch → per-primal modules. Each spring should have its own `ipc/{primal}.rs` leaf modules. |
| **Certification organelle** | `src/certification/` | 4-layer guidestone (bare/discovery/parity/nucleus) as a library module, not a standalone binary. |
| **Validation scenarios** | `src/validation/scenarios/` | `ScenarioMeta` + `ScenarioRegistry` with tiered execution (Rust-only vs Live IPC). |
| **UniBin consolidation** | `src/bin/neuralspring_unibin/` | Single binary with `certify`, `validate`, `serve`, `status`, `version` subcommands. |
| **Feature gate discipline** | `src/lib.rs` | 11 modules gated behind `#[cfg(feature = "barracuda")]`. Pattern: primal deps always optional, code gated, default features enable them. |
| **Centralized config** | `src/config.rs` | All env vars, socket resolution, family ID resolution, capability constants in one module. No ad-hoc strings. |
| **Fossilization** | `fossilRecord/` | Archive pre-extinction code with provenance READMEs. Don't delete — fossilize. |
| **Deprecated migration** | `src/lib.rs` L129-133 | `#[deprecated(since, note)]` on old modules with migration guidance. |

### 4b. NUCLEUS Composition Patterns for neuralAPI / biomeOS

neuralSpring's deployment via biomeOS neuralAPI follows:

1. **Socket resolution**: `config::resolve_biomeos_socket_dir()` — 4-tier hierarchy (`$BIOMEOS_SOCKET_DIR` → `$XDG_RUNTIME_DIR/biomeos/` → `/run/user/{uid}/biomeos/` → `temp_dir()/biomeos/`)
2. **Family isolation**: `config::resolve_family_id()` — `$FAMILY_ID` → `$BIOMEOS_FAMILY_ID` → `"default"`
3. **Capability discovery**: `CompositionContext::from_live_discovery_with_fallback()` — discovers primals by capability, not by name
4. **Bond type**: Metallic (shared-memory transport within NUCLEUS)
5. **Trust model**: InternalNucleus (primals trust each other within the composition)
6. **Health triad**: `health.liveness`, `health.readiness`, `health.check` — all wired and validated

### 4c. neuralAPI Deployment Surface

neuralSpring advertises **30 capabilities** via Songbird discovery:

- **Science** (14): spectral_analysis, anderson_localization, hessian_eigen, agent_coordination, ipr, disorder_sweep, training_trajectory, evoformer_block, structure_module, folding_health, gpu_dispatch, cross_spring_provenance, cross_spring_benchmark, precision_routing
- **Health** (3): liveness, readiness, check
- **Inference** (3): complete, embed, models
- **Provenance** (4): begin, record, complete, status
- **Routing** (2): primal.forward, primal.discover
- **Meta** (4): capability.list, compute.offload, identity.get, mcp.tools.list

---

## 5. Quality Gates

| Gate | Status |
|------|--------|
| `cargo test --workspace --lib` | **1,448 PASS** (1,295 lib + 73 forge + 80 playGround) |
| `cargo fmt --check` | **0 diffs** |
| `cargo clippy --lib` | **pre-existing doc_markdown suggestions only** |
| unsafe code | **zero** |
| `#[allow()]` | **zero** (all `#[expect(reason)]`) |
| TODO/FIXME/HACK/DEBT | **zero** in src/ |
| Mocks in production | **zero** |
| Feature gate consistency | **11 modules** gated behind barracuda |
| Deploy graph metadata | **V145/S195** synchronized |

---

## 6. Archive Sweep Results

- **No** `.bak`, `.old`, `.tmp`, `.orig` files found
- **No** stale standalone scripts (all referenced in docs or README)
- **3 fossilized patterns** in `fossilRecord/` with provenance READMEs
- **1 deprecated module** in `src/lib.rs` (`ipc_dispatch`) with migration note
- **TBD markers** remain in `specs/coral_forge_assessment/MSA_DATABASE_PLAN.md` and `metalForge/MIXED_HARDWARE_DESIGN.md` — these are planning docs, not production code

---

## 7. Open Gaps Summary (for primalSpring audit)

| # | Gap | Owner | Status |
|---|-----|-------|--------|
| 5 | NestGate weight storage IPC | NestGate | open |
| 6 | BearDog/Songbird Tower (BTSP session) | BearDog, Songbird | wip |
| 9 | barraCuda plasma_dispersion feature-gate bug | barraCuda | open (workaround) |
| 10 | 17 shader absorption candidates | barraCuda, coralReef | tracking |
| 11 | 18 barraCuda JSON-RPC surface gaps | barraCuda | open |

All other gaps (1-4, 7-8, 12-14, R1-R13, CE1-CE8) are **resolved** or **implemented**.
