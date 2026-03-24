<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring V125 — Cross-Team Evolution Handoff

**Session**: S175 (March 24, 2026)
**Audience**: All springs, all primals, biomeOS integrators
**Previous**: V124 (S174 — deep audit, zero debt, provenance alignment)

---

## Summary

neuralSpring at V125/S175 is the **ML and spectral analysis validation harness**
for ecoPrimals. ~1,400 tests, 261 binaries, 68 modules, 232+ named tolerances,
zero debt. This handoff documents patterns, capabilities, and integration
surfaces for other teams to leverage.

---

## 1. Patterns We Export (Adopt These)

### ValidationSink (Ecosystem-Converged)

All four springs (wetSpring, airSpring, groundSpring, neuralSpring) now have
`ValidationSink` traits for machine-readable validation output. New springs
and primals with validation harnesses should adopt this pattern.

**neuralSpring implementation** (`src/validation/sink.rs`):
- Trait: `ValidationSink` with `on_check(&Check)` + `on_finish(name, passed, total, success)`
- 5 implementations: `StdoutSink`, `JsonSink<W>`, `NdjsonSink<W>`, `CollectingSink`, `SilentSink`
- Harness methods: `emit_to_sink()`, `emit_json()`
- 12 tests covering all sinks

### Upstream Contract Pinning

neuralSpring pins expected barraCuda tolerance values as constants
(`UPSTREAM_HYDRO_CROP_COEFFICIENT`, `UPSTREAM_PHYSICS_ANDERSON_EIGENVALUE`, etc.)
and tests they haven't silently drifted. Other springs consuming barraCuda
tolerances should adopt this pattern.

### Provenance Integrity Tests

4 tests verify all 49 registered Python baselines:
- Existence on disk
- `# Provenance:` header present
- SPDX license header present
- Content stability (hash + size)

Springs with Python baselines should adopt similar integrity checks.

### Cast Lint Deny

`cast_possible_truncation` and `cast_sign_loss` promoted to `deny` in workspace
lints. All cast sites covered by `#[expect]` with reasons. Recommended for all
springs — it catches silent precision loss.

---

## 2. Capabilities We Provide

16 JSON-RPC capabilities registered with biomeOS:

| Capability | Domain | Description |
|-----------|--------|-------------|
| `science.spectral_analysis` | Physics | Full spectral decomposition with phase classification |
| `science.anderson_localization` | Physics | Anderson localization (eigenvalues, IPR, LSR) |
| `science.hessian_eigen` | ML | Hessian eigendecomposition for loss landscape |
| `science.agent_coordination` | Ecology | Multi-agent quorum sensing via spectral graph theory |
| `science.ipr` | Physics | Inverse participation ratio |
| `science.disorder_sweep` | Physics | Anderson disorder sweep |
| `science.training_trajectory` | ML | Weight matrix spectral evolution during training |
| `science.evoformer_block` | Biology | AlphaFold evoformer block |
| `science.structure_module` | Biology | AlphaFold structure module |
| `science.folding_health` | Biology | coralForge model health check |
| `science.gpu_dispatch` | Infra | Arbitrary Dispatcher ops (44+ ops, GPU/CPU) |
| `science.cross_spring_provenance` | Meta | WGSL shader provenance report |
| `science.cross_spring_benchmark` | Bench | Dispatcher benchmark (7 ops) |
| `science.precision_routing` | Infra | Precision routing (fp64 strategy, bandwidth tier) |
| `health.liveness` | Health | K8s-style liveness probe |
| `health.readiness` | Health | K8s-style readiness probe |

### Deploy Graph

`graphs/neuralspring_deploy.toml` (V124/S174): 5-phase biomeOS deployment —
BearDog → Songbird → optional (ToadStool + NestGate + provenance trio) →
neuralSpring → validation → provenance recording. All dependencies are
capability-based (`by_capability`), not name-hardcoded.

---

## 3. For barraCuda Team

(See also the detailed `NEURALSPRING_BARRACUDA_ABSORPTION_HANDOFF_S174_MAR24_2026.md`)

### Absorption Candidates

4 generic f64 WGSL ops from neuralSpring's validation domain that are
reusable across springs:
- `introgression_forward_f64.wgsl` — generic matrix·vector chain
- `population_wright_fisher_f64.wgsl` — stochastic discrete iteration
- `hmm_spatial_f64.wgsl` — tridiagonal / banded matrix ops
- `multi_obj_fitness_f64.wgsl` — parallel reduction with Bessel correction

### Bessel Correction Documentation

`GPU_MULTI_OBJ_BESSEL_F64` (3e-3): GPU uses sample std (n-1), CPU uses
population std (n). Documented algorithmic difference, not a bug.

### Feature Proposal: `domain-fold`

Group neuralSpring's 15 domain-specific WGSL shaders under a `domain-fold`
feature gate in barraCuda. This is non-breaking — disabled by default.

---

## 4. For toadStool Team

### GPU Dispatch Surface

neuralSpring runs GPU through in-process `Dispatcher` (barracuda tensor API),
not via toadStool RPC. However, the deploy graph declares toadStool as an
optional dependency for:
- Hardware capability enrichment (`gpu.features`)
- Workload orchestration for multi-GPU

### Evolution Path

When toadStool absorbs `compute.dispatch` as a first-class RPC, neuralSpring
can optionally delegate GPU work via IPC instead of in-process. The
`compute.offload` handler already exists in our RPC surface.

---

## 5. For coralReef Team

8 neuralSpring WGSL shaders in coralReef's corpus. coralForge integration
(df64 core streaming) produces ~14-digit precision on FP32 cores. Three-tier
precision validated: f32/f64/df64.

### Known Issue

`enable f64;` in WGSL triggers PTXAS silent-zero regression on Ada Lovelace.
Fix implemented locally in `pipeline_cache.rs` — pending upstream absorption
into coralReef's shader compilation pipeline.

---

## 6. For biomeOS / primalSpring Team

neuralSpring registers via `nucleus.register` + `capability.register` on
`biomeos.sock`. Heartbeat loop active. Deploy graph at
`graphs/neuralspring_deploy.toml` is current (V124/S174).

### Integration Points

- `biomeos::register_with_biomeos()` — registers 16 capabilities
- `biomeos::deregister_from_nucleus()` — clean shutdown on SIGINT/SIGTERM
- `heartbeat_loop()` — periodic liveness signal
- `discover_primal_socket()` — capability-based discovery via orchestrator
- Socket path: `${XDG_RUNTIME_DIR}/biomeos/neuralspring-{family_id}.sock`

---

## 7. For All Springs

### Patterns to Adopt

| Pattern | Where | Why |
|---------|-------|-----|
| `ValidationSink` | `src/validation/sink.rs` | Machine-readable CI output |
| Upstream contract pinning | `src/tolerances/mod.rs` | Detect silent dependency drift |
| Provenance integrity tests | `src/provenance/mod.rs` | Baseline script stability |
| Cast lint deny | `Cargo.toml` workspace lints | Catch silent precision loss |
| `OrExit<T>` | `src/validation/mod.rs` | Panic-free unwrap in binaries |
| `require!` macro | `src/validation/mod.rs` | Graceful error recording |
| `gpu_tensor!` macro | `src/validation/mod.rs` | Boilerplate reduction |
| `PROVENANCE_REGISTRY` | `src/provenance/mod.rs` | Centralized baseline tracking |

### Patterns We Absorbed From You

| Pattern | From | Session |
|---------|------|---------|
| `ValidationSink` | wetSpring V134, airSpring V010, groundSpring V121 | S175 |
| `OrExit<T>` | wetSpring V123 | S159 |
| `OnceLock GPU probe` | groundSpring V116 | S168b |
| `IpcError` typed enum | healthSpring V31, rhizoCrypt V13 | S160 |
| Cast lint deny | airSpring V0.9.0 | S168b |
| `normalize_method` | barraCuda | S172 |
| `PRIMAL_DOMAIN` | healthSpring V34 | S168b |

---

## 8. Metrics

| Metric | Value |
|--------|-------|
| Lib tests | 1,211 |
| Forge tests | 73 |
| PlayGround tests | 80 |
| Total tests | ~1,400 |
| Validation binaries | 261 |
| Modules | 68 |
| Tolerances | 232+ |
| Cast lints | Denied |
| `#[allow()]` | Zero |
| unsafe | Zero (forbid) |
| C deps | Zero |
| Files > 1000 LOC | Zero |
| Clippy warnings | Zero |
| Production mocks | Zero |
