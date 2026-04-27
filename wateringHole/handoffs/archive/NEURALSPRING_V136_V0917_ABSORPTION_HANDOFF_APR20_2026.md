# neuralSpring V136 — primalSpring v0.9.17 Absorption Handoff

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Date**: April 20, 2026
**Session**: S185
**primalSpring**: v0.9.17 (Phase 45)
**guideStone Standard**: v1.2.0
**guideStone Binary**: `neuralspring_guidestone` v0.3.0
**guideStone Level**: 3 (bare ALL PASS — 29/29 checks, P1-P5 certified)

---

## What Changed

### `is_skip_error` Adoption

Replaced 7 manual error-matching arms in `neuralspring_guidestone.rs` with
`primalspring::composition::is_skip_error()`. This unified predicate covers:

- Connection errors (absent primals)
- Protocol errors (HTTP-on-UDS — Songbird, petalTongue)
- Transport mismatches (BTSP dialect — BearDog)

**Before** (Phase 3 domain parity + Phase 4 additive NUCLEUS):
```rust
Err(e) if e.is_connection_error() => { v.check_skip(...) }
// Phase 4 had separate arms:
Err(e) if e.is_connection_error() => { ... }
Err(e) if e.is_protocol_error() => { ... }
```

**After**:
```rust
Err(e) if is_skip_error(&e) => { v.check_skip(...) }
```

Phase 3 functions now correctly skip on protocol errors (previously only
caught connection errors). Phase 4 functions collapsed from 2 skip arms to 1.

### guideStone Standard v1.2.0

Doc references updated from v1.1.0. v1.2.0 additions:

- **Tolerance hierarchy** as ecosystem standard (`EXACT_PARITY_TOL` through
  `STOCHASTIC_SEED_TOL`). neuralSpring already uses `IPC_ROUND_TRIP_TOL`
  for all IPC parity checks — compliant.
- **`call_or_skip` / `is_skip_error`** absorbed into `primalspring::composition`.
  neuralSpring adopted `is_skip_error`; `call_or_skip` available for future use.
- **"Domain functions are local compositions"** pattern documented.
  neuralSpring's domain science composes from generic primitives proven
  correct via IPC — already follows this pattern.

### primalSpring v0.9.17 Integration

No new library API. Checksums, ValidationResult, IPC, composition all
unchanged from v0.9.16. The v0.9.17 delta is deployment validation and
operational contracts:

- Local NUCLEUS 12/12, Docker 12/12, remote fetch 13/13, Pixel aarch64 staged
- genomeBin v5.1: 46 binaries across 6 target triples
- `start_primal.sh` CLI-audited for every primal's actual interface

---

## Operational Awareness (Level 4 Deployment)

These requirements are documented in `docs/PRIMAL_GAPS.md` Gap 13 and apply
when deploying a live NUCLEUS for Level 4 validation:

| Primal | Change | Detail |
|--------|--------|--------|
| coralReef | CLI | `--port` → `--rpc-bind` for TCP; `--tarpc-bind unix://` for UDS (iter84) |
| beardog | Env | `BEARDOG_FAMILY_SEED` required for production BTSP |
| songbird | Env | `SONGBIRD_SECURITY_PROVIDER=beardog` required |
| nestgate | Env | `NESTGATE_JWT_SECRET` required (random Base64) |

### genomeBin v5.1

46 binaries across 6 target triples:
- x86_64-unknown-linux-musl
- aarch64-unknown-linux-musl
- armv7-unknown-linux-musleabihf
- x86_64-pc-windows-msvc
- aarch64-linux-android
- riscv64gc-unknown-linux-musl

Level 4 deployment path is clear: pull plasmidBin, launch NUCLEUS via
`start_primal.sh`, run `neuralspring_guidestone` externally.

---

## Level 4 Path

1. Deploy live NUCLEUS from plasmidBin (12 primals)
2. Run `primalspring_guidestone` — must exit 0 (base certification)
3. Run `neuralspring_guidestone` — all 7 `PROTO_NUCLEATE_VALIDATION_CAPABILITIES`
   must return PASS (not SKIP): `stats.mean`, `tensor.matmul`, `tensor.create`,
   `compute.dispatch`, `crypto.hash`, `inference.complete`, `inference.embed`
4. Document any gaps in `docs/PRIMAL_GAPS.md` and hand back

---

## Manifest Note

`downstream_manifest.toml` shows `guidestone_readiness = 2` for neuralSpring.
Actual status is Level 3 (29/29 bare ALL PASS since S184). The manifest is
upstream's responsibility — flagged for primalSpring to reconcile.

---

## Primal Use & Evolution Patterns

### Dependency Profile

| Primal | Usage | Feature Gate |
|--------|-------|-------------|
| barraCuda | Library (250+ files) + IPC (9 `IpcMathClient` methods) | default |
| primalSpring | Library (composition API, checksums, validation, tolerances) | `guidestone` |
| toadStool | IPC only (`compute.dispatch`) | `primal` |
| coralReef | Shader compilation reference only | none |
| BearDog | IPC only (`crypto.hash`) | `primal` |
| Songbird | IPC only (discovery) | `primal` |
| Squirrel | IPC only (`inference.complete`, `inference.embed`) | `primal` |

### IPC Migration Summary

- **9 methods wired** via `IpcMathClient` (S182): `stats.mean`, `tensor.matmul`,
  `tensor.create`, `tensor.add`, `compute.dispatch`, `crypto.hash`,
  `inference.complete`, `inference.embed`, `inference.models`
- **18 barraCuda surface gaps** (Gap 11): Methods with no JSON-RPC equivalent
  (eigenvalue decomposition, Pearson correlation, chi-squared, spectral density,
  ESN network, neural network ops, belief propagation, etc.)
- **Gap 11 unchanged** in v0.9.17 — barraCuda IPC surface still at 14 methods

### NUCLEUS Composition Patterns

neuralSpring's deploy graph (`neuralspring_deploy.toml`, V136/S185) extends the
proto-nucleate with a nest_atomic fragment and the neuralspring binary node:

- **Fragments**: `tower_atomic`, `node_atomic`, `nest_atomic`, `meta_tier`
  (superset of proto-nucleate which uses `tower_atomic`, `node_atomic`, `meta_tier`)
- **Bonding**: Metallic (InternalNucleus, UDS-only, BTSP at Tower boundary)
- **Resolve**: `resolve = true` in `[graph.metadata]` — fragments merge as base
  layer, profile applies delta only (per primalSpring graph consolidation pattern)
- **Topological waves**: biomeOS deploys in dependency order — beardog first
  (Tower trust boundary), then Node-tier (toadstool, barracuda, coralreef),
  then Nest-tier, then meta-tier (biomeos, squirrel)

The proto-nucleate (pure primal, no spring binary) declares 7 validation
capabilities. The deploy graph adds neuralspring as a niche node with 30
capabilities — this is the Level 2 "Rust proof" graph. For Level 4+ (primal
proof), the guideStone validates the proto-nucleate externally without being
part of the graph.

### neuralAPI / biomeOS Deployment

neuralSpring provides AI inference for the ecosystem. The deployment path:

1. **biomeOS deploys NUCLEUS** — `biomeos deploy --graph nucleus_complete.toml`
   spawns all primals on UDS sockets
2. **neuralspring primal starts** — `neuralspring --socket /run/biomeos/neuralspring.sock`
   registers 30 capabilities including `science.*`, `inference.*`, `health.*`
3. **Squirrel discovers neuralspring** — any spring adding Squirrel to its
   composition gains `inference.*` capabilities automatically
4. **neuralAPI routing** — biomeOS neural-api tier resolves capabilities to
   endpoints; `discover_by_capability("tensor")` queries biomeOS first, then
   falls back to socket file scanning with family isolation

The `neuralspring_primal` binary (feature-gated: `primal`) implements:
- tarpc service with typed request/response structs
- JSON-RPC handlers for all 30 capabilities
- `try_squirrel_route()` fallback for inference requests
- Tower Atomic startup discovery (BearDog + Songbird probes)
- Health triad: `health.liveness`, `health.readiness`, `health.check`

### Ecosystem Learnings

1. **`is_skip_error` is the correct skip predicate** — manual `is_connection_error()`
   misses protocol errors. All IPC error arms should use `is_skip_error`.
2. **Tolerance hierarchy matters** — using `IPC_ROUND_TRIP_TOL` (not `EXACT`)
   for IPC parity checks prevents false failures from JSON serialization noise.
   Domain-specific tolerances (e.g., SEMF, plaquette) should cite their source.
3. **guideStone standard evolves with ecosystem** — v1.1.0 → v1.2.0 absorbed
   patterns from 5 springs. Domain springs should track standard version.
4. **genomeBin multi-arch** — 6 target triples means Level 4 validation can
   run on any platform with pre-built binaries. No source compilation needed.
5. **Operational contracts are the Level 4 gap** — the science code is ready;
   the deployment environment setup (env vars, CLI flags) is the real work.
6. **Feature-gate IPC dependencies** — neuralSpring gates `primalspring` behind
   `guidestone` and `tarpc`/`clap` behind `primal`. This keeps the core library
   (169 files of barracuda math) free of IPC dependencies. Other springs should
   follow this pattern — science library is default, IPC wiring is opt-in.
7. **`IpcMathClient` as validation window** — the temporary typed IPC client
   proved math works through NUCLEUS before the guideStone took over. Springs
   at Level 1–2 should build this intermediate step rather than jumping to
   full guideStone immediately.
8. **30 capabilities is the deployment floor** — health triad + identity +
   MCP tools + domain science + inference routing. Springs that expose fewer
   capabilities will not pass biomeOS niche deployment validation.
9. **`composed` feature is vestigial** — neuralSpring defined it as an alias
   for `primal` with no unique surface. Springs should avoid phantom features.

### For Upstream Primal Teams

**barraCuda**: 18 IPC surface gaps (Gap 11) block full domain science parity.
Priority methods: `linalg.eigenvalues` (PCA, spectral analysis), `stats.pearson`
(correlation analysis), `spectral.power_spectrum` (time-series), `nn.forward`
(neural network inference). These are the most-used library calls that have no
JSON-RPC equivalent.

**toadStool**: `compute.dispatch` works for probes. neuralSpring would benefit
from `compute.benchmark` for performance regression testing in guideStone
Level 4+ validation.

**Squirrel**: `inference.register_provider` — does this method exist? neuralSpring
needs to register as an inference backend so other springs can discover it.
Currently Squirrel discovery is one-way (neuralSpring finds Squirrel).

**coralReef**: iter84 CLI change (`--rpc-bind`) documented. neuralSpring's
8 shaders in coralReef corpus are validated via forge. No new shader needs
identified for Level 4.

**biomeOS**: Level 4 deployment requires `nucleus_launcher.sh` or equivalent
in plasmidBin. Currently not present — neuralSpring's `validate_clean_machine.sh`
uses env-driven socket discovery without a launcher.

---

## Files Changed (S185)

| File | Change |
|------|--------|
| `src/bin/neuralspring_guidestone.rs` | `is_skip_error` adoption, version v0.3.0, standard v1.2.0 |
| `docs/GUIDESTONE_PROPERTIES.md` | v1.2.0, v0.3.0, S185 evidence |
| `docs/PRIMAL_GAPS.md` | v0.9.17 integration notes, operational requirements |
| `CHANGELOG.md` | S185 entry |
| `README.md` | Session header update |
| `EVOLUTION_READINESS.md` | v0.9.17, V136, genomeBin v5.1 |
| `whitePaper/README.md` | Session update |
| `whitePaper/baseCamp/README.md` | Session update |
| `experiments/README.md` | Session update |
| `specs/README.md` | Session update |
| `graphs/neuralspring_deploy.toml` | V136/S185 |
| `src/bin/validate_composition_evolution.rs` | Doc ref V136/S185 |

---

**Hand back to**: primalSpring (manifest `guidestone_readiness` 2→3),
barraCuda (Gap 11 surface expansion), biomeOS (Level 4 deployment tooling)
