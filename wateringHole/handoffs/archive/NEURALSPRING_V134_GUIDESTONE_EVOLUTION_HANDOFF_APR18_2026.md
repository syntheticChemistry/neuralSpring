# neuralSpring V134 — guideStone Evolution Handoff

**Date**: April 18, 2026
**Session**: S183
**Spring version**: 0.1.0
**primalSpring**: v0.9.15
**barraCuda**: v0.3.12
**guideStone readiness**: Level 1 → Level 2 (properties documented)

---

## What Changed

### 1. `neuralspring_guidestone` binary

New self-validating NUCLEUS deployable following the guideStone standard
(`primalSpring/wateringHole/GUIDESTONE_COMPOSITION_STANDARD.md`).

**Binary**: `neuralspring_guidestone` (feature-gated: `guidestone`)
**Source**: `src/bin/neuralspring_guidestone.rs`

4-phase validation:

| Phase | Description | Primals needed |
|-------|-------------|----------------|
| 1. Bare Properties | P1 determinism, P2 provenance, P4 ecoBin, P5 tolerances | No |
| 2. Discovery + Liveness | `validate_liveness` on tensor/security/compute/ai | Yes |
| 3. Domain Science Parity | 7 capabilities via `validate_parity`/`validate_parity_vec` | Yes |
| 4. Additive NUCLEUS | BearDog signing receipt, Songbird discovery | Yes (graceful skip) |

Exit codes: 0 = certified, 1 = regression, 2 = bare only (no NUCLEUS)

Uses `primalspring::composition` API directly:
- `CompositionContext::from_live_discovery_with_fallback()` for UDS+TCP dual discovery
- `validate_liveness()` for Phase 2
- `validate_parity()` / `validate_parity_vec()` for Phase 3
- `hash_bytes()` / `resolve_capability()` for Phase 4

### 2. `primalspring` optional dependency

Added as optional path dep with `guidestone` feature gate:
```toml
primalspring = { path = "../primalSpring/ecoPrimal", optional = true }

[features]
guidestone = ["primalspring", "primal"]
```

### 3. guideStone Properties documented

New `docs/GUIDESTONE_PROPERTIES.md` with 5 certified properties:

| Property | Status |
|----------|--------|
| 1. Deterministic Output | CERTIFIED |
| 2. Reference-Traceable | PARTIAL (not yet machine-readable JSON) |
| 3. Self-Verifying | PARTIAL (no CHECKSUMS file) |
| 4. Environment-Agnostic | CERTIFIED |
| 5. Tolerance-Documented | CERTIFIED |

### 4. Gap 11 confirmed still open

The primalSpring v0.9.15 blurb claims expanded barraCuda IPC surface
(`stats.correlation`, `linalg.solve`, `spectral.fft`, etc.). Verified
against `barraCuda/crates/barracuda-core/src/ipc/methods/mod.rs`:
**these methods do NOT exist in `REGISTERED_METHODS`**. barraCuda still
has exactly 32 JSON-RPC methods. The 18 gaps documented in Gap 11 remain
accurate.

### 5. PRIMAL_GAPS.md — Gap 13 added

New section documenting guideStone evolution status, readiness matrix,
Level 3/4/5 blockers, and validation window documentation.

---

## For barraCuda

Gap 11 (18 JSON-RPC surface gaps) remains the primary blocker for full
domain science parity. The guideStone binary can validate the 7 existing
`PROTO_NUCLEATE_VALIDATION_CAPABILITIES` today. Expanding to full domain
coverage requires:
- `linalg.eigh` (eigendecomposition)
- `stats.pearson` (correlation)
- `stats.chi_squared`
- `stats.shannon` (entropy)
- `linalg.solve`
- `nn.forward`

The blurb's claim of expanded surface (stats.correlation, linalg.solve,
spectral.fft, etc.) does not match the actual codebase. Please advise
whether these are planned or if the documentation needs correction.

## For primalSpring

- `downstream_manifest.toml` entry for neuralspring should update
  `guidestone_readiness = 2` (was 1)
- Base certification: `primalspring_guidestone` must pass (exit 0) before
  domain guideStones can certify. Current guideStone level for primalSpring
  is 1 — domain guideStones will block at Level 4 until this is certified.

## For toadStool

- `compute.dispatch` is validated in Phase 3 of the guideStone. No new
  toadStool requirements.

## For Squirrel

- `inference.complete` and `inference.embed` are validated in Phase 3.
  Currently skip when Squirrel is unavailable. Squirrel provider
  registration remains an open gap (Gap 1).

## For All Springs

The guideStone pattern (`neuralspring_guidestone`) can serve as a reference
for other domain springs at Level 1:

```rust
// Add to Cargo.toml:
// primalspring = { path = "../primalSpring/ecoPrimal", optional = true }
// [features]
// guidestone = ["primalspring", "primal"]

use primalspring::composition::{
    CompositionContext, validate_liveness, validate_parity,
};
use primalspring::validation::ValidationResult;
use primalspring::tolerances;

let mut v = ValidationResult::new("myspring guideStone");
let mut ctx = CompositionContext::from_live_discovery_with_fallback();

// Phase 2: Discovery
let alive = validate_liveness(&mut ctx, &mut v, &["tensor", "security"]);
if alive == 0 {
    v.finish();
    std::process::exit(v.exit_code_skip_aware());
}

// Phase 3: Domain parity
validate_parity(
    &mut ctx, &mut v, "my_check", "tensor", "stats.mean",
    serde_json::json!({"data": [1.0, 2.0, 3.0]}),
    "result", 2.0, tolerances::IPC_ROUND_TRIP_TOL,
);

v.finish();
std::process::exit(v.exit_code_skip_aware());
```

---

## Files Changed

| File | Change |
|------|--------|
| `Cargo.toml` | +primalspring optional dep, +guidestone feature, +neuralspring_guidestone bin |
| `src/bin/neuralspring_guidestone.rs` | NEW — 4-phase guideStone binary |
| `docs/GUIDESTONE_PROPERTIES.md` | NEW — 5 properties documented |
| `docs/PRIMAL_GAPS.md` | Gap 11 note, Gap 13 (guideStone evolution) |
| `README.md` | S183 header, V134 references |
| `CHANGELOG.md` | S183 entry |
| `experiments/README.md` | Exp 130 entry |
| `EVOLUTION_READINESS.md` | S183 header, guideStone section |
| `specs/README.md` | S183 header, V134 references |
| `whitePaper/README.md` | S183 header, V134 references |
| `whitePaper/baseCamp/README.md` | S183 header, validation chain updated |
| `wateringHole/handoffs/` | V133 → archive, V134 created |

## Verification

```bash
cargo check --features guidestone              # ✓ clean
cargo clippy --features guidestone -- \
  -W clippy::pedantic -W clippy::nursery       # ✓ 0 warnings
cargo fmt --check                              # ✓ 0 diffs
cargo deny check                               # ✓ advisories ok, bans ok, licenses ok, sources ok
```

Tests: 1,234 lib + 73 forge + 80 playGround. 268 binaries. 520+ `.rs` files.

---

## Primal Use & Evolution Status

### Current primal dependency profile

| Primal | Usage | Level |
|--------|-------|-------|
| **barraCuda** | 234 files, library-level (`barracuda::` crate). Heavy: ops, dispatch, stats, nn, linalg, spectral, nautilus | Library (L2) + IPC (L5 for 3 methods) |
| **toadStool** | IPC only (`compute.dispatch`). playGround `ToadStoolClient` for `compute.submit`/`status`/`result` | IPC (L5) |
| **BearDog** | IPC only (`crypto.hash`). Tower Atomic discovery probes (`health.liveness`) | IPC (L5) |
| **Songbird** | IPC only. Tower Atomic discovery probes (`health.liveness`) | IPC (L5) |
| **Squirrel** | IPC only (`inference.complete`/`embed`/`models`). `try_squirrel_route` fallback in handlers | IPC (L5) |
| **coralReef** | Optional forge bridge (`coralreef_bridge.rs`). Not yet IPC-wired in main lib | Library (forge only) |
| **NestGate** | Not used (spring-deploy only, not proto-nucleate) | Not wired |
| **rhizoCrypt** | `primal_names::RHIZOCRYPT` defined but no active IPC wiring | Not wired |
| **loamSpine** | `primal_names::LOAMSPINE` defined but no active IPC wiring | Not wired |
| **sweetGrass** | `primal_names::SWEETGRASS` defined but no active IPC wiring | Not wired |
| **petalTongue** | IPC via `PetalTonguePushClient` (visualization push). RetryPolicy + CircuitBreaker | IPC (active) |
| **biomeOS** | IPC via `BiomeosClient` in playGround. Socket registration in primal binary | IPC (active) |

### IPC migration summary (barraCuda)

234 files use `barracuda::` as a library dependency. Only 3 methods are wired
for IPC today (`tensor.matmul`, `tensor.create`, `stats.mean`). Gap 11 documents
18 additional methods that lack JSON-RPC equivalents. For the guideStone to reach
Level 4+, these must migrate:
- **Immediate** (existing IPC methods): `stats.mean`, `tensor.matmul`, `tensor.create`
- **Composable** (via existing methods): `dot`, `l2_norm`, `rmse`, `mae`
- **Blocked** (no IPC equivalent): `eigh`, `pearson`, `chi_squared`, `solve_f64_cpu`,
  `esn_v2::*`, `nn::SimpleMlp`, `belief_propagation`, `graph_laplacian`, `nautilus::*`

### Composition patterns for NUCLEUS deployment via biomeOS

neuralSpring's deploy graph (`graphs/neuralspring_deploy.toml`) defines a
sequential coordination pattern:

```
biomeOS orchestrates → BearDog + Songbird (Tower) → toadStool + barraCuda + coralReef (Node)
  → NestGate + provenance trio (Nest) → Squirrel (Meta) → neuralspring (spring binary)
```

The spring binary registers with biomeOS at startup, advertises 30 capabilities
via `capability.register`, and serves domain science over JSON-RPC. The guideStone
binary validates this composition externally — it is NOT a graph node, it runs
OUTSIDE the NUCLEUS and certifies that the composition is sound.

### Patterns learned for ecosystem evolution

1. **Dual IPC stacks**: neuralSpring has sync IPC (`validation/composition.rs` +
   `IpcMathClient`) for validation binaries and async IPC (`playGround/ipc_client.rs`)
   for interactive tooling. Springs should standardize on `primalspring::composition`
   for guideStone-level work; playground async stacks are development tooling.

2. **ToadStool method naming**: `IpcMathClient` uses `compute.dispatch` while
   playGround's `ToadStoolClient` uses `compute.submit`/`status`/`result`. The
   proto-nucleate manifest declares `compute.dispatch`. toadStool should clarify
   which is canonical.

3. **`composed` feature unused**: neuralSpring defines `composed = ["primal"]`
   but no code gates on it. Originally intended for IPC-only composition paths.
   The `guidestone` feature now serves this role. Other springs should use
   `guidestone` as the standard feature gate for composition API access.

4. **Provenance trio names defined but unwired**: `primal_names::RHIZOCRYPT`,
   `LOAMSPINE`, `SWEETGRASS` exist as constants but have zero active IPC
   clients. These are ready for wiring when Nest Atomic composition is needed.

5. **guideStone inherits primalspring_guidestone**: Domain guideStones should NOT
   re-validate base composition correctness. `primalspring_guidestone` (exit 0)
   certifies that. Domain guideStones validate only their domain science on top.
