# neuralSpring → biomeOS NUCLEUS Integration Handoff

**Date**: February 27, 2026
**From**: neuralSpring (Session 88+)
**To**: biomeOS / ToadStool / ecosystem primals
**Type**: New integration — neuralSpring as biomeOS science primal
**Validation**: `validate_biomeos_spectral` 29/29 PASS

---

## Summary

neuralSpring is now registered as a science capability provider in the biomeOS
ecosystem, following the wetSpring pattern. This handoff documents the integration
points, capability translations, and deployment requirements for the biomeOS team.

## What Was Built

### 1. Capability Registry (`config/capability_registry.toml`)

7 neuralSpring science capabilities registered under `[translations.science]`:

| Capability | Method | Description |
|-----------|--------|-------------|
| `science.spectral_analysis` | `science.spectral_analysis` | Full eigendecomposition → IPR/LSR |
| `science.anderson_localization` | `science.anderson_localization` | Disorder sweep with Anderson Hamiltonians |
| `science.hessian_eigen` | `science.hessian_eigen` | Loss landscape Hessian eigenanalysis |
| `science.agent_coordination` | `science.agent_coordination` | Multi-agent coordination spectral |
| `science.ipr` | `science.ipr` | Inverse participation ratio |
| `science.disorder_sweep` | `science.disorder_sweep` | IPR vs disorder strength |
| `science.training_trajectory` | `science.training_trajectory` | Spectral evolution over epochs |

New domain registered: `[domains.neural_science]` with provider `neuralspring`.

### 2. Pipeline Graph (`graphs/neuralspring_spectral_pipeline.toml`)

7-phase orchestration graph:

```
Phase 1: Check ToadStool, NestGate, neuralSpring health
Phase 2: Anderson localization analysis (capability.call)
Phase 3: Hessian eigenanalysis (capability.call)
Phase 4: Agent coordination spectral (capability.call)
Phase 5: Training trajectory analysis (capability.call)
Phase 6: Store all results in NestGate with provenance
Phase 7: Validate pipeline health and result integrity
```

### 3. Primal Adapter (`neuralspring_primal`)

JSON-RPC 2.0 server binary, feature-gated behind `--features primal`:

- Socket: `$XDG_RUNTIME_DIR/biomeos/neuralspring-{family_id}.sock`
- Follows biomeOS 5-tier socket resolution (env, XDG, /run/user, Android, /tmp)
- Line-delimited JSON-RPC over Unix sockets
- Concurrent request handling (semaphore-bounded at 4)
- All 7 capabilities + health endpoint

### 4. SDK Evolution (`biomeos-types`, `biomeos-primal-sdk`)

- `PrimalCapability::science()` factory method added to `biomeos-types`
- `providers_for_capability` updated: `("science", _) => &["wetspring", "neuralspring"]`
- `capability_from_primal_name("neuralspring")` → `PrimalCapability::science()`

### 5. Integration Validator (`validate_biomeos_spectral`)

29 checks exercising the full round-trip:

- Health check (status, capabilities list)
- IPR computation (exact parity with CPU at 1e-12)
- Disorder sweep (3 points, exact parity with CPU)
- Spectral analysis (eigenvalue count, IPR > 0, LSR in [0,1])
- Anderson localization (IPR increases with disorder)
- Hessian eigenanalysis (10 exact eigenvalues, trace = 55.0)
- Agent coordination (2 disorder points, all IPR > 0)
- Training trajectory (11 epochs, valid IPR/entropy)
- Error handling (method_not_found returns proper RPC error)

## NUCLEUS Deployment Status

All 7 plasmidBin binaries available at `plasmidBin/stable/x86_64/primals/`:

| Binary | Status | Size |
|--------|--------|------|
| beardog | Built (x86_64 ELF) | Available |
| songbird | Built (x86_64 ELF, 19 MB) | Available |
| toadstool | Built (x86_64 ELF, 10 MB) | Available |
| nestgate | Built (x86_64 ELF) | Available |
| squirrel | Built (x86_64 ELF) | Available |
| biomeos | Built (release) | Available |
| biomeos-api | Built (release) | Available |

Deploy: `biomeos nucleus start --mode full --node-id neuralspring-dev`

## Absorption Targets for biomeOS

1. **neuralspring_primal genomeBin**: Build as static binary and add to
   `plasmidBin/stable/x86_64/primals/neuralspring` for NUCLEUS deployment.

2. **Graph composition**: `neuralspring_spectral_pipeline.toml` can be composed
   with `science_pipeline.toml` for cross-spring analysis.

3. **Capability discovery**: neuralSpring should be auto-discovered by
   `biomeos nucleus start` when the binary is in the search path.

4. **NestGate provenance**: Results stored via `storage.store` include
   `family_id = "neuralspring"` for provenance tracking.

## Evolution Path

```
Current:  neuralSpring primal (JSON-RPC, local socket)
Next:     NUCLEUS deployment (auto-start with biomeos nucleus)
Future:   Cross-spring pipelines (wetSpring → neuralSpring spectral)
          GPU-accelerated capabilities (route eigensolve to ToadStool)
          Plasmodium collective (multi-NUCLEUS coordination)
```

---

**Gate**: Eastgate
**Validation**: `validate_biomeos_spectral` 29/29 PASS
**Dependencies**: `biomeos-primal-sdk` (path dep), `biomeos-types` (transitive)
