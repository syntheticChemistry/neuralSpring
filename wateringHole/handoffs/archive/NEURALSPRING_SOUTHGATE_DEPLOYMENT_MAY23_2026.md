# neuralSpring — southGate Covalent Gate Deployment

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Date:** May 23, 2026
**Session:** S216 — Post-Primordial Covalent Gate Deployment
**Version:** V172
**Gate:** southGate
**Hardware:** Ryzen 7 5800X3D, 128GB DDR4, Pop!_OS 22.04
**Directive:** primalSpring Wave 46+ — Post-Primordial Covalent Gate Deployment

---

## Deployment Summary

First live NUCLEUS deployment for neuralSpring. `composition_nucleus.sh` expanded
from 8 primals to the full 13-primal stack, matching the upstream
`nucleus_launcher.sh` patterns from `infra/plasmidBin/`. All 13 plasmidBin
binaries present and executable.

### Gate Assignment Confirmed

| Gate | Springs | Hardware |
|------|---------|----------|
| **southGate** | primalSpring (coord), **neuralSpring** | Ryzen 7 5800X3D, 128GB DDR4 |

### Deployment Command

```bash
COMPOSITION_NAME=southgate FAMILY_ID=southgate \
PETALTONGUE_LIVE=false NODE_ID=southgate \
  ./tools/composition_nucleus.sh start
```

---

## Primal Status (13/13 attempted)

| Primal | Phase | UDS Socket | Status | Notes |
|--------|-------|-----------|--------|-------|
| biomeOS | 0 | neural-api-southgate.sock | **UP** | cleartext bootstrap |
| BearDog | 1 | beardog-southgate.sock | **UP** | BTSP crypto root |
| Songbird | 1 | songbird-southgate.sock | **UP** | discovery.register unknown |
| skunkBat | 1 | — | **SKIPPED** | TCP-only, no UDS `--socket` flag |
| toadStool | 2 | toadstool-southgate.sock | **UP** | .jsonrpc.sock also created |
| barraCuda | 2 | math-southgate.sock (symlinked) | **UP** | barracuda-southgate.sock alias |
| coralReef | 2 | coralreef-core-southgate.sock (symlinked) | **UP** | coralreef-southgate.sock alias |
| NestGate | 2 | nestgate-southgate.sock | **UP** | JWT dev-mode |
| Squirrel | 2 | abstract @squirrel | **UP** | not file-discoverable |
| rhizoCrypt | 3 | — (TCP 9400/9401) | **UP** | ignores RHIZOCRYPT_SOCKET env |
| loamSpine | 3 | — | **CRASHED** | upstream Tokio double-runtime bug |
| sweetGrass | 3 | sweetgrass-southgate.sock | **UP** | |
| petalTongue | 4 | petaltongue-southgate.sock | **UP** | server mode, no CLI socket arg |

**Result: 9/13 primals accessible via UDS, 2 TCP/abstract only, 2 non-functional**

---

## Validation Results

### Proto-Nucleate Capabilities (Level 5 Proof)

| Capability | Primal | Result | Detail |
|-----------|--------|--------|--------|
| `stats.mean` | barraCuda | **PASS** | composition=3, local=3, diff=0.00e0 |
| `tensor.create` | barraCuda | **PASS** | shape=[2,3], dtype=f32, tensor_id returned |
| `tensor.matmul` | barraCuda | **FAIL** | API evolution: `lhs_id` required (tensor lifecycle) |
| `compute.dispatch` | toadStool | **FAIL** | `health.liveness` returns -32601 Method not found |
| `crypto.hash` | BearDog | **PASS** | BLAKE3 hash len=44, deterministic |
| `inference.complete` | Squirrel | **SKIP** | abstract socket not file-discoverable |
| `inference.embed` | Squirrel | **SKIP** | same |

**Summary: 3 PASS, 2 FAIL, 2 SKIP** (exit 1)

### guideStone Certification (v0.4.0)

| Phase | Checks | PASS | FAIL | SKIP |
|-------|--------|------|------|------|
| Phase 1: Bare Properties | 29 | 29 | 0 | 0 |
| Phase 2: Discovery + Liveness | 4 | 1 | 1 | 2 |
| Phase 3: Domain Science Parity | 8 | 5 | 1 | 2 |
| Phase 4: Additive NUCLEUS | 2 | 1 | 0 | 1 |
| **Total** | **37** | **30** | **1** | **6** |

Phase 1 CHECKSUMS refreshed (5 files updated since S211). All 29/29 PASS after refresh.

### Composition Validators

| Validator | Exit | Detail |
|-----------|------|--------|
| `validate_proto_nucleate_capabilities` | 1 | 2 PASS, 5 FAIL (3 barraCuda partial, 2 toadStool/squirrel) |
| `validate_nucleus_composition` | 1 | 6/7 primals discovered, biomeOS SKIP (graph.deploy not discoverable) |
| `validate_science_composition` | 2 | honest skip — neuralspring primal not running |
| `validate_inference_composition` | 1 | Squirrel discovery failed |
| `validate_composition_evolution` | 1 | 6 live, 3 skipped (biomeOS + neuralspring primal + health triad) |
| `validate_primal_discovery` | 1 | 7/8 discovered, biomeOS SKIP |

---

## Launcher Changes (`tools/composition_nucleus.sh`)

Expanded from 8 to 13 primals with correct dependency ordering:

| Phase | Added Primals | Wiring |
|-------|--------------|--------|
| 0 | biomeOS | cleartext bootstrap, `neural-api` subcommand, `-u FAMILY_ID` env |
| 1 | skunkBat | `server --no-uds`, `SKUNKBAT_FAMILY_ID` env |
| 2 | coralReef | `server`, `CORALREEF_FAMILY_ID` + symlink `coralreef-core-*` → `coralreef-*` |
| 2 | NestGate | `daemon --socket-only --dev`, `NESTGATE_JWT_SECRET` env |
| 2 | Squirrel | `server --socket`, Ollama + provider sockets |

Additional fixes:
- Stale socket cleanup at `cmd_start` (prevents EADDRINUSE)
- `SONGBIRD_SECURITY_PROVIDER=beardog` (name, not socket path)
- Domain aliases expanded: `shader`, `storage`, `ai`, `inference`, `orchestration`
- Stop order expanded: all 13 primals in reverse dependency order

---

## Gaps Discovered (Upstream Hand-backs)

| # | Gap | Owner | Priority | Impact |
|---|-----|-------|----------|--------|
| D1 | `tensor.matmul` requires `lhs_id` (tensor lifecycle API) | barraCuda | HIGH | proto-nucleate parity broken |
| D2 | `health.liveness` returns -32601 | toadStool | HIGH | Level 4 liveness check fails |
| D3 | Squirrel abstract socket | Squirrel | MEDIUM | inference.* SKIP in all validators |
| D4 | rhizoCrypt no UDS listener | rhizoCrypt | MEDIUM | provenance trio not reachable via UDS |
| D5 | loamSpine double-runtime panic | loamSpine | MEDIUM | permanence layer unavailable |
| D6 | `discovery.register` unknown | Songbird | LOW | registry seeding fails |
| D7 | skunkBat no UDS `--socket` | skunkBat | LOW | defense primal not composable via UDS |

---

## Files Changed

| File | Change |
|------|--------|
| `tools/composition_nucleus.sh` | Expanded 8→13 primals, stale cleanup, domain aliases |
| `validation/CHECKSUMS` | Refreshed 5 files (guidestone, validation, config, Cargo.toml) |
| `docs/PRIMAL_GAPS.md` | Level 4 status update, Gap 29 deployment findings |
| `CHANGELOG.md` | V172/S216 entry |
| `README.md` | S216/V172 version update |
| `EVOLUTION_READINESS.md` | S216/V172 session update |
| `graphs/neuralspring_deploy.toml` | S216 version, STATUS update |
| `sporeprint/validation-summary.md` | S216 live deployment metrics |
| `experiments/results/gap-status.json` | Gap 29 added, resolved 29/29 |
| `CONTROL_EXPERIMENT_STATUS.md` | S216 session update |
| `wateringHole/handoffs/NEURALSPRING_EASTGATE_DEPLOYMENT_MAY23_2026.md` | This document |

---

## Next Steps

1. **barraCuda**: Absorb tensor lifecycle API (create → matmul → release) in
   `validate_proto_nucleate_capabilities` baseline expectations
2. **toadStool**: Request `health.liveness` implementation or adapt validator to
   use `compute.dispatch` probe as liveness equivalent
3. **Squirrel**: File hand-back requesting file-based UDS socket option
4. **loamSpine**: File hand-back for double-runtime fix
5. **Multi-domain validation**: Run alongside primalSpring on southGate to test
   socket contention and capability collision scenarios

---

**Filed to:** `infra/wateringHole/handoffs/` (cross-referenced)
**Spring:** neuralSpring V172 / S216
**Gate:** southGate (confirmed)
**Mesh status:** 1/5 gates deployed (southGate)
