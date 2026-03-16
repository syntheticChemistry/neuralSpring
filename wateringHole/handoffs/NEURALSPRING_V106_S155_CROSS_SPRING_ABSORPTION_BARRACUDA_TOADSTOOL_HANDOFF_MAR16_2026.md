# neuralSpring V106 → ToadStool/BarraCUDA Cross-Spring Absorption Handoff

**Date:** March 16, 2026
**From:** neuralSpring S155 (1301 tests, 4500+ checks, 260 binaries, 47 modules)
**To:** ToadStool/BarraCUDA team
**Authority:** wateringHole (ecoPrimals Core Standards)
**Supersedes:** V105 S154 Niche Architecture Handoff (Mar 15)
**Pins:** barraCuda v0.3.5 (`0649cd0`), toadStool S146+ (`751b3849`), coralReef Iter 49
**License:** AGPL-3.0-or-later

---

## Executive Summary

Session 155 absorbed cross-spring patterns from 4 springs (hotSpring, groundSpring,
wetSpring, airSpring) and 5 primals (coralReef, loamSpine, rhizoCrypt, sweetGrass,
biomeOS). Key deliverables:

- **`primal_names.rs`** — ecosystem-aligned module with 11 primal + 4 domain constants
- **`control/tolerances.py`** — shared Python tolerance vocabulary (80+ constants)
- **Deploy graph** — provenance trio nodes (rhizoCrypt, loamSpine, sweetGrass)
- **Zero duplicate string literals** for primal names across entire codebase

---

## Part 1: What Changed (S155)

| Artifact | Change | Pattern Source |
|----------|--------|---------------|
| `src/primal_names.rs` | NEW: 11 primal constants + `domains::` module (DAG, COMMIT, PROVENANCE, COMPUTE) | airSpring/groundSpring |
| `src/config.rs` | EVOLVED: `*_NAME_HINT` → delegates to `primal_names::*`; `PETALTONGUE_SOCKET_*` → delegates to `primal_names::PETALTONGUE` | Zero duplicate strings |
| `control/tolerances.py` | NEW: 80+ Python tolerance constants mirroring `src/tolerances/mod.rs` | wetSpring/airSpring |
| `graphs/neuralspring_deploy.toml` | EVOLVED: Phase 2b provenance trio (rhizoCrypt, loamSpine, sweetGrass) with `fallback = "skip"` | wetSpring/rhizoCrypt |

---

## Part 2: Ecosystem Evolution Reviewed

### Springs Pulled & Absorbed

| Spring | Version | Key Learnings |
|--------|---------|---------------|
| hotSpring | v0.6.31+ | **VFIO D3hot VRAM breakthrough** (Exp 062–065), `coral-glowplug` PCIe broker, HBM2 training sequences |
| groundSpring | V106 | **`primal_names.rs`** module, typed `BiomeOsError` (7 `#[non_exhaustive]` variants), 876+ tests |
| wetSpring | V120 | **`graphs/wetspring_deploy.toml`** with provenance trio, **`scripts/tolerances.py`**, typed forge errors |
| airSpring | v0.8.2+ | **`primal_names.rs`** with provenance trio domains, **`ipc/timeseries.rs`** (cross-spring data exchange `v1`), **`TOLERANCE_REGISTRY.md`** |

### Primals Pulled & Absorbed

| Primal | Version | Key Evolution |
|--------|---------|---------------|
| coralReef | Iter 49 | `coral-glowplug` crate (sovereign PCIe broker), GV100 per-runlist registers, HBM2 training, bar cartography, 1842+ tests |
| loamSpine | v0.8.8 | Edition 2024, JSON-RPC batch, proptest, enriched `capability.list`, 1123 tests |
| rhizoCrypt | v0.13.0-dev | Edition 2024, `graphs/rhizocrypt_deploy.toml`, `capability_registry.toml` (23 methods), 1177 tests |
| sweetGrass | v0.7.12 | Edition 2024, chaos tests, identity constants, 903 tests |

---

## Part 3: Primitives Consumed (100+ files, 45+ submodules)

Unchanged from V105 — full inventory in previous handoff. Key domains:

- **device**: `WgpuDevice`, driver profiles, precision routing
- **tensor/dispatch**: matmul, softmax, gelu, variance dispatch
- **stats**: 30+ functions (correlation, diversity, ET0, bootstrap)
- **ops::bio**: 16 GPU ops (evolution, HMM, stochastic, spatial)
- **nn**: `SimpleMlp` for WDM surrogates
- **spectral**: eigensolver, IPR, phase classification
- **nautilus**: brain, drift monitor, generation records

---

## Part 4: ToadStool Integration Points

neuralSpring's deploy graph now declares:

```toml
# ToadStool — GPU compute (optional)
[nodes.primal]
by_capability = "compute"
[nodes.operation.params]
features = ["gpu", "f64"]
[nodes.constraints]
fallback = "skip"
```

**Ready for**: `compute.dispatch.submit` (toadStool S155b), `shader.compile.wgsl.multi`
(multi-device compilation), `compute.hardware.auto_init_all` (multi-GPU parallel init).

neuralSpring already validates multi-GPU on RTX 4070 + TITAN V NVK (133/133 + 384/384).

---

## Part 5: coralReef Evolution — `coral-glowplug`

The new `coral-glowplug` crate is a sovereign PCIe device lifecycle broker:
- Starts at boot, binds GPUs via VFIO
- Keeps VFIO fds open (prevents D3hot VRAM loss)
- Exposes Unix socket for toadStool

neuralSpring currently uses wgpu for GPU access. When sovereign dispatch matures
(`CoralReefDevice` → `coral-glowplug` → bare metal), neuralSpring is ready:
- All dispatch goes through `barracuda::dispatch::*` (no raw wgpu in library code)
- `Fp64Strategy` precision routing already wired
- `metalForge/forge` bridges to coralReef compile path

---

## Part 6: Remaining Evolution Targets

| Target | Priority | Description |
|--------|----------|-------------|
| 6 local WGSL shaders | P2 | xoshiro128ss, swarm_nn_scores, logsumexp_reduce, stencil_cooperation, rk45_adaptive, wright_fisher_step |
| 4 CPU-only reimplementations | P3 | `primitives::softmax`, `coral_forge::layer_norm`, `coral_forge::softmax_rows`, `lenet::relu` |
| Provenance trio IPC | P1 | Wire `rhizoCrypt`/`loamSpine`/`sweetGrass` experiment tracking |
| Edition 2024 migration | P1 | Follow airSpring/loamSpine/rhizoCrypt/sweetGrass pattern |
| `sovereign_resolves_poisoning()` | P1 | hotSpring v0.6.31 pattern for f64 ML on consumer GPUs |

---

## Quality Gates (at handoff time)

| Gate | Status |
|------|--------|
| `cargo fmt --all --check` | 0 diffs |
| `cargo clippy --workspace -- -W pedantic -W nursery` | 0 warnings |
| `cargo test --lib` | 1128 passed, 0 failed |
| Production `panic!()` | 0 |
| `unsafe` blocks | 0 (`#![forbid(unsafe_code)]` on 3 crate roots) |
| Hardcoded primal names | 0 (all via `primal_names::*`) |
| Python tolerance parity | 80+ constants in `control/tolerances.py` |
| Deploy graph phases | 7 (Tower Atomic + provenance trio + ToadStool/NestGate + neuralSpring + validation + provenance record) |
