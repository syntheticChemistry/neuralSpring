# neuralSpring V140 — Cross-Spring Composition Parity Response

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Date:** May 8, 2026
**Session:** S190
**From:** neuralSpring
**To:** primalSpring, all primal teams, all spring teams
**Audit:** primalSpring Phase 60 cross-spring composition parity (v0.9.25)
**Score:** GOOD (neuralSpring)

---

## Summary

This handoff responds to primalSpring Phase 60's cross-spring composition
parity audit. neuralSpring addresses all evolution targets within its
control:

| Target | Status | Details |
|--------|--------|---------|
| barraCuda `optional = true` | **DONE** | Root + playGround, feature flag + default-enabled |
| Registry cross-sync test | **DONE** | 10+ shared methods validated against canonical 389 |
| exp094 replication | **DONE** | `experiments/exp094_neuralspring_composition_parity/` |
| Additional deploy graphs | **DONE** | 3 new (inference, spectral, math) — total 4 |
| PRIMAL_GAPS 1-4 update | **DONE** | All 4 → IMPLEMENTED |
| Documentation + handoff | **DONE** | CHANGELOG, README, gap-status.json updated |

## What Changed

### 1. barraCuda `optional = true` (Universal Target)

`barracuda`, `wgpu`, and `neural-spring-forge` are now `optional = true`
in root `Cargo.toml` with a `barracuda` feature flag (default-enabled).
GPU-centric modules gated behind `#[cfg(feature = "barracuda")]`:

- `evolved`, `gpu`, `gpu_dispatch`, `gpu_ops`, `gpu_shader_validation`
- `nautilus_bridge`, `training_monitor`, `wdm_surrogate`, `wdm_transport`
- `validation::gpu` sub-module

playGround similarly updated. metalForge/forge retains direct barracuda
dep (it's the bridge crate). Full `--no-default-features` compilation is
the next evolution step — transitive module gating needed.

### 2. Registry Cross-Sync Test (Universal Target)

New test `registry_methods_in_primalspring_canonical` in `src/config.rs`:
- `include_str!` of primalSpring's `config/capability_registry.toml`
- Validates 10+ shared methods (health.*, inference.*, compute.offload,
  identity.get, capability.list, mcp.tools.list) appear in canonical
- Documents neuralSpring-only methods as intentionally absent

### 3. exp094 Composition Parity Crate

`experiments/exp094_neuralspring_composition_parity/` — first experiment
crate in neuralSpring, following primalSpring's exp094/exp095 template.

Validates:
- **Tower Atomic**: BearDog health, Songbird discovery, crypto hash
  determinism, capability resolution (security/compute/storage)
- **Node Atomic**: barraCuda `stats.mean` parity, shader capabilities,
  toadStool compute dispatch
- **Nest Atomic**: NestGate store/retrieve roundtrip, provenance trio
- **Niche Science**: `stats.mean` parity (5-elem = 3.0, exact),
  `stats.std_dev` parity (8-elem ≈ 2.138, 1e-6), spectral analysis probe
- **Niche Inference**: probe `inference.complete`, `inference.embed`,
  `inference.models` via Squirrel IPC
- **Cross-Atomic**: hash payload → store in NestGate → retrieve → match

### 4. Deploy Graphs (3 New, 4 Total)

| Graph | Domain | Composition |
|-------|--------|-------------|
| `neuralspring_deploy.toml` | Full NUCLEUS | Existing (V137) |
| `neuralspring_inference_pipeline.toml` | Inference | Squirrel → barraCuda → inference → provenance |
| `neuralspring_spectral_analysis.toml` | Science | BearDog → eigendecomp → IPR → NestGate → provenance |
| `composition/neuralspring_math_pipeline.toml` | Compute | tensor → mean → dispatch (minimal Node Atomic) |

### 5. PRIMAL_GAPS Items 1-4 → IMPLEMENTED

Items 1 (inference surface), 2 (barraCuda IPC), 3 (coralReef IPC), 4
(toadStool IPC) now reflect primalSpring exp094 validation. All have
corresponding IPC validation in neuralSpring's exp094.

## Remaining Gaps

| Item | Status | Blocker |
|------|--------|---------|
| NestGate weight storage (gap 5) | open | storage.retrieve for model weights |
| Tower BTSP session (gap 6) | wip | BearDog BTSP + Songbird mesh |
| 18 barraCuda IPC surface gaps (gap 11) | open | upstream barraCuda JSON-RPC |
| GuideStone L4-L5 (gap 13) | partial | live NUCLEUS deployment + 18 IPC gaps |

## For primalSpring

- exp094 registered in workspace; scorecard can re-audit
- Registry cross-sync test ensures shared methods stay aligned
- barracuda optional = true satisfies universal target #1
- 4 deploy graphs vs previous 1

## For barraCuda Team

- 18 JSON-RPC surface gaps blocking L5: `eigh`, `pearson`, `chi_squared`,
  `svd`, `batch_ipr`, `spatial_payoff`, `multi_obj_fitness`, `pairwise_l2`,
  `pairwise_hamming`, `pairwise_jaccard`, `hmm_forward_log`,
  `locus_variance`, `swarm_nn_forward`, `hill_gate`, `plasma_dispersion`,
  `ode_batch`, `element_wise_reduce`, `anderson_sweep`
- These are documented in `experiments/results/gap-status.json` gap 11

## Codebase Stats

- **Tests:** 1,387 (1,234 lib + 73 forge + 80 playGround)
- **Experiments:** 134
- **Binaries:** 269
- **Deploy graphs:** 4
- **Experiment crates:** 1
- **Notebooks:** 10 (5 sporePrint + 5 paper baselines)
- **Named tolerances:** 233
- **GuideStone:** L3 (29/29 bare checks)
- **Deny clean:** yes
- **barracuda optional:** yes (S190)

---

*License: AGPL-3.0-or-later*
