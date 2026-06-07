# neuralSpring — Degradation Behavior

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

> Per primalSpring Wave 20 PM (lithoSpore R1): documents what happens when
> each dependency is unreachable. Science never fails due to primal absence.

**Date**: 2026-06-06 | **Session**: S225 | **Pattern**: `has_capability()` before `call()`

---

## Principle

**Science logic is never gated behind primal availability.** IPC enriches
(provenance, GPU acceleration, inference routing) but never blocks. Every
IPC call path has a defined fallback.

---

## Per-Primal Degradation

| Primal | Unreachable Behavior | Fallback | Impact |
|--------|---------------------|----------|--------|
| **barraCuda** | GPU dispatch unavailable | CPU reference math (`gpu_or_cpu` fallback in `Dispatcher`) | Slower execution, same results |
| **toadStool** | `compute.dispatch` fails | Local `Dispatcher` executes workload directly | No remote compute, local fallback |
| **coralReef** | Shader compilation unavailable | Pre-compiled shaders or CPU fallback | No JIT shaders, GPU ops use existing pipelines |
| **bearDog** | Security/identity unavailable | Primal starts without BTSP session | Reduced security posture, science continues |
| **squirrel** | Inference unavailable | `has_squirrel()` returns false; inference methods return `Err` | No AI inference, science unaffected |
| **skunkBat** | Audit log forwarding fails | Log warning, continue | No audit trail, science unaffected |
| **nestGate** | Content storage unavailable | Weight load from local filesystem; `nest.store` returns `Err` | No content-addressed storage, local files used |
| **songBird** | Discovery unavailable | `CapabilityRouter` returns no socket; fallback to env var paths | Degraded discovery, manual socket paths work |
| **rhizoCrypt** | DAG session fails | Provenance not recorded; `store_science_result()` returns `Ok` with `recorded: false` | No DAG provenance |
| **loamSpine** | Spine entry fails | Braid created without permanence anchor | No spine ledger entry |
| **sweetGrass** | Braid creation fails | DAG + spine recorded without attribution braid | Partial provenance (valid) |

## GPU-Dependent Paths

| Path | No GPU Available | GPU Present but Op Fails |
|------|-----------------|-------------------------|
| `Dispatcher::gpu_or_cpu()` | Returns CPU result (transparent) | Logs warning, returns CPU result |
| `dispatch_capability_gpu()` | Falls back to `dispatch_capability()` (CPU) | Falls back to CPU |
| `execute_graph_gpu()` | All stages run CPU-only; `substrate_used = "CPU"` | Failed stage falls back to CPU; `substrate_used = "Mixed"` |
| `execute_graph_live()` → `node.compute` | `dispatch_compute_signal()` returns `None` → `ctx.call()` → local | Falls back through cascade |
| `GpuPreferred` substrate | CPU execution, `actual_substrate = CpuOnly` | GPU execution, `actual_substrate = GpuPreferred` |
| `GpuOnly` substrate | CPU execution (fallback), `actual_substrate = CpuOnly` | GPU execution |

## Provenance Trio Degradation

Per `PROVENANCE_TRIO_INTEGRATION_GUIDE.md` — partial completion is valid:

| State | Meaning | Action |
|-------|---------|--------|
| Full (DAG + spine + braid) | Complete provenance | Ideal state |
| DAG + spine (no braid) | Attribution without permanence | Record braid ID for retry |
| DAG only (no spine/braid) | Merkle root covers computation | Flag for backfill |
| None (all trio unreachable) | Science runs without provenance | `Ok` with empty `primals_reached` |

**Rules**:
- `provenance_dispatch::store_science_result()` returns `Ok` even on partial trio
- `provenance_dispatch::commit_session_signal()` returns `Ok` even if spine/braid fail
- Domain logic **never** returns `Err` due to provenance failure
- Deploy graph nodes use `fallback = "skip"` for all trio primals

## IPC Error Classification

neuralSpring uses `primalspring::composition::is_skip_error()` to classify:
- **Skip errors** (primal not deployed): treated as `None` → fallback
- **Real errors** (primal deployed but call failed): treated as `Err` → logged, CPU fallback
- **Protocol errors** (HTTP-on-UDS, wrong format): treated as skip

This classification ensures that missing primals never cause test failures
or science errors — only enrichment is lost.
