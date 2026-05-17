# neuralSpring V167 — GPU Parity + Compute Dispatch Evolution

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**From:** neuralSpring (S211)
**To:** primalSpring, toadStool, barraCuda, coralReef teams
**Date:** 2026-05-17
**Version:** V167

---

## Summary

neuralSpring's 6-stage science pipeline is now **fully GPU-dispatchable**. 4 stages
promoted from `CpuOnly` to `GpuPreferred`, PCIe P2P bridge wired into `Dispatcher`,
typed toadStool workload submission added, and `node.compute` signal dispatch
integrated into the live pipeline executor.

## Changes

### Phase 1: 4 Stages Promoted to GpuPreferred

| Stage | GPU Method | Dispatcher API |
|-------|-----------|----------------|
| `digester_anderson` | eigensolve + disorder sweep | `eigh()`, `disorder_sweep()` |
| `isomorphic_reservoir` | eigensolve per reservoir matrix | `eigh()` × 3 |
| `wdm_ensemble_qs` | replicator dynamics | `replicator_step()` |
| `introgression_nn` | HMM Viterbi chain | `detect_introgression()` |

All 4 retain CPU fallback — `GpuPreferred` substrate routes through GPU when
present, falls back to CPU reference implementations when absent.

### Phase 2: Composition Graph Substrates

`metalForge/forge/src/graph.rs` `composition_pipeline()`: 4 `StageNode` entries
updated from `MixedSubstrate::CpuOnly` to `MixedSubstrate::GpuPreferred`.
`PipelineGraph::stages()` public accessor added.

### Phase 3: GPU Parity Validation

`s_gpu_parity` scenario (10/10) — structural checks via `include_str!` on
`dispatch.rs` source: verifies all 6 stages have GPU functions and are routed
in `dispatch_capability_gpu()`. Track: `GpuParity`, Tier: `Rust`.

### Phase 4: ToadStool Compute Dispatch Wiring

`src/ipc/toadstool.rs`:
- `ComputeWorkload` — typed workload struct (capability, data, substrate_hint)
- `WorkloadResult` — typed result struct (success, actual_substrate, output, elapsed_us)
- `compute_dispatch_workload()` — submit structured workload via IPC
- `compute_dispatch_pipeline()` — submit entire pipeline graph for remote execution

`execute_graph_live()` now prefers `node.compute` signal dispatch for GPU-tagged
stages when `CompositionContext` is available.

### Phase 5: Mixed Hardware Substrate

`Dispatcher`:
- `PcieBridge` probed at construction (sysfs IOMMU group detection on Linux)
- `pcie_p2p_available()` — check P2P DMA capability
- `pcie_transfer_cost()` — estimate cross-device transfer cost

`graphs/neuralspring_deploy.toml`: `[nodes.substrate]` section added to
`germinate_toadstool` documenting GPU stage coverage and P2P detection.

## Metrics

| Metric | Before | After |
|--------|--------|-------|
| GPU-dispatchable stages | 2/6 | **6/6** |
| Validation scenarios | 9 | **10** |
| Tests | 734 | **739** |
| Substrates: GpuOnly | 2 | 2 |
| Substrates: GpuPreferred | 0 | **4** |
| PCIe P2P detection | none | **wired** |

## Files Modified

| File | Change |
|------|--------|
| `src/nucleus_pipeline/dispatch.rs` | 4 `stage_*_gpu()` functions + wiring |
| `metalForge/forge/src/graph.rs` | 4 substrates CpuOnly→GpuPreferred + `stages()` accessor |
| `src/nucleus_pipeline/executor.rs` | `node.compute` preference in `execute_graph_live()`, `dispatch_compute_signal()` |
| `src/validation/scenarios/s_gpu_parity.rs` | New scenario |
| `src/validation/scenarios/mod.rs` | Register scenario 10 |
| `src/ipc/toadstool.rs` | Typed workload submission + pipeline dispatch |
| `src/gpu_dispatch/mod.rs` | PCIe P2P bridge integration |
| `graphs/neuralspring_deploy.toml` | Substrate documentation |
| `docs/PRIMAL_GAPS.md` | Gaps 19–23 resolved |

## Upstream Absorption Opportunities

- **toadStool**: `ComputeWorkload` struct maps directly to `toadstool.compute.submit` API.
  Consider absorbing the typed workload protocol for all delta springs.
- **barraCuda**: `Dispatcher` now wraps `PcieBridge` — could move bridge probing
  into `barracuda::unified_hardware` for ecosystem-wide P2P awareness.
- **coralReef**: mixed substrate routing can inform shader compilation priority
  (GpuPreferred stages may need JIT shader paths).
- **All springs**: `node.compute` signal dispatch pattern in `execute_graph_live()`
  is reusable for any spring that needs GPU-aware live composition.

## Primal Use & Evolution — neuralSpring's NUCLEUS Composition Patterns

### Current Primal Integration (7 IPC modules)

| Primal | Module | Capabilities Used | Signal API |
|--------|--------|-------------------|------------|
| **barraCuda** | `src/ipc/barracuda.rs` | `tensor.*`, `stats.*`, `precision.route` | `node.compute` dispatch |
| **toadStool** | `src/ipc/toadstool.rs` | `compute.dispatch`, `toadstool.validate`, `toadstool.list_workloads` | `node.compute` dispatch |
| **coralReef** | `src/ipc/coralreef.rs` | `shader.compile.wgsl` | — |
| **bearDog** | `src/ipc/beardog.rs` | `security`, `crypto`, `identity` | — |
| **squirrel** | `src/ipc/squirrel.rs` | `ai.query`, `inference.*` | — |
| **skunkBat** | `src/ipc/skunkbat.rs` | `defense`, `threat`, `metadata` | `security.audit_log` |
| **nestGate** | `src/ipc/nestgate.rs` | `content.put/get/exists` | `nest.store`, `nest.commit` |

### NUCLEUS Composition Patterns Learned

1. **Signal dispatch hierarchy**: `node.compute` for GPU workloads → `ctx.call()` for direct primal methods → local fallback. This three-tier cascade in `execute_graph_live()` gives maximum flexibility.

2. **Substrate-aware routing**: `GpuPreferred` stages try GPU first, fall back to CPU. `GpuOnly` stages require GPU or fail. This pattern (implemented in `dispatch_capability_gpu`) is reusable for any spring with mixed compute needs.

3. **Typed workload protocol**: `ComputeWorkload { capability, data, substrate_hint }` → `WorkloadResult { success, actual_substrate, output, elapsed_us }` — structured enough for provenance but flexible enough for any science domain.

4. **PCIe topology awareness**: Probing IOMMU groups at `Dispatcher` construction gives P2P transfer cost estimates without runtime overhead. Other springs with cross-device dispatch should adopt this pattern.

5. **Provenance chain**: `nest.store` (content) → `nest.commit` (session) → provenance braid. Science results get content-addressed storage with full audit trail. `store_science_result()` wraps this for any computation output.

6. **Deploy graph as substrate contract**: The `[nodes.substrate]` section in `neuralspring_deploy.toml` declares what hardware capabilities the spring advertises. biomeOS can use this for intelligent niche placement.

### biomeOS / neuralAPI Deployment Insights

- **`primal.announce`** with fallback to legacy `nucleus.register` + `capability.register` — ensures backward compatibility during ecosystem-wide signal API adoption.
- **`CompositionContext::dispatch()`** vs `ctx.call()` — dispatch for composed signals (multi-primal chains like `nest.store`), call for direct primal methods.
- **Health triad** (`health.liveness`, `health.readiness`, `health.check`) — already wired. biomeOS can probe these for niche health without domain knowledge.
- **Feature gating** — `barracuda`, `primalspring`, `guidestone`, `composed` features keep the binary lean. IPC-first (`default = []`) means the primal binary works without any optional primals.

### Recommendations for Other Springs

- **groundSpring / ludoSpring**: If you have CPU-only compute stages, adopt the `GpuPreferred` substrate pattern with `Dispatcher::gpu_or_cpu()` fallback. neuralSpring's `stage_*_gpu()` functions show the pattern.
- **airSpring / hotSpring**: The `ComputeWorkload` typed protocol is ready for adoption — submit structured workloads to toadStool instead of raw JSON.
- **All springs**: Consider `execute_graph_live()` as a template for live IPC pipeline execution. The `dispatch_compute_signal()` → `dispatch_capability_live()` → `dispatch_capability()` cascade handles offline primals gracefully.

## Quality Gates

- `cargo check --workspace` — clean
- `cargo clippy --workspace` — clean (pre-existing warnings only)
- `cargo test --workspace` — 739 tests, 737 pass, 2 pre-existing env-dependent skips
- All new GPU code behind `#[cfg(feature = "barracuda")]`
- Validation scenario skip-tolerant when no GPU present
