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

## Quality Gates

- `cargo check --workspace` — clean
- `cargo clippy --workspace` — clean (pre-existing warnings only)
- `cargo test --workspace` — 739 tests, 737 pass, 2 pre-existing env-dependent skips
- All new GPU code behind `#[cfg(feature = "barracuda")]`
- Validation scenario skip-tolerant when no GPU present
