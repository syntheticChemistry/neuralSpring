# neuralSpring baseCamp — Cross-Spring Handoff (V3: Upstream Rewired)

**Date**: February 24, 2026
**neuralSpring Session**: 56 (ToadStool S53 sync + upstream rewiring)
**Audience**: hotSpring team, wetSpring team, ToadStool/BarraCUDA team, gen3 thesis committee
**Purpose**: Report upstream rewiring completion and updated cross-spring coordination
**Previous**: V2 (Session 50 — core implementation, CPU-only)

---

## Part 1: What Changed Since V2

V2 reported CPU-only implementations with 82 checks. V3 reports **GPU-promoted,
dispatch-validated, and upstream-rewired** baseCamp modules with 147 checks:

| Sub-thesis | Module | Checks (V2) | Checks (V3) | Upstream Rewired |
|:----------:|--------|:-----------:|:-----------:|:----------------:|
| nS-01: Weight Hamiltonians | `weight_spectral.rs` | 15 | 21 | — |
| nS-02: Information Flow | `information_flow.rs` | 15 | 22 | — |
| nS-03: Loss Landscapes | `loss_landscape.rs` | 19 | 27 | `numerical_hessian` |
| nS-04: Neural PGMs | `neural_pgm.rs` | 15 | 21 | `belief_propagation_chain` |
| nS-05: Multi-Agent QS | `agent_coordination.rs` | 18 | 23 | `graph_laplacian`, `disordered_laplacian` |
| **Total** | **5 modules** | **82** | **114 CPU + 14 GPU + 19 dispatch = 147** | **4 functions** |

---

## Part 2: GPU Promotion Complete

All 5 sub-theses now have Dispatcher methods routing to GPU or CPU fallback.
Three new validators confirm dispatch correctness:

| Validator | Checks | Scope |
|-----------|--------|-------|
| `validate_basecamp_gpu` | 14 | Pure GPU eigensolve + stats + L2 |
| `validate_basecamp_dispatch` | 19 | Dispatcher routing (all 4 baseCamp methods) |
| `validate_barracuda_parity` | 34 | CPU↔GPU parity across all science domains |

---

## Part 3: Upstream Rewiring Proven

4 baseCamp functions now delegate to upstream BarraCUDA (ToadStool `9404fdb4`):

| Function | Upstream | Tests Passing |
|----------|----------|:-------------:|
| `graph_laplacian` | `barracuda::linalg::graph` | 23/23 |
| `disordered_laplacian` | `barracuda::linalg::graph` | 23/23 |
| `belief_propagation_chain` | `barracuda::linalg::graph` | 21/21 |
| `numerical_hessian` | `barracuda::numerical` | 27/27 |

This proves the handoff → absorb → rewire cycle end-to-end. neuralSpring
implementations are now thin wrappers over shared BarraCUDA primitives that
benefit all Springs.

---

## Part 4: Cross-Spring Relevance

### For hotSpring

- `numerical_hessian` (now upstream) is directly usable for nuclear physics
  loss landscape analysis and HFB parameter sensitivity
- `graph_laplacian` applies to lattice QCD topology analysis
- The Dispatcher `gpu_or_cpu` pattern matches hotSpring's simulation routing needs

### For wetSpring

- `belief_propagation_chain` (now upstream) is directly usable for phylogenetic
  HMM posterior computation and amplicon denoising inference
- `graph_laplacian` applies to ecological network analysis (species interaction graphs)
- The `disordered_laplacian` models heterogeneous community structure

### For ToadStool/BarraCUDA

- All 4 rewired functions are now proven on real workloads across 478 lib tests
- Absorption targets: Dispatcher pattern (generic routing), PCIe cost model
- See companion document: `NEURALSPRING_TOADSTOOL_BARRACUDA_ABSORPTION_V21_FEB24_2026.md`

---

## Part 5: metalForge Mixed-Hardware Dispatch

Session 56 added comprehensive PCIe bandwidth tier modeling:

| Tier | Bandwidth | Latency | Use Case |
|------|-----------|---------|----------|
| PCIe 4.0 x16 | 31.5 GB/s | 2 µs | Discrete GPU |
| PCIe 4.0 x4 | 7.9 GB/s | 5 µs | NPU / M.2 |
| PCIe 5.0 x16 | 63.0 GB/s | 1.5 µs | Next-gen GPU |
| Shared memory | ~200 GB/s | 0.1 µs | Same die / SoC |

The `chained_transfer_cost` function models multi-hop paths (GPU→CPU→NPU)
and `compare_transfer_paths` enables P2P-vs-staged decision making. Validated
by 36 checks in `validate_metalforge_pcie`.

---

## Grand Total

| Metric | Value |
|--------|-------|
| Papers | 25 + 5 baseCamp sub-theses |
| Total checks | 2010+ (206 Py + 1810+ Rust/GPU) |
| Functions rewired to upstream | 9 (5 f64 ops + 4 baseCamp) |
| Lib tests | 478 PASS |
| Validation binaries | 155 PASS |
| Quality gates | fmt ✓ · clippy (pedantic+nursery) ✓ · doc ✓ |
| Open data | 100% — no proprietary sources |
