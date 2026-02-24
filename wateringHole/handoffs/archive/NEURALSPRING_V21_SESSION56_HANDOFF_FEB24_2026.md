# neuralSpring V21 — Session 56: ToadStool S53 Sync + Upstream Rewiring + Dispatch Expansion

**Date**: February 24, 2026
**From**: neuralSpring (ML surrogates, scholarly reproduction, isomorphic learning)
**To**: ToadStool / BarraCUDA core team
**Session**: 56
**ToadStool HEAD**: `9404fdb4`
**Previous**: [V20 Sessions 54–55](archive/NEURALSPRING_V20_SESSION55_HANDOFF_FEB24_2026.md)
**License**: AGPL-3.0-or-later

---

## Executive Summary

| Metric | V20 (S55) | V21 (S56) | Delta |
|--------|-----------|-----------|-------|
| Total checks | 1950+ | 2010+ | +60 |
| Lib tests | 459 | 478 | +19 |
| Forge tests | 26 | 30 | +4 |
| Validation binaries | 142 | 155 | +13 |
| Functions rewired to upstream | 5 (f64 ops) | 9 (f64 ops + 4 baseCamp) | +4 |
| Named tolerances | 90 | 95 | +5 |
| Clippy warnings | 0 | 0 | — |

Session 56 pulled ToadStool `9404fdb4`, rewired 4 baseCamp functions to use
upstream BarraCUDA modules absorbed from our prior handoffs (Sessions 51–53),
created 3 new comprehensive validators, and updated all documentation.

---

## Part 1: What Changed

### 1.1 ToadStool S53 Sync — Upstream Now Has Our baseCamp Math

ToadStool absorbed 4 neuralSpring-originated modules between Sessions 51–53:

| New Upstream Module | Origin | What It Does |
|---------------------|--------|-------------|
| `barracuda::linalg::graph` | Sub-04/05 handoff | Graph Laplacians, disordered Laplacians, belief propagation chains |
| `barracuda::numerical` | Sub-03 handoff | Numerical Hessian via central finite differences |
| `barracuda::ops::bio::swarm_nn` | Paper 015 handoff | Swarm neural network forward pass |
| `barracuda::ops::bio::xoshiro128ss` | Paper 011 handoff | GPU-friendly xoshiro128** PRNG |

### 1.2 Upstream Rewiring — 4 Local Functions → Thin Wrappers

| Local Function | Module | Now Delegates To |
|----------------|--------|-----------------|
| `graph_laplacian` | `agent_coordination` | `barracuda::linalg::graph::graph_laplacian` |
| `disordered_laplacian` | `agent_coordination` | `barracuda::linalg::graph::disordered_laplacian` |
| `belief_propagation_chain` | `neural_pgm` | `barracuda::linalg::graph::belief_propagation_chain` |
| `numerical_hessian` | `loss_landscape` | `barracuda::numerical::numerical_hessian` |

Public APIs preserved. All 478 lib tests pass unchanged. The redundant local
implementations were removed; only the thin wrapper (public function signature
delegating to upstream) remains.

### 1.3 Three New Validators

| Validator | Checks | What It Proves |
|-----------|--------|----------------|
| `validate_basecamp_dispatch` | 19 | All 4 baseCamp Dispatcher methods route correctly to GPU/CPU |
| `validate_barracuda_parity` | 34 | CPU↔GPU parity across linalg, stats, spectral, activations, reductions, distance, biology |
| `validate_metalforge_pcie` | 36 | PCIe bandwidth tiers, P2P vs staged, chained multi-hop, substrate selection, bridge API |

### 1.4 metalForge Enhancements

- `BandwidthTier` enum: PCIe 4.0 x16/x4, PCIe 5.0 x16, SharedMemory
- `transfer_cost_for_tier()`: tier-aware transfer cost modeling
- `chained_transfer_cost()`: multi-hop GPU→CPU→NPU cost
- `compare_transfer_paths()`: direct P2P vs CPU-staged comparison

---

## Part 2: What ToadStool Should Absorb Next

### 2.1 High Priority — Dispatcher Patterns

neuralSpring's `Dispatcher` now has 29 methods covering all science domains.
The `gpu_or_cpu` and `mixed_dispatch` patterns are well-proven and would
benefit all Springs:

```
gpu_or_cpu(workload_bytes, |device| async { /* GPU path */ }, || { /* CPU path */ })
mixed_dispatch(workload_bytes, |device| async { /* GPU */ }, || { /* CPU */ })
```

**Why ToadStool cares**: This is the core routing logic. Every Spring
reinvents it locally. Absorbing into BarraCUDA as a trait or generic
dispatch function eliminates per-Spring boilerplate.

### 2.2 Medium Priority — metalForge PCIe Cost Model

The `BandwidthTier` + `chained_transfer_cost` infrastructure is generic
and not neuralSpring-specific. It belongs in BarraCUDA's device layer:

| Component | Lines | Tested |
|-----------|-------|--------|
| `BandwidthTier` enum | 30 | 8 checks |
| `transfer_cost_for_tier` | 10 | Covered |
| `chained_transfer_cost` | 15 | 8 checks |
| `compare_transfer_paths` | 12 | 4 checks |

### 2.3 Evolving — baseCamp GPU Shader Candidates

These baseCamp primitives are validated on GPU via `Dispatcher` and are
candidates for dedicated WGSL shaders:

| Primitive | Current GPU Path | Shader Opportunity |
|-----------|-----------------|-------------------|
| `weight_to_hamiltonian` (W^T·W) | Tensor matmul | Fused symmetric matmul shader |
| `numerical_hessian` | CPU (upstream) | GPU parallel finite differences |
| `graph_laplacian` | CPU (upstream) | Sparse GPU Laplacian (degree - adjacency) |
| `belief_propagation_chain` | GEMV chain (CPU) | GPU batch GEMV pipeline |

---

## Part 3: Learnings for BarraCUDA Evolution

### 3.1 The Handoff → Absorb → Rewire Cycle Works

Session 56 proved the full loop:
1. neuralSpring implements + validates a function locally
2. Hands it off via wateringHole with documented API + tests
3. ToadStool absorbs it into BarraCUDA (generalized, optimized)
4. neuralSpring rewires to use upstream (thin wrapper)
5. All tests still pass — zero API disruption

This is the target pattern for all future function evolution.

### 3.2 f64 Is Non-Negotiable for Science

All 4 rewired functions use f64. The upstream `barracuda::linalg::graph` and
`barracuda::numerical` modules correctly use f64 throughout. This continues
the lesson from Session 53 (f32→f64 typed ops): science workloads need f64
for Hessian eigenvalues, graph Laplacian spectra, and belief propagation
normalization.

### 3.3 PCIe Bandwidth Tiers Matter for Mixed-Hardware

The PCIe 4.0 x4 path (NPU) is 4× slower than x16 (GPU). For small buffers
(<16KB) the latency dominates and CPU staging is competitive. For large
buffers (>1MB) the bandwidth dominates and P2P wins. ToadStool's device
layer should expose tier information so dispatch logic can make informed
routing decisions.

### 3.4 Tolerance Registry Scales Well

neuralSpring now has 95 named, justified tolerance constants in a centralized
registry (`src/tolerances/`). Every validation check uses a named constant,
not an inline magic number. This pattern should be adopted ecosystem-wide.

---

## Part 4: Full Validation State

### Quality Gates (all pass)

| Gate | Status |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery` | 0 warnings |
| `cargo doc --no-deps` | 0 warnings |
| `cargo test --lib` | 478 PASS |
| `cargo test -p neural-spring-forge --lib` | 30 PASS |
| `cargo run --release --bin validate_all` | 155 binaries PASS |

### Paper Stack Coverage

| Tier | Coverage | Notes |
|------|----------|-------|
| Python baselines (Py) | 25/25 (100%) | 206 checks |
| Rust native (Rs) | 25/25 (100%) | 478 lib tests |
| BarraCUDA CPU (bC) | 24/25 (96%) | Exp 005 analytical only |
| GPU Tensor (gT) | 23/25 (92%) | Exp 005 analytical, Study 005 integer |
| metalForge WGSL (mF) | 15/25 (60%) | Phase 0++ only |
| GPU Pipeline (gP) | 15/25 (60%) | Phase 0++ only |
| Cross-dispatch (xD) | 15/15 (100%) | All applicable |
| Mixed-hardware (mH) | 14/14 (100%) | baseCamp |
| Dispatch parity | 89/89 (100%) | 3 new validators |

### Open Data

All 25 papers + 5 baseCamp sub-theses use computationally generated data from
published parameters. No external datasets, no API dependencies, no proprietary
sources. Full inventory: `specs/DATA_PROVENANCE.md`.

---

## Appendix: Files Modified in Session 56

| File | Change |
|------|--------|
| `src/agent_coordination.rs` | Rewired `graph_laplacian`, `disordered_laplacian` → upstream |
| `src/neural_pgm.rs` | Rewired `belief_propagation_chain` → upstream |
| `src/loss_landscape.rs` | Rewired `numerical_hessian` → upstream |
| `src/tolerances/mod.rs` | +5 GPU dispatch tolerance constants |
| `src/tolerances/registry.rs` | Registered 5 new constants (total: 95) |
| `src/bin/validate_basecamp_dispatch.rs` | **New** — 19 checks |
| `src/bin/validate_barracuda_parity.rs` | **New** — 34 checks |
| `src/bin/validate_metalforge_pcie.rs` | **New** — 36 checks |
| `metalForge/forge/src/mixed.rs` | BandwidthTier, chained transfers |
| `Makefile` / `justfile` | `validate-dispatch` target |
| 12 documentation files | Updated counts, dates, absorption status |
