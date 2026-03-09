# neuralSpring → ToadStool/BarraCUDA V91 Handoff

**Date**: March 9, 2026
**From**: neuralSpring (Session 133)
**To**: ToadStool/BarraCUDA/coralReef teams
**License**: AGPL-3.0-or-later
**Supersedes**: V90 (S132), V88 (S130)
**ToadStool pin**: S130+ HEAD (`bfe7977b`)
**BarraCUDA pin**: v0.3.3 at `a898dee`
**coralReef pin**: Iteration 10 at `d29a734`

---

## Executive Summary

neuralSpring Session 133 completes Phases 5–7 of the buildout plan: metalForge PCIe P2P wiring, biomeOS pipeline DAG coordination, and petalTongue full streaming integration. All work is locally evolved in absorption-friendly layout for ToadStool and BarraCUDA to absorb.

### Key Metrics

| Metric | Value |
|--------|-------|
| validate_all | **220/220** PASS (218 standard + 2 feature-gated) |
| Lib tests | 957 + 71 forge + 9 integration |
| Validation binaries | 246 |
| Clippy (pedantic+nursery) | 0 warnings |
| Doc warnings | 0 |
| Unsafe code | `#![forbid(unsafe_code)]` |
| Files ≤1000 LOC | ALL |
| Upstream rewires | 46 functions + 6 shader sources |
| BarraCUDA submodules | 45+ |
| BarraCUDA import sites | 128+ |
| New forge modules | 3 (`graph.rs`, `pcie_bridge` extensions, `mixed` extensions) |
| New visualization modules | 1 (`stream.rs`) |

---

## Part 1: metalForge PCIe P2P — Absorption Target for `barracuda::unified_hardware`

### What neuralSpring Built

| Component | Location | Purpose |
|-----------|----------|---------|
| `TransferStrategy` enum | `metalForge/forge/src/pcie_bridge.rs` | P2P vs CPU-staged decision |
| `PcieBridge::transfer_buffer_strategy()` | same | IOMMU-based strategy selection |
| `MixedSubstrate::NpuToGpuP2P` | `metalForge/forge/src/mixed.rs` | Explicit P2P bypass variant |
| `mixed_substrate_p2p()` | same | P2P-aware substrate routing |
| `detect_p2p()` | same | Linux sysfs IOMMU group probe |

### Absorption Guidance

ToadStool's `barracuda::unified_hardware` already has `BandwidthTier`. neuralSpring's additions fit naturally:

1. **`TransferStrategy`** → `barracuda::unified_hardware::transfer::TransferStrategy`
2. **`PcieBridge`** → `barracuda::unified_hardware::PcieBridge` (generalizes beyond neuralSpring)
3. **`detect_p2p()`** → move to `barracuda::unified_hardware::probe` (shared by all springs)
4. **`NpuToGpuP2P`** → add to `barracuda::unified_hardware::SubstrateType` (currently 8 variants)

### Validation

- `validate_nucleus_compute_dispatch`: 43/43 PASS (was 39 — +4 PCIe transfer strategy checks)
- `validate_nucleus_pcie_mixed_pipeline`: 38/38 PASS
- `validate_nucleus_tower`: 22/22 PASS (feature-gated `--features primal`)
- All three now in `validate_all` (Tower via `FEATURE_BINARIES`)

---

## Part 2: biomeOS Pipeline DAG — Absorption Target for `barracuda::pipeline::graph`

### What neuralSpring Built

| Component | Location | Purpose |
|-----------|----------|---------|
| `StageNode` | `metalForge/forge/src/graph.rs` | Capability-addressed pipeline stage |
| `PipelineGraph` | same | DAG with Kahn's topological sort |
| `PipelineExecution` | same | Per-stage result tracking + throughput |
| `StageOutput` | same | Typed output (Scalar/Vector/Map/Empty) |
| `spectral_pipeline()` | same | Diamond DAG: eigensolve → IPR/LSR → entropy |
| `population_genetics_pipeline()` | same | Linear: AF → π → FST → entropy |
| `folding_pipeline()` | same | Linear: EvoFormer → Structure → health |

### Key Design Decisions

- **Capability-addressed**: stages reference `"science.eigensolve"` not specific primals — biomeOS resolves at runtime
- **Substrate-aware**: each stage has a `MixedSubstrate` preference for GPU/CPU/NPU routing
- **Structural validation**: cycle detection, duplicate ID rejection, dangling edge detection
- **No runtime dependency**: pure graph algorithms, no wgpu/tokio/serde required

### Absorption Guidance

This fits directly into ToadStool's orchestration layer:

1. **`PipelineGraph`** → `barracuda::pipeline::Graph` or `toadstool::orchestration::PipelineGraph`
2. **`StageNode`** → generalize to all primals (not just neuralSpring capabilities)
3. **`spectral_pipeline()`** → move to spring-specific config, keep graph engine in ToadStool
4. **`PipelineExecution`** → integrate with ToadStool's `StreamingDispatch` for live monitoring

### Validation

- `validate_biomeos_graph`: 32/32 PASS (DAG construction, topo sort, cycle detection, execution tracking)
- `validate_biomeos_spectral`: 29/29 PASS (feature-gated, live JSON-RPC pipeline)
- 15 unit tests in `metalForge/forge/src/graph.rs`

---

## Part 3: petalTongue StreamSession — Absorption Target for `petaltongue-client`

### What neuralSpring Built

| Component | Location | Purpose |
|-----------|----------|---------|
| `StreamSession` | `src/visualization/stream.rs` | Session lifecycle with backpressure |
| `SessionStats` | same | Messages/sec, bytes, error rate, uptime |
| `push_replace()` | `src/visualization/ipc_push.rs` | Replace binding data in place |
| `query_capabilities()` | same | Query renderer capabilities |
| 64KB IPC buffer | same | Parity with healthSpring |

### Key Design Decisions

- **Backpressure**: `error_rate() > 0.1` triggers `backpressure_active()` — callers should throttle
- **Atomic stats**: `AtomicU64` counters — safe for concurrent streaming
- **No compile-time petalTongue dependency**: pure JSON-RPC over Unix socket
- **Domain**: `"neural"` (electric blue/magenta palette)

### Absorption Guidance

healthSpring and wetSpring have their own `PetalTonguePushClient` copies. ToadStool should absorb into a shared crate:

1. **`StreamSession`** → `petaltongue-client::StreamSession` (shared by all springs)
2. **`PetalTonguePushClient`** → `petaltongue-client::PushClient` (deduplicate from 3 springs)
3. **IPC discovery** → standardize the 3-tier socket resolution across all springs
4. **Backpressure** → promote to ecosystem convention (all springs should respect error rates)

### Validation

- `validate_petaltongue_scenarios`: 31/31 PASS (scenario builders, streaming, mock IPC roundtrips)
- 46 visualization unit tests (render, append, gauge, replace, discovery, error handling)
- 5 scenario builders + `full_study()` combiner confirmed

---

## Part 4: BarraCUDA Usage Review — Current State

### Modules Used (12+ top-level)

| Module | Usage | Absorption Status |
|--------|-------|-------------------|
| `barracuda::device` | `WgpuDevice`, `GpuDriverProfile`, `Fp64Strategy`, `PrecisionRoutingAdvice` | Fully absorbed |
| `barracuda::dispatch` | `dispatch_for`, `softmax_dispatch`, `gelu_dispatch`, `matmul_dispatch` | Fully absorbed |
| `barracuda::ops::bio` | `HmmBatchForwardF64`, `hmm_viterbi`, `HillGateGpu`, `SwarmNnParams` | Fully absorbed |
| `barracuda::ops::rk45_adaptive` | `Rk45AdaptiveGpu`, `Rk45DispatchArgs` | Fully absorbed |
| `barracuda::ops::mha` | `MultiHeadAttention` | Fully absorbed |
| `barracuda::spectral` | `BatchIprGpu`, `level_spacing_ratio`, `spectral_bandwidth` | Fully absorbed |
| `barracuda::stats` | `shannon`, `pearson_correlation`, `bootstrap_ci`, hydrology | Fully absorbed |
| `barracuda::tensor` | `Tensor`, `from_data` | Fully absorbed |
| `barracuda::shaders::provenance` | `cross_spring_shaders`, `evolution_report` | Fully absorbed |
| `barracuda::unified_hardware` | `BandwidthTier` | Fully absorbed |
| `barracuda::linalg` | `eigh_f64` | Fully absorbed |
| `barracuda::error` | `BarracudaError`, `Result` | Fully absorbed |

### Shortcomings

ALL 17 shortcomings (S-01 through S-17) are resolved upstream. neuralSpring has zero local workarounds.

### Shader Evolution

- 13/17 metalForge shaders absorbed upstream (8 identical, 5 generalized)
- 4 still local: `head_split`, `head_concat`, `xoshiro128ss`, `swarm_nn_scores`
- All local shaders use `fn main` entry point (naga-safe)

---

## Part 5: New Work for ToadStool/BarraCUDA to Absorb

### Priority 1: Pipeline Graph Engine

The DAG-based pipeline execution in `graph.rs` is spring-agnostic and ready for absorption. Key value: capability-addressed stages enable dynamic primal routing via biomeOS.

**Action**: Absorb `PipelineGraph`, `StageNode`, `PipelineExecution` into ToadStool's orchestration layer. Remove spring-specific pipeline definitions (those stay in neuralSpring).

### Priority 2: PCIe Transfer Strategy

The `TransferStrategy` + `detect_p2p()` combination provides runtime P2P capability detection that benefits all springs with mixed hardware.

**Action**: Absorb into `barracuda::unified_hardware`. The sysfs probe is Linux-specific but conservative (returns `false` on non-Linux).

### Priority 3: petalTongue Client Dedup

Three springs have independent `PetalTonguePushClient` implementations. The `StreamSession` pattern (backpressure + stats) should be shared.

**Action**: Create `petaltongue-client` crate in ecoPrimals. All springs lean on it.

### Priority 4: Feature-Gated Validation Pattern

neuralSpring's `FEATURE_BINARIES` pattern in `validate_all.rs` enables feature-gated validators to run alongside standard ones. This pattern should be adopted by hotSpring and wetSpring.

---

## Part 6: Paper Controls Confirmation

All 26 Phase 0++ papers + baseCamp + WDM + coralForge use open data and open systems:

| Data Source | Papers | Status |
|-------------|--------|--------|
| SRA (NCBI) | 001–005, 014–015 | Public accession numbers documented |
| Zenodo | 006–009, 019–021 | DOIs in validation binaries |
| EPA (STORET) | 001 (FAO-56) | Public water quality data |
| PDB | nF-01/02/03 | Public protein structures |
| Synthetic | 010–013, 016–018, 022–025 | Fully seeded, reproducible |
| Public ML datasets | Paper 026 (Chuna LSTM) | UCI Blood Glucose |

Zero proprietary or paywalled sources. AGPL-3.0-or-later throughout.

---

## Appendix: Validation Tiers (Current State)

| Tier | Count | Status |
|------|-------|--------|
| Python control (Py) | 331/331 | **COMPLETE** |
| Rust CPU (Rs) | 957 lib + 9 integration | **COMPLETE** |
| BarraCUDA CPU (bC) | 96% of papers | **ALL GREEN** |
| BarraCUDA GPU Tensor (gT) | 92% of papers | **ALL GREEN** |
| metalForge WGSL (mF) | 100% applicable | **ALL PASS** |
| GPU Pipeline (gP) | 100% applicable | **ALL PASS** |
| Cross-dispatch (xD) | 53/53 parity | **ALL PASS** |
| Mixed-hardware (mH) | 43+38+22 NUCLEUS | **ALL PASS** |
| Multi-GPU (mG) | RTX 4070 + TITAN V NVK | **ALL PASS** |
| Forge tests | 71/71 | **ALL PASS** |
| validate_all | 220/220 | **ALL PASS** |
