# neuralSpring → ToadStool/BarraCUDA Handoff V67 — Deep Debt + Cross-Spring Evolution

**Date**: March 1, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/BarraCUDA team
**License**: AGPL-3.0-or-later
**Covers**: Session 100 — Deep debt execution, capability-based primal discovery, cross-spring rewiring (hotSpring proxy.rs + GPU ESN via barracuda Tensors), 4 unused deps removed, +19 lib tests, zero clippy warnings
**Supersedes**: V66 (NUCLEUS Tower + nS-01 real-data pipeline)

---

## Executive Summary

- **Deep debt execution complete**: Hardcoded primal references → capability-based runtime discovery, magic timeouts → named constants, 4 unused cargo deps removed, zero clippy pedantic+nursery warnings across all 218 binaries
- **Cross-spring rewiring validated**: hotSpring `proxy.rs` diagnostics (bandwidth, condition_number, phase) absorbed into `WeightSpectralResult`; GPU ESN inference via `barracuda::tensor::Tensor` ops (matmul, add, tanh) — `validate_cross_spring_rewire` 41/41 PASS
- **Test coverage expanded**: 727 → 746 lib tests (+19): anderson_localization +10, gpu_dispatch/basecamp +8, bench refactored
- **Zero debt remaining**: 0 mocks in production, 0 unsafe, 0 bare unwrap, 0 hardcoded primal names, 0 unused deps, all files < 1000 LOC, 100% SPDX

---

## Part 1: What Changed in Session 100

### Primal Discovery Evolution

The `neuralspring_primal` binary had a hardcoded `"nestgate"` reference for `data.*` method forwarding — a wateringHole violation (primals should discover others at runtime).

**Before** (S99):
```rust
forward_to_primal("nestgate", method, params).await
```

**After** (S100):
```rust
discover_data_primal_and_forward(method, params).await
```

The new function:
1. Tries capability-based discovery via biomeOS orchestrator (`capability.resolve`)
2. Falls back to probing the socket directory for data-capable primals
3. Returns a clear error if no primal can handle the capability

**BarraCUDA relevance**: None direct — this is biomeOS/primal-layer. But it demonstrates the pattern: primals self-discover, no compile-time coupling.

### Cross-Spring Evolution: hotSpring → neuralSpring

| Origin | Feature | Target | Validation |
|--------|---------|--------|------------|
| hotSpring `proxy.rs` | `spectral_bandwidth` | `WeightSpectralResult::bandwidth` | 41/41 PASS |
| hotSpring `proxy.rs` | `spectral_condition_number` | `WeightSpectralResult::condition_number` | 41/41 PASS |
| hotSpring `proxy.rs` | `classify_phase` (Extended/Critical/Localized) | `WeightSpectralResult::phase` | 41/41 PASS |
| hotSpring `esn_v2` | GPU ESN via Tensor ops | `wdm_esn::classify_via_barracuda` | 41/41 PASS |

### BarraCUDA Tensor ESN Bridge

`classify_via_barracuda` performs ESN 2-step recurrence + readout using raw `barracuda::tensor::Tensor` ops:

```
input → Tensor::from_data
state = tanh(W_in @ input + W_res @ state + bias)
readout = W_out @ state
argmax(readout) → label
```

This bypasses `barracuda::esn_v2::ESN::predict` (which has an internal shape mismatch in `import_weights`/`predict_return_state`). Direct Tensor control gives:
- Correct GPU execution with explicit matrix shapes
- f32 GPU ↔ f64 CPU parity within `TENSOR_TRANSCENDENTAL_F32` tolerance
- Deterministic across runs

**BarraCUDA action item**: The `esn_v2::ESN` module has a shape bug — `train()` stores readout as `[reservoir_size, output_size]` but `set_readout_weights()` expects `[output_size, reservoir_size]`. neuralSpring works around this with direct Tensor ops, but hotSpring/wetSpring may hit the same issue.

### Unused Dependencies Removed

| Removed | Reason |
|---------|--------|
| `biomeos-primal-sdk` | Never imported — primal uses custom JSON-RPC |
| `uuid` | No direct use (was transitive) |
| `chrono` | No direct use (comment: "chrono eliminated") |
| `log` | No direct use (`env_logger` used directly) |

Added `tokio` features (`io-util`, `net`, `signal`, `fs`, `time`) that were previously transitive.

---

## Part 2: BarraCUDA Usage Inventory (S100)

| Category | Count |
|----------|-------|
| Import sites | 130+ across 208 files |
| Submodules used | 20+ (device, tensor, ops::bio, ops::linalg, ops::fft, stats, spectral, dispatch, staging, pipeline, shaders, special, nn, sample) |
| Function rewires | 44 upstream |
| WGSL shaders absorbed | 21 + 15 coralForge df64 |
| CPU→GPU dispatch ops | 47 (~97% of production math) |
| Lib tests exercising barracuda | 746 |
| Validation binaries | 218 |

### Cross-Spring Provenance (what flows where)

| Origin | Contribution | Used By |
|--------|-------------|---------|
| **hotSpring** `proxy.rs` | bandwidth, condition_number, phase classification | neuralSpring `weight_spectral.rs` |
| **hotSpring** `esn_v2` | GPU ESN via Tensor ops | neuralSpring `wdm_esn.rs` |
| **hotSpring** df64 | f64 precision shaders (naga downcast) | neuralSpring coralForge (15 WGSL shaders) |
| **hotSpring** ValidationHarness | Validation pattern | barracuda (absorbed upstream) |
| **wetSpring** DiversityFusionGpu | Fused Shannon+Simpson+Pielou | neuralSpring `bench_cross_spring_evolution` |
| **wetSpring** bio stats | shannon, simpson, chao1, bray_curtis | neuralSpring via barracuda::stats |
| **wetSpring** HMM f64 | Forward algorithm on GPU | neuralSpring `validate_gpu_hmm_forward` |
| **neuralSpring** batch fitness | GPU batch fitness eval | barracuda::ops::bio (absorbed) |
| **neuralSpring** pairwise ops | L2, Jaccard, Hamming | barracuda::ops::bio (absorbed) |
| **neuralSpring** weight_spectral | ESD, level_spacing, MP bounds | barracuda::stats, barracuda::spectral |
| **neuralSpring** SimpleMlp | CPU MLP inference | barracuda::nn (absorbed) |

---

## Part 3: Absorption Opportunities for ToadStool

### Priority 1: esn_v2 Shape Bug Fix

The `esn_v2::ESN` module's `train()` stores readout weights transposed relative to what `set_readout_weights()`/`predict_return_state()` expect. neuralSpring works around this, but other springs will hit it.

### Priority 2: WeightSpectralResult Extensions

neuralSpring now computes `bandwidth`, `condition_number`, and `phase` from eigenvalues. These are zero-dependency scalar functions that could be promoted to `barracuda::spectral`:

```rust
pub fn spectral_bandwidth(eigenvalues: &[f64]) -> f64
pub fn spectral_condition_number(eigenvalues: &[f64]) -> f64
pub fn classify_phase(lsr: f64) -> SpectralPhase
```

### Priority 3: anderson_localization Coverage

neuralSpring now has 16 unit tests for Anderson localization (up from 5). These test patterns could inform barracuda's eigendecomposition test suite — especially the disorder sweep monotonicity and two-particle symmetry checks.

---

## Part 4: Quality Metrics

| Metric | Value |
|--------|-------|
| Lib tests | **746** |
| Integration tests | 9 |
| Forge tests | 43 |
| Validation binaries | **218** |
| validate_all | 200/200 |
| Clippy warnings | **0** (pedantic + nursery) |
| Unsafe code | **0** (forbidden) |
| Bare unwrap | **0** in library |
| Mocks in production | **0** |
| Files > 1000 LOC | **0** |
| SPDX headers | **100%** |
| Named tolerances | 139+ |
| External deps | 9 (all pure Rust) |
| Unused deps | **0** (4 removed in S100) |

---

## Part 5: Remaining Work

| Item | Status | Notes |
|------|--------|-------|
| esn_v2 shape bug workaround | **Active** | neuralSpring uses direct Tensor ops |
| nS-01 Paper A real data | **Pipeline ready** | Awaiting pretrained model downloads |
| WDM nW-05 GPU ESN | **Validated** | classify_via_barracuda 41/41 PASS |
| All 17 shortcomings | **RESOLVED** | S-01 through S-17 |
| wright_fisher WGSL parse | **Pre-existing** | 2 validators skip (naga parse issue) |

---

*neuralSpring Session 100 — Deep debt execution + cross-spring evolution rewiring.*
*V67 supersedes V66. Archive V66.*
