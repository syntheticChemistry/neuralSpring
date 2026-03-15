# neuralSpring → barraCuda/toadStool: Deep Debt Execution Handoff

**Date**: 2026-03-15
**From**: neuralSpring V103 / Session 152
**To**: barraCuda team, toadStool team, coralReef team
**License**: AGPL-3.0-or-later
**Covers**: Session 152 deep debt execution (tolerance centralization, capability-based discovery, shared infrastructure)
**Supersedes**: V102 S151 Deep Audit Handoff (Mar 15, 2026)
**Pins**: barraCuda v0.3.5 at `0649cd0`, toadStool S146 at `751b3849`, coralReef Iteration 33

---

## Executive Summary

Session 152 executed all findings from the S151 comprehensive ecosystem audit:

- **Tolerance centralization**: 15+ hardcoded numeric literals across 9 bench/validation
  binaries replaced with named `tolerances::` constants. New `IPR_CROSS_PYTHON` (0.005)
  added to spectral category with provenance documentation. Zero inline magic numbers
  in any code path.
- **Capability-based discovery evolved**: `PrimalClient::discover()` now uses
  `discover_by_capability("science.spectral_analysis")` with name hint fallback.
  coralReef bridge restructured to scan capability manifests before socket name-matching.
- **biomeOS path constants**: `BIOMEOS_SOCKET_SUBDIR` extracted in both `ipc_client`
  and `coralreef_bridge` — eliminates inline `"biomeos"` string literals.
- **Shared validation infrastructure**: `validate_tensor_binary()` with `BinaryTensorInputs`
  struct, `gen_test_f64()` helper. Compresses future GPU validation binaries.
- **Code quality**: 0 clippy (pedantic+nursery), 0 doc warnings, 3 pre-existing warnings
  fixed. All 1249 tests pass.

---

## Part 1: Absorption Status — Fully Lean

No change from V102. 46 upstream rewires, 219 import files, 45+ submodules consumed.
4 local-only WGSL shaders remain (HEAD_SPLIT, HEAD_CONCAT, XOSHIRO128SS, SWARM_NN_SCORES).

## Part 2: Upstream Evolution Observations

### P0 — Blocking

None. All P0 items from V102 remain resolved.

### P1 — TensorSession Adoption Gap

neuralSpring's main library still uses per-op `Tensor` dispatch. playGround demonstrates
7–45× speedup with `TensorSession::with_device` (hot dispatch vs cold). Prime candidates
for `TensorSession` fused pipelines in the main library:

| Pipeline | Ops | Current Dispatch | Benefit |
|----------|-----|-----------------|---------|
| HMM forward chain | matmul → logsumexp → accumulate (×T steps) | Per-step cold | Session reuse across T steps |
| ODE integration loop | RK4/RK45 step (×N iterations) | Per-step cold | Session reuse across N steps |
| Attention pipeline | Q/K/V projection → SDPA → output projection | Per-op cold | Single session for full pass |
| Spectral analysis | eigensolve → IPR → statistics | Per-op cold | Session reuse for batch |

**Action for barraCuda**: Document `TensorSession` patterns for multi-op pipelines
in upstream docs/examples so springs can adopt without playGround as reference.

### P2 — Benchmark Infrastructure

| Gap | Description | Upstream Action |
|-----|-------------|----------------|
| barraCuda CPU dispatch benchmark | Python baselines benchmark Rust library calls, not `barracuda::dispatch` routing | Consider `bench_barracuda_cpu_dispatch` pattern in upstream |
| Kokkos parity data | `bench_kokkos_parity.rs` scaffolded (9 ops) but uses placeholder timing data | Needs real Kokkos-CUDA runs on matching GPU hardware |
| `BARRACUDA_KOKKOS_GPU_BENCHMARK_RESULTS_MAR04_2026.md` | Referenced in code but missing | Generate or retract reference |

## Part 3: Tolerance Registry Evolution

neuralSpring's `tolerances/` module now has 80+ named constants across 22 categories.
New addition this session:

| Constant | Value | Category | Provenance |
|----------|-------|----------|------------|
| `IPR_CROSS_PYTHON` | 0.005 | spectral | `control/isomorphic_reservoir/isomorphic_reservoir_baseline.json` (seed=42) |

The `all_tolerances()` registry provides runtime introspection — primals can discover
available tolerances without hardcoded knowledge. This pattern is available for
adoption by other springs and upstream.

## Part 4: Discovery Pattern for Other Primals

neuralSpring now demonstrates the complete capability-based discovery pattern:

```rust
// Primary: discover by capability (self-knowledge only)
let socket = discover_by_capability("science.spectral_analysis", "neuralspring")?;

// coralReef: capability manifests first, socket scan fallback
if let Some(path) = discover_by_capability_manifest() { ... }
else if let Some(path) = discover_by_socket_scan() { ... }
```

**Recommendation for toadStool**: Adopt this pattern for all inter-primal discovery.
Springs should never hardcode primal names — only required capabilities.

## Part 5: Validation Metrics (unchanged from V102)

| Category | Count |
|----------|-------|
| Python baselines | 397/397 PASS |
| Rust lib tests | 1115 |
| Forge tests | 73 |
| playGround tests | 61 |
| Integration tests | 13 |
| Validation binaries | 260 |
| validate_all | 220/220 PASS |
| Line coverage | 91.66% |
| Clippy warnings | 0 (pedantic+nursery) |
| Doc warnings | 0 |

## Action Items

### barraCuda

1. Document `TensorSession` patterns for multi-op fused pipelines (P1)
2. Generate or retract `BARRACUDA_KOKKOS_GPU_BENCHMARK_RESULTS` reference (P2)
3. Consider `bench_barracuda_cpu_dispatch` pattern for CPU parity benchmarks (P2)

### toadStool

1. Adopt capability-based discovery pattern for inter-primal routing
2. Publish `TensorSession` usage examples in upstream docs

### coralReef

1. Publish capability manifest at `$XDG_RUNTIME_DIR/ecoPrimals/coralreef.json` for
   capability-first discovery (neuralSpring bridge now scans manifests before sockets)

### neuralSpring

1. Adopt `TensorSession` for HMM/ODE/attention pipelines when upstream docs available
2. Run real Kokkos-CUDA benchmarks when hardware access available
3. Monitor 4 validation binaries approaching 950 LOC for splitting triggers
