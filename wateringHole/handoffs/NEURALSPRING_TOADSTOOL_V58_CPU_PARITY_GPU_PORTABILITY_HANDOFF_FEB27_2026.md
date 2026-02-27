# neuralSpring → ToadStool Handoff V58

**Date**: February 27, 2026
**neuralSpring HEAD**: `validate_barracuda_cpu_bench` + `bench_portability_tiers`
**ToadStool HEAD**: `e96576ee` (703 f64-canonical WGSL shaders)
**Scope**: BarraCUDA CPU parity benchmarks + GPU portability proof

---

## Summary

This handoff adds two authoritative proof binaries that complete the
three-tier portability chain: Python/NumPy → BarraCUDA CPU → BarraCUDA GPU.

### 1. `validate_barracuda_cpu_bench` (25/25 PASS)

Cross-language benchmark proving BarraCUDA CPU is pure math and faster
than interpreted language (Python/NumPy) across 11 paper domains.

| Domain | Papers | Python µs | Rust µs | Speedup |
|--------|--------|-----------|---------|---------|
| HMM Forward | 016-018 | 13,138 | 84 | 157× |
| NK Fitness | 011 | 14,682 | 18 | 821× |
| Pairwise L2 | 012 | 119 | 0.4 | 315× |
| Pairwise Hamming | 017 | 430 | 35 | 13× |
| Pairwise Jaccard | 024 | 2,110 | 142 | 15× |
| Replicator Dynamics | 019 | 36,659 | 151 | 243× |
| RK4 GRN | 020 | 25,567 | 375 | 68× |
| Commutator ||[A,B]||_F | 022 | 24 | 84 | 0.3×† |
| Hill Gate 50×50 | 021 | 527 | 3 | 212× |
| Multi-Obj Fitness | 014 | 3,020 | 3 | 1,104× |
| Swarm NN Forward | 015 | 11,239 | 39 | 290× |

**Geometric mean: 83.6×**

†Commutator is the only domain where NumPy is faster because it delegates
64×64 matrix multiply to optimized BLAS; our pure Rust matmul is intentionally
naive for portability. This is expected and documented.

### 2. `bench_portability_tiers` (9/9 PASS)

CPU→GPU portability proof across 7 domains with ToadStool streaming.

| Domain | Papers | CPU µs | GPU µs | Parity |
|--------|--------|--------|--------|--------|
| HMM Forward | 016-018 | 76 | 52,794 | ✓ (1.6e-7) |
| Batch Fitness | 011-013 | 7 | 1,315 | ✓ (0.0e0) |
| Pairwise L2 | 012 | 126 | 1,472 | ✓ (7.9e-9) |
| Eigensolve+IPR | 022-023 | 9,921 | 9,921 | ✓ (GPU-resident) |
| Spatial Payoff | 019 | 12 | 1,476 | ✓ |
| Dispatcher | All | 8 | 7,302 | ✓ (9.8e-5) |
| Pairwise Hamming | 017 | 88 | 1,325 | ✓ (2.4e-8) |

GPU being slower than CPU for small workloads is expected — the overhead
is upload + readback. ToadStool's unidirectional streaming eliminates
per-op round trips for production-scale workloads.

---

## Portability Chain (Complete)

```text
Tier 1: Python/NumPy (interpreted)     — 263 open-data checks, 11 benchmarks
           ↓ 83.6× faster
Tier 2: BarraCUDA CPU (pure Rust)      — 668 lib tests, 39/39 parity (1e-10)
           ↓ same math, verified parity
Tier 3: BarraCUDA GPU (WGSL dispatch)  — 9/9 portability checks, 10/10 pure GPU
           ↓ ToadStool streaming
Tier 4: Sovereign GPU Pipeline         — 28/28 streaming spectral pipeline
```

---

## Validation Matrix

| Metric | Count |
|--------|-------|
| Total binaries | 175 |
| validate_all | 174/175 PASS |
| Library tests | 668 |
| Total checks | 3034+ |
| Python baselines | 263 checks |
| CPU↔Python parity | 39/39 (1e-10) |
| CPU→GPU portability | 9/9 |
| Pure GPU workload | 10/10 |
| Cross-system dispatch | 46/46 |
| Mixed hardware | 14/14 |

---

## Recommendations for ToadStool

1. **BLAS integration**: Consider adding optimized BLAS-backed matmul for
   small dense matrices (64×64). This closes the one domain where NumPy
   outperforms pure Rust.

2. **Streaming benchmark pattern**: The `bench_portability_tiers` pattern
   of measuring upload+compute+readback separately would be valuable as a
   ToadStool-native benchmark to demonstrate streaming advantages.

3. **Continued absorption**: All 11 benchmark domains use BarraCUDA
   primitives; continued absorption of neuralSpring-evolved algorithms
   benefits all Springs.

---

*AGPL-3.0-or-later*
