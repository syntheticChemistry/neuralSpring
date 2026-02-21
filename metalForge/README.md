# metalForge — ML Dispatch & Hardware Characterization

**Parent**: ecoPrimals/neuralSpring
**License**: AGPL-3.0-or-later

---

## Philosophy

Every ML inference result in ecoPrimals originates in shader math (WGSL).
ToadStool dispatches that math to whatever silicon is available. metalForge
exists to **characterize how hardware affects ML dispatch** — where per-op
overhead dominates, where memory bandwidth saturates, and where cache tiling
unlocks real throughput.

Following the hotSpring pattern: we don't just benchmark. We probe
dispatch latency, measure workgroup occupancy, profile shared memory
pressure, and find the crossover points where CPU software rasterization
beats GPU for small tensors.

---

## Hardware Inventory

| Substrate | Device | Driver | Key Characteristic |
|-----------|--------|--------|-------------------|
| **GPU** | NVIDIA RTX 4070 (12 GB) | Vulkan (proprietary) | 5888 CUDA cores, 186 GB/s bandwidth |
| **CPU** | llvmpipe (software) | Mesa | Single-threaded WGSL, useful for CI and correctness |

---

## Key Findings

### 1. Per-Op Dispatch Overhead (S-01)

The single biggest performance bottleneck in BarraCUDA ML inference is
**per-op command submission**. Each `Tensor` operation creates its own
`CommandEncoder`, dispatches one compute pass, and submits individually.

| Pipeline | Per-Op | Fused | Speedup |
|----------|--------|-------|---------|
| MLP (4→64→64→10) | 4.5 ms | **97 µs** | **46×** |
| Transformer (d=32, h=4) | 12.8 ms | **164 µs** | **78×** |

**Root cause**: `queue.submit()` on Vulkan has ~500 µs fixed overhead.
An MLP with 9 ops pays 9 × 500 µs = 4.5 ms in submission alone.

**Local fix** (fossilized): `evolved::fused_pipeline` — single encoder, single submit.
Now replaced by `TensorSession` (S-01/S-11 absorbed).

### 2. Matmul Cache Tiling (S-02)

BarraCUDA's `matmul.wgsl` reads K elements from global memory per output
element. No shared memory, no tiling, no vectorization.

| Shader | Backend | Technique | Result |
|--------|---------|-----------|--------|
| `matmul.wgsl` (stock) | CPU | Global reads | 3× slower than NumPy |
| `matmul_cpu_tiled.wgsl` | CPU | 32×32, vec4, 8×4 µkernel | **1.1× faster than NumPy** |
| `matmul_gpu_evolved.wgsl` | GPU | 16×16, 2×2 µkernel, double-buf | **104× faster than NumPy** |

**Key insight**: CPU tiling matters even for software rasterizers because
it dramatically reduces global memory traffic via shared memory reuse.

### 3. GPU Crossover Points

| Scale (FLOPs) | Python | CPU | GPU | Winner |
|---------------|--------|-----|-----|--------|
| 3.1M (MLP large) | 3.0 ms | 2.7 ms | 178 µs | GPU |
| 103M (TF medium) | 59 ms | 15.1 ms | 566 µs | GPU |
| 6.6B (TF xlarge) | 232 ms | 1.42 s | 17.8 ms | GPU |

**Crossover**: GPU always wins when using fused dispatch. CPU beats Python
only at ≥3M FLOPs. Below ~1M FLOPs, Python's NumPy (calling OpenBLAS)
wins due to zero dispatch overhead.

### 4. BarraCUDA CPU Math Precision

| Primitive | Test Case | Precision | Notes |
|-----------|-----------|-----------|-------|
| `rk45_solve` | 4D ODE, t=20 | **machine ε** | Matches hand-rolled RK4 exactly |
| `solve_f64` | Dense linear system | **machine ε** | LU-based, excellent |
| `chi_squared_sf` | LRT p-value | **1e-10** | Matches scipy reference |
| `eigh_f64` | n=8 symmetric | **~1e-3** | Jacobi eigensolver accuracy gap |
| `eigh_f64` | n=16 symmetric | **~0.1** | Degrades with dimension |
| `stats::variance` | Population stats | **machine ε** | Two-pass algorithm, stable |

**Critical gap**: `eigh_f64` uses Jacobi iteration, which has O(n³) cost
and ~1e-3 relative error for n≥8. LAPACK achieves 1e-14. This affects
spectral analysis (Paper 022) and Anderson localization (Paper 023).
ToadStool's NAK eigensolver (GPU) may resolve this.

---

## Evolution Targets

| Target | Previous | Current | Status |
|--------|----------|---------|--------|
| Fused dispatch | ~~`evolved::fused_pipeline`~~ (fossilized) | ToadStool `TensorSession` | **ABSORBED** (S-01/S-11) |
| Tiled matmul | ~~local WGSL shaders~~ (fossilized) | ToadStool `KernelRouter` | **ABSORBED** (S-02) |
| Layer norm/log-softmax | ~~local GPU-resident ops~~ (fossilized) | Native `Tensor::*_wgsl()` | **ABSORBED** (S-08/S-09) |
| MHA z-dispatch | `evolved::mha` (active, S-03b) | Native `ops::mha` hangs | **PARTIAL** (S-03 → S-03b) |
| Eigensolver | Jacobi (1e-3 at n=8) | ToadStool NAK eigensolve | S-12 outstanding |
| Batched fitness | CPU loop | `batch_fitness_eval.wgsl` | **Validated** (20/20) |
| HMM chain | Sequential matmul | `hmm_forward_log.wgsl` | **Validated** (13/13) |
| Multi-system ODE | Sequential rk45 | `rk4_parallel.wgsl` | **Validated** (8/8) |
| Pairwise Jaccard | CPU O(N²×G) | `pairwise_jaccard.wgsl` | **Validated** (6/6) |
| Locus variance | CPU loop per locus | `locus_variance.wgsl` | **Validated** (7/7) |
| Spatial payoff | CPU stencil loop | `spatial_payoff.wgsl` | **Validated** (5/5) |
| Batch IPR | CPU eigenvector scan | `batch_ipr.wgsl` | **Validated** (5/5) |
| Pairwise Hamming | CPU O(N²×L) | `pairwise_hamming.wgsl` | **Validated** (5/5) |
| Pairwise L2 | CPU O(N²×D) | `pairwise_l2.wgsl` | **Validated** (15/15) |
| Multi-obj fitness | CPU per-chunk stats | `multi_obj_fitness.wgsl` | **Validated** (6/6) |
| Swarm NN forward | CPU per-controller | `swarm_nn_forward.wgsl` | **Validated** (9/9) |
| Hill AND gate | CPU elementwise | `hill_gate.wgsl` | **Validated** (9/9) |
| Cross-dispatch | Manual routing | `barracuda::dispatch` | **Validated** (16/16) |

---

## Directory Structure

```
metalForge/
├── README.md              ← this file
├── CROSS_SYSTEM_DISPATCH.md ← GPU→CPU→NPU dispatch strategy
├── gpu/
│   └── nvidia/
│       ├── DISPATCH.md    ← RTX 4070 dispatch latency measurements
│       └── HARDWARE.md    ← RTX 4070 hardware characterization
├── ABSORPTION_MANIFEST.md ← comprehensive absorption inventory (hotSpring pattern)
├── shaders/               ← Phase 3+4c+4e WGSL evolution (16 shaders, absorption candidates)
│   ├── ABSORPTION_TRACKER.md  ← lifecycle tracker for all shaders
│   ├── hmm_forward_log.wgsl   ← HMM forward pass, log-domain (Papers 016–018)
│   ├── batch_fitness_eval.wgsl ← Parallel population fitness (Papers 011–015)
│   ├── rk4_parallel.wgsl      ← Multi-system ODE integration (Papers 020–021)
│   ├── mean_reduce.wgsl       ← Scalar mean reduction (chained after fitness)
│   ├── pairwise_jaccard.wgsl  ← Pairwise Jaccard distance (Paper 024)
│   ├── locus_variance.wgsl    ← Per-locus AF variance (Paper 025)
│   ├── spatial_payoff.wgsl    ← PD spatial stencil (Paper 019)
│   ├── batch_ipr.wgsl         ← Batch IPR computation (Papers 022–023)
│   ├── pairwise_hamming.wgsl  ← Pairwise Hamming distance (Paper 017)
│   ├── pairwise_l2.wgsl      ← Pairwise L2 distance (Paper 012 — MODES)
│   ├── multi_obj_fitness.wgsl ← Multi-objective fitness (Paper 014 — Directed Evo)
│   ├── swarm_nn_forward.wgsl ← Batch NN forward pass (Paper 015 — Swarm Robotics)
│   ├── hill_gate.wgsl        ← Two-input Hill AND gate (Paper 021 — Signal)
│   ├── head_split.wgsl       ← GPU head split for MHA: [B,S,D] → [B,H,S,D/H]
│   ├── head_concat.wgsl      ← GPU head concat for MHA: [B,H,S,D/H] → [B,S,D]
│   └── xoshiro128ss.wgsl     ← GPU-parallel PRNG, Xoshiro128** (all stochastic)
└── fossils/               ← Absorbed evolved code (see FOSSIL_RECORD.md)
    ├── evolved_s01_s11/   ← Deprecated workaround modules (~2,864 LOC)
    └── bench/             ← Deprecated fused benchmarks (~1,127 LOC)
```

---

## Shader Evolution Workflow

Following the hotSpring pattern, metalForge now includes a `shaders/`
directory for WGSL evolution. The lifecycle is:

1. **Evolve**: Write WGSL targeting a specific paper workload
2. **Orchestrate**: Rust dispatch code in `src/evolved/` using raw `wgpu::Buffer`
3. **Validate**: `ValidationHarness` binary against Python controls
4. **Benchmark**: Add to `gpu/nvidia/` dispatch characterization
5. **Handoff**: Document in `wateringHole/handoffs/` for ToadStool
6. **Retire**: When ToadStool absorbs, remove local code

### Active Shader Evolutions (Phase 3c + 4c + 4e — 16 shaders, 123/123 PASS)

| Shader | Target Workload | GPU Strategy | Rust Export |
|--------|----------------|--------------|-------------|
| `hmm_forward_log.wgsl` | HMM forward chain (016–018) | Log-domain logsumexp, one thread/state | `hmm::WGSL_HMM_FORWARD_LOG` |
| `batch_fitness_eval.wgsl` | EA population eval (011–015) | One thread/individual, dot-product fitness | `evolved::WGSL_BATCH_FITNESS_EVAL` |
| `rk4_parallel.wgsl` | ODE integration (020–021) | One thread/system, full RK4 stepping | `evolved::WGSL_RK4_PARALLEL` |
| `mean_reduce.wgsl` | Fitness aggregation | Workgroup reduction → scalar | `evolved::WGSL_MEAN_REDUCE` |
| `pairwise_jaccard.wgsl` | Pangenome distance (024) | One thread/pair, O(G) per pair | `pangenome_selection::WGSL_PAIRWISE_JACCARD` |
| `locus_variance.wgsl` | Locus AF variance (025) | One thread/locus, population variance | `meta_population::WGSL_LOCUS_VARIANCE` |
| `spatial_payoff.wgsl` | PD stencil (019) | One thread/cell, Moore neighborhood | `game_theory::WGSL_SPATIAL_PAYOFF` |
| `batch_ipr.wgsl` | IPR computation (022–023) | One thread/eigenvector, sum of 4th powers | `anderson_localization::WGSL_BATCH_IPR` |
| `pairwise_hamming.wgsl` | Hamming distance (017) | One thread/pair, count diffs | `sate_alignment::WGSL_PAIRWISE_HAMMING` |
| `head_split.wgsl` | MHA head split | [B,S,D] → [B,H,S,D/H] data movement | `evolved::WGSL_HEAD_SPLIT` |
| `head_concat.wgsl` | MHA head concat | [B,H,S,D/H] → [B,S,D] data movement | `evolved::WGSL_HEAD_CONCAT` |
| `xoshiro128ss.wgsl` | GPU PRNG (all stochastic) | One thread/stream, 4×u32 state | `rng::WGSL_XOSHIRO128SS` |
| `pairwise_l2.wgsl` | MODES pairwise L2 (012) | One thread/pair, Euclidean in feature space | `modes::WGSL_PAIRWISE_L2` |
| `multi_obj_fitness.wgsl` | Multi-obj fitness (014) | One thread/(individual,objective), mean+std | `directed_evolution::WGSL_MULTI_OBJ_FITNESS` |
| `swarm_nn_forward.wgsl` | Swarm NN forward (015) | One thread/(controller,eval), 1→4→5 MLP | `swarm_robotics::WGSL_SWARM_NN_FORWARD` |
| `hill_gate.wgsl` | Hill AND gate (021) | One thread/(cdg,ai), two-input Hill | `signal_integration::WGSL_HILL_GATE` |

Following the hotSpring pattern, each WGSL shader is exported as a `pub const`
from its parent Rust library module. ToadStool/BarraCUDA can absorb these by
importing the constant and copying the WGSL source directly.

See `shaders/ABSORPTION_TRACKER.md` for the full tracker.

---

## Relationship to hotSpring metalForge

| hotSpring metalForge | neuralSpring metalForge |
|---------------------|------------------------|
| GPU f64 native throughput | ML dispatch overhead profiling |
| NPU (Akida) characterization | Matmul cache tiling analysis |
| Cache line behavior | Workgroup occupancy vs tensor size |
| Register space probing | Shared memory pressure in tiled kernels |
| MD cell-list shaders (WGSL) | HMM/ODE/fitness shaders (WGSL) |
| Physics → ToadStool absorption | ML/bio → ToadStool absorption |

Both feed findings and shaders to ToadStool via `wateringHole/handoffs/`.

---

*Hardware characterization + shader evolution for ML dispatch — following the hotSpring metalForge pattern.*
