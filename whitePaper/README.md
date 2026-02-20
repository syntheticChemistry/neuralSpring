# neuralSpring White Paper

## The Isomorphic Learning Engine

**Status**: Working draft — Phase 2 complete
**Date**: February 20, 2026
**License**: AGPL-3.0-or-later

---

### Document Index

| Document | Audience | Description |
|----------|----------|-------------|
| [STUDY.md](STUDY.md) | Technical | Main study: experiments, results, BarraCUDA evolution |
| [BARRACUDA_EVOLUTION.md](BARRACUDA_EVOLUTION.md) | ToadStool team | Shader evolution narrative: Python → CPU → GPU |
| `specs/BENCHMARK_ANALYSIS.md` | Engineering | Full 3-way benchmark with analysis |
| `specs/TOADSTOOL_HANDOFF.md` | Engineering | 11 BarraCUDA shortcomings + local fixes |
| `specs/EVOLUTION_MAPPING.md` | Engineering | Tier A/B/C module-by-module GPU promotion map |
| `wateringHole/handoffs/` | Cross-project | Formal handoffs (date-stamped) |

---

### What This Study Is

neuralSpring validates machine learning primitives on consumer hardware using
BarraCUDA's WGSL shader library — the same library hotSpring uses for nuclear
physics. The central claim: **all neural architectures decompose into six
fundamental primitives**, and a single engine optimizing those primitives in
WGSL serves every domain.

### What This Study Is Not

- Not a machine learning framework (no training loops, no autograd)
- Not a competitor to PyTorch/JAX (uses them as baselines)
- Not limited to ML — the Phase 0++ papers span evolutionary biology, phylogenetics, game theory, spectral analysis, and regulatory networks
- Not GPU-dependent — all validation runs on CPU (llvmpipe) and GPU (Vulkan)

---

### Three Questions

1. **Can neural surrogates replace equation chains?**
   Yes. MLP surrogate for FAO-56 ET₀ achieves R²>0.999 with 2000 training
   samples. Same 6-layer pipeline replaces the full Penman-Monteith chain.

2. **Can compiled WGSL shaders beat Python/NumPy for ML inference?**
   Yes, at scale. GPU (RTX 4070) is **104× faster** than single-thread Python
   at 103M FLOPs. CPU (llvmpipe) is **3.9× faster** at the same scale.
   Both execute the same WGSL source — ToadStool compiles to x86 or Vulkan.

3. **Does the hotSpring progression (Python > CPU > GPU) hold for ML?**
   Yes, at crossover scales. The 3-way benchmark achieves
   **GPU < CPU < Python** at MLP large (3.1M FLOPs) and Transformer medium
   (103M FLOPs). GPU dominates CPU by 4–80× at every scale.

---

### Key Results Summary

**Phase 0/0+/0++**: 190/190 Python PASS (48 synthetic + 31 scholarly + 111 paper reproductions)
**Phase 1**: 532/532 Rust validation PASS (167 native + 242 BarraCUDA primitives + 123 CPU ports)
**Grand Total**: 722/722 PASS

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Synthetic baselines — 5 experiments, 48 checks | **Complete** |
| 0+ | Scholarly reproductions — 5 studies, 31 checks | **Complete** |
| 0++ | Paper reproductions — 13 papers, 111 checks | **Complete** |
| 1a | Rust validation layer — 167 native checks (16 binaries) | **Complete** |
| 1b | BarraCUDA validation — 242 checks (10 domains) | **Complete** |
| 1c | Fused pipeline — 46–78× speedup | **Complete** |
| 1d | 3-way benchmark + double-buffered shaders | **Complete** |
| 2 | BarraCUDA CPU ports — 13 modules, 123 checks | **Complete** |

#### 3-Way Benchmark Highlights (Phase 1d)

| Scale | Py(1t) | CPU | GPU | GPU/CPU |
|-------|--------|-----|-----|---------|
| MLP large (3.1M FLOPs) | 3.0 ms | **2.7 ms** | **178 µs** | 15× |
| TF medium (103M FLOPs) | 59 ms | **15.1 ms** | **566 µs** | 27× |
| TF xlarge (6.6B FLOPs) | 232 ms | 1.42 s | **17.8 ms** | **80×** |

Correctness: max diff 1.49e-8 (MLP), 1.10e-6 (Transformer) — same WGSL,
same math, both backends.

---

### The Isomorphism Theorem

All neural architectures decompose into compositions of six fundamental primitives:

1. **GEMM** (matrix multiply) — 60–90% of all FLOPs
2. **Attention** (scaled dot-product) — learned routing
3. **Normalization** (LN/BN/RMS) — scale stabilization
4. **Nonlinearity** (ReLU/GELU/SiLU) — feature carving
5. **Reduction** (sum/mean/max) — aggregation
6. **Gating** (sigmoid × value) — information filtering

A single engine optimizing these 6 ops in WGSL serves every domain:
language (llama.cpp), protein (OpenFold), vision (ViT), physics (hotSpring),
time series (weather), evolution (Dolson), phylogenetics (Liu), game theory
(Waters), spectral analysis (Kachkovskiy), and quantized deployment.

---

### Phase 0++ — Scholarly Reproduction Catalog

The 13 Phase 0++ papers span 4 faculty research groups and 5 scientific
disciplines, demonstrating that the same computational primitives (GEMM,
reduction, softmax, elementwise, ODE integration, eigendecomposition) appear
across all domains:

| Faculty | Papers | Disciplines | Key Primitives |
|---------|--------|-------------|----------------|
| **Dolson** (MSU CS) | 011–015 | Evolutionary computation, swarm robotics | Tournament/lexicase selection, fitness GEMM, mutation |
| **Liu** (MSU CSE) | 016–018 | Phylogenetics, genomics, alignment | HMM forward/backward (GEMM chain), NJ tree, introgression |
| **Waters** (MSU Micro) | 019–021 | Game theory, regulatory biology | Replicator dynamics (softmax), ODE (Hill functions), AND gate |
| **Kachkovskiy** (MSU Math) | 022–023 | Spectral theory, quantum mechanics | Eigendecomposition, commutator, IPR, localization |

#### Cross-Domain Primitive Usage

| Primitive | Dolson | Liu | Waters | Kachkovskiy |
|-----------|--------|-----|--------|-------------|
| GEMM / MatMul | Fitness evaluation | HMM forward/backward, distance matrix | Payoff matrix | — |
| Reduction (sum/mean) | Population statistics | Log-likelihood | Population averages | IPR, norm |
| Softmax / Boltzmann | Counterdiabatic driving | — | Replicator dynamics | — |
| Elementwise | Mutation, fitness | Sequence operations | ODE derivatives | Hamiltonian construction |
| Eigendecomposition | — | — | — | Jacobi eigensolver, spectral analysis |
| ODE integration | — | — | RK4 for GRN, signal dynamics | — |

---

### BarraCUDA Shader Evolution

The same pattern hotSpring demonstrated for nuclear physics — Python control,
then Rust/WGSL evolution — applies to ML inference:

| Stage | What Happened | Result |
|-------|---------------|--------|
| Python control | NumPy/PyTorch baselines for all 23 experiments | 190/190 PASS |
| BarraCUDA validation | 242 checks across 10 modules (CPU + GPU) | 242/242 PASS |
| Fused pipeline | Single-encoder dispatch, eliminate per-op overhead | **46–78× over per-op** |
| BLAS-evolved CPU shader | 32×32 tiles, vec4, 8×4 micro-kernel, k-unroll | CPU beats Py at 3M+ FLOPs |
| Double-buffered GPU shader | Load/compute overlap, 2×2 micro-kernel | **10–12% faster at scale** |
| 4-tier router | DeviceCapabilities-driven matmul selection | Best kernel per dispatch |

See [BARRACUDA_EVOLUTION.md](BARRACUDA_EVOLUTION.md) for the full technical narrative.

---

### Phase 2 — BarraCUDA CPU Port Findings

All 13 Phase 0++ modules ported to BarraCUDA CPU math, proving the
hand-rolled Rust math is reproducible via BarraCUDA's pure-Rust primitives:

| Primitive | Modules Using It | Precision Finding |
|-----------|-----------------|-------------------|
| `rk45_solve` | regulatory, signal, game | Machine-precision agreement with hand-rolled RK4 |
| `eigh_f64` | spectral, anderson | ~1e-3 at n=8, ~0.1 at n=16 — Jacobi eigensolver accuracy gap |
| `solve_f64` | hmm, swarm | Machine precision for linear systems |
| `chi_squared_sf` | introgression | Correctly reproduces LRT p-values |
| `stats::variance` | all 13 modules | Cross-validates hand-rolled statistics |
| `pearson_correlation` | modes | Validates complexity trend analysis |

**Key discovery:** BarraCUDA's Jacobi eigensolver (`eigh_f64`) has a
significant accuracy gap at n≥8. This is flagged as the #1 ToadStool
handoff item — a GPU-accelerated Lanczos or divide-and-conquer eigensolver
would resolve this for both hotSpring (nuclear physics) and neuralSpring
(spectral analysis).

---

### BarraCUDA Evolution Opportunities from Phase 0++

The Phase 0++ papers reveal new algorithmic patterns that could benefit from
BarraCUDA GPU acceleration:

| Pattern | Papers | Current | Opportunity |
|---------|--------|---------|-------------|
| **Batch fitness evaluation** | 011–015 | CPU loop over population | GPU-parallel fitness eval (batch GEMM) |
| **HMM forward/backward chain** | 016–018 | Sequential matrix multiply | Batched GEMM chain with log-domain numerics |
| **Replicator dynamics** | 019 | CPU ODE integration | GPU elementwise + softmax per timestep |
| **ODE integration (RK4)** | 020–021 | CPU RK4 loop | GPU-parallel multi-system ODE integration |
| **Eigendecomposition** | 022–023 | CPU Jacobi iteration | GPU tridiagonal eigensolver (Householder → bisection) |
| **Distance matrix** | 017 | O(N²) pairwise | GPU-parallel pairwise computation |
| **Spatial cooperation** | 019 | CPU neighborhood scan | GPU stencil convolution |

---

### Cross-Spring Connection

| Spring | Provides | neuralSpring Uses |
|--------|----------|-------------------|
| airSpring | FAO-56 ET₀ model | Surrogate target, real weather data |
| groundSpring | Noise labels, uncertainty | Training robustness, domain gap quantification |
| hotSpring | Physics surrogates (RBF), BarraCUDA patterns | Shader evolution methodology, benchmark patterns |
| wetSpring | Taxonomy pipelines | HMM chains for phylogenetics (Papers 016–018), metagenomics bridge |

---

### Research Questions Answered (23 Papers)

1. **Can neural surrogates replace equation chains?** Yes — MLP for FAO-56 at R²>0.999
2. **Is self-attention correct from scratch?** Yes — NumPy matches PyTorch to <1e-10
3. **Can LSTM learn weather patterns?** Yes — R²≈0.93, NSE=0.849 on real ERA5
4. **Does transfer learning work across climates?** Yes — 200 NM samples recover domain gap
5. **Are architectures isomorphic?** Yes — 6 primitives, all in BarraCUDA
6. **Can PINNs solve PDEs?** Yes — Burgers' equation to 5.1% L2 error
7. **Can operators be learned?** Yes — DeepONet to 1.2% L2 error
8. **Does quantization preserve accuracy?** Yes — INT8: 0.017% loss, INT4: 0.79%
9. **Can WGSL beat Python for ML?** Yes — CPU 3.9× faster at 103M, GPU 104× faster
10. **Does the hotSpring progression hold?** Yes — GPU < CPU < Python at crossover
11. **Can evolution be controlled?** Yes — counterdiabatic driving outperforms naive (Paper 011)
12. **Can open-endedness be measured?** Yes — MODES metrics distinguish open vs closed (Paper 012)
13. **Do EAs behave as ecological communities?** Yes — niche dynamics, FDS (Paper 013)
14. **Does lexicase outperform tournament?** Yes — higher diversity + Pareto (Paper 014)
15. **Do heterogeneous controllers help?** Yes — more diversity, comparable fitness (Paper 015)
16. **Is HMM forward/backward a GEMM chain?** Yes — bridges neuralSpring → wetSpring (Paper 016)
17. **Can iterative coestimation improve alignment?** Yes — SATé refinement (Paper 017)
18. **Can introgression be detected via HMM?** Yes — PhyloNet-HMM + LRT (Paper 018)
19. **Does QS resolve cooperation dilemmas?** Yes — game theory + spatial (Paper 019)
20. **Can one gene produce multiple strategies?** Yes — bistability in GRN (Paper 020)
21. **Is signal integration a biological AND gate?** Yes — two-input Hill (Paper 021)
22. **Do skip connections reduce commutativity?** Yes — residual layers commute better (Paper 022)
23. **Does disorder produce localization?** Yes — Aubry-André transition at W_c=2t (Paper 023)

---

### Reproduction

```bash
# Phase 0/0+/0++ Python baselines (190/190)
pip install -r control/requirements.txt
bash scripts/run_all_baselines.sh

# Rust validation (532/532)
cargo test
make validate

# 3-way benchmark (Python vs CPU vs GPU)
cargo run --release --bin bench_scaling
```

---

### Next Phase: BarraCUDA GPU Evolution

Phase 2 (BarraCUDA CPU ports) is complete: all 13 Phase 0++ modules validated
against BarraCUDA CPU math primitives (123 checks). The next step is
**BarraCUDA GPU acceleration** via ToadStool's unidirectional streaming,
massively reducing dispatch and round-trips.

Priority targets for GPU promotion:
1. **HMM forward/backward** (Papers 016–018) — GEMM chain, direct port
2. **Batch fitness evaluation** (Papers 011–015) — parallel population eval
3. **Eigendecomposition** (Papers 022–023) — tridiagonal eigensolver
4. **ODE integration** (Papers 020–021) — parallel multi-system RK4
5. **Distance matrix** (Paper 017) — O(N²) pairwise → GPU parallel

---

*23 papers. 5 disciplines. 4 faculty. 190 Python + 532 Rust = 722 total checks.
All green. Paper queue cleared. Phase 2 complete. Ready for BarraCUDA GPU evolution.*
