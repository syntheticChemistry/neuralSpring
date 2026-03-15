# neuralSpring White Paper

## The Isomorphic Learning Engine

**Status**: Phase 5h+ — **4500+ total checks**, ALL GREEN, ~97% GPU promotion, CPU↔Python parity 41/41, 38.6× faster than Python (15 domains, honest geomean), 260 binaries, **220/220 validate\_all**. coralForge unified. 80+ named tolerances (centralized registry), zero debt, 0 clippy pedantic+nursery warnings, 0 doc warnings. 1115 lib + 63 playGround unit + 13 integration + 73 forge tests. 46 upstream rewires, 219 barracuda import files, 45+ submodules. barraCuda v0.3.5 at `0649cd0`, wgpu 28. playGround: Squirrel MCP adapter + HuggingFace Model Lab + compute triangle (ToadStool/coralReef IPC clients). 16 petalTongue scenario builders, `neuralspring_ecosystem_dashboard`, `config.rs` centralized identity. Streaming FASTA/FASTQ/VCF parsers, CPU BLAST pipeline, Kokkos parity harness.
**Date**: March 14, 2026 (Sessions 40–150 — S150: Compute triangle integration (ToadStool+coralReef IPC, hot/cold dispatch benchmarks). S149: HuggingFace Model Lab. S148: Squirrel MCP adapter. S147: Deep debt, V100 handoff. 1115 lib tests, 260 binaries. barraCuda v0.3.5 at `0649cd0`, toadStool S146+, coralReef Phase 10)
**License**: AGPL-3.0-or-later

---

### Document Index

| Document | Audience | Description |
|----------|----------|-------------|
| [STUDY.md](STUDY.md) | Technical | Main study: experiments, results, BarraCUDA evolution |
| [BARRACUDA_EVOLUTION.md](BARRACUDA_EVOLUTION.md) | ToadStool team | Shader evolution narrative: Python → CPU → GPU |
| [CROSS_SPRING_SHADER_LINEAGE.md](CROSS_SPRING_SHADER_LINEAGE.md) | All teams | Cross-spring shader evolution: hotSpring, wetSpring, neuralSpring → BarraCUDA |
| [baseCamp/README.md](baseCamp/README.md) | Faculty/paper | Per-faculty briefings + baseCamp research program (Biophysical AI Interpretability) |
| [baseCamp/extensions.md](baseCamp/extensions.md) | Thesis committee | 5 sub-theses, 15 grounding papers, 28 experiments — novel cross-domain AI research |
| `specs/BENCHMARK_ANALYSIS.md` | Engineering | Full 3-way benchmark with analysis |
| `specs/TOADSTOOL_HANDOFF.md` | Engineering | BarraCUDA shortcomings — all 17 resolved (S-01 through S-17) |
| `specs/PURE_GPU_ROADMAP.md` | Engineering | Pure GPU roadmap — Phase A+B+C complete (44 ops, ~97% GPU coverage) |
| `specs/EVOLUTION_MAPPING.md` | Engineering | Tier A/B/C module-by-module GPU promotion map |
| `experiments/README.md` | Engineering | Experiment journals (001–096, hotSpring pattern) |
| `wateringHole/handoffs/` | Cross-project | S139 handoff (Session 139 — visualization evolution + deep debt) |

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
   at 103M FLOPs. Pure Rust is **83.6× geomean** faster than Python across 11 Phase 0++
   computational kernels. Both execute the same WGSL source — ToadStool compiles
   to x86 or Vulkan.

3. **Does the hotSpring progression (Python > CPU > GPU) hold for ML?**
   Yes, at crossover scales. The 3-way benchmark achieves
   **GPU < CPU < Python** at MLP large (3.1M FLOPs) and Transformer medium
   (103M FLOPs). GPU dominates CPU by 4–80× at every scale.

4. **Is the math truly portable across GPU architectures and drivers?**
   Yes. All 218 validators produce **bit-identical** results on RTX 4070
   (proprietary NVIDIA Vulkan) and TITAN V (NVK open-source driver).
   Same WGSL source, different GPU generations and driver stacks.

5. **Can production math run entirely on GPU?**
   ~97%. Sessions 45-46 + 66 promoted 44 CPU-bound operations to GPU via
   `gpu_dispatch::Dispatcher` — HMM forward/backward/Viterbi, meta-population
   statistics, replicator dynamics, Hill activation, and more. Validated
   on both GPUs (47/47 PASS across Phase A + B validators).

---

### Key Results Summary

**Phase 0/0+/0++**: 397/397 Python PASS (48 synthetic + 31 scholarly + 127 paper reproductions + 30 pub exp + 27 WDM + 19 coralForge + 1 isomorphic fix + 114 extended baselines)
**Phase 1–5h+**: 4000+ Rust+GPU validation PASS (1115 lib + 63 playGround unit + 13 integration + 73 forge tests + 260 binaries across 47+ modules)
**Grand Total**: 4500+ PASS — **ALL GREEN** across all applicable tiers
**Multi-GPU**: 220 validators on RTX 4070, 384+ additional on TITAN V (NVK) — **bit-identical**
**GPU Promotion**: 47 CPU-bound ops → GPU dispatch (Phase A: 27, Phase B: 11, Phase C: 6, +3 upstream). ~97% of production math on GPU.
**Mixed-Hardware**: `Dispatcher::mixed_dispatch()` wired to metalForge cost model (GPU↔NPU↔CPU routing).
**Benchmarks**: Pure Rust **38.6× faster** than Python/NumPy (15 domains, honest geomean); GPU **104× faster** at 103M FLOPs
**Visualization**: 16 petalTongue scenario builders, ecosystem dashboard, config centralization (S139)

Phase 5e achieved pure GPU promotion and mixed-hardware dispatch: **44 CPU-bound
operations promoted to GPU dispatch**, validated on both RTX 4070 and TITAN V (NVK).
All 218 validators pass on RTX 4070 with bit-identical results on TITAN V. The
`gpu_dispatch::Dispatcher` provides capability-based routing: GPU when available,
CPU fallback otherwise. `mixed_dispatch()` extends this with metalForge's
cross-device cost model for GPU↔NPU↔CPU substrate selection.

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Synthetic baselines — 5 experiments, 48 checks | **Complete** |
| 0+ | Scholarly reproductions — 5 studies, 31 checks | **Complete** |
| 0++ | Paper reproductions — 15 papers, 127 checks | **Complete** |
| 1a | Rust validation layer — 1048 lib + 9 integration + 71 forge tests, 233 binaries, 41+ modules | **Complete** |
| 1b | BarraCUDA validation — 272 checks (12 domains incl. FFT) | **Complete** |
| 1c | Fused pipeline — 46–78× speedup | **Complete** |
| 1d | 3-way benchmark + double-buffered shaders | **Complete** |
| 2 | BarraCUDA CPU ports — 24/25 papers, 203 checks (96%) | **Complete** |
| 3a | BarraCUDA FFT validation — 24 analytical checks | **Complete** |
| 3b | GPU streaming (`StatefulPipeline` + `UnidirectionalPipeline`) | **Complete** |
| 3c | Shader evolution (17 WGSL shaders, 108 checks) | **Complete** |
| 3d | Cross-dispatch (6 validators, 49 checks, 15/15 papers) | **Complete** |
| 4 | Phase 4a–4e: GPU pipelines, PRNG, MHA, eigendecomposition | **Complete** |
| 5a | GPU Tensor validation (7 original domains, 43 checks) | **Complete** |
| 5b | Full-stack buildout (bC 24/25, gT 23/25, xD 15/15) | **Complete** |
| 5c | Upstream parity, spectral theory, capability dispatch, cross-eigensolver | **Complete** |
| 5d | Multi-GPU (RTX 4070 + TITAN V NVK), benchmarks (83.6×), stochastic GPU pipelines | **Complete** |
| 5e | Pure GPU promotion — 44 ops via gpu_dispatch, ~97% math on GPU (Phase A+B+C) | **Complete** |
| 5f | WDM surrogates — 5 Python baselines + 6 Rust validators (CPU + BarraCUDA GPU) | **Complete** |
| 5g | baseCamp GPU pure — all 5 sub-theses validated on GPU with scalar readback | **Complete** |
| 5h | ToadStool S66 absorption — 6 rewires, 15 df64 shaders aligned to `compile_shader_df64` | **Complete** |

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

The 15 Phase 0++ papers span 4 faculty research groups and 5 scientific
disciplines, demonstrating that the same computational primitives (GEMM,
reduction, softmax, elementwise, ODE integration, eigendecomposition) appear
across all domains:

| Faculty | Papers | Disciplines | Key Primitives |
|---------|--------|-------------|----------------|
| **Dolson** (MSU CS) | 011–015 | Evolutionary computation, swarm robotics | Tournament/lexicase selection, fitness GEMM, mutation |
| **Liu** (MSU CSE) | 016–018 | Phylogenetics, genomics, alignment | HMM forward/backward (GEMM chain), NJ tree, introgression |
| **Waters** (MSU Micro) | 019–021 | Game theory, regulatory biology | Replicator dynamics (softmax), ODE (Hill functions), AND gate |
| **Kachkovskiy** (MSU Math) | 022–023 | Spectral theory, quantum mechanics | Eigendecomposition, commutator, IPR, localization |
| **Liu** (MSU CSE) | 024–025 | Genomics, population genetics | Pangenome selection, meta-population dynamics |

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
| Python control | NumPy/PyTorch baselines for all 27 papers + WDM + pub exp | 397/397 PASS |
| BarraCUDA validation | 272 checks across 12 modules (CPU + GPU + FFT) | 272/272 PASS |
| Fused pipeline | Single-encoder dispatch, eliminate per-op overhead | **46–78× over per-op** |
| BLAS-evolved CPU shader | 32×32 tiles, vec4, 8×4 micro-kernel, k-unroll | CPU beats Py at 3M+ FLOPs |
| Double-buffered GPU shader | Load/compute overlap, 2×2 micro-kernel | **10–12% faster at scale** |
| 4-tier router | DeviceCapabilities-driven matmul selection | Best kernel per dispatch |

See [BARRACUDA_EVOLUTION.md](BARRACUDA_EVOLUTION.md) for the full technical narrative.

---

### Phase 2 — BarraCUDA CPU Port Findings

24/25 papers ported to BarraCUDA CPU math (96%), proving the
hand-rolled Rust math is reproducible via BarraCUDA's pure-Rust primitives:

| Primitive | Modules Using It | Precision Finding |
|-----------|-----------------|-------------------|
| `rk45_solve` | regulatory, signal, game | Machine-precision agreement with hand-rolled RK4 |
| `eigh_f64` | spectral, anderson | 1.75e-14 at n=32 — Householder+QR (S-12 absorbed at `77f70b2e`) |
| `solve_f64` | hmm, swarm | Machine precision for linear systems |
| `chi_squared_sf` | introgression | Correctly reproduces LRT p-values |
| `stats::variance` | all 15 modules | Cross-validates hand-rolled statistics |
| `pearson_correlation` | modes | Validates complexity trend analysis |

**Key discovery (resolved):** BarraCUDA's original Jacobi eigensolver had a
significant accuracy gap at n≥8. This was resolved as S-12 — ToadStool
absorbed neuralSpring's Householder+QR implementation at `77f70b2e`, achieving
machine-epsilon accuracy (1.75e-14 at n=32). ToadStool also added a GPU-native
NAK eigensolve (`batched_eigh_nak_optimized_f64.wgsl`) for both hotSpring
and neuralSpring use cases.

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

### Phase 5b — Full-Stack Validation (ALL GREEN)

The final validation layer: do BarraCUDA GPU `Tensor` operations produce
identical results to CPU f64 references? Phase 5b expands from the initial
7 domains to **23 papers** across all tiers. S-14, S-15, S-16 **RESOLVED** upstream (`a4996b34` S39); S-17 **RESOLVED** upstream (`c82c23d1` S58).

| Tier | Coverage | Status |
|------|----------|--------|
| Python control (Py) | 25/25 (100%) | **ALL PASS** |
| Rust CPU (Rs) | 25/25 (100%) | **ALL PASS** |
| BarraCUDA CPU (bC) | 24/25 (96%) | **ALL GREEN** |
| BarraCUDA GPU Tensor (gT) | 23/25 (92%) | **ALL GREEN** |
| metalForge WGSL (mF) | 15/25 (60%) | **ALL PASS** |
| GPU Pipeline (gP) | 15/25 (60%) | **ALL PASS** |
| Cross-dispatch (xD) | 15/15 (100%) | **ALL GREEN** |

The validation progression proves math portability at each level:
1. Open data + Python → reproducible science
2. Rust native → same math, type-safe
3. BarraCUDA CPU → pure Rust math matches
4. BarraCUDA GPU Tensor → math is portable CPU → GPU
5. metalForge WGSL → domain-specific GPU kernels validated
6. GPU Pipeline → end-to-end multi-kernel chains
7. Cross-dispatch → CPU ↔ GPU parity via routing

**Bug resolution:**
- **S-16** (transpose dispatch): **RESOLVED** upstream (`a4996b34` S39: Transpose dispatch fixed, `const TILE: u32 = 16`)
- **S-15** (matmul hang): **RESOLVED** upstream (`a4996b34` S39: Matmul hang fixed)
- **S-14** (naive matmul): **RESOLVED** upstream (`a4996b34` S39: Naive matmul tier removed)

Full handoff: `wateringHole/handoffs/`

---

### Cross-Spring Connection

| Spring | Provides | neuralSpring Uses |
|--------|----------|-------------------|
| airSpring | FAO-56 ET₀ model | Surrogate target, real weather data |
| groundSpring | Noise labels, uncertainty | Training robustness, domain gap quantification |
| hotSpring | MD primitives (RK4/RK45, Boltzmann, energy minimization) | Shader evolution methodology, benchmark patterns, **baseCamp nS-03 (loss landscapes as energy landscapes)** |
| wetSpring | Anderson QS framework, HMM phylogenetics, QS game theory | HMM chains (Papers 016–018), **baseCamp nS-01 (weight Hamiltonians), nS-05 (multi-agent QS)** |

---

### Research Questions Answered (25 Papers)

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
24. **Can pangenome selection improve fitness?** Yes — selection on pangenome graph structure (Paper 024)
25. **Do meta-populations exhibit source-sink dynamics?** Yes — spatial structure affects gene flow (Paper 025)

---

### Reproduction

```bash
# Phase 0/0+/0++ Python baselines (397/397)
pip install -r control/requirements.txt
bash scripts/run_all_baselines.sh

# Rust validation (1115 lib + 63 playGround unit + 13 integration + 73 forge tests + 260 binaries)
cargo test
cargo run --release --bin validate_all

# Tensor op benchmark (native BarraCUDA)
cargo run --release --bin bench_barracuda_tensor
```

---

### Phase 3 — GPU Evolution (Complete)

Phase 3a (BarraCUDA FFT validation): 24 analytical checks against
ToadStool's Cooley-Tukey WGSL shader. Absorbed directly from ToadStool.

Phase 3b (GPU-resident streaming): `StatefulPipeline` and
`UnidirectionalPipeline` validated (10/10 PASS).

Phase 3c (shader evolution): 17 WGSL shaders validated, all following
the hotSpring lifecycle: evolve-locally, handoff, retire-when-absorbed.

Phase 3d (cross-dispatch): GPU-CPU parity validated across 4 dispatch
binaries (41 checks).

#### GPU Promotion Priorities

| Priority | Target | Papers | Strategy |
|----------|--------|--------|----------|
| 1 | **HMM forward/backward** | 016–018 | GEMM chain → `StatefulPipeline` |
| 2 | **Batch fitness evaluation** | 011–015 | Population eval → batch GEMM shader |
| 3 | **Eigendecomposition** | 022–023 | Tridiagonal eigensolver (Householder → bisection) |
| 4 | **ODE integration (GPU RK4)** | 020–021 | Multi-system parallel ODE integration |
| 5 | **Distance matrix** | 017 | O(N²) pairwise → GPU parallel |
| 6 | **FFT-accelerated spectral** | 022–023 | Eigenvalue estimation via FFT-based Krylov |
| 7 | **Spatial cooperation** | 019 | Neighborhood scan → GPU stencil convolution |

#### metalForge Shader Evolution

Local WGSL shaders are developed in `metalForge/shaders/` following the
hotSpring pattern: evolve → validate → handoff → ToadStool absorbs → retire.
See `metalForge/README.md` for the development workflow and absorption tracker.

---

*27 papers + 5 studies + 6 baseCamp sub-theses + WDM surrogates + coralForge (nF-01/02/03) + 3 pub experiments + playGround. 5 disciplines. 4 faculty. 47+ modules. 1115 lib + 63 playGround unit + 13 integration + 73 forge tests. 397 Python + 4000+ Rust/GPU = 4500+ total checks.
Phase 5h+: ALL GREEN — bC 24/25 (96%) · gT 23/25 (92%) · xD 15/15 (100%) · uP 13/13 · mG 384/384 · pG 10/10 · cS 46/46 · xSE 55/55 · sfGPU 37/37 coralForge. 47 CPU→GPU promotions, 46 upstream rewires, 219 barracuda import files. 260 binaries, 220/220 validate\_all. 0 clippy pedantic+nursery, 0 doc warnings, 0 unsafe, 0 mocks, 0 unused deps. V100 handoff. barraCuda v0.3.5 at `0649cd0`, toadStool S146+, coralReef Phase 10. 16 petalTongue scenario tracks, ecosystem dashboard, playGround compute triangle, Squirrel MCP adapter.*
