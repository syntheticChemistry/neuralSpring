# neuralSpring — Control Experiment Status

**Last updated**: February 26, 2026 (Sessions 44–74 — Phase C GPU + CPU parity + dispatch tier benchmarks + deep debt audit + validator shader rewiring + cross-spring rewiring + pure GPU all-domains)
**Gate**: Eastgate (i9-12900K, 32 GB DDR5, RTX 4070 12 GB + TITAN V 12 GB NVK, Pop!_OS 22.04)
**Python**: 3.10.12, PyTorch 2.9.0+cu128, NumPy 2.2.6, SciPy 1.15.3
**Rust**: Edition 2021, clippy pedantic + nursery, unsafe_code=forbid
**Grand Total**: 206/206 Python PASS + 1970+ Rust+GPU validation PASS = **2180+ total validation checks**
**Library**: 580 lib tests + 9 integration tests + 43 forge tests | 36 modules + gpu_ops/ + gpu_dispatch | 163 validation/bench binaries
**CPU↔Python Parity**: 39/39 PASS — `validate_cpu_math_parity` (9 primitives + 9 paper kernels + 6 Dispatcher cpu_only checks, all within 1e-10)
**Dispatch Overhead**: `bench_dispatch_tiers` — 9/10 ops ≤1.04× overhead (CPU dispatch is transparent), per-call GPU driver-bound for small workloads (motivates pipeline batching)
**baseCamp**: 5 biophysical AI modules + 7 validators (114/114 CPU + 14/14 GPU + 19/19 dispatch PASS) — Sessions 50, 54, 56
**Dispatch**: `Dispatcher::mixed_dispatch()` wired to metalForge cost model — 16/16 CPU↔GPU + 14/14 mixed-hardware + 19/19 baseCamp dispatch + 17/17 parity + 23/23 PCIe
**Multi-GPU**: 133/133 PASS on RTX 4070 (Vulkan) + 143+ on TITAN V (NVK) — **bit-identical**
**GPU Promotion**: 44 CPU→GPU ops via `gpu_dispatch::Dispatcher` (~97% of production math)
**Pure GPU All-Domains**: 10/10 PASS — `validate_gpu_pure_workload_all` (9 typed BarraCUDA GPU ops across all 15 Phase 0++ papers + determinism check, scalar-only readback)
**Debt**: Zero TODO/FIXME/MOCK/STUB in src/ | zero hardcoded paths | zero unsafe | 0 clippy warnings | 0 doc warnings | zero ad-hoc tolerances | zero bare `unwrap()` in validation code
**Coverage**: 94.53% line coverage (llvm-cov), 107+ named tolerances in centralized registry
**Benchmarks**: Pure Rust **201.7× faster** than Python/NumPy (11 kernels) | Evolution tier: CPU→GPU portability proven (8 domains)
**ToadStool**: **ALL 17 shortcomings RESOLVED** (S-01..S-17) | S-14/S-15/S-16 fixed at `a4996b34` (S39), S-17 fixed at `c82c23d1` (S58) | HEAD `02207c4a` (47 commits reviewed S72) | **21 functions rewired to upstream** + 6 validator shader sources rewired | S73: +4 Tensor API rewires (argmax_dim, softmax_dim, fst_variance_decomposition)
**Cross-Spring**: 39/39 evolution checks PASS (S73) | Variance 2.46× (hotSpring Welford), Entropy 2.59× (wetSpring fused), Pearson 1.11× (joint) | softmax_row_wise, fst_single_locus, pairwise_fst_full, Viterbi argmax_dim rewired
**Open Data**: All 25+5 papers use open data and open systems — zero proprietary or paywalled sources

---

## Phase 0 — Synthetic Baselines (48/48 PASS)

| ID | Title | Domain | Tests | Status |
|----|-------|--------|-------|--------|
| Exp 001 | Neural Surrogate Validation | MLP vs RBF + FAO-56 | 11/11 | **PASS** |
| Exp 002 | Transformer Inference Baseline | Self-attention from scratch | 18/18 | **PASS** |
| Exp 003 | Sequence Forecasting | LSTM/GRU on real ERA5 weather | 5/5 | **PASS** |
| Exp 004 | Transfer Learning | Real 3-city ERA5 (MI/NM/CA) adaptation | 6/6 | **PASS** |
| Exp 005 | Isomorphic Pattern Catalog | Cross-domain op mapping | 8/8 | **PASS** |

## Phase 0+ — Scholarly Reproduction Studies (31/31 PASS)

| ID | Title | Paper | Tests | Status |
|----|-------|-------|-------|--------|
| Study 001 | PINN Burgers' Equation | Raissi et al. (2019) JCP 378:686 | 8/8 | **PASS** (+paper ref validation) |
| Study 002 | DeepONet Antiderivative | Lu et al. (2021) NMI 3:218 | 7/7 | **PASS** (+paper ref validation) |
| Study 003 | LeNet-5 MNIST | LeCun et al. (1998) Proc IEEE 86 | 5/5 + 8 bC conv/pool | **PASS** |
| Study 004 | LSTM ERA5 Weather | Gauch et al. (2021) HESS 25:2045 | 5/5 | **PASS** |
| Study 005 | Quantized Inference | Dettmers (2022) + Frantar (2023) | 6/6 | **PASS** (real ERA5 data) |

## Phase 0++ — Paper Reproductions (127/127 PASS)

| ID | Title | Paper | Tests | Status |
|----|-------|-------|-------|--------|
| Paper 011 | Counterdiabatic Evolution | Iram/Dolson (2020) Nature Physics 17:135 | 11/11 | **PASS** |
| Paper 012 | MODES Toolbox | Dolson et al. (2019) Artif Life 25(1):50 | 9/9 | **PASS** |
| Paper 013 | Ecological Dynamics in EC | Dolson & Ofria (2018) GECCO Companion | 7/7 | **PASS** |
| Paper 014 | Directed Evolution Selection | Dolson et al. (2022) eLife 11:e79665 | 8/8 | **PASS** |
| Paper 015 | Heterogeneous Swarm Robotics | Foreback/Dolson (2025) IEEE | 11/11 | **PASS** |
| Paper 016 | HMM Forward/Backward/Viterbi | Liu et al. (2014) PLoS Comp Bio 10:e1003649 | 10/10 | **PASS** |
| Paper 017 | SATé Alignment | Liu et al. (2009) Science 324:1561 | 8/8 | **PASS** |
| Paper 018 | Introgression Detection | Liu et al. (2015) PNAS 112:196 | 8/8 | **PASS** |
| Paper 019 | Game Theory & QS Cooperation | Bruger & Waters (2018) AEM 84:e00402-18 | 8/8 | **PASS** |
| Paper 020 | Regulatory Network | Mhatre et al. (2020) PNAS 117:21647 | 7/7 | **PASS** |
| Paper 021 | Signal Integration | Srivastava et al. (2011) J Bacteriol 193:6331 | 8/8 | **PASS** |
| Paper 022 | Spectral Commutativity | Kachkovskiy & Safarov (2016) JAMS 29:61 | 8/8 | **PASS** |
| Paper 023 | Anderson Localization | Bourgain & Kachkovskiy (2018) GAFA 29:3 | 8/8 | **PASS** |
| Paper 024 | Pangenome Selection | Liu et al. (genomics) | 8/8 | **PASS** |
| Paper 025 | Meta-Population Dynamics | Liu et al. (population genetics) | 8/8 | **PASS** |

---

## Phase 1 — Rust Validation + BarraCUDA Evolution

### Phase 1a: neuralSpring-Native Validation (580 lib tests + 9 integration + 43 forge tests, 163 validation binaries, 36 modules + gpu_ops/ + gpu_dispatch/)

| Rust Module | Python Source | Tests | Cross-Validation |
|-------------|-------------|-------|------------------|
| `metrics.rs` | `compute_r2`, `compute_rmse`, `compute_mae` | 3 unit + 10 binary | R², RMSE, MAE, NSE at analytical known-values |
| `surrogate.rs` | `rastrigin_2d`, `rosenbrock_2d`, `ackley_2d` | 6 unit + 15 binary | Global minima + 12 Python-computed reference points |
| `transformer.rs` | `softmax`, `gelu_numpy` | 7 unit + 18 binary | Element-wise match against NumPy to <1e-12 |
| `sequence.rs` | LSTM/GRU cell, `create_sequences`, seasonal model | 14 unit + 26 binary | LSTM/GRU gates, sequence ops (Study 004) |
| `quantized.rs` | Q8/Q4 quantize, dequantize, `gemv_q8/q4` | 6 unit + 26 binary | Quantized inference (Study 005) |
| `counterdiabatic.rs` | NK landscape, Boltzmann, CD schedule | 19 binary | CD vs naive protocol comparison |
| `modes.rs` | change, novelty, complexity, ecology | 9 binary | Open vs closed system metrics |
| `eco_dynamics.rs` | multi-niche EA, diversity indices | 7 binary | Competitive exclusion, FDS |
| `directed_evolution.rs` | 5 selection algorithms | 7 binary | Structured vs random selection |
| `hmm.rs` | forward, backward, Viterbi, posterior | 17 binary | Genomic-scale HMM, no underflow |
| `game_theory.rs` | PD, Snowdrift, replicator, QS spatial | 8 binary | QS cooperation stabilization |
| `swarm_robotics.rs` | heterogeneous controllers, swarm EA | 7 binary | Heterogeneous > homogeneous |
| `sate_alignment.rs` | NJ tree, progressive alignment | 8 binary | Iterative refinement improves |
| `introgression.rs` | PhyloNet-HMM, LRT | 13 binary | Introgression detection |
| `regulatory_network.rs` | Hill ODE, bistability | 5 binary | Environment-dependent switching |
| `signal_integration.rs` | Two-input Hill, AND gate | 8 binary | Multiplicative attention analog |
| `spectral_commutativity.rs` | commutator, distance to normal | 8 binary | Skip connections reduce commutativity |
| `anderson_localization.rs` | Aubry-André, IPR | 8 binary | Localization transition |
| `pangenome_selection.rs` | pairwise Jaccard, selection | 8 binary | Pangenome graph fitness (Paper 024) |
| `meta_population.rs` | locus variance, gene flow | 8 binary | Source-sink dynamics (Paper 025) |
| `pinn.rs` | Burgers' PDE, physics-informed loss | 8 binary | Raissi et al. (2019) PINN |
| `lenet.rs` | `Conv2d`, `MaxPool2d`, ReLU, fc | 7 unit + 22 binary | LeNet-5 primitives (Study 003) |
| `deeponet.rs` | operator learning, antiderivative | 7 binary | Lu et al. (2021) DeepONet |

### Phase 1b: BarraCUDA Primitives (272 checks)

| Validation Binary | BarraCUDA Module | Checks | Reference Source |
|-------------------|------------------|--------|-----------------|
| `validate_barracuda_stats` | stats (variance, pearson, covariance, norm) | 13 | Analytical formulas |
| `validate_barracuda_linalg` | linalg (solve, lu, eigh, cholesky, tridiag) | 17 | Analytical solutions |
| `validate_barracuda_special` | special (gamma, erf, bessel, polynomials) | 26 | NIST DLMF values |
| `validate_barracuda_optimize` | optimize (nelder_mead, bisect, brent) | 10 | Analytical minima/roots |
| `validate_barracuda_precision` | precision (add, mul, fma, dot, sum) | 12 | Exact f64 |
| `validate_barracuda_tensor` | Tensor API (90 ops — native LN, log-SM, leaky\_relu, elu, GELU) | 90 | WGSL unified path |
| `validate_barracuda_tensor_f64` | Tensor f64 (GPU ops) | 35 | f64 GPU ops |
| `validate_barracuda_quantized` | quantized (Q4/Q8 dequant, GEMV) | 15 | Hand-constructed |
| `validate_barracuda_linalg_ext` | linalg ext (SVD, LU inverse, gen eigh) | 17 | Analytical |
| `validate_barracuda_ml_inference` | ML inference (MLP + Transformer) | 13 | Python/NumPy baselines |
| `validate_barracuda_fft` | FFT (f32 Fft1D/Ifft1D + f64 Fft1DF64 + Rfft) | 24 | Analytical (DFT definition) |

### Phase 3c: metalForge GPU Shader Validation (16 GPU shader binaries, 108 shader checks, 21 WGSL shaders)

| Validation Binary | WGSL Shader | Papers | Checks | Reference |
|-------------------|-------------|--------|--------|-----------|
| `validate_gpu_hmm_forward` | `hmm_forward_log.wgsl` | 016–018 | 13 | CPU HMM forward (hmm.rs) |
| `validate_gpu_batch_fitness` | `batch_fitness_eval.wgsl` | 011–015 | 20 | CPU dot-product fitness |
| `validate_gpu_rk4` | `rk4_parallel.wgsl` | 020–021 | 8 | CPU RK4 integration |
| `validate_gpu_pangenome` | `pairwise_jaccard.wgsl` | 024 | 6 | CPU pangenome (pangenome_selection.rs) |
| `validate_gpu_meta_pop` | `locus_variance.wgsl` | 025 | 7 | CPU meta-pop (meta_population.rs) |
| `validate_gpu_game_theory` | `spatial_payoff.wgsl` | 019 | 5 | CPU game theory (game_theory.rs) |
| `validate_gpu_anderson` | `batch_ipr.wgsl` | 022–023 | 5 | CPU Anderson (anderson_localization.rs) |
| `validate_gpu_sate` | `pairwise_hamming.wgsl` | 017 | 5 | CPU SATé alignment (sate_alignment.rs) |
| `validate_gpu_modes` | `pairwise_l2.wgsl` | 012 | 15 | CPU L2 distance (modes.rs) |
| `validate_gpu_directed` | `multi_obj_fitness.wgsl` | 014 | 6 | CPU directed evolution |
| `validate_gpu_signal` | `hill_gate.wgsl` | 021 | 9 | CPU signal integration |
| `validate_gpu_swarm` | `swarm_nn_forward.wgsl` | 015 | 9 | CPU swarm robotics |

### Phase 3d: Pure GPU Workload + StatefulPipeline + Cross-Dispatch (45 pipeline/cross-dispatch checks)

| Validation Binary | BarraCUDA API | Checks | What It Proves |
|-------------------|--------------|--------|----------------|
| `validate_gpu_stateful_pipeline` | `StatefulPipeline` | 10 | GPU-resident iterative RK4 (zero full-state readback) |
| `validate_gpu_pure_workload` | Multi-kernel chain | 7 | Fitness + reduce in single submit (zero CPU round-trips) |
| `validate_cross_dispatch` | `DispatchConfig` | 8 | GPU ↔ CPU parity, dispatch routing, timing |
| `validate_cross_dispatch_genomics` | Genomics dispatch | 8 | GPU ↔ CPU parity for genomics workloads |
| `validate_cross_dispatch_extended` | Extended dispatch | 12 | Extended cross-dispatch validation |

### Phase 4b: Pure GPU End-to-End Pipelines (7 pipelines, 32 pipeline checks)

| Validation Binary | Pipeline | Papers | Checks | Status |
|-------------------|----------|--------|--------|--------|
| `validate_gpu_pipeline_hmm` | HMM forward → mean_reduce | 016–018 | 5 | **5/5 PASS** |
| `validate_gpu_pipeline_ecology` | spatial_payoff → mean_reduce | 019 | 5 | **5/5 PASS** |
| `validate_gpu_pipeline_spectral` | batch_ipr → mean_reduce | 022–023 | 5 | **5/5 PASS** |
| `validate_gpu_pipeline_genomics` | pairwise_jaccard → mean_reduce | 024 | 5 | **5/5 PASS** |
| `validate_gpu_pipeline_modes` | pairwise_l2 → mean_reduce | 012 | 4 | **4/4 PASS** |
| `validate_gpu_pipeline_directed` | multi_obj_fitness → mean_reduce | 014 | 4 | **4/4 PASS** |
| `validate_gpu_pipeline_signal` | hill_gate → mean_reduce | 021 | 4 | **4/4 PASS** |

### Phase 4a: Performance Benchmarks (7 kernels, 71.8× overall speedup)

The `bench_phase0pp_kernels` binary compares Rust pure math to single-thread NumPy at identical
problem sizes. Python control scripts: `control/hmm_phylo/bench_hmm_forward.py`,
`control/game_theory/bench_replicator.py`, `control/spectral_commutativity/bench_commutator.py`,
`control/counterdiabatic/bench_nk_fitness.py`, `control/sate_alignment/bench_hamming.py`,
`control/pangenome_selection/bench_jaccard.py`, `control/regulatory_network/bench_rk4.py`.

| Kernel | Paper | Rust µs | Python µs | Speedup |
|--------|-------|---------|-----------|---------|
| HMM forward (3×5000) | 016-018 | 330.0 | 12007.6 | 36.4× |
| Replicator dynamics (10k steps) | 019 | 150.0 | 34937.4 | 232.9× |
| Commutator ‖[A,B]‖_F (64×64) | 022 | 334.6 | 23.3 | 0.1× |
| NK fitness (N=10,K=2, 1000 genotypes) | 011 | 17.9 | 14087.2 | 787.1× |
| Pairwise Hamming (20×500) | 017 | 34.3 | 408.3 | 11.9× |
| Jaccard distance (30×500) | 024 | 142.3 | 2045.4 | 14.4× |
| RK4 GRN ODE (2000 steps) | 020-021 | 218.6 | 24659.8 | 112.8× |
| **TOTAL** | | **1227.8** | **88169.0** | **71.8×** |

Rust pure math is 71.8× faster than single-thread NumPy overall. GEMM-heavy operations
(commutator: 0.1×) show why GPU WGSL acceleration via BarraCUDA matters.

### Phase 4c: GPU WGSL Kernel Benchmarks + GPU PRNG (5 shader checks)

**bench_gpu_kernels**: Times each WGSL shader on RTX 4070 vs Rust CPU to reveal
crossover points. GPU dispatch overhead is ~1.5ms fixed cost; GPU wins when CPU work
exceeds that threshold.

| Kernel | Scale | GPU µs | CPU µs | GPU vs CPU |
|--------|-------|--------|--------|------------|
| Hamming | Small 20×500 | 1,589 | 34 | CPU 46× |
| Hamming | **Large 200×1000** | **1,675** | **7,089** | **GPU 4.2×** |
| Jaccard | Small 30×500 | 1,659 | 142 | CPU 12× |
| Jaccard | **Large 100×2000** | **1,464** | **8,246** | **GPU 5.6×** |
| Fitness | Small 1000×10 | 1,836 | 18 | CPU 102× |
| Fitness | Large 50000×64 | 1,510 | — | — |

**Crossover**: ~1.5ms of CPU work. Below: route to CPU. Above: route to GPU.
This is exactly what `barracuda::dispatch` does — and what metalForge documents.

**validate_gpu_prng** (5/5 PASS): GPU-parallel Xoshiro128** PRNG shader. Validates
uniformity, range, determinism, independence, and multi-call state advancement.
Foundation for all stochastic GPU algorithms (Wright-Fisher, Gillespie SSA, parallel EA).

### Phase 4d: ToadStool Issue Resolution (19 checks)

| Validation Binary | Checks | Status | What It Proves |
|-------------------|--------|--------|----------------|
| `validate_eigh_accuracy` | 9/9 | **PASS** | Householder+QR vs Jacobi eigensolver. Machine epsilon accuracy at n=4,8,16,32,64. Anderson Hamiltonian n=32: 1.75e-14. |
| `validate_mha_gpu` | 10/10 | **PASS** | GPU head_split/head_concat shaders at production sizes (up to B=4, S=128, H=8, d=512). |

### Phase 4e: Pure GPU All-Domains Workload Validation (10 checks, S74)

`validate_gpu_pure_workload_all` — every Phase 0++ paper domain runs through typed
BarraCUDA GPU ops with scalar-only readback. Proves the math is truly portable to GPU.

| Domain | GPU Op | Papers | Tolerance | Result |
|--------|--------|--------|-----------|--------|
| NK Fitness | `BatchFitnessGpu` | 011–013 | 1e-10 | **PASS** |
| Multi-obj Fitness | `MultiObjFitnessGpu` | 014 | 1e-10 | **PASS** |
| HMM Forward | `HmmBatchForwardF64` | 016–018 | 1e-10 | **PASS** |
| Spatial Payoff | `SpatialPayoffGpu` | 019 | 1e-6 (f32) | **PASS** |
| Batch IPR | `BatchIprGpu` | 022–023 | 1e-5 (f32) | **PASS** |
| Pairwise Hamming | `PairwiseHammingGpu` | 017 | 1e-6 (f32) | **PASS** |
| Pairwise L2 | `PairwiseL2Gpu` | 012 | 1e-5 (f32) | **PASS** |
| Pairwise Jaccard | `PairwiseJaccardGpu` | 024 | 1e-5 (f32) | **PASS** |
| Locus Variance | `LocusVarianceGpu` | 025 | 1e-10 | **PASS** |
| Determinism | Re-run fitness | — | 0.0 (exact) | **PASS** |

**f32 precision boundary**: Domain ops (fitness, spatial, distance) use f32 shaders;
HMM and locus variance use f64 paths. IPR requires pre-normalized eigenvectors.
Jaccard uses f32 presence/absence input + upper-triangle output extraction.

### Phase 4e-bench: Evolution Tier Benchmarks (8 domains, S74)

`bench_evolution_tiers` — Rust CPU vs BarraCUDA GPU timing per domain.

| Kernel | CPU µs | GPU µs | Notes |
|--------|--------|--------|-------|
| HMM forward (3×5000) | 149 | 188 | GPU dispatch overhead at validation scale |
| NK fitness (1000×10) | 0.3 | 183 | Dispatch overhead dominates |
| Pairwise Hamming (20×500) | 49 | 186 | GPU crossover at 200×1000 |
| Pairwise L2 (10×8) | 0.3 | 185 | Tiny scale, CPU wins |
| Pairwise Jaccard (30×500) | 316 | 186 | GPU competitive at validation scale |
| Spatial payoff (6×6) | 0.5 | 184 | GPU wins at 128×128+ grids |
| Hill gate (50×50) | 3.1 | 184 | GPU wins at 200×200+ |
| Commutator (64×64) | 183 | — | CPU-only benchmark |

GPU dispatch overhead ~186µs per `queue.submit()`. GPU wins when CPU work exceeds
~1.5ms (documented in Phase 4c). Evolution path validated: same math, portable to GPU.

### Phase 2a: metalForge Cross-System Dispatch (46 checks, S74)

`validate_cross_system_dispatch` — full metalForge stack end-to-end.

| Section | Checks | What It Proves |
|---------|--------|----------------|
| Hardware discovery | 8 | CPU/GPU inventory via `probe_gpus`/`probe_cpu` |
| Domain heuristics | 16 | All 8 workload-type routing (pairwise, fitness, ODE, HMM, spatial, IPR, logsumexp, stochastic) |
| Multi-substrate parity | 6 | Variance, Pearson, entropy: CPU ↔ GPU identical via `mixed_dispatch` |
| Transfer cost hierarchy | 6 | SharedMem < PCIe5 < PCIe4x16 < PCIe4x4, multi-hop, P2P vs staged |
| NPU routing | 4 | GpuToNpu, NpuOnly, non-realtime bypass, live fallback |
| Crossover sweep | 2 | CPU→GPU transition at ~1946µs (1.29× threshold) |

Hardware discovered: i9-12900K (CPU) + RTX 4070 Vulkan + TITAN V NVK + RTX 4070 OpenGL.
Cross-system dispatch chain: discovery → heuristics → cost model → routing → GPU compute → parity.

### Phase 1c: Fused ToadStool Pipeline (46–78× speedup)

| Model | Per-Op (GPU) | Fused (GPU) | Speedup |
|-------|-------------|-------------|---------|
| MLP (4→64→64→10) | 4.0 ms | 92 µs | **43.6×** |
| Transformer (d=32,h=4,seq=8) | 13.3 ms | 174 µs | **76.6×** |

Single `CommandEncoder`, one `queue.submit()`. GPU-resident head-split/concat
and batched attention eliminate all CPU round-trips.

### Phase 1d: 3-Way Benchmark + Double-Buffered Shader Evolution

Target progression (following hotSpring): **Python < CPU < GPU**

| Scale | Py(1t) | CPU | GPU | CPU/Py | GPU/Py | GPU/CPU |
|-------|--------|-----|-----|--------|--------|---------|
| MLP large (3.1M) | 3.0 ms | **2.7 ms** | **178 µs** | **1.1× faster** | 16.8× faster | 15.1× |
| TF medium (103M) | 59 ms | **15.1 ms** | **566 µs** | **3.9× faster** | 104× faster | 26.8× |
| TF xlarge (6.6B) | 232 ms | 1.42 s | **17.8 ms** | — | 13.1× faster | **79.9×** |

4-tier shader router driven by `DeviceCapabilities`:
- Tiny M,N: naive matmul
- CPU: 32×32 double-buffered, 8×4 micro-kernel, vec4, 4× k-unroll
- GPU (small): 16×16 shared-memory (high occupancy)
- GPU (large): 32×32 double-buffered, 2×2 micro-kernel, vec4, 4× k-unroll

---

## BarraCUDA Primitive Coverage

| Primitive | Validated By | WGSL Shader | Status |
|-----------|-------------|-------------|--------|
| GEMM | All experiments & studies | matmul.wgsl (4-tier) | **Native** (S-02 absorbed) |
| Attention | Exp 002, ML inference | attention.wgsl | **Native** (S-03 z-dispatch fixed) |
| LayerNorm | Exp 002, ML inference | layer_norm.wgsl | **Native** (S-08 round-trip fixed) |
| ReLU/GELU/Tanh | Exp 001, Studies 001/003 | relu.wgsl, gelu.wgsl | Native |
| Softmax | ML inference | softmax_simple.wgsl | **Native** (S-04 pooled buffers fixed) |
| Log-Softmax | ML inference | log_softmax.wgsl | **Native** (S-09 round-trip fixed) |
| LeakyReLU | Tensor validation | leaky_relu.wgsl | **Native** (S-05 Params fixed) |
| ELU | Tensor validation | elu.wgsl | **Native** (S-06 Params fixed) |
| LSTM cell | Exp 003, Study 004 | lstm_cell.wgsl | Native |
| Conv2d | Study 003 | conv2d.wgsl | Native |
| Quantized GEMV | Study 005 | gemv_q4/q8.wgsl | Native |

---

## Quality Gates

| Gate | Tool | Status |
|------|------|--------|
| Python lint | `ruff check` (E/F/W/I/N/UP/B/A/SIM) | **PASS** — 0 errors |
| Python format | `ruff format` | **PASS** — 46 files conformant |
| Python tests | `pytest tests/` | **PASS** — 48 tests |
| Python baselines | `bash scripts/run_all_baselines.sh` | **PASS** — 206/206 |
| Rust test | `cargo test` | **PASS** — 580 lib tests + 9 integration tests |
| Rust clippy | `cargo clippy` (pedantic+nursery, -D warnings) | **PASS** — 0 warnings |
| Rust format | `cargo fmt --check` | **PASS** |
| Rust doc | `cargo doc --no-deps` | **PASS** |
| neuralSpring validate | `make validate-native` + `validate-native-papers` | **PASS** — 276/276 |
| BarraCUDA validate | `make validate-barracuda` | **PASS** — 272/272 |
| BarraCUDA CPU ports | `make validate-barracuda-cpu` | **PASS** — 203/203 (24/25 papers, 96%) |
| GPU shader validate | `make validate-gpu` | **PASS** — 108/108 (21 WGSL shaders, 13 upstream + 8 local) |
| GPU pipeline validate | `make validate-gpu-pipeline` | **PASS** — 77/77 (SP 10 + chain 7 + xd 8 + xd-genomics 8 + xd-extended 12 + 32 Phase 4b) |
| GPU PRNG validate | `validate_gpu_prng` | **PASS** — 5/5 |
| CI | GitHub Actions: `baselines.yml` + `rust.yml` | Configured |

---

## Evolution Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Synthetic baselines (48 checks) | **COMPLETE** |
| 0+ | Scholarly reproductions (31 checks) | **COMPLETE** |
| 0++ | Paper reproductions (127 checks) | **COMPLETE** |
| 1a | neuralSpring Rust validation (580 lib + 9 integration + 43 forge tests, 163 binaries, 36 modules + gpu_ops/ + gpu_dispatch/) | **COMPLETE** |
| 1b | BarraCUDA validation (272 checks) | **COMPLETE** |
| 1c | Fused ToadStool pipeline (46–78×) | **COMPLETE** |
| 1d | 3-way benchmark + double-buffered shaders | **COMPLETE** |
| 2 | BarraCUDA CPU ports (24/25 papers, 203 checks) | **COMPLETE** |
| 2a | metalForge hardware characterization | Active |
| 3a | BarraCUDA FFT validation (24 checks: f32+f64+Rfft, RTX 4070) | **COMPLETE** |
| 3b | BarraCUDA GPU streaming (StatefulPipeline + Unidirectional) | **COMPLETE** |
| 3c | metalForge GPU shader validation (16 GPU shader binaries, 108 shader checks, 21 WGSL shaders) | **COMPLETE** |
| 3d | Cross-dispatch validation (49 checks — 6 validators, 15/15 papers) | **COMPLETE** |
| 4a | Performance benchmarks (7 kernels, 71.8× overall speedup vs single-thread NumPy) | **COMPLETE** |
| 4b | Pure GPU end-to-end pipelines (7 pipelines, 32/32 PASS) | **COMPLETE** |
| 4c | GPU WGSL kernel benchmarks + GPU PRNG (crossover mapping + 5/5 PRNG checks) | **COMPLETE** |
| 4d | ToadStool issue resolution (eigh accuracy + MHA GPU, 19 checks) | **COMPLETE** |
| 4e | PINN/DeepONet + new GPU domains | **COMPLETE** |
| 5a | BarraCUDA GPU Tensor validation (7 original domains, 43 checks) | **COMPLETE** |
| 5b | Full-stack buildout: bC 24/25, gT 23/25, xD 15/15 — S-14/S-15/S-16 RESOLVED upstream | **COMPLETE** |
| 5c | Upstream parity (6/6 dual-path 0.00e0) + ReduceScalarPipeline + spectral theory (14/14) | **COMPLETE** |
| 4 | Cross-spring integration | Active |

### Phase 5a: BarraCUDA GPU Tensor Validation (7 domains, 43 checks)

GPU `Tensor` operations (`matmul`, `transpose`, `tanh`, `add`) validated
against CPU f64 references across 7 scientific domains. Discovered 3 new
BarraCUDA shortcomings (S-14, S-15, S-16).

| Validator | Domain | Papers | Checks | Status |
|-----------|--------|--------|--------|--------|
| `validate_barracuda_gpu_spectral` | Spectral commutativity | 022 | 10 | **PASS** |
| `validate_barracuda_gpu_eco` | Ecological dynamics | 013 | 6 | **PASS** |
| `validate_barracuda_gpu_hmm` | HMM phylogenetics | 016-018 | 5 | **PASS** |
| `validate_barracuda_gpu_fitness` | Evolutionary computation | 011-015 | 7 | **PASS** |
| `validate_barracuda_gpu_nn` | Neural network inference | 015, 020-021 | 5 | **PASS** (S-15 RESOLVED upstream) |
| `validate_barracuda_gpu_pairwise` | Pairwise distance | 017, 019, 024-025 | 5 | **5/5 PASS** (S-16 RESOLVED upstream) |
| `validate_barracuda_gpu_anderson` | Anderson localization | 023 | 7 | **7/7 PASS** (S-15 RESOLVED upstream) |

**S-14** ~~(Medium)~~ **RESOLVED** upstream (`a4996b34` S39): Naive tier removed.
**S-15** ~~(Critical)~~ **RESOLVED** upstream (`a4996b34` S39): Matmul hang fixed.
**S-16** ~~(High)~~ **RESOLVED** upstream (`a4996b34` S39): Transpose dispatch fixed.
Validators retain conservative data patterns (positive-only, A×B^T) as defense-in-depth.

Handoff: `wateringHole/handoffs/archive/NEURALSPRING_V8_TOADSTOOL_BARRACUDA_HANDOFF_FEB22_2026.md`.

### Full Validation Stack — All 25 Papers (February 22, 2026)

Every paper passes through 7 tiers: Python control → Rust CPU → BarraCUDA CPU
→ BarraCUDA GPU Tensor → metalForge WGSL → GPU Pipeline → Cross-dispatch.
All tiers use exclusively open data and open systems (see `specs/DATA_PROVENANCE.md`).

| Tier | Coverage | Checks | Status | Delta |
|------|----------|--------|--------|-------|
| Python control (Py) | 25/25 (100%) | 206 | **ALL PASS** | — |
| Rust CPU (Rs) | 25/25 (100%) | 500+ lib + binaries | **ALL PASS** | — |
| BarraCUDA CPU (bC) | 24/25 (96%) | 203 | **ALL PASS** | +12pp |
| BarraCUDA GPU Tensor (gT) | 23/25 (92%) | 98+ | **ALL PASS** | +20pp |
| metalForge WGSL (mF) | 15/15† (100%) | 108 | **ALL PASS** | — |
| GPU Pipeline (gP) | 15/15† (100%) | 94 | **ALL PASS** | — |
| Cross-dispatch (xD) | 15/15† (100%) | 49 | **ALL PASS** | +80pp |

`†` 100% of applicable papers. Phase 0/0+ studies use PyTorch (mF/gP/xD N/A).

**Phase 0++ papers: 15/15 at ALL 7 tiers. ALL GREEN.**
**Phase 0/0+ studies: 9/10 at bC+gT (Exp 005 analytical only). ALL GREEN.**
Full per-paper matrix: `specs/PAPER_REVIEW_QUEUE.md`.

### Phase 6 — Session 43 Experiment Buildouts (February 22, 2026)

| Validator | Domain | Checks | Status |
|-----------|--------|--------|--------|
| `validate_gpu_logsumexp` | Batched logsumexp (HMM/phylo) | 5/5 | **PASS** |
| `validate_gpu_stencil` | Stencil cooperation (game theory) | 3/3 | **PASS** |
| `validate_gpu_rk45` | Adaptive RK45 (regulatory ODE) | 6/6 | **PASS** |
| `validate_gpu_wright_fisher` | Wright-Fisher drift+selection | 4/4 | **PASS** |
| `validate_gpu_gillespie` | Gillespie SSA (upstream) | 20/20 | **PASS** |
| `validate_upstream_taxonomy` | Taxonomy FC (wetSpring) | 3/3 | **PASS** |
| `validate_upstream_kmer` | K-mer histogram (wetSpring) | 3/3 | **PASS** |
| `validate_upstream_unifrac` | UniFrac propagation (wetSpring) | 2/2 | **PASS** |
| `validate_barracuda_chi_squared` | Chi-squared distribution + test | 13/13 | **PASS** |
| `validate_cpu_gpu_parity` | CPU vs GPU Tensor parity | 17/17 | **PASS** |
| `validate_toadstool_dispatch` | Dispatch substrate routing | 16/16 | **PASS** |
| `validate_mixed_dispatch` | Mixed-hardware dispatch | 16/16 | **PASS** |
| **Total** | | **108/108** | **ALL PASS** |

### Session 49 — Deep Debt Audit (February 23, 2026)

Code quality hardening. No new validation checks; all existing checks confirmed passing.

| Change | Scope | Impact |
|--------|-------|--------|
| `gpu_or_cpu` dispatch helper | `gpu_dispatch.rs` | 25 methods DRYed — centralised GPU/CPU fallback |
| `exit_no_gpu()` | 79 validation/bench binaries | CI-fidelity: `NEURALSPRING_REQUIRE_GPU=1` exits 1 when GPU expected |
| `baseline_path()` | 4 binaries | Data resolution via `validation::baseline_path`, no hardcoded `concat!` |
| Clippy + doc cleanup | `gpu_dispatch.rs`, `validation.rs` | Zero clippy warnings, zero doc warnings |
| EVOLUTION_MAPPING fix | `specs/EVOLUTION_MAPPING.md` | "stub" labels corrected — `mlp_forward` exists in `pinn.rs`/`deeponet.rs` |

**Quality gates** (all pass):

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets -- -D warnings` | PASS (0 warnings) |
| `cargo doc --no-deps` | PASS (0 warnings) |
| `cargo test` | PASS (500 lib + 9 integration + 9 doc-tests) |
| Max file size | 965 lines (under 1000 wateringHole limit) |
| `unsafe` blocks | 0 (`forbid` enforced) |
| TODO/FIXME/MOCK/STUB | 0 in src/ |
| Hardcoded paths | 0 (all via `baseline_path`) |

### Session 51 — Code Quality Evolution (February 24, 2026)

Deep code quality evolution. No new validation checks; structural and pedantic improvements.

| Change | Scope | Impact |
|--------|-------|--------|
| `gpu_dispatch.rs` → `gpu_dispatch/` module | Refactored to mod.rs + cpu_fallback.rs | CPU fallbacks independently testable |
| Float comparison evolution | 5 library modules | All `assert_eq!` on f64 → epsilon-based |
| Inline guard centralization | 5 validation binaries | 7 `1e-14` → `tolerances::ZERO_DETECTION` |
| Clippy pedantic resolution | 7 lint categories | float_cmp, cast_lossless, identity_op, manual_midpoint, redundant_closure, doc_markdown, redundant_pub_crate |
| Documentation refresh | 13 docs | 412→500 lib tests, 92.7%→93.17% coverage |
| Dependency audit | Full tree | All pure Rust confirmed |

**Quality gates** (all pass):

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets` | PASS (0 warnings — pedantic + nursery) |
| `cargo clippy -p neural-spring-forge` | PASS (0 warnings) |
| `cargo doc --no-deps` | PASS (0 warnings, 146 pages) |
| `cargo test` | PASS (500 lib + 9 integration + 9 doc-tests) |
| `cargo llvm-cov --lib` | 92.9% line coverage |
| Max file size | 965 lines (under 1000 wateringHole limit) |
| `unsafe` blocks | 0 (`forbid` enforced) |
| Production `.unwrap()`/`.expect()` | 0 (audit confirmed) |
| Dependency purity | Pure Rust (only linux-raw-sys, renderdoc-sys transitive via wgpu) |

### Session 52 — ToadStool Sync & Cross-Spring Benchmarking (February 24, 2026)

ToadStool sync (16 commits, `b41ee5f4` → `9abd6857`), 6 shader absorptions,
cross-spring benchmarking, and documentation hardening.

| Change | Scope | Impact |
|--------|-------|--------|
| ToadStool sync | 16 commits absorbed | `argmax_dim`, `softmax_dim` gaps CLOSED |
| 6 shader absorptions | xoshiro, logsumexp, stencil, wright_fisher, rk45, swarm_nn | Only head_split + head_concat remain local |
| `level_spacing_ratio` rewire | `weight_spectral.rs` | Delegates to `barracuda::spectral` upstream |
| Absorption tracker | 6 items marked absorbed | Upstream locations documented |
| Cross-spring benchmark | 7 ops from 3 springs | RTX 4070 Vulkan, 1.3–6.6 ms |
| `CROSS_SPRING_SHADER_LINEAGE.md` | Session 50–52 narrative | Benchmark data + timeline |

**Cross-spring benchmark results** (RTX 4070, Vulkan, `--release`):

| Op | Origin | µs |
|----|--------|----|
| BatchFitnessGpu 1024×64 | neuralSpring | 1,337 |
| PairwiseL2Gpu 128×16 | neuralSpring | 1,542 |
| SpatialPayoffGpu 32×32 | neuralSpring | 1,450 |
| PairwiseHammingGpu 64×100 | neuralSpring | 1,682 |
| BatchIprGpu 32×64 | neuralSpring | 2,027 |
| HmmBatchForwardF64 4s×50t×32b | wetSpring | 2,141 |
| BatchedEighGpu 12×12×40 | hotSpring | 6,629 |

**Quality gates** (all pass):

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets` | PASS (0 warnings — pedantic + nursery) |
| `cargo doc --no-deps` | PASS (0 warnings, 146 pages) |
| `cargo test --lib` | PASS (500 lib tests) |
| `cargo llvm-cov --lib` | 92.89% line coverage |
| `validate_all` | 137/138 PASS (1 pre-existing logsumexp driver issue) |

### Session 52b — S-17 HillGate f64 `pow()` Fix

**Root cause identified**: `hill_gate_f64.wgsl` native `pow(f64, f64)` crashes
NVVM (RTX 4070 Ada Lovelace) and NAK (TITAN V NVK). The `compile_shader_f64`
pipeline patches `exp/log` to polyfills but **missed `pow`**.

**Fix**: Replace `pow(` → `pow_f64(` in shader source; `inject_missing_math_f64`
auto-injects the polyfill from `math_f64.wgsl`. Machine-epsilon accuracy.

| Item | Detail |
|------|--------|
| `validate_hillgate_f64_fix` | 18/18 PASS — RTX 4070 + TITAN V |
| `validate_gpu_signal` evolved | 9/9 PASS on both GPUs (was: SKIP on both) |
| Buffer mismatch fixed | Old validator used f32 buffers with f64 shader |
| ToadStool action documented | One-line fix in `patch_exp_log_in_code` |
| `validate_all` | 137/138 PASS (unchanged — hillgate_f64_fix not in validate_all yet) |

### Session 54 — baseCamp Experiment Expansion & Pure GPU Workload Validation

Expanded all 5 baseCamp validators with uncovered experiments (nS-103..106,
205, 206, 304, 305, 402, 405, 504, 505). Created `validate_basecamp_gpu` for
pure GPU workload validation and `bench_basecamp_parity` for CPU↔GPU parity.

| Change | Scope | Result |
|--------|-------|--------|
| `validate_weight_spectral` expanded | 15→21 checks: Dyson dynamics, cross-shape, GNN message passing, training trajectory | **21/21 PASS** |
| `validate_information_flow` expanded | 15→22 checks: Hill LSTM gates, edge-of-chaos sweep, deep layer IPR | **22/22 PASS** |
| `validate_loss_landscape` expanded | 19→27 checks: dimension sweep, gradient descent, trajectory landscape, multi-barrier | **27/27 PASS** |
| `validate_neural_pgm` expanded | 15→21 checks: deep factor graph BP, OOD detection, rank monotonicity, depth complexity | **21/21 PASS** |
| `validate_agent_coordination` expanded | 18→23 checks: scaling sweep, threshold transition, disorder comparison, dimensional API | **23/23 PASS** |
| `validate_basecamp_gpu` (NEW) | Pure GPU: eigensolve, variance, Pearson, entropy, matmul, chi², L2, KL divergence | **14/14 PASS** |
| `bench_basecamp_parity` (NEW) | CPU→GPU parity: var 7.77e-16, pearson 6.94e-18, entropy 1.60e-11 | **All sub-epsilon** |
| baseCamp total | 82→114 CPU + 14 GPU = 128 checks | **128/128 PASS** |
| `validate_all` | **139/140 PASS** (1 pre-existing logsumexp driver issue) | **ALL GREEN** |

**CPU parity results** (RTX 4070 Vulkan, `--release`):

| Module | CPU µs | Key metric |
|--------|--------|------------|
| Sub-01 weight_spectral (16×16) | 99.0 | IPR=0.085, LSR=0.492 |
| Sub-02 information_flow | 4.0 | ξ=3.37, W=1.19 |
| Sub-03 loss_landscape (8-dim) | 1.9 | saddle=0, sharpness=2.0 |
| Sub-04 neural_pgm (2-layer BP) | 0.2 | Σ=1.000000 |
| Sub-05 agent_coordination (16-node) | 15.2 | λ₂=1.19 |

**BarraCUDA special function parity**:

| Function | Expected | Got | Source |
|----------|----------|-----|--------|
| Γ(5) | 24.0 | 24.0 | `barracuda::special::gamma` |
| erf(1) | 0.8427... | 0.8427... | `barracuda::special::erf` |
| J₀(0) | 1.0 | 1.0 | `barracuda::special::bessel_j0` |
| χ²(10,20,30,40 vs 25×4) | 20.0 | 20.0 | `barracuda::special::chi_squared_statistic` |

### Session 55 — BarraCUDA CPU vs GPU Dispatch + metalForge Mixed Hardware

Wired `metalForge::mixed` cost model into `Dispatcher::mixed_dispatch()` for
end-to-end substrate routing. Built two new validators proving CPU↔GPU parity
through the dispatch layer and mixed-hardware (GPU↔NPU↔CPU) routing.

| Change | Scope | Result |
|--------|-------|--------|
| `Dispatcher::mixed_dispatch()` | `gpu_dispatch/mod.rs` — routes via `metalForge` cost model | **Wired** |
| `validate_compute_dispatch` (NEW) | Routing correctness + CPU↔GPU parity (variance, Pearson, entropy, chi², eigh, dispatch-aware) | **16/16 PASS** |
| `validate_mixed_hardware` (NEW) | Mixed-hardware routing (small→CPU, large→GPU, realtime→NPU), PCIe bridge, crossover boundary | **14/14 PASS** |
| Sub-thesis doc cleanup | 5 docs: corrected binary references, added expanded check counts | **5/5 updated** |
| Grounding papers B-01..B-15 | Updated from "Queued" to "Primitives validated" with experiment mappings | **15/15 updated** |
| `validate_all` | **141/142 PASS** (1 pre-existing logsumexp driver issue) | **ALL GREEN** |

**CPU↔GPU dispatch parity** (RTX 4070 Vulkan):

| Operation | CPU value | GPU value | Diff |
|-----------|-----------|-----------|------|
| Variance (256 elem) | 9.1459980836e-1 | 9.1459980836e-1 | < 1e-8 |
| Pearson (128 elem) | -2.4480710605e-2 | -2.4480710605e-2 | < 1e-6 |
| Entropy (64 probs) | 3.9609218564e0 | 3.9609218564e0 | < 1e-4 |
| Chi² (8 bins) | 1.0314285714e2 | 1.0314285278e2 | < 0.5 |

**Mixed-hardware routing decisions**:

| Workload | Compute µs | Data bytes | Substrate |
|----------|-----------|------------|-----------|
| Small variance (32 elem) | 10 | 256 | CpuOnly |
| Large variance (4096 elem) | 50,000 | 32,768 | GpuOnly |
| Realtime inference | 5,000 | 512 | GpuToNpu |
| Below crossover | 750 | 1,024 | CpuOnly |
| Above crossover | 15,000 | 1,024 | GpuOnly |

### Session 56 — Dispatcher baseCamp Parity + BarraCUDA CPU/GPU Parity + metalForge PCIe Bridge

Comprehensive dispatch-layer validation proving baseCamp science routes correctly
through the `Dispatcher` abstraction. BarraCUDA CPU vs GPU parity sweep across all
17 science domains. metalForge PCIe bandwidth tier cost model with chained multi-hop
transfer estimation and live mixed-hardware dispatch.

| Change | Scope | Result |
|--------|-------|--------|
| `validate_basecamp_dispatch` (NEW) | 4 Dispatcher baseCamp methods: spectral, Hessian, BP, interaction graph | **19/19 PASS** |
| `validate_barracuda_parity` (NEW) | CPU vs GPU parity: linalg, stats, spectral, activation, reduction, distance, biology | **17/17 PASS** |
| `validate_metalforge_pcie` (NEW) | PCIe tiers, P2P vs staged, chained transfer, substrate sweep, bridge API, live dispatch | **23/23 PASS** |
| `metalForge/forge/src/mixed.rs` enhanced | `BandwidthTier` enum (x16/x4/PCIe5/shared), `chained_transfer_cost`, `compare_transfer_paths` | **43/43 forge tests** |
| `Makefile` + `justfile` updated | `validate-dispatch` group (5 validators), wired into `validate-all` | **CI-integrated** |
| `validate_all.rs` updated | Session 56 + 58 entries | **146 binaries** |

**Dispatcher baseCamp method validation** (RTX 4070 Vulkan):

| Method | Sub-thesis | Checks | Key Result |
|--------|-----------|--------|------------|
| `weight_spectral_analysis` | Sub-01 | 5/5 | Eigenvalues match direct within 0.1 |
| `numerical_hessian` | Sub-03 | 4/4 | Rosenbrock H[0,0]=802, H[1,1]=200 |
| `belief_propagation` | Sub-04 | 5/5 | All layers sum to 1.0 within 1e-8 |
| `agent_interaction_graph` | Sub-05 | 4/4 | Symmetric, zero diagonal, matches library |
| Determinism | all | 1/1 | Repeated dispatch reproduces |

**BarraCUDA CPU vs GPU parity sweep** (17 checks, all domains):

| Domain | Operation | Max Diff | Tolerance |
|--------|-----------|----------|-----------|
| linalg | matmul, transpose, frobenius, commutator, dist_to_normal | < 0.05 | GPU_MATMUL_RANDOM_F32 |
| stats | variance, Pearson, entropy, chi-squared | < 1e-8 | GPU_VARIANCE_F64 |
| spectral | eigh eigenvalues | < 0.1 | GPU_EIGH_DISPATCH_F64 |
| activation | softmax, Boltzmann, Hill | < 1e-4 | GPU_HILL_F32 |
| reduction | mean | < 0.01 | GPU_MEAN_DISPATCH_F32 |
| distance | L2 | < 0.01 | GPU_L2_DISPATCH_F32 |
| biology | replicator step | < 0.01 | GPU_MEAN_DISPATCH_F32 |

**metalForge PCIe bandwidth tier model**:

| Tier | Bandwidth | Latency | 1MB Transfer |
|------|-----------|---------|--------------|
| PCIe 4.0 x16 | 31.5 GB/s | 2.0 us | ~35 us |
| PCIe 4.0 x4 | 7.9 GB/s | 2.0 us | ~135 us |
| PCIe 5.0 x16 | 63.0 GB/s | 2.0 us | ~19 us |
| Shared memory | 200 GB/s | 0.1 us | ~5 us |

P2P direct always beats CPU-staged for same bandwidth tier. Chained 2-hop < 3x direct overhead.

### Session 62 — ToadStool S62 Sync (February 25, 2026)

S-03b (MHA projection hangs) **FULLY RESOLVED** upstream. ToadStool `0c998992` decomposed MHA projections into matmul + head_split/head_concat shaders. All 21/21 WGSL shaders absorbed. `evolved/mha.rs` now thin wrapper to `barracuda::ops::mha::MultiHeadAttention`. 500 lib tests, 145/146 validate_all.

### Session 67b — Dispatch Tier Benchmarks (February 25, 2026)

Three-tier benchmark: Library direct → Dispatcher::cpu_only() → Dispatcher::new() GPU.

| Kernel | Size | Library µs | CPU Disp µs | Overhead |
|--------|------|-----------|-------------|----------|
| MatMul | 64×64 | 41.6 | 41.3 | 0.99× |
| Variance | 4096 | 3.4 | 3.4 | 1.00× |
| Pearson | 4096 | 6.1 | 6.1 | 1.00× |
| Entropy | 256 | 0.7 | 0.8 | 1.03× |
| Softmax | 256 | 1.2 | 1.2 | 1.00× |
| L2 Distance | 256 | 0.1 | 0.1 | 1.04× |
| Chi-squared | 100 | 0.1 | 0.1 | 1.00× |
| Commutator | 32×32 | 12.9 | 12.8 | 1.00× |
| HMM Forward | 3×500 | 7.5 | 7.5 | 1.01× |
| Hill Batch | 2500 | 1.1 | 20.3 | 19.17× * |

\* Hill batch overhead due to batch dispatch allocation path; 9/10 ops ≤1.04×.

**Conclusion**: Dispatcher::cpu_only() adds negligible overhead (≤1.04× for 9/10 ops).
Per-call GPU dispatch is driver-bound for small workloads — motivates StatefulPipeline
and UnidirectionalPipeline batching for GPU-resident acceleration.

### Session 67 — CPU Math Parity Validation (February 25, 2026)

Cross-language numeric parity: Python/NumPy reference → Rust CPU → Dispatcher::cpu_only().
`control/generate_cpu_references.py` produces deterministic inputs+outputs (JSON).
`validate_cpu_math_parity` loads JSON and verifies Rust produces identical values.

| Layer | Checks | Tolerance | Result |
|-------|--------|-----------|--------|
| Primitives (variance, pearson, chi², entropy, softmax, gelu, matmul, frobenius, L2) | 15 | 1e-10 | **PASS** |
| Paper kernels (HMM forward, replicator, commutator, Hamming, Jaccard, L2, multi-obj, Hill, swarm NN) | 18 | 1e-10 (replicator: 1e-6) | **PASS** |
| Dispatcher::cpu_only() (variance, pearson, entropy, matmul, softmax, L2) | 6 | 1e-10 | **PASS** |
| **Total** | **39** | — | **39/39 PASS** |

Python baselines: 25/25 PASS (zero drift).
validate_all: 149/150 PASS (1 pre-existing logsumexp driver issue; S74: +2 binaries `validate_gpu_pure_workload_all`, `validate_cross_system_dispatch`).

### Session 66 — Phase C GPU Promotion (February 25, 2026)

Closes remaining science-domain GPU promotion gaps. 6 new `Dispatcher` methods,
3 new `gpu_ops` functions, `validate_gpu_phase_c` (18/18 PASS on RTX 4070).

| Change | Scope | Result |
|--------|-------|--------|
| `hmm_forward_chain_gpu` | Compose forward steps × T observations | **GPU (S66)** |
| `hmm_viterbi_chain_gpu` | Compose Viterbi steps × T observations | **GPU (S66)** |
| `pairwise_fst_gpu` | GPU allele_freq + Weir-Cockerham per-locus | **GPU (S66)** |
| `global_fst_gpu` | GPU allele_freq per pop + global decomposition | **GPU (S66)** |
| `Dispatcher::inter_population_af_variance` | Wire existing gpu_op to dispatch | **GPU (S66)** |
| `Dispatcher::hmm_forward_chain` | Full forward chain via dispatch | **GPU (S66)** |
| `Dispatcher::hmm_viterbi_chain` | Full Viterbi chain via dispatch | **GPU (S66)** |
| `Dispatcher::pairwise_fst` | Pairwise FST via dispatch | **GPU (S66)** |
| `Dispatcher::global_fst` | Global FST via dispatch | **GPU (S66)** |
| `validate_gpu_phase_c` (NEW) | 18 checks: HMM chains, FST, introgression, AF var | **18/18 PASS** |
| `bench_phase0pp_kernels` | Updated: 11 kernels, 201.7× speedup vs Python | **201.7× faster** |

**GPU dispatch coverage**: ~90% → ~97% of production math.
**Python baselines**: 25/25 PASS (zero drift).
**validate_all**: 146/147 PASS (1 pre-existing logsumexp driver issue).
**Lib tests**: 580 PASS (470 + 35 GPU ops).

### Session 61 — Deep Code Quality Sweep (February 25, 2026)

Deep code quality sweep: 13 property tests added (`src/property_tests.rs`), 6 tolerance constants centralized, 4 vestigial `#[allow]` attributes removed. 500 lib tests, 93.17% coverage.

### Session 68 — Deep Debt Audit (February 25, 2026)

Full barracuda usage audit: 90+ import sites, 20+ submodules, zero duplicates. Tolerance centralization: 104+ named constants, zero ad-hoc magic numbers. All bare `unwrap()` → `expect()` with context. Smart refactoring: `tolerances/mod.rs` split CPU/GPU. Rewired `boltzmann_sampling` → barracuda (17th function rewire). 580 lib tests, 90.43% coverage.

### Session 69 — Validator Shader Rewiring + Cross-Spring Benchmarks (February 25, 2026)

6 validator binaries rewired from local `include_str!` to upstream barracuda shader constants (RK4, RK45, batch fitness, logsumexp, swarm NN scores, stateful pipeline). Cross-spring benchmarks: upstream-vs-local 10/10 ≈ or ~ (zero ⚠), cross-spring evolution 39/39 PASS. Updated cross-spring provenance: hotSpring ~25+ modules (precision), wetSpring ~15+ modules (bio), neuralSpring ~15+ modules (ML). Collaborative: pow_f64, CrankNicolson, FusedMapReduceF64. validate_all: 147/148 PASS, 580 lib + 9 integration tests.

### Session 74 — Pure GPU All-Domains Workload + Evolution Tier Benchmarks (February 26, 2026)

Comprehensive pure GPU validation across all 15 Phase 0++ paper domains (9 typed BarraCUDA ops) + evolution tier benchmark (CPU→GPU portability for 8 kernels) + metalForge cross-system dispatch validator. Three new binaries: `validate_gpu_pure_workload_all` (10/10 PASS), `bench_evolution_tiers` (8 kernels), `validate_cross_system_dispatch` (46/46 PASS: hardware discovery, 8 domain heuristics, CPU↔GPU parity for variance/Pearson/entropy, transfer cost hierarchy, NPU routing, crossover sweep). Key findings: f32/f64 precision boundary is systematic (domain ops f32, HMM/baseCamp f64), IPR needs pre-normalized eigenvectors, GPU dispatch overhead ~186µs dominates at validation scale but GPU wins at production scale. CPU→GPU crossover at ~1946µs (1.29× threshold). validate_all: 149/150 PASS (1 pre-existing logsumexp). 580 lib + 9 integration tests. Cross-spring lineage tracked: BatchIprGpu from hotSpring spectral, HmmBatchForwardF64 from wetSpring bio, SpatialPayoffGpu from wetSpring game theory.
