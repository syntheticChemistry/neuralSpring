# neuralSpring — Control Experiment Status

**Last updated**: February 23, 2026 (Sessions 44–49 — multi-GPU + benchmarks + pure GPU promotion + deep audit)
**Gate**: Eastgate (i9-12900K, 32 GB DDR5, RTX 4070 12 GB + TITAN V 12 GB NVK, Pop!_OS 22.04)
**Python**: 3.10.12, PyTorch 2.9.0+cu128, NumPy 2.2.6, SciPy 1.15.3
**Rust**: Edition 2021, clippy pedantic + nursery, unsafe_code=forbid, 92.7% line coverage
**Grand Total**: 206/206 Python PASS + 1600+ Rust+GPU validation PASS = **1800+ total validation checks**
**Library**: 374 lib tests + 9 integration tests | 31 modules + 2 evolved + gpu_ops/ + gpu_dispatch | 133 validation/bench binaries
**Multi-GPU**: 133/133 PASS on RTX 4070 (Vulkan) + 143+ on TITAN V (NVK) — **bit-identical**
**GPU Promotion**: 38 CPU→GPU ops via `gpu_dispatch::Dispatcher` (~90% of production math)
**Benchmarks**: Pure Rust **178.5× faster** than Python/NumPy (11 kernels)
**ToadStool**: All 12 shortcomings (S-01..S-12) **ABSORBED** | S-16 FIXED | S-14/S-15 workaround | 2 new fixes (Session 44)
**Open Data**: All 25 papers use open data and open systems — zero proprietary or paywalled sources

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

### Phase 1a: neuralSpring-Native Validation (374 lib tests + 9 integration tests, 133 validation binaries, 31 modules + 2 evolved + gpu_ops/ + gpu_dispatch)

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
| Rust test | `cargo test` | **PASS** — 374 lib tests + 9 integration tests |
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
| 1a | neuralSpring Rust validation (374 lib tests + 9 integration tests, 133 binaries, 31 modules + 2 evolved + gpu_ops/ + gpu_dispatch) | **COMPLETE** |
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
| 5b | Full-stack buildout: bC 24/25, gT 23/25, xD 15/15 — S-16 fixed, S-15 root-caused | **COMPLETE** |
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
| `validate_barracuda_gpu_nn` | Neural network inference | 015, 020-021 | 5 | **PASS** (S-15 workaround) |
| `validate_barracuda_gpu_pairwise` | Pairwise distance | 017, 019, 024-025 | 5 | **5/5 PASS** (S-16 fixed) |
| `validate_barracuda_gpu_anderson` | Anderson localization | 023 | 7 | **7/7 PASS** (S-15 workaround) |

**S-14** (Medium): Naive matmul hang on small square matrices in complex binaries.
Workaround: A × B^T pattern.
**S-15** (Critical): `Tensor::matmul` hangs when many elements have magnitude ≤ 0.1
(RTX 4070 Vulkan driver bug). Root-caused via diagnostic: not exact zeros but small
magnitudes trigger the hang across all matmul tiers. Workaround: dense data ≥ 0.5.
**S-16** ~~(High)~~ **FIXED**: transpose dispatch used `optimal_workgroup_size(256)` instead of
shader's `@workgroup_size(16,16)`. One-line fix: `const TILE: u32 = 16`.

Handoff: `wateringHole/handoffs/NEURALSPRING_V8_TOADSTOOL_BARRACUDA_HANDOFF_FEB22_2026.md`.

### Full Validation Stack — All 25 Papers (February 22, 2026)

Every paper passes through 7 tiers: Python control → Rust CPU → BarraCUDA CPU
→ BarraCUDA GPU Tensor → metalForge WGSL → GPU Pipeline → Cross-dispatch.
All tiers use exclusively open data and open systems (see `specs/DATA_PROVENANCE.md`).

| Tier | Coverage | Checks | Status | Delta |
|------|----------|--------|--------|-------|
| Python control (Py) | 25/25 (100%) | 206 | **ALL PASS** | — |
| Rust CPU (Rs) | 25/25 (100%) | 374+ lib + binaries | **ALL PASS** | — |
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
