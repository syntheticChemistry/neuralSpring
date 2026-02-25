# neuralSpring — BarraCUDA Requirements

**Last Updated**: February 23, 2026 (Session 44 — multi-GPU + benchmarks)
**Purpose**: GPU kernel requirements, gap analysis, and evolution priorities

---

## Current Primitive Coverage (Validated in Phase 0/0+)

### The 6 Isomorphic Primitives

| # | Primitive | BarraCUDA Shader | Validated By | Status |
|---|-----------|-----------------|-------------|--------|
| 1 | GEMM | `gemm_f64.wgsl` | Exp 001 (surrogate), Study 001 (PINN), Study 002 (DeepONet) | Validated |
| 2 | Attention | `attention.wgsl` | Exp 002 (transformer), Study 002 (DeepONet branch-trunk) | Validated |
| 3 | Normalization | `layer_norm.wgsl`, `batch_norm.wgsl`, `rmsnorm.wgsl` | Exp 002 (transformer LayerNorm) | Validated |
| 4 | Nonlinearity | `nn::ReLU`, activation shaders | Exp 001 (ReLU), Exp 002 (GELU) | Validated |
| 5 | Reduction | `FusedMapReduceF64` | Exp 005 (isomorphic catalog) | Validated |
| 6 | Gating | `lstm_cell.wgsl` | Exp 003 (LSTM), Study 004 (ERA5 LSTM) | Validated |

### Extended Primitives (Phase 0+)

| Primitive | BarraCUDA Shader | Validated By | Status |
|-----------|-----------------|-------------|--------|
| Conv2d | `conv2d.wgsl` | Study 003 (LeNet-5 MNIST) | Validated |
| MaxPool | pooling shader | Study 003 (LeNet-5 MNIST) | Validated |
| Autograd (backprop) | `fd_gradient_f64.wgsl` | Study 001 (PINN PDE residual) | Validated |
| Quantized GEMV (INT8) | `gemv_q8.wgsl` | Study 005 (quantized inference) | Validated |
| Quantized GEMV (INT4) | `gemv_q4.wgsl`, `dequant_q4.wgsl` | Study 005 (quantized inference) | Validated |
| MSE loss | `mse_loss` | Study 001 (PINN), Study 002 (DeepONet) | Validated |
| Adam optimizer | `nn::Optimizer::Adam` | Study 001 (PINN training) | Validated |

---

## Gaps for Faculty Extension Papers

### Critical (P0) — Required for top-priority papers

| Need | Paper | Why | Effort |
|------|-------|-----|--------|
| **Evolutionary optimization (GA/ES)** | Dolson: Iram et al. 2020 (counterdiabatic), MODES 2019 | Population-level optimization with selection, mutation, crossover. Need parallel fitness evaluation + selection on GPU | Medium — population GEMM + tournament selection |
| **Fitness landscape evaluation** | Dolson: all papers | Parallel evaluation of fitness across large populations. Already have GEMM; need population management | Low — orchestration around existing GEMM |

### Important (P1) — Required for tier 1-2 papers

| Need | Paper | Why | Effort |
|------|-------|-----|--------|
| **HMM Viterbi decoding** | Liu: PhyloNet-HMM 2014 | Forward/backward/Viterbi on state-space models. Matrix chain in log-space — need log-sum-exp shader | Medium |
| **Log-sum-exp** | Liu: HMM, phylogenetics | Numerically stable summation in log-probability space. Fundamental for any probabilistic model on GPU | Low |
| **Gillespie stochastic simulation** | Waters: cooperation dynamics 2018 | Parallel stochastic trajectories. Need GPU PRNG + exponential sampling + event selection | Medium |
| **Game-theoretic payoff matrix** | Waters: Bruger 2018, Mhatre 2020 | Parallel evaluation of strategy payoffs across population. GEMM-based | Low |

### Stretch (P2) — Longer horizon

| Need | Paper | Why | Effort |
|------|-------|-----|--------|
| **MODES metric computation** | Dolson 2019 | Phylogenetic analysis on agent histories — Shannon diversity + lineage metrics over evolutionary time | Medium |
| **Phylogenetic likelihood** | Liu: SATé 2009, cophylogenetics 2023 | Felsenstein pruning on trees. GEMM at each internal node, parallel across trees | High |
| **L-BFGS optimizer** | Raissi 2019 (PINN improvement) | Study 001 used Adam-only (5.1% L2 error). Paper achieves 0.06% with Adam + L-BFGS. Adding L-BFGS closes the gap | Medium |
| **Directed evolution framework** | Dolson 2022 (eLife) | Artificial selection methods for microbial optimization. Connects neuralSpring to wetSpring wet lab | Medium |
| **Lanczos eigensolve** | Kachkovskiy: JAMS 2016, GAFA 2018 | Hessian eigenvalue analysis for understanding loss landscape curvature. Large sparse matrix diagonalization — shared with hotSpring/groundSpring | Medium |
| **Sparse matrix-vector product** | Kachkovskiy (all) | Inner-loop of Lanczos. Required for sparse attention, Hessian-vector products, and spectral analysis of weight matrices | Medium |

---

## Existing ToadStool Kernels That Apply

| ToadStool Kernel | neuralSpring Extension Use |
|-----------------|--------------------------|
| `gemm_f64.wgsl` | Population fitness evaluation, HMM transitions, game payoff matrices |
| `FusedMapReduceF64` | Population statistics, MODES metrics, trajectory averaging |
| `BatchedEighGpu` | Covariance eigendecomposition for population analysis |
| `attention.wgsl` | Regulatory network attention (Waters), multi-agent interaction |
| `lstm_cell.wgsl` | Temporal evolution patterns, state-space model updates |
| `conv2d.wgsl` | Spatial evolution patterns (Dolson swarm robotics) |
| `gemv_q4/q8.wgsl` | Quantized inference for deployed models across all springs |

---

## BarraCUDA Evolution Path for neuralSpring

```
Phase 0/0+ (DONE — Python/PyTorch)     Phase 1 (GPU — NEXT)
────────────────────────────            ────────────────────
PyTorch MLP training        ────────→   BarraCUDA MLP (GEMM + ReLU + Adam)
PyTorch LSTM                ────────→   BarraCUDA LSTM (lstm_cell.wgsl)
PyTorch Conv2d              ────────→   BarraCUDA Conv2d (conv2d.wgsl)
PyTorch quantized           ────────→   BarraCUDA Q4/Q8 (gemv_q4/q8.wgsl)
N/A                         ────────→   Evolutionary optimization (NEW)
N/A                         ────────→   HMM Viterbi (NEW)
N/A                         ────────→   Gillespie simulation (NEW)
N/A                         ────────→   L-BFGS optimizer (NEW)

Phase 1 (GPU)                           Phase 2 (Applications)
─────────────                           ──────────────────────
BarraCUDA MLP              ────────→    Live ET₀ surrogate for Penny Irrigation
BarraCUDA LSTM             ────────→    Real-time weather forecasting
BarraCUDA Q4               ────────→    llama.cpp parity (Squirrel inference)
Evolutionary optimization  ────────→    MODES metrics on BarraCUDA evolution
HMM Viterbi                ────────→    Metagenomic classification (wetSpring)
```

---

## Cross-Spring Kernel Sharing

neuralSpring validates the ML primitives that all springs consume:

| Primitive | neuralSpring Validates | Springs That Use It |
|-----------|----------------------|-------------------|
| MLP surrogate | Exp 001, Study 001-002 | airSpring (ET₀), hotSpring (physics) |
| LSTM | Exp 003, Study 004 | airSpring (weather), wetSpring (time series) |
| Transfer learning | Exp 004 | airSpring (climate adaptation), wetSpring (cross-site) |
| Quantized inference | Study 005 | Squirrel (sovereign AI), all production deployments |
| Conv2d | Study 003 | wetSpring (spectral image analysis), future vision |
| Autograd | Study 001 | hotSpring (force computation), groundSpring (sensitivity) |
| Attention | Exp 002 | biomeOS PathwayLearner, neuralAPI capability routing |

### BarraCUDA Primitives Validated (Phase 1b — February 2026)

`barracuda` path dependency added; 12 validation binaries call `barracuda::*` directly.
The `tensor` binary exercises the **unified Tensor/WGSL path** — same shaders that run on GPU.

| Binary | Module | Checks | Status |
|--------|--------|--------|--------|
| `validate_barracuda_stats` | `stats::{variance, pearson, cov, spearman, norm_*}` | 13 | **PASS** |
| `validate_barracuda_linalg` | `linalg::{solve, lu_*, eigh, cholesky, tridiag}` | 17 | **PASS** |
| `validate_barracuda_special` | `special::{gamma, erf, bessel_*, legendre, hermite, laguerre}` | 26 | **PASS** |
| `validate_barracuda_optimize` | `optimize::{nelder_mead, bisect, brent}` | 10 | **PASS** |
| `validate_barracuda_precision` | `shaders::precision::cpu` (add, mul, fma, dot, sum) | 12 | **PASS** |
| `validate_barracuda_tensor` | Tensor API: relu, gelu, sigmoid, softmax, layer\_norm, matmul, mse\_loss + tanh, exp, log, sqrt, div, scalar ops, reductions, swish, mish, losses, transpose, evolved ops | 90 | **PASS** |
| `validate_barracuda_tensor_f64` | f64 GPU ops: roundtrip, SumReduce, FusedMapReduce, NormReduce, VarianceReduce, WeightedDot, MaxAbsDiff, CosineSimilarity | 35 | **PASS** |
| `validate_barracuda_quantized` | `shaders::quantized` (dequant Q4/Q8, GEMV) | 15 | **PASS** |
| `validate_barracuda_linalg_ext` | `linalg::{svd_*, lu_inverse, gen_eigh}` | 17 | **PASS** |
| `validate_barracuda_ml_inference` | ML inference: MLP + Transformer end-to-end vs Python/NumPy baselines | 13 | **PASS** |
| `validate_barracuda_fft` | FFT: Cooley-Tukey 1D f32/f64, inverse round-trip, Parseval, known DFT pairs, Rfft | 24 | **PASS** |
| `validate_barracuda_logsumexp` | LogSumExp: numerically stable summation in log-probability space (HMM, softmax) | 5 | **PASS** |
| **Total** | | **272** | **ALL PASS** |

### BarraCUDA GPU Tensor — CPU vs GPU Validation (Phase 5a — February 22, 2026)

7 domain-specific validators exercise `Tensor::matmul`, `transpose`, `tanh`, `add`
on a live RTX 4070 Vulkan backend. Each check compares GPU f32 output against
CPU f64 reference with calibrated tolerances.

| Binary | Domain | Checks | Status | Notes |
|--------|--------|--------|--------|-------|
| `validate_barracuda_gpu_spectral` | Kachkovskiy spectral | 10/10 | **PASS** | `A x B^T` workaround for S-14 |
| `validate_barracuda_gpu_eco` | Dolson eco dynamics | 6/6 | **PASS** | Fitness matrices, carrying capacity |
| `validate_barracuda_gpu_hmm` | Liu HMM | 5/5 | **PASS** | Transition, emission, Viterbi score |
| `validate_barracuda_gpu_fitness` | Dolson fitness landscapes | 7/7 | **PASS** | NK landscape, epistasis, ruggedness |
| `validate_barracuda_gpu_nn` | PINN / DeepONet | 5/5 | **PASS** | MLP forward, tanh activations. S-15 workaround |
| `validate_barracuda_gpu_pairwise` | SATé / Pangenome | 5/5 | **PASS** | S-16 fixed (transpose dispatch) |
| `validate_barracuda_gpu_anderson` | Anderson localization | 7/7 | **PASS** | S-15 workaround (data ≥ 0.5) |
| **Total** | | **98+** | **ALL GREEN** | S-16 fixed, S-15 root-caused, 23/25 gT coverage |

**Shortcomings discovered in Phase 5a:**

| ID | Summary | Severity | Status |
|----|---------|----------|--------|
| S-14 | Naive matmul hang — small square inputs (N < 32) in complex binaries | Medium | Workaround applied (`A x B^T`) |
| S-15 | Matmul hang — negative or sparse f32 input data | Critical | Blocks anderson validator |
| S-16 | 2D transpose dispatches wrong workgroup count (256 vs 16) | High | Blocks Gram matrix; fix identified |

Full diagnosis and reproduction steps: `wateringHole/handoffs/archive/NEURALSPRING_V6_BARRACUDA_GPU_HANDOFF_FEB22_2026.md`
Current handoff: `wateringHole/handoffs/NEURALSPRING_TOADSTOOL_V30_S67_HANDOFF_FEB25_2026.md`

---

neuralSpring is the validation layer. ToadStool is the implementation layer. The springs are the application layers.
