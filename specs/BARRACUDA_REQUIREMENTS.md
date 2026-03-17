# neuralSpring — BarraCUDA Requirements

**Last Updated**: March 17, 2026 (Sessions 44–163 — 220/220 validate_all, 55/55 dispatch parity, 260 binaries, 1152 lib + 70 playGround + 73 forge tests, barraCuda v0.3.5 at `0649cd0`, 216 import files, V114 handoff)
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

*Updated Feb 26, 2026 (Session 75): All P0 and P1 gaps RESOLVED. 4 of 6 P2 gaps RESOLVED.*

### Critical (P0) — RESOLVED

| Need | Paper | Resolution | Status |
|------|-------|-----------|--------|
| **Evolutionary optimization (GA/ES)** | Dolson: Iram et al. 2020, MODES 2019 | `BatchFitnessGpu`, `MultiObjFitnessGpu`, `WrightFisherGpu` (S39+) | **RESOLVED** |
| **Fitness landscape evaluation** | Dolson: all papers | `BatchFitnessGpu` + `barracuda::stats::variance` (S39+) | **RESOLVED** |

### Important (P1) — RESOLVED

| Need | Paper | Resolution | Status |
|------|-------|-----------|--------|
| **HMM Viterbi decoding** | Liu: PhyloNet-HMM 2014 | `HmmBatchForwardF64` + `Tensor::argmax_dim(0)` (S39+, rewired S73) | **RESOLVED** |
| **Log-sum-exp** | Liu: HMM, phylogenetics | `LogSumExp` op + `logsumexp_reduce.wgsl` (S42+) | **RESOLVED** |
| **Gillespie stochastic simulation** | Waters: cooperation dynamics 2018 | `GillespieGpu` + `xoshiro128ss.wgsl` GPU PRNG (S39+) | **RESOLVED** |
| **Game-theoretic payoff matrix** | Waters: Bruger 2018, Mhatre 2020 | `SpatialPayoffGpu` (S39+) | **RESOLVED** |

### Stretch (P2) — Mostly resolved

| Need | Paper | Resolution | Status |
|------|-------|-----------|--------|
| **MODES metric computation** | Dolson 2019 | `PairwiseL2Gpu` + `barracuda::stats::shannon` (S39+, S64) | **RESOLVED** |
| **Phylogenetic likelihood** | Liu: SATé 2009, cophylogenetics 2023 | `FelsensteinGpu`, `FlatTree` (S39+) | **RESOLVED** |
| **L-BFGS optimizer** | Raissi 2019 (PINN improvement) | `barracuda::optimize::LbfgsGpu` available in barraCuda v0.3.5 (requires `gpu` feature) | **AVAILABLE** |
| **Directed evolution framework** | Dolson 2022 (eLife) | `MultiObjFitnessGpu`, `WrightFisherGpu`, `BatchedMultinomialGpu` (S39+, S61) | **RESOLVED** |
| **Lanczos eigensolve** | Kachkovskiy: JAMS 2016, GAFA 2018 | `BatchedEighGpu`, `eigh_f64`, `sparse_eigh` (S39+) | **RESOLVED** |
| **Sparse matrix-vector product** | Kachkovskiy (all) | `SparseGemmF64`, `cg_solve`, `bicgstab_solve` (S39+, S52) | **RESOLVED** |

---

## Existing `BarraCUDA` Kernels That Apply

| `BarraCUDA` Kernel | neuralSpring Extension Use |
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
Phase 0/0+ (DONE)                       Phase 1 (GPU — DONE)
────────────────────────────            ────────────────────
PyTorch MLP training        ────────→   BarraCUDA MLP (GEMM + ReLU + Adam)          ✓
PyTorch LSTM                ────────→   BarraCUDA LSTM (lstm_cell.wgsl)              ✓
PyTorch Conv2d              ────────→   BarraCUDA Conv2d (conv2d.wgsl)               ✓
PyTorch quantized           ────────→   BarraCUDA Q4/Q8 (gemv_q4/q8.wgsl)           ✓
N/A                         ────────→   Evolutionary optimization (BatchFitnessGpu)  ✓
N/A                         ────────→   HMM Viterbi (HmmBatchForwardF64+argmax)      ✓
N/A                         ────────→   Gillespie simulation (GillespieGpu)          ✓
N/A                         ────────→   L-BFGS optimizer (LbfgsGpu v0.3.5)           AVAILABLE

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
| `validate_barracuda_tensor` | Tensor API: relu, gelu, sigmoid, softmax, layer\_norm, matmul, mse\_loss + tanh, exp, log, sqrt, div, scalar ops, reductions, swish, mish, losses, transpose, evolved ops | 86 | **PASS** |
| `validate_barracuda_tensor_f64` | f64 GPU ops: roundtrip, SumReduce, FusedMapReduce, NormReduce, VarianceReduce, WeightedDot, MaxAbsDiff, CosineSimilarity | 35 | **PASS** |
| `validate_barracuda_quantized` | `shaders::quantized` (dequant Q4/Q8, GEMV) | 15 | **PASS** |
| `validate_barracuda_linalg_ext` | `linalg::{svd_*, lu_inverse, gen_eigh}` | 17 | **PASS** |
| `validate_barracuda_ml_inference` | ML inference: MLP + Transformer end-to-end vs Python/NumPy baselines | 13 | **PASS** |
| `validate_barracuda_fft` | FFT: Cooley-Tukey 1D f32/f64, inverse round-trip, Parseval, known DFT pairs, Rfft | 24 | **PASS** |
| `validate_barracuda_logsumexp` | LogSumExp: numerically stable summation in log-probability space (HMM, softmax) | 5 | **PASS** |
| **Total** | | **268** | **ALL PASS** |

### BarraCUDA GPU Tensor — CPU vs GPU Validation (Phase 5a — February 22, 2026)

7 domain-specific validators exercise `Tensor::matmul`, `transpose`, `tanh`, `add`
on a live RTX 4070 Vulkan backend. Each check compares GPU f32 output against
CPU f64 reference with calibrated tolerances.

| Binary | Domain | Checks | Status | Notes |
|--------|--------|--------|--------|-------|
| `validate_barracuda_gpu_spectral` | Kachkovskiy spectral | 10/10 | **PASS** | S-14 **RESOLVED** upstream (`a4996b34` S39) |
| `validate_barracuda_gpu_eco` | Dolson eco dynamics | 6/6 | **PASS** | Fitness matrices, carrying capacity |
| `validate_barracuda_gpu_hmm` | Liu HMM | 5/5 | **PASS** | Transition, emission, Viterbi score |
| `validate_barracuda_gpu_fitness` | Dolson fitness landscapes | 7/7 | **PASS** | NK landscape, epistasis, ruggedness |
| `validate_barracuda_gpu_nn` | PINN / DeepONet | 5/5 | **PASS** | MLP forward, tanh activations. S-15 **RESOLVED** upstream (`a4996b34` S39) |
| `validate_barracuda_gpu_pairwise` | SATé / Pangenome | 5/5 | **PASS** | S-16 **RESOLVED** upstream (`a4996b34` S39: transpose dispatch) |
| `validate_barracuda_gpu_anderson` | Anderson localization | 7/7 | **PASS** | S-15 **RESOLVED** upstream (`a4996b34` S39) |
| **Total** | | **98+** | **ALL GREEN** | S-14/S-15/S-16/S-17 **RESOLVED** upstream, 23/25 gT coverage |

**Shortcomings discovered in Phase 5a (all now RESOLVED upstream):**

| ID | Summary | Severity | Status |
|----|---------|----------|--------|
| S-14 | Naive matmul hang — small square inputs (N < 32) in complex binaries | Medium | **RESOLVED** upstream (`a4996b34` S39: Naive tier removed) |
| S-15 | Matmul hang — negative or sparse f32 input data | Critical | **RESOLVED** upstream (`a4996b34` S39: Matmul hang fixed) |
| S-16 | 2D transpose dispatches wrong workgroup count (256 vs 16) | High | **RESOLVED** upstream (`a4996b34` S39: Transpose dispatch fixed) |

Full diagnosis and reproduction steps: `wateringHole/handoffs/archive/NEURALSPRING_V6_BARRACUDA_GPU_HANDOFF_FEB22_2026.md`
Current handoff: `wateringHole/handoffs/NEURALSPRING_TOADSTOOL_V40_S76_MODERN_REWIRING_BENCHMARK_HANDOFF_FEB26_2026.md`

### Session 68 — BarraCUDA Usage Audit

Full inventory of barracuda consumption: 90+ import sites across 60+ files.

| Module | Items | Scope |
|--------|-------|-------|
| `device` | `WgpuDevice`, `GpuDriverProfile`, `Fp64Strategy` | 31+ files |
| `tensor` | `Tensor`, `Tensor::from_data` | 28+ files |
| `ops::bio` | 17 GPU ops | 40+ files |
| `ops::mha` | `MultiHeadAttention` | 1 file |
| `ops::linalg` | `BatchedEighGpu`, `eigh_householder_qr` | 3 files |
| `ops::fft` | `Fft1D`, `Fft1DF64`, `Ifft1D`, `Rfft` | 1 file |
| `ops::fused_map_reduce_f64` | `FusedMapReduceF64`, `MapOp`, `ReduceOp` | 5 files |
| `stats` | `pearson_correlation`, `variance`, `covariance` | 8+ files |
| `special` | `chi_squared_statistic`, `gamma`, `erf`, `bessel_j0` | 6+ files |
| `spectral` | `BatchIprGpu`, Anderson, Lanczos | 8+ files |
| `dispatch` | 9 typed dispatch functions | 12+ files |
| Other | `pipeline`, `staging`, `numerical`, `linalg::graph` | 15+ files |

**Zero duplicate math** — two intentional divergences documented:
- `cpu_fallback::variance` (population ÷N) vs barracuda (sample ÷(N-1))
- `primitives.rs` (independent CPU reference for validation independence)

---

neuralSpring is the validation layer. ToadStool is the implementation layer. The springs are the application layers.
