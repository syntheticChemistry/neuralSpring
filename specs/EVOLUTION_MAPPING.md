# neuralSpring — Evolution Mapping: Rust Module → WGSL Shader → Pipeline Stage

**Last Updated**: February 19, 2026
**Purpose**: Concrete mapping from Phase 0 Python → Phase 1 Rust → Phase 2 GPU

---

## Tier Classification

| Tier | Meaning | Criteria |
|------|---------|----------|
| **A** (rewire) | Direct port — pure math, no framework dependencies | NumPy-only implementations, analytical known-values |
| **B** (adapt) | Needs adaptation — training loops, data dependencies | PyTorch training, real data, stochastic |
| **C** (new) | New implementation — no Python equivalent | GPU-specific (flash attention, fused kernels) |

---

## Module-by-Module Mapping

### Tier A — Direct Rewire (ready for Rust port)

| Python Module | Rust Module | WGSL Shader | Pipeline Stage | Status |
|---------------|-------------|-------------|----------------|--------|
| `transformer/` softmax | `transformer::softmax` | `attention.wgsl` (softmax stage) | Inference | **VALIDATED** (18 checks) |
| `transformer/` GELU | `transformer::gelu` | elementwise | Inference | **VALIDATED** (18 checks) |
| `transformer/` LayerNorm | `transformer::layer_norm` (stub) | `layer_norm.wgsl` | Inference | Implement norm |
| `transformer/` SDPA | `transformer::sdpa` (stub) | `attention.wgsl` | Inference | Implement QKV matmul |
| `surrogate/` Rastrigin | `surrogate::rastrigin_2d` | N/A (test function) | Validation | **VALIDATED** (15 checks) |
| `surrogate/` Rosenbrock | `surrogate::rosenbrock_2d` | N/A (test function) | Validation | **VALIDATED** (15 checks) |
| `surrogate/` Ackley | `surrogate::ackley_2d` | N/A (test function) | Validation | **VALIDATED** (15 checks) |
| `surrogate/` R²/RMSE/MAE | `metrics::*` | `FusedMapReduceF64` | Validation | **VALIDATED** (10 checks) |

### Tier A+ — BarraCUDA CPU Primitives (validated 2026-02-19)

Direct `barracuda::*` calls validated against analytical / NIST DLMF baselines.
Follows hotSpring pattern: `ValidationHarness`, hardcoded expected, exit 0/1.

| BarraCUDA Module | Validation Binary | Checks | Status |
|------------------|-------------------|--------|--------|
| `stats::{variance, std_dev, pearson, covariance, spearman, norm_*}` | `validate_barracuda_stats` | 13 | **PASS** |
| `linalg::{solve_f64, lu_det, lu_solve, eigh_f64, cholesky_f64, tridiag}` | `validate_barracuda_linalg` | 17 | **PASS** |
| `special::{gamma, factorial, erf, bessel, legendre, hermite, laguerre}` | `validate_barracuda_special` | 26 | **PASS** |
| `optimize::{nelder_mead, bisect, brent}` | `validate_barracuda_optimize` | 10 | **PASS** |
| `shaders::precision::cpu` (add, mul, fma, dot, kahan\_sum) | `validate_barracuda_precision` | 12 | **PASS** |
| **Tensor API** (relu, gelu, sigmoid, softmax, layer\_norm, matmul, mse\_loss + tanh, exp, log, sqrt, div, scalar ops, reductions, swish, mish, losses, transpose, evolved ops) | `validate_barracuda_tensor` | 84 | **PASS** |
| **Tensor f64 API** (roundtrip, SumReduce, FusedMapReduce, NormReduce, VarianceReduce, WeightedDot, MaxAbsDiff, CosineSimilarity) | `validate_barracuda_tensor_f64` | 35 | **PASS** |
| `shaders::quantized` (dequant Q4/Q8, GEMV) | `validate_barracuda_quantized` | 15 | **PASS** |
| `linalg::{svd\_\*, lu\_inverse, gen\_eigh}` | `validate_barracuda_linalg_ext` | 17 | **PASS** |
| **ML Inference** (MLP + Transformer end-to-end vs Python baselines) | `validate_barracuda_ml_inference` | 13 | **PASS** |
| **Total** | **10 binaries** | **242** | **ALL PASS** |

### Tier B — Adapt (needs training infrastructure)

| Python Module | Rust Module | WGSL Shader | Pipeline Stage | Blocker |
|---------------|-------------|-------------|----------------|---------|
| `surrogate/` MLP forward | `surrogate::mlp_forward` (stub) | `gemm_f64.wgsl` + `nn::ReLU` | Inference | BarraCUDA `nn::Layer` |
| `surrogate/` MLP training | `surrogate::mlp_train` (stub) | `gemm_f64.wgsl` + `nn::Optimizer::Adam` | Training | BarraCUDA autograd |
| `sequence/` LSTM cell | — | `lstm_cell.wgsl` | Inference | BarraCUDA LSTM primitive |
| `sequence/` GRU cell | — | `gru_cell.wgsl` | Inference | BarraCUDA GRU primitive |
| `pinn/` autograd | — | `fd_gradient_f64.wgsl` | Training | Reverse-mode AD in BarraCUDA |
| `lenet/` Conv2d | — | `conv2d.wgsl` | Inference | BarraCUDA Conv2d |
| `lenet/` MaxPool | — | `max_pool2d.wgsl` | Inference | BarraCUDA pooling |
| `deeponet/` Branch-Trunk | — | `gemm_f64.wgsl` × 2 | Inference | Compose from MLP |
| `quantized/` INT8 GEMV | — | `gemv_q8.wgsl` | Deployment | BarraCUDA Q8 kernels |
| `quantized/` INT4 GEMV | — | `gemv_q4.wgsl` | Deployment | BarraCUDA Q4 kernels |
| `transfer/` freeze+finetune | — | selective gradient | Training | BarraCUDA param freeze |

### Tier C — New (GPU-specific, no Python equivalent)

| Capability | WGSL Shader | Pipeline Stage | Blocker |
|------------|-------------|----------------|---------|
| Flash attention | `flash_attention.wgsl` | Inference | Algorithm implementation |
| Fused LayerNorm+GELU | fused kernel | Inference | Kernel fusion framework |
| Batched GEMM | `gemm_f64.wgsl` (batched) | Training | Batch dispatch |
| Population fitness eval | `gemm_f64.wgsl` + selection | Evolution (Dolson) | GA/ES framework |
| HMM Viterbi | log-sum-exp + traceback | Genomics (Liu) | New primitive |
| Gillespie SSA | GPU PRNG + exp sampling | Biology (Waters) | New primitive |

---

## Promotion Checklist

For each Rust module → GPU promotion:

- [ ] Python baseline passes with documented provenance
- [ ] Rust implementation matches Python to documented tolerance
- [ ] WGSL shader exists in BarraCUDA or is planned
- [ ] Validation binary follows hotSpring pattern (exit 0/1)
- [ ] Performance meets or exceeds Python baseline
- [ ] Test coverage ≥ 90% (analytical + round-trip + determinism)

---

## Current Status (February 2026)

| Phase | Status | Coverage |
|-------|--------|----------|
| Phase 0 (Python baselines) | **75/75 PASS** | 10 experiments, 48 pytest |
| Phase 1a (neuralSpring Rust) | **43/43 PASS** | 9 modules, 34 unit tests, 3 validation binaries |
| Phase 1b (BarraCUDA) | **242/242 PASS** | 10 validation binaries, incl. unified Tensor/WGSL path (84), tensor_f64 (35), ml_inference (13), evolved ops |
| Phase 1c (Fused pipeline) | **43–78× speedup** | Single-encoder dispatch, GPU-resident head-split/concat, batched attention |
| Phase 2 (GPU shaders) | **Planned** | Mapping documented above |
| Phase 3 (Sovereign pipeline) | **Planned** | Depends on Phase 2 |
