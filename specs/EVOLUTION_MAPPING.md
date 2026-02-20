# neuralSpring — Evolution Mapping: Rust Module → WGSL Shader → Pipeline Stage

**Last Updated**: February 20, 2026
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

### Tier A — Direct Rewire (validated, ready for GPU promotion)

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

### Tier A — Phase 0++ Paper Reproductions (validated, ready for GPU promotion)

All Phase 0++ modules are pure math, deterministic (seed=42), and use no
external dependencies beyond `crate::rng::Rng`. They are ideal Tier A
candidates for BarraCUDA CPU port and subsequent GPU promotion.

| Python Module | Rust Module | Checks | WGSL Shader Target | Key Primitive |
|---------------|-------------|--------|--------------------|----|
| `counterdiabatic/` | `counterdiabatic.rs` | 19 | `gemm_f64` + `softmax.wgsl` | NK fitness, Boltzmann |
| `modes/` | `modes.rs` | 9 | `reduce_sum` + `elementwise` | Change/novelty/complexity |
| `eco_dynamics/` | `eco_dynamics.rs` | 7 | batch `gemm_f64` + `reduce_sum` | Multi-niche EA |
| `directed_evolution/` | `directed_evolution.rs` | 7 | batch `gemm_f64` + `reduce_max` | 5 selection algorithms |
| `swarm_robotics/` | `swarm_robotics.rs` | 7 | batch `gemm_f64` | Heterogeneous controllers |
| `hmm_phylo/` | `hmm.rs` | 17 | `gemm_f64` chain (log-domain) | Forward/backward/Viterbi |
| `sate_alignment/` | `sate_alignment.rs` | 8 | `gemm_f64` (distance matrix) | NJ tree + alignment |
| `introgression/` | `introgression.rs` | 13 | `gemm_f64` chain + log-sum-exp | PhyloNet-HMM + LRT |
| `game_theory/` | `game_theory.rs` | 8 | `gemm_f64` + `softmax.wgsl` | Replicator, QS spatial |
| `regulatory_network/` | `regulatory_network.rs` | 5 | `elementwise` | Hill ODE + RK4 |
| `signal_integration/` | `signal_integration.rs` | 8 | `elementwise` | Two-input Hill AND gate |
| `spectral_commutativity/` | `spectral_commutativity.rs` | 8 | `gemm_f64` | Commutator [A,B] |
| `anderson_localization/` | `anderson_localization.rs` | 8 | `tridiag` + `eigh_f64` | Aubry-André, IPR |

### Tier A+ — BarraCUDA CPU Primitives (validated 2026-02-19)

Direct `barracuda::*` calls validated against analytical / NIST DLMF baselines.

| BarraCUDA Module | Validation Binary | Checks | Status |
|------------------|-------------------|--------|--------|
| `stats::{variance, std_dev, pearson, covariance, spearman, norm_*}` | `validate_barracuda_stats` | 13 | **PASS** |
| `linalg::{solve_f64, lu_det, lu_solve, eigh_f64, cholesky_f64, tridiag}` | `validate_barracuda_linalg` | 17 | **PASS** |
| `special::{gamma, factorial, erf, bessel, legendre, hermite, laguerre}` | `validate_barracuda_special` | 26 | **PASS** |
| `optimize::{nelder_mead, bisect, brent}` | `validate_barracuda_optimize` | 10 | **PASS** |
| `shaders::precision::cpu` (add, mul, fma, dot, kahan\_sum) | `validate_barracuda_precision` | 12 | **PASS** |
| **Tensor API** (84 ops including evolved) | `validate_barracuda_tensor` | 84 | **PASS** |
| **Tensor f64 API** (GPU reductions + fused maps) | `validate_barracuda_tensor_f64` | 35 | **PASS** |
| `shaders::quantized` (dequant Q4/Q8, GEMV) | `validate_barracuda_quantized` | 15 | **PASS** |
| `linalg::{svd\_\*, lu\_inverse, gen\_eigh}` | `validate_barracuda_linalg_ext` | 17 | **PASS** |
| **ML Inference** (MLP + Transformer end-to-end) | `validate_barracuda_ml_inference` | 13 | **PASS** |
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
| Batched GEMM | `gemm_f64.wgsl` (batched) | Training / EA | Batch dispatch |
| Population fitness eval | batch `gemm_f64` + selection | Evolution (Dolson 011–015) | GA/ES framework |
| HMM forward (fused) | `hmm_forward_log.wgsl` | Genomics (Liu 016–018) | Log-domain matmul chain |
| Pairwise distance | `pairwise_distance.wgsl` | Alignment (Liu 017) | One thread per pair |
| GPU ODE integrator (RK4) | `rk4_batch.wgsl` | Biology (Waters 020–021) | Elementwise RHS |
| Spatial stencil | `stencil_1d.wgsl` | Cooperation (Waters 019) | Neighbor averaging |
| Tridiag eigensolver | `tridiag_eigh.wgsl` | Spectral (Kachkovskiy 022–023) | Bisection + inverse iteration |
| GPU PRNG (Xoshiro256**) | `xoshiro256ss.wgsl` | All stochastic algorithms | `jump()` for independent streams |
| Gillespie SSA | GPU PRNG + exp sampling | Biology (Waters) | New primitive |

---

## GPU Promotion Priority

Based on cross-paper primitive usage and BarraCUDA impact:

| Priority | Primitive | Papers Served | Effort | Impact |
|----------|-----------|---------------|--------|--------|
| 1 | Batch GEMM/GEMV | 011–015 (5 papers) | Medium | Parallel population eval |
| 2 | Pairwise distance kernel | 017 | Low | Simple, high-value |
| 3 | GPU-parallel RK4 | 020–021 | Medium | Multi-system ODE |
| 4 | Fused HMM forward | 016–018 | Medium | Log-domain matmul chain |
| 5 | Tridiagonal eigensolver | 022–023 | High | Specialized for structure |
| 6 | Spatial stencil | 019 | Low | Reuse conv1d |
| 7 | GPU PRNG | All stochastic | Medium | Foundation for parallel EA |

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
| Phase 0 (Python baselines) | **190/190 PASS** | 23 experiments, 48 pytest |
| Phase 1a (neuralSpring Rust) | **167/167 PASS** | 20 modules, 109 unit tests, 16 validation binaries |
| Phase 1b (BarraCUDA) | **242/242 PASS** | 10 validation binaries, incl. Tensor/WGSL (84), tensor_f64 (35), ml_inference (13) |
| Phase 1c (Fused pipeline) | **46–78× speedup** | Single-encoder dispatch, GPU-resident ops |
| Phase 2 (BarraCUDA CPU port) | **Planned** | All 13 Phase 0++ modules are Tier A |
| Phase 3 (GPU acceleration) | **Planned** | 7 new primitives identified above |
| Phase 4 (Sovereign pipeline) | **Planned** | Unidirectional streaming via ToadStool |
