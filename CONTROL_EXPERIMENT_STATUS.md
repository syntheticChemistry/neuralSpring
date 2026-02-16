# neuralSpring — Control Experiment Status

**Last updated**: February 16, 2026
**Gate**: Eastgate (i9-12900K, 32 GB DDR5, RTX 4070 12GB, Pop!_OS 22.04)
**Python**: 3.10, PyTorch 2.9.0+cu128
**Grand Total**: 75/75 quantitative checks PASS (48 Phase 0 + 27 Phase 0+)

---

## Phase 0 — Synthetic Baselines (48/48 PASS)

| ID | Title | Domain | Tests | Status |
|----|-------|--------|-------|--------|
| Exp 001 | Neural Surrogate Validation | MLP vs RBF + FAO-56 | 11/11 | **PASS** |
| Exp 002 | Transformer Inference Baseline | Self-attention from scratch | 18/18 | **PASS** |
| Exp 003 | Sequence Forecasting | LSTM/GRU weather time series | 5/5 | **PASS** |
| Exp 004 | Transfer Learning | Michigan → NM/CA adaptation | 6/6 | **PASS** |
| Exp 005 | Isomorphic Pattern Catalog | Cross-domain op mapping | 8/8 | **PASS** |

## Phase 0+ — Scholarly Reproduction Studies (27/27 PASS)

| ID | Title | Paper | Tests | Status |
|----|-------|-------|-------|--------|
| Study 001 | PINN Burgers' Equation | Raissi et al. (2019) JCP 378:686 | 6/6 | **PASS** |
| Study 002 | DeepONet Antiderivative | Lu et al. (2021) NMI 3:218 | 5/5 | **PASS** |
| Study 003 | LeNet-5 MNIST | LeCun et al. (1998) Proc IEEE 86 | 5/5 | **PASS** |
| Study 004 | LSTM ERA5 Weather | Gauch et al. (2021) HESS 25:2045 | 5/5 | **PASS** |
| Study 005 | Quantized Inference | Dettmers (2022) + Frantar (2023) | 6/6 | **PASS** |

---

## Phase 0+ Study Details

### Study 001: PINN Burgers' Equation (6/6)

**Paper**: Raissi, Perdikaris, Karniadakis (2019) "Physics-informed neural networks" JCP

| Check | Result | Status |
|-------|--------|--------|
| IC validation (Cole-Hopf) | error < 1e-6 | PASS |
| BC validation | error < 1e-15 | PASS |
| Training convergence | best loss < 0.01 | PASS |
| L2 relative error < 15% | ~5.1% (paper: 0.06% with L-BFGS) | PASS |
| Shock steepening detected | gradient increase confirmed | PASS |
| Op analysis | GEMM + tanh + autograd | PASS |

### Study 002: DeepONet Antiderivative (5/5)

**Paper**: Lu, Jin, Pang, Zhang, Karniadakis (2021) "Learning nonlinear operators" NMI

| Check | Result | Status |
|-------|--------|--------|
| Data generation | 1000 train, 200 test | PASS |
| Mean L2 error < 5% | ~1.2% | PASS |
| RMSE < 0.05 | confirmed | PASS |
| Specific operators (2/3 < 0.1) | confirmed | PASS |
| Architecture analysis | Branch-trunk ≈ encoder-decoder | PASS |

### Study 003: LeNet-5 MNIST (5/5)

**Paper**: LeCun, Bottou, Bengio, Haffner (1998) Proc. IEEE

| Check | Result | Status |
|-------|--------|--------|
| MNIST loaded | 60,000 train, 10,000 test | PASS |
| Test accuracy ≥ 98.5% | 98.89% (paper: 99.05%) | PASS |
| All digits ≥ 95% | confirmed | PASS |
| Feature map dimensions | (1,16,5,5) correct | PASS |
| Op analysis | Conv2d + MaxPool + FC | PASS |

### Study 004: LSTM ERA5 Weather (5/5)

**Paper**: Gauch et al. (2021) HESS 25:2045 (methodology; data is real ERA5)

| Check | Result | Status |
|-------|--------|--------|
| Data loaded (real ERA5) | 1461 days, Open-Meteo | PASS |
| NSE > 0.80 | 0.849 | PASS |
| RMSE < 5.0°C | 3.46°C | PASS |
| Multi-horizon analysis | 1d/3d/7d completed | PASS |
| Op analysis | lstm_cell + GEMM | PASS |

### Study 005: Quantized Inference (6/6)

**Papers**: Dettmers et al. (2022), Frantar et al. (2023)

| Check | Result | Status |
|-------|--------|--------|
| FP32 baseline R² > 0.99 | 0.9998 | PASS |
| INT8 degradation < 1% | 0.017% | PASS |
| INT4 degradation < 5% | 0.79% | PASS |
| Throughput benchmark | completed | PASS |
| Memory analysis | FP32→Q4 = 8× compression | PASS |
| BarraCUDA mapping | gemv_q4/q8 validated | PASS |

---

## BarraCUDA Primitive Coverage

| Primitive | Validated By | WGSL Shader |
|-----------|-------------|-------------|
| GEMM | All experiments & studies | gemm_f64.wgsl |
| Attention | Exp 002 | attention.wgsl, mha_output.wgsl |
| LayerNorm | Exp 002 | layer_norm.wgsl |
| ReLU/GELU/Tanh | Exp 001, Study 001, 003 | nn::ReLU |
| LSTM cell | Exp 003, Study 004 | lstm_cell.wgsl |
| Conv2d | Study 003 | conv2d.wgsl |
| MaxPool | Study 003 | max_pool2d.wgsl |
| Autograd | Study 001 | fd_gradient_f64.wgsl |
| Branch-Trunk (DeepONet) | Study 002 | elementwise_mul + sum_reduce |
| Quantized GEMV | Study 005 | gemv_q4.wgsl, gemv_q8.wgsl |
| Dequantization | Study 005 | dequant_q4.wgsl, dequant_q8.wgsl |

---

## Evolution Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Synthetic baselines (48 checks) | **COMPLETE** |
| 0+ | Scholarly reproductions (27 checks) | **COMPLETE** |
| 1 | BarraCUDA Rust port | Planned |
| 2 | Quantized inference on GPU | Planned |
| 3 | Cross-spring integration | Planned |
