# neuralSpring — Methodology

## Phase 0: Python/PyTorch Baselines

### Validation Framework

Each experiment follows the same structure:
1. **Benchmark definition** — expected results, acceptance criteria
2. **Implementation** — Python/PyTorch reference code
3. **Validation** — automated PASS/FAIL checks against criteria
4. **Isomorphic analysis** — map ops to BarraCUDA WGSL shaders

### Experiment Details

#### Experiment 001: Neural Surrogate Validation (11 checks)

**Objective**: Compare MLP neural surrogates against RBF interpolation.

| Check | Target |
|-------|--------|
| Rastrigin RBF R² | ≥ 0.40 (multimodal — intentionally hard) |
| Rastrigin MLP R² | ≥ 0.40 |
| Rosenbrock RBF R² | ≥ 0.95 |
| Rosenbrock MLP R² | ≥ 0.95 |
| Ackley RBF R² | ≥ 0.90 |
| Ackley MLP R² | ≥ 0.90 |
| FAO-56 RBF R² | ≥ 0.95 |
| FAO-56 MLP R² | ≥ 0.95 |
| FAO-56 MLP RMSE | ≤ 0.15 mm/day |
| Training efficiency | Completed |
| Op count analysis | Completed |

**Key finding**: MLP (4,673 params) achieves R²=0.999 on FAO-56, matching RBF.
Rastrigin exposes the limitation of random sampling for multimodal functions —
exactly why hotSpring's SparsitySampler matters.

#### Experiment 002: Transformer Inference Baseline (18 checks)

**Objective**: Implement self-attention from scratch, validate against PyTorch.

| Check Category | Count |
|----------------|-------|
| Softmax properties | 3 |
| SDPA correctness | 3 |
| Causal attention | 2 |
| Multi-head attention | 3 |
| LayerNorm + GELU + FFN | 4 |
| Full transformer block | 2 |
| Isomorphic catalog | 1 |

**Key finding**: NumPy SDPA matches PyTorch to <1e-10. The transformer block
(8 MatMuls + Softmax + LayerNorm + GELU) is identical across llama.cpp,
OpenFold, and ViT — confirmed computationally.

#### Experiment 003: Sequence Forecasting (5 checks)

**Objective**: Train LSTM/GRU on Michigan weather time series.

| Check | Target |
|-------|--------|
| LSTM competitive with persistence | Within 0.10 R² |
| LSTM R² | > 0.80 |
| GRU R² | > 0.80 |
| Horizon sweep | Completed |
| Op analysis | Completed |

**Key finding**: LSTM and GRU both achieve R²≈0.93 on 1-day Tmax forecasts.
Persistence is a strong baseline for autocorrelated weather data — the real
advantage of neural models shows at longer horizons.

#### Experiment 004: Transfer Learning (6 checks)

**Objective**: Transfer Michigan ET₀ model to New Mexico and California.

| Check | Target |
|-------|--------|
| Source model R² | > 0.95 |
| Domain gap NM | Detected |
| Domain gap CA | Detected |
| Fine-tuning NM | Improves |
| Fine-tuning CA | Improves |
| From-scratch baseline | Completed |

**Key finding**: Michigan→NM domain gap is 0.33 R² (different climate regime).
Fine-tuning just the head layer with 200 NM samples bridges most of the gap.
Same pattern as vision transfer (freeze backbone, retrain head).

#### Experiment 005: Isomorphic Pattern Catalog (8 checks)

**Objective**: Map shared primitives across language, protein, vision, physics, time series.

| Check | Target |
|-------|--------|
| Architecture survey | 6 architectures |
| GEMM universality | All architectures |
| Nonlinearity universality | All architectures |
| Cross-domain sharing | > 0 pairs |
| BarraCUDA coverage | ≥ 70% |
| Quantization path | Analyzed |
| Isomorphism theorem | Stated |
| PyTorch op trace | Correct output |

**Key finding**: Six fundamental primitives explain all architectures.
BarraCUDA has WGSL shaders for all six. The Rust evolution team needs
to optimize 6 ops, not 600.

### Grand Total: 48 checks across 5 experiments

## Phase 0+: Scholarly Reproduction Studies

Each reproduction follows the same framework as Phase 0 but targets a published result:
1. **Paper identification** — peer-reviewed paper with reproducible methodology
2. **Faithful reimplementation** — Python/PyTorch following the paper's method
3. **Validation against published results** — quantitative checks vs paper's reported values
4. **Tolerance justification** — documented rationale for any departure from exact reproduction
5. **Isomorphic analysis** — map new ops to BarraCUDA WGSL shaders

### Study 001: PINN Burgers' Equation (6 checks)

**Paper**: Raissi, Perdikaris, Karniadakis (2019) "Physics-informed neural networks." JCP 378:686-707

| Check | Target |
|-------|--------|
| IC validation (Cole-Hopf exact) | error < 1e-6 |
| BC validation | error < 1e-15 |
| Training convergence | best loss < 0.01 |
| L2 relative error | < 15% (paper: 0.06% with L-BFGS) |
| Shock steepening detected | gradient increase confirmed |
| Op analysis (autograd) | GEMM + tanh + autograd mapped |

**Tolerance**: 5.1% vs paper's 0.06% — our Adam-only optimizer vs paper's Adam + L-BFGS. L-BFGS is a P1 BarraCUDA gap.

### Study 002: DeepONet Antiderivative (5 checks)

**Paper**: Lu et al. (2021) "Learning nonlinear operators." Nature Machine Intelligence 3:218-229

| Check | Target |
|-------|--------|
| Data generation (1000 train) | Polynomial basis functions |
| Mean L2 error | < 5% |
| RMSE | < 0.05 |
| Specific operators (2/3 < 0.1) | u(x)=1, u(x)=x, u(x)=sin(πx) |
| Architecture analysis | Branch-trunk ≈ encoder-decoder |

### Study 003: LeNet-5 MNIST (5 checks)

**Paper**: LeCun et al. (1998) "Gradient-based learning." Proc IEEE 86(11):2278-2324

| Check | Target |
|-------|--------|
| MNIST loaded | 60,000 train, 10,000 test |
| Test accuracy | ≥ 98.5% (paper: ~99.05%) |
| All digits ≥ 95% | Per-digit accuracy |
| Feature map dimensions | (1,16,5,5) correct |
| Op analysis (Conv + Pool + FC) | Mapped to BarraCUDA |

### Study 004: LSTM ERA5 Weather (5 checks)

**Data**: Open-Meteo ERA5 reanalysis (ECMWF Copernicus), East Lansing MI, 2020-2023

| Check | Target |
|-------|--------|
| Data loaded (real ERA5) | ≥ 1000 days |
| NSE | > 0.80 |
| RMSE | < 5.0°C |
| Multi-horizon analysis | 1d/3d/7d completed |
| Op analysis (LSTM cell) | Mapped to BarraCUDA |

### Study 005: Quantized Inference (6 checks)

**Papers**: Dettmers et al. (2022) NeurIPS, Frantar et al. (2023) ICLR

| Check | Target |
|-------|--------|
| FP32 baseline R² | > 0.99 |
| INT8 degradation | < 1% R² |
| INT4 degradation | < 5% R² |
| Throughput benchmark | Completed |
| Memory analysis | FP32→Q4 compression ratio |
| BarraCUDA mapping | gemv_q4/q8 validated |

### Grand Total: 27 checks across 5 studies

## Combined: 75 checks across 10 experiments (48 Phase 0 + 27 Phase 0+)

## Phase 1–5b: Rust Validation

The Rust layer cross-validates Python baselines using hardcoded expected values (hotSpring pattern).
BarraCUDA integration extended it to 1750+ Rust+GPU checks across 142 validation binaries.

### neuralSpring-native

| Component | Checks | Pattern |
|-----------|--------|---------|
| `validate_surrogate` | 15 | Global minima + known-values for Rastrigin, Rosenbrock, Ackley |
| `validate_transformer` | 18 | Softmax properties + GELU known-values |
| `validate_metrics` | 10 | R², RMSE, MAE, NSE against analytical expectations |
| 15 Phase 0++ validation binaries | 188 | Paper-specific checks across all domains |
| Rust library tests | 459 | Cross-language validation |
| Rust integration tests | 9 | Cross-module consistency verification |

### BarraCUDA (275+ checks)

| Component | Checks | Pattern |
|-----------|--------|---------|
| stats, linalg, special, optimize | 66 | Analytical / NIST DLMF reference values |
| precision (f64 CPU shaders) | 12 | Exact f64 |
| Tensor API (CPU + GPU) | 90 | WGSL unified path, all activations/ops |
| Tensor f64 API | 35 | f64 GPU ops |
| quantized (Q4/Q8) | 15 | Hand-constructed test vectors |
| linalg ext (SVD, gen eigh) | 17 | Analytical solutions |
| ML inference (MLP + Transformer) | 13 | Python/NumPy baselines |
| FFT (f32/f64/Rfft) | 24 | Analytical DFT definition |
| LogSumExp | 5 | Log-domain stability |

### BarraCUDA CPU Ports (170 checks)

17 modules validated against BarraCUDA CPU math primitives (`rk45_solve`,
`eigh_f64`, `solve_f64`, `chi_squared_sf`, `stats::variance`, `pearson_correlation`).

### GPU Shaders + Pipelines (180+ checks)

17 WGSL shaders in `metalForge/shaders/`, validated via GPU shader binaries,
cross-dispatch binaries, and 7 pure-GPU pipeline binaries.

### Quality Gates

`cargo clippy` (pedantic + nursery), `cargo fmt`, `cargo doc`, `unsafe_code = "forbid"`.
Centralized `tolerances/` module with 58 named constants. `require!` macro for graceful
GPU error handling.

## Evolution Roadmap

| Phase | Focus | Validates | Status |
|-------|-------|-----------|--------|
| 0 | Python/PyTorch baselines (48 checks) | Science correctness | **COMPLETE** |
| 0+ | Scholarly reproductions (31 checks) | Published result fidelity | **COMPLETE** |
| 0++ | Paper reproductions (127 checks) | 15 papers, 4 faculty | **COMPLETE** |
| 1a | neuralSpring Rust validation (910 workspace tests, IPC-first) | Cross-language agreement | **COMPLETE** |
| 1b | BarraCUDA validation (275+ checks) | WGSL shader correctness | **COMPLETE** |
| 1c | Fused ToadStool pipeline | Single-encoder dispatch | **COMPLETE** |
| 1d | 3-way benchmark + evolved shaders | Double-buffered, 4-tier routing | **COMPLETE** |
| 2 | BarraCUDA CPU ports (170 checks) | CPU math fidelity | **COMPLETE** |
| 3 | GPU shader evolution (17 WGSL + pipelines) | GPU-CPU parity | **COMPLETE** |
| 4 | Performance + domain expansion | PINN, DeepONet, MHA, eigh | **COMPLETE** |
| 5a | BarraCUDA GPU Tensor | Spectral + eco GPU validation | **COMPLETE** |
| 5b | Upstream fixes (S-13, S-14/S-15/S-16/S-17) | Pool sync; S-14–S-17 **RESOLVED** upstream | **COMPLETE** |
