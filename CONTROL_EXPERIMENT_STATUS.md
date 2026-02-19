# neuralSpring — Control Experiment Status

**Last updated**: February 19, 2026
**Gate**: Eastgate (i9-12900K, 32 GB DDR5, RTX 4070 12GB, Pop!_OS 22.04)
**Python**: 3.10.12, PyTorch 2.9.0+cu128, NumPy 2.2.6, SciPy 1.15.3
**Rust**: Edition 2021, clippy pedantic + nursery, unsafe_code=forbid
**Grand Total**: 75/75 Python PASS (48 Phase 0 + 27 Phase 0+) + 285/285 Rust validation PASS (43 native + 242 BarraCUDA)

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
| Softmax (full pipeline) | ML inference | softmax_simple.wgsl |
| Multi-Head Attention | ML inference | attention_matmul/softmax/apply.wgsl (evolved MHA) |
| GELU (pipeline) | ML inference | gelu.wgsl |
| MLP end-to-end | ML inference | matmul + add + relu + softmax |
| Transformer encoder block | ML inference | LayerNorm + MHA + FFN + residuals |

---

## Quality Gates (updated 2026-02-19)

| Gate | Tool | Status |
|------|------|--------|
| Python lint | `ruff check` (E/F/W/I/N/UP/B/A/SIM) | **PASS** — 0 errors |
| Python format | `ruff format` | **PASS** — 14 files conformant |
| Python tests | `pytest tests/` | **PASS** — 48 tests |
| Python baselines | `bash scripts/run_all_baselines.sh` | **PASS** — 75/75 |
| Rust test | `cargo test` | **PASS** — 34 unit tests |
| Rust clippy | `cargo clippy` (pedantic+nursery, -D warnings) | **PASS** — 0 warnings |
| Rust format | `cargo fmt --check` | **PASS** |
| Rust doc | `cargo doc --no-deps` | **PASS** |
| neuralSpring validate | `make validate-native` (surrogate, transformer, metrics) | **PASS** — 43/43 |
| BarraCUDA validate | `make validate-barracuda` (stats, linalg, special, optimize, precision, tensor, tensor_f64, quantized, linalg_ext, ml_inference) | **PASS** — 242/242 |
| CI | GitHub Actions: `baselines.yml` (Python), `rust.yml` (Rust) | Configured |

## Audit Remediation (2026-02-18)

Key fixes applied during comprehensive audit:

- **Silent-pass bug**: All 8 PyTorch-dependent scripts now return exit 77 (SKIP) when PyTorch is missing instead of silently passing.
- **DeepONet `.repeat()` bug**: Fixed `np.repeat(n, 0)` → `np.tile()` for correct array broadcasting.
- **Transfer learning scope bug**: Fixed `result_ft` used outside loop; strengthened domain gap and fine-tuning checks.
- **Determinism**: Added explicit `torch.manual_seed(42)` + `np.random.seed(42)` to LeNet-5 and isomorphic catalog.
- **ERA5 robustness**: Added 3-retry with exponential backoff, 60s timeout, safer `np.load` (no `allow_pickle`).
- **Dependencies**: Pinned all versions in `control/requirements.txt` for reproducibility.
- **Provenance**: Added provenance docstrings and tolerance justifications to all 10 scripts.
- **Ruff**: Fixed 441 lint issues (unused imports, f-strings, import ordering, unused variables).
- **Rust scaffolding**: Created `Cargo.toml`, `src/lib.rs`, `metrics.rs`, `surrogate.rs`, `transformer.rs`, `sequence.rs`, 3 validation binaries.
- **Cross-validation**: Rust tests hardcode Python-computed values to verify cross-language agreement to <1e-12.
- **Test suite**: 48 Python tests (pytest) + 34 Rust unit tests + 285 validation binary checks (43 native + 242 BarraCUDA).
- **Data provenance**: Documented all datasets in `specs/DATA_PROVENANCE.md`.
- **Infrastructure**: `Makefile`, `justfile`, `.pre-commit-config.yaml`, two GitHub Actions CI workflows.

### BarraCUDA CPU Integration (2026-02-19)

Following the hotSpring pattern, `barracuda` is a direct path dependency. Nine
validation binaries call `barracuda::*` primitives and compare against analytical /
NIST DLMF / Python-derived baselines. No abstraction layer — each Spring evolves independently.

- **`validate_barracuda_stats`** (13 checks): variance, std\_dev, pearson, covariance, spearman, norm\_cdf/pdf/ppf
- **`validate_barracuda_linalg`** (17 checks): solve\_f64, lu\_det, lu\_solve, eigh\_f64, cholesky\_f64, tridiagonal\_solve
- **`validate_barracuda_special`** (26 checks): gamma, factorial, erf/erfc, bessel J0/J1/I0/K0, Legendre, Hermite, Laguerre
- **`validate_barracuda_optimize`** (10 checks): nelder\_mead (Rosenbrock, Rastrigin), bisect, brent
- **`validate_barracuda_precision`** (12 checks): elementwise add/mul/fma, dot product, Kahan sum
- **`validate_barracuda_tensor`** (84 checks): unified Tensor/WGSL path — relu, gelu, sigmoid, softmax, layer\_norm, matmul, add/sub/mul + tanh, exp, log, sqrt, div, scalar ops, reductions, swish, mish, losses, transpose, evolved ops
- **`validate_barracuda_tensor_f64`** (35 checks): f64 GPU ops — roundtrip, SumReduce, FusedMapReduce, NormReduce, VarianceReduce, WeightedDot, MaxAbsDiff, CosineSimilarity
- **`validate_barracuda_quantized`** (15 checks): Q4/Q8 dequantization, quantized GEMV
- **`validate_barracuda_linalg_ext`** (17 checks): SVD, LU inverse, generalized eigendecomposition
- **`validate_barracuda_ml_inference`** (13 checks): MLP (3-layer, softmax) + pre-norm transformer encoder block vs Python/NumPy baselines

Benchmark binaries: `bench_barracuda_tensor`, `bench_mlp_inference`, `bench_transformer_block`, `bench_fused_inference`.

New infrastructure modules: `src/validation.rs` (`ValidationHarness`), `src/tolerances.rs`, `src/provenance.rs`.
Locally evolved ops: `src/evolved/mha.rs` (MHA workaround), `src/evolved/layer_norm.rs`, `src/evolved/log_softmax.rs`.
Fused pipeline: `src/evolved/fused_pipeline.rs` (ShaderCache + helpers), `src/evolved/fused_mlp.rs` (FusedMlp), `src/evolved/fused_transformer.rs` (FusedTransformer).

### Fused ToadStool Pipeline (2026-02-19)

Eliminates per-op dispatch overhead by pre-compiling shaders, pre-allocating
buffers, and recording all compute passes into a single `CommandEncoder`:

| Model | Per-Op (GPU) | Fused (GPU) | Speedup |
|-------|-------------|-------------|---------|
| MLP | 4.0 ms | 92 µs | **43.6×** |
| Transformer | 13.3 ms | 174 µs | **76.6×** |

Includes GPU-resident head-split/concat WGSL shaders and batched fused
attention, eliminating all CPU round-trips from the MHA workaround.

## Evolution Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Synthetic baselines (48 checks) | **COMPLETE** |
| 0+ | Scholarly reproductions (27 checks) | **COMPLETE** |
| 1a | neuralSpring Rust validation (43 checks) | **COMPLETE** — surrogate, transformer, metrics |
| 1b | BarraCUDA validation (242 checks) | **COMPLETE** — stats, linalg, special, optimize, precision, tensor (84), tensor_f64 (35), quantized, linalg_ext, ml_inference (13) |
| 2 | Quantized inference on GPU | Planned |
| 3 | Cross-spring integration | Planned |
