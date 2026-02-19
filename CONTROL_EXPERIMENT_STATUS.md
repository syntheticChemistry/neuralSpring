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

## Phase 1 — Rust Validation + BarraCUDA Evolution

### Phase 1a: neuralSpring-Native Validation (43 checks)

| Rust Module | Python Source | Tests | Cross-Validation |
|-------------|-------------|-------|------------------|
| `metrics.rs` | `compute_r2`, `compute_rmse`, `compute_mae` | 3 unit + 10 binary | R², RMSE, MAE, NSE at analytical known-values |
| `surrogate.rs` | `rastrigin_2d`, `rosenbrock_2d`, `ackley_2d` | 6 unit + 15 binary | Global minima + 12 Python-computed reference points |
| `transformer.rs` | `softmax`, `gelu_numpy` | 7 unit + 18 binary | Element-wise match against NumPy to <1e-12 |
| `sequence.rs` | `create_sequences`, `persistence_forecast`, `seasonal_tmax` | 7 unit | Window construction, sigmoid/tanh gates |

### Phase 1b: BarraCUDA Primitives (242 checks)

| Validation Binary | BarraCUDA Module | Checks | Reference Source |
|-------------------|------------------|--------|-----------------|
| `validate_barracuda_stats` | stats (variance, pearson, covariance, norm) | 13 | Analytical formulas |
| `validate_barracuda_linalg` | linalg (solve, lu, eigh, cholesky, tridiag) | 17 | Analytical solutions |
| `validate_barracuda_special` | special (gamma, erf, bessel, polynomials) | 26 | NIST DLMF values |
| `validate_barracuda_optimize` | optimize (nelder_mead, bisect, brent) | 10 | Analytical minima/roots |
| `validate_barracuda_precision` | precision (add, mul, fma, dot, sum) | 12 | Exact f64 |
| `validate_barracuda_tensor` | Tensor API (84 ops, CPU + GPU) | 84 | WGSL unified path |
| `validate_barracuda_tensor_f64` | Tensor f64 (GPU ops) | 35 | f64 GPU ops |
| `validate_barracuda_quantized` | quantized (Q4/Q8 dequant, GEMV) | 15 | Hand-constructed |
| `validate_barracuda_linalg_ext` | linalg ext (SVD, LU inverse, gen eigh) | 17 | Analytical |
| `validate_barracuda_ml_inference` | ML inference (MLP + Transformer) | 13 | Python/NumPy baselines |

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

| Primitive | Validated By | WGSL Shader | Evolved Variant |
|-----------|-------------|-------------|-----------------|
| GEMM | All experiments & studies | matmul.wgsl | **matmul_cpu_tiled.wgsl**, **matmul_gpu_evolved.wgsl** |
| Attention | Exp 002, ML inference | attention.wgsl | **BATCHED_ATTENTION_WGSL** |
| LayerNorm | Exp 002, ML inference | layer_norm.wgsl | **GPU-resident** (no readback) |
| ReLU/GELU/Tanh | Exp 001, Studies 001/003 | relu.wgsl, gelu.wgsl | — |
| Softmax | ML inference | softmax_simple.wgsl | — |
| Log-Softmax | ML inference | log_softmax.wgsl | **GPU-resident** (no readback) |
| LSTM cell | Exp 003, Study 004 | lstm_cell.wgsl | — |
| Conv2d | Study 003 | conv2d.wgsl | — |
| Quantized GEMV | Study 005 | gemv_q4/q8.wgsl | — |

---

## Quality Gates

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
| neuralSpring validate | `make validate-native` | **PASS** — 43/43 |
| BarraCUDA validate | `make validate-barracuda` | **PASS** — 242/242 |
| CI | GitHub Actions: `baselines.yml` + `rust.yml` | Configured |

---

## Evolution Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Synthetic baselines (48 checks) | **COMPLETE** |
| 0+ | Scholarly reproductions (27 checks) | **COMPLETE** |
| 1a | neuralSpring Rust validation (43 checks) | **COMPLETE** |
| 1b | BarraCUDA validation (242 checks) | **COMPLETE** |
| 1c | Fused ToadStool pipeline (46–78×) | **COMPLETE** |
| 1d | 3-way benchmark + double-buffered shaders | **COMPLETE** |
| 2 | Quantized inference on GPU | Planned |
| 3 | Cross-spring integration | Planned |
