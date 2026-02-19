# neuralSpring — Learning, Surrogates, and Isomorphic Patterns

**The learning layer: ML surrogates, transfer learning, and the shared computational DNA across domains.**

neuralSpring is where models learn. Where airSpring validates clean equations, groundSpring quantifies measurement noise, and hotSpring benchmarks physics simulations, neuralSpring asks: **"can we learn a model that adapts, predicts, and generalizes?"**

```
groundSpring (noise labels) → neuralSpring (learn + adapt) → adapted models for new domains
hotSpring (physics surrogates) → neuralSpring (neural surrogates) → faster-than-simulation predictions
```

## The Core Thesis: Isomorphic Learning Patterns

Across seemingly different domains, the same computational primitives appear:

| Domain | Architecture | Key Ops |
|--------|-------------|---------|
| **Language** (llama.cpp) | Transformer | Embed → Attn → FFN → Norm |
| **Protein** (OpenFold) | Evoformer | MSA Attn → Pair Attn → Structure |
| **Vision** (ResNet/ViT) | CNN/ViT | Conv → Pool → FC / Patch → Attn |
| **Physics Surrogate** | MLP/RBF | Sample → Interpolate → Predict |
| **Time Series** (weather) | LSTM/GRU | Embed → Recur → Decode |

The **isomorphic pattern**: at the primitive level, all of these are compositions of:
- **MatMul** (GEMM/GEMV) — the universal workhorse
- **Attention** (scaled dot-product) — weighted information routing
- **Normalization** (LayerNorm, BatchNorm) — scale stabilization
- **Nonlinearity** (ReLU, GELU, SiLU) — feature carving
- **Reduction** (sum, mean, max) — aggregation
- **Quantization** (Q4, Q8, FP16) — deployment compression

neuralSpring validates these primitives in Python, then hands off to the BarraCUDA team for Rust/WGSL evolution. BarraCUDA already has ~100+ WGSL shaders covering most of these — neuralSpring provides the **test harness** that proves they produce correct learning.

## Current Status: 75/75 Python PASS + 285/285 Rust PASS

### Phase 0 — Synthetic Baselines (48/48)

| Experiment | Domain | Tests | Key Question |
|------------|--------|-------|--------------|
| 001: Neural Surrogate | Function approximation | 11/11 | MLP vs RBF on benchmark + FAO-56 |
| 002: Transformer Inference | Language/Protein foundation | 18/18 | Can we reproduce self-attention from scratch? |
| 003: Sequence Forecasting | Time series (weather) | 5/5 | LSTM/GRU on real Michigan weather data |
| 004: Transfer Learning | Domain adaptation | 6/6 | Michigan ET0 model → different climates |
| 005: Isomorphic Catalog | Cross-domain analysis | 8/8 | Map shared primitives to BarraCUDA ops |

### Phase 0+ — Scholarly Reproductions (27/27)

| Study | Paper | Tests | Key Result |
|-------|-------|-------|------------|
| 001: PINN Burgers | Raissi et al. (2019) JCP | 6/6 | 5.1% L2 error, shock front captured |
| 002: DeepONet | Lu et al. (2021) NMI | 5/5 | 1.2% mean L2 on operator learning |
| 003: LeNet-5 MNIST | LeCun et al. (1998) | 5/5 | 98.89% accuracy (Conv+Pool+FC) |
| 004: LSTM ERA5 | Gauch et al. (2021) HESS | 5/5 | NSE=0.849 on real ERA5 weather |
| 005: Quantized | Dettmers (2022), Frantar (2023) | 6/6 | INT8: 0.017% loss, INT4: 0.79% loss |

### 3-Way Benchmark: Python vs BarraCUDA CPU vs GPU

Target: **Python (slowest) < CPU < GPU (fastest)** — following the hotSpring pattern.

The fused pipeline pre-compiles all shaders, pre-allocates buffers, and records
all compute passes into a **single** `CommandEncoder`. A **4-tier shader router**
driven by `DeviceCapabilities` selects the optimal matmul kernel per dispatch:

| Tier | Shader | Key Technique |
|------|--------|---------------|
| Tiny M,N | naive | Direct global reads |
| CPU | cpu-tiled | 32×32 double-buffered, 8×4 micro-kernel, vec4, 4× k-unroll |
| GPU (small) | tiled | 16×16 shared-memory (high occupancy) |
| GPU (large) | gpu-evolved | 32×32 double-buffered, 2×2 micro-kernel, vec4, 4× k-unroll |

#### Key Results (RTX 4070 + llvmpipe vs Python/NumPy single-thread)

| Scale | Py(1t) | CPU | GPU | CPU/Py | GPU/Py | GPU/CPU |
|-------|--------|-----|-----|--------|--------|---------|
| **MLP large** (3.1M) | 3.0 ms | **2.7 ms** | **178 µs** | **1.1× faster** | **16.8× faster** | 15.1× |
| **TF medium** (103M) | 59 ms | **15.1 ms** | **566 µs** | **3.9× faster** | **104× faster** | 26.8× |
| **TF xlarge** (6.6B) | 232 ms | 1.42 s | **17.8 ms** | — | **13.1× faster** | **79.9×** |

Progression check: **✓ GPU < CPU < Py** at MLP large + TF medium. GPU dominates
CPU at every scale (4–80×). See `specs/BENCHMARK_ANALYSIS.md` for the full
5-scale three-way comparison.

## Quick Start

```bash
# Python baselines (75/75 PASS, ~6 min)
pip install -r control/requirements.txt
bash scripts/run_all_baselines.sh

# Python unit tests (48 tests, <1 sec)
pip install pytest
python3 -m pytest tests/ -v

# Rust validation (34 unit tests + 285 binary checks)
cargo test
make validate              # all 13 validation binaries
make validate-native       # neuralSpring: surrogate, transformer, metrics (43 checks)
make validate-barracuda    # BarraCUDA: stats, linalg, special, optimize, precision,
                           # tensor, tensor_f64, quantized, linalg_ext, ml_inference (242 checks)

# Benchmark fused pipeline (CPU + GPU)
make bench-fused

# All quality gates at once
make check    # or: just check
```

## How neuralSpring Relates to Other Springs

| Spring | What It Provides | What neuralSpring Adds |
|--------|------------------|------------------------|
| hotSpring | Physics surrogates (RBF, SparsitySampler) | Neural surrogates (MLP, attention-based) |
| airSpring | FAO-56 ET0, water balance models | Learned ET0 predictor, transfer to new locations |
| wetSpring | Taxonomy pipelines, PFAS screening | Learned classifiers for noisy spectra |
| groundSpring | Noise characterization, uncertainty labels | Uses noise labels for robust training + adaptation |

## BarraCUDA Connection

BarraCUDA is the **unified math** — the same WGSL shaders run on GPU, CPU, or NPU.
ToadStool provides the execution layer (wgpu) that decides which hardware to target.
neuralSpring calls `barracuda::*` directly — no abstraction layer — matching the hotSpring pattern.
Each Spring evolves independently; the BarraCUDA team absorbs changes asynchronously.

| BarraCUDA Module | neuralSpring Validation | Binary |
|------------------|------------------------|--------|
| `stats::{variance, pearson_correlation, covariance, norm_cdf}` | 13 checks (analytical) | `validate_barracuda_stats` |
| `linalg::{solve_f64, eigh_f64, cholesky_f64, lu_*, tridiag}` | 17 checks (analytical) | `validate_barracuda_linalg` |
| `linalg::{svd_*, lu_inverse, gen_eigh_f64}` | 17 checks (analytical) | `validate_barracuda_linalg_ext` |
| `special::{gamma, erf, bessel, legendre, hermite, laguerre}` | 26 checks (NIST DLMF) | `validate_barracuda_special` |
| `optimize::{nelder_mead, bisect, brent}` | 10 checks (analytical) | `validate_barracuda_optimize` |
| `shaders::precision::cpu` (add, mul, fma, dot, sum) | 12 checks (exact f64) | `validate_barracuda_precision` |
| **Tensor API** (relu, gelu, sigmoid, softmax, layer\_norm, matmul, mse\_loss, evolved ops + tanh, exp, log, sqrt, div, scalar ops, reductions, swish, mish, losses, transpose) | 84 checks (WGSL unified path + evolved ops) | `validate_barracuda_tensor` |
| **Tensor f64 API** (roundtrip, SumReduce, FusedMapReduce, NormReduce, VarianceReduce, WeightedDot, MaxAbsDiff, CosineSimilarity) | 35 checks (f64 GPU ops) | `validate_barracuda_tensor_f64` |
| `shaders::quantized` (dequant Q4/Q8, GEMV) | 15 checks (hand-constructed) | `validate_barracuda_quantized` |
| **ML Inference** (MLP + Transformer end-to-end vs Python) | 13 checks (Python baseline) | `validate_barracuda_ml_inference` |

### Locally Evolved Ops

Where BarraCUDA shortcomings block neuralSpring, we evolve locally (same
pattern as hotSpring). The ToadStool team absorbs at their pace.

| Evolved Module | What It Fixes | Impact |
|----------------|---------------|--------|
| `evolved::layer_norm` | GPU→CPU→GPU round-trip | ~5× (eliminates readback) |
| `evolved::log_softmax` | Same round-trip pattern | ~5× |
| `evolved::mha` | MHA projection dispatch bug | Correctness fix |
| `evolved::fused_pipeline` | Per-op dispatch overhead | **46–78×** |
| `evolved::matmul_cpu_tiled.wgsl` | Naive matmul on CPU | Double-buffered, 8×4 micro-kernel |
| `evolved::matmul_gpu_evolved.wgsl` | No GPU-optimized matmul | Double-buffered, 2×2 micro-kernel |
| `fused_pipeline::MatmulConfig` | No kernel routing | 4-tier `DeviceCapabilities` routing |

Full catalog: `specs/TOADSTOOL_HANDOFF.md` (11 issues).
Formal handoff: `wateringHole/handoffs/NEURALSPRING_TOADSTOOL_HANDOFF_FEB19_2026.md`

## Evolution Roadmap

- **Phase 0**: Python/PyTorch baselines — validate the science **COMPLETE** (75/75)
- **Phase 1a**: neuralSpring Rust validation **COMPLETE** (43 checks — surrogate, transformer, metrics)
- **Phase 1b**: BarraCUDA validation **COMPLETE** (242 checks — 10 domains including ML inference)
- **Phase 1c**: Fused ToadStool pipeline **COMPLETE** (46–78× speedup via single-encoder dispatch)
- **Phase 1d**: 3-way benchmark + double-buffered shaders **COMPLETE** (GPU 80× CPU, CPU beats Py at crossover)
- **Phase 2**: Quantized inference — Q4/Q8 models on consumer GPU
- **Phase 3**: Cross-spring integration — live surrogate for Penny Irrigation

See `specs/EVOLUTION_MAPPING.md` for the Tier A/B/C module-by-module mapping.

## Quality Gates

| Gate | Command | Status |
|------|---------|--------|
| Python lint | `ruff check control/ scripts/ tests/` | 0 errors |
| Python format | `ruff format --check control/ tests/` | 14 files clean |
| Python unit tests | `python3 -m pytest tests/ -v` | 48/48 PASS |
| Python baselines | `bash scripts/run_all_baselines.sh` | 75/75 PASS |
| Rust tests | `cargo test` | 34/34 PASS |
| Rust clippy | `cargo clippy -- -D warnings` | 0 warnings (pedantic+nursery) |
| Rust format | `cargo fmt --check` | clean |
| Rust doc | `cargo doc --no-deps` | clean |
| neuralSpring validate | `make validate-native` | 43/43 PASS |
| BarraCUDA validate | `make validate-barracuda` | 242/242 PASS |

CI: `.github/workflows/baselines.yml` (Python) + `.github/workflows/rust.yml` (Rust)

## Directory Structure

```
neuralSpring/
├── control/                    # Phase 0 Python baselines
│   ├── surrogate/              #   Exp 001: MLP vs RBF surrogates
│   ├── transformer/            #   Exp 002: Self-attention from scratch
│   ├── sequence/               #   Exp 003: LSTM/GRU weather forecasting
│   ├── transfer/               #   Exp 004: Domain adaptation
│   ├── isomorphic/             #   Exp 005: Cross-domain pattern catalog
│   ├── pinn/                   #   Study 001: Physics-informed NN
│   ├── deeponet/               #   Study 002: Operator learning
│   ├── lenet/                  #   Study 003: LeNet-5 MNIST
│   ├── lstm_weather/           #   Study 004: ERA5 weather
│   ├── quantized/              #   Study 005: INT8/INT4 inference
│   └── requirements.txt        #   Pinned dependencies
├── src/                        # Phase 1 Rust library
│   ├── lib.rs                  #   Crate root
│   ├── validation.rs           #   ValidationHarness (hotSpring pattern)
│   ├── tolerances.rs           #   Centralized tolerance constants
│   ├── provenance.rs           #   Python baseline metadata
│   ├── metrics.rs              #   R², RMSE, MAE, NSE
│   ├── surrogate.rs            #   Benchmark functions (Rastrigin, etc.)
│   ├── transformer.rs          #   Softmax, GELU
│   ├── sequence.rs             #   Sequence forecasting primitives
│   └── bin/                    #   hotSpring-pattern validation binaries
│       ├── validate_surrogate.rs     # neuralSpring benchmarks (15 checks)
│       ├── validate_transformer.rs   # neuralSpring softmax/GELU (18 checks)
│       ├── validate_metrics.rs       # neuralSpring R²/RMSE/MAE (10 checks)
│       ├── validate_barracuda_stats.rs    # barracuda stats (13 checks)
│       ├── validate_barracuda_linalg.rs   # barracuda linalg (17 checks)
│       ├── validate_barracuda_special.rs  # barracuda special functions (26 checks)
│       ├── validate_barracuda_optimize.rs # barracuda optimizers (10 checks)
│       ├── validate_barracuda_precision.rs  # barracuda precision (12 checks)
│       ├── validate_barracuda_tensor.rs     # barracuda Tensor/WGSL (84 checks)
│       ├── validate_barracuda_tensor_f64.rs # barracuda Tensor f64 (35 checks)
│       ├── validate_barracuda_quantized.rs  # barracuda quantized (15 checks)
│       ├── validate_barracuda_linalg_ext.rs # barracuda extended linalg (17 checks)
│       ├── validate_barracuda_ml_inference.rs # ML inference MLP+Transformer (13 checks)
│       ├── bench_fused_inference.rs   # Fused pipeline 4-way benchmark
│       └── validate_all.rs          # Meta-binary: runs all validators
│   ├── evolved/                #   Locally evolved BarraCUDA ops
│       ├── fused_pipeline.rs        # ShaderCache + shader router + fused dispatch
│       ├── fused_mlp.rs             # Fused MLP (9 passes, 1 submit)
│       ├── fused_transformer.rs     # Fused Transformer (18 passes, 1 submit)
│       ├── matmul_cpu_tiled.wgsl    # CPU matmul (32x32 double-buffered, vec4, 8x4, k-unroll)
│       ├── matmul_gpu_evolved.wgsl  # GPU matmul (32x32 double-buffered, 2x2, vec4, k-unroll)
│       ├── mha.rs                   # MHA workaround (dispatch bug)
│       ├── layer_norm.rs            # GPU-resident layer norm
│       └── log_softmax.rs           # GPU-resident log-softmax
├── tests/                      # Python unit tests (pytest)
│   ├── conftest.py             #   Shared path configuration
│   ├── test_benchmark_functions.py
│   ├── test_determinism.py
│   └── test_transformer_ops.py
├── specs/                      # Specifications & tracking
│   ├── EVOLUTION_MAPPING.md    #   Python → Rust → GPU mapping
│   ├── DATA_PROVENANCE.md      #   Dataset sources & licenses
│   ├── BARRACUDA_REQUIREMENTS.md
│   ├── TOADSTOOL_HANDOFF.md    #   10 BarraCUDA shortcomings + local fixes
│   ├── BENCHMARK_ANALYSIS.md   #   Python vs BarraCUDA CPU vs GPU analysis
│   └── PAPER_REVIEW_QUEUE.md
├── wateringHole/handoffs/      # Cross-project handoffs (ToadStool/BarraCUDA)
├── scripts/
│   └── run_all_baselines.sh    #   Orchestrates all Python runs
├── .github/workflows/          # CI
│   ├── baselines.yml           #   Python baselines + lint + tests
│   └── rust.yml                #   Rust test + clippy + validate
├── whitePaper/                 # Study documentation
├── Cargo.toml                  # Rust manifest
├── pyproject.toml              # Python tooling config
├── Makefile                    # Task runner (make check, make test, etc.)
├── justfile                    # Task runner alt (just check — requires just)
├── .pre-commit-config.yaml     # Pre-commit hooks (ruff + cargo)
├── CONTROL_EXPERIMENT_STATUS.md
├── README.md
└── LICENSE                     # AGPL-3.0-or-later
```

## Specifications

| Document | Description |
|----------|-------------|
| `specs/EVOLUTION_MAPPING.md` | Tier A/B/C mapping from Python modules → Rust → WGSL shaders |
| `specs/DATA_PROVENANCE.md` | All dataset sources, accession numbers, and licenses |
| `specs/BARRACUDA_REQUIREMENTS.md` | GPU kernel requirements and gap analysis |
| `specs/TOADSTOOL_HANDOFF.md` | 10 BarraCUDA shortcomings and local workarounds |
| `specs/BENCHMARK_ANALYSIS.md` | Python vs BarraCUDA CPU vs GPU + fused pipeline results |
| `specs/PAPER_REVIEW_QUEUE.md` | Papers queued for reproduction, prioritized by faculty |
| `wateringHole/handoffs/` | Formal ToadStool handoff (following hotSpring pattern) |

## License

AGPL-3.0-or-later

---

*Initialized: February 16, 2026 | Audit remediation: February 18, 2026 | BarraCUDA validation: February 19, 2026 | 3-way benchmark + double-buffered shaders: February 19, 2026*
