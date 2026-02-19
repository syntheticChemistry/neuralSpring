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

## Current Status: 75/75 PASS

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

## Quick Start

```bash
# Python baselines (75/75 PASS, ~6 min)
pip install -r control/requirements.txt
bash scripts/run_all_baselines.sh

# Python unit tests (48 tests, <1 sec)
pip install pytest
python3 -m pytest tests/ -v

# Rust validation (23 unit tests + 21 binary checks)
cargo test
cargo run --bin validate_surrogate
cargo run --bin validate_transformer
cargo run --bin validate_metrics

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

BarraCUDA already has the infrastructure; neuralSpring proves it works for learning:

| BarraCUDA Module | neuralSpring Validation |
|------------------|------------------------|
| `nn::Layer` (Linear, Conv2D, etc.) | Exp 001: MLP surrogate training |
| `attention`, `mha`, `flash_attention` | Exp 002: Transformer inference |
| `lstm_cell`, `gru_cell`, `bi_lstm` | Exp 003: Sequence forecasting |
| `gemm_f64`, `gemv_q4/q8` | Exp 005: Isomorphic GEMM patterns |
| `esn_v2`, `snn` | Future: reservoir computing |

## Evolution Roadmap

- **Phase 0**: Python/PyTorch baselines — validate the science **COMPLETE** (75/75)
- **Phase 1**: BarraCUDA Rust port — prove WGSL shaders produce correct gradients **SCAFFOLDED** (4 modules, 3 binaries)
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
| Rust tests | `cargo test` | 23/23 PASS |
| Rust clippy | `cargo clippy -- -D warnings` | 0 warnings (pedantic+nursery) |
| Rust format | `cargo fmt --check` | clean |
| Rust doc | `cargo doc --no-deps` | clean |
| Validation binaries | `cargo run --bin validate_{surrogate,transformer,metrics}` | 21/21 PASS |

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
│   ├── metrics.rs              #   R², RMSE, MAE, NSE
│   ├── surrogate.rs            #   Benchmark functions (Rastrigin, etc.)
│   ├── transformer.rs          #   Softmax, GELU
│   ├── sequence.rs             #   Sequence forecasting primitives
│   └── bin/                    #   hotSpring-pattern validation binaries
│       ├── validate_surrogate.rs
│       ├── validate_transformer.rs
│       └── validate_metrics.rs
├── tests/                      # Python unit tests (pytest)
│   ├── conftest.py             #   Shared path configuration
│   ├── test_benchmark_functions.py
│   ├── test_determinism.py
│   └── test_transformer_ops.py
├── specs/                      # Specifications & tracking
│   ├── EVOLUTION_MAPPING.md    #   Python → Rust → GPU mapping
│   ├── DATA_PROVENANCE.md      #   Dataset sources & licenses
│   ├── BARRACUDA_REQUIREMENTS.md
│   └── PAPER_REVIEW_QUEUE.md
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
| `specs/PAPER_REVIEW_QUEUE.md` | Papers queued for reproduction, prioritized by faculty |

## License

AGPL-3.0-or-later

---

*Initialized: February 16, 2026 | Audit remediation: February 18, 2026*
