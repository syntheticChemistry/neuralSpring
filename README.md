# neuralSpring — Learning, Surrogates, and Isomorphic Patterns

**The learning layer: ML surrogates, transfer learning, scholarly reproduction, and the shared computational DNA across domains.**

neuralSpring is where models learn. Where airSpring validates clean equations, groundSpring quantifies measurement noise, and hotSpring benchmarks physics simulations, neuralSpring asks: **"can we learn a model that adapts, predicts, and generalizes?"**

```
groundSpring (noise labels) → neuralSpring (learn + adapt) → adapted models for new domains
hotSpring (physics surrogates) → neuralSpring (neural surrogates) → faster-than-simulation predictions
```

Named after neural networks — the adaptive, learning counterpart to hotSpring's
physics-driven computational springs. Both feed BarraCUDA the same six
primitives; neuralSpring proves those primitives produce correct learning across
23 scholarly reproductions spanning evolutionary computation, phylogenetics,
game theory, spectral analysis, and regulatory biology.

## The Core Thesis: Isomorphic Learning Patterns

Across seemingly different domains, the same computational primitives appear:

| Domain | Architecture | Key Ops |
|--------|-------------|---------|
| **Language** (llama.cpp) | Transformer | Embed → Attn → FFN → Norm |
| **Protein** (OpenFold) | Evoformer | MSA Attn → Pair Attn → Structure |
| **Vision** (ResNet/ViT) | CNN/ViT | Conv → Pool → FC / Patch → Attn |
| **Physics Surrogate** | MLP/RBF | Sample → Interpolate → Predict |
| **Time Series** (weather) | LSTM/GRU | Embed → Recur → Decode |
| **Evolution** (Dolson) | EA + fitness | Evaluate → Select → Mutate |
| **Phylogenetics** (Liu) | HMM | Forward → Backward → Viterbi |
| **Spectral** (Kachkovskiy) | Eigendecomp | Hamiltonian → Diagonalize → Localize |

The **isomorphic pattern**: at the primitive level, all of these are compositions of:
- **MatMul** (GEMM/GEMV) — the universal workhorse
- **Attention** (scaled dot-product) — weighted information routing
- **Normalization** (LayerNorm, BatchNorm) — scale stabilization
- **Nonlinearity** (ReLU, GELU, SiLU) — feature carving
- **Reduction** (sum, mean, max) — aggregation
- **Quantization** (Q4, Q8, FP16) — deployment compression

neuralSpring validates these primitives in Python, then hands off to the BarraCUDA team for Rust/WGSL evolution. BarraCUDA already has ~100+ WGSL shaders covering most of these — neuralSpring provides the **test harness** that proves they produce correct learning.

## Current Status: 190/190 Python PASS + 409/409 Rust PASS

### Phase 0 — Synthetic Baselines (48/48)

| Experiment | Domain | Tests | Key Question |
|------------|--------|-------|--------------|
| 001: Neural Surrogate | Function approximation | 11/11 | MLP vs RBF on benchmark + FAO-56 |
| 002: Transformer Inference | Language/Protein foundation | 18/18 | Can we reproduce self-attention from scratch? |
| 003: Sequence Forecasting | Time series (weather) | 5/5 | LSTM/GRU on real ERA5 Michigan weather |
| 004: Transfer Learning | Domain adaptation | 6/6 | Real 3-city ERA5 (MI/NM/CA) ET₀ transfer |
| 005: Isomorphic Catalog | Cross-domain analysis | 8/8 | Map shared primitives to BarraCUDA ops |

### Phase 0+ — Scholarly Reproductions (31/31)

| Study | Paper | Tests | Key Result |
|-------|-------|-------|------------|
| 001: PINN Burgers | Raissi et al. (2019) JCP | 8/8 | 5.1% L2 + paper ref (6.7e-4, 2 OOM gap) |
| 002: DeepONet | Lu et al. (2021) NMI | 7/7 | 1.2% L2 + paper ref (MSE 9.27e-7) |
| 003: LeNet-5 MNIST | LeCun et al. (1998) | 5/5 | 98.89% accuracy (Conv+Pool+FC) |
| 004: LSTM ERA5 | Gauch et al. (2021) HESS | 5/5 | NSE=0.849 on real ERA5 weather |
| 005: Quantized | Dettmers (2022), Frantar (2023) | 6/6 | INT8: 0.017% loss, INT4: 0.79% loss (real ERA5 data) |

### Phase 0++ — Paper Reproductions (111/111)

| Paper | Reference | Tests | Key Question |
|-------|-----------|-------|-------------|
| 011: CD Evolution | Iram/Dolson (2020) Nature Physics | 11/11 | Controlled evolution via counterdiabatic driving |
| 012: MODES Toolbox | Dolson et al. (2019) Artif Life | 9/9 | Measuring open-endedness of evolving systems |
| 013: Ecological Dynamics | Dolson & Ofria (2018) GECCO | 7/7 | EA populations as ecological communities |
| 014: Directed Evolution | Dolson et al. (2022) eLife | 8/8 | Lexicase vs tournament for multi-objective |
| 015: Swarm Robotics | Foreback/Dolson (2025) IEEE | 11/11 | Heterogeneous controllers > homogeneous |
| 016: HMM Phylogenetics | Liu et al. (2014) PLoS Comp Bio | 10/10 | Forward/backward as GEMM chain |
| 017: SATé Alignment | Liu et al. (2009) Science | 8/8 | Divide-and-conquer iterative coestimation |
| 018: Introgression | Liu et al. (2015) PNAS | 8/8 | Gene flow detection via PhyloNet-HMM |
| 019: Game Theory & QS | Bruger & Waters (2018) AEM | 8/8 | Quorum sensing resolves cooperation dilemma |
| 020: Regulatory Network | Mhatre et al. (2020) PNAS | 7/7 | One gene → multiple ecological strategies |
| 021: Signal Integration | Srivastava et al. (2011) J Bact | 8/8 | Two-input Hill function = biological AND gate |
| 022: Spectral Commutativity | Kachkovskiy & Safarov (2016) JAMS | 8/8 | Skip connections reduce commutativity distance |
| 023: Anderson Localization | Bourgain & Kachkovskiy (2018) GAFA | 8/8 | Disorder → localization transition |

### Rust Validation (167 native + 242 BarraCUDA = 409 PASS)

Every Python experiment has a companion Rust validation binary following the
hotSpring pattern: `ValidationHarness`, hardcoded expected values, explicit
pass/fail exit codes, centralized tolerances.

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

Progression check: **✓ GPU < CPU < Py** at MLP large + TF medium.

## Quick Start

```bash
# Python baselines (190/190 PASS, ~10 min)
pip install -r control/requirements.txt
bash scripts/run_all_baselines.sh

# Python unit tests (48 tests, <1 sec)
pip install pytest
python3 -m pytest tests/ -v

# Rust validation (109 unit + 6 doc-tests + 409 binary checks)
cargo test
make validate              # all 26 validation binaries
make validate-native       # neuralSpring: 16 binaries (167 checks)
make validate-barracuda    # BarraCUDA: 10 binaries (242 checks)

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
| wetSpring | Taxonomy pipelines, PFAS screening | HMM chains, phylogenetic inference, metagenomics bridge |
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
| **Tensor API** (84 ops incl. activations, losses, evolved) | 84 checks (WGSL unified) | `validate_barracuda_tensor` |
| **Tensor f64 API** (SumReduce, FusedMap, Norm, etc.) | 35 checks (f64 GPU) | `validate_barracuda_tensor_f64` |
| `shaders::quantized` (dequant Q4/Q8, GEMV) | 15 checks (hand-constructed) | `validate_barracuda_quantized` |
| **ML Inference** (MLP + Transformer end-to-end) | 13 checks (Python baseline) | `validate_barracuda_ml_inference` |

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

Full catalog: `specs/TOADSTOOL_HANDOFF.md` (11 issues, all still pending).
Formal handoff: `wateringHole/handoffs/NEURALSPRING_TOADSTOOL_HANDOFF_FEB20_2026.md`

## Evolution Roadmap

- **Phase 0**: Python/PyTorch baselines — validate the science **COMPLETE** (190/190)
- **Phase 1a**: neuralSpring Rust validation **COMPLETE** (167 checks — 16 validation binaries)
- **Phase 1b**: BarraCUDA validation **COMPLETE** (242 checks — 10 domains including ML inference)
- **Phase 1c**: Fused ToadStool pipeline **COMPLETE** (46–78× speedup via single-encoder dispatch)
- **Phase 1d**: 3-way benchmark + double-buffered shaders **COMPLETE** (GPU 80× CPU, CPU beats Py at crossover)
- **Phase 2**: BarraCUDA CPU implementations for Phase 0++ — pure Rust math matching Python baselines
- **Phase 3**: BarraCUDA GPU acceleration — ToadStool unidirectional streaming
- **Phase 4**: Cross-spring integration — live surrogate for Penny Irrigation

See `specs/EVOLUTION_MAPPING.md` for the Tier A/B/C module-by-module mapping.

## Quality Gates

| Gate | Command | Status |
|------|---------|--------|
| Python lint | `ruff check control/ scripts/ tests/` | 0 errors |
| Python format | `ruff format --check control/ tests/` | clean |
| Python unit tests | `python3 -m pytest tests/ -v` | 48/48 PASS |
| Python baselines | `bash scripts/run_all_baselines.sh` | 190/190 PASS |
| Rust tests | `cargo test` | 109 unit + 6 doc-tests PASS |
| Rust clippy | `cargo clippy -- -D warnings` | 0 warnings (pedantic+nursery) |
| Rust coverage | `make coverage` | Target ≥90% (llvm-cov) |
| Rust format | `cargo fmt --check` | clean |
| Rust doc | `cargo doc --no-deps` | clean |
| neuralSpring validate | `make validate-native` | 167/167 PASS |
| BarraCUDA validate | `make validate-barracuda` | 242/242 PASS |

CI: `.github/workflows/baselines.yml` (Python) + `.github/workflows/rust.yml` (Rust + coverage)

## Directory Structure

```
neuralSpring/
├── control/                    # Phase 0 Python baselines (23 experiments)
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
│   ├── counterdiabatic/        #   Paper 011: Counterdiabatic evolution
│   ├── modes/                  #   Paper 012: MODES open-ended evolution
│   ├── eco_dynamics/           #   Paper 013: Ecological dynamics in EC
│   ├── directed_evolution/     #   Paper 014: Directed evolution selection
│   ├── swarm_robotics/         #   Paper 015: Heterogeneous swarm controllers
│   ├── hmm_phylo/              #   Paper 016: HMM forward/backward/Viterbi
│   ├── sate_alignment/         #   Paper 017: SATé divide-and-conquer alignment
│   ├── introgression/          #   Paper 018: Introgression detection (PhyloNet-HMM)
│   ├── game_theory/            #   Paper 019: Game theory & QS cooperation
│   ├── regulatory_network/     #   Paper 020: One gene → multiple strategies
│   ├── signal_integration/     #   Paper 021: Cyclic di-GMP + QS logic gate
│   ├── spectral_commutativity/ #   Paper 022: Skip connections & commutativity
│   ├── anderson_localization/  #   Paper 023: Disorder → localization transition
│   ├── shared/                 #   Shared utilities (Open-Meteo, etc.)
│   └── requirements.txt        #   Pinned dependencies
├── src/                        # Phase 1 Rust library (20 modules)
│   ├── lib.rs                  #   Crate root
│   ├── validation.rs           #   ValidationHarness (hotSpring pattern)
│   ├── tolerances.rs           #   Centralized tolerance constants
│   ├── provenance.rs           #   Python baseline metadata
│   ├── rng.rs                  #   Deterministic Xoshiro256** PRNG
│   ├── metrics.rs              #   R², RMSE, MAE, NSE
│   ├── surrogate.rs            #   Benchmark functions
│   ├── transformer.rs          #   Softmax, GELU
│   ├── sequence.rs             #   Sequence forecasting primitives
│   ├── counterdiabatic.rs      #   NK landscape, CD schedule
│   ├── modes.rs                #   Open-ended evolution metrics
│   ├── eco_dynamics.rs         #   Multi-niche EA, diversity indices
│   ├── directed_evolution.rs   #   5 selection algorithms
│   ├── swarm_robotics.rs       #   Heterogeneous controller EA
│   ├── hmm.rs                  #   Forward/backward/Viterbi/posterior
│   ├── sate_alignment.rs       #   NJ tree + progressive alignment
│   ├── introgression.rs        #   PhyloNet-HMM introgression detection
│   ├── game_theory.rs          #   PD, Snowdrift, replicator, QS spatial
│   ├── regulatory_network.rs   #   GRN ODE with Hill functions
│   ├── signal_integration.rs   #   Two-input Hill AND gate
│   ├── spectral_commutativity.rs # Commutator, distance to normal
│   ├── anderson_localization.rs  # Aubry-André model, IPR, Jacobi eigensolver
│   └── bin/                    #   Validation binaries (26 total)
│       ├── validate_surrogate.rs           # 15 checks
│       ├── validate_transformer.rs         # 18 checks
│       ├── validate_metrics.rs             # 10 checks
│       ├── validate_counterdiabatic.rs     # 19 checks
│       ├── validate_modes.rs               # 9 checks
│       ├── validate_eco_dynamics.rs        # 7 checks
│       ├── validate_directed_evolution.rs  # 7 checks
│       ├── validate_hmm.rs                 # 17 checks
│       ├── validate_game_theory.rs         # 8 checks
│       ├── validate_swarm_robotics.rs      # 7 checks
│       ├── validate_sate_alignment.rs      # 8 checks
│       ├── validate_regulatory_network.rs  # 5 checks
│       ├── validate_signal_integration.rs  # 8 checks
│       ├── validate_introgression.rs       # 13 checks
│       ├── validate_spectral_commutativity.rs # 8 checks
│       ├── validate_anderson_localization.rs  # 8 checks
│       ├── validate_barracuda_*.rs         # 10 BarraCUDA binaries (242 checks)
│       ├── bench_fused_inference.rs        # Fused pipeline 4-way benchmark
│       └── validate_all.rs                 # Meta-binary: runs all 26 validators
│   ├── evolved/                #   Locally evolved BarraCUDA ops
│       ├── fused_pipeline.rs        # ShaderCache + shader router + fused dispatch
│       ├── fused_mlp.rs             # Fused MLP (9 passes, 1 submit)
│       ├── fused_transformer.rs     # Fused Transformer (18 passes, 1 submit)
│       ├── matmul_cpu_tiled.wgsl    # CPU matmul (32x32 double-buffered, vec4)
│       ├── matmul_gpu_evolved.wgsl  # GPU matmul (32x32 double-buffered, vec4)
│       ├── mha.rs                   # MHA workaround (dispatch bug)
│       ├── layer_norm.rs            # GPU-resident layer norm
│       └── log_softmax.rs           # GPU-resident log-softmax
├── tests/                      # Python unit tests (pytest)
├── specs/                      # Specifications & tracking
│   ├── EVOLUTION_MAPPING.md    #   Python → Rust → GPU mapping
│   ├── DATA_PROVENANCE.md      #   Dataset sources & licenses
│   ├── TOADSTOOL_HANDOFF.md    #   11 BarraCUDA shortcomings + local fixes
│   ├── BENCHMARK_ANALYSIS.md   #   Python vs BarraCUDA CPU vs GPU analysis
│   └── PAPER_REVIEW_QUEUE.md   #   23/23 papers — all complete
├── wateringHole/handoffs/      # Cross-project handoffs (ToadStool/BarraCUDA)
├── whitePaper/                 # Study documentation
├── scripts/
│   └── run_all_baselines.sh    #   Orchestrates all 23 Python runs
├── .github/workflows/          # CI
│   ├── baselines.yml           #   Python baselines + lint + tests
│   └── rust.yml                #   Rust test + clippy + validate (26 binaries)
├── Cargo.toml                  # Rust manifest
├── Makefile                    # Task runner
├── justfile                    # Task runner alt (just)
├── CONTROL_EXPERIMENT_STATUS.md
├── README.md
└── LICENSE                     # AGPL-3.0-or-later
```

## Specifications

| Document | Description |
|----------|-------------|
| `specs/EVOLUTION_MAPPING.md` | Tier A/B/C mapping from Python modules → Rust → WGSL shaders |
| `specs/DATA_PROVENANCE.md` | All dataset sources, accession numbers, and licenses |
| `specs/TOADSTOOL_HANDOFF.md` | 11 BarraCUDA shortcomings and local workarounds |
| `specs/BENCHMARK_ANALYSIS.md` | Python vs BarraCUDA CPU vs GPU + fused pipeline results |
| `specs/PAPER_REVIEW_QUEUE.md` | 23 papers — all complete, queued for BarraCUDA CPU port |
| `wateringHole/handoffs/` | Formal ToadStool handoffs (date-stamped, following hotSpring pattern) |

## License

AGPL-3.0-or-later

---

*Initialized: February 16, 2026 | Phase 0++ complete: February 20, 2026 | 23 papers, 190 Python + 409 Rust = 599 total checks | Paper queue cleared — ready for BarraCUDA CPU evolution*
