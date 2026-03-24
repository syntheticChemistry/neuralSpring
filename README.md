# neuralSpring — Learning, Surrogates, and Isomorphic Patterns

**Last updated**: March 24, 2026 (Session S176 — deep audit: IPC resilience wired, environment centralization, GPU module refactor, integration test expansion).

**The learning layer: ML surrogates, transfer learning, scholarly reproduction, and the shared computational DNA across domains.**

neuralSpring is where models learn. Where airSpring validates clean equations, groundSpring quantifies measurement noise, and hotSpring benchmarks physics simulations, neuralSpring asks: **"can we learn a model that adapts, predicts, and generalizes?"**

```
groundSpring (noise labels) → neuralSpring (learn + adapt) → adapted models for new domains
hotSpring (physics surrogates) → neuralSpring (neural surrogates) → faster-than-simulation predictions
```

Named after neural networks — the adaptive, learning counterpart to hotSpring's
physics-driven computational springs. Both feed BarraCUDA the same six
primitives; neuralSpring proves those primitives produce correct learning across
27 scholarly reproductions and 5 novel composition experiments spanning
evolutionary computation, phylogenetics, game theory, spectral analysis,
population genetics, regulatory biology, biomedical time-series prediction,
and cross-domain reservoir computing.

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

neuralSpring validates these primitives in Python, then hands off to the BarraCUDA team for Rust/WGSL evolution. BarraCUDA has 806+ WGSL shaders covering all of these — neuralSpring provides the **test harness** that proves they produce correct learning across all six primitives.

## Current Status: 397/397 Python PASS + 4500+ Rust+GPU PASS = **4900+ total validation checks**

**S176: Deep audit execution — clippy zero-warning gate restored, provenance environment centralization (20 literals → 2 named constants), IPC resilience wired (RetryPolicy + CircuitBreaker into PetalTongue), GPU module refactor (gpu.rs → gpu/mod.rs + gpu/tests.rs), integration tests expanded (9 → 12, full 49/49 provenance coverage), doc reconciliation. ~1,403 tests, 261 binaries, 466 `.rs` files. V126 handoff.**

**S175: Ecosystem absorption — ValidationSink (5 sinks, 12 tests), cast deny, provenance integrity (4 tests). V125 handoff.**

**S174: Deep audit execution — zero `#[allow()]`, tolerance fidelity (all literals centralized), self-knowledge compliance (dead hints removed, origins neutralized, petalTongue gated), 49 Python provenance headers, CONTRIBUTING.md + SECURITY.md. ~1,385 tests, 261 binaries, 464 `.rs` files. V124 handoff.**

**S173: Typed errors & doc hygiene — spring-level `thiserror` errors, `nucleus_pipeline`/module layout, stale reference cleanup. V123 handoff.**

**S172: Deep evolution & ecosystem absorption — DeviceCapabilities migration (last deprecated GpuDriverProfile usage removed across 11 files), workspace lint inheritance ([workspace.lints] single source of truth), 163 playGround missing-docs resolved, normalize_method IPC absorption, 3 validation binaries smart-refactored by responsibility (942→209 max, 913→189 max, 900→137 max), config centralization (8 env vars), #[allow]→#[expect] complete. 1,380 tests, 0 clippy, 0 fmt, 0 doc warnings. V122 handoff.**

**S171: Deep debt audit execution — `PipelineError` typed error (nucleus_pipeline .expect()→Result),
`POSITIVE_DATA_GUARD` + `R2_DENOMINATOR_FLOOR` named constants (primitives.rs), 2 bench_* removed
from validate_all, metalForge forge lint parity (unwrap_used/expect_used), barraCuda version refs
refreshed (v0.3.5→v0.3.7 across 4 specs + ABSORPTION_TRACKER), 6 new proptests (FASTQ/VCF/WDM),
0 doc warnings on main crate (2 link fixes). 1356 tests (1203 lib + 73 forge + 80 playGround),
0 clippy (pedantic+nursery), 0 fmt, 0 doc warnings. V121 handoff.**

**S170: UniBin compliance — primary binary `neuralspring`, barraCuda v0.3.7, 1320 tests, 0 clippy. V120.**

**S168–S169** (condensed): Deep debt — `expected_source()` provenance fix (9→49+ mappings),
66 clippy→zero, `ipc_client.rs` 885→448 LOC, `TensorSession`/`StatefulPipeline` wired to `Dispatcher`,
`CONTEXT.md`, AGPL LICENSE, graceful shutdown, TCP fallback, zero-copy streaming. V119–V120 handoffs.

**S164–S167** (condensed): Ecosystem evolution — `mul_add()` FMA sweep, `pearson_r` centralized,
`#[allow()]`→`#[expect(reason)]` (zero remaining), ecoBin CI, `capability_registry.toml`,
MSRV 1.87, `total_cmp()`, Edition 2024, health probes, RetryPolicy/CircuitBreaker, 28 proptests.
V115–V118 handoffs.

**S157–S162** (condensed): Modern idiomatic Rust — `IpcError` typed enum, `call_typed()`,
`discover_primal()`, `DispatchOutcome`, `safe_cast`, zero `eprintln!` workspace-wide, Tower Atomic
(reqwest/ring removed — zero C deps), `OrExit<T>`, `deny.toml`. V108–V113 handoffs.

**S155–S156** (condensed): Cross-spring absorption (`primal_names.rs`, `tolerances.py`, provenance
trio), IPC bug fixes (probe_capabilities format, coralreef_bridge socket), typed BiomeOsClient,
3 validators to ValidationHarness. V106–V107 handoffs.

**S146–S154** (condensed): Industry GPU parity (barraCuda beats cuBLAS/cuFFT at target scales),
playGround compute triangle (ToadStool/coralReef IPC, hot/cold benchmarks 7–45×), Squirrel MCP
adapter (16 tools), HuggingFace Model Lab (GPT-2 on barraCuda), niche architecture (`niche.rs`,
deploy graph), capability-based discovery across all clients, tolerance centralization (80+ named),
deep debt (zero magic numbers, zero hardcoded primal names). V99–V107 handoffs.

**barraCuda v0.3.7** at `0649cd0` (standalone, extracted from ToadStool S89): ALL 17 shortcomings resolved.
216 files consume barracuda (211 src + 5 playGround), 178 binaries, 71 wgpu files. 14+ modules
exercised (stats, dispatch, ops, linalg, tensor, device, spectral, numerical, nautilus, shaders, nn,
error, unified\_hardware, prelude). 47 CPU→GPU dispatch ops (~97% GPU), 42 metalForge WGSL shaders.
ToadStool S146 (`751b3849`), coralReef Iteration 49 (`coral-glowplug`). Pure Rust **83.6×** faster
than Python (geomean, 15 domains). **coralForge** — sovereign structure prediction (AlphaFold2/3
Evoformer, IPA, diffusion, pairformer, confidence).

### petalTongue Visualization (Sessions 135–139)

16 domain scenario builders covering all 8 `DataChannel` types (TimeSeries, Spectrum, Gauge, Bar, Scatter3D, **Heatmap**, **Distribution**, **FieldMap**). S139 added search results, streaming I/O quality, Kokkos GPU parity, and industry coverage scenarios. Live training dashboard via `TrainingVisualizer` streaming spectral diagnostics to petalTongue. `neuralspring_ecosystem_dashboard` binary for rendering all 16 tracks simultaneously. 56/56 petalTongue validation checks. `scripts/visualize.sh` for offline/live/render/ecosystem modes. `config.rs` centralizes primal identity, env var names, petalTongue domain/theme.
S130–S150 condensed: Upstream rewires (ToadStool S130→S146, barraCuda v0.3.7, coralReef Iter 7→49),
petalTongue visualization (16 scenario builders, live training dashboard), streaming parsers (FASTA/FASTQ/VCF),
CPU BLAST pipeline, Kokkos parity harness, industry gap analysis, composition experiments (Exp 097–101),
NUCLEUS pipeline executor, playGround (Squirrel MCP, Model Lab, compute triangle), deep debt
(zero inline magic numbers, capability-based discovery, `config.rs`). V88–V101 handoffs.

**Validation tiers**: 24/25 bC (96%) | 23/25 gT (92%) | 15/15 xD (100%) | 10/10 pure GPU all-domains |
5/5 baseCamp sub-theses GPU | 5 WDM surrogates (33/33 Py + 160/160 Rs+GPU) |
3 pub experiments (Py 30/30 + Rs 44/44 + GPU 30/30 + Pipeline 13/13 + Mixed 43/43) |
Phase 4 shader validation 22/22 | Streaming spectral pipeline 28/28 |
NUCLEUS compute dispatch 39/39 | `BarraCUDA` absorption readiness 294/294 |
Dispatch parity 30/30 (CPU↔GPU identical for 26 ops) | Mixed-hardware dispatch 47/47 |
WDM+coralForge CPU↔GPU parity 39/39 | metalForge WDM+coralForge NUCLEUS 41/41 |
Multi-GPU RTX 4070 + TITAN V (NVK): 384/384 bit-identical | CPU↔Python parity 39/39 (1e-10).
Cross-spring rewire: 41/41 (`validate_cross_spring_rewire`) | modern bench 28/28 (`bench_cross_spring_modern`).
S121 rewire: 80/80 (`validate_barracuda_s121_rewire`) — SimpleMlp EOS/Transport + HMM Viterbi/forward dispatcher parity.
**Debt**: Zero TODO/FIXME/MOCK/STUB | zero unsafe (`#![forbid(unsafe_code)]` on all 3 crates) | zero inline magic numbers | zero `#[allow()]` (all `#[expect(reason)]`) | zero unfulfilled expectations | zero C dependencies (Tower Atomic) | 100% SPDX headers | zero mocks in production | all files ≤1000 LOC | `deny.toml` supply-chain hygiene | `OrExit<T>` zero-panic binaries | structured logging (`log::info!/warn!/debug!`) | `temp-env` safe env testing (Rust 2024 ready) | zero `eprintln!` workspace-wide | `safe_cast` module (checked GPU dispatch params) | `resilient_call()` circuit breaker | `DispatchOutcome` RPC classification | 4-format `parse_capability_list()` | `discover_primal()` generic socket discovery | Rust Edition 2024 | proptest property-based testing | MSRV pinned (`rust-version = "1.87"`) | `solve_symmetric` → `barracuda::linalg::solve` | ~1,385 Rust tests (1,199 lib + 72 forge + 80 playGround + 9 integration + 25 tokio).
See `wateringHole/handoffs/` for active handoffs.

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

### Phase 0++ — Paper Reproductions (127/127)

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
| 024: Pangenome Selection | Anderson (2024) | 8/8 | Gene gain/loss dynamics, selection signatures |
| 025: Meta-Population | Anderson (2024) | 8/8 | FST, isolation-by-distance, thermal adaptation |

### baseCamp — Biophysical AI Interpretability (176/176)

Novel cross-domain research applying validated physics/biology primitives
to understanding AI systems as physical systems. 6 library modules,
10 validation binaries composing existing primitives (`eigh`, `anderson_localization`,
`hmm`, `game_theory`, `swarm_robotics`, `immunological_anderson`) with novel analysis pipelines.

| Module | Sub-thesis | Validation | Checks | Key Primitive |
|--------|-----------|-----------|--------|---------------|
| `weight_spectral` | nS-01: Weight Matrices as Disordered Hamiltonians | `validate_weight_spectral` | 21/21 | ESD, IPR, level spacing ratio, Dyson dynamics |
| `information_flow` | nS-02: Information Flow as Wave Propagation | `validate_information_flow` | 22/22 | Depth scale, gate disorder, Hill activation, edge-of-chaos |
| `loss_landscape` | nS-03: Loss Landscapes as Energy Landscapes | `validate_loss_landscape` | 27/27 | Numerical Hessian, Boltzmann, gradient descent, barriers |
| `neural_pgm` | nS-04: Neural Networks as PGMs | `validate_neural_pgm` | 21/21 | Belief propagation, effective rank, OOD detection |
| `agent_coordination` | nS-05: Multi-Agent AI as Quorum Sensing | `validate_agent_coordination` | 23/23 | Graph Laplacian, QS signaling, Anderson transition |
| `immunological_anderson` | nS-06: Immunological Anderson Localization | `validate_immunological_anderson` | 20/20 | AD classification, Pielou evenness, Hill dose-response |
| `immunological_anderson` | nS-06 extended: Gonzales/PK/Lattice/MATRIX | `validate_immunological_anderson_extended` | 28/28 | Dose-response, PK decay, tissue lattice, MATRIX scoring |
| — | GPU parity | `validate_basecamp_gpu` | 14/14 | Pure GPU workload validation |
| — | CPU↔GPU dispatch | `validate_compute_dispatch` | 16/16 | BarraCUDA CPU vs GPU parity |
| — | Mixed hardware | `validate_mixed_hardware` | 14/14 | GPU↔NPU↔CPU dispatch routing |

15 grounding papers (B-01 through B-15): **Primitives validated** (Sessions 54–55).
See `whitePaper/baseCamp/extensions.md` for the full research program.

### WDM Surrogates — Warm Dense Matter (153/153 Rust + 33/33 Python = 186/186)

Machine learning surrogates for warm dense matter plasma properties, extending
hotSpring's MD/DFT physics into ML territory. Open data baselines with full
Python↔Rust parity validation.

| Item | Paper | Py | Rs | GPU | Key Primitive |
|------|-------|----|----|-----|---------------|
| nW-01 | Stanton-Murillo transport coefficients | 4/4 | 30/30 | — | `barracuda::nn::SimpleMlp` 3→H→3, log-space normalization |
| nW-02 | EOS surrogate P(ρ,T), E(ρ,T) | 9/9 | 36/36 | 15/15 | `barracuda::nn::SimpleMlp` 2→H→2, signed-log output |
| nW-03 | S(q,ω) LSTM peak predictor | 5/5 | 27/27 | — | LSTM reservoir on MD time series, R²=0.98 |
| nW-04 | Classical→WDM transfer learning | 4/4 | 6/6 | — | Pre-train MLP on classical, fine-tune on WDM |
| nW-05 | ESN WDM regime classifier | 5/5 | 39/39 | — | ESN classifier, 96.5% accuracy |

**WDM surrogate queue fully closed**: nW-01 through nW-05 all complete.
See `specs/PAPER_REVIEW_QUEUE.md` for the full WDM pipeline.

### Phase 5b — Full-Stack Validation (23 domains, 98+ GPU Tensor checks)

BarraCUDA `Tensor` ops (`matmul`, `transpose`, `tanh`, `sigmoid`, `add`, `mul`)
validated against CPU f64 references across **23 papers** (15 Phase 0++ + 8 Phase 0/0+).
S-14/S-15/S-16 **RESOLVED** upstream (`a4996b34` S39).

| Validator | Domain | Status |
|-----------|--------|--------|
| `validate_barracuda_gpu_spectral` | Spectral (022) | **PASS** (10) |
| `validate_barracuda_gpu_eco` | Ecology (013) | **PASS** (6) |
| `validate_barracuda_gpu_hmm` | HMM (016-018) | **PASS** (5) |
| `validate_barracuda_gpu_fitness` | Evolution (011-015) | **PASS** (7) |
| `validate_barracuda_gpu_nn` | Neural nets | **PASS** (5) |
| `validate_barracuda_gpu_pairwise` | Pairwise distance | **PASS** (5) — S-16 fixed |
| `validate_barracuda_gpu_anderson` | Anderson (023) | **PASS** (7) — S-15 RESOLVED upstream |
| `validate_barracuda_gpu_modes` | MODES (012) | **PASS** (5) |
| `validate_barracuda_gpu_directed` | Directed Evo (014) | **PASS** (5) |
| `validate_barracuda_gpu_swarm` | Swarm (015) | **PASS** (6) |
| `validate_barracuda_gpu_game` | Game Theory (019) | **PASS** (6) |
| `validate_barracuda_gpu_introgression` | Introgression (018) | **PASS** (5) |
| `validate_barracuda_gpu_regulatory` | Regulatory (020) | **PASS** (5) |
| `validate_barracuda_gpu_signal` | Signal (021) | **PASS** (6) |
| `validate_barracuda_gpu_meta_pop` | Meta-pop (025) | **PASS** (5) |
| `validate_barracuda_gpu_transformer` | Transformer (Exp 002) | **PASS** (7) |
| `validate_barracuda_surrogate` | Surrogate (Exp 001) | **PASS** (7) |
| `validate_barracuda_transfer` | Transfer (Exp 004) | **PASS** (7) |
| `validate_barracuda_sequence` | Sequence (Exp 003) | **PASS** (7) |
| `validate_barracuda_lenet` | LeNet-5 (Study 003) | **PASS** (5) |
| `validate_barracuda_lstm` | LSTM (Study 004) | **PASS** (6) |

**Cross-dispatch (xD)**: 15/15 Phase 0++ papers have GPU ↔ CPU parity validation.
6 cross-dispatch binaries, 49 checks, all PASS.

**Upstream parity (uP)**: 10/10 GPU validators have dual-path local↔upstream parity checks (9 bit-identical, 1 Bessel diff 1.95e-3).
`ReduceScalarPipeline` f64 mean validated (5.55e-17 diff). `barracuda::spectral` theory stack validated (17/17 PASS).
**Capability-based dispatch**: 12 validators + evolved HMM use `Gpu::dispatch_1d()` with runtime hardware validation.
Cross-eigensolver: dense Householder+QR vs tridiag Sturm bisection agree at machine epsilon (2.89e-15 at n=64).

#### Rust Validation (3080+ PASS across 210 validation binaries)

Every Python experiment has a companion Rust validation binary following the
hotSpring pattern: `ValidationHarness`, centralized `tolerances/` module (225 named
constants with justification comments), explicit pass/fail exit codes. Library code: 1195 lib tests + 9
integration tests. baseCamp modules add 82 analytical checks + GPU pure 5/5 sub-theses.
WDM surrogates add 6 Rust validators (CPU + BarraCUDA GPU): nW-01 transport 30/30,
nW-02 EOS 36/36 + GPU 15/15, nW-03 S(q,ω) 27/27, nW-04 transfer 6/6, nW-05 ESN 39/39.

### Phase 2 — BarraCUDA CPU Ports (203/203)

24/25 papers validated against BarraCUDA CPU math primitives (96% coverage):

| Binary | Paper | BarraCUDA Primitives | Checks |
|--------|-------|---------------------|--------|
| `validate_barracuda_spectral` | 022 | `linalg::eigh_f64` | 10/10 |
| `validate_barracuda_anderson` | 023 | `linalg::eigh_f64` | 7/7 |
| `validate_barracuda_regulatory` | 020 | `numerical::rk45_solve` | 6/6 |
| `validate_barracuda_signal` | 021 | `numerical::rk45_solve` | 14/14 |
| `validate_barracuda_hmm` | 016 | `stats::variance`, `linalg::solve_f64` | 14/14 |
| `validate_barracuda_introgression` | 018 | `special::chi_squared_sf/cdf` | 11/11 |
| `validate_barracuda_counterdiabatic` | 011 | `stats::variance` | 7/7 |
| `validate_barracuda_modes` | 012 | `stats::variance`, `pearson_correlation` | 7/7 |
| `validate_barracuda_eco` | 013 | `stats::variance` | 6/6 |
| `validate_barracuda_directed` | 014 | `stats::variance` | 7/7 |
| `validate_barracuda_swarm` | 015 | `linalg::solve_f64`, `stats::variance` | 10/10 |
| `validate_barracuda_sate` | 017 | `stats::variance` | 6/6 |
| `validate_barracuda_game` | 019 | `numerical::rk45_solve`, `stats::variance` | 5/5 |
| `validate_barracuda_pangenome` | 024 | `stats::variance`, `stats::pearson_correlation` | 12/12 |
| `validate_barracuda_meta_pop` | 025 | `stats::variance`, `stats::pearson_correlation` | 12/12 |
| `validate_barracuda_pinn` | 001 | `barracuda::tensor::{matmul, tanh}` | 14/14 |
| `validate_barracuda_deeponet` | 002 | `barracuda::tensor::{matmul, dot}` | 9/9 |

Key finding: `rk45_solve` achieves machine-precision agreement with hand-rolled RK4.
`eigh_f64` upgraded to Householder+QR at `77f70b2e` (S-12 absorbed) — 1.75e-14 at n=32.

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
# Python baselines (397/397 PASS, ~10 min)
pip install -r control/requirements.txt
bash scripts/run_all_baselines.sh
bash control/check_drift.sh        # drift detection (re-runs baselines)

# Python unit tests (48 tests, <1 sec)
pip install pytest
python3 -m pytest tests/ -v

# Rust validation (1195 lib + 9 integration)
cargo test --lib --test integration
cargo run --release --bin validate_all   # all validation binaries

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
barraCuda is now a standalone primal (`../barraCuda/crates/barracuda` v0.3.7 at `0649cd0`), extracted from `ToadStool` at S89.
`ToadStool` dispatches across hardware; `BarraCUDA` provides the universal math engine.
neuralSpring calls `barracuda::*` directly — no abstraction layer — matching the hotSpring pattern.
Each Spring evolves independently; the barraCuda team absorbs changes asynchronously.

| BarraCUDA Module | neuralSpring Validation | Binary |
|------------------|------------------------|--------|
| `stats::{variance, pearson_correlation, covariance, norm_cdf}` | 13 checks (analytical) | `validate_barracuda_stats` |
| `linalg::{solve_f64, eigh_f64, cholesky_f64, lu_*, tridiag}` | 17 checks (analytical) | `validate_barracuda_linalg` |
| `linalg::{svd_*, lu_inverse, gen_eigh_f64}` | 17 checks (analytical) | `validate_barracuda_linalg_ext` |
| `special::{gamma, erf, bessel, legendre, hermite, laguerre}` | 26 checks (NIST DLMF) | `validate_barracuda_special` |
| `optimize::{nelder_mead, bisect, brent}` | 10 checks (analytical) | `validate_barracuda_optimize` |
| `shaders::precision::cpu` (add, mul, fma, dot, sum) | 12 checks (exact f64) | `validate_barracuda_precision` |
| **Tensor API** (90 ops — native LN, log-SM, leaky_relu, elu) | 90 checks (WGSL unified) | `validate_barracuda_tensor` |
| **Tensor f64 API** (SumReduce, FusedMap, Norm, etc.) | 35 checks (f64 GPU) | `validate_barracuda_tensor_f64` |
| `shaders::quantized` (dequant Q4/Q8, GEMV) | 15 checks (hand-constructed) | `validate_barracuda_quantized` |
| **ML Inference** (MLP + Transformer end-to-end) | 13 checks (Python baseline) | `validate_barracuda_ml_inference` |
| **FFT** (Cooley-Tukey 1D f32, inverse, Parseval) | 12 checks (analytical DFT) | `validate_barracuda_fft` |
| **LogSumExp** (numerical stability for HMM/softmax) | 5 checks (analytical) | `validate_barracuda_logsumexp` |

### `BarraCUDA` Absorption (all 17 shortcomings ABSORBED)

All 17 neuralSpring shortcomings (S-01..S-17) have been absorbed by `BarraCUDA`.
S-12 (eigensolver accuracy) resolved via Householder+QR — `src/eigh.rs` now
delegates to upstream. S-14..S-17 resolved matmul hang, transpose dispatch, and
pow transcendental issues. Session 89: 3 new BarraCUDA ops wired (HillGateGpu,
MultiObjFitnessGpu, SwarmNnGpu) with dispatch parity 30/30 and mixed-hardware
dispatch 47/47.

| Shortcoming | Fix | Validated |
|-------------|-----|-----------|
| S-01 Per-op dispatch | `TensorSession` single-encoder batch | ✓ |
| S-02 Naive matmul | 4-tier `KernelRouter` | ✓ |
| S-03 MHA z-dispatch | `workgroups_z = seq_len` | ✓ |
| S-04 Softmax pooled | `params.size` uniform | ✓ |
| S-05 leaky_relu Params | `{size, negative_slope}` | ✓ (90/90 PASS) |
| S-06 elu Params | `{size, alpha}` | ✓ (90/90 PASS) |
| S-07 from_buffer pub | `pub fn from_buffer()` | ✓ |
| S-08 layer_norm round-trip | `from_pooled_buffer` | ✓ (native test) |
| S-09 log_softmax round-trip | `from_pooled_buffer` | ✓ (native test) |
| S-10 science_limits CPU | `new_cpu_relaxed()` | ✓ (gpu.rs rewired) |
| S-11 TensorSession limited | ML ops in SessionOp | ✓ |
| S-12 eigh_f64 accuracy | Householder+QR (`77f70b2e`) | ✓ (1.75e-14 at n=32) |

### Shortcomings (Phase 5a/5b) — ALL RESOLVED

| # | Shortcoming | Severity | Status |
|---|-------------|----------|--------|
| S-14 | Naive matmul hang (small square matrices, complex binaries) | Medium | **RESOLVED** upstream (`a4996b34` S39: Naive tier removed) |
| S-15 | Matmul hang when elements have magnitude ≤ 0.1 (RTX 4070 Vulkan) | Critical | **RESOLVED** upstream (`a4996b34` S39) |
| S-16 | 2D transpose dispatch: `optimal_workgroup_size` (256) vs tile size (16) | High | **RESOLVED** upstream (`a4996b34` S39: `const TILE: u32 = 16`) |
| S-17 | `pow(f64,f64)` crashes NVVM/NAK on Ada Lovelace + Volta | High | **RESOLVED** upstream (`c82c23d1` S58: `patch_transcendentals_in_code` covers pow) |

Validators retain conservative data patterns (positive-only, A×B^T) as defense-in-depth.
Full details: `EVOLUTION_READINESS.md` | `wateringHole/handoffs/`

### metalForge GPU Shader Validation (Phase 3c+3d)

| Shader / API | Validation Binary | Checks | Status |
|--------------|-------------------|--------|--------|
| `hmm_forward_log.wgsl` | `validate_gpu_hmm_forward` | 13 | **PASS** |
| `batch_fitness_eval.wgsl` | `validate_gpu_batch_fitness` | 20 | **PASS** |
| `rk4_parallel.wgsl` | `validate_gpu_rk4` | 8 | **PASS** |
| `pairwise_jaccard.wgsl` | `validate_gpu_pangenome` | 6 | **PASS** |
| `locus_variance.wgsl` | `validate_gpu_meta_pop` | 7 | **PASS** |
| `spatial_payoff.wgsl` | `validate_gpu_game_theory` | 5 | **PASS** |
| `batch_ipr.wgsl` | `validate_gpu_anderson` | 5 | **PASS** |
| `pairwise_hamming.wgsl` | `validate_gpu_sate` | 5 | **PASS** |
| `StatefulPipeline` (RK4) | `validate_gpu_stateful_pipeline` | 10 | **PASS** |
| Multi-kernel chain | `validate_gpu_pure_workload` | 7 | **PASS** |
| `DispatchConfig` parity | `validate_cross_dispatch` | 8 | **PASS** |
| `DispatchConfig` genomics | `validate_cross_dispatch_genomics` | 8 | **PASS** |
| `DispatchConfig` extended | `validate_cross_dispatch_extended` | 12 | **PASS** |
| `pairwise_l2.wgsl` | `validate_gpu_modes` | 15 | **PASS** |
| `multi_obj_fitness.wgsl` | `validate_gpu_directed` | 6 | **PASS** |
| `swarm_nn_forward.wgsl` | `validate_gpu_swarm` | 9 | **PASS** |
| `hill_gate.wgsl` | `validate_gpu_signal` | 9 | **PASS** |
| Phase 4b pipelines | `validate_gpu_pipeline_{hmm,ecology,spectral,genomics,modes,directed,signal}` | 32 | **PASS** (HMM→mean_reduce, spatial_payoff→mean_reduce, batch_ipr→mean_reduce, pairwise_jaccard→mean_reduce, pairwise_l2, multi_obj_fitness, hill_gate) |
| `logsumexp_reduce.wgsl` | `validate_gpu_logsumexp` | 5 | **PASS** (Session 43) |
| `stencil_cooperation.wgsl` | `validate_gpu_stencil` | 3 | **PASS** (Session 43) |
| `rk45_adaptive.wgsl` | `validate_gpu_rk45` | 6 | **PASS** (Session 43) |
| `wright_fisher_step.wgsl` | `validate_gpu_wright_fisher` | 4 | **PASS** (Session 43) |
| `GillespieGpu` (upstream) | `validate_gpu_gillespie` | 20 | **PASS** (Session 43) |
| `TaxonomyFcGpu` (upstream) | `validate_upstream_taxonomy` | 3 | **PASS** (Session 43) |
| `KmerHistogramGpu` (upstream) | `validate_upstream_kmer` | 3 | **PASS** (Session 43) |
| `UniFracPropagateGpu` (upstream) | `validate_upstream_unifrac` | 2 | **PASS** (Session 43) |
| `chi_squared` (upstream CPU) | `validate_barracuda_chi_squared` | 13 | **PASS** (Session 43) |
| CPU vs GPU parity (Tensor) | `validate_cpu_gpu_parity` | 17 | **PASS** (Session 43) |
| Dispatch routing (metalForge) | `validate_toadstool_dispatch` | 16 | **PASS** (Session 43) |
| Mixed-hardware dispatch | `validate_mixed_dispatch` | 16 | **PASS** (Session 43) |

Lifecycle tracker: `metalForge/shaders/ABSORPTION_TRACKER.md`

## Evolution Roadmap

- **Phase 0**: Python/PyTorch baselines — validate the science **COMPLETE** (397/397 — 27 papers + 5 WDM + baseCamp + coralForge + pub experiments)
- **Phase 1a**: neuralSpring Rust validation **COMPLETE** (~1,385 Rust tests: 1,199 lib + 72 forge + 80 playGround + 9 integration + 25 tokio, 261 binaries, 67 modules + gpu_ops/ + gpu_dispatch/)
- **Phase 1b**: BarraCUDA validation **COMPLETE** (272 checks — 12 domains incl. ML inference, FFT f32/f64/Rfft, LogSumExp)
- **Phase 1c**: Fused `ToadStool` pipeline **COMPLETE** (46–78× speedup via single-encoder dispatch)
- **Phase 1d**: 3-way benchmark + double-buffered shaders **COMPLETE** (GPU 80× CPU, CPU beats Py at crossover)
- **Phase 2**: BarraCUDA CPU implementations — **COMPLETE** (203 checks — 24/25 papers, 96% coverage)
- **Phase 5b**: Full-stack validation buildout — **COMPLETE** (bC 24/25, gT 23/25, xD 15/15 — all green)
- **Phase 2a**: metalForge hardware characterization — dispatch, cache, bandwidth profiling
- **Phase 3a**: BarraCUDA FFT validation **COMPLETE** (24 checks — f32/f64/Rfft, Parseval, inverse, known pairs)
- **Phase 3b**: BarraCUDA GPU streaming **COMPLETE** (`StatefulPipeline` — 10/10 PASS)
- **Phase 3c**: metalForge GPU shader evolution **COMPLETE** (21 WGSL shaders — 13 upstream + 8 local)
- **Phase 3d**: Pure GPU workload + cross-dispatch **COMPLETE** (45 checks — SP 10 + chain 7 + xd 8 + xd-genomics 8 + xd-extended 12)

### Phase 4a — Performance Benchmarks

The `bench_phase0pp_kernels` binary compares pure Rust math (neuralSpring) to single-thread NumPy at identical problem sizes. Run: `cargo run --release --bin bench_phase0pp_kernels -- --with-python`.

**Python control scripts** (one per kernel):

| Kernel | Script |
|--------|--------|
| HMM forward (3×5000) | `control/hmm_phylo/bench_hmm_forward.py` |
| Replicator dynamics (10k steps) | `control/game_theory/bench_replicator.py` |
| Commutator ‖[A,B]‖_F (64×64) | `control/spectral_commutativity/bench_commutator.py` |
| NK fitness (N=10,K=2, 1000 genotypes) | `control/counterdiabatic/bench_nk_fitness.py` |
| Pairwise Hamming (20×500) | `control/sate_alignment/bench_hamming.py` |
| Jaccard distance (30×500) | `control/pangenome_selection/bench_jaccard.py` |
| RK4 GRN ODE (2000 steps) | `control/regulatory_network/bench_rk4.py` |

**Summary:**

| Kernel | Paper | Rust µs | Python µs | Speedup |
|--------|-------|---------|-----------|---------|
| HMM forward (3×5000) | 016-018 | 330.0 | 12007.6 | 36.4× |
| Replicator dynamics (10k steps) | 019 | 150.0 | 34937.4 | 232.9× |
| Commutator ‖[A,B]‖_F (64×64) | 022 | 334.6 | 23.3 | 0.1× |
| NK fitness (N=10,K=2, 1000 genotypes) | 011 | 17.9 | 14087.2 | 787.1× |
| Pairwise Hamming (20×500) | 017 | 34.3 | 408.3 | 11.9× |
| Jaccard distance (30×500) | 024 | 142.3 | 2045.4 | 14.4× |
| RK4 GRN ODE (2000 steps) | 020-021 | 218.6 | 24659.8 | 112.8× |
| **TOTAL** | | **1227.8** | **88169.0** | **71.8×** |

Rust pure math is 71.8× faster than single-thread NumPy overall. GEMM-heavy operations (commutator: 0.1×) show why GPU WGSL acceleration via BarraCUDA matters.

- **Phase 4a**: Performance benchmarks **COMPLETE** (7 kernels, 71.8× overall — see above)
- **Phase 4b**: Pure GPU end-to-end pipelines **COMPLETE** (7 pipelines, 32/32 PASS — HMM, ecology, spectral, genomics, modes, directed, signal covering Papers 016–024; 3d+4b combined 77/77 PASS for pure GPU + cross-dispatch)
- **Phase 4c**: GPU WGSL kernel benchmarks + GPU PRNG **COMPLETE** — Crossover mapping (GPU wins at >1.5ms CPU work) + Xoshiro128** PRNG shader (5/5 PASS, `xoshiro128ss.wgsl`). Foundation for stochastic GPU algorithms.
- **Phase 4d**: `BarraCUDA` issue resolution **COMPLETE** — S-12 Householder+QR eigensolver (9/9 PASS), S-03b **FULLY RESOLVED** upstream (`ToadStool` `0c998992`: matmul + head_split/head_concat absorbed). New: `src/eigh.rs`, `validate_eigh_accuracy`, `validate_mha_gpu` (upstream wrapper).
- **Phase 4e**: PINN/DeepONet + new GPU domains **COMPLETE** (PINN 16+14, DeepONet 17+9, GPU modes 15, directed 6, swarm 9, signal 9 + 3 pipelines 12)
- **Phase 5e**: Pure GPU promotion — **COMPLETE** (47 CPU→GPU ops via `gpu_dispatch::Dispatcher`, ~97% math on GPU, Phase A 27/27 + Phase B 20/20 + Phase C 18/18 PASS on RTX 4070 + TITAN V NVK)
- **Phase 4**: metalForge shader evolution toward `BarraCUDA` absorption — **Active**
  - Evolve library modules to inline WGSL (hotSpring pattern)
  - Replace hand-rolled math with `barracuda::*` primitives
  - Cross-spring integration (GPU → CPU → NPU)

See `specs/EVOLUTION_MAPPING.md` for the Tier A/B/C module-by-module mapping.

## Quality Gates

| Gate | Command | Status |
|------|---------|--------|
| Python lint | `ruff check control/ scripts/ tests/` | 0 errors |
| Python format | `ruff format --check control/ tests/` | clean |
| Python unit tests | `python3 -m pytest tests/ -v` | 48/48 PASS |
| Python baselines | `bash scripts/run_all_baselines.sh` | 397/397 PASS |
| Rust tests | `cargo test` | 1217 PASS (1195 lib + 9 forge + 13 doc) |
| Rust clippy | `cargo clippy -- -D warnings` | 0 warnings (pedantic+nursery), 0 `#[allow(` in production code |
| Rust coverage | `cargo llvm-cov --lib` | 91.66% line coverage |
| Rust format | `cargo fmt --check` | clean |
| Rust doc | `cargo doc --no-deps` | clean |
| neuralSpring validate | `cargo run --release --bin validate_all` | 220/220 binaries PASS |
| BarraCUDA CPU validate | `make validate-barracuda` | 272/272 PASS |
| BarraCUDA CPU ports | `make validate-barracuda-cpu` | 203/203 PASS (24/25 papers) |
| GPU Tensor validate | Phase 5b validators | 98+ checks (23/25 gT, S-15/S-16 resolved) |
| GPU shader validate | `make validate-gpu` | 108/108 PASS (16 domain shaders) |
| GPU pipeline validate | `make validate-gpu-pipeline` | 77/77 PASS |
| Cross-dispatch | 6 xD validators | 49/49 PASS (15/15 Phase 0++ papers) |
| GPU PRNG validate | `validate_gpu_prng` | 5/5 PASS |
| Phase 4d validate | `validate_eigh_accuracy` + `validate_mha_gpu` | 10/10 PASS (eigh 9 + upstream MHA wrapper 1) |

CI: `.github/workflows/baselines.yml` (Python) + `.github/workflows/rust.yml` (Rust + coverage)

## Directory Structure

```
neuralSpring/
├── control/                    # Phase 0 Python baselines (25 experiments)
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
│   ├── pangenome_selection/   #   Paper 024: Pangenome selection dynamics
│   ├── meta_population/       #   Paper 025: Meta-population differentiation
│   ├── wdm/                    #   WDM surrogates: EOS (nW-02), transport (nW-01), S(q,ω) (nW-03), transfer (nW-04), ESN regime (nW-05)
│   ├── shared/                 #   Shared utilities (Open-Meteo, etc.)
│   └── requirements.txt        #   Pinned dependencies
├── src/                        # Rust library (41 modules + 2 evolved + config + gpu_ops/ + gpu_dispatch/ + streaming/ + search/ + visualization/)
│   ├── lib.rs                  #   Crate root
│   ├── validation.rs           #   ValidationHarness (hotSpring pattern)
│   ├── tolerances/             #   Centralized tolerance constants + runtime introspection
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
│   ├── hmm.rs                  #   Forward/backward/Viterbi/posterior (flat row-major GPU-ready)
│   ├── sate_alignment.rs       #   NJ tree + progressive alignment
│   ├── introgression.rs        #   PhyloNet-HMM introgression detection
│   ├── game_theory.rs          #   PD, Snowdrift, replicator, QS spatial
│   ├── regulatory_network.rs   #   GRN ODE with Hill functions
│   ├── signal_integration.rs   #   Two-input Hill AND gate
│   ├── spectral_commutativity.rs # Commutator, distance to normal (flat row-major GPU-ready)
│   ├── anderson_localization.rs  # Aubry-André model, IPR
│   ├── pangenome_selection.rs   # PA matrix, gene frequency, selection dynamics
│   ├── meta_population.rs       # FST, Mantel test, thermal adaptation
│   ├── eigh.rs                  #   Eigensolver → delegates to barracuda (S-12 absorbed)
│   ├── weight_spectral.rs       #   baseCamp nS-01: Weight matrix spectral analysis
│   ├── information_flow.rs      #   baseCamp nS-02: Information flow as wave propagation
│   ├── loss_landscape.rs        #   baseCamp nS-03: Loss landscape characterization
│   ├── neural_pgm.rs            #   baseCamp nS-04: Neural networks as PGMs
│   ├── agent_coordination.rs    #   baseCamp nS-05: Multi-agent QS coordination
│   ├── pinn.rs                  #   Physics-informed NN (Raissi et al.)
│   ├── deeponet.rs              #   Operator learning (Lu et al.)
│   ├── primitives.rs            #   Consolidated math: Shannon, Hill, sigmoid, RK4
│   ├── wdm_surrogate.rs         #   nW-02: WDM EOS surrogate (P, E vs ρ, T)
│   ├── wdm_transport.rs         #   nW-01: WDM transport surrogate (D*, η*, λ*)
│   ├── fft.rs                   #   FFT validation helpers (analytical DFT refs)
│   ├── gpu.rs                   #   GPU device wrapper (Gpu::new(), NEURALSPRING_BACKEND)
│   ├── gpu_ops/                 #   41 GPU-accelerated ops (6 submodules: linalg, activation, reduction, bio, population, eigensolver)
│   ├── gpu_dispatch/            #   Capability-based GPU/CPU dispatch (Dispatcher)
│   └── bin/                    #   261 binaries (validate + bench)
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
│       ├── validate_wdm_*.rs              # 6 WDM validators (nW-01, nW-02 CPU+GPU, nW-03, nW-04, nW-05)
│       ├── validate_barracuda_*.rs         # 14 BarraCUDA primitives (272+) + 24 CPU/GPU ports (203+)
│       ├── validate_gpu_*.rs              # 16+ GPU shader binaries (108+ checks)
│       ├── validate_cross_dispatch*.rs    # 6 cross-dispatch validators (49 checks, 15/15 papers)
│       ├── validate_wdm_coral_parity.rs   # CPU↔GPU domain parity for WDM+coralForge (39 checks)
│       ├── validate_metalforge_wdm_coral.rs # metalForge NUCLEUS WDM+coralForge (41 checks)
│       ├── validate_eigh_accuracy.rs      # Householder+QR eigensolver (9 checks)
│       ├── validate_mha_gpu.rs            # GPU head_split/head_concat (10 checks)
│       ├── bench_*.rs                     # 6 benchmark binaries
│       └── validate_all.rs                 # Meta-binary: runs all 220 validators + 2 feature-gated
│   ├── evolved/                #   Active evolutions (2 modules)
│       ├── mod.rs                   # WGSL shader exports (batch_fitness, rk4, mean_reduce)
│       └── mha.rs                   # MHA — thin wrapper to barracuda::ops::mha::MultiHeadAttention (S-03b resolved)
├── tests/                      # Python unit tests (pytest)
├── metalForge/                 # Hardware characterization + shader evolution
│   ├── CROSS_SYSTEM_DISPATCH.md #  GPU→CPU→NPU dispatch strategy
│   ├── ABSORPTION_MANIFEST.md  #   Comprehensive absorption inventory
│   ├── forge/                  #   Rust crate: shader catalog + bindings + dispatch + bridge
│   ├── gpu/nvidia/DISPATCH.md  #   RTX 4070 dispatch latency
│   ├── shaders/                #   WGSL shaders (21 files — 17 original + 4 Session 43)
│   └── fossils/                #   Absorbed evolved code (FOSSIL_RECORD.md)
├── specs/                      # Specifications & tracking
│   ├── EVOLUTION_MAPPING.md    #   Python → Rust → GPU mapping
│   ├── DATA_PROVENANCE.md      #   Dataset sources & licenses
│   ├── TOADSTOOL_HANDOFF.md    #   12 BarraCUDA shortcomings — all absorbed
│   ├── BENCHMARK_ANALYSIS.md   #   Python vs BarraCUDA CPU vs GPU analysis
│   ├── PAPER_REVIEW_QUEUE.md   #   25/25 papers — all complete + baseCamp controls
│   ├── BARRACUDA_REQUIREMENTS.md # BarraCUDA primitive requirements
│   ├── BARRACUDA_USAGE.md      #   Module-level barracuda usage inventory
│   ├── CROSS_SPRING_EVOLUTION.md # Cross-spring shader/primitive provenance
│   └── PURE_GPU_ROADMAP.md     #   Pure GPU target: all math on GPU
├── wateringHole/               # Cross-project handoffs (ToadStool/BarraCUDA)
│   ├── README.md              #   Active handoffs index (following wetSpring pattern)
│   ├── handoffs/              #   Formal handoff documents
│   │   ├── NEURALSPRING_V126_*.md # Current handoffs (V126/S176)
│   │   └── archive/           #   Superseded handoffs (V1–V125 + NestGate/biomeOS/Songbird V1)
├── experiments/                # Experiment journals (hotSpring pattern)
│   └── README.md              #   Journal index (001-123+)
├── whitePaper/                 # Study documentation
│   ├── baseCamp/              #   Per-faculty research briefings
├── scripts/
│   ├── run_all_baselines.sh    #   Orchestrates all 39 Python runs (25 papers + 5 WDM + ML inference + 5 coralForge + 3 pub + 2 nS-06)
│   ├── download_pretrained.py  #   Download pretrained models for nS-01 Paper A (safetensors)
│   └── visualize.sh            #   petalTongue visualization: dump scenarios / live dashboard / render
├── .github/workflows/          # CI
│   ├── baselines.yml           #   Python baselines + lint + tests
│   └── rust.yml                #   Rust test + clippy + validate (261 binaries)
├── CHANGELOG.md                # Release history
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
| `specs/TOADSTOOL_HANDOFF.md` | 17 BarraCUDA shortcomings (S-01–S-17) — all resolved upstream |
| `specs/CROSS_SPRING_EVOLUTION.md` | Cross-spring shader/primitive provenance (hotSpring/wetSpring/neuralSpring) |
| `specs/BENCHMARK_ANALYSIS.md` | Python vs BarraCUDA CPU vs GPU + fused pipeline results |
| `specs/PAPER_REVIEW_QUEUE.md` | 26 papers — all complete + baseCamp + WDM controls |
| `whitePaper/BARRACUDA_EVOLUTION.md` | Shader evolution narrative: Python → CPU → GPU |
| `metalForge/forge/` | Rust crate: shader catalog, binding layouts, dispatch routing, bridge |
| `metalForge/ABSORPTION_MANIFEST.md` | Comprehensive absorption inventory (APIs, shaders, counts) |
| `metalForge/CROSS_SYSTEM_DISPATCH.md` | GPU → CPU → NPU dispatch strategy and validated paths |
| `metalForge/shaders/ABSORPTION_TRACKER.md` | Shader lifecycle (evolve → validate → absorb → retire) |
| `whitePaper/baseCamp/` | Per-faculty research briefings (5 groups, 15 papers) |
| `wateringHole/handoffs/` | Formal ToadStool/BarraCUDA/coralReef handoffs (V126 current: Session 176, barraCuda v0.3.7) |
| `experiments/README.md` | Experiment journals (Sessions 40–176, hotSpring pattern) |
| `CHANGELOG.md` | Release history and session-level changes |

## License

AGPL-3.0-or-later

---

## Part of ecoPrimals

This repo is a domain validation spring in the
[ecoPrimals](https://github.com/ecoPrimals) sovereign computing ecosystem.
Springs reproduce published scientific results using pure Rust and
[barraCuda](https://github.com/ecoPrimals/barraCuda) GPU primitives.

See [wateringHole](https://github.com/ecoPrimals/wateringHole) for ecosystem
documentation and standards.

---

*Initialized: February 16, 2026 | Sessions 40–176: March 24, 2026 | 27 papers + 5 novel compositions + 6 baseCamp sub-theses + 5 WDM surrogates + coralForge + 3 publication experiments | 397 Python + 4000+ Rust+GPU = 4500+ validation checks | ~1,403 Rust tests (1,211 lib + 73 forge + 80 playGround + 12 integration + 25 tokio) | ALL 17 shortcomings RESOLVED upstream (S-01–S-17) | 68 modules, 261 binaries, 466 `.rs` files, 42 WGSL shaders | 232+ named tolerances (centralized registry + `control/tolerances.py` Python mirror + upstream contract pins), 0 clippy (pedantic+nursery+cast deny, all-features), 0 fmt, 0 doc warnings, 100% SPDX, 0 `#[allow(` | barraCuda v0.3.7 at `0649cd0`, nautilus absorbed, 92% coverage | 46 upstream rewires, 250+ barracuda import files | V126 handoff (IPC resilience + environment centralization + GPU refactor) | playGround: Squirrel MCP + HuggingFace Model Lab + compute triangle (ToadStool/coralReef clients) + 70 unit + 13 integration tests | `enable f64;` PTXAS fix | ecoBin compliant (zero C deps in main crates) | capability-based IPC discovery | 16 MCP tools, 16 capabilities | 21 petalTongue scenario tracks + ecosystem dashboard + composition visualization | nucleus\_pipeline Tower→Node→Nest executor | niche deployment: `src/niche.rs` + `graphs/neuralspring_deploy.toml` (biomeOS BYOB Steps 1–4 + provenance trio) | `src/primal_names.rs` (zero duplicate primal name strings) | cross-spring absorption from hotSpring/groundSpring/wetSpring/airSpring + coralReef/loamSpine/rhizoCrypt/sweetGrass*
