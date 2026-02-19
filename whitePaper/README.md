# neuralSpring White Paper

## The Isomorphic Learning Engine

**Status**: Phase 0 + Phase 0+ complete — **75/75 quantitative checks pass** (48 synthetic + 27 scholarly)
**Rust**: Phase 1 scaffolded — 4 library modules, 3 validation binaries, 23 unit tests, 21 binary checks

### Key Results — Phase 0 (Synthetic Baselines)

| Experiment | Domain | Tests | Key Finding |
|------------|--------|-------|-------------|
| 001 Neural Surrogate | Function approx + FAO-56 | 11/11 | MLP (4,673 params) replaces FAO-56 chain at R²=0.999, RMSE=0.07 mm/day |
| 002 Transformer | Self-attention mechanics | 18/18 | NumPy SDPA matches PyTorch to <1e-10. Same ops as llama.cpp/OpenFold/ViT |
| 003 Sequence | LSTM/GRU weather forecast | 5/5 | LSTM R²≈0.93 on Michigan Tmax, competitive with persistence baseline |
| 004 Transfer | Michigan→NM/CA adaptation | 6/6 | Domain gap 0.33 R² (NM); fine-tuning with 200 samples bridges it |
| 005 Isomorphic | Cross-domain op catalog | 8/8 | 6 primitives explain ALL architectures. BarraCUDA covers all 6 |

### Key Results — Phase 0+ (Scholarly Reproductions)

| Study | Paper | Tests | Key Finding |
|-------|-------|-------|-------------|
| 001 PINN Burgers | Raissi et al. (2019) JCP | 6/6 | L2 error 5.1% with Adam-only (paper: 0.06% with L-BFGS). Validates MLP + autograd for PDE solving |
| 002 DeepONet | Lu et al. (2021) Nat Mach Intel | 5/5 | 1.2% mean L2 on operator learning. Branch-trunk = encoder-decoder attention |
| 003 LeNet-5 MNIST | LeCun et al. (1998) | 5/5 | 98.89% accuracy. Validates Conv2d + MaxPool + FC pipeline |
| 004 LSTM ERA5 | Real Open-Meteo data | 5/5 | NSE=0.849, RMSE=3.46°C on 4 years of Michigan weather |
| 005 Quantized | INT8/INT4 inference | 6/6 | INT8: 0.017% accuracy loss, INT4: 0.79% loss. Same pipeline as llama.cpp GGML |

### The Isomorphism Theorem

All neural architectures decompose into compositions of six fundamental primitives:

1. **GEMM** (matrix multiply) — 60-90% of all FLOPs
2. **Attention** (scaled dot-product) — learned routing
3. **Normalization** (LN/BN/RMS) — scale stabilization
4. **Nonlinearity** (ReLU/GELU/SiLU) — feature carving
5. **Reduction** (sum/mean/max) — aggregation
6. **Gating** (sigmoid × value) — information filtering

A single engine optimizing these 6 ops in WGSL serves every domain.

### Key Research Questions Answered

1. **Can neural surrogates replace equation chains?** Yes — MLP surrogate for FAO-56 achieves R²>0.999 with 2000 training samples
2. **Is self-attention correct from scratch?** Yes — NumPy matches PyTorch to machine precision
3. **Can LSTM learn weather patterns?** Yes — R²≈0.93 for 1-day Tmax forecasts
4. **Does transfer learning work across climates?** Yes — fine-tuning with 200 NM samples recovers most of the domain gap
5. **Are architectures isomorphic?** Yes — 6 primitives, all in BarraCUDA
6. **Can PINNs solve PDEs from scratch?** Yes — Burgers' equation solved to 5.1% L2 error with Adam-only
7. **Can operators be learned?** Yes — DeepONet learns the antiderivative operator to 1.2% L2
8. **Does quantization preserve accuracy?** Yes — INT8 costs 0.017%, INT4 costs 0.79%

### Cross-Spring Connection

| Spring | Provides | neuralSpring Uses |
|--------|----------|-------------------|
| airSpring | FAO-56 ET₀ model | Surrogate target, real weather data |
| groundSpring | Noise labels, uncertainty | Training robustness, domain gap quantification |
| hotSpring | Physics surrogates (RBF) | Neural surrogate comparison (MLP vs RBF) |
| wetSpring | Taxonomy pipelines | Future: learned classifiers, HMM for metagenomics |

---

## Next Phase: Paper Review Candidates

neuralSpring's Phase 0/0+ validates ML primitives in isolation. The faculty network reveals three professors whose work drives the next phase — applying validated primitives to real scientific problems.

### Constrained Evolution & Evolutionary Computation (Dolson)

Emily Dolson's work is the closest published analog to the constrained evolution methodology described in `gen3/CONSTRAINED_EVOLUTION_FORMAL.md`. Reproducing her work would externally validate the theoretical framework underlying all of ecoPrimals.

| Priority | Paper | Why |
|----------|-------|-----|
| **Tier 1** | Iram, Dolson et al. (2020) "Controlling the speed and trajectory of evolution with counterdiabatic driving." Nature Physics | **Critical**: Counterdiabatic protocols for steering evolution under constraint. This is the physics formalization of what ecoPrimals does with Rust's type system. Reproducing the computational protocol validates the gen3 thesis |
| **Tier 1** | Dolson et al. (2019) "The MODES Toolbox: Measurements of Open-Ended Dynamics in Evolving Systems." Artificial Life 25(1):50-73 | Metrics for measuring whether a system produces genuine novelty. Apply MODES metrics to BarraCUDA's own evolution — does constrained evolution produce open-ended innovation? |
| **Tier 2** | Dolson & Ofria (2018) "Ecological Theory Provides Insights about Evolutionary Computation." GECCO | Ecological dynamics in evolutionary algorithms. Maps directly to primal competition/cooperation in biomeOS — primals as species in an ecosystem |
| **Tier 2** | Dolson et al. (2022) "Artificial selection methods from evolutionary computing show promise for directed evolution of microbes." eLife 11:e79665 | Computational → wet lab bridge. If selection algorithms work for microbes, they work for BarraCUDA shader optimization |
| **Tier 2** | Foreback, Bohm, Dolson (2025) "Leveraging Heterogeneous Controller Representations for Evolutionary Swarm Robotics." IEEE | Heterogeneous controller representations = different primals with different architectures. Swarm robotics optimization ↔ NUCLEUS deployment optimization |

### HMM & Phylogenetic Inference (Liu)

Kevin Liu's sequence analysis methods exercise the same GEMM + state-space primitives that neuralSpring validates. His HMM work is a natural bridge to learned sequence models.

| Priority | Paper | Why |
|----------|-------|-----|
| **Tier 1** | Liu et al. (2014) "An HMM-based Comparative Genomic Framework for Detecting Introgression in Eukaryotes." PLoS Comp Bio 10:e1003649 | PhyloNet-HMM = Hidden Markov Model on genomic data. Validates LSTM/sequence model primitives from a completely different angle — HMM forward/backward/Viterbi are matrix chain multiplications |
| **Tier 2** | Liu et al. (2009) "Rapid and accurate large-scale coestimation of sequence alignments and phylogenetic trees." Science 324:1561-1564 | SATé's divide-and-conquer + iterative refinement = surrogate + optimization loop. Benchmark for GEMM-heavy computation at massive scale |
| **Tier 2** | Liu et al. (2015) "Interspecific Introgressive Origin of Genomic Diversity in the House Mouse." PNAS 112:196-201 | Gene flow detection = transfer learning analog. Introgression between species = knowledge transfer between domains. Exp 004 (transfer learning) from a genomics perspective |

### Game Theory & Cooperation Dynamics (Waters)

Christopher Waters' quorum sensing work frames bacterial cooperation as an evolutionary game theory problem — directly connected to neuralSpring's optimization landscape analysis.

| Priority | Paper | Why |
|----------|-------|-----|
| **Tier 2** | Bruger & Waters (2018) "Maximizing Growth Yield and Dispersal via QS Promotes Cooperation." AEM 84:e00402-18 | Game-theoretic optimization of cooperative behavior. Evolutionary strategy landscapes — the bacterial "fitness landscape" is the same mathematical object as a neural network's loss landscape |
| **Tier 2** | Mhatre et al. (2020) "One gene, multiple ecological strategies: a biofilm regulator is a capacitor for sustainable diversity." PNAS 117:21647-21657 | Single regulatory node enabling phenotypic diversity = single constrained system producing diverse specialized primals |
| **Tier 2** | Srivastava et al. (2011) "Integration of Cyclic di-GMP and Quorum Sensing in the Control of vpsT and aphA." J Bacteriology 193:6331-41 | Multi-input regulatory network = attention mechanism analog. Multiple noisy signals integrated through learned weights |

### Phase 1 Rust Scaffolding (February 2026)

The audit produced a Rust validation layer that cross-checks Python baselines:

| Rust Module | Python Source | Tests | Cross-Validation |
|-------------|-------------|-------|------------------|
| `metrics.rs` | `compute_r2`, `compute_rmse`, `compute_mae` | 3 unit + 10 binary | R², RMSE, MAE, NSE at analytical known-values |
| `surrogate.rs` | `rastrigin_2d`, `rosenbrock_2d`, `ackley_2d` | 6 unit + 5 binary | Global minima + 12 Python-computed reference points |
| `transformer.rs` | `softmax`, `gelu_numpy` | 7 unit + 6 binary | Element-wise match against NumPy to <1e-12 |
| `sequence.rs` | `create_sequences`, `persistence_forecast`, `seasonal_tmax` | 7 unit | Window construction, sigmoid/tanh gates |

Quality gates: `cargo clippy` (pedantic+nursery, `-D warnings`), `cargo fmt`, `cargo doc`, `unsafe_code = "forbid"`.

See `specs/EVOLUTION_MAPPING.md` for the Tier A/B/C module promotion path.

### BarraCUDA Primitive Coverage After Extensions

| Primitive | Phase 0/0+ Status | Faculty Extension Target |
|-----------|-------------------|------------------------|
| GEMM | Validated (surrogate, transformer, PINN) | Liu: sequence alignment, phylogenetic likelihood |
| Attention | Validated (transformer, DeepONet) | Waters: regulatory network graph attention |
| Normalization | Validated (LayerNorm, BatchNorm) | Stable across all extensions |
| Conv2d | Validated (LeNet-5) | Dolson: spatial evolution patterns |
| LSTM cell | Validated (ERA5 weather) | Liu: HMM forward/backward (equivalent structure) |
| Autograd | Validated (PINN Burgers) | Bazavov: lattice QCD force computation |
| Quantized GEMV | Validated (INT8/INT4) | Deployment: all models at inference |
| Evolutionary optimization | NOT YET | **Gap**: Dolson's MODES + counterdiabatic protocols |
| Gillespie simulation | NOT YET | **Gap**: Waters' stochastic c-di-GMP dynamics |
| HMM Viterbi | NOT YET | **Gap**: Liu's PhyloNet-HMM decoding |
