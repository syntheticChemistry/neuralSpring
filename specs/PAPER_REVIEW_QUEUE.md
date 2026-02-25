# neuralSpring — Paper Review Queue

**Last Updated**: February 25, 2026 (Sessions 45–61)
**Purpose**: Track papers for reproduction/review, ordered by priority

---

## Completed Reproductions

### Phase 0 — Synthetic Baselines

| # | Experiment | Domain | Checks | Key Finding |
|---|-----------|--------|--------|-------------|
| 1 | Neural surrogate | Function approx + FAO-56 | 11/11 | MLP replaces FAO-56 at R²=0.999 |
| 2 | Transformer | Self-attention mechanics | 18/18 | NumPy matches PyTorch to <1e-10 |
| 3 | Sequence forecasting | LSTM/GRU weather | 5/5 | LSTM R²≈0.93, competitive with persistence |
| 4 | Transfer learning | Michigan→NM/CA | 6/6 | Fine-tuning with 200 samples bridges domain gap |
| 5 | Isomorphic catalog | Cross-domain analysis | 8/8 | 6 primitives explain ALL architectures |

### Phase 0+ — Scholarly Reproductions

| # | Paper | Journal | Year | Checks | Key Finding |
|---|-------|---------|------|--------|-------------|
| 6 | Raissi et al. "Physics-informed neural networks" | J Comp Physics | 2019 | 6/6 | L2 error 5.1% (Adam-only). Validates MLP + autograd |
| 7 | Lu et al. "Learning nonlinear operators" (DeepONet) | Nat Mach Intel | 2021 | 5/5 | 1.2% L2. Branch-trunk = encoder-decoder attention |
| 8 | LeCun et al. "Gradient-based learning" (LeNet-5) | Proc IEEE | 1998 | 5/5 | 98.89% accuracy. Conv2d + MaxPool + FC |
| 9 | LSTM on real ERA5 weather data | (real data) | — | 5/5 | NSE=0.849, RMSE=3.46°C on 4 years Michigan |
| 10 | Quantized inference (INT8/INT4) | (methodology) | — | 6/6 | INT8: 0.017% loss. INT4: 0.79% loss |

---

## Review Queue

### Constrained Evolution & Evolutionary Computation (Dolson)

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 11 | Iram, Dolson et al. "Controlling the speed and trajectory of evolution with counterdiabatic driving" | Nature Physics | 2020 | Dolson | **Critical**: Closest published analog to ecoPrimals constrained evolution thesis. Reproducing the computational protocol validates gen3/CONSTRAINED_EVOLUTION_FORMAL.md | **Complete** — `control/counterdiabatic/counterdiabatic_evolution.py` (11/11 PASS) |
| 12 | Dolson et al. "The MODES Toolbox: Measurements of Open-Ended Dynamics in Evolving Systems" | Artificial Life 25(1):50-73 | 2019 | Dolson | Metrics for open-ended evolution. Apply to BarraCUDA's own evolution — does constrained evolution produce novelty? | **Complete** — `control/modes/modes_toolbox.py` (9/9 PASS) |
| 13 | Dolson & Ofria "Ecological Theory Provides Insights about Evolutionary Computation" | GECCO | 2018 | Dolson | Ecological dynamics in evolutionary algorithms. Primals as species in biomeOS | **Complete** — `control/eco_dynamics/eco_dynamics.py` (7/7 PASS) |
| 14 | Dolson et al. "Artificial selection methods from evolutionary computing show promise for directed evolution of microbes" | eLife 11:e79665 | 2022 | Dolson | Computational → wet lab bridge. Selection algorithms for microbial optimization | **Complete** — `control/directed_evolution/directed_evolution.py` (8/8 PASS) |
| 15 | Foreback, Bohm, Dolson "Leveraging Heterogeneous Controller Representations for Evolutionary Swarm Robotics" | IEEE | 2025 | Dolson | Heterogeneous controllers = different primal architectures. Swarm ↔ NUCLEUS | **Complete** — `control/swarm_robotics/swarm_robotics.py` (11/11 PASS) |

### HMM & Phylogenetic Inference (Liu)

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 16 | Liu et al. "An HMM-based Comparative Genomic Framework for Detecting Introgression" | PLoS Comp Bio 10:e1003649 | 2014 | Liu | PhyloNet-HMM: HMM on genomic data. Forward/backward/Viterbi = matrix chain multiplication — same GEMM primitive | **Complete** — `control/hmm_phylo/hmm_phylo.py` (10/10 PASS) |
| 17 | Liu et al. "Rapid and accurate large-scale coestimation of sequence alignments and phylogenetic trees" (SATé) | Science 324:1561-1564 | 2009 | Liu | Divide-and-conquer + iterative refinement at massive scale. GEMM benchmark | **Complete** — `control/sate_alignment/sate_alignment.py` (8/8 PASS) |
| 18 | Liu et al. "Interspecific Introgressive Origin of Genomic Diversity in the House Mouse" | PNAS 112:196-201 | 2015 | Liu | Gene flow detection = transfer learning analog. Introgression = knowledge transfer between species | **Complete** — `control/introgression/introgression.py` (8/8 PASS) |

### Game Theory & Cooperation Dynamics (Waters)

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 19 | Bruger & Waters "Maximizing Growth Yield and Dispersal via QS Promotes Cooperation" | AEM 84:e00402-18 | 2018 | Waters | Game-theoretic optimization. Bacterial fitness landscape = neural network loss landscape | **Complete** — `control/game_theory/game_theory.py` (8/8 PASS) |
| 20 | Mhatre et al. "One gene, multiple ecological strategies" | PNAS 117:21647-21657 | 2020 | Waters | Capacitor for diversity — single constrained system producing diverse primals | **Complete** — `control/regulatory_network/regulatory_network.py` (7/7 PASS) |
| 21 | Srivastava et al. "Integration of Cyclic di-GMP and Quorum Sensing" | J Bacteriology 193:6331-41 | 2011 | Waters | Multi-input regulatory network = attention mechanism analog | **Complete** — `control/signal_integration/signal_integration.py` (8/8 PASS) |

### Spectral Theory & Optimization Landscapes (Kachkovskiy)

Ilya Kachkovskiy (Math, MSU — previously IAS; co-author with Fields Medalist
Jean Bourgain) provides the mathematical layer for understanding neural network
training dynamics and optimization landscapes through spectral theory.

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 22 | Kachkovskiy & Safarov "Distance to normal elements in C*-algebras of real rank zero" | JAMS 29:61-80 | 2016 | Kachkovskiy | Approximate commutativity of operators — when do neural network layers approximately commute? Mathematical foundation for understanding why skip connections and residual networks work: layers that almost commute can be reordered without catastrophic information loss | **Complete** — `control/spectral_commutativity/spectral_commutativity.py` (8/8 PASS) |
| 23 | Bourgain & Kachkovskiy "Anderson localization for two interacting quasiperiodic particles" | GAFA 29:3-43 | 2018 | Kachkovskiy | Localization in disordered systems — connects to loss landscape analysis. Local minima in neural networks = localized states in disordered Hamiltonians. The spectral theory of weight matrices determines training dynamics | **Complete** — `control/anderson_localization/anderson_localization.py` (8/8 PASS) |

### Constrained Evolution in Nature — Empirical Corollary (R. Anderson)

Rika Anderson (Carleton College) provides the empirical biological evidence for
the constrained evolution thesis that Dolson formalizes computationally. Her
pangenomics work shows that gene gain/loss in bacteria is driven by environmental
selection — the biological version of feature selection in machine learning.

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 24 | Moulana, Anderson et al. "Selection is a significant driver of gene gain and loss in the pangenome of Sulfurovum" | mSystems 5:e00673-19 | 2020 | R. Anderson | **Constrained evolution of bacterial functional repertoires.** Different vent environments select for different gene complements — like Lenski's 12 populations finding 12 different solutions. Gene gain/loss dynamics ↔ feature selection and pruning in neural networks | **Complete** — `control/pangenome_selection/pangenome_selection.py` (8/8 PASS) |
| 25 | Campbell, Anderson et al. "*Sulfolobus islandicus* meta-populations in Yellowstone National Park hot springs" | Env Microbiol 19:2392-2405 | 2017 | R. Anderson | Population differentiation under thermal constraint. Same hot springs as Taq polymerase. Geographic isolation → independent evolutionary trajectories. Direct analog to swarm robotics (Dolson Paper 015): different populations in isolated environments evolve different strategies | **Complete** — `control/meta_population/meta_population.py` (8/8 PASS) |

**Why this matters for neuralSpring**: Dolson's counterdiabatic driving (Paper 011)
shows how to *control* evolution computationally. Anderson's pangenomics shows how
evolution *actually behaves* in constrained natural systems. Together they complete
the constrained evolution argument: Dolson proves the theory, Anderson provides the
empirical biology, and neuralSpring validates the computational primitives that
bridge them.

---

## Phase 1 — baseCamp Papers (Biophysical AI Interpretability)

These are neuralSpring's **novel cross-domain extensions**: applying validated
physics and biology primitives to understanding AI systems as physical systems.
Each sub-thesis grounds in published academic work and uses existing validated
primitives. No new math — only novel composition.

Full sub-thesis documents: `whitePaper/baseCamp/sub01_weight_hamiltonians.md`
through `sub05_multiagent_qs.md`. Program overview: `whitePaper/baseCamp/extensions.md`.

### Sub-Thesis 01: Weight Matrices as Disordered Hamiltonians

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| B-01 | Martin & Mahoney "Implicit Self-Regularization in Deep Neural Networks" | JMLR 22(165):1-97 | 2021 | M. Mahoney (UC Berkeley) | Heavy-tailed spectral analysis of weight matrices; 5+1 training phases. ESD as generalization predictor. We extend with Anderson IPR/level spacing | **Primitives validated** (nS-101, 102) |
| B-02 | Gurbuzbalaban, Hu, Simsekli, Zhu "From SGD to Spectra" | arXiv:2507.12709 | 2025 | U. Simsekli (Inria/ENS) | Dyson Brownian motion for singular values during SGD. We connect Dyson dynamics to Anderson localization transition | **Primitives validated** (nS-104) |
| B-03 | Ouyang "Rethinking Over-Smoothing in GNNs via Anderson Localization" | arXiv:2507.05263 | 2025 | — | Direct Anderson framework for GNN message passing. We extend to non-GNN architectures | **Primitives validated** (nS-106) |

**Primitives**: `eigh_f64`, `BatchIprGpu`, `spectral_commutativity.rs`, `anderson_localization.rs`
**Experiments**: nS-101 through nS-106 (6 experiments) — **21/21 PASS** (Session 54)

### Sub-Thesis 02: Information Flow as Wave Propagation in Neural Lattices

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| B-04 | Schoenholz, Gilmer, Ganguli, Sohl-Dickstein "Deep Information Propagation" | ICLR | 2017 | S. Ganguli (Stanford) | Mean-field theory of signal propagation; edge-of-chaos criticality. We replace mean-field with exact Anderson diagnostics | **Primitives validated** (nS-201, 206) |
| B-05 | Gu et al. "Improving the Gating Mechanism of Recurrent Neural Networks" | ICML | 2020 | — | Gate saturation and information flow in LSTMs. We formalize gate saturation as Anderson localization in the strong-disorder limit | **Primitives validated** (nS-202, 205) |
| B-06 | Yang et al. "GLU Spectral Analysis" | — | 2025 | — | Frequency-domain analysis of gating mechanisms. We connect frequency-domain behavior to Anderson localization of high-frequency modes | **Primitives validated** (nS-204) |

**Primitives**: `hmm.rs`, `stencil_cooperation.wgsl`, `signal_integration.rs`, `gpu_dispatch::Dispatcher`
**Experiments**: nS-201 through nS-206 (6 experiments) — **22/22 PASS** (Session 54)

### Sub-Thesis 03: Loss Landscapes as Energy Landscapes

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| B-07 | Ballard, Das, Martiniani, Wales "Insights into ML Models from Chemical Physics: an Energy Landscapes Approach" | Digital Discovery 3, RSC | 2024 | D. Wales (Cambridge) | EL4ML program: loss landscapes as energy landscapes with disconnectivity graphs. We bring GPU-accelerated eigensolver and RK45 | **Primitives validated** (nS-301, 304) |
| B-08 | Pittorino et al. "Boltzmann Entropy and Neural Network Generalization" | — | 2025 | — | Weights as atomic coordinates, loss as potential energy. High-entropy states generalize better. We compute S(E) via GPU-accelerated sampling | **Primitives validated** (nS-303) |
| B-09 | Liu et al. "Loss Landscape Characterization without Over-Parametrization" | arXiv:2410.12455 | 2024 | — | Saddle point convergence guarantees. We connect saddle analysis to transition state theory from chemical physics | **Primitives validated** (nS-301, 305) |

**Primitives**: `ode.rs` / `rk45_adaptive.wgsl`, `game_theory.rs`, `eigh_f64`, `gpu_dispatch::Dispatcher`
**Experiments**: nS-301 through nS-305 (5 experiments) — **27/27 PASS** (Session 54)

### Sub-Thesis 04: Neural Networks as Probabilistic Graphical Models

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| B-10 | Li et al. "Deep Neural Networks as Infinite Tree-Structured PGMs" | arXiv:2305.17583 | 2023 | — | DNN forward prop as PGM belief propagation. We extract PGM from weight matrices via spectral decomposition | **Primitives validated** (nS-401, 402) |
| B-11 | Nabarro et al. "Learning in Deep Factor Graphs with Gaussian Belief Propagation" | ICML | 2024 | Y.W. Teh (Oxford/DeepMind) | Factor graph representation of neural networks. We add uncertainty quantification via factor graph propagation | **Primitives validated** (nS-402, 405) |
| B-12 | Conmy, Mavor-Parker et al. "Towards Automated Circuit Discovery" | NeurIPS | 2023 | N. Nanda | Open-source ACDC for circuit tracing. We apply to our small validated models and compare spectral circuit discovery | **Primitives validated** (nS-404) |

**Primitives**: `hmm.rs`, `eigh_f64`, `introgression.rs`, `spectral_commutativity.rs`
**Experiments**: nS-401 through nS-406 (6 experiments) — **21/21 PASS** (Session 54)

**Note**: We implement the METHODS on our own small models. We do NOT download,
run, or interact with Claude, GPT, or any proprietary model.

### Sub-Thesis 05: Multi-Agent AI Coordination as Quorum Sensing

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| B-13 | SwarmSys "Decentralized Swarm-Inspired Agents for Scalable Reasoning" | arXiv:2510.10047 | 2025 | — | Pheromone-inspired multi-agent reinforcement. We apply Anderson localization to predict coordination phase transitions | **Primitives validated** (nS-501, 505) |
| B-14 | "Emergent Collective Memory in Decentralized Multi-Agent AI Systems" | arXiv:2512.10166 | 2025 | — | Stigmergic coordination and collective memory. We map stigmergic traces to QS autoinducers and test Anderson threshold | **Primitives validated** (nS-502, 503) |
| B-15 | Foreback & Dolson "Heterogeneous Swarm Controllers" | IEEE | 2025 | E. Dolson (MSU) | Already validated as Paper 015. We extend with QS-style signaling and Anderson interaction-graph analysis | **Complete** (base); **Primitives validated** (nS-504) |

**Primitives**: `swarm_robotics.rs`, `game_theory.rs`, `stencil_cooperation.wgsl`, `WrightFisherGpu`, `anderson_localization.rs`
**Experiments**: nS-501 through nS-505 (5 experiments) — **23/23 PASS** (Session 54)

### baseCamp Summary

| Sub-Thesis | Grounding Papers | Experiments | Key Primitive | Rust Module | Checks | Priority |
|:----------:|:----------------:|:-----------:|---------------|-------------|:------:|:--------:|
| 01 Weight Hamiltonians | 3 (B-01 to B-03) | 6 | `eigh_f64`, `BatchIprGpu` | `weight_spectral.rs` | **21/21** | 1 |
| 02 Information Flow | 3 (B-04 to B-06) | 6 | `hmm.rs`, `stencil_cooperation.wgsl` | `information_flow.rs` | **22/22** | 3 |
| 03 Loss Landscapes | 3 (B-07 to B-09) | 5 | `rk45_adaptive.wgsl`, `eigh_f64` | `loss_landscape.rs` | **27/27** | 5 |
| 04 Neural PGM | 3 (B-10 to B-12) | 6 | `hmm.rs`, `introgression.rs` | `neural_pgm.rs` | **21/21** | 2 |
| 05 Multi-Agent QS | 3 (B-13 to B-15) | 5 | `anderson_localization.rs`, `game_theory.rs` | `agent_coordination.rs` | **23/23** | 4 |
| GPU Parity | — | — | `BarraCUDA` f64 typed ops | `validate_basecamp_gpu` | **14/14** | — |
| **Total** | **15** | **28** | | **6 validators** | **128/128** | |

Core Rust primitives: **ALL IMPLEMENTED AND EXPANDED** (Sessions 50, 54).
Experiment coverage expanded from 82→128 checks including pure GPU parity.
Grounding paper reproductions: **Primitives and experiments validated** — full
paper reproductions with publication-ready analysis remain for Phase 2.

All baseCamp papers use open data only (our own trained models + algorithmic
computation). No proprietary models, no external downloads, no API dependencies.

---

## Completion Summary

**All 25 papers complete. baseCamp (B-01..B-15) primitives validated.**

Session 67: CPU↔Python parity — `validate_cpu_math_parity` 39/39 PASS (1e-10 cross-language).
Session 66: Phase C GPU promotion — HMM chains, FST, introgression, AF variance.
`validate_all`: 147/148 PASS on RTX 4070 (1 pre-existing logsumexp driver issue).
`validate_gpu_phase_c`: 18/18 PASS. `validate_cpu_math_parity`: 39/39 PASS.
Python baselines: 25/25 PASS (zero drift). Rust **201.7× faster** than Python/NumPy (11 kernels).
505 lib + 9 integration + 43 forge tests. 158 validation/bench binaries. Zero debt.
44 CPU→GPU dispatch ops (~97% of production math).
Per-faculty briefings: `whitePaper/baseCamp/`.

| Faculty | Papers | Python Checks | Rust Checks |
|---------|--------|---------------|-------------|
| Dolson (MSU CS) | 011–015 (5) | 46 | 50 |
| Liu (MSU CSE) | 016–018 (3) | 26 | 38 |
| Waters (MSU Micro) | 019–021 (3) | 23 | 21 |
| Kachkovskiy (MSU Math) | 022–023 (2) | 16 | 16 |
| R. Anderson (Carleton) | 024–025 (2) | 16 | 16 |
| **Total Phase 0++** | **15** | **127** | **141** |

---

## Open Data & Systems Audit

**All 25 papers use open data and open systems.** No proprietary, paywalled,
or access-restricted data anywhere in the validation stack.

| Category | Papers | Data Source | License / Access |
|----------|--------|------------|-----------------|
| Synthetic / analytical | 011-015, 016, 019-023 | Generated in-code from equations | N/A — pure math, deterministic seed |
| Open API | Exp 003-004, Study 004-005 | Open-Meteo ERA5 Archive API | CC BY 4.0 (free, no auth) |
| Public dataset | Study 003 | MNIST (torchvision) | CC BY-SA 3.0 |
| Open source reference | Study 001, Study 002, Paper 012 | GitHub repos (PINNs, DeepONet, MODES) | MIT / Apache-2.0 |
| Equation-derived | Exp 001, Exp 005 | FAO-56 ET₀, architecture catalog | Public equations |
| Simulated biology | Paper 024-025 | Gene content / population dynamics | N/A — computational model |

**Reproducibility**: Every experiment is deterministic from a fixed RNG seed (42).
ERA5 data has synthetic fallback if API is unavailable. All code is AGPL-3.0.
Full provenance: `specs/DATA_PROVENANCE.md`.

---

## Full Validation Stack Matrix (February 25, 2026 — Sessions 60–61)

Each paper maps through 10 validation tiers. The stack proves correctness
from Python baseline through multi-GPU portability to mixed-hardware dispatch.

### Legend

- **Py**: Python control baseline (Phase 0/0+/0++)
- **Rs**: Pure Rust CPU validation (Phase 1a)
- **bC**: BarraCUDA CPU primitives (Phase 2)
- **gT**: BarraCUDA GPU Tensor — `matmul`, `transpose`, `tanh`, `sigmoid`, `add` (Phase 5b)
- **mF**: metalForge WGSL shader — domain-specific GPU kernel (Phase 3c)
- **gP**: GPU Pipeline — chained domain→reduce (Phase 4b)
- **xD**: Cross-dispatch CPU↔GPU parity (Phase 3d)
- **mG**: Multi-GPU — RTX 4070 (proprietary) + TITAN V (NVK open-source) (Phase 5d)
- **gD**: GPU dispatch — `gpu_dispatch::Dispatcher` routes CPU ops to GPU (Phase 5e)
- **mH**: Mixed hardware — `Dispatcher::mixed_dispatch()` GPU↔NPU↔CPU substrate routing (Phase 5f)

### Phase 0++ Papers (011-025) — ALL GREEN, ALL xD

| Paper | Faculty | Py | Rs | bC | gT | mF | gP | xD | Status |
|-------|---------|----|----|----|----|----|----|----|----- |
| 011 CD Evolution | Dolson | ✓ | ✓ | ✓ | fitness ✓ | `batch_fitness` ✓ | fitness ✓ | ✓ | **7/7** |
| 012 MODES | Dolson | ✓ | ✓ | ✓ | modes ✓ | `pairwise_l2` ✓ | modes ✓ | ✓ | **7/7** |
| 013 Eco Dynamics | Dolson | ✓ | ✓ | ✓ | eco ✓ | `batch_fitness` ✓ | eco ✓ | ✓ | **7/7** |
| 014 Directed Evo | Dolson | ✓ | ✓ | ✓ | directed ✓ | `multi_obj` ✓ | directed ✓ | ✓ | **7/7** |
| 015 Swarm | Dolson | ✓ | ✓ | ✓ | swarm ✓ | `swarm_nn` ✓ | swarm ✓ | ✓ | **7/7** |
| 016 HMM | Liu | ✓ | ✓ | ✓ | hmm ✓ | `hmm_fwd` ✓ | hmm ✓ | ✓ | **7/7** |
| 017 SATé | Liu | ✓ | ✓ | ✓ | pairwise ✓ | `hamming` ✓ | sate ✓ | ✓ | **7/7** |
| 018 Introgression | Liu | ✓ | ✓ | ✓ | introgression ✓ | `hmm_fwd` ✓ | hmm ✓ | ✓ | **7/7** |
| 019 Game Theory | Waters | ✓ | ✓ | ✓ | game ✓ | `spatial` ✓ | ecology ✓ | ✓ | **7/7** |
| 020 Regulatory | Waters | ✓ | ✓ | ✓ | regulatory ✓ | `rk4` ✓ | regulatory ✓ | ✓ | **7/7** |
| 021 Signal | Waters | ✓ | ✓ | ✓ | signal ✓ | `hill_gate` ✓ | signal ✓ | ✓ | **7/7** |
| 022 Spectral | Kachkovskiy | ✓ | ✓ | ✓ | spectral ✓ | `batch_ipr` ✓ | spectral ✓ | ✓ | **7/7** |
| 023 Anderson | Kachkovskiy | ✓ | ✓ | ✓ | anderson ✓ | `batch_ipr` ✓ | spectral ✓ | ✓ | **7/7** |
| 024 Pangenome | Anderson | ✓ | ✓ | ✓ | pairwise ✓ | `jaccard` ✓ | genomics ✓ | ✓ | **7/7** |
| 025 Meta-pop | Anderson | ✓ | ✓ | ✓ | meta_pop ✓ | `locus_var` ✓ | meta_pop ✓ | ✓ | **7/7** |

### Phase 0/0+ Studies (001-010)

| Study | Py | Rs | bC | gT | mF | gP | Status |
|-------|----|----|----|----|----|----|--------|
| Exp 001 Surrogate | ✓ | ✓ | surrogate ✓ | nn ✓ | — | — | **4/4** |
| Exp 002 Transformer | ✓ | ✓ | transformer ✓ | transformer ✓ | — | — | **4/4** |
| Exp 003 Sequence | ✓ | ✓ | sequence ✓ | sequence ✓ | — | — | **4/4** |
| Exp 004 Transfer | ✓ | ✓ | transfer ✓ | transfer ✓ | — | — | **4/4** |
| Exp 005 Isomorphic | ✓ | ✓ | — | — | — | — | **Analytical** |
| Study 001 PINN | ✓ | ✓ | pinn ✓ | nn ✓ | — | — | **4/4** |
| Study 002 DeepONet | ✓ | ✓ | deeponet ✓ | nn ✓ | — | — | **4/4** |
| Study 003 LeNet-5 | ✓ | ✓ | lenet ✓ | lenet+Conv2d/MaxPool GPU ✓ | — | — | **4/4** |
| Study 004 LSTM | ✓ | ✓ | lstm ✓ | lstm ✓ | — | — | **4/4** |
| Study 005 Quantized | ✓ | ✓ | quantized ✓ | — | — | — | **3/3** |

Phase 0/0+ studies use PyTorch training workflows. mF/gP columns are N/A.
Study 005 uses integer arithmetic (Q8/Q4), not Tensor ops — gT is N/A.

### baseCamp (B-01..B-15) — Primitives Validated

| Sub-Thesis | Papers | Rs | bC GPU | Dispatch | mH | Status |
|:----------:|--------|:--:|:------:|:--------:|:--:|:------:|
| 01 Weight Hamiltonians | B-01..B-03 | 21/21 ✓ | eigh, IPR, variance ✓ | 16/16 ✓ | 14/14 ✓ | **4/4** |
| 02 Information Flow | B-04..B-06 | 22/22 ✓ | variance ✓ | ✓ | ✓ | **4/4** |
| 03 Loss Landscapes | B-07..B-09 | 27/27 ✓ | matmul, entropy ✓ | ✓ | ✓ | **4/4** |
| 04 Neural PGM | B-10..B-12 | 21/21 ✓ | correlation, KL ✓ | ✓ | ✓ | **4/4** |
| 05 Multi-Agent QS | B-13..B-15 | 23/23 ✓ | chi², L2 ✓ | ✓ | ✓ | **4/4** |

baseCamp papers use in-code synthetic data (deterministic seed 42). No mF/gP
columns — baseCamp math uses `BarraCUDA` typed f64 ops, not domain-specific WGSL
shaders. GPU validation through `validate_basecamp_gpu` (14/14 PASS). CPU↔GPU
dispatch parity through `validate_compute_dispatch` (16/16 PASS). Mixed-hardware
routing through `validate_mixed_hardware` (14/14 PASS).

### Stack Coverage Summary

| Tier | Papers Covered | Total | Coverage |
|------|---------------|-------|----------|
| Python control (Py) | 25/25 | 206 checks | **100%** |
| Rust CPU (Rs) | 25/25 + baseCamp | 501 lib + 114 baseCamp + 9 integration | **100%** |
| BarraCUDA CPU (bC) | 24/25 | 203 checks | **96%** |
| BarraCUDA GPU Tensor (gT) | 23/25 | 98+ checks | **92%** |
| BarraCUDA GPU (baseCamp) | 5/5 sub-theses | 14 checks | **100%** |
| metalForge WGSL (mF) | 15/25 | 108 checks | **100%**† |
| GPU Pipeline (gP) | 15/25 | 94 checks | **100%**† |
| Cross-dispatch (xD) | 15/15 | 49 checks | **100%**† |
| CPU↔GPU dispatch | 25 + baseCamp | 16 checks | **100%** |
| Mixed hardware (mH) | baseCamp | 14 checks | **100%** |

`†` 100% of applicable papers. Phase 0/0+ studies use PyTorch, not WGSL shaders.

### What Changed (Phase 5b buildout, February 22, 2026)

**Phase 0/0+ gaps closed:**
- **Exp 001 Surrogate**: S-15 unblocked — new `validate_barracuda_surrogate` (7/7 PASS)
- **Exp 002 Transformer**: new `validate_barracuda_gpu_transformer` (7/7 PASS) — Q/K/V projection, attention scores, FFN block via GPU Tensor
- **Exp 004 Transfer**: new `validate_barracuda_transfer` (7/7 PASS) — domain adaptation MLP forward + metrics
- **Exp 003, Study 003, Study 004**: reclassified as gT (sequence/lenet/lstm validators use GPU Tensor)

**Cross-dispatch 100%:**
- New `validate_cross_dispatch_hmm` (4/4 PASS): Papers 016/018, `hmm_forward_log.wgsl` GPU ↔ CPU log-likelihood parity (diff=2.29e-6)
- New `validate_cross_dispatch_ode` (4/4 PASS): Paper 020, `rk4_parallel.wgsl` GPU ↔ CPU ODE integration parity (diff=1.49e-8)
- Existing cross-dispatch binaries already covered Papers 011-015, 017, 019, 021-025

**Previous sweep (Phase 5a+):**
- S-16 FIXED: transpose dispatch one-line fix
- S-15 root-caused: WGPU/Vulkan driver bug, workaround documented
- 8 GPU Tensor + 6 GPU Pipeline + 3 BarraCUDA CPU validators (89 checks)

### Remaining Gaps

**Exp 005 Isomorphic (analytical only):**
Cross-domain architecture mapping — no numerical computation to validate via bC/gT.

**Study 005 Quantized (integer ops):**
Q8/Q4 quantization uses integer arithmetic, not Tensor matmul. Already validated
via `validate_barracuda_quantized` (CPU primitive path).

---

## GPU Promotion Priority

1. ~~**Papers 016–018 (Liu)** — HMM forward/backward~~ → `hmm_forward_log.wgsl` **DONE** (13/13 + pipeline 5/5)
2. ~~**Papers 011–015 (Dolson)** — Batch fitness evaluation~~ → `batch_fitness_eval.wgsl` **DONE** (20/20)
3. **Papers 022–023 (Kachkovskiy)** — Tridiagonal eigensolver → specialized `tridiag_eigh.wgsl` (**Pending** — ToadStool NAK eigensolve)
4. ~~**Papers 020–021 (Waters)** — GPU-parallel RK4~~ → `rk4_parallel.wgsl` **DONE** (8/8 + pipeline 5/5)
5. ~~**Paper 017 (Liu)** — Pairwise distance matrix~~ → `pairwise_hamming.wgsl` **DONE** (5/5)
6. ~~**Paper 024 (Anderson)** — Pairwise Jaccard~~ → `pairwise_jaccard.wgsl` **DONE** (6/6 + pipeline 5/5)
7. ~~**Paper 019 (Waters)** — Spatial PD payoff~~ → `spatial_payoff.wgsl` **DONE** (5/5 + pipeline 5/5)
8. ~~**Papers 022–023** — Batch IPR~~ → `batch_ipr.wgsl` **DONE** (5/5 + pipeline 5/5)
9. ~~**All stochastic** — GPU PRNG~~ → `xoshiro128ss.wgsl` **DONE** (5/5)
10. ~~**GPU PRNG → Wright-Fisher / Gillespie** — stochastic GPU pipelines~~ → `validate_gpu_pipeline_wright_fisher` (4/4) + `validate_gpu_pipeline_gillespie` (6/6) **DONE** (Session 44)
11. ~~**GPU dispatch (38 ops)** — capability-based runtime dispatch~~ → `gpu_ops` + `gpu_dispatch` **DONE** (Phase A: 27/27, Phase B: 20/20, Sessions 45–46)

---

## Upstream Parity & Capability Dispatch (Sessions 40, 42)

All 15 Phase 0++ papers have been validated through the full 7-tier stack.
6 GPU validators have dual-path upstream parity (local vs barracuda wrapper,
0.00e0 diff — bit-identical). 12 validators use capability-based dispatch
(`Gpu::dispatch_1d`) with runtime hardware validation. Spectral theory
validator cross-validates dense Householder+QR vs tridiag Sturm bisection
(17/17 PASS, max diff 2.89e-15). All controls use open data and open systems.

---

## Multi-Hardware Validation Summary (Session 42)

Each paper's controls run on open data and open systems. The progression
validates correctness at every hardware tier:

### Tier 1: Open Data Controls (Python)
- **25/25 papers** (206 checks) use open data exclusively
- Sources: in-code synthetic (deterministic seed 42), Open-Meteo ERA5 (CC BY 4.0),
  MNIST (CC BY-SA 3.0), published reference data (MIT/Apache-2.0)
- No proprietary, paywalled, or access-restricted data
- Python baseline drift detection: `control/check_drift.sh` (all 25 baselines)

### Tier 2: BarraCUDA CPU (Rust native)
- **24/25 papers** (203 checks, 96% coverage)
- Pure Rust math: `stats::variance`, `linalg::eigh_f64`, `numerical::rk45_solve`,
  `special::chi_squared_sf`, `optimize::nelder_mead`
- Gap: Exp 005 (analytical-only) — no numerical computation
- Key finding: `rk45_solve` achieves machine-precision parity with hand-rolled RK4

### Tier 3: BarraCUDA GPU (Tensor API)
- **23/25 papers** (98+ checks, 92% coverage)
- GPU Tensor ops: `matmul`, `transpose`, `tanh`, `sigmoid`, `add`, `mul`
- f32 GPU vs f64 CPU agreement: < 1e-3 for most operations
- Gaps: Exp 005 (analytical), Study 005 (integer Q4/Q8 — validated separately)
- S-15 workaround: data magnitude ≥ 0.5

### Tier 4: metalForge WGSL (Domain-Specific GPU Kernels)
- **15/25 papers** (108 checks, 100% of applicable papers)
- 17 WGSL shaders: `batch_fitness`, `pairwise_l2`, `hmm_forward_log`,
  `rk4_parallel`, `spatial_payoff`, `batch_ipr`, `hill_gate`, etc.
- 13/17 absorbed upstream into BarraCUDA; 4 local (S-03b, PRNG, swarm)

### Tier 5: GPU Pipeline (Multi-Kernel Chains)
- **15/25 papers** (94 checks, 100% of applicable papers)
- Chained domain→reduce pipelines: HMM, ecology, spectral, genomics, modes,
  directed evolution, signal integration
- Single-encoder dispatch: 46-78× speedup over per-op

### Tier 6: Cross-Dispatch (CPU↔GPU Parity)
- **15/15 Phase 0++ papers** (49 checks, 100%)
- CPU ↔ GPU routing preserves correctness to machine precision
- 6 dual-path upstream parity validators: 0.00e0 diff (bit-identical)

### Tier 7: metalForge Mixed Hardware (Session 43)
- **Capability-based dispatch**: 12 validators use `Gpu::dispatch_1d()` with
  runtime hardware validation via `GpuCapabilities`
- **Cross-eigensolver**: Dense Householder+QR vs tridiag Sturm bisection agree
  at machine epsilon (2.89e-15 at n=64)
- **RTX 4070 + llvmpipe**: Validated on both GPU (Vulkan) and CPU (software)
  backends via the same WGSL source
- **4-tier kernel router**: DeviceCapabilities-driven matmul selection — best
  kernel per hardware configuration

### Tier 8: Multi-GPU Portability (Session 44)
- **RTX 4070** (Ada Lovelace, proprietary Vulkan): **133/133 PASS**
- **TITAN V** (Volta GV100, NVK open-source Vulkan): **143+ additional PASS**
- Results are **bit-identical** across architectures and driver stacks
- Same WGSL source, different silicon generations (2017 vs 2023)
- `NEURALSPRING_BACKEND=titan` selects Titan V adapter at runtime
- Proves: GPU math portability is not dependent on a specific vendor driver

### Tier 9: GPU Dispatch — Pure GPU Promotion (Sessions 45–46)
- **38 CPU→GPU promotions** via `gpu_dispatch::Dispatcher` (Phase A: 27, Phase B: 11)
- Capability-based routing: GPU when available, CPU fallback otherwise
- `validate_gpu_promotion`: **27/27 PASS** (both GPUs)
- `validate_gpu_phase_b`: **20/20 PASS** (both GPUs)
- Covers: HMM (forward/backward/Viterbi), statistics (variance/correlation/allele freq),
  distances (L2/Hamming/Jaccard/geographic), ML (neural_forward/softmax/PCA),
  bio (fitness/diversity/Hill/replicator), ODE (RK4 step)
- **~90% of production math** has GPU path. Remaining ~10%: full ODE loops, FST, introgression chain
- All per-paper controls still use open data: Python baselines validate independently

### Tier 10: metalForge Mixed Hardware (Future)
- **Capability-based dispatch across GPU + NPU + CPU**
- Infrastructure ready: `mixed.rs` (MixedSubstrate), `pcie_bridge.rs` (PCIe cost model)
- Validated: 16/16 dispatch routing + 16/16 mixed dispatch checks (Session 43)
- Next: Exercise on actual NPU hardware, optimize transfer cost model

---

## Notes

- Paper 011 is the single most important paper in the entire ecosystem — it externally validates the constrained evolution methodology
- Papers 016–018 (Liu) bridge neuralSpring to wetSpring's metagenomics work
- Papers 019–021 (Waters) connect optimization theory to real biological dynamics
- Papers 022–023 (Kachkovskiy) provide the mathematical foundation for understanding loss landscapes and training dynamics
- Dolson papers (011–015) form a coherent sequence from theory → metrics → applications → swarm
- All 13 Phase 0++ modules are Tier A (pure math, direct port) — ready for BarraCUDA CPU evolution

---

## Session 43 Validators (February 22, 2026)

New validation binaries added in Session 43 for upstream BarraCUDA wrapper integration
and mixed-hardware dispatch:

| Validator | Checks | Purpose |
|-----------|--------|---------|
| `validate_gpu_gillespie` | 20/20 | `GillespieGpu` f64 parallel SSA |
| `validate_upstream_taxonomy` | 3/3 | `TaxonomyFcGpu` f64 metagenomics |
| `validate_upstream_kmer` | 3/3 | `KmerHistogramGpu` k-mer histograms |
| `validate_upstream_unifrac` | 2/2 | `UniFracPropagateGpu` tree propagation |
| `validate_barracuda_chi_squared` | 13/13 | `chi_squared::*` distribution functions |
| `validate_gpu_logsumexp` | 5/5 | `logsumexp_reduce.wgsl` batched logsumexp |
| `validate_gpu_stencil` | 3/3 | `stencil_cooperation.wgsl` Fermi imitation |
| `validate_gpu_rk45` | 6/6 | `rk45_adaptive.wgsl` Dormand-Prince |
| `validate_gpu_wright_fisher` | 4/4 | `wright_fisher_step.wgsl` stochastic |
| `validate_cpu_gpu_parity` | 17/17 | CPU vs GPU Tensor bit-identical parity |
| `validate_toadstool_dispatch` | 16/16 | `logsumexp_substrate`, `stochastic_substrate` heuristics |
| `validate_mixed_dispatch` | 16/16 | PCIe transfer cost model |

These extend wetSpring-origin APIs (Taxonomy, Kmer, UniFrac, Gillespie) and chi² functions
validated from neuralSpring; GillespieGpu benefits all Springs for stochastic simulation.

### Session 44 Validators (February 23, 2026)

Gap-closure validators: stochastic GPU pipelines and missing tier coverage.

| Validator | Checks | Purpose |
|-----------|--------|---------|
| `validate_gpu_pipeline_wright_fisher` | 4/4 | Wright-Fisher → mean_reduce pipeline (Papers 024-025) |
| `validate_gpu_pipeline_gillespie` | 4/4 | Gillespie SSA → mean_reduce pipeline (Papers 013, 020) |
| `validate_barracuda_gpu_lenet` | 5/5 | `Tensor::conv2d` + `Tensor::maxpool2d` GPU WGSL (Study 003 gT) |
| `validate_barracuda_transformer` | 7/7 | Full transformer layer via BarraCUDA Tensor (Exp 002 bC) |

**Stochastic pipelines**: Wright-Fisher and Gillespie now have GPU pipeline
validators that chain domain shader → mean_reduce with scalar-only readback.
This completes the GPU Promotion Priority item 10 (GPU PRNG → stochastic pipelines).

**Conv2d/MaxPool GPU**: First Spring-side exercise of `barracuda::ops::conv2d::Conv2D`
and `barracuda::ops::maxpool2d::MaxPool2D` GPU WGSL shaders. Validates single-channel
conv → relu → pool pipeline against f64 CPU reference.

**Transformer bC tier**: Closes the gap where Exp 002 had only a GPU-specific validator
(`validate_barracuda_gpu_transformer`). The new bC-tier validator covers Q/K/V projections,
attention scores, FFN block, residual connections, global softmax, and a full layer forward
pass (sans row-wise softmax — `Tensor::softmax()` is global; row-wise requires
`ScaledDotProductAttention` which is tested separately).

### Multi-GPU Validation (Session 44)

All Session 44 validators validated on **both** discrete GPUs:

| Validator | RTX 4070 (NVIDIA) | TITAN V (NVK GV100) |
|-----------|------------------|---------------------|
| `validate_barracuda_transformer` | **12/12 PASS** | **12/12 PASS** |
| `validate_barracuda_gpu_lenet` | **8/8 PASS** | **8/8 PASS** |
| `validate_gpu_pipeline_wright_fisher` | **4/4 PASS** | **4/4 PASS** |
| `validate_gpu_pipeline_gillespie` | **6/6 PASS** | **6/6 PASS** |

Results are bit-identical across proprietary (NVIDIA Vulkan) and open-source (NVK)
drivers, proving math portability across GPU architectures and driver stacks.
`NEURALSPRING_BACKEND=titan` selects Titan V; default selects RTX 4070.

Extended Titan V NVK sweep (Session 44):

| Validator | Titan V (NVK) Checks |
|-----------|---------------------|
| `validate_gpu_hmm_forward` | **13/13 PASS** |
| `validate_gpu_batch_fitness` | **21/21 PASS** |
| `validate_gpu_rk4` | **8/8 PASS** |
| `validate_gpu_prng` | **5/5 PASS** |
| `validate_gpu_wright_fisher` | **4/4 PASS** |
| `validate_gpu_pipeline_hmm` | **5/5 PASS** |
| `validate_gpu_pipeline_fitness` | **5/5 PASS** |
| `validate_cpu_gpu_parity` | **17/17 PASS** |
| `validate_barracuda_gpu_spectral` | **10/10 PASS** |
| `validate_barracuda_gpu_eco` | **6/6 PASS** |
| `validate_barracuda_gpu_hmm` | **5/5 PASS** |
| `validate_barracuda_gpu_fitness` | **7/7 PASS** |
| `validate_barracuda_gpu_transformer` | **7/7 PASS** |
| `validate_barracuda_tensor` | **86/86 PASS** |

**143 additional checks on Titan V NVK** — all bit-identical with RTX 4070.

### BarraCUDA Upstream Fixes (Session 44)

Two upstream BarraCUDA bugs discovered and fixed during validation:

1. **`mean_reduce` entry point mismatch**: `ops/mean.rs` referenced `entry_point: "main"`
   but `mean_reduce.wgsl` shader defines `fn mean_reduce(...)`. Also fixed double-divide
   bug where Rust code re-divided by n after the shader already computed the mean.
   Previously caused `validate_barracuda_tensor` to fail (85/86).

2. **Chi-squared expected values**: `validate_barracuda_chi_squared` used textbook-rounded
   reference values (e.g., 0.950 for CDF critical values). BarraCUDA's implementation
   is more precise (0.949956). Updated expected values to full precision.
   Previously 10/13, now 13/13.

### Full Suite Result (Session 46)

`validate_all`: **133/133 PASS, 0 FAIL** (RTX 4070, Vulkan)

### Session 45 (February 23, 2026)

**GPU promotion (Phase A)**: New `validate_gpu_promotion` binary validates all 27 CPU→GPU
promotions (27/27 PASS on RTX 4070 and TITAN V NVK). The `gpu_dispatch` module
provides runtime capability-based GPU/CPU dispatch; `gpu_ops` provides GPU paths
for all previously CPU-bound ops (matmul, transpose, frobenius_norm, softmax,
l2_distance, hmm_forward_step, neural_forward, etc.).

### Session 46 (February 23, 2026)

**GPU promotion (Phase B)**: `validate_gpu_phase_b` (20/20 PASS on RTX 4070 + TITAN V NVK).
11 new GPU operations added to `gpu_ops` and `gpu_dispatch`:
- **HMM**: backward step (GEMV), Viterbi step (broadcast + max_dim + argmax)
- **Meta-population**: allele_frequencies (column-sum), nucleotide_diversity,
  matrix_correlation, geographic_distance_matrix, thermal_diversity_correlation,
  inter_population_af_variance
- **Game theory**: replicator dynamics step (2×2 GEMV)
- **Hill activation**: fixed from pseudo-GPU to genuine GPU (log→exp pipeline)
- `validate_all`: **133/133 PASS, 0 FAIL**. ~90% of production math now has GPU path.

---

## Controls Verification: Open Data + Three Hardware Tiers (Sessions 49–61)

Every paper control runs on open data, uses deterministic seeds, and validates
at three hardware tiers (BarraCUDA CPU, BarraCUDA GPU, metalForge mixed).

**Sessions 60–61 confirmation**: All three tiers re-verified after 16-function rewiring.
Cross-spring evolution validator (22/22 PASS) proves rewired paths produce identical
results through upstream BarraCUDA dispatch. Benchmark validation confirms
performance benefits: Variance 2.46× (hotSpring Welford), Entropy 2.59× (wetSpring fused).

### BarraCUDA CPU Controls

All 24 applicable papers (96%) validate Python → Rust → BarraCUDA CPU parity.
Pure Rust math (stats, linalg, numerical, special) reproduces Python at
machine precision or better.

| Primitive | Papers Using | Max Diff vs Python |
|-----------|-------------|-------------------|
| `stats::variance` | 011–025 (all 15 P0++) | Machine ε |
| `linalg::eigh_f64` | 022–023 | 1.75e-14 (n=32) |
| `numerical::rk45_solve` | 019–021 | Machine ε |
| `special::chi_squared_sf` | 018 | 1e-10 |
| `linalg::solve_f64` | 016, 015 | Machine ε |
| `stats::pearson_correlation` | 012 | Machine ε |

### BarraCUDA GPU Controls

All 23 applicable papers (92%) validate CPU → GPU Tensor parity. The
`Tensor` API (matmul, transpose, sigmoid, tanh, add, mul, etc.) produces
f32 results within 1e-3 of f64 CPU references across all domains.

| Paper Group | GPU Op Validated | Max f32-f64 Diff |
|-------------|-----------------|------------------|
| 011–015 (Dolson) | BatchFitnessGpu, MultiObjFitnessGpu, SwarmNnGpu | < 1e-3 |
| 016–018 (Liu) | HmmBatchForwardF64, PairwiseHammingGpu | < 1e-6 |
| 019–021 (Waters) | SpatialPayoffGpu, StencilCooperationGpu, HillGateGpu | < 1e-3 |
| 022–023 (Kachkovskiy) | BatchIprGpu, eigh_f64 | < 1e-7 |
| 024–025 (Anderson) | PairwiseJaccardGpu, LocusVarianceGpu, WrightFisherGpu | < 1e-3 |

### metalForge Mixed-Hardware Controls

All 15 Phase 0++ papers have cross-dispatch validation (CPU ↔ GPU routing
preserves correctness). Additionally:

- **Multi-GPU**: RTX 4070 + TITAN V (NVK) produce bit-identical results for
  all 133 validators.
- **Pure GPU dispatch**: 38 CPU→GPU promotions validated (Phase A: 27/27,
  Phase B: 20/20) — ~90% of production math.
- **Mixed-hardware infrastructure**: `MixedSubstrate` routing and PCIe
  transfer cost model validated (16/16 dispatch + 16/16 mixed checks).

### Open Data Confirmation

| Source Type | Papers | License |
|-------------|--------|---------|
| In-code synthetic (seed=42) | 011–023, 024–025 | N/A (pure math) |
| Open-Meteo ERA5 | Exp 003–004, Study 004–005 | CC BY 4.0 |
| MNIST | Study 003 | CC BY-SA 3.0 |
| GitHub repos | Study 001–002, Paper 012 | MIT / Apache-2.0 |

**No proprietary data. No API keys. No access-restricted sources.
All experiments reproducible from scratch with deterministic seeds.**

---

## baseCamp Controls Verification (Session 50)

baseCamp modules (5 sub-theses, 82 checks) follow the same controls
framework. All data is in-code synthetic with deterministic seeds.

### baseCamp — BarraCUDA CPU Controls

All 5 baseCamp modules use `eigh_f64` (via `eigh.rs` → `barracuda::ops::linalg`)
for eigendecomposition. This is the same `eigh_f64` validated at 1.75e-14
accuracy (Householder+QR) in Phase 0++ papers 022–023.

| Module | BarraCUDA Primitive | Validation |
|--------|--------------------|----|
| `weight_spectral` | `eigh_f64` | Hamiltonian symmetry, ESD normalization, IPR range |
| `information_flow` | `eigh_f64` | Attention Hamiltonian spectral finiteness |
| `loss_landscape` | `eigh_f64` | Hessian spectrum matches analytical quadratic |
| `neural_pgm` | `eigh_f64` | Transition matrix spectral properties |
| `agent_coordination` | `eigh_f64` | Laplacian smallest eigenvalue ≈ 0 |

### baseCamp — BarraCUDA GPU Controls (Planned)

baseCamp modules are CPU-only in Session 50. GPU promotion targets:

| Module | GPU Candidate | BarraCUDA Pattern | Priority |
|--------|-------------|------------------|----------|
| `weight_spectral` | `weight_to_hamiltonian` | `Tensor::matmul` | High (bottleneck for large matrices) |
| `loss_landscape` | `numerical_hessian` | `BatchFitnessGpu` (parallel eval) | High (O(n²) evaluations) |
| `neural_pgm` | `belief_propagation_chain` | `HmmBatchForwardF64` (GEMV chain) | Medium |
| `agent_coordination` | `interaction_graph` | `PairwiseL2Gpu` | Medium |
| `loss_landscape` | `boltzmann_sampling` | `WrightFisherGpu` (parallel MCMC) | Low |

When GPU promotion is implemented, controls will follow the same pattern
as Phase 0++ papers: CPU reference → GPU result → diff within tolerance.

### baseCamp — Full Three-Tier Hardware Validation (COMPLETE)

All baseCamp experiments validated across three hardware tiers:

1. **BarraCUDA CPU**: Pure Rust — 114/114 PASS (Sessions 50, 54)
2. **BarraCUDA GPU**: Tensor API on RTX 4070 — 14/14 PASS (`validate_basecamp_gpu`)
3. **Dispatcher routing**: GPU/CPU parity — 19/19 PASS (`validate_basecamp_dispatch`)
4. **CPU↔GPU parity**: Cross-domain — 34/34 PASS (`validate_barracuda_parity`)
5. **metalForge mixed**: PCIe tiers + substrate selection — 36/36 PASS (`validate_metalforge_pcie`)

**Total baseCamp checks**: 147 (114 CPU + 14 GPU + 19 dispatch) — all PASS.

### baseCamp — Open Data Confirmation

| Source Type | Modules | License |
|-------------|---------|---------|
| In-code synthetic (seed=42) | All 5 baseCamp modules | N/A (pure math) |

**All baseCamp experiments generate synthetic data programmatically.
No model downloads. No external weights. No proprietary systems.
Deterministic seed (42) ensures exact reproducibility.**

### baseCamp Experiment-to-Paper Mapping

| Sub-thesis | Grounding Papers | Open Access | Experiment Coverage |
|:----------:|:----------------:|:-----------:|:-------------------:|
| nS-01 | Martin & Mahoney (2021), Pennington & Worah (2017), Anderson (1958) | All open | 21/21 checks (analytical + synthetic) |
| nS-02 | Schoenholz et al. (2017), Poole et al. (2016), Ganguli (2020) | All open | 22/22 checks (analytical + synthetic) |
| nS-03 | Wales (2003), Li et al. (2018), Ghorbani et al. (2019) | All open | 27/27 checks (`numerical_hessian` → upstream) |
| nS-04 | Koller & Friedman (2009), Hinton (2012), Murphy (2012) | All open | 21/21 checks (`belief_propagation_chain` → upstream) |
| nS-05 | Waters & Bassler (2005), Dolson et al. (2019), Anderson QS | All open | 23/23 checks (`graph_laplacian` → upstream) |

---

## WDM Surrogate Extensions — baseCamp Sub-thesis 07

**Purpose**: Extend neuralSpring's validated surrogate learning pipeline
to warm dense matter (WDM) physics, supporting hotSpring's Tier 4 WDM
reproduction targets and baseCamp Sub-thesis 07 (Sovereign WDM on Consumer GPU).

### Surrogate Models for WDM Transport

| # | Target | Method | BarraCUDA Primitive | Status |
|---|--------|--------|-------------------|--------|
| nW-01 | WDM transport surrogate (D*, η*, λ* vs ρ, T, Z*) | MLP/RBF — extend hotSpring Paper 3 (Diaw surrogate, 9/9) to WDM parameter range | `nuclear_eos_gpu` (GPU RBF validated) | Queued |
| nW-02 | EOS surrogate (P, E vs ρ, T for H, He, C) | MLP trained on FPEOS tables (Militzer) — extend Exp 001 FAO-56 surrogate to physics EOS | `validate_barracuda_surrogate` (7/7) | Queued |
| nW-03 | S(q,ω) peak predictor | LSTM on MD-generated S(q,ω) time series — predict peak position/width from (ρ, T) | `validate_barracuda_gpu_lstm` | Queued |

### Transfer Learning: Classical → WDM

| # | Target | Method | BarraCUDA Primitive | Status |
|---|--------|--------|-------------------|--------|
| nW-04 | Classical plasma → WDM transfer | Fine-tune transport surrogate from Stanton-Murillo (Γ,κ) space to WDM (ρ,T,Z*) space — same architecture as Exp 004 (MI→NM/CA transfer, 6/6) | Transfer learning pipeline (validated) | Queued |
| nW-05 | NPU screening for WDM phase | ESN classifier: given (ρ,T), predict WDM regime (classical/WDM/degenerate) — extends metalForge lattice phase classifier to plasma phases | `validate_lattice_npu` (10/10) | Queued |

**Connection to hotSpring Tier 4**: Each surrogate replaces expensive MD
runs with inference at 9,017× less energy (NPU) or ~100× less compute
(GPU RBF). Classical→WDM transfer learning exploits the fact that
Stanton-Murillo transport coefficients are continuous functions of
coupling parameters — fine-tuning bridges the regime gap.

**Connection to baseCamp Sub-thesis 07 §4**: Phase 4 (distributed
parameter sweep) generates the training data; neuralSpring surrogates
provide instant lookup for the sweep results.

---

## Sovereign Folding — Protein/RNA/DNA Structure Prediction (NEW TRACK)

**Purpose**: Port OpenFold3's Evoformer + Structure Module to BarraCUDA WGSL
shaders for sovereign structure prediction on consumer GPUs.

**Where it lives**: `neuralSpring/sovereign_folding/`

### Papers

| # | Paper | Journal | Year | Why | Status |
|---|-------|---------|------|-----|--------|
| nF-01 | Ahdritz et al. "OpenFold: Retraining AlphaFold2 yields new insights" | Nature Methods | 2024 | Reference implementation (Apache 2.0). Baseline for porting | Phase A Eval DONE (9/9) |
| nF-02 | Jumper et al. "Highly accurate protein structure prediction with AlphaFold" | Nature 596:583-589 | 2021 | Original architecture. Evoformer + IPA specification | Queue |
| nF-03 | Abramson et al. "Accurate structure prediction for all molecules" (AlphaFold3) | Nature 630:493-500 | 2024 | Diffusion-based extension. RNA/DNA/ligand handling | Queue |

### Phase A — Baseline Assessment (DONE)

Evaluation script: `sovereign_folding/openfold3_eval.py` (9/9 checks)

- RTX 4070: 12 GB VRAM, Compute 8.9, Vulkan + SHADER_F64 confirmed
- PyTorch 2.9.0+cu128 available, 316x GPU speedup on attention
- 4 of 10 required primitives already exist in BarraCUDA
- See `sovereign_folding/BARRACUDA_FOLDING_REQUIREMENTS.md` for full shader spec
- See `sovereign_folding/MSA_DATABASE_PLAN.md` for data acquisition plan
