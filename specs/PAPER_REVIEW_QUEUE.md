# neuralSpring — Paper Review Queue

**Last Updated**: March 14, 2026 (Sessions 45–150 — 27/27 papers complete. Paper queue CLOSED. Extension phase: playGround (Squirrel MCP, HuggingFace Model Lab, compute triangle). S150: ToadStool/coralReef IPC integration, hot/cold dispatch benchmarks)
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

### Bioprocess ML Prediction (Liao / ADREC)

Wei Liao (ADREC Director, MSU BAE) — the author interviewed with ADREC before
the Sandia internship. Liao's group applied ML to predict anaerobic digestion
performance from operational parameters. Profile: `whitePaper/attsi/non-anon/contact/liao/README.md`

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 27 | Wang et al. "Prediction of anaerobic digestion performance and identification of critical operational parameters using machine learning algorithms" | Bioresour Technol 298:122495 | 2020 | Liao | **ML for biogas yield prediction from operational parameters** — same ESN architecture as nW-05 (WDM classifier), different domain (bioprocess engineering). Validates that neuralSpring surrogates generalize to environmental/bioprocess systems. Tests ESN on digester operational parameters (T, pH, OLR, HRT, VS/TS) | **Complete + bC/gT** — `control/digestion_prediction/digestion_prediction.py` (9/9 PASS), `digestion_prediction.rs` (11 tests), `validate_digestion_prediction` (36/36 PASS), `validate_barracuda_digestion` (23/23 PASS: bC CPU 6, gT GPU 17, GPU↔CPU parity ≤7.1e-5) |

**Why this matters for neuralSpring**: We have LSTM weather (Exp 3/9, R²=0.93-0.849),
LSTM glucose (Paper 026, 26/26), ESN WDM regime (nW-05, 96.5%), and ESN HAB sentinel
(wetSpring Exp114-119). Wang 2020 adds a fourth domain — bioprocess engineering —
proving the isomorphic learning thesis (Exp 5) extends to engineered biological
systems. The paper uses random forests and gradient boosting; reproducing it with
ESN/LSTM demonstrates that reservoir computing matches or exceeds their approach
while running on GPU/NPU at sovereign speeds.

**Connection to baseCamp Paper 16**: Anaerobic digester process monitoring via ESN
regime classifier = Paper 04 (sentinels) applied to ADREC digesters. The QS
dynamics that drive community stability in digesters are what Paper 16 models.

### Biomedical Time-Series Prediction (Chuna)

Thomas Chuna (Physics & CMSE, MSU) — referred by Murillo (March 4, 2026).
Before joining the Murillo Group for plasma physics, Chuna worked on LSTM-based
blood glucose prediction. Profile: `whitePaper/attsi/non-anon/contact/murillo/chuna_profile.md`

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 26 | Chuna "Setting Limits on Neural Network's Predictive Capacity in T1D Blood Glucose Concentration" | arXiv:2005.09051 | 2020 | Chuna | LSTM time series on CGM data — same architecture as Exp 3/9 (weather LSTM), different domain. Explores prediction horizon limits. Validates LSTM primitives on biomedical data | **Complete + bC/gT** — `control/glucose_prediction/glucose_prediction.py` (9/9 PASS), `glucose_prediction.rs` (11 tests), `validate_glucose_prediction` (26/26 PASS), `validate_barracuda_glucose_prediction` (25/25 PASS: bC CPU 11, gT GPU 14, GPU↔CPU parity ≤1.07e-6) |

**Why this matters for neuralSpring**: We already have LSTM weather validation
(Exp 3: synthetic, 5/5; Exp 9: real ERA5, 5/5). Chuna's paper uses the same
LSTM architecture on a fundamentally different domain (biomedical time series
vs meteorological). Reproducing it validates that our LSTM primitives generalize
across domains — the isomorphic learning thesis (Exp 5). The prediction horizon
analysis (how far ahead can LSTM reliably predict?) maps directly to the
forecasting limits we quantified in Exp 9 (48h optimal for ERA5). Public CGM
datasets exist (OhioT1DM, OpenAPS). Additionally, Chuna is now in the Murillo
Group working on plasma physics — his cross-domain trajectory (biomedical ML →
computational physics) directly parallels the ecoPrimals argument that the same
primitives work across fields.

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

### Sub-Thesis 06: Anderson Localization in Immunological Signaling (NEW — Session 105)

Extension of Anderson localization from microbial QS (Papers 01, 05, 06) to
immunological cytokine signaling. Cytokines (IL-4, IL-13, IL-31) as diffusible
signals propagating through disordered tissue. Connects to Fajgenbaum drug
repurposing paradigm (MATRIX, ARPA-H $48.3M) via geometry-aware scoring.

| # | Citation | Year | Key Data | neuralSpring Target |
|---|----------|------|----------|---------------------|
| B-16 | Gonzales AJ et al. "IL-31: its role in canine pruritus and naturally occurring canine AD" *Vet Dermatol* 24:48-53 | 2013 | IL-31 elevated in AD dog serum; induces pruritus | Anderson lattice: IL-31 as diffusible signal, W from tissue heterogeneity |
| B-17 | Gonzales AJ et al. "Oclacitinib (APOQUEL) is a novel JAK inhibitor" *J Vet Pharmacol Ther* 37:317-324 | 2014 | JAK1 IC50 = 10 nM; blocks IL-2/4/6/13/31 | Dose-response modeling: IC50 as Anderson barrier height |
| B-18 | Gonzales AJ et al. "IL-31-induced pruritus in dogs: a novel experimental model" *Vet Dermatol* 27:34-e10 | 2016 | Standardized pruritus model; oclacitinib vs steroids at 1/6/11/16 hr | LSTM time series: pruritus prediction, controlled Anderson perturbation |
| B-19 | Fleck TJ,...,Gonzales AJ "Onset and duration of lokivetmab in IL-31 pruritus" *Vet Dermatol* 32:681-e182 | 2021 | Cytopoint: 3 hr onset, dose-dependent duration (14/28/42 days) | ESN classifier: pharmacokinetic decay as signal extinction |
| B-20 | McCandless EE,...,Fici GJ "Allergen-induced IL-31 by canine Th2 cells" *Vet Immunol Immunopathol* 157:42-48 | 2014 | IL-31 targets: immune + skin + neural cells | Three-compartment Anderson lattice: multi-cell-type disorder |
| B-21 | Fajgenbaum DC et al. "Pathogenic PI3K/AKT/mTOR in iMCD" *J Clin Invest* | 2019 | Pathway-based drug repurposing; mTOR cross-talks JAK/STAT | Anderson-augmented MATRIX score: pathway × tissue geometry |

**Novel contribution**: No prior work applies Anderson localization to cytokine
propagation in tissue. No prior work adds spatial geometry to drug repurposing scoring.
**Dimensional promotion–collapse duality**: AD scratching (2D→3D) is the inverse of
Paper 06 tillage collapse (3D→2D) — same physics, opposite biological outcome.

**neuralSpring connections**:
- ESN regime classifier (nW-05, 96.5%): classify AD skin state from cytokine profile
- LSTM time series (nW-03, R²=0.98): predict pruritus r(t) from treatment + time
- `MultiHeadWdmClassifier` (Session 105): multi-head ESN for regime + uncertainty
- `TrainingMonitor` (Session 105): DriftMonitor for pharmacokinetic trajectory tracking
- `Dispatcher::kl_divergence` (Session 105): distribution shift in cytokine profiles

**Status**: Proposal — literature grounded, Gonzales catalog complete, Anderson mapping
drafted. Awaiting wetSpring Exp 270-274 for computational validation.

### baseCamp Summary

| Sub-Thesis | Grounding Papers | Experiments | Key Primitive | Rust Module | Checks | Priority |
|:----------:|:----------------:|:-----------:|---------------|-------------|:------:|:--------:|
| 01 Weight Hamiltonians | 3 (B-01 to B-03) | 6 | `eigh_f64`, `BatchIprGpu` | `weight_spectral.rs` | **21/21** | 1 |
| 02 Information Flow | 3 (B-04 to B-06) | 6 | `hmm.rs`, `stencil_cooperation.wgsl` | `information_flow.rs` | **22/22** | 3 |
| 03 Loss Landscapes | 3 (B-07 to B-09) | 5 | `rk45_adaptive.wgsl`, `eigh_f64` | `loss_landscape.rs` | **27/27** | 5 |
| 04 Neural PGM | 3 (B-10 to B-12) | 6 | `hmm.rs`, `introgression.rs` | `neural_pgm.rs` | **21/21** | 2 |
| 05 Multi-Agent QS | 3 (B-13 to B-15) | 5 | `anderson_localization.rs`, `game_theory.rs` | `agent_coordination.rs` | **23/23** | 4 |
| 06 Immunological Anderson | 6 (B-16 to B-21) | 0 (proposed) | `wdm_esn.rs`, `anderson_localization.rs` | `wdm_esn.rs`, `training_monitor.rs` | **0/0** | 6 |
| GPU Parity | — | — | `BarraCUDA` f64 typed ops | `validate_basecamp_gpu` | **14/14** | — |
| **Total** | **21** | **28** | | **6 validators** | **128/128** | |

Core Rust primitives: **ALL IMPLEMENTED AND EXPANDED** (Sessions 50, 54, 105).
Experiment coverage expanded from 82→128 checks including pure GPU parity.
Grounding paper reproductions: **Primitives and experiments validated** — full
paper reproductions with publication-ready analysis remain for Phase 2.
Sub-thesis 06 (immunological Anderson) adds 6 new grounding papers (B-16..B-21)
pending wetSpring computational validation.

All baseCamp papers use open data only (our own trained models + algorithmic
computation). No proprietary models, no external downloads, no API dependencies.

---

## Completion Summary

**All 27 papers complete. Paper 027 (Liao/Wang 2020 ML digestion prediction) complete — validates isomorphic ESN generalization to bioprocess engineering (R²=0.84, 36/36 Rust + 23/23 bC/gT PASS). baseCamp (B-01..B-15) primitives validated. baseCamp Sub-thesis 06 (B-16..B-21, immunological Anderson) added — proposal stage, awaiting wetSpring Exp 270-274. All 5 WDM surrogates (nW-01..nW-05) complete. nF-03 AlphaFold3 Phase C (confidence heads) complete. Paper 026 (Chuna LSTM glucose prediction) complete — validates LSTM prediction horizon limits, isomorphic cross-domain generalization (biomedical ↔ meteorological).**

Session 139: Visualization evolution + deep debt. 16 petalTongue scenario tracks, ecosystem dashboard, config centralization, streaming parsers, BLAST pipeline, Kokkos parity. 1048 lib + 71 forge + 9 integration tests. 233 binaries. 220/220 validate_all. S139 handoff. All paper controls confirmed: open data only (SRA, Zenodo, EPA, PDB, synthetic). BarraCUDA CPU + GPU validated for all applicable papers. metalForge mixed hardware + NUCLEUS Tower/Node/Nest deployment via biomeOS atomic graphs: 43+38+22 PASS.
Session 133: Phase 5–7 buildout. metalForge PCIe P2P, biomeOS pipeline DAG (3 canonical pipelines), petalTongue StreamSession. Feature-gated validate_all. 957 lib + 71 forge + 9 integration tests. 232 binaries. 220/220 validate_all. V91 handoff.
Session 111: CPU benchmark expanded 11→14 domains (Papers 013, 023, 025). 3 new Python bench scripts. 31/31 PASS, 38.6× geomean. Full 10-tier pyramid validated. 210/210 validate_all.
Session 109: Paper queue spec update. 861 lib tests, 229 binaries, 210/210 validate_all.
Session 105: Deep Evolution + baseCamp Paper 12. MultiHeadWdmClassifier (barracuda MultiHeadEsn), TrainingMonitor (brain-inspired FSM), Dispatcher::kl_divergence, dispatch_and_read→Result. NUCLEUS protocol alignment. 5 large-file refactors (validation, provenance, weight_spectral, meta_population, gpu_ops/bio). baseCamp Sub-thesis 06 (B-16..B-21) added: Anderson localization in immunological signaling — Gonzales catalog, Fajgenbaum bridge, dimensional promotion–collapse duality.
Session 93: Deep debt evolution + nF-03 Phase C. dispatch_ops.rs (842→7 domain files), gpu_ops/mod.rs (668→38+tests_ops). Iterator evolution (diffusion.rs, pairformer.rs, counterdiabatic.rs, cpu_fallback.rs, meta_population.rs). Self-identification→env!("CARGO_PKG_NAME"). .unwrap()→.expect(). nF-03 Phase C confidence heads (pLDDT, PAE, pDE, ranking: Py 19/19, Rs 16/16, 7 unit tests). 201 binaries, 685 lib tests, 189/189 validators. 39 Python drift baselines.
Session 92: nF-03 AlphaFold3 Phase A+B buildout — diffusion primitives (Py 29/29, Rs 26/26), Pairformer block (Py 14/14, Rs 13/13). 2 new Python controls, 2 new Rust validators, 11 new unit tests. 196 binaries, 680 lib tests, 184/184 validators in validate_all. 38 Python drift baselines.
Session 88: Publication experiment buildout — Exp-050 (training trajectory spectral analysis, Py 11/11, Rs 12/12), Exp-052 (Hessian eigenanalysis at trained minima, Py 8/8, Rs 14/14), Exp-053 (Anderson multi-agent coordination, Py 11/11, Rs 18/18). 3 new Python controls, 3 new Rust validators, 175 binaries, 668 lib tests, 163 validators in validate_all.
Session 87: WDM surrogate queue closed — nW-03 (LSTM S(q,ω) peak predictor, Py 5/5, Rs 27/27) and nW-05 (ESN regime classifier, Py 5/5, Rs 39/39). 175 binaries, 623 lib tests, 158/158 validators.
Session 86: V50 handoff — WDM buildout complete, 170 binaries, 611 lib tests, 154/154 validators.
Session 83: WDM surrogate buildout — nW-01 transport (Py 4/4, Rs 30/30), nW-02 EOS wired (Py 9/9, Rs 36/36, GPU 15/15), nW-04 transfer (Py 4/4, Rs 6/6). `wdm_transport.rs` new module. 4 new validators in `validate_all` (154 total). 611 lib + 43 forge tests. `check_drift.sh` expanded to 29 baselines.
Session 81: Deep debt evolution — 129+ named tolerances (25 new), spectral_entropy→barracuda (39th rewire), cross-platform probe, PyTorch seeding. Zero inline magic numbers.
Session 80: Comprehensive debt audit — 604 lib tests, 93.5% coverage, wdm_surrogate 97.6%, basecamp 90.6%.
Session 70: Deep audit II — 93.5% coverage (580 tests), tolerance macro refactor, streaming I/O, 100% SPDX (211/211 files), all files ≤1000 lines.
Session 68-69: Deep debt audit — zero ad-hoc tolerances, zero bare `unwrap()`, 107+ named tolerances. 6 validator shader sources → upstream constants.
Session 67: CPU↔Python parity — `validate_cpu_math_parity` 39/39 PASS (1e-10 cross-language).
Session 66: Phase C GPU promotion — HMM chains, FST, introgression, AF variance.
`validate_all`: 207/207 PASS on RTX 4070.
`validate_gpu_phase_c`: 18/18 PASS. `validate_cpu_math_parity`: 39/39 PASS.
Python baselines: 25/25+5 WDM PASS (zero drift). Rust **38.6× faster** than Python/NumPy (14 domains, all Phase 0++ papers).
869 lib + 9 integration + 43 forge tests. 229 validation/bench binaries. Zero debt.
53 CPU→GPU dispatch ops (100% of Dispatcher GPU paths covered).
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

## Full Validation Stack Matrix (March 2, 2026 — Sessions 60–109)

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
| **026 Glucose (Chuna)** | ✓ | ✓ | glucose ✓ | LSTM+readout ✓ | — | — | **4/4** |
| **027 Digestion (Liao)** | ✓ | ✓ | digestion ✓ | ESN+readout ✓ | — | — | **4/4** |

Phase 0/0+ studies use PyTorch training workflows. mF/gP columns are N/A.
Study 005 uses integer arithmetic (Q8/Q4), not Tensor ops — gT is N/A.
Paper 026 uses LSTM reservoir (same primitives as Exp 003/004, nW-03).

### baseCamp (B-01..B-21) — Primitives Validated

| Sub-Thesis | Papers | Rs | bC GPU | Dispatch | mH | Status |
|:----------:|--------|:--:|:------:|:--------:|:--:|:------:|
| 01 Weight Hamiltonians | B-01..B-03 | 21/21 ✓ | eigh, IPR, variance ✓ | 16/16 ✓ | 14/14 ✓ | **4/4** |
| 02 Information Flow | B-04..B-06 | 22/22 ✓ | variance ✓ | ✓ | ✓ | **4/4** |
| 03 Loss Landscapes | B-07..B-09 | 27/27 ✓ | matmul, entropy ✓ | ✓ | ✓ | **4/4** |
| 04 Neural PGM | B-10..B-12 | 21/21 ✓ | correlation, KL ✓ | ✓ | ✓ | **4/4** |
| 05 Multi-Agent QS | B-13..B-15 | 23/23 ✓ | chi², L2 ✓ | ✓ | ✓ | **4/4** |
| 06 Immunological Anderson | B-16..B-21 | 53/53 ✓ | KL, Shannon ✓ | 3/3 ✓ | 7/7 ✓ | **4/4** |

baseCamp papers use in-code synthetic data (deterministic seed 42). No mF/gP
columns — baseCamp math uses `BarraCUDA` typed f64 ops, not domain-specific WGSL
shaders. GPU validation through `validate_basecamp_gpu` (18/18 PASS). CPU↔GPU
dispatch parity through `validate_compute_dispatch` (19/19 PASS). Mixed-hardware
routing through `validate_mixed_hardware` (21/21 PASS).

### Stack Coverage Summary

| Tier | Papers Covered | Total | Coverage |
|------|---------------|-------|----------|
| Python control (Py) | 26/26 + 5 WDM + 3 pub exp | 272 checks | **100%** |
| Rust CPU (Rs) | 26/26 + baseCamp + WDM + pub exp | 880 lib + 9 integration | **100%** |
| BarraCUDA CPU (bC) | 25/26 | 214 checks | **96%** |
| BarraCUDA GPU Tensor (gT) | 24/26 | 112+ checks | **92%** |
| BarraCUDA GPU (baseCamp) | 6/6 sub-theses | 18 checks | **100%** |
| metalForge WGSL (mF) | 15/25 | 108 checks | **100%**† |
| GPU Pipeline (gP) | 15/25 | 94 checks | **100%**† |
| Cross-dispatch (xD) | 15/15 | 49 checks | **100%**† |
| CPU↔GPU dispatch | 26 + baseCamp | 19 checks | **100%** |
| Mixed hardware (mH) | baseCamp | 21 checks | **100%** |

`†` 100% of applicable papers. Phase 0/0+ studies use PyTorch, not WGSL shaders.

**S74 addition**: `validate_gpu_pure_workload_all` (10/10 PASS) proves all 15 Phase 0++
paper domains run through typed BarraCUDA GPU ops (scalar-only readback). This is the
comprehensive "pure GPU" proof that complements the per-domain pipeline validators.
`bench_evolution_tiers` characterizes the CPU→GPU portability across 8 domains.

### What Changed (Phase 5b buildout, February 22, 2026)

**Phase 0/0+ gaps closed:**
- **Exp 001 Surrogate**: S-15 **RESOLVED** upstream (`a4996b34` S39) — new `validate_barracuda_surrogate` (7/7 PASS)
- **Exp 002 Transformer**: new `validate_barracuda_gpu_transformer` (7/7 PASS) — Q/K/V projection, attention scores, FFN block via GPU Tensor
- **Exp 004 Transfer**: new `validate_barracuda_transfer` (7/7 PASS) — domain adaptation MLP forward + metrics
- **Exp 003, Study 003, Study 004**: reclassified as gT (sequence/lenet/lstm validators use GPU Tensor)

**Cross-dispatch 100%:**
- New `validate_cross_dispatch_hmm` (4/4 PASS): Papers 016/018, `hmm_forward_log.wgsl` GPU ↔ CPU log-likelihood parity (diff=2.29e-6)
- New `validate_cross_dispatch_ode` (4/4 PASS): Paper 020, `rk4_parallel.wgsl` GPU ↔ CPU ODE integration parity (diff=1.49e-8)
- Existing cross-dispatch binaries already covered Papers 011-015, 017, 019, 021-025

**Previous sweep (Phase 5a+):**
- S-14/S-15/S-16 **RESOLVED** upstream (`a4996b34` S39: Naive tier removed, matmul hang fixed, transpose dispatch)
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
3. **Papers 022–023 (Kachkovskiy)** — Tridiagonal eigensolver → specialized `tridiag_eigh.wgsl` (**Pending** — `BarraCUDA` NAK eigensolve)
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
- **25/25 papers + 5 WDM + 3 publication experiments** (263 checks) use open data exclusively
- Sources: in-code synthetic (deterministic seed 42), Open-Meteo ERA5 (CC BY 4.0),
  MNIST (CC BY-SA 3.0), published reference data (MIT/Apache-2.0), FPEOS tables
  (Militzer), Stanton-Murillo transport model
- No proprietary, paywalled, or access-restricted data
- Python baseline drift detection: `control/check_drift.sh` (all 39 baselines — 25 papers + 5 WDM + 3 pub exp + 5 coralForge + 2 nS-06, ML inference doesn't produce baselines)
- **S88+ publication experiments**: Exp-050 (Py 11/11), Exp-052 (Py 8/8), Exp-053 (Py 11/11) — all synthetic/algorithmic data

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
- S-15 **RESOLVED** upstream (`a4996b34` S39); previously workaround: data magnitude ≥ 0.5

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

### Tier 9: GPU Dispatch — Pure GPU Promotion (Sessions 45–46, 66)
- **44 CPU→GPU promotions** via `gpu_dispatch::Dispatcher` (Phase A: 27, Phase B: 11, Phase C: 6)
- Capability-based routing: GPU when available, CPU fallback otherwise
- `validate_gpu_promotion`: **27/27 PASS** (both GPUs)
- `validate_gpu_phase_b`: **20/20 PASS** (both GPUs)
- `validate_gpu_phase_c`: **18/18 PASS** (Session 66: HMM chains, FST, introgression, AF variance)
- Covers: HMM (forward/backward/Viterbi + full chains), statistics (variance/correlation/allele freq),
  distances (L2/Hamming/Jaccard/geographic), ML (neural_forward/softmax/PCA),
  bio (fitness/diversity/Hill/replicator), ODE (RK4 step), population genetics (FST, AF variance)
- **~97% of production math** has GPU path. Remaining ~3%: full ODE loops, spatial stencil cooperation
- All per-paper controls still use open data: Python baselines validate independently
- **CPU↔Python parity** (Session 67): `validate_cpu_math_parity` 39/39 PASS (1e-10)
  confirms Rust CPU = Python/NumPy for 9 primitives + 9 paper kernels
- **Dispatch overhead** (Session 67b): `bench_dispatch_tiers` ≤1.04× for 9/10 ops
  — Dispatcher::cpu_only() is transparent

### Tier 10: metalForge Mixed Hardware (Future)
- **Capability-based dispatch across GPU + NPU + CPU**
- Infrastructure ready: `mixed.rs` (MixedSubstrate), `pcie_bridge.rs` (PCIe cost model)
- Validated: 16/16 dispatch routing + 16/16 mixed dispatch checks (Session 43)
- Next: Exercise on actual NPU hardware, optimize transfer cost model

---

## Controls Audit: BarraCUDA CPU → GPU → metalForge (Sessions 67–74)

Confirming all papers have controls across the three hardware tiers:

### BarraCUDA CPU Controls

| Validator | Papers | Checks | BarraCUDA Primitives |
|-----------|--------|--------|---------------------|
| `validate_cpu_math_parity` | All | 39/39 | 9 primitives + 9 kernels + 6 Dispatcher |
| `validate_barracuda_{domain}` | 24/25 | 203 | stats, linalg, special, numerical |
| `validate_barracuda_parity` | All 17 domains | 17/17 | CPU vs GPU parity per domain |
| **Gap**: Exp 005 (analytical only) | | | No numerical ops |
| **Gap**: Tridiagonal eigensolver | 022-023 | | Pending `BarraCUDA` NAK |

### BarraCUDA GPU Controls

| Validator | Papers | Checks | GPU Operations |
|-----------|--------|--------|---------------|
| `validate_barracuda_gpu_{domain}` | 23/25 | 98+ | Tensor matmul, transpose, tanh, sigmoid |
| `validate_gpu_phase_c` | 016-018, 024-025 | 18/18 | HMM chains, FST, introgression, AF var |
| `validate_basecamp_gpu` | baseCamp 01-05 | 14/14 | eigh, variance, Pearson, entropy, matmul |
| `bench_dispatch_tiers` | 10 kernels | — | Three-tier overhead characterization |
| **Gap**: Exp 005, Study 005 | | | Analytical / integer arithmetic |

### metalForge Mixed Hardware Controls

| Validator | Scope | Checks | Substrates |
|-----------|-------|--------|-----------|
| `validate_mixed_hardware` | Mixed dispatch | 14/14 | GPU↔NPU↔CPU routing |
| `validate_mixed_dispatch` | PCIe bridge | 16/16 | Transfer cost model |
| `validate_compute_dispatch` | CPU↔GPU parity | 16/16 | Dispatcher routing |
| `validate_metalforge_pcie` | PCIe tiers | 23/23 | Bandwidth + latency |

**Audit result**: All 25 papers + 5 baseCamp sub-theses have controls at every
applicable tier. Two known gaps (Exp 005 analytical, tridiag eigensolver pending NAK).
All controls use open data and open systems exclusively.

**Session 70 addendum**: Deep audit II — 93.5% coverage (580 tests, up from 505/90.43%).
Tolerance registry refactored to `tolerance_registry!` macro (891→257 lines, 107+ named).
100% SPDX AGPL-3.0-or-later compliance (211/211 files). All files ≤1000 lines.
BarraCUDA usage inventory: 90+ import sites, 60+ files, 20+ submodules, zero duplicate math.
Remaining uncovered lines (5.5%) are exclusively GPU error-handling branches.

**Session 81 addendum**: 25 new named tolerances (spectral, population genetics, game theory,
quantization, GPU, hardware). 21 validation binaries swept for inline magic numbers.
`spectral_entropy` rewired to `barracuda::stats::shannon_from_frequencies` (39th function).
Cross-platform probe gating. 7 PyTorch scripts fully seeded. 129+ total named tolerances.
All controls verified passing across BarraCUDA CPU, GPU, and metalForge mixed hardware.

**Session 88+ addendum**: 3 publication experiments (Exp-050/052/053) each have open-data
Python controls + Rust CPU validators using BarraCUDA primitives (eigh_f64, BatchIprGpu,
numerical_hessian, graph_laplacian, stencil_cooperation). Papers A/C/D data-ready.
Total: 263 Python checks, 861 lib tests, 210/210 validate_all. V54 `ToadStool` handoff
documents barracuda evolution surface, absorption targets, and cross-spring alignment.
All controls use open data and open systems exclusively.

**Session 88+ publication experiment GPU buildout**: 4 new validators push Exp-050/052/053
through the full GPU validation progression:

| Validator | Checks | Tier | What It Proves |
|-----------|--------|------|----------------|
| `validate_barracuda_training_trajectory` | 9/9 | GPU | Eigensolve → IPR → variance on GPU (Exp-050 Paper A) |
| `validate_barracuda_hessian_eigen` | 10/10 | GPU | Hessian eigensolve → spectral diagnostics on GPU (Exp-052 Paper D) |
| `validate_barracuda_anderson_multiagent` | 11/11 | GPU | Laplacian → disordered eigensolve → IPR + L2 on GPU (Exp-053 Paper C) |
| `validate_publication_gpu_pipeline` | 13/13 | Pipeline + metalForge | BatchIprGpu pure pipeline, Dispatcher CPU↔GPU parity, mixed-hardware routing |

**Session 88+ CPU parity and portability benchmarks**: 2 new validators:
- `validate_barracuda_cpu_bench` (31/31): Python/NumPy vs pure Rust across 14 paper domains (all Phase 0++), 38.6× geometric mean speedup
- `bench_portability_tiers` (9/9): CPU→GPU portability proof, 7 domains, `ToadStool` streaming

Total: **229 binaries**, **210/210 validate_all**.
Also fixed `validate_wdm_sqw` JSON schema mismatch (`spec_mean` → `series_mean` compat): 0/1 → 26/27.

**Session 88+ Phase 4 shader + streaming pipeline**: 2 new validators close direct
WGSL shader validation and `ToadStool` streaming proof gaps:

| Validator | Checks | Tier | What It Proves |
|-----------|--------|------|----------------|
| `validate_gpu_shader_phase4` | 22/22 | WGSL direct | HMM backward (1.19e-7), Viterbi (exact), matrix correlation (<1e-6), linear regression (slope 2.503 vs true 2.5) |
| `validate_streaming_spectral_pipeline` | 28/28 | Streaming | Batch eigensolve→IPR→stats (8 Hamiltonians), Anderson disorder sweep (6 W values, IPR 0.09→0.79), Dispatcher parity (1.6e-14) |

Total: **229 binaries**, **210/210 validate_all**.
`ToadStool` streaming pattern validated: unidirectional dispatch preserves scientific conclusions.

**Session 88+ debt reduction addendum**: Barracuda usage audit complete — 90+ import sites,
60+ files, 20+ submodules, 42 upstream rewires, zero duplicate math.
18 `unwrap_or_else(|e| panic!(...))` sites evolved to `.expect()` across WDM tests and
validation binaries. 11 manual loop sites evolved to idiomatic iterators (`chunks_exact`,
`flat_map`, `zip`) in `basecamp.rs` and `coral_forge.rs`. Control matrix confirmed:
all papers have controls at open data (Py), BarraCUDA CPU (Rs), BarraCUDA GPU (Tensor),
and metalForge mixed hardware tiers. Zero clippy warnings, zero fmt diffs.

**Session 74 addendum**: Pure GPU all-domains + cross-system dispatch — three new validators:

| Validator | Checks | Tier | What It Proves |
|-----------|--------|------|----------------|
| `validate_gpu_pure_workload_all` | 10/10 | GPU | All 15 Phase 0++ domains run through typed BarraCUDA GPU ops (scalar-only readback) |
| `validate_cross_system_dispatch` | 46/46 | metalForge | Full cross-system stack: hardware discovery, domain heuristics, CPU↔GPU parity, transfer cost model, NPU routing, crossover sweep |
| `bench_evolution_tiers` | — | Bench | CPU→GPU portability characterization across 8 domains |

This closes the "pure GPU final workload validation" milestone: every paper domain
has a typed GPU op validator, and metalForge's cross-system dispatch is proven
end-to-end (GPU→NPU→CPU). Total: **2480+ checks**, **229 binaries**, **210/210 validate_all**.

**Session 86 addendum**: WDM surrogate buildout — 4 new validators added:
- `validate_wdm_transport` (30/30): nW-01 Stanton-Murillo transport MLP
- `validate_wdm_eos` (36/36): nW-02 EOS surrogate P(ρ,T), E(ρ,T)
- `validate_barracuda_wdm_eos` (15/15): nW-02 GPU tier via `Tensor::matmul`
- `validate_wdm_transfer` (6/6): nW-04 classical→WDM transfer learning

**Session 87 addendum**: WDM surrogate queue closed — 2 new validators:
- `validate_wdm_sqw` (27/27): nW-03 LSTM reservoir on MD density fluctuation time series, predicts plasmon frequency (R²=0.98) and Landau damping (R²=0.98)
- `validate_wdm_esn` (39/39): nW-05 ESN regime classifier (Classical/WDM/Degenerate), 96.5% accuracy, bit-exact Rust↔Python score parity

New modules: `wdm_sqw.rs` (LSTM reservoir + pooled readout), `wdm_esn.rs` (ESN 2-step recurrence + ridge readout). Both use open data (synthetic from physics equations, seed=42). 31 baselines in `check_drift.sh`.

All WDM surrogates use open data (FPEOS tables from Militzer, Stanton-Murillo
transport model, synthetic plasma spectra). All controls deterministic with documented seeds.

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
Cross-spring evolution validator (39/39 PASS) proves rewired paths produce identical
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

## baseCamp Controls Verification (Sessions 50–105)

baseCamp modules (6 sub-theses, 167 checks) follow the same controls
framework. All data is in-code synthetic with deterministic seeds.

### baseCamp — BarraCUDA CPU Controls

All 6 baseCamp modules use `eigh_f64` (via `eigh.rs` → `barracuda::ops::linalg`)
for eigendecomposition. This is the same `eigh_f64` validated at 1.75e-14
accuracy (Householder+QR) in Phase 0++ papers 022–023.

| Module | BarraCUDA Primitive | Validation |
|--------|--------------------|----|
| `weight_spectral` | `eigh_f64` | Hamiltonian symmetry, ESD normalization, IPR range |
| `information_flow` | `eigh_f64` | Attention Hamiltonian spectral finiteness |
| `loss_landscape` | `eigh_f64` | Hessian spectrum matches analytical quadratic |
| `neural_pgm` | `eigh_f64` | Transition matrix spectral properties |
| `agent_coordination` | `eigh_f64` | Laplacian smallest eigenvalue ≈ 0 |
| `immunological_anderson` | `eigh_f64`, KL, Shannon | Pielou evenness, IC50, dimensional promotion |

### baseCamp — BarraCUDA GPU Promotions (Session 106 — COMPLETE)

All 4 high/medium priority GPU promotions validated via `validate_barracuda_basecamp`:

| Module | GPU Candidate | BarraCUDA Pattern | Status |
|--------|-------------|------------------|--------|
| `weight_spectral` | `weight_to_hamiltonian` | `mat_mul_gpu` (H² + eigensolve) | **DONE** — 6 checks |
| `loss_landscape` | `numerical_hessian` | `eigh_gpu` + `entropy_gpu` + `mat_mul_gpu` | **DONE** — 6 checks |
| `neural_pgm` | `belief_propagation_chain` | `hmm_forward_chain_gpu` + `mat_mul_gpu` | **DONE** — 5 checks |
| `agent_coordination` | `interaction_graph` | `pairwise_l2_matrix_gpu` + `eigh_gpu` | **DONE** — 7 checks |
| `loss_landscape` | `boltzmann_sampling` | `WrightFisherGpu` (parallel MCMC) | Low (future) |

CPU reference → GPU result → diff within tolerance. 26 total checks PASS.

### baseCamp — Full Three-Tier Hardware Validation (COMPLETE)

All baseCamp experiments validated across three hardware tiers:

1. **BarraCUDA CPU**: Pure Rust — 167/167 PASS (Sessions 50, 54, 105)
2. **BarraCUDA GPU**: Tensor API on RTX 4070 — 18/18 PASS (`validate_basecamp_gpu`)
3. **Dispatcher routing**: GPU/CPU parity — 19/19 PASS (`validate_compute_dispatch`)
4. **CPU↔GPU parity**: Cross-domain — 34/34 PASS (`validate_barracuda_parity`)
5. **metalForge mixed**: PCIe tiers + substrate selection — 36/36 PASS (`validate_metalforge_pcie`)
6. **nS-06 mixed hardware**: NUCLEUS tower/node/nest + PCIe — 7/7 PASS (`validate_mixed_hardware`)

**Total baseCamp checks**: 200+ (167 CPU + 18 GPU + 19 dispatch + 7 mH) — all PASS.

### baseCamp — Open Data Confirmation

| Source Type | Modules | License |
|-------------|---------|---------|
| In-code synthetic (seed=42) | All 6 baseCamp modules | N/A (pure math) |

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
| nS-06 | Gonzales (2013-2024), Fajgenbaum (2019), MATRIX/ARPA-H | All open | 53/53 Rust + 187/187 Rust ext + 20/20 Py + 28/28 Py ext + 4 GPU + 3 dispatch + 7 mH |

---

## WDM Surrogate Extensions — baseCamp Sub-thesis 07

**Purpose**: Extend neuralSpring's validated surrogate learning pipeline
to warm dense matter (WDM) physics, supporting hotSpring's Tier 4 WDM
reproduction targets and baseCamp Sub-thesis 07 (Sovereign WDM on Consumer GPU).

### Surrogate Models for WDM Transport

| # | Target | Method | BarraCUDA Primitive | Status |
|---|--------|--------|-------------------|--------|
| nW-01 | WDM transport surrogate (D*, η*, λ* vs ρ, T, Z*) | MLP/RBF — extend hotSpring Paper 3 (Diaw surrogate, 9/9) to WDM parameter range | `nuclear_eos_gpu` (GPU RBF validated) | **Complete** — Py 4/4, Rs 30/30 |
| nW-02 | EOS surrogate (P, E vs ρ, T for H, He, C) | MLP trained on FPEOS tables (Militzer) — extend Exp 001 FAO-56 surrogate to physics EOS | `validate_barracuda_surrogate` (7/7) | **Complete** — Py 9/9, Rs 36/36, GPU 15/15 |
| nW-03 | S(q,ω) peak predictor | LSTM reservoir on MD density fluctuation time series — predict plasmon ω and damping γ from (ρ, T) | `validate_wdm_sqw` (27/27) | **Complete** — Py 5/5, Rs 27/27 |

### Transfer Learning: Classical → WDM

| # | Target | Method | BarraCUDA Primitive | Status |
|---|--------|--------|-------------------|--------|
| nW-04 | Classical plasma → WDM transfer | Fine-tune transport surrogate from Stanton-Murillo (Γ,κ) space to WDM (ρ,T,Z*) space — same architecture as Exp 004 (MI→NM/CA transfer, 6/6) | Transfer learning pipeline (validated) | **Complete** — Py 4/4, Rs 6/6, GPU 7/7 |
| nW-05 | WDM regime classifier | ESN classifier: given (ρ,T), predict WDM regime (classical/WDM/degenerate) — reservoir computing with ridge regression readout | `validate_wdm_esn` (39/39) | **Complete** — Py 5/5, Rs 39/39 |

**Connection to hotSpring Tier 4**: Each surrogate replaces expensive MD
runs with inference at 9,017× less energy (NPU) or ~100× less compute
(GPU RBF). Classical→WDM transfer learning exploits the fact that
Stanton-Murillo transport coefficients are continuous functions of
coupling parameters — fine-tuning bridges the regime gap.

**Connection to baseCamp Sub-thesis 07 §4**: Phase 4 (distributed
parameter sweep) generates the training data; neuralSpring surrogates
provide instant lookup for the sweep results.

---

## coralForge — Protein/RNA/DNA Structure Prediction (NEW TRACK)

**Purpose**: Port OpenFold3's Evoformer + Structure Module to BarraCUDA WGSL
shaders for sovereign structure prediction on consumer GPUs.

**Where it lives**: `neuralSpring/coral_forge/`

### Papers

| # | Paper | Journal | Year | Why | Status |
|---|-------|---------|------|-----|--------|
| nF-01 | Ahdritz et al. "OpenFold: Retraining AlphaFold2 yields new insights" | Nature Methods | 2024 | Reference implementation (Apache 2.0). Baseline for porting | **Phase B.4 DONE** — Py 25/25, Rs 67/67, GPU 37/37 (15 shaders + pipeline, all AlphaFold2 primitives) |
| nF-02 | Jumper et al. "Highly accurate protein structure prediction with AlphaFold" | Nature 596:583-589 | 2021 | Original architecture. Evoformer + IPA specification | **Phase B DONE** — Py 19/19, Rs 18/18, bC 17/17 (full Evoformer block + IPA + backbone + torsion) |
| nF-03 | Abramson et al. "Accurate structure prediction for all molecules" (AlphaFold3) | Nature 630:493-500 | 2024 | Diffusion-based extension. RNA/DNA/ligand handling | **Phase C DONE** — Py 62/62 (diffusion 29 + pairformer 14 + confidence 19), Rs 55/55 (diffusion 26 + pairformer 13 + confidence 16), bC 13/13 (matmul, dot, mean, var, l2_norm for AF3 ops), 18 unit tests |

### nF-03 Phase A+B — Diffusion + Pairformer (DONE)

Python controls:
- `control/coral_forge/alphafold3_diffusion.py` (29/29 PASS)
- `control/coral_forge/alphafold3_pairformer.py` (14/14 PASS)

Rust modules: `src/coral_forge/diffusion.rs`, `src/coral_forge/pairformer.rs`
Validators:
- `validate_alphafold3_diffusion` (26/26 PASS, max diff 1.24e-14)
- `validate_alphafold3_pairformer` (13/13 PASS, max diff 6.66e-16)

Primitives implemented:
- Cosine/linear noise schedules, forward diffusion, DDPM/DDIM reverse steps
- SE(3)-equivariant noise (COM removal, translation invariance)
- Sinusoidal timestep embedding, pair conditioning
- Pairformer block: TriMul outgoing/incoming + TriAttn + FFN + timestep conditioning
- pLDDT confidence head (Linear → sigmoid)
- PAE confidence head (pair → softmax → expected distance)

Remaining: Phase C (confidence heads integration), Phase D (multi-molecule tokenization),
Phase E (full pipeline + MSA databases via NestGate).

### Phase A — Baseline Assessment (DONE)

Evaluation script: `specs/coral_forge_assessment/openfold3_eval.py` (9/9 checks)

- RTX 4070: 12 GB VRAM, Compute 8.9, Vulkan + SHADER_F64 confirmed
- PyTorch 2.9.0+cu128 available, 316x GPU speedup on attention
- 4 of 10 required primitives already exist in BarraCUDA
- See `coral_forge/BARRACUDA_FOLDING_REQUIREMENTS.md` for full shader spec
- See `coral_forge/MSA_DATABASE_PLAN.md` for data acquisition plan

### Phase B — Complete AlphaFold2 Primitive Validation (DONE)

Python control: `control/coral_forge/evoformer_primitives.py` (25/25 checks)
Rust modules: `src/coral_forge/` (41 unit tests, 67 validation checks)
GPU validator: `validate_coral_forge_gpu` (37/37 checks on RTX 4070)

15 WGSL shaders wired and validated (38 total in shader catalog).
All shaders use **df64 core streaming**: f64 buffer I/O → df64 compute on
FP32 cores → f64 output. `Fp64Strategy::Hybrid` auto-detected on RTX 4070
(1:64 FP64:FP32 ratio). Compilation via `compile_shader_f64_hybrid` which
prepends `df64_core.wgsl` + `df64_transcendentals.wgsl`.

| Shader | Algorithm | Precision tier | Max GPU-CPU diff |
|--------|-----------|----------------|------------------|
| `gelu_f64.wgsl` | Pointwise GELU | Transcendental | 3.41e-4 |
| `triangle_mul_outgoing_f64.wgsl` | Algorithm 11 | Arithmetic | 3.10e-7 |
| `triangle_mul_incoming_f64.wgsl` | Algorithm 12 | Arithmetic | 4.66e-7 |
| `sdpa_scores_f64.wgsl` | QKᵀ/√d (pass 1) | Arithmetic | 6.76e-8 |
| `triangle_attention_f64.wgsl` | Algorithms 13-14 | Arithmetic | 1.54e-7 |
| `softmax_f64.wgsl` | Row-wise softmax (pass 2) | Transcendental | 2.92e-4 |
| `attention_apply_f64.wgsl` | Σ weights × V (pass 3) | Arithmetic | 6.89e-8 |
| `layer_norm_f64.wgsl` | LayerNorm (with `sqrt_df64`) | Arithmetic | 5.58e-7 |
| `sigmoid_f64.wgsl` | Sigmoid gate (`exp_df64`) | Transcendental | (CPU validated) |
| `outer_product_mean_f64.wgsl` | MSA → pair (OPM) | Arithmetic | 6.43e-8 |
| `msa_row_attention_scores_f64.wgsl` | Row attn + pair bias | Arithmetic | 1.06e-7 |
| `msa_col_attention_scores_f64.wgsl` | Col attn (no bias) | Arithmetic | 9.57e-8 |
| `ipa_scores_f64.wgsl` | IPA (SE(3)-equivariant) | Arithmetic | 3.40e-7 |
| `backbone_update_f64.wgsl` | Frame composition | Arithmetic | 3.59e-8 |
| `torsion_angles_f64.wgsl` | Fused `ResNet` + normalize | Arithmetic | 1.10e-7 |

#### Precision hierarchy

The df64 core streaming approach creates two distinct precision tiers, both
significantly better than pure f32 (which would give ~1e-3 to 1e-2 error):

| Tier | Operations | Tolerance | Observed range | Bottleneck |
|------|-----------|-----------|----------------|------------|
| **Arithmetic** | dot products, matrix multiply, accumulation, `sqrt_df64` (Newton-Raphson) | 1e-6 | 3.6e-8 to 5.6e-7 | f32 FMA error tracking in `two_prod`/`two_sum` |
| **Transcendental** | `exp_df64`, `tanh_df64` (degree-6 Horner polynomial) | 5e-4 | 1.7e-4 to 3.4e-4 | Polynomial approximation truncation error |

Full precision ladder (consumer GPU → data-center GPU):

| Level | Mantissa bits | Decimal digits | Source | Use case |
|-------|---------------|----------------|--------|----------|
| fp16 | 10 | ~3 | Native | Inference, training lower precision |
| bf16 | 7 | ~2 | Native | Training dynamic range |
| f32 | 23 | ~7 | Native | Standard GPU compute |
| **df64 (fp48)** | **~48** | **~14** | **Emulated (f32 pairs)** | **Scientific validation on consumer GPUs** |
| f64 | 52 | ~15.9 | Native (data-center only) | Gold standard, 1:2 FP64:FP32 GPUs |

The df64 approach achieves ~9.9x throughput vs native f64 on consumer GPUs
by leveraging the full FP32 core count (RTX 4070: 5888 FP32 cores vs 92
FP64 cores at 1:64 ratio). For scientific validation, df64/fp48 precision
(~14 digits) is sufficient — the limiting factor becomes the polynomial
approximation in transcendentals, not the arithmetic. Production ML
inference will run at f32 or lower; the f64 validation proves correctness.

**Complete AlphaFold2 primitive set**: all 9 Evoformer block operations and
all 3 Structure Module primitives (IPA scores, backbone frame update, torsion
angle prediction) have validated GPU shaders. **3-pass SDPA GPU pipeline**
(scores → softmax → apply): max diff 1.71e-4. IPA shader includes three-term
attention with SE(3)-equivariant point distance through backbone frames.
Backbone update uses quaternion-to-rotation with df64 matrix multiply.
Torsion prediction is a fused `ResNet` + unit circle normalization kernel.

CPU Rust vs Python baseline parity: max diff 3.6e-15 (machine precision).
IPA scores max diff: 3.6e-15. Backbone rotation max diff: 1.1e-16.
Torsion max diff: 1.4e-15. All unit circle constraints preserved (||(sin,cos)|| = 1).
All operations deterministic with seed=42. Open data only (synthetic).

### Phase B.2 — Remaining Evoformer Primitives (DONE)

New primitives added:
- **Outer product mean**: MSA → pair representation bridge. Converts evolutionary
  covariance (MSA sequences) into structural contact information (pair matrix).
  df64 accumulation over sequence dimension.
- **MSA row attention** (with pair bias): Per-sequence attention over residue
  positions. Pair bias from the pair representation injects structural priors.
- **MSA column attention**: Per-position attention across MSA sequences.
  Captures sequence-level relationships at each residue position.

---

## LTEE GuideStone Queue (Barrick/Lenski + Eaves/Woldring)

Targeted paper reproductions for the LTEE Targeted GuideStone artifact — a USB-deployable
validation subsystem of projectNUCLEUS. See `infra/whitePaper/gen4/architecture/GUIDESTONE_LTEE.md`
and `infra/whitePaper/attsi/non-anon/contact/barrick/PAPER_REVIEW_AND_SPRING_TARGETS.md`.

| ID | Paper | What to Reproduce | Exp | Status |
|----|-------|-------------------|-----|--------|
| B1 | Barrick et al. 2009 "Genome evolution" *Nature* | LSTM time-series prediction of mutation accumulation curves; ESN regime classifier for mutation-rate shift detection | TBD | QUEUED |
| B2 | Wiser et al. 2013 "Long-term dynamics" *Science* | LSTM prediction: train on 0-20K, predict 20K-50K; ESN regime detection at inflection points | TBD | QUEUED |
| B3 | Good et al. 2017 "Dynamics of molecular evolution" *Nature* | LSTM allele frequency trajectory prediction; HMM clade state detection; ESN regime classification (sweep vs interference vs coexistence) | TBD | QUEUED |
| B4 | Blount et al. 2008/2012 Citrate innovation | Early warning ESN on pre-citrate allele trajectories; detect potentiation before innovation | TBD | QUEUED |
| B6 | "Measuring the burden of hundreds of BioBricks" 2024 *Nat Comms* | ML prediction of burden from sequence features (GC%, codon usage, promoter strength) | TBD | QUEUED |
| B7 | Tenaillon et al. 2016 "Tempo and mode" *Nature* | ML detection of parallel evolution (same genes mutated independently across populations); transfer learning | TBD | QUEUED |
| B8 | Barrick & Waters 2025 "Phages use contingency loci" *bioRxiv* | ML prediction of contingency loci from sequence features; Anderson disorder mapping | TBD | QUEUED |
| B9 | DFE Evolution in LTEE 2024 *Science* | LSTM prediction of DFE parameters at generation t+1; ESN detection of DFE regime shifts | TBD | QUEUED |
| E2 | Mardikoraem & Woldring 2025 "HOLIgraph" *J Cheminformatics* | GNN for protein-ligand prediction; GPU-accelerated inference via barraCuda | TBD | QUEUED |
| E3 | Dolgikh et al. 2025 "Tuning Yeast Glycosylation for FLS2" *bioRxiv* | ML prediction of glycosylation effects on binding affinity | TBD | QUEUED |
| E4 | Woldring Lab 2024 "Screening macrocyclic peptide libraries" *bioRxiv* | ML ranking of binders from sequence features; transfer learning linear → cyclic | TBD | QUEUED |
| E5 | Woldring Lab 2023 "Single-Cell scFab Libraries" | Antibody pairing prediction from single-cell data; ML for VH/VL pairing | TBD | QUEUED |
