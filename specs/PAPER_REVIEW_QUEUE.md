# neuralSpring — Paper Review Queue

**Last Updated**: February 20, 2026
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

## Completion Summary

**All 25 papers complete as of February 20, 2026.** No queued items remain.

| Faculty | Papers | Python Checks | Rust Checks |
|---------|--------|---------------|-------------|
| Dolson (MSU CS) | 011–015 (5) | 46 | 50 |
| Liu (MSU CSE) | 016–018 (3) | 26 | 38 |
| Waters (MSU Micro) | 019–021 (3) | 23 | 21 |
| Kachkovskiy (MSU Math) | 022–023 (2) | 16 | 16 |
| R. Anderson (Carleton) | 024–025 (2) | 16 | 16 |
| **Total Phase 0++** | **15** | **127** | **141** |

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
10. **GPU PRNG → Wright-Fisher / Gillespie** — stochastic GPU pipelines (**Next**)

---

## Notes

- Paper 011 is the single most important paper in the entire ecosystem — it externally validates the constrained evolution methodology
- Papers 016–018 (Liu) bridge neuralSpring to wetSpring's metagenomics work
- Papers 019–021 (Waters) connect optimization theory to real biological dynamics
- Papers 022–023 (Kachkovskiy) provide the mathematical foundation for understanding loss landscapes and training dynamics
- Dolson papers (011–015) form a coherent sequence from theory → metrics → applications → swarm
- All 13 Phase 0++ modules are Tier A (pure math, direct port) — ready for BarraCUDA CPU evolution
