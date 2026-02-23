# neuralSpring — Paper Review Queue

**Last Updated**: February 22, 2026
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

Session 42 verified: all `cargo fmt`, `clippy` (pedantic), and `doc` gates pass clean. 264 lib + 9 integration tests.

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

## Full Validation Stack Matrix (February 22, 2026)

Each paper maps through 7 validation tiers. The stack proves correctness
from Python baseline through mixed-hardware GPU dispatch.

### Legend

- **Py**: Python control baseline (Phase 0/0+/0++)
- **Rs**: Pure Rust CPU validation (Phase 1a)
- **bC**: BarraCUDA CPU primitives (Phase 2)
- **gT**: BarraCUDA GPU Tensor — `matmul`, `transpose`, `tanh`, `sigmoid`, `add` (Phase 5b)
- **mF**: metalForge WGSL shader — domain-specific GPU kernel (Phase 3c)
- **gP**: GPU Pipeline — chained domain→reduce (Phase 4b)
- **xD**: Cross-dispatch CPU↔GPU parity (Phase 3d)

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
| Study 003 LeNet-5 | ✓ | ✓ | lenet ✓ | lenet ✓ | — | — | **4/4** |
| Study 004 LSTM | ✓ | ✓ | lstm ✓ | lstm ✓ | — | — | **4/4** |
| Study 005 Quantized | ✓ | ✓ | quantized ✓ | — | — | — | **3/3** |

Phase 0/0+ studies use PyTorch training workflows. mF/gP columns are N/A.
Study 005 uses integer arithmetic (Q8/Q4), not Tensor ops — gT is N/A.

### Stack Coverage Summary

| Tier | Papers Covered | Total | Coverage | Delta |
|------|---------------|-------|----------|-------|
| Python control (Py) | 25/25 | 206 checks | **100%** | — |
| Rust CPU (Rs) | 25/25 | 264+ lib + 9 integration checks | **100%** | — |
| BarraCUDA CPU (bC) | 24/25 | 203 checks | **96%** | +12pp (was 84%) |
| BarraCUDA GPU Tensor (gT) | 23/25 | 98+ checks | **92%** | +20pp (was 72%) |
| metalForge WGSL (mF) | 15/25 | 108 checks | **100%**† | — |
| GPU Pipeline (gP) | 15/25 | 94 checks | **100%**† | — |
| Cross-dispatch (xD) | 15/15 | 49 checks | **100%**† | +80pp (was 20%) |

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
10. **GPU PRNG → Wright-Fisher / Gillespie** — stochastic GPU pipelines (**Next**)

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

### Tier 7: metalForge Mixed Hardware
- **Capability-based dispatch**: 12 validators use `Gpu::dispatch_1d()` with
  runtime hardware validation via `GpuCapabilities`
- **Cross-eigensolver**: Dense Householder+QR vs tridiag Sturm bisection agree
  at machine epsilon (2.89e-15 at n=64)
- **RTX 4070 + llvmpipe**: Validated on both GPU (Vulkan) and CPU (software)
  backends via the same WGSL source
- **4-tier kernel router**: DeviceCapabilities-driven matmul selection — best
  kernel per hardware configuration

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
