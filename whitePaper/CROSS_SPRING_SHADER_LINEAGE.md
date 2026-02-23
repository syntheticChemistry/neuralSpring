# Cross-Spring Shader Evolution Lineage

> How three Springs evolved WGSL shaders that ToadStool/BarraCUDA absorbed
> into a universal compute engine — and how each Spring now benefits from
> the others' contributions.

| Field | Value |
|-------|-------|
| ToadStool HEAD | `b41ee5f4` (Session 47 + S45/S46/S49 absorption) |
| Last updated | February 23, 2026 (Sessions 40–48) |
| BarraCUDA shader count | 645+ WGSL (zero CPU-only production math, S49) |

---

## The Three Springs

| Spring | Domain | Key Contribution to BarraCUDA |
|--------|--------|-------------------------------|
| **hotSpring** | Nuclear/particle physics | HFB self-consistent field, lattice QCD, spectral theory, ESN reservoir, precision infrastructure |
| **wetSpring** | Bioinformatics/genomics | Smith-Waterman, Gillespie SSA, Felsenstein, HMM f64 batch, SNP calling, dN/dS, precision polyfills |
| **neuralSpring** | ML validation/evolution | Pairwise ops, batch fitness, spatial payoff, IPR, hill gate, matmul tiers, eigh Householder+QR |

---

## Shader Origin Map

### neuralSpring → BarraCUDA (15 contributions)

| Shader / API | BarraCUDA Location | Absorbed At | Status |
|-------------|-------------------|-------------|--------|
| `matmul_cpu_tiled.wgsl` | `shaders/math/` | `82f953c8` (S-02) | Identical copy |
| `matmul_gpu_evolved.wgsl` | `shaders/math/` | `82f953c8` (S-02) | Identical copy |
| `eigh_f64` (Householder+QR) | `ops::linalg::eigh_f64` | `77f70b2e` (S-12) | Identical copy |
| `pairwise_hamming.wgsl` | `shaders/math/` | `77f70b2e` (S-25) | Identical copy |
| `pairwise_jaccard.wgsl` | `shaders/math/` | `77f70b2e` (S-25) | Identical copy |
| `locus_variance.wgsl` | `shaders/bio/` | `77f70b2e` (S-25) | Identical copy |
| `spatial_payoff.wgsl` | `shaders/math/` | `77f70b2e` (S-25) | Identical copy |
| `batch_ipr.wgsl` | `shaders/spectral/` | `77f70b2e` (S-25) | Identical copy |
| `hmm_forward_log.wgsl` | `shaders/ml/` | `77f70b2e` (S-25) | Identical copy |
| `batch_fitness_eval.wgsl` | `shaders/ml/` | `77f70b2e` (S-25) | Identical copy |
| `rk4_parallel.wgsl` | `shaders/numerical/` | `77f70b2e` (S-25) | Identical copy |
| `pairwise_l2.wgsl` | `shaders/math/` | `5437c170` (S-42) | Generalized variant |
| `multi_obj_fitness.wgsl` | `shaders/bio/` | `5437c170` (S-42) | Generalized variant |
| `hill_gate.wgsl` | `shaders/bio/` | `5437c170` (S-42) | Generalized variant |
| `swarm_nn_forward.wgsl` | `shaders/bio/` | `5437c170` (S-42) | Generalized variant |

### hotSpring → BarraCUDA (20+ contributions)

| Shader / API | BarraCUDA Location | Domain |
|-------------|-------------------|--------|
| `complex_f64.wgsl` | `shaders/math/` | Complex arithmetic (f64) |
| `su3.wgsl` | `shaders/math/` | SU(3) gauge group |
| `wilson_plaquette_f64.wgsl` | `shaders/lattice/` | Lattice QCD plaquette |
| `su3_hmc_force_f64.wgsl` | `shaders/lattice/` | HMC molecular dynamics |
| `higgs_u1_hmc_f64.wgsl` | `shaders/lattice/` | Higgs U(1) Abelian model |
| `dirac_staggered_f64.wgsl` | `shaders/lattice/` | Staggered fermion Dirac op |
| `cg_kernels_f64.wgsl` | `shaders/lattice/` | Conjugate gradient solver |
| `batched_hfb_density_f64.wgsl` | `shaders/science/` | Nuclear HFB density |
| `batched_hfb_hamiltonian_f64.wgsl` | `shaders/science/` | Nuclear HFB Hamiltonian |
| `batched_hfb_potentials_f64.wgsl` | `shaders/science/` | Nuclear HFB potentials |
| `batched_hfb_energy_f64.wgsl` | `shaders/science/` | Nuclear HFB energy |
| `bcs_bisection_f64.wgsl` | `shaders/science/` | BCS gap equation |
| `deformed_wavefunction_f64.wgsl` | `shaders/science/` | Axially-deformed HFB |
| `deformed_density_f64.wgsl` | `shaders/science/` | Deformed nuclear density |
| `deformed_potential_f64.wgsl` | `shaders/science/` | Deformed Skyrme+Coulomb |
| `deformed_hamiltonian_f64.wgsl` | `shaders/science/` | Deformed Hamiltonian |
| `deformed_bcs_f64.wgsl` | `shaders/science/` | Deformed BCS pairing |
| `esn_reservoir_update.wgsl` | `shaders/ml/` | Echo State Network reservoir |
| `esn_readout.wgsl` | `shaders/ml/` | ESN readout layer |
| `spin_orbit_f64.wgsl` | `shaders/grid/` | Spin-orbit coupling |

### wetSpring → BarraCUDA (15+ contributions)

| Shader / API | BarraCUDA Location | Domain |
|-------------|-------------------|--------|
| `smith_waterman_banded_f64.wgsl` | `shaders/bio/` | Sequence alignment |
| `gillespie_ssa_f64.wgsl` | `shaders/bio/` | Stochastic simulation |
| `tree_inference_f64.wgsl` | `shaders/bio/` | Decision tree inference |
| `felsenstein_f64.wgsl` | `shaders/bio/` | Phylogenetic likelihood |
| `hmm_forward_f64.wgsl` | `shaders/bio/` | HMM forward (f64 batch) |
| `ani_batch_f64.wgsl` | `shaders/bio/` | Average Nucleotide Identity |
| `dnds_batch_f64.wgsl` | `shaders/bio/` | dN/dS selection pressure |
| `pangenome_classify.wgsl` | `shaders/bio/` | Pangenome classification |
| `snp_calling_f64.wgsl` | `shaders/bio/` | SNP variant calling |
| `dada2_e_step.wgsl` | `shaders/bio/` | DADA2 E-step denoising |
| `quality_filter.wgsl` | `shaders/bio/` | Read quality filtering |
| `rf_batch_inference.wgsl` | `shaders/bio/` | Random Forest inference |
| `bray_curtis_f64.wgsl` | `shaders/math/` | Bray-Curtis distance |
| `batched_qs_ode_rk4_f64.wgsl` | `shaders/numerical/` | QS/c-di-GMP ODE (RK4) |
| `mean_reduce.wgsl` | `shaders/reduce/` | Mean reduction |

---

## Cross-Spring Benefits — What Each Spring Gets From the Others

### neuralSpring benefits from hotSpring

| From hotSpring | How neuralSpring uses it |
|---------------|------------------------|
| `ReduceScalarPipeline` | Available for mean/sum/max reductions in validators |
| `GpuDriverProfile` + NAK workarounds | Automatic driver-specific shader compilation |
| `SHADER_F64` feature negotiation | Enables `HmmBatchForwardF64` on RTX 4070 |
| ESN `export_weights`/`import_weights` | Weight persistence for evolved echo state networks |
| Spectral theory (Lanczos, Anderson) | Shared `BatchIprGpu` API for Anderson localization |
| `complex_f64.wgsl` | Foundation for any future complex-valued ML ops |

### neuralSpring benefits from wetSpring

| From wetSpring | How neuralSpring uses it |
|---------------|------------------------|
| `HmmBatchForwardF64` | **10⁹× precision improvement** over local f32 HMM (diff: 2.47e-10 vs 0.5 tol) |
| `hmm_forward_f64.wgsl` | f64 batch HMM replaces neuralSpring's per-timestep f32 dispatch |
| `log_f64` precision polyfill | Automatic f64 log accuracy on RTX 4070 (Ada Lovelace) |
| `(zero + literal)` f64 workaround | TS-001 `pow_f64` fix flows through to all f64 ops |
| `SmithWatermanGpu` | Available for sequence alignment validation (Paper 017) |
| `GillespieGpu` | Available for stochastic eco-dynamics simulation |
| `FelsensteinGpu` | Available for phylogenetic likelihood computation |

### wetSpring benefits from neuralSpring

| From neuralSpring | How wetSpring could use it |
|------------------|---------------------------|
| `pairwise_hamming.wgsl` | Genomic distance computation (via `PairwiseHammingGpu`) |
| `pairwise_jaccard.wgsl` | Pangenome PA matrix distances (via `PairwiseJaccardGpu`) |
| `locus_variance.wgsl` | Fst/population genetics (via `LocusVarianceGpu`) |
| `batch_fitness_eval.wgsl` | Selection pressure quantification (via `BatchFitnessGpu`) |
| `batch_ipr.wgsl` | Eigenvector localization in network analysis |
| `matmul_gpu_evolved.wgsl` | General matrix operations (via Tensor API) |

### hotSpring benefits from neuralSpring

| From neuralSpring | How hotSpring could use it |
|------------------|---------------------------|
| `rk4_parallel.wgsl` | ODE integration for nuclear dynamics |
| `hill_gate.wgsl` | Signal integration in nuclear reaction networks |
| `eigh_f64` Householder+QR | Eigenvalue decomposition for nuclear matrix problems |
| `matmul_cpu_tiled.wgsl` | CPU-fallback matmul for testing |

---

## Validation Evidence

### neuralSpring upstream wrapper validation (Feb 22, 2026)

| Validator | Checks | Result | Max Diff |
|-----------|--------|--------|----------|
| `validate_barracuda_bio_ops` | 12 | ALL PASS | 1.91e-6 (SpatialPayoff) |
| `validate_barracuda_hmm_f64` | 11 | ALL PASS | 2.47e-10 (100-obs LL) |
| Library tests | 255 | ALL PASS | — |

### Benchmark: local vs upstream dispatch (RTX 4070, release)

| Kernel | Local µs | Upstream µs | Ratio |
|--------|---------|------------|-------|
| BatchFitness 10K×32 | 1683.7 | 1951.8 | 1.16× |
| Hamming 200×500 | 2291.7 | 2351.3 | 1.03× |
| Jaccard 100×500 | 2745.0 | 2530.4 | 0.92× |
| LocusVariance 50×500 | 2319.4 | 2606.1 | 1.12× |
| SpatialPayoff 256² | 2440.1 | 2353.3 | 0.96× |
| BatchIPR 1000×256 | 1741.3 | 1795.9 | 1.03× |

Upstream wrappers have **negligible overhead** (0.92–1.16×). Jaccard is actually
faster through the wrapper due to better buffer management.

---

## Evolution Timeline

```
Feb 16  hotSpring: complex_f64, su3, spin_orbit → BarraCUDA S-18
        wetSpring: bray_curtis_f64 → BarraCUDA

Feb 20  wetSpring: SmithWaterman, Gillespie, Felsenstein, TreeInference → BarraCUDA
        wetSpring: dada2, quality_filter, snp_calling → BarraCUDA

Feb 21  neuralSpring: 8 identical-copy shaders → BarraCUDA 77f70b2e (S-25)
        wetSpring: hmm_forward_f64, ani, dnds, pangenome_classify → BarraCUDA (S-27)
        hotSpring: ESN shaders, Dirac staggered, CG kernels → BarraCUDA (S-26)

Feb 22  neuralSpring: 5 generalized-variant shaders → BarraCUDA 5437c170 (S-42)
        hotSpring: HFB spherical + deformed (10 shaders) → BarraCUDA (S-39)
        ToadStool: Dead code sweep, S-13 PooledBuffer race fix
        neuralSpring: Rewire to upstream APIs, validate, benchmark

        neuralSpring: 6/6 dual-path upstream parity (0.00e0 bit-identical)
        neuralSpring: ReduceScalarPipeline f64 mean IPR (5.55e-17 diff)
        neuralSpring: barracuda::spectral theory stack validated (14/14 PASS)
        Cross-spring: hotSpring spectral theory → barracuda → neuralSpring validates

Feb 23  Session 47: ToadStool S45/S46/S49 absorbed (c8076a2d, fe573095, 9bd71391)
        MHA S-03b FIXED upstream (z-dimension dispatch) — flows from ToadStool S46
        10 validators rewired: raw wgpu → typed BarraCUDA ops (cross-spring absorption complete)
        HmmBatchForwardF64 (wetSpring) now primary HMM path — evolved/hmm_forward_gpu retired
        BatchedEighGpu (hotSpring) now eigensolve path — eigh_gpu via single dispatch (n≤32)
        bench_cross_spring_evolution: neuralSpring + wetSpring + hotSpring ops in one benchmark

Feb 23  Session 48: Mass typed-op rewiring — 28 binaries converted from raw wgpu to typed BarraCUDA ops
        2 standalone + 15 pipeline + 6 cross-dispatch validators + bench_gpu_kernels
        HillGateGpu f64 graceful skip (RTX 4070 driver limitation)
        f32→f64 data type fixes for 6 ops (upstream S49 sync)
        Validation: 132/133 (validate_barracuda_logsumexp pre-existing driver issue)
```

---

## Session 47 — Typed Op Migration & Cross-Spring Convergence (February 23, 2026)

### MHA S-03b Fix

The MHA projection shader z-dimension dispatch bug was **FIXED upstream** in ToadStool S46
(`fe573095`). The fix flows back to neuralSpring via path dependency. Native
`Tensor::multi_head_attention` projection execution no longer hangs on RTX 4070.

### 10 Validators Rewired to Typed BarraCUDA Ops

All 10 domain validators migrated from raw wgpu dispatch to typed BarraCUDA ops.
Cross-spring absorption is **complete**:

| Validator | Typed Op | Spring |
|-----------|----------|--------|
| `validate_gpu_batch_fitness` | BatchFitnessGpu | neuralSpring |
| `validate_gpu_sate` | PairwiseHammingGpu | neuralSpring |
| `validate_gpu_pangenome` | PairwiseJaccardGpu | neuralSpring |
| `validate_gpu_meta_pop` | LocusVarianceGpu | neuralSpring |
| `validate_gpu_game_theory` | SpatialPayoffGpu | neuralSpring |
| `validate_gpu_directed` | MultiObjFitnessGpu | neuralSpring |
| `validate_gpu_modes` | PairwiseL2Gpu | neuralSpring |
| `validate_gpu_anderson` | BatchIprGpu | neuralSpring |
| `validate_gpu_swarm` | SwarmNnGpu | neuralSpring |
| `validate_gpu_signal` | HillGateGpu | neuralSpring |

### HMM: wetSpring Primary Path

- **Retired**: `evolved/hmm_forward_gpu.rs` (351 lines) — fossil at `metalForge/fossils/evolved_hmm_forward_gpu/`
- **Primary**: `HmmBatchForwardF64` (wetSpring origin) — f64, batch, BarraCUDA shader-first

### Eigensolve: hotSpring Primary Path

- **eigh_gpu**: Via `BatchedEighGpu` (single-dispatch for n≤32)
- **disorder_sweep_gpu**: Batch eigensolve + mean IPR
- **spectrum_chi_squared_gpu**, **selection_coefficient_gpu**: Pangenome GPU dispatch

### bench_cross_spring_evolution

New benchmark demonstrating the full cross-spring cycle — ops from all three Springs:

- **neuralSpring**: BatchFitnessGpu, PairwiseL2Gpu, BatchIprGpu, SpatialPayoffGpu, PairwiseHammingGpu
- **wetSpring**: HmmBatchForwardF64
- **hotSpring**: BatchedEighGpu

---

## Session 48 — Mass Typed Op Rewiring & Cross-Spring Benchmarks (February 23, 2026)

### 28 Binaries Rewired from Raw wgpu to Typed BarraCUDA Ops

Session 48 completed a major rewiring: 28 validation/benchmark binaries converted from raw
wgpu dispatch (include_str! local shaders + manual pipeline/bindgroup/encoder creation)
to modern BarraCUDA typed op APIs. This removes thousands of lines of boilerplate and
validates the upstream ToadStool/BarraCUDA APIs directly.

| Category | Count | Examples |
|----------|-------|----------|
| Standalone validators | 2 | wright_fisher → WrightFisherGpu (f64), stencil → StencilCooperationGpu (f64) |
| Pipeline validators | 15 | All use typed BarraCUDA ops + CPU mean instead of raw wgpu shader chains |
| Cross-dispatch validators | 6 | All use typed BarraCUDA ops |
| Benchmarks | 1 | bench_gpu_kernels.rs — 5 benchmarks now use typed ops |

### HillGateGpu f64 Graceful Skip

On RTX 4070 (driver limitation), HillGateGpu f64 is skipped gracefully. The f32 path
remains validated. This pattern is documented for ToadStool S49 upstream sync.

### Cross-Spring Benchmark Results (RTX 4070, Vulkan)

| Op | Origin | Time (ms) |
|----|--------|-----------|
| BatchFitnessGpu | neuralSpring | 44 |
| PairwiseL2Gpu | neuralSpring | 8.7 |
| BatchIprGpu | neuralSpring | 7 |
| SpatialPayoffGpu | neuralSpring | 5.3 |
| PairwiseHammingGpu | neuralSpring | 5.2 |
| HmmBatchForwardF64 | wetSpring | 7.2 |
| BatchedEighGpu | hotSpring | 17.5 |

### GPU Kernels Benchmark (Typed Ops)

| Kernel | GPU | Rust CPU | GPU Advantage |
|--------|-----|----------|---------------|
| Large Hamming | 5.6 ms | 8.2 ms | 1.4× |
| Large Jaccard | 5.6 ms | 13.3 ms | 2.4× |
| Large Fitness (50000×64) | 6 ms | — | — |

### f32→f64 Evolution Tracking (ToadStool S49 Upstream Sync)

These ops moved from f32 to f64 in ToadStool S49: BatchFitnessGpu, LocusVarianceGpu,
MultiObjFitnessGpu, WrightFisherGpu, StencilCooperationGpu, SwarmNnGpu.

### Validation: 132/133

Only pre-existing `validate_barracuda_logsumexp` driver issue remains. All other
validators PASS.

---

## Files

| Purpose | Path |
|---------|------|
| This document | `whitePaper/CROSS_SPRING_SHADER_LINEAGE.md` |
| neuralSpring absorption tracker | `metalForge/shaders/ABSORPTION_TRACKER.md` |
| neuralSpring absorption manifest | `metalForge/ABSORPTION_MANIFEST.md` |
| Upstream bio ops validator | `src/bin/validate_barracuda_bio_ops.rs` |
| Upstream HMM f64 validator | `src/bin/validate_barracuda_hmm_f64.rs` |
| Local vs upstream benchmark | `src/bin/bench_upstream_vs_local.rs` |
| Spectral theory validator | `src/bin/validate_barracuda_spectral_theory.rs` |
| Cross-spring benchmark | `src/bin/bench_cross_spring_evolution.rs` |
| V15 handoff document | `wateringHole/handoffs/archive/NEURALSPRING_V15_SESSION47_HANDOFF_FEB23_2026.md` |
| V16 handoff document | `wateringHole/handoffs/NEURALSPRING_V16_SESSION48_HANDOFF_FEB23_2026.md` |
