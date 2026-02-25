# Cross-Spring Shader Evolution Lineage

> How three Springs evolved WGSL shaders that ToadStool/BarraCUDA absorbed
> into a universal compute engine — and how each Spring now benefits from
> the others' contributions.

| Field | Value |
|-------|-------|
| ToadStool HEAD | `02207c4a` (Sessions 59–69 sync, Feb 25, 2026) |
| Last updated | February 25, 2026 (Sessions 40–69) |
| BarraCUDA shader count | 645+ WGSL (zero CPU-only production math, S49) |
| Shaders absorbed | 21/21 — all neuralSpring WGSL now upstream |
| `BandwidthTier` detection | Wired into Dispatcher (S64) — `PciE4x16` on RTX 4070 |

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
| `VarianceReduceF64` (Welford) | **3–4.5× faster** than f32 Tensor variance, f64 precision (S-53 rewire) |

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
| `FusedMapReduceF64` | **1.5–2.4× faster** entropy with fused f64 map-reduce (S-53 rewire) |

### neuralSpring benefits from wetSpring + hotSpring combined

| From both Springs | How neuralSpring uses it |
|-------------------|------------------------|
| `CorrelationF64` | Single-dispatch f64 Pearson correlation (wetSpring stats + hotSpring precision) |
| `chi_squared_statistic` | CPU fallback chi-squared via barracuda::special (S-53 rewire) |
| `pow_f64` polyfill (S-17) | hotSpring `math_f64.wgsl` + wetSpring constant fix → `HillGate` f64 works |

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

### neuralSpring validation (Feb 25, 2026, S64)

| Validator | Checks | Result |
|-----------|--------|--------|
| `validate_cross_spring_evolution` | 22 | 22/22 PASS |
| `validate_all` | 146 | 145/146 PASS (1 pre-existing logsumexp) |
| `cargo test --lib` | 500 | 500 PASS |
| `cargo clippy --all-targets` (pedantic + nursery) | — | 0 warnings |
| `BandwidthTier` detection | RTX 4070 | `PciE4x16` detected |

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

## Sessions 50–52 — ToadStool Sync, Code Quality & Absorption (February 24, 2026)

### ToadStool Sync (16 commits absorbed, `b41ee5f4` → `9abd6857`)

Sessions 50–53 of ToadStool absorbed 6 additional neuralSpring shaders and
closed 2 API gaps (`argmax_dim`, `softmax_dim`). The `level_spacing_ratio`
function was rewired from local implementation to upstream `barracuda::spectral`.

### 6 New Shader Absorptions (S50–S52)

| Shader | Upstream API | Session |
|--------|-------------|---------|
| `xoshiro128ss.wgsl` | `barracuda::ops::prng_xoshiro` | S51 |
| `logsumexp_reduce.wgsl` | `barracuda::ops::LogsumexpWgsl` | S51 |
| `stencil_cooperation.wgsl` | `barracuda::StencilCooperationGpu` | S52 |
| `wright_fisher_step.wgsl` | `barracuda::WrightFisherGpu` | S52 |
| `rk45_adaptive.wgsl` | `barracuda::ops::rk45_adaptive` | S51 |
| `swarm_nn_scores.wgsl` | `barracuda::SwarmNnGpu` | S52 |

### S-03b Resolved: head_split / head_concat Absorbed (S62)

`ToadStool` S60–S61 (`0c998992`) decomposed the fused MHA projection into separate
matmul + head_split/head_concat — precisely mirroring neuralSpring's local workaround.
`head_split.wgsl` and `head_concat.wgsl` are now fully absorbed upstream.
`evolved/mha.rs` rewired to a thin wrapper delegating to `barracuda::ops::mha`.
**21/21 neuralSpring shaders absorbed upstream.**

### Code Quality Hardening

- `gpu_dispatch.rs` refactored into `gpu_dispatch/` module (dispatcher + `cpu_fallback.rs`)
- Population-vs-sample variance convention documented (CPU fallback ÷N vs barracuda ÷(N-1))
- 7 inline `1e-14` guards centralized to `tolerances::ZERO_DETECTION`
- All Clippy pedantic + nursery warnings resolved (0 warnings)
- `cargo doc --no-deps` clean (0 warnings, 146 pages)

### Cross-Spring Benchmark Results (RTX 4070, Vulkan, Release, Feb 24, 2026)

| Op | Origin | Size | Time (µs) |
|----|--------|------|-----------|
| `BatchFitnessGpu` | neuralSpring (S-25) | 1024×64 | 1,337 |
| `PairwiseL2Gpu` | neuralSpring (S-42) | 128×16 | 1,542 |
| `BatchIprGpu` | neuralSpring (S-25) | 32×64 | 2,027 |
| `SpatialPayoffGpu` | neuralSpring (S-25) | 32×32 | 1,450 |
| `PairwiseHammingGpu` | neuralSpring (S-25) | 64×100 | 1,682 |
| `HmmBatchForwardF64` | wetSpring (S-39) | 4s×50t×32b | 2,141 |
| `BatchedEighGpu` | hotSpring (S-39) | 12×12×40 | 6,629 |

**Key insight**: All three Springs' shaders run through the same unified
BarraCUDA API on RTX 4070. A neuralSpring user calling `HmmBatchForwardF64`
(wetSpring origin) or `BatchedEighGpu` (hotSpring origin) sees no difference
from calling `BatchFitnessGpu` (neuralSpring origin). The absorption model
works: evolve locally, validate, hand off, absorb upstream, retire local copy.

### `bench_upstream_vs_local` (Known Limitation)

HillGateGpu f64 causes NVVM compilation failure on RTX 4070, which cascades
to device loss. This is a documented driver limitation. The f32 path works.
Benchmarks for the other 9 kernels show negligible wrapper overhead (0.92–1.16×),
consistent with Session 48 findings.

### Validation Score

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets` (pedantic + nursery) | 0 warnings |
| `cargo doc --no-deps` | 0 warnings (146 pages) |
| `cargo test --lib` | 459 PASS |
| `cargo llvm-cov --lib` | 92.89% line coverage |
| `validate_all` | 141/142 PASS (1 pre-existing logsumexp driver issue) |

### Evolution Timeline Update

```
Feb 24  Session 50: baseCamp biophysical AI interpretability (5 modules, 82/82 PASS)
        Session 51: Code quality evolution — gpu_dispatch refactor, clippy, docs
        ToadStool sync: 16 commits (b41ee5f4 → 9abd6857)
          - 6 shaders absorbed (xoshiro, logsumexp, stencil, wright_fisher, rk45, swarm_nn)
          - argmax_dim() and softmax_dim(axis) API gaps CLOSED
          - level_spacing_ratio rewired to barracuda::spectral
          - barracuda::tolerances created (shared ZERO_DETECTION)
        Session 52: S-17 HillGate f64 pow polyfill fix — 3 validators upgraded (0 SKIP)
        Session 53: Final rewiring — 5 ops delegated to upstream f64 typed ops
          - variance_gpu → VarianceReduceF64 (hotSpring Welford)
          - pearson_correlation_gpu → CorrelationF64 (wetSpring + hotSpring)
          - shannon_entropy_gpu → FusedMapReduceF64 (wetSpring fused)
          - cpu_fallback::pearson → barracuda::stats::pearson_correlation
          - cpu_fallback::chi_squared → barracuda::special::chi_squared_statistic
        Only 2 local shaders remain: head_split + head_concat (MHA S-03b)
```

---

## Session 53 — Final Rewiring: f32 Tensor → f64 Upstream Typed Ops (February 24, 2026)

### Rewiring Summary

Five GPU/CPU operations rewired from local f32 Tensor pipelines to upstream
f64 typed BarraCUDA ops. This completes the cross-spring absorption cycle —
neuralSpring now consumes shaders evolved by all three Springs through
unified upstream APIs.

| Operation | Old Path | New Path | Origin Spring |
|-----------|----------|----------|---------------|
| `variance_gpu` | f32 Tensor (mean→sub→sq→mean, 4 dispatches) | `VarianceReduceF64` (Welford, single f64 shader) | hotSpring |
| `pearson_correlation_gpu` | f32 Tensor (3+ dispatches) | `CorrelationF64` (single f64 shader) | wetSpring + hotSpring |
| `shannon_entropy_gpu` | f32 Tensor (log→mul→sum, 3 dispatches) | `FusedMapReduceF64` (fused f64 map-reduce) | wetSpring |
| `cpu_fallback::pearson` | Local Rust implementation | `barracuda::stats::correlation::pearson_correlation` | wetSpring |
| `cpu_fallback::chi_squared` | Local Rust implementation | `barracuda::special::chi_squared_statistic` | wetSpring |

### Benchmark: Old f32 Tensor → New f64 Upstream (10,000 elements)

**RTX 4070 (Ada Lovelace, NVIDIA proprietary driver)**

| Op | f32 Tensor (µs) | f64 Upstream (µs) | Speedup | Origin |
|----|-----------------|-------------------|---------|--------|
| Variance | 7,018 | 2,316 | **3.03×** | hotSpring Welford |
| Pearson | 3,566 | 3,480 | **1.02×** | wetSpring + hotSpring |
| Entropy | 3,989 | 1,662 | **2.40×** | wetSpring fused |

**TITAN V (Volta, NVK open-source driver)**

| Op | f32 Tensor (µs) | f64 Upstream (µs) | Speedup | Origin |
|----|-----------------|-------------------|---------|--------|
| Variance | 13,333 | 2,937 | **4.54×** | hotSpring Welford |
| Pearson | 5,098 | 15,053 | 0.34× (NVK f64 overhead) | wetSpring + hotSpring |
| Entropy | 5,510 | 3,525 | **1.56×** | wetSpring fused |

**Key observations:**

1. **Variance** benefits enormously — Welford's online algorithm in a single
   f64 dispatch eliminates 4 separate f32 dispatches (upload, mean, subtract,
   square, mean). The win is even larger on TITAN V/NVK (4.54×).

2. **Entropy** benefits from fused map-reduce — a single f64 dispatch that
   computes `-sum(p * ln(p))` eliminates 3 f32 dispatches (log, mul, sum).

3. **Pearson** is dispatch-neutral on RTX 4070 but **regresses on TITAN V/NVK**
   where the f64 correlation shader hits NVK's slower f64 path. This is an
   acceptable trade-off: the precision upgrade (f32→f64) matters more than
   the NVK slowdown for correctness-critical scientific computation.

### Cross-Spring Evolution Benchmark (RTX 4070 + TITAN V)

| Op | Origin | RTX 4070 (µs) | TITAN V (µs) |
|----|--------|---------------|-------------|
| `BatchFitnessGpu` 1024×64 | neuralSpring (S-25) | 1,678 | 2,494 |
| `PairwiseL2Gpu` 128×16 | neuralSpring (S-42) | 2,137 | 2,093 |
| `BatchIprGpu` 32×64 | neuralSpring (S-25) | 1,988 | 1,913 |
| `SpatialPayoffGpu` 32×32 | neuralSpring (S-25) | 1,776 | 2,087 |
| `PairwiseHammingGpu` 64×100 | neuralSpring (S-25) | 1,449 | 1,678 |
| `HmmBatchForwardF64` 4s×50t×32b | wetSpring (S-39) | 1,981 | 5,136 |
| `BatchedEighGpu` 12×12×40 | hotSpring (S-39) | 6,190 | 20,106 |

### Validation: 141/142 PASS

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets` (pedantic + nursery) | 0 warnings |
| `cargo test --lib` | 459 PASS |
| `validate_all` | 141/142 PASS |

Only `validate_barracuda_logsumexp` fails (pre-existing driver issue, S-16).

---

## Sessions 62–64 — S-03b Resolved, `BandwidthTier` Wired, Full Benchmark (Feb 25, 2026)

### S-03b: The Write → Absorb → Lean Cycle Completing

`ToadStool` S60–S61 (`0c998992`) independently decomposed the fused MHA projection
into matmul + head_split + head_concat — precisely the approach neuralSpring evolved
locally when upstream's fused dispatch hung on RTX 4070. This is the textbook
absorption cycle:

```text
neuralSpring evolves workaround (S-03b, evolved/mha.rs, 124 LOC)
     → validated against baselines (10/10 MHA checks PASS)
     → ToadStool absorbs the decomposed approach (0c998992)
     → neuralSpring rewires to upstream (evolved/mha.rs → 18 LOC thin wrapper)
     → 21/21 shaders absorbed, 0 local WGSL remaining
```

### `BandwidthTier` + NVK Allocation Guard (S63–S64)

Wired upstream `BandwidthTier::detect_from_adapter_name()` into the Dispatcher.
On initialization, the tier is detected and logged:
```text
[dispatch] GPU available: NVIDIA GeForce RTX 4070 (DiscreteGpu, Vulkan, f64=Hybrid, pcie=PciE4x16)
```

Added `Dispatcher::check_allocation_safe()` which delegates to upstream
`GpuDriverProfile::check_allocation_safe()` for NVK large-buffer protection.

### S63–S64 Cross-Spring Benchmark (RTX 4070, `--release`)

#### Typed GPU Ops — All Three Springs

| Op | Size | Median (µs) | Origin Spring | When |
|----|------|-------------|---------------|------|
| `BatchFitnessGpu` | 1024×64 | 3,033 | neuralSpring (ML) | S-25 |
| `PairwiseL2Gpu` | 128×16 | 3,154 | neuralSpring (MODES) | S-42 |
| `BatchIprGpu` | 32×64 | 2,364 | neuralSpring (Anderson) | S-25 |
| `SpatialPayoffGpu` | 32×32 | 2,901 | neuralSpring (game theory) | S-25 |
| `PairwiseHammingGpu` | 64×100 | 2,678 | neuralSpring (SATé) | S-25 |
| `HmmBatchForwardF64` | 4s×50t×32b | 3,325 | wetSpring (phylo) | S-39 |
| `BatchedEighGpu` | 12×12×40 | 7,402 | hotSpring (nuclear) | S-39 |

#### Rewire Evolution — f32 Tensor → f64 Upstream (10,000 elements)

| Op | f32 Tensor (µs) | f64 Upstream (µs) | Speedup | Cross-Spring Origin |
|----|-----------------|-------------------|---------|---------------------|
| Variance | 9,949 | 2,847 | **3.49×** | hotSpring Welford → `VarianceReduceF64` |
| Pearson | 4,679 | 3,508 | **1.33×** | wetSpring + hotSpring → `CorrelationF64` |
| Entropy | 6,317 | 2,468 | **2.56×** | wetSpring fused → `FusedMapReduceF64` |

**Key insight**: Variance sees the biggest speedup because Welford's online algorithm
(hotSpring origin) replaces 4 separate f32 dispatches with a single f64 dispatch.
Entropy's fused map-reduce (wetSpring origin) similarly collapses 3 dispatches into 1.
Pearson gains f64 precision with modest speedup.

#### Rewired Dispatcher Throughput (upstream dispatch vs CPU ref)

| Method | Origin | n=128 upstream (µs) | n=128 CPU (µs) | Notes |
|--------|--------|---------------------|----------------|-------|
| `matmul` | hotSpring precision | 2,714 | 325 | GPU at 128²; crossover ~256² |
| `softmax` | hotSpring numerics | 4.7 | 4.7 | Parity — dispatch routes to CPU |
| `gelu` | neuralSpring ML | 19.4 | 15.1 | CPU-routed; GPU wins at 4096+ |
| `mean` | hotSpring reduce | 0.4 | 0.4 | Parity |
| `hmm_forward` | wetSpring bio | 0.5 | 0.5 | Parity — CPU optimal at 32 states |

### Validation: 145/146 + 22/22 + 500 PASS

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets` (pedantic + nursery) | 0 warnings |
| `cargo test --lib` | 500 PASS |
| `validate_all` | 145/146 PASS |
| `validate_cross_spring_evolution` | **22/22 PASS** |

### Evolution Timeline Update

```text
Feb 25  Session 62: ToadStool S62 sync (02207c4a)
          - S-03b RESOLVED: MHA head_split/head_concat absorbed upstream
          - evolved/mha.rs rewired to thin wrapper (124 LOC → 18 LOC)
          - 21/21 neuralSpring WGSL shaders now upstream
          - New upstream: BandwidthTier, ComputeDispatch, NvkLargeBufferLimit,
            Conv2dGpu, SpMM f64, TransE f64, PeakDetectF64, cpu-math feature gate
        Session 63: BandwidthTier + NVK guard wired into Dispatcher
          - BandwidthTier::detect_from_adapter_name() → PciE4x16 (RTX 4070)
          - check_allocation_safe() for NVK large-buffer protection
          - Full benchmark: 3.49× variance, 2.56× entropy, 1.33× Pearson
          - All Springs' ops benchmarked: BatchFitnessGpu, HmmBatchForwardF64,
            BatchedEighGpu running through unified BarraCUDA dispatch
```

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
| Rewire evolution benchmark | `src/bin/bench_rewire_evolution.rs` |
| V32 handoff document | `wateringHole/handoffs/NEURALSPRING_TOADSTOOL_V32_S69_CROSS_SPRING_EVOLUTION_HANDOFF_FEB25_2026.md` |
