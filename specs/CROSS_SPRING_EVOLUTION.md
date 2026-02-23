# Cross-Spring Evolution — Shader & Primitive Provenance

> *"We evolve locally, validate rigorously, hand off cleanly, then lean on upstream."*

This document tracks how three ecoPrimals Springs — **hotSpring**, **wetSpring**,
and **neuralSpring** — contribute shaders and primitives to `ToadStool`/`BarraCUDA`,
creating a shared math engine whose capabilities grow with every absorption cycle.

**ToadStool HEAD**: `6ee71f07` + 2 local fixes pending absorption (Feb 23, 2026)
**Multi-GPU**: RTX 4070 (proprietary) + TITAN V (NVK) — bit-identical across all Springs' shaders

---

## The Absorption Cycle

```text
Spring evolves locally   →  validates against baselines  →  metalForge export
       ↓                                                          ↓
Spring leans on upstream ←  ToadStool absorbs             ←  handoff to ToadStool
```

Each Spring operates independently, evolving workarounds and new capabilities in
response to its domain needs. When a primitive matures (validated, documented,
binding layouts specified), it's handed off to `ToadStool`. After absorption, all
three Springs — and any future Springs — benefit.

---

## Provenance by Spring

### hotSpring — Precision Physics & Lattice QCD

hotSpring's domain is molecular dynamics, lattice gauge theory, and precision
numerics. Its contributions established `BarraCUDA`'s f64 math foundation.

| Contribution | Category | BarraCUDA Location | neuralSpring Uses |
|-------------|----------|-------------------|-------------------|
| `complex_f64.wgsl` | f64 complex arithmetic | `ops::lattice::complex_f64` | Yes (ESN, lattice ops) |
| `su3.wgsl` + SU(3) matrix ops | Lattice QCD gauge field | `ops::lattice::su3` | — |
| `wilson_plaquette_f64.wgsl` | Lattice QCD plaquette | `ops::lattice::plaquette` | — |
| `su3_hmc_force_f64.wgsl` | HMC force term | `ops::lattice::hmc_force_su3` | — |
| `higgs_u1_hmc_f64.wgsl` | Higgs U(1) simulation | `ops::lattice::higgs_u1` | — |
| `dirac_staggered_f64.wgsl` | Staggered Dirac operator | `ops::lattice::dirac` | — |
| `cg_kernels_f64.wgsl` | Conjugate gradient solver | `ops::lattice::cg` | — |
| `batched_eigh_nak_optimized_f64.wgsl` | GPU eigensolve (NAK) | `ops::linalg::eigh_f64` | Available |
| Spectral theory module | Lanczos, Sturm bisection, Anderson, Hofstadter | `spectral::*` | Yes (batch IPR) |
| `esn_reservoir_update.wgsl` / `esn_readout.wgsl` | Echo state networks | `esn_v2` | — |
| `CellListGpu` fix | MD neighbor list | `ops::md::neighbor` | — |
| `SubstrateCapability` enum | Device capability detection | `device::substrate` | Yes (dispatch) |
| Hermite, Laguerre, Broyden polynomials | Special functions | `special::*` | Yes (validation) |
| `weighted_dot_f64` | Precision inner product | `ops::weighted_dot_f64` | Yes (f64 validation) |
| `target` WGSL keyword fix | Driver correctness | All shaders | Yes (all GPU ops) |
| `SHADER_F64` adapter detection | f64 GPU capability | `device::wgpu_device` | Yes (f64 dispatch) |

**Impact on neuralSpring**: hotSpring's precision math (`log_f64`, `exp_f64`,
`weighted_dot_f64`) underpins every f64 validation in neuralSpring. The spectral
module provides the `BatchIprGpu` pipeline used by Anderson localization. The
`SHADER_F64` detection enables neuralSpring's dual-backend (CPU/GPU) tensor validation.

### wetSpring — Bioinformatics & Genomics

wetSpring's domain is metagenomics, phylogenetics, and population genetics.
Its contributions established `BarraCUDA`'s bio-compute layer.

| Contribution | Category | BarraCUDA Location | neuralSpring Uses |
|-------------|----------|-------------------|-------------------|
| `smith_waterman_banded_f64.wgsl` | Sequence alignment | `ops::bio::smith_waterman` | Available |
| `gillespie_ssa_f64.wgsl` | Stochastic simulation | `ops::bio::gillespie` | Available |
| `felsenstein_f64.wgsl` | Phylogenetic likelihood | `ops::bio::felsenstein` | Available |
| `tree_inference_f64.wgsl` | Random forest inference | `ops::bio::tree_inference` | Available |
| `rf_batch_inference.wgsl` | Batch RF inference | `ops::bio::rf_inference` | Available |
| `hmm_forward_f64.wgsl` | Batch HMM (f64) | `ops::bio::hmm` | Available |
| `ani_batch_f64.wgsl` | Average nucleotide identity | `ops::bio::ani` | Available |
| `snp_calling_f64.wgsl` | Variant detection | `ops::bio::snp` | Available |
| `dnds_batch_f64.wgsl` | Molecular evolution | `ops::bio::dnds` | Available |
| `pangenome_classify.wgsl` | Gene family classification | `ops::bio::pangenome` | Available |
| `quality_filter.wgsl` | FASTQ quality trimming | `ops::bio::quality_filter` | Available |
| `dada2_e_step.wgsl` | Amplicon denoising | `ops::bio::dada2` | Available |
| `bray_curtis_f64.wgsl` | Diversity distance | `ops::batch_pair_reduce_f64` | Yes (f64 validation) |
| `log_f64` coefficient fix | Precision math | `shaders/math/math_f64.wgsl` | Yes (all f64 shaders) |
| `FusedMapReduceF64` (Shannon, Simpson) | Diversity indices | `ops::fused_map_reduce_f64` | Yes (f64 validation) |
| `cosine_similarity_f64.wgsl` | Distance metric | `ops::cosine_similarity_f64` | Yes (f64 validation) |
| Ada Lovelace NVVM f64 workaround | Driver compatibility | `device::*` | Yes (RTX 4070 support) |

**Impact on neuralSpring**: wetSpring's `log_f64` coefficient fix improved
precision across all f64 shader operations. The `HmmBatchForwardF64` wrapper
provides the f64 batch counterpart to neuralSpring's f32 HMM forward shader.
The Ada Lovelace workaround enables neuralSpring's GPU validation on RTX 4070.

**Session 43 wetSpring parity (validated from neuralSpring):** `TaxonomyFcGpu`,
`KmerHistogramGpu`, and `UniFracPropagateGpu` are wetSpring-origin APIs now
validated from neuralSpring (`validate_upstream_taxonomy`, `validate_upstream_kmer`,
`validate_upstream_unifrac`). `GillespieGpu` benefits all Springs for stochastic
simulation — validated via `validate_gpu_gillespie` (20/20 PASS, f64 conservation).

### neuralSpring — ML Validation & Evolutionary Computation

neuralSpring's domain is reproducing 25 computational biology papers with Rust
validation against Python baselines. Its contributions established `BarraCUDA`'s
ML and evolutionary computation layer.

| Contribution | Category | BarraCUDA Location | Status |
|-------------|----------|-------------------|--------|
| `eigh_householder_qr` | Precision eigensolve | `ops::linalg::eigh_f64` | **Absorbed** (`77f70b2e`) |
| `hmm_forward_log.wgsl` | HMM forward (f32) | `ops::bio::hmm` / `shaders/ml/` | **Absorbed** |
| `batch_fitness_eval.wgsl` | EA fitness evaluation | `ops::bio::batch_fitness` / `shaders/ml/` | **Absorbed** |
| `rk4_parallel.wgsl` | Parallel ODE integration | `ops::rk_stage` / `shaders/numerical/` | **Absorbed** |
| `pairwise_jaccard.wgsl` | Pangenome distance | `ops::bio::pairwise_jaccard` / `shaders/math/` | **Absorbed** |
| `pairwise_hamming.wgsl` | Alignment distance | `ops::bio::pairwise_hamming` / `shaders/math/` | **Absorbed** |
| `locus_variance.wgsl` | FST / allele freq var | `ops::bio::locus_variance` / `shaders/bio/` | **Absorbed** |
| `spatial_payoff.wgsl` | Game theory stencil | `ops::bio::spatial_payoff` / `shaders/math/` | **Absorbed** |
| `batch_ipr.wgsl` | Spectral localization | `spectral::batch_ipr` / `shaders/spectral/` | **Absorbed** |
| `TensorSession` ML ops | Session API extension | `session::{matmul, relu, gelu, softmax, layer_norm}` | **Absorbed** (S-01/S-11) |
| 4-tier `KernelRouter` | Matmul auto-tuning | `ops::matmul` | **Absorbed** (S-02) |
| `pairwise_l2.wgsl` | MODES novelty | *local* | Pending |
| `multi_obj_fitness.wgsl` | Directed evolution | *local* | Pending |
| `swarm_nn_forward.wgsl` | Swarm NN inference | *local* | Pending |
| `hill_gate.wgsl` | Signal AND gate | *local* | Pending |
| `mean_reduce.wgsl` | Scalar reduction | *local* | Pending |
| `head_split.wgsl` / `head_concat.wgsl` | MHA reshape | *local* | Pending (S-03b) |
| `xoshiro128ss.wgsl` | GPU PRNG | *local* | Pending |

**Impact on other Springs**: neuralSpring's `eigh_householder_qr` replaced BarraCUDA's
Jacobi eigensolver with trillion-fold accuracy improvement at n≥8, benefiting all Springs
that use eigendecomposition. The HMM, pairwise distance, and spatial payoff shaders are
now available to wetSpring for its genomics pipelines and to hotSpring for spectral analysis.

---

## Cross-Spring Dependencies

```text
                    ToadStool / BarraCUDA
                    ┌─────────────────────┐
                    │  ops::bio::*         │ ← wetSpring + neuralSpring
                    │  ops::lattice::*     │ ← hotSpring
                    │  ops::linalg::eigh   │ ← neuralSpring
                    │  spectral::*         │ ← hotSpring + neuralSpring
                    │  session::*          │ ← neuralSpring (S-01/S-11)
                    │  shaders/math/*      │ ← hotSpring (f64) + wetSpring (log fix)
                    │  device::*           │ ← hotSpring (SHADER_F64) + wetSpring (Ada)
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         hotSpring       wetSpring      neuralSpring
       (lattice QCD)   (genomics)     (ML validation)
```

### What neuralSpring leans on from other Springs

| What | Source Spring | How neuralSpring uses it |
|------|-------------|------------------------|
| `log_f64` precision fix | wetSpring | All f64 shader math |
| `SHADER_F64` detection | hotSpring | Dual CPU/GPU tensor validation |
| `BatchIprGpu` pipeline | hotSpring (spectral) | Anderson localization GPU |
| `weighted_dot_f64` | hotSpring | f64 validation checks |
| Ada Lovelace f64 workaround | wetSpring | RTX 4070 GPU support |
| `SubstrateCapability` | hotSpring | Cross-dispatch routing |
| `FusedMapReduceF64` | wetSpring | f64 tensor validation |
| `cosine_similarity_f64` | wetSpring | f64 tensor validation |

---

## Benchmark Results (RTX 4070, Feb 22, 2026)

### GPU Tensor Ops (barracuda `77f70b2e`)

| Op | Median | Notes |
|------|--------|-------|
| ReLU | 7 µs | ElementwiseGpu |
| GELU | 12 µs | WgslGpu |
| Sigmoid | 7 µs | ElementwiseGpu |
| Softmax | 3.7 ms | Multi-pass reduction |
| LayerNorm | 170 µs | WgslGpu (stock) |
| MatMul | 3.6 ms | 4-tier KernelRouter (neuralSpring S-02) |
| Add | 8 µs | ElementwiseGpu |
| MSE Loss | 139 µs | Reduction |
| LogSoftmax | 164 µs | WgslGpu (native) |

### GPU Shader Crossover Points

| Kernel | GPU µs | Rust CPU µs | GPU/Rust |
|--------|--------|-------------|----------|
| Hamming 200×1000 | 2,328 | 7,477 | **3.2×** |
| Jaccard 100×2000 | 1,737 | 8,244 | **4.7×** |
| Batch fitness 50k×64 | 1,842 | — | — |
| Spatial 512² | 2,178 | — | — |
| IPR 2000×256 | 1,688 | — | — |

GPU wins at scale; dispatch overhead (~1.5ms) makes CPU faster for small problems.
`BarraCUDA`'s cross-dispatch (`dispatch_for`) routes automatically based on
empirical crossover points codified in `metalForge/forge/src/dispatch.rs`.

### Pure Rust Math Kernels

| Kernel | Papers | Rust µs |
|--------|--------|---------|
| HMM forward (3×5000) | 016–018 | 84.8 |
| Replicator dynamics (10k) | 019 | 151.3 |
| Commutator ‖[A,B]‖_F (64²) | 022 | 111.2 |
| NK fitness (N=10,K=2, 1k) | 011 | 17.9 |
| Pairwise Hamming (20×500) | 017 | 33.9 |
| Jaccard distance (30×500) | 024 | 141.4 |
| RK4 GRN ODE (2k steps) | 020–021 | 192.7 |
| Multi-obj fitness (100×30×3) | 014 | 3.0 |
| Hill gate (50×50) | 021 | 2.8 |
| Swarm NN (20×50) | 015 | 39.0 |
| **Total** | | **778.2** |

---

## Validation Summary (Post-Rewire, Feb 22, 2026)

All validation binaries pass after rewiring to upstream `77f70b2e`:

| Category | Binaries | Checks | Status |
|----------|----------|--------|--------|
| Lib tests | — | 237 + 9 doc | **PASS** |
| Forge crate | — | 18 | **PASS** |
| eigh accuracy (S-12 delegation) | 1 | 9 | **PASS** |
| Anderson localization | 1 | 8 | **PASS** |
| BarraCUDA linalg ext | 1 | 17 | **PASS** |
| GPU HMM forward (rewired) | 1 | 13 | **PASS** |
| GPU batch fitness (rewired) | 1 | 20 | **PASS** |
| GPU RK4 parallel (rewired) | 1 | 8 | **PASS** |
| GPU pangenome/meta-pop/game/anderson/sate | 5 | 28 | **PASS** |
| GPU modes/directed/swarm/signal | 4 | 39 | **PASS** |
| Cross-dispatch (4 binaries) | 4 | 41 | **PASS** |
| BarraCUDA stats/FFT/logsumexp | 3 | 42 | **PASS** |

---

*Cross-spring evolution tracker — every absorption makes all Springs stronger.*
