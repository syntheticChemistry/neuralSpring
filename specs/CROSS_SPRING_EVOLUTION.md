# Cross-Spring Evolution — Shader & Primitive Provenance

> *"We evolve locally, validate rigorously, hand off cleanly, then lean on upstream."*

This document tracks how three ecoPrimals Springs — **hotSpring**, **wetSpring**,
and **neuralSpring** — contribute shaders and primitives to `ToadStool`/`BarraCUDA`,
creating a shared math engine whose capabilities grow with every absorption cycle.

**ToadStool HEAD**: `e96576ee` (Sessions 59–68 sync — 42 upstream rewires, S-03b fully resolved, 21/21 shaders absorbed + 15 coralForge df64 shaders (df64 core streaming), 93.5% coverage, 611 tests, pure GPU all-domains 10/10 PASS, cross-system dispatch 46/46 PASS, cross-spring evolution 52/52 PASS, WDM surrogates validated, Feb 26, 2026)
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
| `pairwise_l2.wgsl` | MODES novelty | `ops::bio::pairwise_l2` / `shaders/math/` | **Absorbed** (closed-form pair decode) |
| `multi_obj_fitness.wgsl` | Directed evolution | `ops::bio::multi_obj_fitness` / `shaders/bio/` | **Absorbed** (Bessel correction) |
| `swarm_nn_forward.wgsl` | Swarm NN inference | `ops::bio::swarm_nn` / `shaders/bio/` | **Absorbed** (generic MLP dims) |
| `hill_gate.wgsl` | Signal AND gate | `ops::bio::hill_gate` / `shaders/bio/` | **Absorbed** (mode 0/1 generalization) |
| ~~`mean_reduce.wgsl`~~ | Scalar reduction | `pipeline::ReduceScalarPipeline` | **Absorbed** (S59 cleanup) |
| `head_split.wgsl` / `head_concat.wgsl` | MHA reshape | `barracuda::ops::mha` | **Absorbed** (S-03b resolved `0c998992`) |
| ~~`xoshiro128ss.wgsl`~~ | GPU PRNG | `ops::prng_xoshiro` | **Absorbed** (S52) |
| `empirical_spectral_density` | Eigenvalue histogram | `stats::empirical_spectral_density` | **Absorbed** (S54, rewired S59) |
| `marchenko_pastur_bounds` | MP spectral bounds | `stats::marchenko_pastur_bounds` | **Absorbed** (S54, rewired S59) |
| `effective_rank` | Entropy-based rank | `linalg::effective_rank` | **Absorbed** (S54, rewired S59) |

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
| `FusedMapReduceF64` | wetSpring | **Production entropy** (2.4× faster, S-53 rewire) |
| `cosine_similarity_f64` | wetSpring | f64 tensor validation |
| `VarianceReduceF64` | hotSpring | **Production variance** (3–4.5× faster, S-53 rewire) |
| `CorrelationF64` | wetSpring + hotSpring | **Production Pearson** (f64 precision, S-53 rewire) |
| `chi_squared_statistic` | wetSpring | **CPU fallback chi²** (S-53 rewire) |
| `pearson_correlation` | wetSpring | **CPU fallback Pearson** (S-53 rewire) |
| `pow_f64` polyfill (S-17) | hotSpring + wetSpring | `HillGate` f64 works on all drivers. **RESOLVED** upstream (`c82c23d1` S58: `patch_transcendentals_in_code` covers `pow`) |

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

## Validation Summary (Post-Rewire S69, Feb 25, 2026)

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets` (pedantic + nursery) | 0 warnings |
| `cargo test --lib` | 604 PASS |
| `cargo test --test integration` | 9 PASS |
| `validate_all` | 147/148 PASS |
| `validate_cross_spring_evolution` | **39/39 PASS** |

Only `validate_barracuda_logsumexp` fails (pre-existing upstream buffer size mismatch).

The cross-spring evolution validator covers all 30 rewired functions: 9 Dispatcher
methods (S58: 7 + S59: gelu, hmm_forward) plus 3 library delegates (ESD, MP bounds,
effective rank) plus `boltzmann_sampling` (S68) plus 4 S73 Tensor API rewires
(Viterbi argmax_dim, softmax_row_wise, fst_single_locus, pairwise_fst_full) plus driver profile checks.

Additionally, 6 validator binaries now source WGSL from upstream barracuda constants
instead of local `include_str!`, further eliminating local shader copies.

---

## Session 53 — Final f64 Typed Op Rewiring (February 24, 2026)

### Rewiring: f32 Tensor → f64 Upstream Typed Ops

Five operations rewired from local f32 Tensor pipelines to upstream f64 typed
BarraCUDA ops, completing the cross-spring absorption cycle.

| Operation | Old Path (f32 Tensor) | New Path (f64 Upstream) | Origin |
|-----------|----------------------|------------------------|--------|
| `variance_gpu` | mean→sub→sq→mean (4 dispatches) | `VarianceReduceF64` (Welford, 1 dispatch) | hotSpring |
| `pearson_correlation_gpu` | dx/dy→mul→sum (3+ dispatches) | `CorrelationF64` (1 dispatch) | wetSpring + hotSpring |
| `shannon_entropy_gpu` | log→mul→sum (3 dispatches) | `FusedMapReduceF64` (fused, 1 dispatch) | wetSpring |
| `cpu_fallback::pearson` | Local Rust impl | `barracuda::stats::pearson_correlation` | wetSpring |
| `cpu_fallback::chi_squared` | Local Rust impl | `barracuda::special::chi_squared_statistic` | wetSpring |

### Benchmark: Old f32 Tensor → New f64 Upstream (10,000 elements)

**RTX 4070 (Ada Lovelace)**

| Op | f32 Tensor (µs) | f64 Upstream (µs) | Speedup | Origin |
|----|-----------------|-------------------|---------|--------|
| Variance | 7,018 | 2,316 | **3.03×** | hotSpring Welford |
| Pearson | 3,566 | 3,480 | **1.02×** | wetSpring + hotSpring |
| Entropy | 3,989 | 1,662 | **2.40×** | wetSpring fused |

**TITAN V (NVK)**

| Op | f32 Tensor (µs) | f64 Upstream (µs) | Speedup | Origin |
|----|-----------------|-------------------|---------|--------|
| Variance | 13,333 | 2,937 | **4.54×** | hotSpring Welford |
| Pearson | 5,098 | 15,053 | 0.34× (NVK f64 overhead) | wetSpring + hotSpring |
| Entropy | 5,510 | 3,525 | **1.56×** | wetSpring fused |

### Cross-Spring Evolution Benchmark (RTX 4070 + TITAN V)

| Op | Origin | RTX 4070 (µs) | TITAN V (µs) |
|----|--------|---------------|-------------|
| `BatchFitnessGpu` 1024×64 | neuralSpring | 1,678 | 2,494 |
| `PairwiseL2Gpu` 128×16 | neuralSpring | 2,137 | 2,093 |
| `BatchIprGpu` 32×64 | neuralSpring | 1,988 | 1,913 |
| `SpatialPayoffGpu` 32×32 | neuralSpring | 1,776 | 2,087 |
| `PairwiseHammingGpu` 64×100 | neuralSpring | 1,449 | 1,678 |
| `HmmBatchForwardF64` 4s×50t×32b | wetSpring | 1,981 | 5,136 |
| `BatchedEighGpu` 12×12×40 | hotSpring | 6,190 | 20,106 |

### Session 58 — Upstream Dispatch Rewiring + GpuDriverProfile

neuralSpring now delegates 7 core Dispatcher methods to upstream `domain_ops`
and uses hotSpring-evolved `GpuDriverProfile` for hardware-adaptive f64 strategy.

| Dispatcher Method | Upstream domain_ops | Cross-Spring Origin |
|-------------------|--------------------|---------------------|
| `mat_mul` | `matmul_dispatch` | hotSpring (tile kernels) |
| `frobenius_norm` | `frobenius_norm_dispatch` | hotSpring (reduction) |
| `transpose` | `transpose_dispatch` | neuralSpring (spectral) |
| `softmax` | `softmax_dispatch` | neuralSpring (ML) |
| `l2_distance` | `l2_distance_dispatch` | neuralSpring (MODES) |
| `mean` | `mean_dispatch` | hotSpring (reduction) |
| `variance` | `variance_dispatch` | hotSpring (Welford) |

GpuDriverProfile on RTX 4070: Ada, NvidiaPtxas, Throttled FP64 → Hybrid
(df64 f32-pair bulk, native f64 reductions). pow(f64) workaround: yes.

### Session 59 — S54-S59 Absorption Cycle Completed

neuralSpring rewired 5 local implementations to upstream BarraCUDA APIs that were
originally contributed *by* neuralSpring and absorbed in S54/S52:

| Function | neuralSpring Module | Upstream API | Absorbed In |
|----------|-------------------|-------------|-------------|
| `empirical_spectral_density` | `weight_spectral` | `barracuda::stats::empirical_spectral_density` | S54 (M-011) |
| `marchenko_pastur_bounds` | `weight_spectral` | `barracuda::stats::marchenko_pastur_bounds` | S54 (M-012) |
| `effective_rank` | `neural_pgm` | `barracuda::linalg::effective_rank` | S54 (H-009) |
| `gelu` (new dispatch) | `gpu_dispatch/dispatch_ops` | `barracuda::dispatch::gelu_dispatch` | S52 |
| `hmm_forward_step` (new dispatch) | `gpu_dispatch/dispatch_ops` | `barracuda::dispatch::hmm_forward_dispatch` | S52 |

Additionally, 3 dead WGSL re-exports (`WGSL_BATCH_FITNESS_EVAL`, `WGSL_RK4_PARALLEL`,
`WGSL_MEAN_REDUCE`) were removed from `evolved/mod.rs` — all callers already use
upstream typed APIs.

Total rewired functions: **25** (S68: 17; S59: 21; S73: +4 Tensor API rewires).

### Sessions 60–61 — Benchmark Validation & Cross-Spring Narrative (Feb 25, 2026)

Updated `validate_cross_spring_evolution` to cover all S59 rewires (22 checks total),
ran full benchmark suite demonstrating cross-spring performance evolution.

#### Rewire Evolution Benchmark (RTX 4070, `--release`)

f32 Tensor paths vs f64 upstream typed ops (10,000 elements):

| Op | f32 Tensor (µs) | f64 Upstream (µs) | Speedup | Cross-Spring Origin |
|----|-----------------|-------------------|---------|---------------------|
| Variance | 5,773 | 2,350 | **2.46×** | hotSpring Welford → `VarianceReduceF64` |
| Pearson | 3,254 | 2,938 | **1.11×** | wetSpring + hotSpring → `CorrelationF64` |
| Entropy | 3,191 | 1,232 | **2.59×** | wetSpring fused → `FusedMapReduceF64` |

#### Cross-Spring Typed GPU Op Benchmark (RTX 4070, `--release`)

| Op | Size | Median (µs) | Origin Spring | Absorption |
|----|------|-------------|---------------|------------|
| `BatchFitnessGpu` | 1024×64 | 1,274 | neuralSpring (ML) | S-25 |
| `PairwiseL2Gpu` | 128×16 | 1,846 | neuralSpring (MODES) | S-42 |
| `BatchIprGpu` | 32×64 | 2,541 | neuralSpring (Anderson) | S-25 |
| `SpatialPayoffGpu` | 32×32 | 1,518 | neuralSpring (game theory) | S-25 |
| `PairwiseHammingGpu` | 64×100 | 1,430 | neuralSpring (SATé) | S-25 |
| `HmmBatchForwardF64` | 4s×50t×32b | 1,743 | wetSpring (phylo) | S-39 |
| `BatchedEighGpu` | 12×12×40 | 5,355 | hotSpring (nuclear) | S-39 |

#### Rewired Dispatcher Throughput (RTX 4070, upstream dispatch vs CPU)

| Method | n | Upstream (µs) | CPU (µs) | Notes |
|--------|---|---------------|----------|-------|
| matmul | 16×16 | 1.3 | 1.2 | CPU faster at small n (expected) |
| matmul | 128×128 | 2,740 | 302 | GPU overhead; crossover at ~256² |
| softmax | 256 | 1.1 | 1.1 | Parity at small n |
| gelu | 256 | 2.2 | 2.2 | Parity — dispatch routes to CPU |
| gelu | 1024 | 9.8 | 9.7 | Still CPU-routed |
| mean | 256 | 0.1 | 0.1 | Parity |
| hmm_fwd | 32 states | 0.5 | 0.5 | Parity — CPU optimal at this scale |

**Key insight**: For the workloads neuralSpring validates (n ≤ 4096), the upstream
dispatch correctly routes to CPU, maintaining zero overhead. GPU benefits appear at
production scales handled by the typed GPU ops (e.g., `BatchFitnessGpu` at 50k genomes,
`PairwiseHammingGpu` at 200+ sequences). The cross-spring dispatch architecture is
working as designed — each Spring benefits from the others' GPU primitives while the
dispatch layer transparently selects the optimal backend.

#### Cross-Spring Evolution Flow (Empirically Validated)

```text
hotSpring precision ──→ df64_core, pow_f64, Welford variance, Lanczos
                        ↓                                      ↓
                     BarraCUDA ←── ToadStool absorption ←── metalForge
                        ↓
wetSpring bio ─────→ HMM forward, fused map-reduce, log_f64 fix, dN/dS
                        ↓
                     BarraCUDA (now has precision + bio)
                        ↓
neuralSpring ML ───→ eigh, batch_fitness, pairwise_l2, spectral density
                        ↓
                     BarraCUDA (now has precision + bio + ML)
                        ↓
               ╔═══════════════════════════╗
               ║  All Springs lean on the  ║
               ║  shared math engine:      ║
               ║  • 694 WGSL shaders       ║
               ║  • 30 rewired functions    ║
               ║  • 6 validator shader      ║
               ║    sources → upstream      ║
               ║  • 117+ upstream APIs      ║
               ╚═══════════════════════════╝
```

### Sessions 62–64 — S-03b Resolved + `BandwidthTier` + Full Benchmark (Feb 25, 2026)

#### S-03b Resolution: Write → Absorb → Lean Complete

The MHA projection hang (S-03b) was resolved upstream in `ToadStool` S60–S61
(`0c998992`). ToadStool independently decomposed the fused projection into matmul +
head_split + head_concat — the exact approach neuralSpring evolved locally.

Result: `evolved/mha.rs` rewired from 124 LOC workaround to 18 LOC thin wrapper.
**21/21 neuralSpring WGSL shaders now absorbed upstream.** Zero local WGSL remaining.

#### `BandwidthTier` + NVK Guard Wired (S63–S64)

Upstream `BandwidthTier::detect_from_adapter_name()` wired into `Dispatcher`. Logs:
```text
[dispatch] GPU available: NVIDIA GeForce RTX 4070 (DiscreteGpu, Vulkan, f64=Hybrid, pcie=PciE4x16)
```

`Dispatcher::check_allocation_safe()` delegates to `GpuDriverProfile` for NVK
large-buffer protection (1.2 GB limit on TITAN V).

#### S63–S64 Benchmark: Rewire Evolution (RTX 4070, 10,000 elements)

| Op | f32 Tensor (µs) | f64 Upstream (µs) | Speedup | Cross-Spring Origin |
|----|-----------------|-------------------|---------|---------------------|
| Variance | 9,949 | 2,847 | **3.49×** | hotSpring Welford |
| Pearson | 4,679 | 3,508 | **1.33×** | wetSpring + hotSpring |
| Entropy | 6,317 | 2,468 | **2.56×** | wetSpring fused map-reduce |

#### S63–S64 Benchmark: Cross-Spring GPU Ops (RTX 4070, `--release`)

| Op | Size | Median (µs) | Origin Spring |
|----|------|-------------|---------------|
| `BatchFitnessGpu` | 1024×64 | 3,033 | neuralSpring (S-25) |
| `PairwiseL2Gpu` | 128×16 | 3,154 | neuralSpring (S-42) |
| `BatchIprGpu` | 32×64 | 2,364 | neuralSpring (S-25) |
| `SpatialPayoffGpu` | 32×32 | 2,901 | neuralSpring (S-25) |
| `PairwiseHammingGpu` | 64×100 | 2,678 | neuralSpring (S-25) |
| `HmmBatchForwardF64` | 4s×50t×32b | 3,325 | wetSpring (S-39) |
| `BatchedEighGpu` | 12×12×40 | 7,402 | hotSpring (S-39) |

#### S63–S64 Validation

| Gate | Result |
|------|--------|
| `validate_cross_spring_evolution` | 39/39 PASS |
| `validate_all` | 145/146 PASS |
| `cargo test --lib` | 500 PASS |
| `cargo clippy --all-targets` | 0 warnings |

---

### Session 69 — Validator Shader Rewiring + Modern Benchmarks (Feb 25, 2026)

#### Validator Shader Source Rewiring

Six validator binaries rewired from local `include_str!` to upstream barracuda
shader constants. The shader content is identical (same absorbed WGSL), but the
source-of-truth now lives in barracuda rather than the local `metalForge/shaders/`
directory — completing the "Lean" phase for shader sources.

| Validator | Shader | Old Source | New Source |
|-----------|--------|-----------|-----------|
| `validate_gpu_rk4` | `rk4_parallel.wgsl` | `include_str!` | `barracuda::ops::rk_stage::WGSL_RK4_PARALLEL` |
| `validate_gpu_rk45` | `rk45_adaptive.wgsl` | `include_str!` | `barracuda::ops::rk45_adaptive::WGSL_RK45_ADAPTIVE` |
| `validate_gpu_stateful_pipeline` | `rk4_parallel.wgsl` | `include_str!` | `barracuda::ops::rk_stage::WGSL_RK4_PARALLEL` |
| `validate_gpu_pure_workload` | `batch_fitness_eval.wgsl` | `include_str!` | `barracuda::ops::bio::batch_fitness::WGSL_BATCH_FITNESS_EVAL` |
| `validate_gpu_logsumexp` | `logsumexp_reduce.wgsl` | `include_str!` | `barracuda::ops::logsumexp::LogSumExp::WGSL_LOGSUMEXP_REDUCE` |
| `validate_gpu_pipeline_swarm` | `swarm_nn_scores.wgsl` | `include_str!` | `barracuda::ops::bio::swarm_nn::WGSL_SWARM_NN_SCORES` |

**Not rewired (blocked)**:
- `mean_reduce.wgsl` — barracuda uses internally but no public `WGSL_MEAN_REDUCE` constant
- `head_split.wgsl` / `head_concat.wgsl` — no upstream equivalent (still local-only)
- `bench_upstream_vs_local.rs` — intentionally uses `include_str!` to compare local vs upstream dispatch

#### Remaining `include_str!` Inventory

| File | Shader | Reason |
|------|--------|--------|
| `validate_gpu_pure_workload.rs` | `mean_reduce.wgsl` | No public upstream constant |
| `validate_mha_gpu.rs` | `head_split.wgsl`, `head_concat.wgsl` | No upstream equivalent |
| `bench_upstream_vs_local.rs` | 10 shaders | Intentional: benchmarks local vs upstream dispatch |

#### Upstream vs Local Shader Benchmark (RTX 4070, `--release`)

All 10 neuralSpring-origin shaders benchmarked — upstream wrappers within negligible
overhead of local manual dispatch:

| Kernel | Origin Paper | Local (µs) | Upstream (µs) | Ratio |
|--------|-------------|-----------|--------------|-------|
| BatchFitness 10k×32 | 011-015 | 1,840 | 2,060 | 1.12× ~ |
| Hamming 200×500 | 017 (SATé) | 1,807 | 1,947 | 1.08× ≈ |
| Jaccard 100×500 | 024 (Pangenome) | 1,972 | 1,849 | 0.94× ≈ |
| LocusVariance 50×500 | 025 (MetaPop) | 2,035 | 2,043 | 1.00× ≈ |
| SpatialPayoff 256² | 019 (GameTheory) | 1,903 | 1,890 | 0.99× ≈ |
| BatchIPR 1k×256 | 022-023 (Anderson) | 1,909 | 2,301 | 1.21× ~ |
| HillGate 100² | 021 (Signal) | 2,101 | 2,003 | 0.95× ≈ |
| MultiObjFitness 5k×4 | 014 (DirEvo) | 1,978 | 1,943 | 0.98× ≈ |
| PairwiseL2 200×50 | 012 (MODES) | 2,031 | 1,940 | 0.96× ≈ |
| SwarmNN 500×20 | 015 (Swarm) | 1,990 | 1,999 | 1.00× ≈ |

≈ = negligible overhead (< 5%), ~ = minor overhead (5–25%)

#### Cross-Spring Evolution Benchmark (RTX 4070, `--release`, S69)

| Op | Size | Median (µs) | Origin Spring | Absorption |
|----|------|-------------|---------------|------------|
| `BatchFitnessGpu` | 1024×64 | 2,000 | neuralSpring (ML) | S-25 |
| `PairwiseL2Gpu` | 128×16 | 1,994 | neuralSpring (MODES) | S-42 |
| `BatchIprGpu` | 32×64 | 2,064 | neuralSpring (Anderson) | S-25 |
| `SpatialPayoffGpu` | 32×32 | 2,102 | neuralSpring (game theory) | S-25 |
| `PairwiseHammingGpu` | 64×100 | 2,027 | neuralSpring (SATé) | S-25 |
| `HmmBatchForwardF64` | 4s×50t×32b | 2,085 | wetSpring (phylo) | S-39 |
| `BatchedEighGpu` | 12×12×40 | 7,497 | hotSpring (nuclear) | S-39 |

#### Cross-Spring Provenance Summary (S69)

Three Springs feed ToadStool/BarraCUDA, each bringing domain expertise:

**hotSpring** (precision physics): ~25+ shaders/modules — df64_core, SU(3) gauge,
CG solver, Lanczos eigensolver, Hermite/Laguerre polynomials, `SHADER_F64` detection,
`SubstrateCapability`, `GpuDriverProfile`, `VarianceReduceF64` (Welford), ESN reservoir.
neuralSpring benefits from hotSpring's precision math in every f64 validation.

**wetSpring** (bioinformatics): ~15+ shaders/modules — Smith-Waterman, Gillespie SSA,
Felsenstein likelihood, HMM forward/backward, dN/dS, pangenome classify, DADA2,
Bray-Curtis distance, `FusedMapReduceF64` (Shannon/Simpson), `log_f64` coefficient fix,
Ada Lovelace NVVM workaround. neuralSpring uses wetSpring's HMM f64 and bio distance ops.

**neuralSpring** (ML validation): ~15+ shaders/modules — pairwise ops (Hamming, Jaccard,
L2), batch fitness, spatial payoff, batch IPR, hill gate, multi-obj fitness, swarm NN,
`eigh_householder_qr` (trillion-fold accuracy vs Jacobi), `TensorSession`, 4-tier
`KernelRouter`, `empirical_spectral_density`, `marchenko_pastur_bounds`, `effective_rank`.
hotSpring and wetSpring benefit from neuralSpring's eigensolve and spectral analysis.

**Collaborative**: `pow_f64` polyfill (hotSpring + wetSpring), `CrankNicolson`
(airSpring + wetSpring + hotSpring), `FusedMapReduceF64` (wetSpring entropy + hotSpring
convergence norms), `GemmF64` cached extension (wetSpring 60× taxonomy speedup).

#### S69 Validation

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets` | 0 warnings |
| `cargo test --lib` | 604 PASS |
| `cargo test --test integration` | 9 PASS |
| `validate_all` | 147/148 PASS |
| `validate_cross_spring_evolution` | 39/39 PASS |
| `bench_upstream_vs_local` | 10/10 ≈ or ~ (zero ⚠) |

---

### Session 73 — Cross-Spring Rewiring: Upstream Tensor APIs + FST (Feb 26, 2026)

Four operations rewired to upstream Tensor APIs and BarraCUDA bio FST decomposition,
completing the absorption cycle for Viterbi, softmax, and F-statistics.

| Rewire | Source File | Before | After | Origin Spring |
|--------|------------|--------|-------|---------------|
| Viterbi argmax_dim | `gpu_ops/bio.rs` | CPU loop over scores_flat | `Tensor::argmax_dim(0)` + `to_vec_u32()` | neuralSpring request → ToadStool S60 |
| softmax_row_wise | `gpu_dispatch/dispatch_ops.rs` | Manual per-row loop | `Tensor::softmax_dim(1)` | neuralSpring V20 → ToadStool S60 |
| fst_single_locus | `meta_population.rs` | θ-only `pairwise_fst` | `fst_variance_decomposition` → (θ, f_is, f_it) | wetSpring S53 → BarraCUDA bio |
| pairwise_fst_full | `meta_population.rs` | θ-only multi-locus | Per-locus upstream → averaged F-statistics | wetSpring S53 → BarraCUDA bio |

**New tolerances**: `DISPATCH_F32_ROUNDTRIP` (1e-6), `DISPATCH_VITERBI_F32` (1e-5)

**Cross-spring evolution validator**: 39/39 PASS

---

### Session 74 — Pure GPU All-Domains + Cross-System Dispatch (Feb 26, 2026)

Three new binaries close the pure GPU and cross-system milestones:

| Binary | Checks | What It Proves |
|--------|--------|----------------|
| `validate_gpu_pure_workload_all` | 10/10 PASS | All 15 Phase 0++ domains via typed BarraCUDA GPU ops (scalar-only readback) |
| `validate_cross_system_dispatch` | 46/46 PASS | Full metalForge stack: hardware discovery → domain heuristics → CPU↔GPU parity → transfer cost → NPU routing → crossover sweep |
| `bench_evolution_tiers` | 8 domains | CPU→GPU portability characterization |

**Cross-spring provenance in pure GPU ops:**
- `BatchIprGpu` from hotSpring spectral primitives
- `SpatialPayoffGpu` from wetSpring game theory stencil
- `HmmBatchForwardF64` from wetSpring phylogenetics
- `PairwiseJaccardGpu` from wetSpring/neuralSpring pangenomics
- All feeding ToadStool's unified GPU sovereign pipeline

**f32/f64 precision boundary:** f64 for log-space accumulation (HMM, fitness, locus variance); f32 for domain ops (IPR, L2, Hamming, Jaccard, spatial payoff).

**GPU dispatch overhead:** ~186µs per `queue.submit()` (structural floor). CPU→GPU crossover at ~1946µs compute.

#### S74 Validation

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets` | 0 warnings |
| `cargo test --lib` | 604 PASS |
| `cargo test --test integration` | 9 PASS |
| `validate_all` | 150/150 PASS |
| `validate_gpu_pure_workload_all` | 10/10 PASS |
| `validate_cross_system_dispatch` | 46/46 PASS |
| `validate_cross_spring_evolution` | 39/39 PASS |

---

### Session 75 — ToadStool S60–S65 Upstream Sync (Feb 26, 2026)

Full sync to ToadStool commits S60–S65 (4 commits, 234 files, ~23K lines).

| Rewire | Module | Upstream |
|--------|--------|----------|
| `r_squared` | `metrics.rs` | `barracuda::stats::r_squared` |
| `rmse` | `metrics.rs`, `deeponet.rs` | `barracuda::stats::rmse` |
| `nse` | `metrics.rs` | `barracuda::stats::nash_sutcliffe` |
| `branch_trunk_dot` | `deeponet.rs` | `barracuda::stats::dot` |
| `l2_relative_error` | `deeponet.rs` | `barracuda::stats::l2_norm` |
| `shannon_entropy_from_counts` | `primitives.rs` | `barracuda::stats::shannon` |
| dot product | `neural_pgm.rs`, `counterdiabatic.rs`, `meta_population.rs` | `barracuda::stats::dot` |

**Cross-spring provenance of new stats module:**
- `rmse`, `mbe`, `nash_sutcliffe`, `r_squared`, `index_of_agreement`, `hit_rate` — airSpring/groundSpring hydrology
- `shannon`, `simpson`, `chao1`, `pielou_evenness`, `bray_curtis`, `rarefaction_curve` — wetSpring biodiversity
- `DiversityFusionGpu` — wetSpring → GPU fused Shannon+Simpson+Pielou
- 8 lattice shaders (SU(3), PRNG PCG) — hotSpring precision physics

**Validators fixed:** logsumexp f32→f64, 3× RK4 shader re-import.

**New benchmark:** `bench_cross_spring_evolution` (15/15 PASS) traces provenance
across airSpring stats, wetSpring bio/GPU diversity, hotSpring precision.

#### S75 Validation

| Gate | Result |
|------|--------|
| `cargo clippy --lib` | 0 warnings |
| `cargo test --lib` | 604 PASS |
| `cargo test -p neural-spring-forge --lib` | 43 PASS |
| `validate_all` | **150/150 PASS** |
| `bench_cross_spring_evolution` | 15/15 PASS |
| Total upstream rewires | **30 functions + 6 shader sources** |

---

### Session 76 — Modern BarraCUDA Rewiring + Benchmark Validation (Feb 26, 2026)

Deep rewiring pass to delegate more local implementations to upstream `BarraCUDA`
primitives, followed by full benchmark sweep on RTX 4070.

#### New Rewires

| Rewire | Module | Upstream | Cross-Spring Origin |
|--------|--------|----------|---------------------|
| `matrix_correlation` | `meta_population.rs` | `barracuda::stats::pearson_correlation` | airSpring/groundSpring S64 |
| `thermal_diversity_correlation` | `meta_population.rs` | `barracuda::stats::pearson_correlation` | airSpring/groundSpring S64 |

#### Benchmark Results (RTX 4070, Release)

**Cross-Spring Evolution Benchmark** (`bench_cross_spring_evolution`):

| Metric | Source | Time (µs/iter) |
|--------|--------|-----------------|
| RMSE | airSpring → `barracuda::stats` | 4.1 |
| R² | airSpring → `barracuda::stats` | 12.4 |
| NSE | airSpring → `barracuda::stats` | 12.5 |
| Index of Agreement | airSpring → `barracuda::stats` | 14.0 |
| dot | shared → `barracuda::stats` | 4.2 |
| l2_norm | shared → `barracuda::stats` | 4.1 |
| Shannon | wetSpring → `barracuda::stats` | 1.8 |
| Simpson | wetSpring → `barracuda::stats` | 0.7 |
| Chao1 | wetSpring → `barracuda::stats` | 0.2 |
| alpha_diversity | wetSpring → `barracuda::stats` | 4.8 |
| Bray-Curtis | wetSpring → `barracuda::stats` | 0.1 |
| DiversityFusion CPU | wetSpring → ToadStool | 61.1 |
| DiversityFusion GPU | wetSpring → ToadStool → GPU | 3569.1 |
| Pearson r | hotSpring + wetSpring → `barracuda::stats` | 26.1 |

**Upstream vs Local GPU Dispatch** (`bench_upstream_vs_local`):

| Kernel | Origin | Local µs | Upstream µs | Ratio |
|--------|--------|----------|-------------|-------|
| BatchFitness 10000×32 | neuralSpring 011-015 | 1624.1 | 1617.5 | 1.00× |
| Hamming 200×500 | neuralSpring SATé | 1780.5 | 2032.0 | 1.14× |
| Jaccard 100×500 | neuralSpring Pangenome | 2266.4 | 1918.0 | 0.85× |
| LocusVariance 50×500 | neuralSpring MetaPop | 1816.5 | 1914.1 | 1.05× |
| SpatialPayoff 256×256 | neuralSpring GameTheory | 1885.9 | 1933.0 | 1.02× |
| BatchIPR 1000×256 | neuralSpring Anderson | 1891.1 | 1723.6 | 0.91× |
| HillGate 100×100 | neuralSpring Signal | 1862.6 | 1646.5 | 0.88× |
| MultiObjFitness 5000×4 | neuralSpring DirEvo | 1597.4 | 1777.6 | 1.11× |
| PairwiseL2 200×50 | neuralSpring MODES | 1768.3 | 1719.6 | 0.97× |
| SwarmNN 500×20 | neuralSpring Swarm | 1626.8 | 1681.1 | 1.03× |

All 10 kernels show negligible overhead (0.85–1.14×), confirming `BarraCUDA`
wrappers add no meaningful cost vs raw `metalForge` dispatch.

**Rewire Evolution GPU** (`bench_rewire_evolution`):

| Kernel | f32 Tensor µs | f64 Evolved µs | Speedup | Origin |
|--------|---------------|-----------------|---------|--------|
| Variance (10000) | 8220.6 | 2569.7 | 3.20× | hotSpring Welford |
| Pearson (10000) | 4361.3 | 3213.6 | 1.36× | wetSpring + hotSpring |
| Shannon (10000) | 4216.2 | 1886.2 | 2.24× | wetSpring fused |

Cross-spring evolved shaders outperform naïve f32 Tensor paths by 1.4–3.2×
through domain-specific algorithmic optimizations (Welford online variance,
fused map-reduce, combined correlation).

#### S76 Validation

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --workspace -- -D warnings` | 0 warnings |
| `cargo test --workspace` | 604 lib + 43 forge + 9 integration PASS |
| `validate_all` | **150/150 PASS** |
| `validate_cross_spring_evolution` | **52/52 PASS** |
| `bench_cross_spring_evolution` | **28/28 PASS** |
| Total upstream rewires | **42 upstream rewires** |

---

*Cross-spring evolution tracker — every absorption makes all Springs stronger. S79: complete.*
