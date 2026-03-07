# neuralSpring — baseCamp: Per-Faculty Research Briefings & Cross-Domain Extensions

**Last Updated**: March 7, 2026 (Sessions 61–130)
**Status**: 26 papers + 6 baseCamp sub-theses + WDM surrogates + coralForge (nF-01/02/03 Phase C) + 3 publication experiments, **4100+ checks**, ~97% GPU promotion, **218/218 validate\_all** (240 binaries), 883 lib tests. S130: upstream rewire — ToadStool S130, BarraCUDA `2a6c072`, coralReef Iteration 7. `PrecisionRoutingAdvice` wired, fused GPU regression gated, coralNAK→coralReef rename. V88 handoff. barraCuda: v0.3.3 at `2a6c072` (`../barraCuda/crates/barracuda`, 708 WGSL shaders, wgpu 28, cross-spring provenance)

## Purpose

Per-faculty validation briefings and cross-domain extension proposals.
Each briefing maps: paper → Python baseline → Rust validation → BarraCUDA CPU →
GPU Tensor → metalForge WGSL → pipeline → cross-dispatch → multi-GPU.

Extension proposals identify where neuralSpring's validated primitives can
serve larger fields of study, cross-domain science, and the gen3 baseCamp
sub-theses — now that we have pure GPU execution for ~97% of production math.

## Faculty Summary

| Faculty | Institution | Track | Papers | Checks | Domains |
|---------|------------|-------|--------|--------|---------|
| Emily Dolson | Michigan State | Evolutionary Computation | 5 (011–015) | 50 | NK fitness, MODES, eco dynamics, directed evolution, swarm robotics |
| Kevin Liu | Michigan State | Phylogenetics / HMM | 3 (016–018) | 38 | HMM forward/backward/Viterbi, SATé alignment, introgression detection |
| Chris Waters | Michigan | Microbial Cooperation | 3 (019–021) | 21 | Game theory, regulatory networks, signal integration |
| Ilya Kachkovskiy | Michigan State | Spectral Theory | 2 (022–023) | 16 | Spectral commutativity, Anderson localization |
| R. Anderson / Campbell | Various | Population Genetics | 2 (024–025) | 16 | Pangenome selection, meta-population dynamics |

**Total**: 15 Phase 0++ papers + Paper 026 (Chuna LSTM glucose), 141+ Phase 0++ checks (all PASS at 7 tiers).

## Validation Chain

```
Python baseline (seed=42) → Rust CPU (provenance) → BarraCUDA CPU
  → GPU Tensor (WGSL) → metalForge shaders → GPU pipeline → cross-dispatch
    → multi-GPU (bit-identical RTX 4070 + TITAN V NVK)
      → gpu_dispatch (~97% pure GPU, Phase C: HMM chains, FST, introgression)
        → CPU↔Python parity (39/39 PASS, 1e-10 cross-language)
```

## Briefings

| File | Faculty | Papers |
|------|---------|--------|
| [dolson.md](dolson.md) | Emily Dolson (MSU) | 011–015: Evolutionary computation |
| [liu.md](liu.md) | Kevin Liu (MSU) | 016–018: Phylogenetics / HMM |
| [waters.md](waters.md) | Chris Waters (UMich) | 019–021: Microbial cooperation |
| [kachkovskiy.md](kachkovskiy.md) | Ilya Kachkovskiy (MSU) | 022–023: Spectral theory |
| [anderson.md](anderson.md) | R. Anderson / Campbell | 024–025: Population genetics |

## baseCamp Research Program: Biophysical AI Interpretability

neuralSpring's novel research program applies validated physics and biology
primitives to understanding AI systems as physical systems. Six sub-thesis
proposals, each grounded in 3-6 published papers and using existing validated
primitives.

| File | Sub-Thesis | Domain Cross |
|------|-----------|--------------|
| [extensions.md](extensions.md) | **Program overview** — sub-theses 01-05, priority, reading order | All |
| [sub01_weight_hamiltonians.md](sub01_weight_hamiltonians.md) | Weight matrices as Anderson Hamiltonians | Random matrix theory x DL |
| [sub02_information_propagation.md](sub02_information_propagation.md) | Information flow as wave propagation | Statistical physics x RNNs |
| [sub03_loss_landscapes.md](sub03_loss_landscapes.md) | Loss landscapes as energy landscapes | Chemical physics x Optimization |
| [sub04_neural_pgm.md](sub04_neural_pgm.md) | Neural networks as probabilistic graphical models | Bayesian inference x Interpretability |
| [sub05_multiagent_qs.md](sub05_multiagent_qs.md) | Multi-agent AI coordination as quorum sensing | Microbial ecology x Multi-agent AI |
| [sub06_immunological_anderson.md](sub06_immunological_anderson.md) | Anderson localization in immunological signaling | Immunology x condensed matter x drug repurposing |

---

## Infrastructure Summary

### Upstream Rewiring (Sessions 56–79 — ToadStool S66)

baseCamp functions delegate to upstream BarraCUDA. Session 58 rewired
7 core Dispatcher methods to `barracuda::dispatch::domain_ops` and wired in
`GpuDriverProfile`. Sessions 77–79 added WDM surrogates and completed
cross-spring rewiring to ToadStool S66 APIs.

| Local Function | Upstream | Sub-thesis | Session |
|----------------|----------|-----------|---------|
| `graph_laplacian` | `barracuda::linalg::graph` | Sub-05 | S56 |
| `disordered_laplacian` | `barracuda::linalg::graph` | Sub-05 | S56 |
| `belief_propagation_chain` | `barracuda::linalg::graph` | Sub-04 | S56 |
| `numerical_hessian` | `barracuda::numerical` | Sub-03 | S56 |
| `mat_mul` | `barracuda::dispatch::matmul_dispatch` | All | S58 |
| `frobenius_norm` | `barracuda::dispatch::frobenius_norm_dispatch` | Sub-01 | S58 |
| `transpose` | `barracuda::dispatch::transpose_dispatch` | Sub-01 | S58 |
| `softmax` | `barracuda::dispatch::softmax_dispatch` | All | S58 |
| `l2_distance` | `barracuda::dispatch::l2_distance_dispatch` | Sub-02 | S58 |
| `mean` | `barracuda::dispatch::mean_dispatch` | All | S58 |
| `variance` | `barracuda::dispatch::variance_dispatch` | All | S58 |
| `softmax_row_wise` | `Tensor::softmax_dim(1)` | Sub-04 (PGM) | S73 |
| `fst_single_locus` | `barracuda::ops::bio::fst_variance_decomposition` | Pop genetics | S73 |
| `pairwise_fst_full` | upstream per-locus decomposition | Pop genetics | S73 |
| Viterbi argmax | `Tensor::argmax_dim(0)` | Sub-04 (HMM) | S73 |
| `metrics::mae` | `barracuda::stats::mae` | WDM surrogates | S78 |
| `shannon_entropy` | `barracuda::stats::shannon_from_frequencies` | All | S78 |
| `hill_activation` | `barracuda::stats::hill` | Sub-02, Waters | S78 |
| `hill_repression` | `barracuda::stats::hill` (inverted) | Sub-02, Waters | S78 |
| `modes::l2_distance` | `barracuda::dispatch::l2_distance_dispatch` | Sub-05 | S78 |
| `complexity_metric` | `barracuda::stats::fit_linear` | MODES | S78 |
| `MlpLayer` (EOS surrogate) | `barracuda::nn::SimpleMlp` | WDM surrogates | S121 |
| `MlpLayer` (Transport surrogate) | `barracuda::nn::SimpleMlp` | WDM surrogates | S121 |
| `hmm_viterbi_chain_gpu` (per-step f32) | `barracuda::ops::bio::hmm_viterbi` (f64 ComputeDispatch) | HMM/Phylo | S121 |

### Hardware Validation

All baseCamp experiments inherit neuralSpring's validated multi-tier pipeline:

1. **BarraCUDA CPU**: Pure Rust, machine-precision agreement with Python
2. **BarraCUDA GPU**: Tensor API, f32-f64 agreement < 1e-3 across all domains
3. **metalForge mixed**: Same answer on CPU, GPU, NPU — multi-substrate dispatch
4. **df64 core streaming** (S88): f64 buffer I/O → df64 compute on FP32 cores → f64
   output. Achieves ~14-digit (fp48) precision on consumer GPUs. Arithmetic ops:
   3.6e-8 to 5.6e-7 max diff. Transcendental ops: 1.7e-4 to 3.4e-4 max diff.
   `Fp64Strategy::Hybrid` auto-detected on RTX 4070 (1:64 FP64:FP32 ratio).

### Performance Summary

| Metric | Value |
|--------|-------|
| Pure Rust vs Python | 83.6× geomean (11 domains); fastest 1104× (multi-obj) |
| CPU↔Python parity | 39/39 PASS (1e-10 cross-language) |
| Dispatch overhead | ≤1.04× for 9/10 ops (transparent) |
| GPU vs Python | Up to 104x (transformer medium) |
| GPU crossover | ~1.5 ms dispatch overhead |
| Multi-GPU | Bit-identical (RTX 4070 + TITAN V NVK) — **384/384 Titan V (S82)** |
| Fused pipeline | 46-78x over per-op dispatch |
| GPU math coverage | ~97% of production operations (Phase C complete) |

### Open Data

All papers use computationally generated data from published parameters.
No external datasets, no API dependencies, no proprietary sources.
See `specs/DATA_PROVENANCE.md` for full inventory.
