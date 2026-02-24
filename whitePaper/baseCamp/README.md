# neuralSpring — baseCamp: Per-Faculty Research Briefings & Cross-Domain Extensions

**Last Updated**: February 24, 2026 (Session 50)
**Status**: 25 papers + 5 baseCamp sub-theses, 1900+ checks, ~90% GPU promotion, zero debt

## Purpose

Per-faculty validation briefings and cross-domain extension proposals.
Each briefing maps: paper → Python baseline → Rust validation → BarraCUDA CPU →
GPU Tensor → metalForge WGSL → pipeline → cross-dispatch → multi-GPU.

Extension proposals identify where neuralSpring's validated primitives can
serve larger fields of study, cross-domain science, and the gen3 baseCamp
sub-theses — now that we have pure GPU execution for ~90% of production math.

## Faculty Summary

| Faculty | Institution | Track | Papers | Checks | Domains |
|---------|------------|-------|--------|--------|---------|
| Emily Dolson | Michigan State | Evolutionary Computation | 5 (011–015) | 50 | NK fitness, MODES, eco dynamics, directed evolution, swarm robotics |
| Kevin Liu | Michigan State | Phylogenetics / HMM | 3 (016–018) | 38 | HMM forward/backward/Viterbi, SATé alignment, introgression detection |
| Chris Waters | Michigan | Microbial Cooperation | 3 (019–021) | 21 | Game theory, regulatory networks, signal integration |
| Ilya Kachkovskiy | Michigan State | Spectral Theory | 2 (022–023) | 16 | Spectral commutativity, Anderson localization |
| R. Anderson / Campbell | Various | Population Genetics | 2 (024–025) | 16 | Pangenome selection, meta-population dynamics |

**Total**: 15 papers, 141 Phase 0++ checks (all PASS at 7 tiers).

## Validation Chain

```
Python baseline (seed=42) → Rust CPU (provenance) → BarraCUDA CPU
  → GPU Tensor (WGSL) → metalForge shaders → GPU pipeline → cross-dispatch
    → multi-GPU (bit-identical RTX 4070 + TITAN V NVK)
      → gpu_dispatch (~90% pure GPU)
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
primitives to understanding AI systems as physical systems. Five sub-thesis
proposals, each grounded in 3 published papers and using existing validated
primitives.

| File | Sub-Thesis | Domain Cross |
|------|-----------|--------------|
| [extensions.md](extensions.md) | **Program overview** — all 5 sub-theses, priority, reading order | All |
| [sub01_weight_hamiltonians.md](sub01_weight_hamiltonians.md) | Weight matrices as Anderson Hamiltonians | Random matrix theory x DL |
| [sub02_information_propagation.md](sub02_information_propagation.md) | Information flow as wave propagation | Statistical physics x RNNs |
| [sub03_loss_landscapes.md](sub03_loss_landscapes.md) | Loss landscapes as energy landscapes | Chemical physics x Optimization |
| [sub04_neural_pgm.md](sub04_neural_pgm.md) | Neural networks as probabilistic graphical models | Bayesian inference x Interpretability |
| [sub05_multiagent_qs.md](sub05_multiagent_qs.md) | Multi-agent AI coordination as quorum sensing | Microbial ecology x Multi-agent AI |

---

## Infrastructure Summary

### Three-Tier Hardware Validation

All baseCamp experiments inherit neuralSpring's validated three-tier pipeline:

1. **BarraCUDA CPU**: Pure Rust, machine-precision agreement with Python
2. **BarraCUDA GPU**: Tensor API, f32-f64 agreement < 1e-3 across all domains
3. **metalForge mixed**: Same answer on CPU, GPU, NPU — multi-substrate dispatch

### Performance Summary

| Metric | Value |
|--------|-------|
| Pure Rust vs Python | 178.5x faster (11 kernels) |
| GPU vs Python | Up to 104x (transformer medium) |
| GPU crossover | ~1.5 ms dispatch overhead |
| Multi-GPU | Bit-identical (RTX 4070 + TITAN V NVK) |
| Fused pipeline | 46-78x over per-op dispatch |
| GPU math coverage | ~90% of production operations |

### Open Data

All papers use computationally generated data from published parameters.
No external datasets, no API dependencies, no proprietary sources.
See `specs/DATA_PROVENANCE.md` for full inventory.
