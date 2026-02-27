# neuralSpring Specifications

**Last Updated**: February 27, 2026 (Sessions 44–88+ — Phase 4 WGSL shader validation, ToadStool streaming pipeline, NUCLEUS atomics, publication experiments, barracuda evolution audit)
**Status**: Phase 5h+ — 263/263 Python + 2710+ Rust+GPU = **3034+ total checks**, ~97% GPU, 39/39 CPU↔Python parity, 83.6× speedup, pure GPU 10/10 PASS, cross-system 46/46 PASS, cross-spring 52/52 PASS, Phase 4 shaders 22/22, streaming pipeline 28/28, 175 binaries, 174/175 validate\_all
**Domain**: ML primitives, transfer learning, surrogates, isomorphic patterns, scholarly reproduction

---

## Quick Status

| Metric | Value |
|--------|-------|
| Phase 0 (Synthetic) | 48/48 PASS — surrogate, transformer, LSTM, transfer, isomorphic catalog |
| Phase 0+ (Scholarly) | 31/31 PASS — PINN Burgers, DeepONet, LeNet-5, LSTM ERA5, quantized inference |
| Phase 0++ (Papers) | 127/127 PASS — 15 papers across Dolson, Liu, Waters, Kachkovskiy, Anderson |
| Rust native validation | 668 lib + 43 forge + 9 integration PASS — 175 binaries, 40 modules + gpu_ops/ + gpu_dispatch |
| BarraCUDA CPU (bC) | 24/25 papers (96%), 203 checks | ALL GREEN |
| BarraCUDA GPU Tensor (gT) | 23/25 papers (92%), 98+ checks | ALL GREEN |
| metalForge WGSL (mF) | 15/25 papers, 17 shaders, 108 checks | ALL PASS |
| GPU Pipeline (gP) | 15/25 papers, 94 checks | ALL PASS |
| Cross-dispatch (xD) | 15/15 Phase 0++ papers, 49 checks | ALL GREEN |
| Code quality | fmt + clippy (pedantic+nursery) + doc: zero warnings |
| Isomorphism Theorem | 6 primitives explain ALL neural architectures. BarraCUDA covers all 6 |
| Faculty (evolution) | Dolson (CSE, MSU) — counterdiabatic evolution, MODES, directed evolution |
| Faculty (genomics) | Liu (CMSE, MSU) — HMM, phylogenetics, introgression |
| Faculty (biology) | Waters (MMG, MSU) — game theory, regulatory networks, signal integration |
| Faculty (math/spectral) | Kachkovskiy (Math, MSU) — spectral commutativity, Anderson localization |
| Faculty (population genetics) | R. Anderson (Carleton) — pangenome selection, meta-population dynamics |

---

## Specifications

### Validation & Reproduction

| Spec | Status | Description |
|------|--------|-------------|
| [PAPER_REVIEW_QUEUE.md](PAPER_REVIEW_QUEUE.md) | **Complete** | 25/25 papers reproduced — 7-tier validation matrix |
| [BARRACUDA_REQUIREMENTS.md](BARRACUDA_REQUIREMENTS.md) | Active | GPU kernel requirements and gap analysis |
| [EVOLUTION_MAPPING.md](EVOLUTION_MAPPING.md) | Active | Python → Rust → GPU module mapping (Tier A/B/C) |
| [DATA_PROVENANCE.md](DATA_PROVENANCE.md) | Active | All dataset sources, accession numbers, licenses |
| [BARRACUDA_USAGE.md](BARRACUDA_USAGE.md) | Active | BarraCUDA usage audit and evolution path |

### Shader Evolution & Hardware

| Spec | Status | Description |
|------|--------|-------------|
| [PURE_GPU_ROADMAP.md](PURE_GPU_ROADMAP.md) | **Active** | Pure GPU roadmap — Phase A+B+C complete (44 ops), ~97% GPU coverage |
| [TOADSTOOL_HANDOFF.md](TOADSTOOL_HANDOFF.md) | Active | 17 shortcomings — **ALL RESOLVED** upstream; 42 upstream rewires (S81); V53 handoff |
| [BENCHMARK_ANALYSIS.md](BENCHMARK_ANALYSIS.md) | Active | Python vs BarraCUDA CPU vs GPU 3-way benchmark |
| [CROSS_SPRING_EVOLUTION.md](CROSS_SPRING_EVOLUTION.md) | Active | Cross-spring shader/primitive provenance |

### Related Documentation

| Document | Location | Description |
|----------|----------|-------------|
| CONTROL_EXPERIMENT_STATUS.md | `../` | All experiments + phases with detailed results |
| EVOLUTION_READINESS.md | `../` | Tier A/B/C shader absorption readiness |
| whitePaper/STUDY.md | `../whitePaper/` | Full study results |
| whitePaper/BARRACUDA_EVOLUTION.md | `../whitePaper/` | Shader evolution narrative |
| whitePaper/METHODOLOGY.md | `../whitePaper/` | Validation framework |
| metalForge/CROSS_SYSTEM_DISPATCH.md | `../metalForge/` | GPU → CPU → NPU dispatch strategy |
| metalForge/shaders/ABSORPTION_TRACKER.md | `../metalForge/` | Shader lifecycle tracker |
| wateringHole/handoffs/ | `../wateringHole/` | V55 ToadStool handoff (current, Session 88+) |

---

## Scope

### neuralSpring IS:
- **ML primitive validation** — proving each neural operation is correct from scratch
- **Scholarly reproduction** — 25 published papers reproduced in Python/PyTorch
- **Isomorphic pattern discovery** — the shared primitives across all architectures
- **BarraCUDA ML roadmap** — identifying which GPU kernels cover which ML workloads
- **Transfer learning framework** — domain adaptation across climates, physics, biology

### neuralSpring IS NOT:
- Domain-specific science (airSpring, wetSpring, hotSpring, groundSpring)
- GPU implementation (ToadStool/BarraCUDA — neuralSpring validates, ToadStool implements)
- Production ML deployment (Squirrel primal handles inference coordination)

### Faculty Coverage:
- **Dolson** (MSU CS): Counterdiabatic evolution, open-ended evolution, directed evolution, swarm robotics (Papers 011–015)
- **Liu** (MSU CSE): HMM inference, phylogenetic alignment, introgression detection (Papers 016–018)
- **Waters** (MSU Micro): Game theory, regulatory networks, signal integration (Papers 019–021)
- **Kachkovskiy** (MSU Math): Spectral commutativity, Anderson localization (Papers 022–023)
- **R. Anderson** (Carleton): Pangenome selection, meta-population dynamics (Papers 024–025)

---

## The Isomorphism Theorem

All neural architectures decompose into 6 fundamental primitives:

| # | Primitive | FLOPs | BarraCUDA Module |
|---|-----------|-------|-----------------|
| 1 | GEMM (matrix multiply) | 60-90% | `Tensor::matmul`, 4-tier KernelRouter |
| 2 | Attention (scaled dot-product) | 10-30% | `Tensor::attention` |
| 3 | Normalization (LN/BN/RMS) | 1-5% | `Tensor::layer_norm_wgsl` |
| 4 | Nonlinearity (ReLU/GELU/SiLU) | 1-5% | `Tensor::relu/gelu/silu` (90/90 PASS) |
| 5 | Reduction (sum/mean/max) | 1-5% | `ReduceScalarPipeline`, `VarianceReduceF64` |
| 6 | Gating (sigmoid × value) | 5-30% | `hill_gate.wgsl`, sigmoid gating |

Optimizing these 6 operations in WGSL serves language, protein, vision, physics,
and time series simultaneously. All 6 validated across 25 papers from 5 disciplines.

---

## Reading Order

**New to neuralSpring** (20 min):
1. This README (5 min)
2. `../whitePaper/README.md` — overview and key results (10 min)
3. PAPER_REVIEW_QUEUE.md — full 7-tier validation matrix (5 min)

**Deep dive** (2 hours):
`../whitePaper/STUDY.md` → `../CONTROL_EXPERIMENT_STATUS.md` → BARRACUDA_REQUIREMENTS.md

**Ecosystem connection**:
`../../wateringHole/` — inter-primal standards and handoffs

---

## License

**AGPL-3.0-or-later** — GNU Affero General Public License v3.0

All neuralSpring code, data, and documentation are aggressively open science.
See `../LICENSE` for full text.
