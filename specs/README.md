# neuralSpring Specifications

**Last Updated**: February 21, 2026
**Status**: Phase 5b active — 206/206 Python + 1100+ Rust+GPU = **1300+ total checks**
**Domain**: ML primitives, transfer learning, surrogates, isomorphic patterns, scholarly reproduction

---

## Quick Status

| Metric | Value |
|--------|-------|
| Phase 0 (Synthetic) | 48/48 PASS — surrogate, transformer, LSTM, transfer, isomorphic catalog |
| Phase 0+ (Scholarly) | 31/31 PASS — PINN Burgers, DeepONet, LeNet-5, LSTM ERA5, quantized inference |
| Phase 0++ (Papers) | 127/127 PASS — 15 papers across Dolson, Liu, Waters, Kachkovskiy, Anderson |
| Phase 1a (native Rust) | 183/183 PASS — 21 validation binaries (incl. pinn, deeponet, sequence) |
| Phase 1b (BarraCUDA) | 268/268 PASS — stats, linalg, special, optimize, precision, tensor (86), tensor_f64 (35), quantized, linalg_ext, ml_inference (13), FFT (24) |
| Phase 2 (CPU ports) | 170/170 PASS — 17 modules |
| Phase 3c (GPU shaders) | 108/108 PASS — 16 WGSL shaders |
| Phase 3d (cross-dispatch) | 41/41 PASS |
| Phase 4 (pipelines+PRNG+MHA+eigh) | PASS — GPU pipelines, PRNG, MHA, eigendecomposition |
| Phase 5a (GPU Tensor) | 16/16 PASS — spectral (10) + eco (6) |
| Phase 5b (upstream fixes) | Active — GELU fix, S-13 pool sync, S-14 Naive matmul |
| Isomorphism Theorem | 6 primitives explain ALL neural architectures. BarraCUDA covers all 6 |
| BarraCUDA primitive coverage | GEMM, Attention, Norm, Conv2d, LSTM, Autograd, Q-GEMV + stats, linalg, special, optimize |
| Faculty (evolution) | Dolson (CSE, MSU) — counterdiabatic evolution, MODES |
| Faculty (genomics) | Liu (CMSE, MSU) — HMM, phylogenetics |
| Faculty (biology) | Waters (MMG, MSU) — game theory, cooperation |
| Faculty (physics) | Bazavov (CMSE + Physics, MSU) — parallel algorithms |

---

## Specifications

### Validation & Reproduction

| Spec | Status | Description |
|------|--------|-------------|
| [PAPER_REVIEW_QUEUE.md](PAPER_REVIEW_QUEUE.md) | **Complete** | 25/25 papers reproduced — now tracking GPU promotion priority |
| [BARRACUDA_REQUIREMENTS.md](BARRACUDA_REQUIREMENTS.md) | Active | GPU kernel requirements and gap analysis |
| [EVOLUTION_MAPPING.md](EVOLUTION_MAPPING.md) | Active | Python → Rust → GPU module mapping (Tier A/B/C) |
| [DATA_PROVENANCE.md](DATA_PROVENANCE.md) | Active | All dataset sources, accession numbers, licenses |

### Shader Evolution & Hardware

| Spec | Status | Description |
|------|--------|-------------|
| [TOADSTOOL_HANDOFF.md](TOADSTOOL_HANDOFF.md) | Active | 11 BarraCUDA shortcomings + metalForge shader evolutions |
| [BENCHMARK_ANALYSIS.md](BENCHMARK_ANALYSIS.md) | Active | Python vs BarraCUDA CPU vs GPU 3-way benchmark |
| `metalForge/shaders/ABSORPTION_TRACKER.md` | Active | Shader evolution lifecycle tracker (hotSpring pattern) |

### Existing Documentation (in parent directories)

| Document | Location | Description |
|----------|----------|-------------|
| CONTROL_EXPERIMENT_STATUS.md | `../` | All experiments + phases with detailed results |
| whitePaper/STUDY.md | `../whitePaper/` | Full study with Phase 0 + Phase 0+ results |
| whitePaper/BARRACUDA_EVOLUTION.md | `../whitePaper/` | Shader evolution narrative through Phase 3 |
| whitePaper/METHODOLOGY.md | `../whitePaper/` | Validation framework for all experiments |
| BARRACUDA_USAGE.md | `specs/` | Comprehensive BarraCUDA usage audit and evolution path |
| metalForge/CROSS_SYSTEM_DISPATCH.md | `../metalForge/` | GPU → CPU → NPU dispatch strategy |
| metalForge/shaders/ABSORPTION_TRACKER.md | `../metalForge/` | Shader lifecycle tracker |
| metalForge/gpu/nvidia/HARDWARE.md | `../metalForge/` | RTX 4070 hardware characterization |

---

## Scope

### neuralSpring IS:
- **ML primitive validation** — proving each neural operation is correct from scratch
- **Scholarly reproduction** — published ML papers reproduced in Python/PyTorch
- **Isomorphic pattern discovery** — finding the shared primitives across all architectures
- **BarraCUDA ML roadmap** — identifying which GPU kernels cover which ML workloads
- **Transfer learning framework** — domain adaptation across climates, physics, biology

### neuralSpring IS NOT:
- Domain-specific science (airSpring, wetSpring, hotSpring, groundSpring)
- GPU implementation (ToadStool/BarraCUDA — neuralSpring validates, ToadStool implements)
- Production ML deployment (Squirrel primal handles inference coordination)

### neuralSpring EXTENDS TO (via faculty):
- **Dolson**: Counterdiabatic evolution, open-ended evolution metrics, directed evolution
- **Liu**: HMM inference, phylogenetic sequence models, comparative genomics ML
- **Waters**: Game-theoretic optimization, cooperation dynamics, regulatory networks
- **Bazavov**: Parallel algorithms for lattice computation, GPU compute patterns

### neuralSpring PROVIDES TO:
- **biomeOS**: PathwayLearner uses validated ML primitives for capability routing
- **Squirrel**: Sovereign AI inference uses validated quantized models
- **NUCLEUS**: Deployment optimization uses isomorphic kernel sharing
- **All springs**: Validated surrogates, transfer learning, uncertainty quantification

---

## The Isomorphism Theorem

All neural architectures decompose into 6 fundamental primitives:

| # | Primitive | FLOPs | BarraCUDA Shader |
|---|-----------|-------|-----------------|
| 1 | GEMM (matrix multiply) | 60-90% | `gemm_f64.wgsl` |
| 2 | Attention (scaled dot-product) | 10-30% | `attention.wgsl` |
| 3 | Normalization (LN/BN/RMS) | 1-5% | `layer_norm.wgsl`, `batch_norm.wgsl`, `rmsnorm.wgsl` |
| 4 | Nonlinearity (ReLU/GELU/SiLU) | 1-5% | `nn::ReLU` + activation shaders |
| 5 | Reduction (sum/mean/max) | 1-5% | `FusedMapReduceF64` |
| 6 | Gating (sigmoid x value) | 5-30% | `lstm_cell.wgsl` |

Optimizing these 6 operations in WGSL serves language, protein, vision, physics, and time series simultaneously.

---

## Reading Order

**New to neuralSpring** (20 min):
1. This README (5 min)
2. `../whitePaper/README.md` — overview and key results (10 min)
3. PAPER_REVIEW_QUEUE.md — what's next (5 min)

**Deep dive** (2 hours):
`../whitePaper/STUDY.md` → `../CONTROL_EXPERIMENT_STATUS.md` → BARRACUDA_REQUIREMENTS.md

**Ecosystem connection**:
`../../whitePaper/gen3/data/FACULTY_SPRING_PROFILES.md` — full faculty-to-spring mapping

---

## License

**AGPL-3.0** — GNU Affero General Public License v3.0

All neuralSpring code, data, and documentation are aggressively open science. See `../LICENSE` for full text. Any derivative work, including network-accessible services using neuralSpring code, must publish source under the same license.
