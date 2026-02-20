# neuralSpring Specifications

**Last Updated**: February 20, 2026
**Status**: Phase 0/0+/0++ complete (190/190 Python) + Phase 1 complete (409/409 Rust: 167 native + 242 BarraCUDA)
**Domain**: ML primitives, transfer learning, surrogates, isomorphic patterns, scholarly reproduction

---

## Quick Status

| Metric | Value |
|--------|-------|
| Phase 0 (Synthetic) | 48/48 PASS — surrogate, transformer, LSTM, transfer, isomorphic catalog |
| Phase 0+ (Scholarly) | 31/31 PASS — PINN Burgers, DeepONet, LeNet-5, LSTM ERA5, quantized inference |
| Phase 0++ (Papers) | 111/111 PASS — 13 papers across Dolson, Liu, Waters, Kachkovskiy |
| Phase 1a (native Rust) | 167/167 PASS — 16 validation binaries |
| Phase 1b (BarraCUDA) | 242/242 PASS — stats, linalg, special, optimize, precision, tensor (84), tensor_f64 (35), quantized, linalg_ext, ml_inference (13) |
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
| [PAPER_REVIEW_QUEUE.md](PAPER_REVIEW_QUEUE.md) | **Complete** | 23/23 papers reproduced — now tracking GPU promotion priority |
| [BARRACUDA_REQUIREMENTS.md](BARRACUDA_REQUIREMENTS.md) | Active | GPU kernel requirements and gap analysis |
| [EVOLUTION_MAPPING.md](EVOLUTION_MAPPING.md) | Active | Python → Rust → GPU module mapping (Tier A/B/C) |
| [DATA_PROVENANCE.md](DATA_PROVENANCE.md) | Active | All dataset sources, accession numbers, licenses |

### Existing Documentation (in parent directories)

| Document | Location | Description |
|----------|----------|-------------|
| CONTROL_EXPERIMENT_STATUS.md | `../` | All 10 experiments with detailed results |
| whitePaper/STUDY.md | `../whitePaper/` | Full study with Phase 0 + Phase 0+ results |
| whitePaper/METHODOLOGY.md | `../whitePaper/` | Validation framework for all experiments |

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
