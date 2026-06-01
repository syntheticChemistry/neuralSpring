<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring — Foundation Seeding Manifest

**Status**: Thread 5 active (expression + ML_SURROGATES wired), Thread 7 seeded | **Session**: S221 | **Date**: Jun 1, 2026

neuralSpring contributes validated science to two foundation threads:

## Thread 5: Evolutionary Biology / LTEE

neuralSpring reproduces 5 Dolson faculty papers as publishable notebooks with
full Python→Rust→GPU validation chains.

### Validated Datasets

| Paper | Dataset | Result | Tolerance | Source |
|-------|---------|--------|-----------|--------|
| 011 | NK fitness landscapes (K=2,3,4) | CD protocol outperforms naive linear schedule | rel 1e-10 | `control/counterdiabatic/` |
| 012 | MODES toolbox metrics (4 open-endedness metrics) | All four discriminate open-ended vs closed | rel 1e-10 | `control/modes/` |
| 013 | Competitive exclusion + niche differentiation | Validated ecological dynamics | rel 1e-10 | `control/eco_dynamics/` |
| 014 | Lexicase selection (directed evolution) | Diversity preservation + fitness improvement | rel 1e-10 | `control/directed_evolution/` |
| 015 | Heterogeneous swarm controllers | Het > Hom diversity; both improve fitness | rel 1e-10 | `control/swarm_robotics/` |

### Provenance

- Python baselines: `control/{domain}/{domain}.py` (NumPy, validated via `run_all_baselines.sh`)
- Rust validators: `src/bin/validate_*.rs` binaries (CPU parity within 1e-10)
- GPU validators: barraCuda dispatch via WGSL shaders (f64-canonical)
- Notebooks: `notebooks/paper-{id}-*.ipynb` (72/72 checks PASS)
- Provenance registry: `src/provenance/references.rs` (49 records)

## Thread 7: Anderson Mathematics

neuralSpring validates Anderson localization spectral properties across 1D/2D/3D
regimes, connecting to groundSpring's ODE/disorder physics and hotSpring's MD.

### Validated Datasets

| Domain | Result | Tolerance | Source |
|--------|--------|-----------|--------|
| 1D Anderson tight-binding | Lyapunov exponent vs disorder W | abs 1e-6 | `control/anderson_localization/` |
| IPR (Inverse Participation Ratio) | IPR scaling with system size | rel 1e-8 | `src/bin/validate_immunological_anderson.rs` |
| Level spacing statistics | Wigner-Dyson to Poisson transition | abs 1e-4 | `src/bin/validate_immunological_anderson.rs` |
| Anderson transition (3D) | W_c ~ 16.5 critical disorder | rel 5% | `src/bin/validate_immunological_anderson_extended.rs` |

### Evoformer / Protein Folding (cross-thread)

| Domain | Result | Tolerance | Source |
|--------|--------|-----------|--------|
| Evoformer MSA attention | Self-attention + outer product update | rel 1e-6 | `src/bin/validate_alphafold3_pairformer.rs` |
| Structure module IPA | Invariant point attention + backbone | rel 1e-6 | `src/bin/validate_alphafold2_evoformer.rs` |
| Folding health (pLDDT/PAE/pTM) | Confidence metrics validated | abs 1e-4 | `src/bin/validate_alphafold3_confidence.rs` |
| ESN multi-target prediction | ESN surrogate vs physics baselines | rel 1e-3 | `control/wdm/esn_regime_classifier.py` |

### BLAKE3 Provenance

All validated results are checksummed via `validation/CHECKSUMS` (BLAKE3, 15
files) and linked to `src/provenance/experiments.rs` for sweetGrass braid
integration.

## Contribution Path

1. ~~Create `data/targets/thread07_anderson_targets.toml` in foundation~~ **DONE** (S201) — 6 neuralSpring targets added
2. ~~Create `data/sources/thread05_ml_surrogates.toml` in foundation~~ **DONE** (S201) — 15 sources
3. ~~Create `data/targets/thread05_ml_surrogates_targets.toml`~~ **DONE** (S201) — 12 targets
4. Register BLAKE3 hashes via sweetGrass braid + NestGate content pipeline

Foundation now at 7/10 threads with sources (was 5/10). neuralSpring
contributed Thread 5 (new) and expanded Thread 7.

*neuralSpring V165 | Session S209 | AGPL-3.0-or-later*
