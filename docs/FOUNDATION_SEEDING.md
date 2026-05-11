<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring — Foundation Seeding Manifest

**Status**: Ready for contribution | **Session**: S198 | **Date**: May 11, 2026

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
- Rust validators: `src/validate_*.rs` binaries (CPU parity within 1e-10)
- GPU validators: barraCuda dispatch via WGSL shaders (f64-canonical)
- Notebooks: `notebooks/paper-{id}-*.ipynb` (72/72 checks PASS)
- Provenance registry: `src/provenance/references.rs` (49 records)

## Thread 7: Anderson Mathematics

neuralSpring validates Anderson localization spectral properties across 1D/2D/3D
regimes, connecting to groundSpring's ODE/disorder physics and hotSpring's MD.

### Validated Datasets

| Domain | Result | Tolerance | Source |
|--------|--------|-----------|--------|
| 1D Anderson tight-binding | Lyapunov exponent vs disorder W | abs 1e-6 | `control/spectral_analysis/` |
| IPR (Inverse Participation Ratio) | IPR scaling with system size | rel 1e-8 | `src/validate_spectral.rs` |
| Level spacing statistics | Wigner-Dyson to Poisson transition | abs 1e-4 | `src/validate_spectral.rs` |
| Anderson transition (3D) | W_c ~ 16.5 critical disorder | rel 5% | `src/validate_anderson.rs` |

### Evoformer / Protein Folding (cross-thread)

| Domain | Result | Tolerance | Source |
|--------|--------|-----------|--------|
| Evoformer MSA attention | Self-attention + outer product update | rel 1e-6 | `src/validate_alphafold3.rs` |
| Structure module IPA | Invariant point attention + backbone | rel 1e-6 | `src/validate_alphafold3.rs` |
| Folding health (pLDDT/PAE/pTM) | Confidence metrics validated | abs 1e-4 | `src/validate_alphafold3.rs` |
| ESN multi-target prediction | ESN surrogate vs physics baselines | rel 1e-3 | `control/wdm/esn_regime_classifier.py` |

### BLAKE3 Provenance

All validated results are checksummed via `validation/CHECKSUMS` (BLAKE3, 15
files) and linked to `src/provenance/experiments.rs` for sweetGrass braid
integration.

## Contribution Path

1. Create `data/targets/thread07_anderson_targets.toml` in foundation with
   neuralSpring's spectral validation results (IPR, localization length,
   level spacing statistics)
2. Create `data/sources/thread05_ltee.toml` in foundation with Dolson's
   5 paper references (already implemented as notebooks)
3. Link `data/targets/thread05_ltee_targets.toml` with evolutionary
   dynamics validation results
4. Register BLAKE3 hashes via sweetGrass braid + NestGate content pipeline

*neuralSpring V147 | Session S198 | AGPL-3.0-or-later*
