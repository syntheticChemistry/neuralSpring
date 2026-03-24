# neuralSpring — Python Control Baselines

Python implementations that serve as ground truth for Rust validation.
Every hardcoded expected value in `src/bin/validate_*.rs` traces back to
a specific run of one of these scripts.

## Environment

```
Python 3.10.12, PyTorch 2.9.0+cu128, NumPy 2.2.6, SciPy 1.15.3
Pinned baseline commit: f9ad0268917a335dce2b1175ea0d77add271b25b
Baseline date: 2026-02-16
```

## Running

Install dependencies and run all baselines:

```bash
pip install -r control/requirements.txt
bash scripts/run_all_baselines.sh
```

Individual experiment:

```bash
python3 control/surrogate/surrogate_validation.py
```

## Structure

| Directory | Experiment | Paper/Source | Checks |
|-----------|-----------|-------------|--------|
| `surrogate/` | Exp 001: MLP vs RBF | FAO-56 ET₀ | 11 |
| `transformer/` | Exp 002: Self-attention | NumPy vs PyTorch | 18 |
| `sequence/` | Exp 003: LSTM/GRU weather | ERA5 / synthetic | 5 |
| `transfer/` | Exp 004: Domain adaptation | ERA5 / synthetic | 6 |
| `isomorphic/` | Exp 005: Cross-domain catalog | Analytical | 8 |
| `pinn/` | Study 001: PINN Burgers | Raissi et al. 2019 | 6 |
| `deeponet/` | Study 002: DeepONet | Lu et al. 2021 | 5 |
| `lenet/` | Study 003: LeNet-5 MNIST | LeCun et al. 1998 | 5 |
| `lstm_weather/` | Study 004: LSTM ERA5 | Real ERA5 data | 5 |
| `quantized/` | Study 005: INT8/INT4 | Quantization methods | 6 |
| `counterdiabatic/` | Paper 011: CD evolution | Iram, Dolson et al. 2020 | 11 |
| `modes/` | Paper 012: MODES toolbox | Dolson et al. 2019 | 9 |
| `eco_dynamics/` | Paper 013: Ecological EA | Dolson & Ofria 2018 | 7 |
| `directed_evolution/` | Paper 014: Directed evolution | Dolson et al. 2022 | 8 |
| `swarm_robotics/` | Paper 015: Swarm controllers | Foreback et al. 2025 | 11 |
| `hmm_phylo/` | Paper 016: HMM phylogenetics | Liu et al. 2014 | 10 |
| `sate_alignment/` | Paper 017: SATé MSA | Liu et al. 2009 | 8 |
| `introgression/` | Paper 018: Introgression | Liu et al. 2015 | 8 |
| `game_theory/` | Paper 019: Game theory & QS | Bruger & Waters 2018 | 8 |
| `regulatory_network/` | Paper 020: Regulatory network | Mhatre et al. 2020 | 7 |
| `signal_integration/` | Paper 021: Signal integration | Srivastava et al. 2011 | 8 |
| `spectral_commutativity/` | Paper 022: C*-algebra distance | Kachkovskiy & Safarov 2016 | 8 |
| `anderson_localization/` | Paper 023: Anderson localization | Bourgain & Kachkovskiy 2018 | 8 |
| `pangenome_selection/` | Paper 024: Pangenome selection | Liu et al. (genomics) | 8 |
| `meta_population/` | Paper 025: Meta-population dynamics | Liu et al. (population genetics) | 8 |
| `training_trajectory/` | Exp-050: Training trajectory spectral analysis | baseCamp Paper A | 11 |
| `hessian_eigenanalysis/` | Exp-052: Hessian eigenanalysis at minima | baseCamp Paper D | 8 |
| `anderson_multiagent/` | Exp-053: Anderson multi-agent coordination | baseCamp Paper C | 11 |
| `coral_forge/` | nF-01/02/03: coralForge structure prediction | AlphaFold2/3 | 106 |
| `shared/` | Open-Meteo ERA5 fetch/cache | CC BY 4.0 | — |
| `ml_inference/` | Benchmark + baseline generation | Scaling analysis | — |
| `immunological_anderson/` | nS-06: Immunological Anderson | Gonzales, Fajgenbaum, McCandless | 48 |

**Total: 417/417 PASS** (48 Phase 0 + 31 Phase 0+ + 127 Phase 0++ + 30 pub exp + 27 WDM + 106 coralForge + 48 nS-06)

## Data Sources

| Source | License | Used By |
|--------|---------|---------|
| ERA5 via Open-Meteo | CC BY 4.0 | sequence, lstm_weather, transfer, quantized |
| MNIST via torchvision | CC BY-SA 3.0 | lenet |
| Synthetic (seed=42) | N/A | All Phase 0++ papers |
| FAO-56 ET₀ (Allen 1998) | Public | surrogate |

## Provenance Protocol

Every baseline must be reproducible. The canonical provenance record
includes five fields, all captured in `src/provenance/`:

| Field | Value |
|-------|-------|
| **Commit** | `f9ad0268917a335dce2b1175ea0d77add271b25b` |
| **Date** | 2026-02-16 |
| **Hardware** | Eastgate (i9-12900K, RTX 4070 12 GB, Pop!_OS 22.04) |
| **Environment** | Python 3.10.12, PyTorch 2.9.0+cu128, NumPy 2.2.6, SciPy 1.15.3 |
| **Command** | `python3 control/<subdir>/<script>.py` |

When regenerating baselines, update `src/provenance/` with the new
commit hash, date, and verify the Rust validators still pass.

## Determinism

All scripts use `seed=42` for reproducibility. Stochastic experiments
(training-based) have relaxed validation thresholds documented in
`src/tolerances/`. Deterministic experiments should produce bitwise-identical
results across runs.

## License

AGPL-3.0-or-later — see repository root `LICENSE`.
