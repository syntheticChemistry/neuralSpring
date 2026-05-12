<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# LTEE B1 — Barrick 2009 Mutation Accumulation

**Paper**: Barrick et al. (2009). Genome evolution and adaptation in a
long-term experiment with *Escherichia coli*. *Nature* 461, 1243–1247.

**Strain**: Ara-1 | **Generations**: 2,000–20,000 (6 time points)

## Pipeline

```
Python baseline → expected_values.json → Rust validation binary
```

### 1. Generate expected values (Python)

```bash
cd /path/to/neuralSpring
python3 control/ltee_mutation_accumulation/ltee_mutation_accumulation.py
```

Outputs `control/ltee_mutation_accumulation/expected_values.json` with:
- Mutation trajectories (total, point, IS, deletion)
- Fitted scalars (rate, power-law exponent, intercept)
- Component rates and dominance hierarchy
- LSTM forward-pass predictions (NumPy RNG, seed 42)
- Interpolation at generation 7,500

### 2. Validate in Rust

```bash
cargo run --bin validate_ltee_b1_mutation_accumulation
```

Structured JSON output for lithoSpore / projectNUCLEUS ingestion:

```bash
cargo run --bin validate_ltee_b1_mutation_accumulation -- --format json
# or: NEURALSPRING_JSON=1 cargo run --bin validate_ltee_b1_mutation_accumulation
```

The Rust binary checks 14 properties (B1-001 through B1-014):
monotonicity, linear fit parity, power-law exponent, LSTM forward
finiteness, neutral model residuals, component rates, interpolation,
and mutation rate bounds.

## Artifacts for lithoSpore

| File | Purpose |
|------|---------|
| `expected_values.json` | Canonical frozen data — lithoSpore module input |
| `ltee_mutation_accumulation.py` | Reproducible baseline generator |
| `validate_ltee_b1_mutation_accumulation` | Rust binary (14 checks, `--format json`) |

## PRNG Note

Python uses `numpy.random.RandomState(42)` for LSTM weight generation.
The Rust binary uses `Xoshiro256++ + Box-Muller` (seed 42). The LSTM
prediction values will differ between languages — `expected_values.json`
records the Python fingerprint. The Rust binary checks finiteness of
its own LSTM forward pass, not numeric match to the Python predictions.

## lithoSpore Module

Target: `ltee-mutation` (ML surrogate ingestion). The `expected_values.json`
fields map directly to lithoSpore's validation targets. The `--format json`
output from the Rust binary provides structured check results for CI
integration.
