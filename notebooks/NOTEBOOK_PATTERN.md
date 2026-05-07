# neuralSpring — Notebook Pattern

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

> Adapted from primalSpring/wetSpring sporePrint pattern for neuralSpring.
> Date: May 7, 2026 | Session: S188

## Purpose

sporePrint notebooks provide public, reproducible, CI-executable evidence of
neuralSpring's validation state. Each notebook loads frozen JSON data from
`experiments/results/` — no live primals needed.

## Cell Structure

Every notebook follows this 4-cell structure:

### Cell 1 — Title (markdown)

- Notebook title and one-line purpose
- Data sources (which `experiments/results/*.json` files)
- "For other springs" adaptation note
- Session and date context

### Cell 2 — Imports + Data Loading (code)

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS = Path('..') / 'experiments' / 'results'

with open(RESULTS / '<file>.json') as f:
    data = json.load(f)
```

### Cells 3–N — Domain Cells (code + markdown)

- matplotlib charts with ecosystem palette:
  - `#2ecc71` — pass / positive
  - `#e74c3c` — fail / negative
  - `#3498db` — info / neutral
- Analysis markdown cells between code cells
- Each chart should have a clear title, axis labels, and legend

### Final Cell — Summary (markdown)

- Validation state table
- Provenance links to `primals.eco`
- Session reference

## Conventions

- Load frozen data via `Path('..') / 'experiments' / 'results'`
- No live primals — all data from frozen JSON
- Use matplotlib for charts
- All cells must execute cleanly: `jupyter nbconvert --execute`
- End each notebook with provenance summary

## Notebooks

| # | File | Focus |
|---|------|-------|
| 01 | `01-composition-validation.ipynb` | Deploy graphs, bond types, capabilities, discovery tiers |
| 02 | `02-benchmark-comparison.ipynb` | Rust vs Python timing, GPU speedups, guideStone phases |
| 03 | `03-ecosystem-evidence.ipynb` | 134 experiments, gap resolution, security posture |
| 04 | `04-cross-spring-connections.ipynb` | Primal consumption matrix, ecosystem flows |
| 05 | `05-btsp-security-deep-dive.ipynb` | Per-primal BTSP posture, security convergence arc |

## Data Sources

All in `experiments/results/`:

- `validation-state.json` — test counts, capabilities, code quality, guideStone
- `experiment-catalog.json` — 134 experiments across 11 domains
- `security-posture.json` — BTSP, cargo-deny, unsafe, checksums
- `cross-spring-matrix.json` — primal consumption, proto-nucleate
- `benchmark-data.json` — Rust vs Python, GPU, multi-GPU
- `gap-status.json` — 14 gaps, 13 resolved, composition evolution
