# neuralSpring V139 — sporePrint Tier 2 Handoff

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Date:** May 7, 2026
**Session:** S188
**From:** neuralSpring
**To:** primalSpring, all spring teams

---

## 1. What We Shipped

neuralSpring is now the third spring (after wetSpring and primalSpring) with
full Tier 2 sporePrint content — 5 public notebooks + 6 frozen JSON datasets,
pushed to main and wired to `notify-sporeprint.yml`.

### Frozen Data (`experiments/results/`)

| File | Contents |
|------|----------|
| `validation-state.json` | 1,387 tests, 269 binaries, 30 capabilities, 233 tolerances, guideStone L3 |
| `experiment-catalog.json` | 134 experiments, 11 domains, 6 faculties, 27 papers, 5 validation tiers |
| `security-posture.json` | BTSP 13/13, cargo-deny (8 bans), forbid(unsafe_code), BLAKE3 (15 files) |
| `cross-spring-matrix.json` | 8 primal dependencies, proto-nucleate (7 caps, 6 deps), barraCuda usage |
| `benchmark-data.json` | 83.6x Rust/Python geomean, 104x GPU peak, 384/384 multi-GPU, 6 primitives |
| `gap-status.json` | 14 main gaps, 13 resolved appendix, 5 composition evolution items |

### Notebooks (`notebooks/`)

| # | Notebook | Focus |
|---|----------|-------|
| 01 | `01-composition-validation.ipynb` | Deploy graphs, bond types, capabilities, discovery tiers, guideStone readiness |
| 02 | `02-benchmark-comparison.ipynb` | Rust vs Python timing (11 domains), GPU, multi-GPU, isomorphic primitives |
| 03 | `03-ecosystem-evidence.ipynb` | 134 experiments, 27 papers, gap resolution, security timeline |
| 04 | `04-cross-spring-connections.ipynb` | 8-primal consumption matrix, ecosystem flow tiers, barraCuda depth |
| 05 | `05-btsp-security-deep-dive.ipynb` | BTSP convergence (Phase 45c), encryption tiers, supply chain, P1-P5 |

### Infrastructure

- `notebooks/NOTEBOOK_PATTERN.md` — cell structure standard, color palette, conventions
- `sporeprint/validation-summary.md` — headline numbers + notebook list
- `.github/workflows/notify-sporeprint.yml` — fires on push to sporePrint content paths

---

## 2. Pattern Followed

Followed the primalSpring/wetSpring pattern exactly:

1. Created `experiments/results/*.json` — frozen data, no live primals needed
2. Created `notebooks/NOTEBOOK_PATTERN.md` — adapted for neuralSpring domains
3. Created 5 notebooks with standard cell structure (title → imports → domain → summary)
4. Used matplotlib with ecosystem palette (#2ecc71 pass, #e74c3c fail, #3498db info)
5. Updated `sporeprint/validation-summary.md` with headline numbers
6. Wired `notify-sporeprint.yml` for CI notification

---

## 3. neuralSpring Headline Numbers

| Metric | Value |
|--------|-------|
| Workspace lib tests | 1,387 (1,234 + 73 + 80) |
| Total validation checks | 4,900+ |
| Experiments | 134 across 11 domains |
| Papers reproduced | 27 (6 faculties) |
| Binaries | 269 (244 validate, 18 bench) |
| Capabilities | 30 (9 domains) |
| Named tolerances | 233 |
| guideStone | Level 3 (29/29 bare, P1-P5) |
| BTSP | 13/13 mandatory |
| Rust vs Python | 83.6x geomean (up to 1104x) |
| GPU vs Python | up to 104x (~97% coverage) |
| Multi-GPU | 384/384 bit-identical |

---

## 4. For Upstream Review

- **primalSpring**: Verify notebook structure matches expected Tier 2 pattern. Confirm
  frozen data format is compatible with any aggregation tooling being built
- **All springs**: neuralSpring's notebooks can serve as a domain-heavy reference —
  the benchmark and cross-spring notebooks show how to visualize consumption matrices
  and multi-tier performance data
- **CI**: `notify-sporeprint.yml` is minimal (content check only). If primalSpring
  has a richer notification pattern (e.g., cross-spring aggregation), we can absorb it

---

## 5. Open Items

- Level 4 guideStone requires live NUCLEUS deployment (tracked in PRIMAL_GAPS §13)
- 18 barraCuda IPC surface gaps block Level 5 (PRIMAL_GAPS §11)
- BTSP session establishment deferred pending BearDog `crypto.btsp_handshake`
- Notebook CI execution (`jupyter nbconvert --execute`) not yet wired into `rust.yml`
  — keeping separate via `notify-sporeprint.yml` per ecosystem convention
