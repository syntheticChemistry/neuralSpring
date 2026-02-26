# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ToadStool/BarraCUDA.
Following the wetSpring/hotSpring pattern: unidirectional Spring → ToadStool flow.

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V47** | `handoffs/NEURALSPRING_TOADSTOOL_V47_TITAN_V_PIPELINE_VALIDATION_HANDOFF_FEB26_2026.md` | Feb 26, 2026 | S82: Titan V pure Rust pipeline validation — 384/384 GPU checks PASS on NVK GV100, `fma(f64)` shader fix, zero RTX 4070 regressions, multi-GPU verification, WGSL abstract-float lessons |

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V46, 61 files).

## Conventions

- Naming: `NEURALSPRING_TOADSTOOL_V{NN}_{TOPIC}_{DATE}.md`
- Direction: neuralSpring → ToadStool (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
