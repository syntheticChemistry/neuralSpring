# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ToadStool/BarraCUDA.
Following the wetSpring/hotSpring pattern: unidirectional Spring → ToadStool flow.

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V56** | `handoffs/NEURALSPRING_TOADSTOOL_V56_E96576EE_UPSTREAM_SYNC_HANDOFF_FEB27_2026.md` | Feb 27, 2026 | ToadStool `e96576ee` sync: `compile_shader_df64` rewired, pin updated across 17 files, universal precision review (703 WGSL all f64 canonical), LogSumExp/PairwiseDistance/BatchedEighGpu confirmed upstream. 171/172 validate\_all, 2970+ checks |
| biomeOS | `handoffs/NEURALSPRING_BIOMEOS_V1_NUCLEUS_INTEGRATION_HANDOFF_FEB27_2026.md` | Feb 27, 2026 | biomeOS NUCLEUS integration — science primal, 7 capabilities, JSON-RPC server, 29/29 PASS |

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V55, 70 files).

## Conventions

- Naming: `NEURALSPRING_TOADSTOOL_V{NN}_{TOPIC}_{DATE}.md`
- Direction: neuralSpring → ToadStool (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
