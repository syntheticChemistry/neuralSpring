# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ToadStool/BarraCUDA.
Following the wetSpring/hotSpring pattern: unidirectional Spring → ToadStool flow.

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V58** | `handoffs/NEURALSPRING_TOADSTOOL_V58_CPU_PARITY_GPU_PORTABILITY_HANDOFF_FEB27_2026.md` | Feb 27, 2026 | CPU parity (83.6× vs Python, 11 domains) + GPU portability (9/9, 7 domains), 175 binaries, 174/175 validate\_all, 3034+ checks |
| biomeOS | `handoffs/NEURALSPRING_BIOMEOS_V1_NUCLEUS_INTEGRATION_HANDOFF_FEB27_2026.md` | Feb 27, 2026 | biomeOS NUCLEUS integration — science primal, 7 capabilities, JSON-RPC server, 29/29 PASS |

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V57, 72 files).

## Conventions

- Naming: `NEURALSPRING_TOADSTOOL_V{NN}_{TOPIC}_{DATE}.md`
- Direction: neuralSpring → ToadStool (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
