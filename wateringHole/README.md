# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### ToadStool/barraCuda

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V100 S147** | `handoffs/NEURALSPRING_V100_S147_DEEP_DEBT_EVOLUTION_HANDOFF_MAR14_2026.md` | Mar 14, 2026 | Deep debt: zero inline magic numbers, zero duplicate math, provenance completeness, capability-based discovery. Absorption candidates: tolerance registry, ValidationHarness, BaselineProvenance. Supersedes V99 |

### coralReef

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V95 S142** | `handoffs/NEURALSPRING_V95_S142_CORALREEF_HANDOFF_MAR10_2026.md` | Mar 10, 2026 | Precision lessons, bridge status, shader inventory, `enable f64;` implications, DF64 characterization |

### barraCuda (Bug Fix)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **—** | `handoffs/NEURALSPRING_ENABLE_F64_FIX_HANDOFF_MAR10_2026.md` | Mar 10, 2026 | Critical: `enable f64;` PTXAS silent-zero regression on Ada Lovelace. Local fix for upstream absorption |

Central wateringHole handoffs (S135–S147) live at `ecoPrimals/wateringHole/handoffs/`:
- `NEURALSPRING_V100_S147_DEEP_DEBT_EVOLUTION_HANDOFF_MAR14_2026.md`
- `NEURALSPRING_S139_TOADSTOOL_BARRACUDA_ABSORPTION_HANDOFF_MAR10_2026.md`
- `NEURALSPRING_S139_VISUALIZATION_EVOLUTION_HANDOFF_MAR10_2026.md`
- `NEURALSPRING_S138_INDUSTRY_GAP_EVOLUTION_HANDOFF_MAR10_2026.md`
- `NEURALSPRING_V92_S137_UPSTREAM_REWIRE_HANDOFF_MAR10_2026.md`
- `NEURALSPRING_V91_S135_PETALTONGUE_VISUALIZATION_HANDOFF_MAR09_2026.md`

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V99 + NestGate V1 + biomeOS V1 + Songbird V1). V74–V99 span S86 through S146. Central handoffs for S135–S147 live in `ecoPrimals/wateringHole/handoffs/`.

## Conventions

- Naming: `NEURALSPRING_{PRIMAL}_V{NN}_{TOPIC}_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
