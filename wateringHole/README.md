# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### toadStool/barraCuda/coralReef (Cross-Spring Absorption)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V106 S155** | `handoffs/NEURALSPRING_V106_S155_CROSS_SPRING_ABSORPTION_BARRACUDA_TOADSTOOL_HANDOFF_MAR16_2026.md` | Mar 16, 2026 | Cross-spring absorption from 4 springs + 5 primals, `primal_names.rs`, `tolerances.py`, provenance trio deploy, 100+ barraCuda primitives consumed, V106 quality gates. Supersedes V105 |

### barraCuda (Bug Fix)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V95** | `handoffs/NEURALSPRING_V95_ENABLE_F64_FIX_HANDOFF_MAR10_2026.md` | Mar 10, 2026 | Critical: `enable f64;` PTXAS silent-zero regression on Ada Lovelace. Local fix for upstream absorption |

### Central wateringHole Copies

Handoffs also published to `ecoPrimals/wateringHole/handoffs/` for cross-project visibility:
- `NEURALSPRING_V106_S155_CROSS_SPRING_ABSORPTION_BARRACUDA_TOADSTOOL_HANDOFF_MAR16_2026.md`
- `NEURALSPRING_V105_S154_NICHE_ARCHITECTURE_BARRACUDA_TOADSTOOL_HANDOFF_MAR15_2026.md`
- `NEURALSPRING_V104_S153_AUDIT_ABSORPTION_BARRACUDA_TOADSTOOL_HANDOFF_MAR15_2026.md`
- `NEURALSPRING_V101_S150_COMPUTE_TRIANGLE_HANDOFF_MAR14_2026.md`

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V105 + NestGate V1 + biomeOS V1 + Songbird V1).

## Conventions

- Naming: `NEURALSPRING_V{NN}_{TOPIC}_HANDOFF_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
