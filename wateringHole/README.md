# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### toadStool/barraCuda/coralReef (Absorption)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V104 S153** | `handoffs/NEURALSPRING_V104_S153_AUDIT_ABSORPTION_BARRACUDA_TOADSTOOL_HANDOFF_MAR15_2026.md` | Mar 15, 2026 | Full ecosystem audit + absorption: tolerance registry, capability unity, ValidationHarness standardization, Kokkos gap, cross-spring patterns. Supersedes V103 |

### barraCuda (Bug Fix)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V95** | `handoffs/NEURALSPRING_V95_ENABLE_F64_FIX_HANDOFF_MAR10_2026.md` | Mar 10, 2026 | Critical: `enable f64;` PTXAS silent-zero regression on Ada Lovelace. Local fix for upstream absorption |

### Central wateringHole Copies

Handoffs also published to `ecoPrimals/wateringHole/handoffs/` for cross-project visibility:
- `NEURALSPRING_V104_S153_AUDIT_ABSORPTION_BARRACUDA_TOADSTOOL_HANDOFF_MAR15_2026.md`
- `NEURALSPRING_V101_S150_COMPUTE_TRIANGLE_HANDOFF_MAR14_2026.md`
- `NEURALSPRING_V100_S147_DEEP_DEBT_EVOLUTION_HANDOFF_MAR14_2026.md`
- `NEURALSPRING_S139_TOADSTOOL_BARRACUDA_ABSORPTION_HANDOFF_MAR10_2026.md`

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V103 + NestGate V1 + biomeOS V1 + Songbird V1).

## Conventions

- Naming: `NEURALSPRING_V{NN}_{TOPIC}_HANDOFF_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
