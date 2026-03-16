# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### V108 S157 — Deep Debt + Idiomatic Rust + Tower Atomic

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V108 S157** | `handoffs/NEURALSPRING_V108_S157_DEEP_DEBT_IDIOMATIC_RUST_HANDOFF_MAR16_2026.md` | Mar 16, 2026 | 5 blanket lint suppressions eliminated, primal binary refactored, error handling evolved, reqwest+ring removed (Tower Atomic), zero C deps, 1128 tests, V108 quality gates. Supersedes V107 |

### Central wateringHole Copies

Handoffs also published to `ecoPrimals/wateringHole/handoffs/` for cross-project visibility:
- `NEURALSPRING_V108_S157_DEEP_DEBT_IDIOMATIC_RUST_HANDOFF_MAR16_2026.md`
- `NEURALSPRING_V107_S156_AUDIT_IPC_FIXES_BARRACUDA_TOADSTOOL_HANDOFF_MAR16_2026.md`
- `NEURALSPRING_V106_S155_CROSS_SPRING_ABSORPTION_BARRACUDA_TOADSTOOL_HANDOFF_MAR16_2026.md`
- `NEURALSPRING_V105_S154_NICHE_ARCHITECTURE_BARRACUDA_TOADSTOOL_HANDOFF_MAR15_2026.md`
- `NEURALSPRING_V104_S153_AUDIT_ABSORPTION_BARRACUDA_TOADSTOOL_HANDOFF_MAR15_2026.md`
- `NEURALSPRING_V101_S150_COMPUTE_TRIANGLE_HANDOFF_MAR14_2026.md`

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V107 + V95 + NestGate V1 + biomeOS V1 + Songbird V1).

## Conventions

- Naming: `NEURALSPRING_V{NN}_{TOPIC}_HANDOFF_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
