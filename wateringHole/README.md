# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### V138 — Session S187 (Deep Debt Cleanup + Ecosystem Handoff)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V138** | `handoffs/NEURALSPRING_V138_DEEP_DEBT_ECOSYSTEM_HANDOFF_APR27_2026.md` | Apr 27, 2026 | Deep debt cleanup: 6 smart refactors (all >800L binaries → companion modules), centralized biomeOS socket discovery, `eprintln!`→`log::`, full codebase audit (zero unsafe/mocks/allow/TODO), BarraCUDA API alignment. Ecosystem handoff for primal + spring teams: NUCLEUS composition patterns, neuralAPI deployment, primal evolution recommendations. |

### Central wateringHole Copies

Active handoffs also published to `ecoPrimals/wateringHole/handoffs/` for cross-project visibility.

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V137 + NestGate V1 + biomeOS V1 + Songbird V1 + barraCuda evolution requests).

## Conventions

- Naming: `NEURALSPRING_V{NN}_{TOPIC}_HANDOFF_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
