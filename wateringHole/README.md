# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### V123 — Session 173

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V123 debt** | `handoffs/NEURALSPRING_V123_DEEP_DEBT_TYPED_ERRORS_MODULE_DECOMPOSITION_HANDOFF_MAR24_2026.md` | Mar 24, 2026 | Typed errors (`thiserror`), nucleus/glucose/immunological module decomposition, explicit barraCuda features, cargo-deny + IPC smoke + rustfmt, dead-code/JSON-RPC/provenance fixes. ~1,385 tests, 0 clippy/fmt/doc. |
| **barraCuda evolution** | `handoffs/NEURALSPRING_BARRACUDA_EVOLUTION_REQUEST_TYPED_ERRORS_DOMAIN_FOLD_MAR24_2026.md` | Mar 24, 2026 | Request: generic f64 ops upstream (`gelu`/`sigmoid`/`layer_norm`/`softmax`), proposed `domain-fold`, typed-error convention alignment. |

### Central wateringHole Copies

Active handoffs also published to `ecoPrimals/wateringHole/handoffs/` for cross-project visibility.

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V120 + V95 + NestGate V1 + biomeOS V1 + Songbird V1 + V121 + V122).

## Conventions

- Naming: `NEURALSPRING_V{NN}_{TOPIC}_HANDOFF_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
