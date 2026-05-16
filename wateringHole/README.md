# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### V165 — Session S209 (Live Composition + Live Data Chains)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V165** | `handoffs/NEURALSPRING_V165_LIVE_COMPOSITION_MAY16_2026.md` | May 16, 2026 | `nest.commit` + `store_science_result()` provenance chain. `node.compute` signal dispatch. `execute_graph_live()` live IPC pipeline. 2 new validation scenarios (9 total). Gaps 15–18 resolved. |

### Central wateringHole Copies

Active handoffs also published to `ecoPrimals/wateringHole/handoffs/` for cross-project visibility.

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V164 + NestGate V1 + biomeOS V1 + Songbird V1 + barraCuda evolution requests).

## Conventions

- Naming: `NEURALSPRING_V{NN}_{TOPIC}_HANDOFF_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
