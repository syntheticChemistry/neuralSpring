# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### V152 — Session S201b (Tier 4 IPC-first + deep debt + LTEE B1 + foundation seeding)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V152** | `handoffs/NEURALSPRING_V152_TIER4_IPC_FIRST_HANDOFF_MAY11_2026.md` | May 11, 2026 | Tier 4 IPC-first (`default = []`). 48 files feature-gated. 241 require-features bins. CPU fallbacks for 12 primitives. `IpcError` typed hierarchy. `capabilities` module (31 constants). `primal_names::display` (20 entries). `[workspace.dependencies]`. `ipc_dispatch` removed. 693 IPC-first / 1,300 barracuda / 1,453 workspace tests. 19 cert (L5). V152 handoff. |
| **V152** (companion) | `handoffs/NEURALSPRING_V152_PRIMAL_EVOLUTION_UPSTREAM_HANDOFF_MAY11_2026.md` | May 11, 2026 | Upstream handoff for primal and spring teams: IPC architecture, capability contracts, NUCLEUS composition patterns, neuralAPI deployment, evolution patterns for downstream absorption. |

### Central wateringHole Copies

Active handoffs also published to `ecoPrimals/wateringHole/handoffs/` for cross-project visibility.

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V151 + NestGate V1 + biomeOS V1 + Songbird V1 + barraCuda evolution requests).

## Conventions

- Naming: `NEURALSPRING_V{NN}_{TOPIC}_HANDOFF_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
