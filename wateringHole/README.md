# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### V153 — Session S202 (River Delta Downstream Seeding)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V153** | `handoffs/NEURALSPRING_V153_RIVER_DELTA_DOWNSTREAM_SEEDING_HANDOFF_MAY12_2026.md` | May 12, 2026 | River Delta downstream seeding: `--format json` all validation binaries (Tier 2 projectNUCLEUS), `CapabilityRouter` IPC evolution, foundation Thread 5 expression, workspace deps consolidated, metrics CPU fallback fix, NUCLEUS workload `PRIMALSPRING_JSON=1`. 867 IPC-first workspace tests. V153 handoff. |

### V152 — Session S201b (Tier 4 IPC-first + deep debt + LTEE B1 + foundation seeding)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V152** | `handoffs/archive/NEURALSPRING_V152_TIER4_IPC_FIRST_HANDOFF_MAY11_2026.md` | May 11, 2026 | Tier 4 IPC-first (`default = []`). 48 files feature-gated. 241 require-features bins. CPU fallbacks for 12 primitives. `IpcError` typed hierarchy. V152 handoff. |
| **V152** (companion) | `handoffs/archive/NEURALSPRING_V152_PRIMAL_EVOLUTION_UPSTREAM_HANDOFF_MAY11_2026.md` | May 11, 2026 | Upstream handoff for primal and spring teams: IPC architecture, capability contracts, NUCLEUS composition patterns. |

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
