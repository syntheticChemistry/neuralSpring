# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### ToadStool/barraCuda

| Version | File | Date | Scope |
|---------|------|------|-------|
| **S139** | `handoffs/NEURALSPRING_TOADSTOOL_V92_S134_DEEP_DEBT_HANDOFF_MAR09_2026.md` | Mar 9, 2026 | S134: Deep debt — activation consolidation, tolerance promotion. 1048 lib + 71 forge tests. 233 binaries. 220/220 validate\_all. Supersedes V91 |

Central wateringHole handoffs (S135–S139) live at `ecoPrimals/wateringHole/handoffs/`:
- `NEURALSPRING_S139_VISUALIZATION_EVOLUTION_HANDOFF_MAR10_2026.md`
- `NEURALSPRING_S139_TOADSTOOL_BARRACUDA_ABSORPTION_HANDOFF_MAR10_2026.md`
- `NEURALSPRING_S138_INDUSTRY_GAP_EVOLUTION_HANDOFF_MAR10_2026.md`
- `NEURALSPRING_V92_S137_UPSTREAM_REWIRE_HANDOFF_MAR10_2026.md`
- `NEURALSPRING_V91_S135_PETALTONGUE_VISUALIZATION_HANDOFF_MAR09_2026.md`

### NestGate (Data Acquisition)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V1** | `handoffs/archive/NEURALSPRING_NESTGATE_V1_DATA_ACQUISITION_MAR01_2026.md` | Mar 1, 2026 | S99 data.* JSON-RPC gap, NCBI/PDB/HuggingFace needs, data volume tiers (1GB–1TB), content-addressed storage, cross-spring data sharing |

### biomeOS/NUCLEUS (Orchestration)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V1** | `handoffs/archive/NEURALSPRING_BIOMEOS_V1_NUCLEUS_INTEGRATION_MAR01_2026.md` | Mar 1, 2026 | S99 primal registration, 11 science capabilities, metalForge↔NUCLEUS alignment, 88/88 checks, LAN multi-gate roadmap, science primal discovery |

### Songbird (Networking)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V1** | `handoffs/archive/NEURALSPRING_SONGBIRD_V1_NETWORK_DISCOVERY_MAR01_2026.md` | Mar 1, 2026 | S99 socket discovery patterns, LAN 10GbE multi-gate vision, bandwidth-aware routing, inter-gate data transfer for science workloads |

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V92 + biomeOS V1 + NestGate V1 + Songbird V1). V74–V92 span S86 through S134. Central handoffs for S135–S139 live in `ecoPrimals/wateringHole/handoffs/`.

## Conventions

- Naming: `NEURALSPRING_{PRIMAL}_V{NN}_{TOPIC}_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
