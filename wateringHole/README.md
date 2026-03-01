# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### ToadStool/BarraCUDA

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V66** | `handoffs/NEURALSPRING_TOADSTOOL_V66_S99_PRIMAL_INTEGRATION_MAR01_2026.md` | Mar 1, 2026 | S99 NUCLEUS Tower validated, nS-01 real-data pipeline (safetensors weight\_loader), 200/200 validate\_all, 216 binaries, 3500+ checks, primal integration |

### NestGate (Data Acquisition)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V1** | `handoffs/NEURALSPRING_NESTGATE_V1_DATA_ACQUISITION_MAR01_2026.md` | Mar 1, 2026 | S99 data.* JSON-RPC gap, NCBI/PDB/HuggingFace needs, data volume tiers (1GB–1TB), content-addressed storage, cross-spring data sharing |

### biomeOS/NUCLEUS (Orchestration)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V1** | `handoffs/NEURALSPRING_BIOMEOS_V1_NUCLEUS_INTEGRATION_MAR01_2026.md` | Mar 1, 2026 | S99 primal registration, 11 science capabilities, metalForge↔NUCLEUS alignment, 88/88 checks, LAN multi-gate roadmap, science primal discovery |

### Songbird (Networking)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V1** | `handoffs/NEURALSPRING_SONGBIRD_V1_NETWORK_DISCOVERY_MAR01_2026.md` | Mar 1, 2026 | S99 socket discovery patterns, LAN 10GbE multi-gate vision, bandwidth-aware routing, inter-gate data transfer for science workloads |

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V65 + biomeOS V1, 80 files).

## Conventions

- Naming: `NEURALSPRING_{PRIMAL}_V{NN}_{TOPIC}_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
