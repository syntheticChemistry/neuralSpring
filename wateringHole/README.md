# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### ToadStool/barraCuda

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V80** | `handoffs/NEURALSPRING_TOADSTOOL_V80_S120_DEEP_DEBT_AUDIT_HANDOFF_MAR03_2026.md` | Mar 3, 2026 | S120: deep debt audit + CI hardening, zero `#[allow(`, `--all-features` CI, barraCuda evolution review. Supersedes V79 |
| V79 | `handoffs/NEURALSPRING_TOADSTOOL_V79_S119_DEEP_LINT_EVOLUTION_HANDOFF_MAR03_2026.md` | Mar 3, 2026 | S119: deep lint evolution, 4 shared validation helpers, 869 lib tests |

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

Superseded handoffs: `handoffs/archive/` (V1–V78 + biomeOS V1). V74 (S86 rewire + nautilus absorption), V75 (S113 cross-spring evolution benchmark), V76 (S115 dispatch parity + NUCLEUS PCIe bypass), V77 (S117 cross-spring shader evolution), V78 (S118 barraCuda standalone rewire).

## Conventions

- Naming: `NEURALSPRING_{PRIMAL}_V{NN}_{TOPIC}_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
