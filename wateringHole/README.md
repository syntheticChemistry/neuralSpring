# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### ToadStool/barraCuda

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V92** | `handoffs/NEURALSPRING_TOADSTOOL_V92_S134_DEEP_DEBT_HANDOFF_MAR09_2026.md` | Mar 9, 2026 | S134: Deep debt — activation consolidation, tolerance promotion, doc alignment. 966 lib + 71 forge tests. 246 binaries. 220/220 validate\_all. 91.66% coverage. Supersedes V91 |

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

Superseded handoffs: `handoffs/archive/` (V1–V91 + biomeOS V1). V74 (S86 rewire + nautilus absorption), V75 (S113 cross-spring evolution benchmark), V76 (S115 dispatch parity + NUCLEUS PCIe bypass), V77 (S117 cross-spring shader evolution), V78 (S118 barraCuda standalone rewire), V79 (S119 deep lint evolution), V80 (S120 deep debt audit + CI hardening), V81 (S121 SimpleMlp rewire + HMM Viterbi f64 ComputeDispatch), V82 (S122–S124 naming rewire + HMM absorption + Paper 026), V83 (S125 wgpu 28 + BarraCUDA v0.3.3 sync), V84 (S126 cross-spring fused op absorption), V85 (S127 Paper 026 full-tier + baseline closure), V86 (S128 VarianceF64 rewire + ToadStool catchup), V87 (S129 struct-based API sync), V88 (S130 upstream rewire + PrecisionRoutingAdvice), V89 (S131 isomorphic coverage), V90 (S132 upstream rewire + cross-spring provenance), V91 (S133 Phase 5–7 buildout — metalForge PCIe P2P, biomeOS DAG, petalTongue StreamSession).

## Conventions

- Naming: `NEURALSPRING_{PRIMAL}_V{NN}_{TOPIC}_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
