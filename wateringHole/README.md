# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### ToadStool/barraCuda

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V87** | `handoffs/NEURALSPRING_TOADSTOOL_V87_S129_API_SYNC_HANDOFF_MAR07_2026.md` | Mar 7, 2026 | S129: Struct-based API sync (HmmForwardArgs, GillespieModel, Conv2dConfig, Pool2dConfig, Rk45DispatchArgs), `#![forbid(unsafe_code)]` enforced, 141+ named tolerances, 12 GPU SIGSEGV root-cause identified (fused ops on llvmpipe), docs aligned. Supersedes V86 |

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

Superseded handoffs: `handoffs/archive/` (V1–V86 + biomeOS V1). V74 (S86 rewire + nautilus absorption), V75 (S113 cross-spring evolution benchmark), V76 (S115 dispatch parity + NUCLEUS PCIe bypass), V77 (S117 cross-spring shader evolution), V78 (S118 barraCuda standalone rewire), V79 (S119 deep lint evolution), V80 (S120 deep debt audit + CI hardening), V81 (S121 SimpleMlp rewire + HMM Viterbi f64 ComputeDispatch), V82 (S122–S124 naming rewire + HMM absorption + Paper 026), V83 (S125 wgpu 28 + BarraCUDA v0.3.3 sync), V84 (S126 cross-spring fused op absorption), V85 (S127 Paper 026 full-tier + baseline closure), V86 (S128 VarianceF64 rewire + ToadStool catchup).

## Conventions

- Naming: `NEURALSPRING_{PRIMAL}_V{NN}_{TOPIC}_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
